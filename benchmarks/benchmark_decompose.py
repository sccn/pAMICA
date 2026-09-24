#!/usr/bin/env python
"""Phase 3 (#87): full-decomposition equivalence + total-time benchmark.

Runs each AMICA backend to a full decomposition (default 2000 iters) on real EEG,
saving total wall-clock time, final log-likelihood, and the unmixing matrix W.
A ``--compare`` pass Hungarian-matches W across backends to show they all recover
the SAME independent components (the equivalence figure), and tabulates total
decompose time and LL.

Three axes (channel x core x FP):
  * channels  -- sweep with --channels (each needs k = frames/ch^2 >= ~20-30 for
    a well-determined decomposition; the k30 data has 147k frames -> k=30 at 70ch)
  * cores     -- --threads for the CPU backends (torch-cpu, native-fortran)
  * precision -- pick the f32/f64 backend names in --backends (torch-cpu/cuda have
    both; mlx/mps are f32-only; native-fortran is f64-only)

numpy is intentionally excluded (too slow at k30 and not a backend to recommend).

Usage:
    # run one machine's backends to 2000 iters, save per-run npz into a dir
    uv run python benchmarks/benchmark_decompose.py --data DATA.npy \
        --channels 70 --iters 2000 --backends native-fortran-f64,torch-cuda-f64,torch-cuda-f32 \
        --out results_cuda/
    # merge every machine's dir and emit the equivalence figure + tables
    uv run python benchmarks/benchmark_decompose.py --compare results_cuda/ results_mac/ \
        --figure equivalence.png
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

N_MIX = 3
SEED = 42
BLOCK_SIZE = 512
_REPO = Path(__file__).resolve().parent.parent
_FORTRAN_TIMEOUT = 36000  # a full 2000-iter fit can be long at k30 frames


def _load_dimsweep():
    """Reuse the Fortran .fdt writer / param renderer / out.txt parser (#85)."""
    path = _REPO / "benchmarks" / "benchmark_dimsweep.py"
    spec = importlib.util.spec_from_file_location("benchmark_dimsweep", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_ds = _load_dimsweep()


# --------------------------------------------------------------------------
# Per-backend fit + extract: returns dict(time, final_ll, W, A). W is the
# unmixing matrix in the (n_comps x n_channels) convention get_unmixing_matrix
# uses (rows = components), so all backends are directly Hungarian-comparable.
# --------------------------------------------------------------------------
def _fit_torch(data, device, dtype_str, iters, threads=None):
    import torch

    from pamica.torch_impl import AMICATorchNG

    dtype = torch.float64 if dtype_str == "f64" else torch.float32
    prev = torch.get_num_threads()
    try:
        if device == "cpu" and threads:
            torch.set_num_threads(int(threads))
        m = AMICATorchNG(
            n_channels=data.shape[0],
            n_models=1,
            n_mix=N_MIX,
            device=device,
            dtype=dtype,
            do_newton=False,
            block_size=BLOCK_SIZE,
            seed=SEED,
        )
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = perf_counter()
        m.fit(data, max_iter=iters, verbose=False)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = perf_counter() - t0
    finally:
        if device == "cpu":
            torch.set_num_threads(prev)
    assert m.final_ll_ is not None
    return {
        "time": elapsed,
        "final_ll": float(m.final_ll_),
        "W": m.get_unmixing_matrix(0),
        "A": m.get_mixing_matrix(0),
    }


def _fit_mlx(data, iters):

    from pamica.mlx_impl import AMICAMLXNG

    m = AMICAMLXNG(
        n_channels=data.shape[0],
        n_models=1,
        n_mix=N_MIX,
        do_newton=False,
        block_size=BLOCK_SIZE,
        seed=SEED,
    )
    t0 = perf_counter()
    m.fit(data, max_iter=iters, verbose=False)  # fit's mx.eval syncs
    elapsed = perf_counter() - t0
    # convert mlx -> numpy FIRST, then index (mlx cannot be indexed by a numpy array)
    A_np = np.array(m.A)
    W_np = np.array(m.W)
    idx = np.array(m.comp_list)[:, 0].astype(int)
    W = W_np[0].T  # (comps x channels), matching torch get_unmixing convention
    A = A_np[idx, :].T  # component rows of model 0 (issue #334 layout)
    assert m.final_ll_ is not None
    return {"time": elapsed, "final_ll": float(m.final_ll_), "W": W, "A": A}


def _fit_fortran(data, iters, binary, threads):
    binary = Path(binary)
    if not (binary.exists() and os.access(binary, os.X_OK)):
        raise RuntimeError(f"native fortran binary not found/executable: {binary}")
    threads = threads or (os.cpu_count() or 4)
    nc, ns = data.shape
    with tempfile.TemporaryDirectory(prefix="amica_decomp_") as td:
        work = Path(td)
        _ds._write_fdt(data, work / "bench.fdt")
        _ds._write_fortran_param(work, nc, ns, iters, threads, 1)
        (work / "bench_out").mkdir(exist_ok=True)
        env = {**os.environ, "OMP_NUM_THREADS": str(threads)}
        t0 = perf_counter()
        res = subprocess.run(
            [str(binary), "input.param"],
            cwd=work,
            env=env,
            capture_output=True,
            text=True,
            timeout=_FORTRAN_TIMEOUT,
        )
        elapsed = perf_counter() - t0
        if res.returncode != 0:
            print(f"    fortran exit {res.returncode}\n{res.stderr[-2000:]}")
            raise RuntimeError(f"native fortran run failed (exit {res.returncode})")
        _secs, final_ll = _ds._parse_fortran_out(work / "bench_out" / "out.txt")
        # Fortran writes W/A as (n x n) float64 in column-major order; W's rows are
        # components (same convention validate_implementations.py matches on).
        W = np.fromfile(work / "bench_out" / "W", dtype=np.float64).reshape(
            nc, nc, order="F"
        )
        A = np.fromfile(work / "bench_out" / "A", dtype=np.float64).reshape(
            nc, nc, order="F"
        )
    return {"time": elapsed, "final_ll": final_ll, "W": W, "A": A}


# backend name -> (kind, kwargs); kind selects the fit function above
_BACKENDS = {
    "torch-cpu-f64": ("torch", {"device": "cpu", "dtype_str": "f64"}),
    "torch-cpu-f32": ("torch", {"device": "cpu", "dtype_str": "f32"}),
    "torch-cuda-f64": ("torch", {"device": "cuda", "dtype_str": "f64"}),
    "torch-cuda-f32": ("torch", {"device": "cuda", "dtype_str": "f32"}),
    "torch-mps-f32": ("torch", {"device": "mps", "dtype_str": "f32"}),
    "mlx-f32": ("mlx", {}),
    "native-fortran-f64": ("fortran", {}),
}


def _fit(name, data, iters, fortran_bin, threads):
    kind, kw = _BACKENDS[name]
    if kind == "torch":
        return _fit_torch(data, iters=iters, threads=threads, **kw)
    if kind == "mlx":
        return _fit_mlx(data, iters=iters)
    return _fit_fortran(data, iters=iters, binary=fortran_bin, threads=threads)


def _platform_tag() -> str:
    import platform

    return f"{platform.system()}-{platform.machine()}"


# --------------------------------------------------------------------------
# Equivalence: Hungarian-match unmixing rows (components) between two runs.
# --------------------------------------------------------------------------
def _match_correlation(W1: np.ndarray, W2: np.ndarray) -> np.ndarray:
    """Best per-component |correlation| after optimal (Hungarian) assignment.
    Rows are unit-normalized so W @ W.T is the cosine similarity; abs handles the
    ICA sign ambiguity, linear_sum_assignment handles the permutation."""
    from scipy.optimize import linear_sum_assignment

    n1 = W1 / (np.linalg.norm(W1, axis=1, keepdims=True) + 1e-12)
    n2 = W2 / (np.linalg.norm(W2, axis=1, keepdims=True) + 1e-12)
    corr = np.abs(n1 @ n2.T)
    row, col = linear_sum_assignment(1.0 - corr)
    return corr[row, col]


def _desphere(W, data, channels):
    """Recompute the symmetric-ZCA sphere from the data and return the de-sphered
    mixing (channels x comps, columns = true sensor-space IC scalp maps, the
    EEGLAB representation) and the source activations (comps x time).

    The saved A is whitened-space loadings; the sphere is far from identity, so
    the sensor-space scalp maps and the variance-explained both need de-sphering.
    Every backend spheres the same data the same way, so the recomputed sphere
    matches each backend's own (symmetric ZCA, no PCA reduction)."""
    x = np.ascontiguousarray(data[:channels]).astype(np.float64)
    x = x - x.mean(axis=1, keepdims=True)
    cov = (x @ x.T) / x.shape[1]
    d, V = np.linalg.eigh(cov)
    d = np.clip(d, 1e-12, None)
    sphere = (V * (1.0 / np.sqrt(d))) @ V.T  # symmetric ZCA
    U = W @ sphere  # comps x channels, unmixing from the original sensors
    ades = np.linalg.pinv(U)  # channels x comps, de-sphered mixing (scalp maps)
    return ades, U @ x


def _variance_order(W, data, channels):
    """EEGLAB-style component order by back-projected variance (highest first):
    ``||desphered mixing col||^2 * var(source)`` (torch/mlx normalize the
    whitened-space A, so the variance lives in the sources, not in A)."""
    ades, sources = _desphere(W, data, channels)
    projvar = (ades**2).sum(axis=0) * sources.var(axis=1)
    return np.argsort(-projvar)


def _k_sweep(runs, figure=None):
    """Data-size hypothesis test (#90): at fixed channels, mean cross-backend
    Hungarian-matched |corr| vs k = frames/ch^2. Equivalence should rise toward
    1.0 as k grows (the decomposition becomes well-determined)."""
    nc = runs[0]["channels"]
    by_f: dict = {}
    for r in runs:
        if r["frames"] <= 0:  # legacy npz without a recorded frame count -> skip
            continue
        by_f.setdefault(r["frames"], []).append(r)
    rows, lines = [], []
    for nf in sorted(by_f):
        g = by_f[nf]
        if len(g) < 2:
            continue
        pair_means, all_c = [], []
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                c = _match_correlation(g[i]["W"], g[j]["W"])
                pair_means.append(c.mean())
                all_c.append(c)
        allc = np.concatenate(all_c)
        k = nf / nc**2
        mean_eq = float(np.mean(pair_means))
        rows.append((nf, k, mean_eq))
        lines.append(
            f"  {nf:8d}  {k:6.1f}    {mean_eq:.4f}   {min(pair_means):.4f}"
            f"     {(allc > 0.95).mean() * 100:5.1f}%   {len(g)}"
        )
    if not rows:  # every frame count had <2 backends -- nothing to compare
        print(
            f"\n  data-size (k) sweep @ {nc} channels: need >=2 backends at a shared "
            "frame count -- nothing to compare"
        )
        return
    print(f"\n=== data-size (k) sweep @ {nc} channels ===")
    print("  frames        k   mean|corr|   min    comps>0.95   n")
    for ln in lines:
        print(ln)
    if figure:
        _plot_ksweep(rows, nc, figure)


def _plot_ksweep(rows, channels, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ks = [r[1] for r in rows]
    eq = [r[2] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ks, eq, "o-", color="#2a7")
    for nf, k, e in rows:
        ax.annotate(
            f"{nf // 1000}k",
            (k, e),
            fontsize=7,
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
        )
    ax.axhline(1.0, ls="--", lw=0.8, color="0.6")
    ax.set_xlabel("k = frames / channels^2")
    ax.set_ylabel("mean cross-backend |correlation|")
    ax.set_title(
        f"AMICA cross-backend IC equivalence vs data size @ {channels} channels\n"
        "(more data -> better-determined decomposition -> higher equivalence)"
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


def _compare(dirs, figure=None, montage=None, topo_figure=None, n_topo=6, data=None):
    full = np.load(data).astype(np.float64) if data else None
    runs: list[dict[str, Any]] = []
    for d in dirs:
        for f in sorted(Path(d).glob("*.npz")):
            z = np.load(f, allow_pickle=True)
            runs.append(
                {
                    "label": str(z["label"]),
                    "channels": int(z["channels"]),
                    "channel_indices": z["channel_indices"]
                    if "channel_indices" in z
                    else np.arange(int(z["channels"])),
                    "frames": int(z["frames"]) if "frames" in z else -1,
                    "time": float(z["time"]),
                    "final_ll": float(z["final_ll"]),
                    "W": z["W"],
                    "A": z["A"],
                }
            )
    if not runs:
        print("no result npz found")
        return

    # data-size (k) sweep (#90): at the top channel count, if multiple frame counts
    # were run, report mean cross-backend equivalence vs k = frames/ch^2. When a
    # k-sweep runs it owns --figure (the headline plot); the per-config matrix below
    # then goes to a derived "{stem}_matrix{suffix}" path so neither overwrites the
    # other.
    top_ch = max(r["channels"] for r in runs)
    top_runs = [r for r in runs if r["channels"] == top_ch]
    # legacy npz (pre-#90) carry no frame count (frames == -1); they can't join a
    # k-sweep and won't group with per-frame npz in the matrix below, so warn rather
    # than silently group them wrong when the two formats are merged in one --compare.
    if any(r["frames"] < 0 for r in top_runs) and any(
        r["frames"] > 0 for r in top_runs
    ):
        print(
            f"  warning: {top_ch}ch mixes legacy npz (no recorded frame count) with "
            "per-frame npz; legacy runs are excluded from the k-sweep and grouped "
            "separately in the equivalence matrix"
        )
    real_frames = {r["frames"] for r in top_runs if r["frames"] > 0}
    matrix_figure = figure
    if len(real_frames) > 1:
        _k_sweep(top_runs, figure)
        if figure:
            p = Path(figure)
            matrix_figure = str(p.with_name(f"{p.stem}_matrix{p.suffix}"))

    # single-config equivalence matrix + topomaps: use the largest (channels, frames)
    by_cf: dict = {}
    for r in runs:
        by_cf.setdefault((r["channels"], r["frames"]), []).append(r)
    target_cf = max(by_cf)
    target = target_cf[0]
    group = by_cf[target_cf]
    labels = [r["label"] for r in group]
    n = len(group)

    print(f"\n=== decompose time + final LL @ {target} channels, sorted by time ===")
    for r in sorted(group, key=lambda x: x["time"]):
        print(f"  {r['label']:24s} {r['time']:8.1f} s   LL={r['final_ll']:+.5f}")

    if n < 2:
        print(
            f"\n  only one backend result at {target} channels -- nothing to "
            "compare (merge more --out dirs for an equivalence matrix)"
        )
        return

    print(f"\n=== IC equivalence: mean Hungarian-matched |corr| @ {target}ch ===")
    M = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            c = _match_correlation(group[i]["W"], group[j]["W"]).mean()
            M[i, j] = M[j, i] = c
    hdr = "                          " + " ".join(f"{lb[:8]:>8}" for lb in labels)
    print(hdr)
    for i, lb in enumerate(labels):
        print(f"  {lb:24s} " + " ".join(f"{M[i, j]:8.3f}" for j in range(n)))
    offdiag = M[~np.eye(n, dtype=bool)]
    print(
        f"\n  cross-backend mean |corr| = {offdiag.mean():.4f} "
        f"(min {offdiag.min():.4f}) -- ~1.0 => same ICs recovered"
    )

    if matrix_figure:
        _plot(labels, M, target, matrix_figure)
    if topo_figure and montage:
        # de-sphere must recompute the sphere from the SAME frames AND channels the
        # group was fit on (#90 --frames truncates frames; #91 selects distributed
        # channels), so slice full to both, not the whole array.
        sel = group[0]["channel_indices"]
        nf = target_cf[1]
        if full is None:
            topo_data = None
        else:
            topo_data = full[sel][:, :nf] if nf > 0 else full[sel]
        _plot_topomaps(group, montage, sel, topo_figure, n_topo, topo_data)


def _load_info(montage_tsv, channel_indices):
    """Build an MNE Info with a scalp montage from a BIDS electrodes.tsv.

    ``channel_indices`` are the 0-based data-channel indices actually used (issue
    #91 selects a distributed subset, so they need not be contiguous); the
    electrode name for data channel ``i`` is ``EEG{i+1:03d}``.

    NOTE: the tsv coordinates may need a rotation to MNE's head frame (nose +y)
    for the absolute orientation to be correct -- to be checked/fixed later. A
    global rotation does NOT affect the cross-backend equivalence (every backend
    shares the montage), only the absolute nose-up orientation of the maps."""
    import csv

    # mne is the optional 'viz' extra, not in the base env; ty can't resolve it
    # there. Lazy-imported so the benchmark's core paths don't require it.
    import mne  # ty: ignore[unresolved-import]

    pos = {}
    with open(montage_tsv) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            try:  # skip channels with an 'n/a' (unlocalized / non-EEG) position
                pos[row["name"]] = (
                    np.array([float(row["x"]), float(row["y"]), float(row["z"])])
                    / 100.0  # tsv is in cm; MNE head frame is meters
                )
            except ValueError:
                continue
    ch_names = [f"EEG{i + 1:03d}" for i in channel_indices]
    # some channels can be unlocalized ('n/a'); topomap only the located subset
    located = np.array([i for i, n in enumerate(ch_names) if n in pos])
    used = [ch_names[i] for i in located]
    dig = mne.channels.make_dig_montage(
        ch_pos={n: pos[n] for n in used}, coord_frame="head"
    )
    info = mne.create_info(used, sfreq=250.0, ch_types="eeg")
    info.set_montage(dig)
    return info, located


def _plot_topomaps(group, montage_tsv, channel_indices, path, n_comps, data=None):
    """Grid of IC scalp maps: rows = backends, cols = components. Columns are
    ordered by the reference's back-projected variance (EEGLAB convention, IC1 =
    highest variance) when ``data`` is given, else by cross-backend match. Each
    cell is that backend's Hungarian-matched, sign-aligned map; identical columns
    down the rows mean every backend recovered the same IC.

    ``channel_indices`` are the 0-based data channels used (issue #91)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # mne: optional 'viz' extra, absent from the base env (see above).
    import mne  # ty: ignore[unresolved-import]
    from scipy.optimize import linear_sum_assignment

    info, located = _load_info(montage_tsv, channel_indices)
    n_ch = len(channel_indices)
    ref = group[0]  # reference backend: the first result loaded (glob order)
    Wref = ref["W"] / (np.linalg.norm(ref["W"], axis=1, keepdims=True) + 1e-12)
    n_ref = Wref.shape[0]

    # per-backend Hungarian match to the reference + a per-ref-component match score
    matches = []
    score = np.zeros(n_ref)
    for r in group:
        Wb = r["W"] / (np.linalg.norm(r["W"], axis=1, keepdims=True) + 1e-12)
        corr = Wref @ Wb.T  # signed cosine between ref and this backend
        _row, col = linear_sum_assignment(-np.abs(corr))
        matches.append((col, corr))
        if r is not ref:
            score += np.abs(corr[np.arange(n_ref), col])
    score /= max(len(group) - 1, 1)

    # column order: EEGLAB back-projected variance if data available, else match
    if data is not None:
        order = _variance_order(ref["W"], data, n_ch)[:n_comps]
    else:
        order = np.argsort(-score)[: min(n_comps, n_ref)]

    fig, axes = plt.subplots(
        len(group),
        len(order),
        figsize=(1.5 * len(order), 1.5 * len(group) + 0.5),
        squeeze=False,
    )
    for bi, r in enumerate(group):
        col, corr = matches[bi]
        # de-sphered mixing = true sensor-space IC scalp maps (EEGLAB); the saved
        # A is whitened-space. Fall back to A only if no data to recompute the sphere.
        ades = _desphere(r["W"], data, n_ch)[0] if data is not None else r["A"]
        for ci, i in enumerate(order):
            j = col[i]
            sign = 1.0 if corr[i, j] >= 0 else -1.0
            scalp = ades[located, j] * sign  # column j = component j's scalp map
            ax = axes[bi][ci]
            mne.viz.plot_topomap(scalp, info, axes=ax, show=False, contours=4)
            if bi == 0:
                # IC index is the variance rank (IC1 = highest variance); r = match
                ax.set_title(f"IC{ci + 1}\n(r={score[i]:.3f})", fontsize=8)
        axes[bi][0].text(
            -0.35,
            0.5,
            r["label"].split("@")[0],
            rotation=90,
            fontsize=7,
            va="center",
            ha="center",
            transform=axes[bi][0].transAxes,
        )
    fig.suptitle(
        f"AMICA IC scalp maps @ {n_ch} channels -- matched across backends "
        "(rows), same IC (columns)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0.02, 0, 1, 0.97))
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


def _plot(labels, M, channels, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(labels)
    # square figure + equal-aspect cells so the matrix reads as a true square.
    fig, ax = plt.subplots(figsize=(0.85 * n + 3.5, 0.85 * n + 3.0))
    # stretch the color range to the actual off-diagonal spread so the small
    # differences (e.g. fortran ~0.90 vs torch/mlx 1.000) are visible, not washed
    # out by a fixed 0.9-1.0 scale.
    off = M[~np.eye(n, dtype=bool)]
    vmin = np.floor(off.min() * 100) / 100
    im = ax.imshow(M, vmin=vmin, vmax=1.0, cmap="viridis", aspect="equal")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    mid = (vmin + 1.0) / 2
    for i in range(n):
        for j in range(n):
            ax.text(
                j,
                i,
                f"{M[i, j]:.3f}",
                ha="center",
                va="center",
                color="white" if M[i, j] < mid else "black",
                fontsize=8,
            )
    ax.set_title(
        f"AMICA cross-backend IC equivalence @ {channels} channels\n"
        "mean Hungarian-matched |correlation| of unmixing components"
    )
    fig.colorbar(im, ax=ax, label="mean |corr|", fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", help="path to the (channels, frames) real-EEG .npy")
    ap.add_argument("--channels", default="70")
    ap.add_argument(
        "--frames",
        default=None,
        help="data-size sweep (#90): comma-separated frame counts (first N samples). "
        "Default: all frames. At fixed channels this sweeps k = frames/ch^2.",
    )
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--backends", default="auto")
    ap.add_argument(
        "--fortran-bin",
        default=str(_REPO / "benchmarks/fortran/amica15"),
        dest="fortran_bin",
    )
    ap.add_argument("--out", help="directory to write per-run npz into")
    ap.add_argument("--compare", nargs="+", help="result dirs -> equivalence report")
    ap.add_argument("--figure", help="write the equivalence heatmap PNG here")
    ap.add_argument("--montage", help="BIDS electrodes.tsv for IC scalp-map topomaps")
    ap.add_argument(
        "--topo-figure", dest="topo_figure", help="write the IC topomap grid PNG here"
    )
    ap.add_argument(
        "--n-topo",
        type=int,
        default=6,
        dest="n_topo",
        help="# components in the topomap grid",
    )
    args = ap.parse_args()

    if args.compare:
        _compare(
            args.compare,
            args.figure,
            args.montage,
            args.topo_figure,
            args.n_topo,
            args.data,
        )
        return 0
    if not args.data or not args.out:
        ap.error("--data and --out are required unless --compare is given")

    full = np.load(args.data).astype(np.float64)
    channels = [int(c) for c in args.channels.split(",") if int(c) <= full.shape[0]]
    backends = list(_BACKENDS) if args.backends == "auto" else args.backends.split(",")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = _platform_tag()
    print(f"platform {tag} | channels {channels} | iters {args.iters} | {backends}")

    # clamp to the available frames and dedupe up front so two requested counts that
    # collapse to the same length don't re-fit identical data and clobber each other's npz.
    frame_list = (
        sorted({min(int(f), full.shape[1]) for f in args.frames.split(",")})
        if args.frames
        else [full.shape[1]]
    )
    for nc in channels:
        # #91: with a montage, pick spatially-distributed (whole-head) channels
        # rather than the first nc electrodes in file order (a spatial cluster);
        # the selected data-channel indices are saved so the topomaps use them.
        if args.montage and nc < full.shape[0]:
            from channel_selection import (
                positions_for_channels,
                select_distributed_channels,
            )

            sel = select_distributed_channels(
                positions_for_channels(args.montage, full.shape[0]), nc
            )
        else:
            sel = np.arange(nc)
        for nf in frame_list:
            data = np.ascontiguousarray(full[sel][:, :nf])
            k = nf / len(sel) ** 2
            print(f"\n{len(sel)}ch, {nf} frames (k={k:.1f})")
            for b in backends:
                label = f"{b}@{tag}"
                try:
                    t0 = perf_counter()
                    res = _fit(b, data, args.iters, args.fortran_bin, args.threads)
                    fname = out / f"{b}_{nc}ch_{nf}f_{tag}.npz"
                    np.savez_compressed(
                        fname,
                        label=label,
                        backend=b,
                        channels=len(sel),
                        channel_indices=sel,
                        frames=nf,
                        iters=args.iters,
                        time=res["time"],
                        final_ll=res["final_ll"],
                        W=res["W"],
                        A=res["A"],
                    )
                    print(
                        f"  {b:24s} {res['time']:8.1f} s   LL={res['final_ll']:+.5f}"
                        f"   [{perf_counter() - t0:.0f}s wall]  -> {fname.name}"
                    )
                except Exception as exc:  # noqa: BLE001 - report and continue
                    print(f"  {b:24s} FAILED: {type(exc).__name__}: {str(exc)[:80]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
