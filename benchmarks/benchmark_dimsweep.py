#!/usr/bin/env python
"""Cross-platform result + performance benchmark for AMICA backends (issue #77).

Sweeps the channel count on real 70-channel EEG (OpenNeuro ds002718 sub-002, the
Wakeman-Henson faces data) and, for every backend the current host supports,
records BOTH performance (ms/iteration, warmed, min-of-repeats) and results
(converged log-likelihood) so the two questions the epic (#74) has deferred can
finally be answered together:

  * do the backends/platforms/precisions agree on the *result*?
  * where does an Apple/NVIDIA GPU actually *beat* the CPU?

Backends (auto-detected per platform):
  numpy-cpu-f64   AMICA_NumPy (the float64 reference implementation)
  torch-cpu-f64   AMICATorchNG, CPU, float64 (the Fortran-parity path)
  torch-cpu-f32   AMICATorchNG, CPU, float32
  torch-mps-f32   AMICATorchNG, Apple GPU (MPS), float32
  torch-cuda-f64  AMICATorchNG, NVIDIA GPU, float64
  torch-cuda-f32  AMICATorchNG, NVIDIA GPU, float32
  mlx-f32         AMICAMLXNG, Apple GPU (MLX), float32
  native-fortran-f64  amica15 compiled from source (MPI+OpenMP), the reference
                  (#85; timed via amica's own per-iter stamps, startup-immune;
                  build it with benchmarks/fortran/build_amica.sh -- works on both
                  Apple Silicon and the x86 host, just not built by default / in CI)

All backends run at matched settings (n_models=1, n_mix=3, pdftype=0,
do_newton=False, same block_size/seed/channels/samples/iters) so the comparison
is apples-to-apples. Emits JSON (``--out``) so a Mac run (cpu/mps/mlx) and a CUDA
host run can be merged by ``--report``.

Data: pass ``--data path/to/ds002718_sub-002_eeg70.npy`` -- a (70, n_samples)
float64 array of the 70 EEG channels (see ``benchmarks/README_dimsweep.md`` for
how to fetch/extract it; not committed).

``--threads`` (#86) adds a CPU core-count scaling sweep: the CPU backends
(torch-cpu, numpy, native-fortran) are run at each thread count in the list, so
the report shows where cores catch the GPU. GPU backends run once (thread-
independent). torch-cpu uses ``set_num_threads``, numpy uses ``threadpoolctl``,
native-fortran uses ``OMP_NUM_THREADS``.

Usage:
    uv run python benchmarks/benchmark_dimsweep.py --data DATA --out mac.json
    uv run python benchmarks/benchmark_dimsweep.py --data DATA --backends torch-cuda-f64,torch-cuda-f32 --out cuda.json
    uv run python benchmarks/benchmark_dimsweep.py --data DATA --backends torch-cpu-f64,numpy-cpu-f64,native-fortran-f64 --threads 4,8,16,32 --out scaling.json
    uv run python benchmarks/benchmark_dimsweep.py --report mac.json cuda.json scaling.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import statistics
import subprocess
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

# Matched fit settings for every backend.
N_MIX = 3
SEED = 42
BLOCK_SIZE = 512

# Native-Fortran backend (#85): paths + run limits.
_REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DIR = _REPO_ROOT / "pamica" / "sample_data"
_PARAM_TEMPLATE = SAMPLE_DIR / "input.param"
_DEFAULT_FORTRAN_BIN = _REPO_ROOT / "benchmarks" / "fortran" / "amica15"
_FORTRAN_TIMEOUT = 3600  # seconds; a crashed/hung run must not hang the sweep


# --------------------------------------------------------------------------
# Backend adapters: each returns (ms_per_iter, final_ll) for one fit.
# --------------------------------------------------------------------------
def _torch_available(device: str) -> bool:
    try:
        import torch
    except ImportError:
        return False
    if device == "cpu":
        return True
    if device == "mps":
        return torch.backends.mps.is_available()
    if device == "cuda":
        return torch.cuda.is_available()
    return False


def _share_kw_torch(share) -> dict[str, Any]:
    # share_start/share_iter tuned so a merge actually fires inside the short
    # benchmark budget (share_iter must be > 6); no-op when share is off.
    return (
        dict(share_comps=True, share_start=5, share_iter=8, comp_thresh=0.99)
        if share
        else {}
    )


def _run_torch(
    data, device, dtype_str, iters, repeats, n_models=1, share=False, threads=None
):
    import torch

    from pamica.torch_impl import AMICATorchNG

    dtype = torch.float64 if dtype_str == "f64" else torch.float32

    def one(n_iter):
        m = AMICATorchNG(
            n_channels=data.shape[0],
            n_models=n_models,
            n_mix=N_MIX,
            device=device,
            dtype=dtype,
            do_newton=False,
            block_size=BLOCK_SIZE,
            seed=SEED,
            **_share_kw_torch(share),
        )
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = perf_counter()
        m.fit(data, max_iter=n_iter, verbose=False)
        if device == "cuda":
            torch.cuda.synchronize()
        assert m.final_ll_ is not None
        return perf_counter() - t0, float(m.final_ll_)

    # CPU-scaling knob (#86): pin intra-op threads for the CPU device (no-op on
    # cuda/mps). set_num_threads is process-global, so save/restore it around this
    # run -- otherwise a pinned count would leak into a later backend/run.
    prev_threads = torch.get_num_threads()
    try:
        if device == "cpu" and threads:
            torch.set_num_threads(int(threads))
        one(min(5, iters))  # warmup (kernel compile / context init)
        times, ll = [], float("nan")
        for _ in range(repeats):
            t, ll = one(iters)
            times.append(t)
    finally:
        torch.set_num_threads(prev_threads)
    return min(times) / iters * 1000.0, ll


def _run_mlx(data, iters, repeats, n_models=1):
    from pamica.mlx_impl import AMICAMLXNG

    def one(n_iter):
        m = AMICAMLXNG(
            n_channels=data.shape[0],
            n_models=n_models,
            n_mix=N_MIX,
            do_newton=False,
            block_size=BLOCK_SIZE,
            seed=SEED,
        )
        t0 = perf_counter()
        m.fit(data, max_iter=n_iter, verbose=False)  # fit's mx.eval syncs
        assert m.final_ll_ is not None
        return perf_counter() - t0, float(m.final_ll_)

    one(min(5, iters))
    times, ll = [], float("nan")
    for _ in range(repeats):
        t, ll = one(iters)
        times.append(t)
    return min(times) / iters * 1000.0, ll


def _run_numpy(data, iters, repeats, n_models=1, share=False, threads=None):
    import logging

    from threadpoolctl import threadpool_limits

    from pamica.numpy_impl.core import AMICA as AMICA_NumPy

    # NumPy uses share_int (torch uses share_iter).
    share_kw: dict[str, Any] = (
        dict(share_comps=True, share_start=5, share_int=8, comp_thresh=0.99)
        if share
        else {}
    )

    def one(n_iter):
        m = AMICA_NumPy(
            num_models=n_models,
            num_mix=N_MIX,
            max_iter=n_iter,
            do_newton=False,
            # Match the fixed block_size and skip AMICA_NumPy's do_opt_block
            # auto-tune, which otherwise runs a wall-clock block-size search
            # INSIDE fit() -- unmatched settings + timing pollution vs torch/mlx.
            block_size=BLOCK_SIZE,
            do_opt_block=False,
            use_tqdm=False,
            verbose=False,
            **share_kw,
        )
        # AMICA_NumPy logs per-iteration progress at INFO on the "AMICA" logger
        # (its constructor resets that logger to INFO). Raise the level to WARNING
        # so the sweep log stays clean WITHOUT swallowing the real divergence/NaN
        # warnings and the non-convergence error, which stay visible on stderr.
        logging.getLogger("AMICA").setLevel(logging.WARNING)
        t0 = perf_counter()
        m.fit(data)
        elapsed = perf_counter() - t0
        # A non-finite final LL is a divergence, not a fast success -- surface it
        # (AMICA_NumPy has NaN'd on long fits historically, #39/#41). Raising here
        # makes it an "error" row instead of a bogus timing with a nan LL.
        if not m.converged:
            raise RuntimeError(
                f"AMICA_NumPy did not converge (non-finite LL) at "
                f"{data.shape[0]}ch x {data.shape[1]} samples, {n_iter} iters"
            )
        # AMICA_NumPy.ll is already per-sample-per-channel normalized (issue
        # #212), the same scale the torch/mlx backends report, so the results
        # column is directly comparable (numpy-f64 matches torch-cpu-f64 to ~5
        # digits, as the parity suite guarantees) with no rescaling here.
        ll = float(m.ll[-1])
        return elapsed, ll

    # CPU-scaling knob (#86): cap the BLAS/OpenMP thread pool numpy's vectorized
    # ops dispatch to. threadpoolctl limits at runtime (numpy's thread count is
    # otherwise fixed at import from OMP/OPENBLAS env), so a single process can
    # sweep it. limits=None leaves the pool at its default.
    with threadpool_limits(limits=int(threads) if threads else None):
        one(min(3, iters))
        times, ll = [], float("nan")
        for _ in range(repeats):
            t, ll = one(iters)
            times.append(t)
    return min(times) / iters * 1000.0, ll


# --------------------------------------------------------------------------
# Native-Fortran backend (#85): drives an amica15 compiled from source (see
# benchmarks/fortran/build_amica.sh). Timing comes from amica's OWN per-iter
# stamps in out.txt -- startup-immune (init/PCA/sphering happen before iter 1),
# unlike wrapping the whole process in perf_counter.
# --------------------------------------------------------------------------
def _write_fdt(data: np.ndarray, path: Path) -> None:
    """Write (n_channels, n_samples) EEG as amica's raw float32 .fdt: a
    (data_dim=channels, field_dim=samples) array in column-major (channel-
    fastest) order -- the exact inverse of numpy_impl.data.load_data_file's
    ``reshape(..., order="F")``. Byte-identical to EEGLAB's own .fdt (locked by
    test_write_fdt_roundtrip)."""
    path.write_bytes(np.ascontiguousarray(data).astype("<f4").tobytes(order="F"))


def _write_fortran_param(
    work: Path, nc: int, ns: int, iters: int, threads: int, n_models: int
) -> None:
    """Render input.param for one benchmark fit: start from the committed
    template and override only the benchmark knobs so the algorithm config
    matches the other backends (n_mix=3, pdftype=0, do_newton off, fixed
    block_size). use_min_dll/use_grad_norm are forced OFF so amica runs the full
    ``iters`` budget (matched, fixed-length runs like torch/mlx)."""
    overrides = {
        "files": "files ./bench.fdt",
        "outdir": "outdir ./bench_out/",
        "block_size": f"block_size {BLOCK_SIZE}",
        "do_opt_block": "do_opt_block 0",
        "num_models": f"num_models {n_models}",
        "max_threads": f"max_threads {threads}",
        "num_mix_comps": f"num_mix_comps {N_MIX}",
        "pdftype": "pdftype 0",
        "max_iter": f"max_iter {iters}",
        "num_samples": "num_samples 1",
        "data_dim": f"data_dim {nc}",
        "field_dim": f"field_dim {ns}",
        "do_newton": "do_newton 0",
        "do_sphere": "do_sphere 1",
        "do_mean": "do_mean 1",
        "doPCA": "doPCA 1",
        "pcakeep": f"pcakeep {nc}",
        "use_min_dll": "use_min_dll 0",
        "use_grad_norm": "use_grad_norm 0",
        "write_LLt": "write_LLt 0",
        "do_history": "do_history 0",
        "share_comps": "share_comps 0",
    }
    seen: set[str] = set()
    lines: list[str] = []
    for line in _PARAM_TEMPLATE.read_text().splitlines():
        key = line.split()[0] if line.strip() else ""
        if key in overrides:
            lines.append(overrides[key])
            seen.add(key)
        else:
            lines.append(line)
    lines.extend(v for k, v in overrides.items() if k not in seen)
    (work / "input.param").write_text("\n".join(lines) + "\n")


_ITER_RE = re.compile(
    r"iter\s+\d+\s+lrate\s*=\s*\S+\s+LL\s*=\s*(\S+).*?\(\s*([\d.]+)\s*s,"
)


def _parse_fortran_out(out_txt: Path) -> tuple[list[float], float]:
    """Parse amica's out.txt -> (per_iteration_seconds, final_ll). Each iter
    line ends '( <sec> s, <cum_h> h)'; <sec> is that iteration's compute time."""
    secs: list[float] = []
    last_ll = float("nan")
    # errors="replace" (not "ignore") so a corrupt byte surfaces as U+FFFD and
    # breaks that line's regex match cleanly, rather than silently vanishing and
    # risking a still-matching but numerically wrong stamp.
    for line in out_txt.read_text(errors="replace").splitlines():
        m = _ITER_RE.search(line)
        if m:
            last_ll = float(m.group(1))
            secs.append(float(m.group(2)))
    return secs, last_ll


def _run_fortran(
    data, iters, repeats, n_models=1, binary=None, threads=None
) -> tuple[float, float]:
    """Time a native amica fit. Per repeat: fresh workdir, write .fdt + param,
    run the binary, read amica's per-iter seconds. ms/iter = MEAN of the timed
    iters (dropping iter 1 as first-touch warmup), reported as the min over
    repeats.

    amica prints per-iteration time to only ~0.01 s (10 ms) resolution, so the
    mean over the timed iters is used. NOTE this floors precision at ~10 ms: when
    every iteration rounds to the same stamp the mean is still that stamp, so run
    at a size whose per-iteration time is well above 10 ms (the 70-ch benchmark
    is ~30-70 ms) for the number to be meaningful. A config whose per-iteration
    time rounds to 0.00 s (below the floor -- only tiny channel/sample counts)
    raises instead of returning a bogus 0.0. Raises on a nonzero exit too -- a
    crashed run must not masquerade as a fast one (mirrors
    benchmark_runtime.time_fortran)."""
    binary = Path(binary or _DEFAULT_FORTRAN_BIN)
    if not (binary.exists() and os.access(binary, os.X_OK)):
        raise RuntimeError(f"native fortran binary not found/executable: {binary}")
    threads = threads or (os.cpu_count() or 4)
    nc, ns = data.shape
    per_iter_ms: list[float] = []
    final_ll = float("nan")
    for _ in range(repeats):
        with tempfile.TemporaryDirectory(prefix="amica_bench_") as td:
            work = Path(td)
            _write_fdt(data, work / "bench.fdt")
            _write_fortran_param(work, nc, ns, iters, threads, n_models)
            (work / "bench_out").mkdir(exist_ok=True)
            env = {**os.environ, "OMP_NUM_THREADS": str(threads)}
            res = subprocess.run(
                [str(binary), "input.param"],
                cwd=work,
                env=env,
                capture_output=True,
                text=True,
                timeout=_FORTRAN_TIMEOUT,
            )
            if res.returncode != 0:
                # main()'s per-backend handler truncates the exception message to
                # ~70 chars, so print the full captured output here first -- a
                # native crash (bad LAPACK/MPI linkage, malformed param, segfault)
                # has many platform-specific causes and the transcript is the only
                # place to see why.
                print(
                    f"    native fortran exit {res.returncode}\n"
                    f"    --- stderr ---\n{res.stderr[-2000:]}\n"
                    f"    --- stdout ---\n{res.stdout[-2000:]}"
                )
                raise RuntimeError(
                    f"native fortran run failed (exit {res.returncode}); "
                    "see stderr/stdout printed above"
                )
            secs, final_ll = _parse_fortran_out(work / "bench_out" / "out.txt")
            if len(secs) < 2:
                raise RuntimeError(
                    f"native fortran produced <2 timed iterations (got {len(secs)}"
                    "); cannot compute per-iteration timing"
                )
            mean_s = statistics.mean(secs[1:])
            if mean_s == 0.0:
                raise RuntimeError(
                    f"native fortran per-iteration time rounds to 0.00 s for "
                    f"{data.shape[0]}ch x {data.shape[1]} samples -- below amica's "
                    "~10 ms out.txt stamp resolution; use a larger channel/sample "
                    "count (the 70-ch benchmark size is well above the floor)"
                )
            per_iter_ms.append(mean_s * 1000.0)
    return min(per_iter_ms), final_ll


def _fortran_available(binary=None) -> bool:
    binary = Path(binary or _DEFAULT_FORTRAN_BIN)
    return binary.exists() and os.access(binary, os.X_OK)


_BACKENDS = {
    "numpy-cpu-f64": ("numpy", {}),
    "torch-cpu-f64": ("torch", {"device": "cpu", "dtype_str": "f64"}),
    "torch-cpu-f32": ("torch", {"device": "cpu", "dtype_str": "f32"}),
    "torch-mps-f32": ("torch", {"device": "mps", "dtype_str": "f32"}),
    "torch-cuda-f64": ("torch", {"device": "cuda", "dtype_str": "f64"}),
    "torch-cuda-f32": ("torch", {"device": "cuda", "dtype_str": "f32"}),
    "mlx-f32": ("mlx", {}),
    "native-fortran-f64": ("fortran", {}),
}


def _available(
    name: str, n_models: int = 1, share: bool = False, fortran_bin=None
) -> bool:
    kind, kw = _BACKENDS[name]
    if kind == "torch":
        return _torch_available(kw["device"])
    if kind == "mlx":
        # AMICAMLXNG supports single- and multi-model (#81); sharing not yet.
        if share:
            return False
        try:
            # mlx is a compiled extension with no type stubs; ty cannot resolve
            # it statically even when installed.
            import mlx.core as mx  # ty: ignore[unresolved-import]

            return mx.default_device().type == mx.DeviceType.gpu
        except Exception:
            return False
    if kind == "fortran":
        # amica15 supports num_models > 1, but this adapter is only validated
        # single-model in Phase 1 (#85); component sharing is hardcoded off (not
        # wired here), so only the share config is gated out.
        return not share and _fortran_available(fortran_bin)
    if kind == "numpy":
        return True
    return False


def _is_cpu(name: str) -> bool:
    """A backend whose runtime is set by CPU thread count (swept by --threads)."""
    kind, kw = _BACKENDS[name]
    if kind in ("numpy", "fortran"):
        return True
    if kind == "torch":
        return kw.get("device") == "cpu"
    return False


def _run_backend(
    name,
    data,
    iters,
    repeats,
    n_models=1,
    share=False,
    fortran_bin=None,
    threads=None,
):
    # threads: CPU thread count for this run (None = backend default). Applied to
    # torch-cpu (set_num_threads), numpy (threadpool_limits), native-fortran (OMP);
    # ignored by the GPU backends.
    kind, kw = _BACKENDS[name]
    if kind == "torch":
        return _run_torch(
            data,
            iters=iters,
            repeats=repeats,
            n_models=n_models,
            share=share,
            threads=threads,
            **kw,
        )
    if kind == "mlx":
        return _run_mlx(data, iters=iters, repeats=repeats, n_models=n_models)
    if kind == "fortran":
        return _run_fortran(
            data,
            iters=iters,
            repeats=repeats,
            n_models=n_models,
            binary=fortran_bin,
            threads=threads,
        )
    return _run_numpy(
        data,
        iters=iters,
        repeats=repeats,
        n_models=n_models,
        share=share,
        threads=threads,
    )


def _platform_info() -> dict:
    info: dict[str, Any] = {"machine": platform.machine(), "system": platform.system()}
    info["nproc"] = os.cpu_count()  # context for the CPU-scaling sweep (#86)
    try:
        import torch

        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    try:
        info["loadavg"] = [round(x, 1) for x in os.getloadavg()]
    except OSError:
        pass
    return info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", help="path to the (70, n_samples) real-EEG .npy")
    ap.add_argument("--channels", default="16,32,48,70")
    ap.add_argument(
        "--montage",
        help="BIDS electrodes.tsv; when given, reduced channel counts use "
        "spatially-distributed (whole-head) subsets instead of the first N (#91)",
    )
    ap.add_argument("--samples", type=int, default=30000)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--backends", default="auto")
    ap.add_argument("--out", help="write results JSON here")
    ap.add_argument("--report", nargs="+", help="merge these result JSONs into tables")
    ap.add_argument("--n-models", type=int, default=1, dest="n_models")
    ap.add_argument(
        "--share", action="store_true", help="enable component sharing (n_models>1)"
    )
    ap.add_argument(
        "--fortran-bin",
        default=str(_DEFAULT_FORTRAN_BIN),
        dest="fortran_bin",
        help="native amica binary (default benchmarks/fortran/amica15; build it "
        "with benchmarks/fortran/build_amica.sh)",
    )
    ap.add_argument(
        "--fortran-threads",
        type=int,
        default=None,
        dest="fortran_threads",
        help="OMP threads for the native-fortran backend when NOT sweeping "
        "(default: all cores). Overridden by --threads.",
    )
    ap.add_argument(
        "--threads",
        default=None,
        help="CPU core-count scaling sweep (#86): comma-separated thread counts "
        "(e.g. 4,8,16,32). Applied to the CPU backends (torch-cpu, numpy, "
        "native-fortran); GPU backends run once, thread-independent.",
    )
    args = ap.parse_args()

    if args.report:
        _report(args.report)
        return 0

    if not args.data:
        ap.error("--data is required unless --report is given")

    n_models, share = args.n_models, args.share
    config = f"m{n_models}" + ("+share" if share else "")  # e.g. m1, m2, m2+share

    full = np.load(args.data).astype(np.float64)
    channels = [int(c) for c in args.channels.split(",") if int(c) <= full.shape[0]]
    if args.backends == "auto":
        backends = [
            b for b in _BACKENDS if _available(b, n_models, share, args.fortran_bin)
        ]
    else:
        requested = args.backends.split(",")
        backends = [
            b for b in requested if _available(b, n_models, share, args.fortran_bin)
        ]
        # Do not silently drop an explicitly requested backend: an unbuilt
        # native-fortran binary or a mistyped/unavailable device would otherwise
        # leave the sweep quietly benchmarking fewer backends than asked (or
        # nothing at all) while still exiting 0.
        for b in requested:
            if b not in _BACKENDS:
                print(f"WARNING: unknown backend {b!r} (skipped)")
            elif b not in backends:
                print(
                    f"WARNING: requested backend {b!r} is unavailable on this host "
                    f"(config m{n_models}{'+share' if share else ''}"
                    + (
                        f"; native binary {args.fortran_bin!r} not built?"
                        if b == "native-fortran-f64"
                        else ""
                    )
                    + ") -- skipped"
                )

    thread_sweep = [int(t) for t in args.threads.split(",")] if args.threads else None
    if thread_sweep and any(t < 1 for t in thread_sweep):
        ap.error("--threads values must be >= 1")

    info = _platform_info()
    print(f"platform: {info}")
    print(
        f"data {full.shape} | config {config} | channels {channels} | "
        f"samples {args.samples} | iters {args.iters} x {args.repeats} | {backends}"
        + (f" | threads {thread_sweep}" if thread_sweep else "")
    )

    def _thread_points(b):
        # CPU backends sweep the thread list; GPU backends run once (thread-
        # independent, recorded as threads=None). Without --threads, fortran runs
        # at its resolved OMP count (so the row records the real thread count, not
        # None -- None is reserved for genuinely thread-independent GPU rows), and
        # torch/numpy use their library default (None).
        if thread_sweep and _is_cpu(b):
            return thread_sweep
        if b == "native-fortran-f64":
            return [args.fortran_threads or (os.cpu_count() or 4)]
        return [None]

    rows = []
    for nc in channels:
        # #91: with a montage, time whole-head distributed subsets rather than the
        # first nc electrodes (a spatial cluster). Channel count is unchanged, so the
        # timing is unaffected; this only makes the reduced montages physical.
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
        # a montage with fewer localized electrodes than requested yields fewer
        # channels than nc; record/print the actual count so the sweep is honest.
        n_sel = len(sel)
        data = np.ascontiguousarray(full[sel][:, : args.samples])
        for b in backends:
            for t in _thread_points(b):
                tag = f"@{t}t" if t is not None else ""
                try:
                    ms, ll = _run_backend(
                        b,
                        data,
                        args.iters,
                        args.repeats,
                        n_models,
                        share,
                        fortran_bin=args.fortran_bin,
                        threads=t,
                    )
                    rows.append(
                        {
                            "config": config,
                            "channels": n_sel,
                            "backend": b,
                            "threads": t,
                            "ms_per_iter": ms,
                            "final_ll": ll,
                        }
                    )
                    print(
                        f"  [{config}] {n_sel:3d}ch {b:18s}{tag:5s} "
                        f"{ms:9.2f} ms/it   LL={ll:+.5f}"
                    )
                except Exception as exc:  # noqa: BLE001 - report and continue
                    print(
                        f"  [{config}] {n_sel:3d}ch {b:18s}{tag:5s} "
                        f"FAILED: {type(exc).__name__}: {str(exc)[:60]}"
                    )
                    rows.append(
                        {
                            "config": config,
                            "channels": n_sel,
                            "backend": b,
                            "threads": t,
                            "error": str(exc)[:120],
                        }
                    )

    out = {"platform": info, "config": vars(args), "rows": rows}
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"wrote {args.out}")
    return 0


def _report(paths):
    """Merge per-platform JSONs into ms/it + LL tables, one block per config.
    Thread-less rows keep the channels x backend tables; --threads sweep rows
    (#86) get an extra CPU-scaling block (threads x backend, per channel count)."""
    # merged[config][channels][backend] = row  (summary view: keep the fastest row
    # seen per backend, thread-less or swept -- see the ms_per_iter comparison below)
    merged: dict = {}
    # scaling[config][channels][threads][backend] = row  (thread-swept rows only)
    scaling: dict = {}
    # errored[config][backend] = count  (so an all-failing backend is not invisible)
    errored: dict = {}
    ok_backends: dict = {}
    plats = []
    for p in paths:
        with open(p) as f:
            d = json.load(f)
        plats.append(d.get("platform", {}))
        for r in d["rows"]:
            cfg = r.get("config", "m1")  # rows from before the config axis => m1
            if "error" in r:
                errored.setdefault(cfg, {})[r["backend"]] = (
                    errored.setdefault(cfg, {}).get(r["backend"], 0) + 1
                )
                continue
            ok_backends.setdefault(cfg, set()).add(r["backend"])
            th = r.get("threads")  # None for GPU/thread-less rows
            if th is not None:
                scaling.setdefault(cfg, {}).setdefault(r["channels"], {}).setdefault(
                    th, {}
                )[r["backend"]] = r
            slot = merged.setdefault(cfg, {}).setdefault(r["channels"], {})
            prev = slot.get(r["backend"])
            # summary view keeps the fastest CPU number per backend across the swept
            # thread counts (the "best cores can do" vs the GPU)
            cur_ms = r.get("ms_per_iter")
            prev_ms = prev.get("ms_per_iter") if prev else None
            if prev is None or (
                cur_ms is not None and (prev_ms is None or cur_ms < prev_ms)
            ):
                slot[r["backend"]] = r

    for cfg in sorted(merged):
        by_ch = merged[cfg]
        backends = sorted({b for ch in by_ch.values() for b in ch})
        chans = sorted(by_ch)

        def table(field, fmt):
            hdr = "channels | " + " | ".join(f"{b:>18}" for b in backends)
            print(hdr)
            print("-" * len(hdr))
            for c in chans:
                cells = [
                    f"{format(by_ch[c][b][field], fmt):>18}"
                    if b in by_ch[c] and by_ch[c][b].get(field) is not None
                    else f"{'-':>18}"
                    for b in backends
                ]
                print(f"{c:8d} | " + " | ".join(cells))

        print(f"\n########## config: {cfg} ##########")
        print("=== performance: ms / iteration (fastest CPU thread count) ===")
        table("ms_per_iter", ".2f")
        print("=== results: converged log-likelihood ===")
        table("final_ll", "+.5f")

        # CPU core-count scaling (#86): threads x backend, one block per channels
        if cfg in scaling:
            for c in sorted(scaling[cfg]):
                by_t = scaling[cfg][c]
                cpu_bes = sorted({b for t in by_t.values() for b in t})
                print(f"\n=== CPU scaling: ms/iteration @ {c} channels ===")
                hdr = "threads  | " + " | ".join(f"{b:>18}" for b in cpu_bes)
                print(hdr)
                print("-" * len(hdr))
                for t in sorted(by_t):
                    cells = [
                        f"{by_t[t][b]['ms_per_iter']:>18.2f}"
                        if b in by_t[t] and by_t[t][b].get("ms_per_iter") is not None
                        else f"{'-':>18}"
                        for b in cpu_bes
                    ]
                    print(f"{t:8d} | " + " | ".join(cells))
                # GPU reference line (thread-less) so you can see where cores catch up
                gpu = {
                    b: r
                    for b, r in merged[cfg].get(c, {}).items()
                    if not _is_cpu(b) and r.get("ms_per_iter") is not None
                }
                for b, r in sorted(gpu.items()):
                    print(f"   [gpu] {b}: {r['ms_per_iter']:.2f} ms/it")

        # Surface backends that errored on EVERY attempted row -- otherwise they
        # would silently vanish from the tables (indistinguishable from "not run").
        all_failed = {
            b: n
            for b, n in errored.get(cfg, {}).items()
            if b not in ok_backends.get(cfg, set())
        }
        if all_failed:
            print("=== failed (errored on all attempts; see raw JSON) ===")
            for b, n in sorted(all_failed.items()):
                print(f"   {b}: FAILED x{n}")
    print(f"\nplatforms: {plats}")


if __name__ == "__main__":
    raise SystemExit(main())
