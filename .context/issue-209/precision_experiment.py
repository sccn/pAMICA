#!/usr/bin/env python
"""Matched-stopping precision experiment prep (issue #209).

The prior float32-vs-float64 numbers in ``docs/guides/validation.md`` (the
k=152 rows, ``.context/issue-90/ksweep_findings.md:56``) confound precision
with stopping: the f32 arm hit the natural-gradient lrate floor at iter 1735
and stopped early, so it was compared against a full-2000-iter f64 run. This
script removes that confound. Per the issue's design-decision comment, every
arm runs to an IDENTICAL fixed iteration budget with every early-stop path
disabled (or, for the unconstrained diagnostic below, left at its default),
so precision is the only variable in the precision contrast.

Two separate contrasts (never mixed in one row):
  * precision       same implementation (AMICATorchNG), same device, same
                     seed/init/block_size -- f32 vs f64.
  * implementation  same precision (f64) -- native Fortran (amica15) vs
                     AMICATorchNG, CPU only (the reference binary has no GPU
                     path). Skipped automatically if the binary at
                     --fortran-bin is missing, not executable, or fails to
                     run (e.g. a macOS Mach-O binary on a Linux CUDA host).

Per the issue's second comment, the early-stop *behavior* is itself a
candidate float32 finding, not noise to engineer away, so this script reports
BOTH:
  1. matched   -- every arm forced to the full budget, no self-termination
                  (the clean contrast the validation.md numbers should rest
                  on). Final LL (three flavors: the returned/keep_best value,
                  the raw last-iteration value, and the trajectory max, so a
                  late-iteration overshoot is visible) + Hungarian-matched
                  component correlation.
  2. unconstrained -- the SAME seed/init/precision run again with the
                  backend's default stopping criteria left ON, capped at the
                  largest requested budget, recording whether/when/why it
                  would have stopped on its own.

Suppressing early stopping without patching AMICATorchNG (the open question
in the issue thread): use_min_dll=False and use_grad_norm=False turn off the
two standalone per-iteration stop checks; minlrate=0.0 and min_nd=-1.0 make
the *unconditional* decrease-branch checks unreachable too (lrate stays > 0
under repeated multiplicative decay; a vector norm is never < 0). All four
are public AMICATorchNG constructor kwargs. The Fortran binary takes the
mirror-image four param-file overrides (see ``_write_fortran_param``); amica15
reads min_grad_norm with a fixed-decimal F15.12 field, so 0 there (not a
negative literal) is the safe unreachable floor.

Usage (bundled 32-channel sample, CPU, small budget -- the local smoke check):
    uv run python .context/issue-209/precision_experiment.py \\
        --device cpu --budgets 20 --seeds 0 --k-values 5

Usage (the real run, on the CUDA box -- see .context/issue-209/README.md):
    uv run python .context/issue-209/precision_experiment.py --device cuda

Real EEG only (bundled ``pamica/sample_data/eeglab_data.fdt``, or a larger
recording passed via --data as a (channels, frames) float .npy -- e.g. the
ds002718 sub-002 recording used in ``.context/issue-90/ksweep_findings.md``,
not bundled). No synthetic data.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
_SAMPLE_DIR = _REPO_ROOT / "pamica" / "sample_data"
_BUNDLED_FDT = _SAMPLE_DIR / "eeglab_data.fdt"
_BUNDLED_CHANNELS = 32
_BUNDLED_FRAMES = 30504
_DEFAULT_FORTRAN_BIN = _SAMPLE_DIR / "amica15mac"
_PARAM_TEMPLATE = _SAMPLE_DIR / "input.param"
_FORTRAN_TIMEOUT = 3600  # seconds; a crashed/hung run must not hang the sweep

_DEFAULT_BUDGETS = "2000,6000"  # issue #209 comment: matched to a fixed budget
_DEFAULT_SEEDS = "0,1,2"
_DEFAULT_K_VALUES = "10,20,30"  # k = frames / channels**2


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def _load_data(data_path: str | None, channels: int | None) -> tuple[np.ndarray, str]:
    """Real EEG only. Default: the bundled 32-channel sample. ``--data`` is the
    "optional larger-data hook": a (channels, frames) float array saved with
    ``np.save`` (e.g. a larger real recording, not bundled -- see README)."""
    from pamica.torch_impl.utils import load_eeglab_data

    if data_path is None:
        full = load_eeglab_data(
            str(_BUNDLED_FDT), data_dim=_BUNDLED_CHANNELS, field_dim=_BUNDLED_FRAMES
        ).astype(np.float64)
        source = f"bundled:{_BUNDLED_FDT.relative_to(_REPO_ROOT)}"
    else:
        full = np.load(data_path).astype(np.float64)
        source = f"external:{data_path}"
    if channels is not None and channels < full.shape[0]:
        full = np.ascontiguousarray(full[:channels])
    return full, source


def _frame_count(k: float, channels: int, available: int) -> tuple[int, float]:
    """Frame count for a target k = frames/channels**2, clamped to what the
    data provides (and floored at 4x channels so a fit is never absurdly
    underdetermined). Returns (frames, the k actually achieved)."""
    frames = min(int(round(k * channels**2)), available)
    frames = max(frames, min(channels * 4, available))
    return frames, frames / channels**2


# --------------------------------------------------------------------------
# Component correlation: Hungarian-matched |cosine similarity| of unmixing
# rows (issue #90 / validate_implementations.py convention -- row-normalize,
# abs handles the ICA sign ambiguity, linear_sum_assignment handles the
# permutation).
# --------------------------------------------------------------------------
def _match_correlation(w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
    from scipy.optimize import linear_sum_assignment

    n1 = w1 / (np.linalg.norm(w1, axis=1, keepdims=True) + 1e-12)
    n2 = w2 / (np.linalg.norm(w2, axis=1, keepdims=True) + 1e-12)
    corr = np.abs(n1 @ n2.T)
    row, col = linear_sum_assignment(1.0 - corr)
    return corr[row, col]


# --------------------------------------------------------------------------
# torch backend (AMICATorchNG)
# --------------------------------------------------------------------------
def _fit_torch(
    data: np.ndarray,
    *,
    dtype_str: str,
    seed: int,
    device: str,
    max_iter: int,
    block_size: int,
    n_mix: int,
    do_newton: bool,
    matched: bool,
) -> dict[str, Any]:
    import torch

    from pamica.torch_impl import AMICATorchNG

    dtype = torch.float64 if dtype_str == "f64" else torch.float32
    kwargs: dict[str, Any] = dict(
        n_channels=data.shape[0],
        n_models=1,
        n_mix=n_mix,
        device=device,
        dtype=dtype,
        do_newton=do_newton,
        block_size=block_size,
        seed=seed,
    )
    if matched:
        # See module docstring: this is the full suppression of every
        # AMICATorchNG early-stop path, using only public constructor kwargs.
        kwargs.update(use_min_dll=False, use_grad_norm=False, minlrate=0.0, min_nd=-1.0)
    m = AMICATorchNG(**kwargs)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    m.fit(data, max_iter=max_iter, verbose=False)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    ll_hist = np.asarray(m.ll_history, dtype=float)
    degenerate = m.stop_reason in AMICATorchNG._DEGENERATE_STOP_REASONS
    return {
        "final_ll": float(m.final_ll_) if m.final_ll_ is not None else float("nan"),
        "last_iter_ll": float(ll_hist[-1]) if ll_hist.size else float("nan"),
        "max_ll": float(ll_hist.max()) if ll_hist.size else float("nan"),
        "n_iters": int(ll_hist.size),
        "stop_reason": m.stop_reason,
        "degenerate": bool(degenerate),
        "wall_s": elapsed,
        "W": None if degenerate else m.get_unmixing_matrix(0),
    }


# --------------------------------------------------------------------------
# Fortran reference binary (amica15). CPU only -- the reference has no GPU
# path, so this backend runs the same regardless of --device.
# --------------------------------------------------------------------------
def _write_fdt(data: np.ndarray, path: Path) -> None:
    """amica's raw float32 .fdt: (channels, frames), column-major (channel-
    fastest), the exact convention ``load_eeglab_data`` reads back."""
    path.write_bytes(np.ascontiguousarray(data).astype("<f4").tobytes(order="F"))


def _write_fortran_param(
    work: Path,
    nc: int,
    ns: int,
    iters: int,
    threads: int,
    n_mix: int,
    block_size: int,
    do_newton: bool,
    matched: bool,
) -> None:
    """Render input.param from the committed template, overriding only the
    experiment knobs. When ``matched``, every early-stop path is disabled --
    the Fortran mirror of the four AMICATorchNG kwargs in ``_fit_torch``
    (amica15.f90:1058's unconditional ``(lrate .le. minlrate) .or. (ndtmpsum
    .le. min_nd)`` decrease-branch check, plus the standalone
    use_min_dll/use_grad_norm blocks at :1078-1097)."""
    overrides = {
        "files": "files ./run.fdt",
        "outdir": "outdir ./run_out/",
        "block_size": f"block_size {block_size}",
        "do_opt_block": "do_opt_block 0",
        "num_models": "num_models 1",
        "max_threads": f"max_threads {threads}",
        "num_mix_comps": f"num_mix_comps {n_mix}",
        "pdftype": "pdftype 0",
        "max_iter": f"max_iter {iters}",
        "num_samples": "num_samples 1",
        "data_dim": f"data_dim {nc}",
        "field_dim": f"field_dim {ns}",
        "do_newton": f"do_newton {1 if do_newton else 0}",
        "do_sphere": "do_sphere 1",
        "do_mean": "do_mean 1",
        "doPCA": "doPCA 1",
        "pcakeep": f"pcakeep {nc}",
        "write_LLt": "write_LLt 0",
        "do_history": "do_history 0",
        "share_comps": "share_comps 0",
        "do_reject": "do_reject 0",
    }
    if matched:
        overrides["use_min_dll"] = "use_min_dll 0"
        overrides["use_grad_norm"] = "use_grad_norm 0"
        # Format-field floor, not a negative literal: minlrate is read with an
        # E15.3 field (scientific notation is fine) but min_grad_norm with a
        # fixed-decimal F15.12 field, which does not reliably round-trip a
        # negative value. 0 is already unreachable: lrate stays > 0 under
        # repeated multiplicative decay and ndtmpsum (a vector norm) is never
        # < 0, so ``<= 0`` requires an exact-zero hit, same reasoning as the
        # torch side's minlrate=0.0/min_nd=-1.0.
        overrides["minlrate"] = "minlrate 0.000e+00"
        overrides["min_grad_norm"] = "min_grad_norm 0.000000000000"
    else:
        overrides["use_min_dll"] = "use_min_dll 1"
        overrides["use_grad_norm"] = "use_grad_norm 1"
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


_ITER_RE = re.compile(r"iter\s+(\d+)\s+lrate\s*=\s*\S+\s+LL\s*=\s*(\S+)")
_STOP_MESSAGES = (
    ("Got NaN", "nan_ll"),
    ("likelihood increasing by less than", "min_dll"),
    ("norm of weight gradient less than", "grad_norm"),
    # amica15.f90:1060 prints this one shared message for BOTH the
    # lrate<=minlrate and ndtmpsum<=min_nd decrease-branch conditions -- the
    # Fortran source does not disambiguate them the way AMICATorchNG's
    # separate "lrate_floor"/"grad_norm_floor" messages do.
    ("minimum change threshold met", "lrate_or_gradnorm_floor"),
)


def _parse_fortran_run(text: str, max_iter: int) -> tuple[int, float, str]:
    n_iters, final_ll = 0, float("nan")
    for m in _ITER_RE.finditer(text):
        n_iters = int(m.group(1))
        final_ll = float(m.group(2))
    stop_reason = "max_iter"
    for needle, reason in _STOP_MESSAGES:
        if needle in text:
            stop_reason = reason
            break
    return n_iters, final_ll, stop_reason


def fortran_available(binary: str | Path) -> tuple[bool, str]:
    binary = Path(binary)
    if not binary.exists():
        return False, f"binary not found: {binary}"
    if not os.access(binary, os.X_OK):
        return False, f"binary not executable: {binary}"
    return True, ""


def _fit_fortran(
    data: np.ndarray,
    *,
    max_iter: int,
    binary: str | Path,
    threads: int | None,
    n_mix: int,
    block_size: int,
    do_newton: bool,
    matched: bool,
) -> dict[str, Any]:
    ok, reason = fortran_available(binary)
    if not ok:
        return {"available": False, "reason": reason}
    threads = threads or (os.cpu_count() or 4)
    nc, ns = data.shape
    with tempfile.TemporaryDirectory(prefix="amica_precision_") as td:
        work = Path(td)
        _write_fdt(data, work / "run.fdt")
        _write_fortran_param(
            work, nc, ns, max_iter, threads, n_mix, block_size, do_newton, matched
        )
        (work / "run_out").mkdir(exist_ok=True)
        env = {**os.environ, "OMP_NUM_THREADS": str(threads)}
        t0 = time.perf_counter()
        try:
            res = subprocess.run(
                [str(binary), "input.param"],
                cwd=work,
                env=env,
                capture_output=True,
                text=True,
                timeout=_FORTRAN_TIMEOUT,
            )
        except OSError as exc:  # wrong binary format (e.g. mac Mach-O on Linux)
            return {"available": False, "reason": f"exec failed: {exc}"}
        elapsed = time.perf_counter() - t0
        if res.returncode != 0:
            return {
                "available": False,
                "reason": f"exit {res.returncode}: {res.stderr[-300:].strip()}",
            }
        out_txt = work / "run_out" / "out.txt"
        text = out_txt.read_text(errors="replace") if out_txt.exists() else ""
        n_iters, final_ll, stop_reason = _parse_fortran_run(text, max_iter)
        w_path = work / "run_out" / "W"
        w = (
            np.fromfile(w_path, dtype=np.float64).reshape(nc, nc, order="F")
            if w_path.exists()
            else None
        )
    return {
        "available": True,
        "final_ll": final_ll,
        "n_iters": n_iters,
        "stop_reason": stop_reason,
        # amica15's do_opt_block=0 blocking loop silently produces an all-NaN
        # gradient / LL stuck at exactly 0.0 (never a "Got NaN" print, so
        # stop_reason alone reads as a clean "max_iter") if block_size exceeds
        # the frame count -- observed while smoke-testing this script at a
        # too-small k. Callers clamp block_size to <= frames precisely to
        # avoid this, but flag it defensively too: a real fit's LL on EEG data
        # is never exactly 0.0.
        "degenerate": stop_reason == "nan_ll" or final_ll == 0.0,
        "wall_s": elapsed,
        "W": w,
    }


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def _config_header(args: argparse.Namespace, channels: int, source: str) -> dict:
    import torch

    info: dict[str, Any] = {
        "device": args.device,
        "block_size": args.block_size,
        "n_mix": args.n_mix,
        "do_newton": args.do_newton,
        "budgets": args.budgets,
        "seeds": args.seeds,
        "k_values": args.k_values,
        "channels": channels,
        "data_source": source,
        "torch_version": torch.__version__,
        "platform": f"{platform.system()}-{platform.machine()}",
        "nproc": os.cpu_count(),
    }
    if args.device == "cuda" and torch.cuda.is_available():
        info["cuda_device"] = torch.cuda.get_device_name(0)
    return info


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "--device cuda requested but torch.cuda.is_available() is False"
        )

    full, source = _load_data(args.data, args.channels)
    channels, available_frames = full.shape
    print(
        f"data: {source}  channels={channels}  available_frames={available_frames}\n"
        f"device={args.device}  block_size={args.block_size}  n_mix={args.n_mix}  "
        f"do_newton={args.do_newton}"
    )

    fbin_ok, fbin_reason = (
        (False, "skipped via --skip-fortran")
        if args.skip_fortran
        else fortran_available(args.fortran_bin)
    )
    print(
        f"fortran binary: {args.fortran_bin}  "
        + ("available" if fbin_ok else f"UNAVAILABLE ({fbin_reason})")
    )

    matched_rows: list[dict] = []
    precision_corr_rows: list[dict] = []
    implementation_corr_rows: list[dict] = []
    unconstrained_rows: list[dict] = []

    max_budget = max(args.budgets)
    # Mirrors _fit_fortran's own threads-or-cpu_count fallback, needed here to
    # compute the per-thread block-size clamp below before the binary runs.
    threads = args.threads or (os.cpu_count() or 4)
    t_start = time.perf_counter()

    for k in args.k_values:
        frames, k_actual = _frame_count(k, channels, available_frames)
        data = np.ascontiguousarray(full[:, :frames])
        # amica15's fixed-block-size path (do_opt_block=0) silently produces
        # an all-NaN gradient (LL stuck at exactly 0.0, no error, exit 0) if
        # block_size exceeds the frame count -- discovered while smoke-testing
        # this script. torch's own blocking loop already collapses to one
        # block in that case (min(start+block_size, n_samples)), so this
        # clamp only ever changes Fortran's behavior, never torch's.
        block_size = min(args.block_size, frames)
        # amica15 additionally splits `frames` across `max_threads` OpenMP
        # segments BEFORE chunking each segment into `block_size` blocks: if
        # block_size exceeds one segment's length (frames // threads), the
        # same all-NaN failure occurs. Confirmed by bisection (frames=5120,
        # threads=4: block_size=1280 -> a real fit, 1281 -> all-NaN). Torch
        # has no such per-thread split, so only the Fortran call gets this
        # extra clamp.
        fortran_block_size = max(1, min(block_size, frames // threads))
        if block_size != args.block_size:
            print(
                f"  k~{k_actual:.1f}: block_size clamped {args.block_size} -> "
                f"{block_size} (frames={frames})"
            )
        if fortran_block_size != block_size:
            print(
                f"  k~{k_actual:.1f}: fortran block_size further clamped "
                f"{block_size} -> {fortran_block_size} (frames={frames}, "
                f"threads={threads}; amica15 splits frames across threads "
                "before blocking)"
            )
        for seed in args.seeds:
            # --- unconstrained diagnostic: default stops, capped at the
            # largest requested budget. One fit per precision, independent of
            # the "budgets" list (see module docstring, part 2).
            for dtype_str in ("f32", "f64"):
                r = _fit_torch(
                    data,
                    dtype_str=dtype_str,
                    seed=seed,
                    device=args.device,
                    max_iter=max_budget,
                    block_size=block_size,
                    n_mix=args.n_mix,
                    do_newton=args.do_newton,
                    matched=False,
                )
                row = {
                    "k_requested": k,
                    "k_actual": k_actual,
                    "frames": frames,
                    "block_size": block_size,
                    "seed": seed,
                    "precision": dtype_str,
                    "cap_iter": max_budget,
                    "stopped_early": r["n_iters"] < max_budget,
                    **{
                        key: r[key]
                        for key in (
                            "n_iters",
                            "stop_reason",
                            "degenerate",
                            "final_ll",
                            "wall_s",
                        )
                    },
                }
                unconstrained_rows.append(row)
                print(
                    f"  [unconstrained] k~{k_actual:5.1f} seed={seed} {dtype_str} "
                    f"-> stop={r['stop_reason']:<22s} iter={r['n_iters']:5d}/"
                    f"{max_budget}  LL={r['final_ll']:.5f}  ({r['wall_s']:.1f}s)"
                )

            for budget in args.budgets:
                # --- matched: every early-stop path off, fixed budget.
                fits: dict[str, dict] = {}
                for dtype_str in ("f32", "f64"):
                    r = _fit_torch(
                        data,
                        dtype_str=dtype_str,
                        seed=seed,
                        device=args.device,
                        max_iter=budget,
                        block_size=block_size,
                        n_mix=args.n_mix,
                        do_newton=args.do_newton,
                        matched=True,
                    )
                    fits[dtype_str] = r
                    matched_rows.append(
                        {
                            "k_requested": k,
                            "k_actual": k_actual,
                            "frames": frames,
                            "block_size": block_size,
                            "seed": seed,
                            "budget": budget,
                            "implementation": "torch",
                            "precision": dtype_str,
                            "final_ll": r["final_ll"],
                            "last_iter_ll": r["last_iter_ll"],
                            "max_ll": r["max_ll"],
                            "overshoot": r["max_ll"] - r["final_ll"],
                            "stop_reason": r["stop_reason"],
                            "degenerate": r["degenerate"],
                            "wall_s": r["wall_s"],
                        }
                    )
                    print(
                        f"  [matched k~{k_actual:5.1f} seed={seed} budget={budget}] "
                        f"torch-{dtype_str} stop={r['stop_reason']:<10s} "
                        f"LL={r['final_ll']:.5f}  ({r['wall_s']:.1f}s)"
                    )

                if not fits["f32"]["degenerate"] and not fits["f64"]["degenerate"]:
                    corr = _match_correlation(fits["f32"]["W"], fits["f64"]["W"])
                    precision_corr_rows.append(
                        {
                            "k_requested": k,
                            "k_actual": k_actual,
                            "seed": seed,
                            "budget": budget,
                            "mean_abs_corr": float(corr.mean()),
                            "min_abs_corr": float(corr.min()),
                            "ll_gap_f32_minus_f64": (
                                fits["f32"]["final_ll"] - fits["f64"]["final_ll"]
                            ),
                        }
                    )

                # --- implementation: Fortran f64 vs torch f64, same budget.
                if fbin_ok:
                    fr = _fit_fortran(
                        data,
                        max_iter=budget,
                        binary=args.fortran_bin,
                        threads=threads,
                        n_mix=args.n_mix,
                        block_size=fortran_block_size,
                        do_newton=args.do_newton,
                        matched=True,
                    )
                    if fr["available"]:
                        matched_rows.append(
                            {
                                "k_requested": k,
                                "k_actual": k_actual,
                                "frames": frames,
                                "block_size": fortran_block_size,
                                "seed": seed,
                                "budget": budget,
                                "implementation": "fortran",
                                "precision": "f64",
                                "final_ll": fr["final_ll"],
                                "last_iter_ll": fr["final_ll"],
                                "max_ll": fr["final_ll"],
                                "overshoot": 0.0,
                                "stop_reason": fr["stop_reason"],
                                "degenerate": fr["degenerate"],
                                "wall_s": fr["wall_s"],
                            }
                        )
                        print(
                            f"  [matched k~{k_actual:5.1f} seed={seed} budget={budget}] "
                            f"fortran-f64 stop={fr['stop_reason']:<10s} "
                            f"LL={fr['final_ll']:.5f}  ({fr['wall_s']:.1f}s)"
                        )
                        if not fr["degenerate"] and not fits["f64"]["degenerate"]:
                            corr = _match_correlation(fr["W"], fits["f64"]["W"])
                            implementation_corr_rows.append(
                                {
                                    "k_requested": k,
                                    "k_actual": k_actual,
                                    "seed": seed,
                                    "budget": budget,
                                    "mean_abs_corr": float(corr.mean()),
                                    "min_abs_corr": float(corr.min()),
                                    "ll_gap_fortran_minus_torch": (
                                        fr["final_ll"] - fits["f64"]["final_ll"]
                                    ),
                                }
                            )
                    else:
                        print(f"  [matched] fortran skipped: {fr['reason']}")
                        # A hard failure (e.g. wrong exec format on this
                        # platform) fails identically on every other call;
                        # stop retrying and just report it once.
                        fbin_ok = False
                        fbin_reason = fr["reason"]

    elapsed_total = time.perf_counter() - t_start
    fortran_ran = any(r["implementation"] == "fortran" for r in matched_rows)
    config = _config_header(args, channels, source)
    config["fortran_available"] = fortran_ran
    config["fortran_status"] = "ok" if fortran_ran else fbin_reason
    config["wall_s_total"] = elapsed_total
    return {
        "config": config,
        "matched": matched_rows,
        "precision_correlation": precision_corr_rows,
        "implementation_correlation": implementation_corr_rows,
        "unconstrained": unconstrained_rows,
    }


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def _print_summary(results: dict) -> None:
    matched = results["matched"]
    unconstrained = results["unconstrained"]
    n_matched = len(matched)
    n_max_iter = sum(1 for r in matched if r["stop_reason"] == "max_iter")
    n_degenerate = sum(1 for r in matched if r["degenerate"])
    print("\n=== summary ===")
    print(
        f"matched runs: {n_matched}  stopped via max_iter (expected): "
        f"{n_max_iter}/{n_matched}  degenerate: {n_degenerate}"
    )
    if n_matched and n_max_iter != n_matched - n_degenerate:
        print(
            "  WARNING: a non-degenerate matched run did not reach max_iter -- "
            "stop suppression did not fully hold; check the offending row's "
            "stop_reason above before trusting the matched-budget numbers."
        )
    n_early = sum(1 for r in unconstrained if r["stopped_early"])
    print(f"unconstrained runs: {len(unconstrained)}  stopped early: {n_early}")
    reasons: dict[str, int] = {}
    for r in unconstrained:
        if r["stopped_early"]:
            reasons[r["stop_reason"]] = reasons.get(r["stop_reason"], 0) + 1
    if reasons:
        print(f"  early-stop reasons: {reasons}")
    if results["precision_correlation"]:
        corrs = [r["mean_abs_corr"] for r in results["precision_correlation"]]
        print(
            f"precision contrast (f32 vs f64): mean|corr| over "
            f"{len(corrs)} (k,seed,budget) points = {np.mean(corrs):.4f} "
            f"(min {np.min(corrs):.4f})"
        )
    if results["implementation_correlation"]:
        corrs = [r["mean_abs_corr"] for r in results["implementation_correlation"]]
        print(
            f"implementation contrast (fortran-f64 vs torch-f64): mean|corr| "
            f"over {len(corrs)} (k,seed,budget) points = {np.mean(corrs):.4f} "
            f"(min {np.min(corrs):.4f})"
        )
    else:
        print(
            f"implementation contrast: not run ({results['config']['fortran_status']})"
        )


def _json_safe(obj: Any) -> Any:
    """Replace non-finite floats (a degenerate fit's NaN LL) with None so the
    output is strict JSON, not Python's ``NaN``-literal extension."""
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def _write_json(results: dict, path: Path) -> None:
    path.write_text(json.dumps(_json_safe(results), indent=2, sort_keys=True) + "\n")
    print(f"wrote {path}")


def _md_table(rows: list[dict], columns: list[str]) -> list[str]:
    if not rows:
        return ["_(no rows)_"]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for r in rows:
        cells = []
        for c in columns:
            v = r.get(c)
            cells.append(f"{v:.5f}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _write_markdown(results: dict, path: Path) -> None:
    cfg = results["config"]
    lines = [
        "# Matched-stopping precision experiment (issue #209)",
        "",
        f"Config: `{json.dumps(cfg, sort_keys=True)}`",
        "",
        "## Matched-budget contrast (all early stops disabled)",
        "",
        *_md_table(
            results["matched"],
            [
                "k_requested",
                "k_actual",
                "block_size",
                "seed",
                "budget",
                "implementation",
                "precision",
                "final_ll",
                "last_iter_ll",
                "max_ll",
                "overshoot",
                "stop_reason",
                "wall_s",
            ],
        ),
        "",
        "## Precision correlation (torch f32 vs torch f64, matched budget)",
        "",
        *_md_table(
            results["precision_correlation"],
            [
                "k_requested",
                "seed",
                "budget",
                "mean_abs_corr",
                "min_abs_corr",
                "ll_gap_f32_minus_f64",
            ],
        ),
        "",
        "## Implementation correlation (fortran f64 vs torch f64, matched budget)",
        "",
        *_md_table(
            results["implementation_correlation"],
            [
                "k_requested",
                "seed",
                "budget",
                "mean_abs_corr",
                "min_abs_corr",
                "ll_gap_fortran_minus_torch",
            ],
        ),
        "",
        "## Unconstrained (default stopping, capped at max(budgets))",
        "",
        *_md_table(
            results["unconstrained"],
            [
                "k_requested",
                "block_size",
                "seed",
                "precision",
                "cap_iter",
                "n_iters",
                "stopped_early",
                "stop_reason",
                "final_ll",
            ],
        ),
        "",
    ]
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path}")


def _parse_ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def _parse_floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument(
        "--budgets",
        default=_DEFAULT_BUDGETS,
        help="comma-separated matched iteration budgets (default: %(default)s)",
    )
    ap.add_argument(
        "--seeds",
        default=_DEFAULT_SEEDS,
        help="comma-separated seeds, identical init across precisions "
        "(default: %(default)s)",
    )
    ap.add_argument(
        "--k-values",
        default=_DEFAULT_K_VALUES,
        dest="k_values",
        help="comma-separated k = frames/channels**2 targets; the frame count "
        "is derived and clamped to the available data (default: %(default)s)",
    )
    ap.add_argument(
        "--channels",
        type=int,
        default=None,
        help="channels to use (default: all channels in the data)",
    )
    ap.add_argument(
        "--data",
        default=None,
        help="optional (channels, frames) float .npy for a larger real "
        "recording; default is the bundled 32-channel sample",
    )
    ap.add_argument("--block-size", type=int, default=8192, dest="block_size")
    ap.add_argument("--n-mix", type=int, default=3, dest="n_mix")
    ap.add_argument("--do-newton", action="store_true", dest="do_newton")
    ap.add_argument(
        "--fortran-bin", default=str(_DEFAULT_FORTRAN_BIN), dest="fortran_bin"
    )
    ap.add_argument(
        "--skip-fortran",
        action="store_true",
        dest="skip_fortran",
        help="force-skip the implementation contrast",
    )
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument(
        "--out",
        default=str(_SCRIPT_DIR),
        help="directory for the .json/.md outputs (default: next to this script)",
    )
    ap.add_argument(
        "--tag",
        default=None,
        help="output filename tag (default: platform-machine-device)",
    )
    args = ap.parse_args()
    args.budgets = _parse_ints(args.budgets)
    args.seeds = _parse_ints(args.seeds)
    args.k_values = _parse_floats(args.k_values)

    results = run_experiment(args)
    _print_summary(results)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"{platform.system()}-{platform.machine()}-{args.device}"
    _write_json(results, out_dir / f"precision_experiment_{tag}.json")
    _write_markdown(results, out_dir / f"precision_experiment_{tag}.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
