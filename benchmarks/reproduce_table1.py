#!/usr/bin/env python
"""Reproduce paper.md's Table 1 parity numbers against the Fortran reference
(issue #144): a single, portable entry point, replacing the workstation-
specific ``run_5seed_newton0.sh`` that used to live under
``.context/issue-144-parity-data-adequacy/`` (hardcoded ``cd
~/pamica-issue144``, ``/tmp`` paths, a 24-thread count, and an assumed CUDA
GPU).

Two tiers, because Table 1's rows need different data:

* **bundled** -- ``pamica/sample_data/eeglab_data.fdt`` (32 channels, 30504
  frames, k~30), no download. Covers the rows the paper attributes to the
  bundled sample: Amari distance, the score-function/sufficient-statistics
  bit-exactness check, and the four multi-model ensemble rows.
* **external** -- a real, well-determined recording (OpenNeuro ds002718
  sub-002, 70 channels, k~153; manual download, see
  ``benchmarks/README_dimsweep.md``). Covers the headline single-model
  log-likelihood and component-correlation rows, which the bundled sample is
  too short to settle on its own (below k~60 the decomposition is
  under-determined and backends diverge for legitimate reasons -- see
  ``.context/issue-90/ksweep_findings.md``).

The Fortran side runs through :class:`pamica.AMICANative`, which resolves the
right platform binary via :mod:`pamica.native.resolver` (downloading and
SHA-256-checking a release asset on first use, or honoring
``PAMICA_NATIVE_BINARY``/``--fortran-binary``), so this works on Linux and
Windows too, not just the bundled macOS ``amica15mac`` fixture. The compute
device is auto-detected (CUDA > MPS > CPU) with a printed fallback to CPU
when the parity dtype (float64) is not representable on the detected device
(MPS has no float64 support).

**Cost**: the full external-tier protocol (5 sequential Fortran fits at 2000
iterations on a 70-channel, 747k-frame recording, plus 5 matching Python
fits) is hours, not minutes, even on a fast machine -- see
``benchmarks/README_dimsweep.md`` ("Reproducing Table 1") for measured
wall-clock, download size, and what changes without a GPU before running it.
There is no cheaper reduced-iteration substitute for that tier: below k~60
the decomposition is under-determined, so a shorter run would reproduce a
noisier number, not a faster version of the same one.

Usage
-----
    # Bundled tier (no download): Amari distance, score/sufficient-statistics
    # bit-exactness, and the four multi-model rows. ~minutes to ~an hour
    # depending on --n-seeds/--max-iter/--multimodel-runs.
    uv run python benchmarks/reproduce_table1.py --tier bundled

    # External tier (manual download required; hours -- see
    # benchmarks/README_dimsweep.md before running):
    uv run python benchmarks/reproduce_table1.py --tier external \\
        --data benchmarks/data/ds002718_sub-002_eeg70_full.npy

    # Both tiers back to back.
    uv run python benchmarks/reproduce_table1.py --tier both \\
        --data benchmarks/data/ds002718_sub-002_eeg70_full.npy

    # A fast smoke test (NOT the paper's protocol -- fewer seeds/iterations
    # so the pipeline can be checked end to end in minutes; the printed
    # numbers will be noisier than Table 1's, especially the multi-model
    # rows, which are estimator-spread-dominated even at full budget):
    uv run python benchmarks/reproduce_table1.py --tier bundled \\
        --n-seeds 2 --max-iter 100 --multimodel-runs 3 --multimodel-max-iter 30
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from scipy import stats
from scipy.optimize import linear_sum_assignment

from pamica import AMICA, AMICANative
from pamica.native import resolver
from pamica.numpy_impl.core import AMICA as AMICA_NumPy
from pamica.torch_impl import AMICATorchNG
from pamica.torch_impl.core import _log_pdf_and_deriv, _score
from pamica.torch_impl.utils import load_eeglab_data

REPO = Path(__file__).resolve().parent.parent
SAMPLE_DIR = REPO / "pamica" / "sample_data"
BUNDLED_FDT = SAMPLE_DIR / "eeglab_data.fdt"
BUNDLED_PARAMS_JSON = SAMPLE_DIR / "sample_params.json"
DEFAULT_OUT_DIR = REPO / "benchmarks" / "results" / "reproduce_table1"

# Protocol defaults matching docs/guides/validation.md's "Reproducing these
# results" section (the same config that produced Table 1's numbers): 5-seed,
# 2000-iteration single-model sweeps; a 20-run, 100-iteration multi-model
# ensemble. Seed ranges are disjoint from other reproduction scripts in the
# repo only to avoid confusing cross-references in logs, not for correctness.
DEFAULT_N_SEEDS = 5
DEFAULT_SINGLE_MAX_ITER = 2000
DEFAULT_ENSEMBLE_N = 20
DEFAULT_ENSEMBLE_MAX_ITER = 100
BUNDLED_SEED_START = 301
EXTERNAL_SEED_START = 201
MULTIMODEL_SEED_START = 1


# ---------------------------------------------------------------------------
# Parity metrics (Amari, Cichocki & Yang 1996; Hungarian-matched correlation),
# reimplemented here rather than imported from validate_implementations.py --
# the repo's other bundled-sample reproduction scripts
# (.context/issue-144-parity-data-adequacy/bundled_sample_newton0.py,
# .context/issue-27/multimodel_ensemble.py) each define their own copy too,
# rather than pulling in the top-level CLI script as a library.
# ---------------------------------------------------------------------------


def xcorr(Wa: np.ndarray, Wb: np.ndarray) -> np.ndarray:
    """Hungarian-matched |correlation| per component."""
    na = Wa / (np.linalg.norm(Wa, axis=1, keepdims=True) + 1e-12)
    nb = Wb / (np.linalg.norm(Wb, axis=1, keepdims=True) + 1e-12)
    corr = np.abs(na @ nb.T)
    r, c = linear_sum_assignment(1 - corr)
    return corr[r, c]


def corr_metric(Wa: np.ndarray, Wb: np.ndarray) -> float:
    """Scalar mean Hungarian-matched correlation (for the ensemble pairwise
    comparisons, which need one number per run pair, not per component)."""
    return float(xcorr(Wa, Wb).mean())


def _amari_index(gain: np.ndarray) -> float:
    n = gain.shape[0]
    abs_gain = np.abs(gain)
    row_max = abs_gain.max(axis=1)
    col_max = abs_gain.max(axis=0)
    if np.any(row_max == 0) or np.any(col_max == 0):
        raise ValueError("amari_distance: a row or column is all-zero (degenerate W)")
    row_term = (abs_gain.sum(axis=1) / row_max - 1).sum()
    col_term = (abs_gain.sum(axis=0) / col_max - 1).sum()
    return (row_term + col_term) / (2 * n * (n - 1))


def amari_distance(Wa: np.ndarray, Wb: np.ndarray) -> float:
    """Amari distance between two square unmixing matrices: permutation- and
    scale-invariant by construction, so unlike ``xcorr`` it needs no Hungarian
    assignment step. 0 for a perfect match up to row permutation/scaling."""
    forward = _amari_index(Wa @ np.linalg.pinv(Wb))
    backward = _amari_index(Wb @ np.linalg.pinv(Wa))
    return float((forward + backward) / 2)


def model_amari(Wa_stacked: np.ndarray, Wb_stacked: np.ndarray, nw: int) -> float:
    """Best-pairing mean Amari distance between two stacked (2*nw, nw)
    2-model unmixing matrices. Which Fortran model corresponds to which
    pamica model is not identified, so both label pairings are tried and the
    lower-distance one is kept, per run pair (matches
    ``.context/issue-27/amari_distance.py``'s treatment)."""
    Wa = [Wa_stacked[:nw], Wa_stacked[nw:]]
    Wb = [Wb_stacked[:nw], Wb_stacked[nw:]]
    best = math.inf
    for pairing in ((0, 1), (1, 0)):
        ds = [amari_distance(Wa[i], Wb[j]) for i, j in enumerate(pairing)]
        best = min(best, float(np.mean(ds)))
    return best


def pairwise(
    A: np.ndarray, B: np.ndarray, same: bool, metric: Callable[..., float]
) -> np.ndarray:
    """All A[i]-vs-B[j] metric values; ``same=True`` skips the diagonal and
    the lower triangle (A and B are the same ensemble)."""
    return np.array(
        [
            metric(A[i], B[j])
            for i in range(len(A))
            for j in range(len(B))
            if not (same and j <= i)
        ]
    )


def perm_test_not_worse(
    Fs: np.ndarray,
    Gs: np.ndarray,
    metric: Callable[..., float],
    higher_is_worse: bool,
    n_perm: int = 20000,
    seed: int = 0,
) -> float:
    """Run-level permutation test: is between-implementation agreement worse
    than Fortran's own run-to-run agreement? The pairwise values are not
    independent (each of the ``2N`` runs appears in ``~2N-1`` pairs), so a
    Mann-Whitney/TOST on the pairwise values directly would be
    pseudoreplicated; this permutes the runs as intact units instead. Returns
    the one-sided p-value for "between is worse than within-Fortran".
    """
    rng = np.random.default_rng(seed)
    allW = np.concatenate([Fs, Gs], axis=0)
    m = len(allW)
    n = len(Fs)
    P = np.zeros((m, m))
    for i in range(m):
        for j in range(i + 1, m):
            P[i, j] = P[j, i] = metric(allW[i], allW[j])

    sign = 1.0 if higher_is_worse else -1.0

    def gap(mask: np.ndarray) -> float:
        a = np.flatnonzero(mask)
        b = np.flatnonzero(~mask)
        within = P[np.ix_(a, a)][np.triu_indices(a.size, 1)].mean()
        betw = P[np.ix_(a, b)].mean()
        return sign * (betw - within)

    true_mask = np.zeros(m, dtype=bool)
    true_mask[:n] = True
    obs_gap = gap(true_mask)
    ge = 1
    for _ in range(n_perm):
        mask = np.zeros(m, dtype=bool)
        mask[rng.permutation(m)[:n]] = True
        if gap(mask) >= obs_gap:
            ge += 1
    return ge / (n_perm + 1)


# ---------------------------------------------------------------------------
# Device / binary resolution -- no assumed CUDA, no assumed binary path.
# ---------------------------------------------------------------------------


def resolve_device(explicit: str) -> torch.device:
    """Pick the compute device for the natural-gradient (NG) backend, which
    computes in float64 for Fortran parity. Prints what was chosen so a
    reviewer never has to guess. MPS has no float64 support, so MPS (auto- or
    explicitly selected) always falls back to CPU here -- these runs are
    float64 throughout, unlike the float32 GPU benchmarks elsewhere in
    ``benchmarks/``.
    """
    if explicit == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        if explicit == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "--device cuda requested, but torch.cuda.is_available() is False."
            )
        device = torch.device(explicit)

    if device.type == "mps":
        print(
            "device: MPS detected/requested, but these parity runs use "
            "float64 (Fortran reference precision) and MPS has no float64 "
            "support; falling back to CPU."
        )
        return torch.device("cpu")
    if device.type == "cuda":
        print(f"device: CUDA ({torch.cuda.get_device_name(0)})")
    else:
        print("device: CPU" + (" (no GPU detected)" if explicit == "auto" else ""))
    return device


def resolve_binary(explicit: Path | None, version: str) -> Path:
    """Resolve the native AMICA reference binary: an explicit ``--fortran-
    binary``/``PAMICA_NATIVE_BINARY`` override, or the platform release asset
    via :mod:`pamica.native.resolver` (downloaded and SHA-256-verified on
    first use, then cached). Works on macOS/Linux/Windows, unlike the
    bundled ``sample_data/amica15mac`` (macOS x86_64 only)."""
    if explicit is not None:
        path = Path(explicit).resolve()
        if not path.exists():
            raise FileNotFoundError(f"--fortran-binary {explicit} does not exist")
        print(f"Fortran reference binary: {path} (explicit override)")
        return path
    try:
        path = resolver.resolve(version)
    except Exception as exc:
        raise RuntimeError(
            f"Could not resolve a native AMICA binary for this platform: {exc}\n"
            "Fixes: set PAMICA_NATIVE_BINARY to a locally built binary (see "
            "native/build.sh), or pass --fortran-binary explicitly."
        ) from exc
    print(f"Fortran reference binary: {path} (native/{version})")
    return path


# ---------------------------------------------------------------------------
# Single-model comparison (paper's "Single" rows): shared by both tiers --
# the bundled sample gives the Amari distance row, the external recording
# gives the headline log-likelihood/correlation rows. Newton is disabled
# (do_newton=0) to isolate the algorithm from initialization, and Fortran's
# own early-stopping is disabled (use_min_dll=0, use_grad_norm=0) so both
# sides run the full max_iter budget -- otherwise Fortran can stop well short
# of max_iter while AMICATorchNG (no early-stopping equivalent) keeps
# optimizing, drifting weakly-determined components to a different, still-
# valid optimum: an asymmetry, not real disagreement.
# ---------------------------------------------------------------------------


def run_single_model_sweep(
    data: np.ndarray,
    seeds: list[int],
    max_iter: int,
    threads: int,
    device: torch.device,
    binary: Path,
    label: str,
) -> dict[str, float]:
    nw = data.shape[0]
    fortran_Ws: dict[int, np.ndarray] = {}
    fortran_lls: dict[int, float] = {}
    ng_lls: list[float] = []
    corr_means: list[float] = []
    corr_mins: list[float] = []
    amaris: list[float] = []

    for seed in seeds:
        print(
            f"[{label}] seed {seed}: Fortran ({threads} threads, do_newton=0)...",
            flush=True,
        )
        t0 = time.perf_counter()
        try:
            eng = AMICANative(
                binary=binary,
                threads=threads,
                max_threads=threads,
                timeout=3600,
                n_models=1,
                n_mix=3,
                max_iter=max_iter,
                do_newton=0,
                use_min_dll=0,
                use_grad_norm=0,
                block_size=512,
            )
            eng.fit(data)
        except (RuntimeError, TimeoutError) as exc:
            print(
                f"[{label}] seed {seed}: Fortran FAILED, skipping this seed: {exc}",
                flush=True,
            )
            continue
        assert eng.output_ is not None
        fort_dt = time.perf_counter() - t0
        W_f = eng.output_.W[:, :, 0]
        ll_f = float(eng.output_.LL[-1])
        fortran_Ws[seed] = W_f
        fortran_lls[seed] = ll_f
        print(
            f"[{label}] seed {seed}: Fortran done in {fort_dt:.0f}s, LL={ll_f:.4f}",
            flush=True,
        )

        print(
            f"[{label}] seed {seed}: pamica (device={device}, do_newton=0)...",
            flush=True,
        )
        t0 = time.perf_counter()
        model = AMICA(n_models=1, n_mix=3, device=device, verbose=False)
        model.fit(
            data,
            max_iter=max_iter,
            lrate=0.05,
            do_mean=True,
            do_sphere=True,
            do_approx_sphere=True,
            do_newton=False,
            seed=seed,
            block_size=512,
            minlrate=1e-8,
            lratefact=0.5,
            maxdecs=3,
            newt_start=50,
            newt_ramp=10,
            newtrate=1.0,
            rho0=1.5,
            minrho=1.0,
            maxrho=2.0,
            rholrate=0.05,
            rholratefact=0.5,
            invsigmin=0.0,
            invsigmax=100.0,
            doscaling=True,
            scalestep=1,
        )
        ng_dt = time.perf_counter() - t0
        if not model.converged_:
            print(
                f"[{label}] seed {seed}: pamica fit ended degenerate "
                f"(stop_reason={model.stop_reason_!r}); skipping this seed",
                flush=True,
            )
            continue
        assert model.final_ll_ is not None  # converged_ guarantees a fitted LL
        W_ng = model.get_unmixing_matrix(0)
        ll_ng = float(model.final_ll_)
        corrs = xcorr(W_f, W_ng)
        amari = amari_distance(W_f, W_ng)
        corr_means.append(float(corrs.mean()))
        corr_mins.append(float(corrs.min()))
        amaris.append(amari)
        ng_lls.append(ll_ng)
        print(
            f"[{label}] seed {seed}: pamica done in {ng_dt:.0f}s, LL={ll_ng:.4f}, "
            f"mean_corr={corrs.mean():.4f}, min_corr={corrs.min():.4f}, amari={amari:.4f}",
            flush=True,
        )

    ff_means: list[float] = []
    ff_mins: list[float] = []
    ff_amaris: list[float] = []
    for a, b in itertools.combinations(fortran_Ws, 2):
        corrs = xcorr(fortran_Ws[a], fortran_Ws[b])
        ff_means.append(float(corrs.mean()))
        ff_mins.append(float(corrs.min()))
        ff_amaris.append(amari_distance(fortran_Ws[a], fortran_Ws[b]))

    def _stat(values: list[float], fn: Callable[[np.ndarray], float]) -> float:
        return float(fn(np.array(values))) if values else float("nan")

    return {
        "n_channels": nw,
        "n_seeds_requested": len(seeds),
        "n_seeds_ok": len(corr_means),
        "n_fortran_ok": len(fortran_Ws),
        "ng_vs_fortran_corr_mean": _stat(corr_means, np.mean),
        "ng_vs_fortran_corr_min": _stat(corr_mins, np.min),
        "ng_vs_fortran_corr_sd": _stat(corr_means, np.std),
        "ng_vs_fortran_amari_mean": _stat(amaris, np.mean),
        "ng_vs_fortran_amari_sd": _stat(amaris, np.std),
        "fortran_vs_fortran_corr_mean": _stat(ff_means, np.mean),
        "fortran_vs_fortran_corr_min": _stat(ff_mins, np.min),
        "fortran_vs_fortran_amari_mean": _stat(ff_amaris, np.mean),
        "fortran_ll_mean": _stat(list(fortran_lls.values()), np.mean),
        "fortran_ll_sd": _stat(list(fortran_lls.values()), np.std),
        "ng_ll_mean": _stat(ng_lls, np.mean),
        "ng_ll_sd": _stat(ng_lls, np.std),
        "ll_gap_of_means": abs(
            _stat(ng_lls, np.mean) - _stat(list(fortran_lls.values()), np.mean)
        ),
    }


# ---------------------------------------------------------------------------
# Multi-model ensemble (paper's "Multi" rows, bundled sample only): N Fortran
# + N pamica fits (n_models=2), compared by within-Fortran / within-pamica /
# between distributions of Hungarian correlation and Amari distance, plus a
# run-level permutation test and a log-likelihood KS test. Newton stays on
# (the multi-model default), matching .context/issue-27/multimodel_ensemble.py.
# ---------------------------------------------------------------------------


def run_multimodel_ensemble(
    data: np.ndarray,
    n_runs: int,
    max_iter: int,
    threads: int,
    device: torch.device,
    binary: Path,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    nw = data.shape[0]
    Fs: list[np.ndarray] = []
    F_ll: list[float] = []
    Gs: list[np.ndarray] = []
    G_ll: list[float] = []

    for i in range(n_runs):
        print(f"[multimodel] Fortran {i + 1}/{n_runs}...", flush=True)
        try:
            eng = AMICANative(
                binary=binary,
                threads=threads,
                max_threads=threads,
                timeout=1800,
                n_models=2,
                n_mix=3,
                max_iter=max_iter,
            )
            eng.fit(data)
        except RuntimeError as exc:
            print(f"[multimodel] Fortran run {i}: FAILED, skipping: {exc}", flush=True)
            continue
        assert eng.output_ is not None
        Fs.append(np.vstack([eng.output_.W[:, :, 0], eng.output_.W[:, :, 1]]))
        F_ll.append(float(eng.output_.LL[-1]))

    for i in range(n_runs):
        print(f"[multimodel] pamica {i + 1}/{n_runs}...", flush=True)
        model = AMICA(n_models=2, n_mix=3, device=device, verbose=False)
        model.fit(
            data,
            max_iter=max_iter,
            do_newton=True,
            seed=MULTIMODEL_SEED_START + i,
            block_size=512,
            minlrate=1e-8,
            lratefact=0.5,
            maxdecs=3,
            newt_start=50,
            newt_ramp=10,
            newtrate=1.0,
            rho0=1.5,
            minrho=1.0,
            maxrho=2.0,
            rholrate=0.05,
            rholratefact=0.5,
            invsigmin=1e-8,
            invsigmax=100.0,
            doscaling=True,
            scalestep=1,
        )
        if not model.converged_:
            print(
                f"[multimodel] pamica run {i}: ended degenerate "
                f"(stop_reason={model.stop_reason_!r}), skipping",
                flush=True,
            )
            continue
        assert model.final_ll_ is not None  # converged_ guarantees a fitted LL
        Gs.append(
            np.vstack([model.get_unmixing_matrix(0), model.get_unmixing_matrix(1)])
        )
        G_ll.append(float(model.final_ll_))

    Fs_a, Gs_a = np.array(Fs), np.array(Gs)
    F_ll_a, G_ll_a = np.array(F_ll), np.array(G_ll)

    def amari_pair(a: np.ndarray, b: np.ndarray) -> float:
        return model_amari(a, b, nw)

    corr_within_f = pairwise(Fs_a, Fs_a, True, corr_metric)
    corr_within_g = pairwise(Gs_a, Gs_a, True, corr_metric)
    corr_between = pairwise(Gs_a, Fs_a, False, corr_metric)
    corr_p = perm_test_not_worse(Fs_a, Gs_a, corr_metric, higher_is_worse=False)

    amari_within_f = pairwise(Fs_a, Fs_a, True, amari_pair)
    amari_within_g = pairwise(Gs_a, Gs_a, True, amari_pair)
    amari_between = pairwise(Gs_a, Fs_a, False, amari_pair)
    amari_p = perm_test_not_worse(Fs_a, Gs_a, amari_pair, higher_is_worse=True)

    ks_p = (
        float(stats.ks_2samp(G_ll_a, F_ll_a).pvalue)
        if len(F_ll_a) and len(G_ll_a)
        else float("nan")
    )

    summary = {
        "n_runs_requested": n_runs,
        "n_fortran_ok": len(Fs),
        "n_ng_ok": len(Gs),
        "corr_within_fortran_mean": float(corr_within_f.mean()),
        "corr_within_fortran_sd": float(corr_within_f.std()),
        "corr_within_ng_mean": float(corr_within_g.mean()),
        "corr_within_ng_sd": float(corr_within_g.std()),
        "corr_between_mean": float(corr_between.mean()),
        "corr_between_sd": float(corr_between.std()),
        "corr_diff_between_minus_within_fortran": float(
            corr_between.mean() - corr_within_f.mean()
        ),
        "corr_perm_p_not_worse": corr_p,
        "amari_within_fortran_mean": float(amari_within_f.mean()),
        "amari_within_fortran_sd": float(amari_within_f.std()),
        "amari_within_ng_mean": float(amari_within_g.mean()),
        "amari_within_ng_sd": float(amari_within_g.std()),
        "amari_between_mean": float(amari_between.mean()),
        "amari_between_sd": float(amari_between.std()),
        "amari_diff_between_minus_within_fortran": float(
            amari_between.mean() - amari_within_f.mean()
        ),
        "amari_perm_p_not_worse": amari_p,
        "fortran_ll_mean": float(F_ll_a.mean()) if len(F_ll_a) else float("nan"),
        "fortran_ll_sd": float(F_ll_a.std()) if len(F_ll_a) else float("nan"),
        "ng_ll_mean": float(G_ll_a.mean()) if len(G_ll_a) else float("nan"),
        "ng_ll_sd": float(G_ll_a.std()) if len(G_ll_a) else float("nan"),
        "ll_ks_p": ks_p,
    }
    raw = {"Fs": Fs_a, "Gs": Gs_a, "F_ll": F_ll_a, "G_ll": G_ll_a}
    return summary, raw


# ---------------------------------------------------------------------------
# Score functions / sufficient statistics (paper's bit-exactness row,
# bundled tier). No fitting: the log-density/score check is a pure formula
# identity (matches pamica/tests/torch_tests/test_ng_pdf_families.py, no data
# needed); the sufficient-statistics check compares one real-data block
# (matches test_ng_backend.py::test_sufficient_stats_match_numpy_reference).
# ---------------------------------------------------------------------------

_LOG4 = math.log(4.0)
_LSQ2PI = math.log(2.506628274)
_LNSUB = math.log(4.132731354)
_LNSUP = math.log(1.858073988)


def _fortran_z0(y: np.ndarray, code: int) -> np.ndarray:
    """Literal amica15.f90 log-density (alpha=beta=1, mu=0 so y=b)."""
    if code == 2:  # Gaussian, :1314
        return -0.5 * y * y - _LSQ2PI
    if code == 3:  # logistic, :1327
        return -2.0 * np.log(np.cosh(0.5 * y)) - _LOG4
    if code == 4:  # sub-Gaussian cosh+, :1340
        return -0.5 * y * y + np.log(np.cosh(y)) - _LNSUB
    if code == 1:  # super-Gaussian cosh-, :1352
        return -0.5 * y * y - np.log(np.cosh(y)) - _LNSUP
    raise ValueError(code)


def _fortran_fp(y: np.ndarray, code: int) -> np.ndarray:
    """Literal amica15.f90 score (:1465-1472)."""
    return {2: y, 3: np.tanh(y / 2.0), 4: y - np.tanh(y), 1: y + np.tanh(y)}[code]


def check_score_functions() -> dict[str, float]:
    y = torch.linspace(-8.0, 8.0, 65, dtype=torch.float64)
    y = y[y.abs() > 1e-3]
    rho = torch.full_like(y, 1.5)

    max_log_pdf_diff = 0.0
    max_score_diff = 0.0
    for code in (2, 3, 4, 1):
        pdt = torch.full_like(y, code, dtype=torch.long)
        log_pdf, _ = _log_pdf_and_deriv(y, rho, pdt)
        ref_log_pdf = _fortran_z0(y.numpy(), code)
        max_log_pdf_diff = max(
            max_log_pdf_diff, float(np.max(np.abs(log_pdf.numpy() - ref_log_pdf)))
        )

        fp = _score(y, rho, pdt)
        ref_fp = _fortran_fp(y.numpy(), code)
        max_score_diff = max(max_score_diff, float(np.max(np.abs(fp.numpy() - ref_fp))))

    return {"max_log_pdf_diff": max_log_pdf_diff, "max_score_diff": max_score_diff}


def check_sufficient_statistics(data: np.ndarray) -> dict[str, float]:
    nw = data.shape[0]
    ng = AMICATorchNG(
        n_channels=nw, n_models=1, n_mix=3, seed=42, device="cpu",
        dtype=torch.float64, block_size=256,
    )  # fmt: skip
    x_t = ng._preprocess(data)  # noqa: SLF001 -- reused test-only accessor
    ng._initialize_parameters()  # noqa: SLF001
    block = x_t[:, :256].contiguous()
    ng_upd = ng._get_block_updates(block)  # noqa: SLF001

    assert (
        ng.comp_list is not None
        and ng.A is not None
        and ng.W is not None
        and ng.c is not None
        and ng.mu is not None
        and ng.alpha is not None
        and ng.beta is not None
        and ng.rho is not None
        and ng.gm is not None
    )
    npm = AMICA_NumPy(num_models=1, num_mix=3, do_newton=False)
    npm.data_dim = nw
    npm.num_comps = nw
    npm.num_models = 1
    npm.num_mix = 3
    npm.block_size = 256
    npm.comp_list = ng.comp_list.cpu().numpy()
    npm.A = ng.A.cpu().numpy().copy()
    npm.W = ng.W.cpu().numpy().copy()
    npm.c = ng.c.cpu().numpy().copy()
    npm.mu = ng.mu.cpu().numpy().copy()
    npm.alpha = ng.alpha.cpu().numpy().copy()
    npm.beta = ng.beta.cpu().numpy().copy()
    npm.rho = ng.rho.cpu().numpy().copy()
    npm.gm = ng.gm.cpu().numpy().copy()
    np_upd = npm._get_block_updates(block.cpu().numpy())  # noqa: SLF001

    keys = [
        "dgm",
        "dalpha_n",
        "dmu_n",
        "dmu_d",
        "dbeta_n",
        "dbeta_d",
        "drho_n",
        "dWtmp",
        "dc_numer",
    ]
    max_diff = 0.0
    for key in keys:
        a = np.asarray(ng_upd[key].cpu().numpy(), dtype=np.float64)
        b = np.asarray(np_upd[key], dtype=np.float64).reshape(a.shape)
        max_diff = max(max_diff, float(np.max(np.abs(a - b))))
    return {"max_sufficient_stats_diff": max_diff}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

_RULE = "-" * 78


def _row(title: str, paper: str, *measured: str) -> None:
    print(_RULE)
    print(f"Row: {title}")
    print(f"  paper:    {paper}")
    for line in measured:
        print(f"  measured: {line}")


def print_bundled_report(
    nw: int, field: int, single: dict, multi: dict, score: dict
) -> None:
    """Print the bundled-tier rows, skipping any section a ``--skip-*`` flag
    left empty rather than crashing on a missing key."""
    k = field / nw**2
    print("=" * 78)
    print(
        f"BUNDLED tier: pamica/sample_data/eeglab_data.fdt ({nw}ch x {field}fr, k={k:.1f})"
    )
    print("=" * 78)
    if single:
        _row(
            "Single | Amari distance (bundled)",
            "0.006",
            f"pamica vs Fortran: mean={single['ng_vs_fortran_amari_mean']:.4f} "
            f"(n={single['n_seeds_ok']}/{single['n_seeds_requested']} seeds ok)",
            f"Fortran vs Fortran: mean={single['fortran_vs_fortran_amari_mean']:.4f}",
        )
    if score:
        _row(
            "Single | Score functions, sufficient statistics",
            "exact, ~1e-15",
            f"log-pdf max|diff|={score['max_log_pdf_diff']:.2e}, "
            f"score max|diff|={score['max_score_diff']:.2e} (formula identity, no data)",
            f"sufficient-statistics max|diff|={score['max_sufficient_stats_diff']:.2e} "
            "(one real-data block vs the NumPy reference)",
        )
    if multi:
        _row(
            "Multi | Correlation, one run: cross; within-Fortran",
            "0.65; 0.64 (sd 0.05)",
            f"between={multi['corr_between_mean']:.3f} (sd {multi['corr_between_sd']:.3f}); "
            f"within-Fortran={multi['corr_within_fortran_mean']:.3f} "
            f"(sd {multi['corr_within_fortran_sd']:.3f})",
        )
        _row(
            "Multi | Amari, one run: cross; within-Fortran",
            "0.163; 0.174 (sd 0.02)",
            f"between={multi['amari_between_mean']:.3f} (sd {multi['amari_between_sd']:.3f}); "
            f"within-Fortran={multi['amari_within_fortran_mean']:.3f} "
            f"(sd {multi['amari_within_fortran_sd']:.3f})",
        )
        _row(
            "Multi | Ensemble agreement, cross - within-Fortran",
            "correlation +0.011 (p=0.96); Amari -0.011 (p>0.999)",
            f"correlation diff={multi['corr_diff_between_minus_within_fortran']:+.3f} "
            f"(perm p={multi['corr_perm_p_not_worse']:.3f})",
            f"Amari diff={multi['amari_diff_between_minus_within_fortran']:+.3f} "
            f"(perm p={multi['amari_perm_p_not_worse']:.3f})",
        )
        _row(
            "Multi | Ensemble log-likelihood: Fortran; pamica",
            "-3.3539; -3.3629 (KS p=6e-5)",
            f"Fortran={multi['fortran_ll_mean']:.4f} (sd {multi['fortran_ll_sd']:.3f}); "
            f"pamica={multi['ng_ll_mean']:.4f} (sd {multi['ng_ll_sd']:.3f}); "
            f"KS p={multi['ll_ks_p']:.2e} "
            f"(n={multi['n_fortran_ok']}/{multi['n_runs_requested']} Fortran, "
            f"{multi['n_ng_ok']}/{multi['n_runs_requested']} pamica runs ok)",
        )
    print(_RULE)


def print_external_report(nw: int, field: int, single: dict) -> None:
    if not single:
        return
    k = field / nw**2
    print("=" * 78)
    print(f"EXTERNAL tier: {nw}ch x {field}fr, k={k:.1f}")
    print("=" * 78)
    _row(
        "Single | Log-likelihood gap (on002718)",
        "within ~0.0005 of -3.6993",
        f"Fortran mean={single['fortran_ll_mean']:.4f} (sd {single['fortran_ll_sd']:.4f}); "
        f"pamica mean={single['ng_ll_mean']:.4f} (sd {single['ng_ll_sd']:.4f}); "
        f"gap={single['ll_gap_of_means']:.4f} "
        f"(n={single['n_seeds_ok']}/{single['n_seeds_requested']} seeds ok)",
    )
    _row(
        "Single | Component correlation (on002718)",
        "0.998",
        f"pamica vs Fortran: mean={single['ng_vs_fortran_corr_mean']:.4f} "
        f"(min={single['ng_vs_fortran_corr_min']:.4f}, sd {single['ng_vs_fortran_corr_sd']:.4f})",
        f"Fortran vs Fortran self-consistency: mean={single['fortran_vs_fortran_corr_mean']:.4f} "
        f"(min={single['fortran_vs_fortran_corr_min']:.4f})",
    )
    print(_RULE)


def _json_default(obj: object) -> object:
    # The report dict is built entirely from explicit float()/int() casts, so
    # this is a defensive fallback (fail loudly, not a silent misencoding) for
    # a stray numpy scalar rather than an expected code path; raw arrays
    # (Fs/Gs/LL histories) are saved separately as .npz, never embedded here.
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    raise TypeError(f"not JSON-serializable: {type(obj)}")


# ---------------------------------------------------------------------------
# Tier orchestration
# ---------------------------------------------------------------------------


def load_bundled_data() -> tuple[np.ndarray, dict]:
    with open(BUNDLED_PARAMS_JSON) as f:
        params = json.load(f)
    data_dim = params["data_dim"]
    field_dim = params["field_dim"][0]
    data = load_eeglab_data(
        str(BUNDLED_FDT), data_dim=data_dim, field_dim=field_dim, dtype=np.float32
    ).astype(np.float64)
    return data, params


def run_bundled_tier(
    args: argparse.Namespace, device: torch.device, binary: Path
) -> dict:
    print("\n### bundled tier ###")
    t_start = time.perf_counter()
    data, _params = load_bundled_data()
    nw, field = data.shape

    seed_start = args.seed_start if args.seed_start is not None else BUNDLED_SEED_START
    seeds = list(range(seed_start, seed_start + args.n_seeds))

    single: dict = {}
    if not args.skip_single_model:
        single = run_single_model_sweep(
            data, seeds, args.max_iter, args.threads, device, binary, label="bundled"
        )

    multi: dict = {}
    ensemble_raw: dict = {}
    if not args.skip_multimodel:
        multi, ensemble_raw = run_multimodel_ensemble(
            data,
            args.multimodel_runs,
            args.multimodel_max_iter,
            args.threads,
            device,
            binary,
        )
        np.savez(
            args.out_dir / "bundled_multimodel_ensemble.npz",
            Fs=ensemble_raw["Fs"],
            Gs=ensemble_raw["Gs"],
            F_ll=ensemble_raw["F_ll"],
            G_ll=ensemble_raw["G_ll"],
        )

    score: dict = {}
    if not args.skip_score_check:
        score = check_score_functions()
        score.update(check_sufficient_statistics(data))

    wall_clock = time.perf_counter() - t_start
    if single or multi or score:
        print_bundled_report(nw, field, single, multi, score)
    print(f"bundled tier wall-clock: {wall_clock:.0f}s")

    return {
        "tier": "bundled",
        "n_channels": nw,
        "n_frames": field,
        "k": field / nw**2,
        "wall_clock_s": wall_clock,
        "single_model": single,
        "multimodel": multi,
        "score_functions": score,
    }


def run_external_tier(
    args: argparse.Namespace, device: torch.device, binary: Path
) -> dict:
    print("\n### external tier ###")
    t_start = time.perf_counter()
    data = np.load(args.data).astype(np.float64)
    nw, field = data.shape
    print(f"data: {args.data} ({nw}ch x {field}fr, k={field / nw**2:.1f})")

    seed_start = args.seed_start if args.seed_start is not None else EXTERNAL_SEED_START
    seeds = list(range(seed_start, seed_start + args.n_seeds))

    single = run_single_model_sweep(
        data, seeds, args.max_iter, args.threads, device, binary, label="external"
    )
    wall_clock = time.perf_counter() - t_start
    print_external_report(nw, field, single)
    print(f"external tier wall-clock: {wall_clock:.0f}s")

    return {
        "tier": "external",
        "n_channels": nw,
        "n_frames": field,
        "k": field / nw**2,
        "wall_clock_s": wall_clock,
        "single_model": single,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Reproduce paper.md's Table 1 parity numbers against the Fortran "
            "reference (issue #144). See benchmarks/README_dimsweep.md for "
            "wall-clock and download-size estimates before running the "
            "external tier."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--tier",
        choices=["bundled", "external", "both"],
        default="bundled",
        help="bundled needs no download; external needs a manual OpenNeuro "
        "download (see benchmarks/README_dimsweep.md) and can take hours",
    )
    p.add_argument(
        "--data",
        type=Path,
        default=None,
        help="external-tier data: a (n_channels, n_frames) float32/float64 "
        ".npy of OpenNeuro ds002718 sub-002 (see benchmarks/README_dimsweep.md). "
        "Required for --tier external/both.",
    )
    p.add_argument(
        "--n-seeds", type=int, default=DEFAULT_N_SEEDS, help="single-model sweep size"
    )
    p.add_argument(
        "--seed-start",
        type=int,
        default=None,
        help="first seed for the single-model sweep (default: tier-specific)",
    )
    p.add_argument(
        "--max-iter",
        type=int,
        default=DEFAULT_SINGLE_MAX_ITER,
        help="single-model iteration budget (both tiers)",
    )
    p.add_argument(
        "--multimodel-runs",
        type=int,
        default=DEFAULT_ENSEMBLE_N,
        help="bundled-tier multi-model ensemble size",
    )
    p.add_argument(
        "--multimodel-max-iter",
        type=int,
        default=DEFAULT_ENSEMBLE_MAX_ITER,
        help="bundled-tier multi-model iteration budget",
    )
    p.add_argument(
        "--threads",
        type=int,
        default=os.cpu_count() or 4,
        help="OMP_NUM_THREADS for Fortran",
    )
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument(
        "--fortran-binary",
        type=Path,
        default=None,
        help="explicit reference-binary path, overriding PAMICA_NATIVE_BINARY "
        "and the platform-release resolver",
    )
    p.add_argument(
        "--native-version",
        default="latest",
        help="release tag to resolve the binary from",
    )
    p.add_argument(
        "--skip-single-model",
        action="store_true",
        help="bundled tier: skip the Amari row",
    )
    p.add_argument(
        "--skip-multimodel",
        action="store_true",
        help="bundled tier: skip the multi-model rows",
    )
    p.add_argument(
        "--skip-score-check",
        action="store_true",
        help="bundled tier: skip the bit-exactness row",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"reproduce_table1: tier={args.tier}, threads={args.threads}, out_dir={args.out_dir}"
    )
    device = resolve_device(args.device)
    binary = resolve_binary(args.fortran_binary, args.native_version)

    if args.tier in ("external", "both") and args.data is None:
        print(
            "ERROR: --tier external/both requires --data (a downloaded "
            "ds002718 .npy); see benchmarks/README_dimsweep.md for the "
            "download recipe and cost.",
            file=sys.stderr,
        )
        return 2

    report: dict[str, dict] = {}
    if args.tier in ("bundled", "both"):
        report["bundled"] = run_bundled_tier(args, device, binary)
    if args.tier in ("external", "both"):
        report["external"] = run_external_tier(args, device, binary)

    report_path = args.out_dir / "report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=_json_default)
    print(f"\nwrote {report_path}")

    n_short = 0
    for tier_report in report.values():
        single = tier_report.get("single_model") or {}
        if single and single.get("n_seeds_ok", 0) < single.get("n_seeds_requested", 0):
            n_short += 1
        multi = tier_report.get("multimodel") or {}
        if multi and (
            multi.get("n_fortran_ok", 0) < multi.get("n_runs_requested", 0)
            or multi.get("n_ng_ok", 0) < multi.get("n_runs_requested", 0)
        ):
            n_short += 1
    if n_short:
        print(
            f"\nWARNING: {n_short} sweep(s) had fewer converged runs than "
            "requested (see the FAILED/degenerate lines above); the reported "
            "means are over fewer seeds/runs than --n-seeds/--multimodel-runs.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
