"""Seeded multi-model log-likelihood ensemble against the pinned binary
(issue #351 items 3 and 5): ADR 0003's ``keep_best`` figures and the paper's
"convergence speed" explanation, re-measured under epic #324.

Protocol of ``.context/issue-51/ensemble_ll.py`` (ADR 0003): the bundled
sample EEG, ``n_models=2``, three mixtures, the reference's ``input.param``
schedule (``lrate`` 0.05, Newton from iteration 50, ``newtrate`` 1.0, block
size 512), N = 20 fits per implementation, pamica seeds 0-19 on
``AMICATorchNG`` with the #51 keyword set. Changes, each recorded in
``findings.md``:

* the reference is the pinned v0.3.3 native binary through ``AMICANative``,
  seeded 0-19 and single-threaded, so every run is reproducible (#51 used the
  clock-seeded bundled ``amica15mac``);
* each fit runs the longest budget (300 iterations) once, and the 100- and
  200-iteration figures are read from the prefix of its trajectory: a fit's
  iterations do not depend on ``max_iter``, and ``verify`` checks that on the
  first seeds (separate 100-iteration fits with ``keep_best`` on and off give
  exactly the prefix's values, on both sides).

At a budget of K iterations, "return-last" is ``ll_history[K-1]`` (the
likelihood ``keep_best=False`` reports) and ``keep_best`` is the best of the
first K (restored when it beats the last by more than ``_KEEP_BEST_TOL``).

    uv run python .context/issue-351/keep_best_ensemble.py run OUT.npz [N]
    uv run python .context/issue-351/keep_best_ensemble.py verify OUT.json [n]
    uv run python .context/issue-351/keep_best_ensemble.py report OUT.npz
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FDT = REPO / "pamica/sample_data/eeglab_data.fdt"
NW, FIELD = 32, 30504
BUDGETS = (100, 200, 300)
DELTA_LL = 0.01  # the #51 TOST margin on the mean LL


def load_data() -> np.ndarray:
    from pamica.torch_impl.utils import load_eeglab_data

    return load_eeglab_data(str(FDT), data_dim=NW, field_dim=FIELD).astype(np.float64)


def fortran_kwargs(max_iter: int, seed: int) -> dict:
    """``AMICANative`` keywords of the reference runs, every protocol setting
    explicit (``_reference_settings``), seeded and single-threaded."""
    from _reference_settings import multimodel_reference_kwargs

    return dict(
        threads=1, max_threads=1, timeout=3600, seed=seed,
        **multimodel_reference_kwargs(max_iter),
    )  # fmt: skip


def run_fortran(data: np.ndarray, seed: int, max_iter: int) -> np.ndarray:
    from pamica import AMICANative
    from pamica.native import resolver

    eng = AMICANative(
        **fortran_kwargs(max_iter, seed), binary=resolver.resolve("v0.3.3")
    )
    eng.fit(data)
    assert eng.output_ is not None
    ll = np.asarray(eng.output_.LL, dtype=np.float64)
    return ll[ll != 0]


def run_ng(data: np.ndarray, seed: int, max_iter: int, keep_best: bool = True):
    from pamica.torch_impl import AMICATorchNG

    ng = AMICATorchNG(
        n_channels=NW, n_models=2, n_mix=3, block_size=512, lrate=0.05, minlrate=1e-8,
        lratefact=0.5, maxdecs=3, do_newton=True, newt_start=50, newt_ramp=10,
        newtrate=1.0, rho0=1.5, minrho=1.0, maxrho=2.0, rholrate=0.05,
        rholratefact=0.5, invsigmin=1e-8, invsigmax=100.0, doscaling=True,
        scalestep=1, seed=seed, device="cpu", keep_best=keep_best,
    )  # fmt: skip
    ng.fit(data, max_iter=max_iter, verbose=False)
    return ng


def at_budget(hist: np.ndarray, k: int, tol: float) -> tuple[float, float, bool]:
    """(return-last, keep_best, restored) at a budget of ``k`` iterations."""
    h = hist[:k]
    last, best = float(h[-1]), float(h.max())
    restored = best - last > tol
    return last, (best if restored else last), restored


def cmd_run(out: Path, n: int) -> None:
    from pamica.torch_impl.core import _KEEP_BEST_TOL

    data = load_data()
    kmax = max(BUDGETS)
    F_hist = np.full((n, kmax), np.nan)
    G_hist = np.full((n, kmax), np.nan)
    G_final = np.full(n, np.nan)
    G_stop, G_fallbacks = [], []
    t0 = time.time()
    for k in range(n):
        f = run_fortran(data, k, kmax)
        F_hist[k, : f.size] = f
        ng = run_ng(data, k, kmax)
        h = np.asarray(ng.ll_history, dtype=np.float64)
        G_hist[k, : h.size] = h
        G_final[k] = ng.final_ll_
        G_stop.append(str(ng.stop_reason))
        G_fallbacks.append(int(ng.n_newton_fallbacks))
        print(
            f"seed {k}: Fortran LL[{f.size}]={f[-1]:.4f} NG iters={h.size} "
            f"last={h[-1]:.4f} final={ng.final_ll_:.4f} stop={ng.stop_reason} "
            f"fallbacks={ng.n_newton_fallbacks} ({time.time() - t0:.0f}s)",
            flush=True,
        )
    np.savez(
        out, F_hist=F_hist, G_hist=G_hist, G_final=G_final,
        G_stop=np.array(G_stop), G_fallbacks=np.array(G_fallbacks),
        keep_best_tol=_KEEP_BEST_TOL, budgets=np.array(BUDGETS),
    )  # fmt: skip
    cmd_report(out)


def cmd_verify(out: Path, n: int) -> None:
    """Separate 100-iteration fits equal the prefix of a 300-iteration fit."""
    from pamica.torch_impl.core import _KEEP_BEST_TOL

    data = load_data()
    rows = []
    for k in range(n):
        f300 = run_fortran(data, k, 300)
        f100 = run_fortran(data, k, 100)
        g300 = np.asarray(run_ng(data, k, 300).ll_history)
        g100_on = run_ng(data, k, 100, keep_best=True)
        g100_off = run_ng(data, k, 100, keep_best=False)
        last, kb, restored = at_budget(g300, 100, _KEEP_BEST_TOL)
        rows.append(
            {
                "seed": k,
                "fortran_ll100_separate_minus_prefix": float(f100[-1] - f300[99]),
                "ng_history_prefix_max_abs_diff": float(
                    np.abs(np.asarray(g100_on.ll_history) - g300[:100]).max()
                ),
                "ng_keep_best_off_final_minus_prefix_last": float(
                    g100_off.final_ll_ - last
                ),
                "ng_keep_best_on_final_minus_prefix_best": float(
                    g100_on.final_ll_ - kb
                ),
                "restored_at_100": bool(restored),
            }
        )
        print(rows[-1], flush=True)
    out.write_text(json.dumps(rows, indent=2))


def _tost(F: np.ndarray, G: np.ndarray) -> float:
    diff = G.mean() - F.mean()
    se = np.sqrt(G.var(ddof=1) / G.size + F.var(ddof=1) / F.size)
    return float(
        max(
            stats.norm.sf((diff + DELTA_LL) / se),
            stats.norm.cdf((diff - DELTA_LL) / se),
        )
    )


def cmd_report(out: Path) -> dict:
    d = np.load(out)
    F_hist, G_hist, tol = d["F_hist"], d["G_hist"], float(d["keep_best_tol"])
    n = F_hist.shape[0]
    rep: dict = {"n": n, "keep_best_tol": tol, "budgets": {}}
    for K in BUDGETS:
        F = np.array([F_hist[i, :K][~np.isnan(F_hist[i, :K])][-1] for i in range(n)])
        last, best, restored = [], [], []
        for i in range(n):
            h = G_hist[i][~np.isnan(G_hist[i])]
            a, b, r = at_budget(h, min(K, h.size), tol)
            last.append(a)
            best.append(b)
            restored.append(r)
        G_last, G_best = np.array(last), np.array(best)
        entry = {
            "fortran_mean": float(F.mean()),
            "fortran_sd": float(F.std()),
            "return_last_mean": float(G_last.mean()),
            "return_last_sd": float(G_last.std()),
            "keep_best_mean": float(G_best.mean()),
            "keep_best_sd": float(G_best.std()),
            "sd_ratio_return_last": float(G_last.std() / F.std()),
            "sd_ratio_keep_best": float(G_best.std() / F.std()),
            "gap_return_last": float(G_last.mean() - F.mean()),
            "gap_keep_best": float(G_best.mean() - F.mean()),
            "ks_p_keep_best": float(stats.ks_2samp(G_best, F).pvalue),
            "ks_p_return_last": float(stats.ks_2samp(G_last, F).pvalue),
            "tost_p_keep_best": _tost(F, G_best),
            "n_restored": int(np.sum(restored)),
            "restored_seeds": [int(i) for i in np.flatnonzero(restored)],
            "max_restore_gain": float((G_best - G_last).max()),
        }
        rep["budgets"][str(K)] = entry
    F100 = np.array([F_hist[i, 99] for i in range(n)])
    rep["keep_best_gap_to_fortran_at_100"] = {
        str(K): rep["budgets"][str(K)]["keep_best_mean"] - float(F100.mean())
        for K in BUDGETS
    }
    rep["ng_stop_reasons"] = sorted(set(d["G_stop"].tolist()))
    rep["ng_newton_fallbacks_total"] = int(d["G_fallbacks"].sum())
    out.with_suffix(".json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    return rep


def main() -> None:
    cmd, out = sys.argv[1], Path(sys.argv[2])
    if cmd == "run":
        cmd_run(out, int(sys.argv[3]) if len(sys.argv) > 3 else 20)
    elif cmd == "verify":
        cmd_verify(out, int(sys.argv[3]) if len(sys.argv) > 3 else 2)
    elif cmd == "report":
        cmd_report(out)
    else:
        raise SystemExit(f"unknown command {cmd!r}")


if __name__ == "__main__":
    main()
