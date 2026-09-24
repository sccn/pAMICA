"""The ±0.05 equivalence margin of the #27 multi-model study, on saved ensembles
(issue #351 item 3). No refitting.

The #27 record states its acceptance criterion as "between not worse than
within-Fortran; TOST within a margin" of ±0.05 on the mean Hungarian-matched
correlation. Its first TOST was computed on the 190/400 pairwise values as if
they were independent, which the record later retired as pseudoreplicated (each
run appears in ~39 pairs) in favor of the run-level permutation test. For each
ensemble given, this script reports, for the correlation and the Amari distance:

* ``diff``: mean(between) - mean(within-Fortran), and whether it lies inside
  ±0.05 (the documented descriptive check, stated for the correlation only);
* ``pairwise_tost_p``: the retired pairwise TOST with the same margin, for
  continuity with the old record (its p-value is not valid, as the record says);
* ``run_bootstrap_90ci``: a run-level bootstrap 90% interval of ``diff``
  (runs resampled with replacement within each implementation, 5000 draws),
  which respects the shared-run dependence; an interval inside ±0.05 is the
  run-level analogue of the TOST at the 5% level.

    uv run python .context/issue-351/equivalence_check.py OUT.json NAME=PATH.npz [NAME=PATH.npz ...]
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DELTA = 0.05


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def analyze(Fs: np.ndarray, Gs: np.ndarray, metric) -> dict:
    allW = np.concatenate([Fs, Gs])
    m, n = len(allW), len(Fs)
    P = np.zeros((m, m))
    for i in range(m):
        for j in range(i + 1, m):
            P[i, j] = P[j, i] = metric(allW[i], allW[j])
    F = np.arange(n)
    G = np.arange(n, m)
    wF = P[np.ix_(F, F)][np.triu_indices(n, 1)]
    bt = P[np.ix_(G, F)].ravel()
    diff = float(bt.mean() - wF.mean())
    se = np.sqrt(bt.var(ddof=1) / bt.size + wF.var(ddof=1) / wF.size)
    p_tost = float(
        max(stats.norm.sf((diff + DELTA) / se), stats.norm.cdf((diff - DELTA) / se))
    )
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(5000):
        f = rng.choice(F, n, replace=True)
        g = rng.choice(G, len(G), replace=True)
        # within-Fortran over distinct draws only (a run is not paired with itself)
        sub = P[np.ix_(f, f)]
        mask = f[:, None] != f[None, :]
        w = sub[np.triu(mask, 1)].mean()
        b = P[np.ix_(g, f)].mean()
        boot.append(b - w)
    lo, hi = np.percentile(boot, [5, 95])
    return {
        "diff_between_minus_within_fortran": diff,
        "inside_pm_0.05": bool(abs(diff) < DELTA),
        "pairwise_tost_p_retired": p_tost,
        "run_bootstrap_90ci": [float(lo), float(hi)],
        "run_bootstrap_90ci_inside_pm_0.05": bool(lo > -DELTA and hi < DELTA),
    }


def main() -> None:
    out = Path(sys.argv[1])
    ama27 = _load(REPO / ".context/issue-27/amari_distance.py", "ama27_eq")
    rep = {}
    for arg in sys.argv[2:]:
        name, path = arg.split("=", 1)
        d = np.load(path)
        rep[name] = {
            "corr": analyze(d["Fs"], d["Gs"], ama27.xcorr),
            "amari": analyze(d["Fs"], d["Gs"], ama27.model_amari),
        }
        print(name, json.dumps(rep[name], indent=1), flush=True)
    out.write_text(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
