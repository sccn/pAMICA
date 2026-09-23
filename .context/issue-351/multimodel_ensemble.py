"""Multi-model distributional-equivalence study (issue #27), re-run under epic
#324 (issue #351 item 3). Same shape as ``.context/issue-27/multimodel_ensemble.py``
and ``amari_distance.py``, whose analysis functions and figure it reuses:
N = 20 reference and N = 20 pamica fits of the bundled sample EEG
(``n_models=2``, three mixtures, 100 iterations, Newton from iteration 50),
compared by the within-Fortran, within-pamica and between distributions of the
Hungarian-matched correlation and the Amari distance, with the run-level
permutation test and a Kolmogorov-Smirnov test on the final log-likelihoods.

The fits are ``benchmarks/reproduce_table1.py``'s bundled-tier ensemble (so the
paper's Table 1 rows and the figure come from the same 40 fits): the pinned
v0.3.3 native binary through ``AMICANative`` (clock-seeded, as the #27
reference runs were) and ``AMICA`` seeds 1-20 with the #27 keyword set.

    # analyze an existing ensemble (the bundled tier's .npz)
    uv run python .context/issue-351/multimodel_ensemble.py --from-npz PATH
    # or fit a new one first (writes ensemble.npz next to this script)
    uv run python .context/issue-351/multimodel_ensemble.py [N] [--threads T]

Writes ``multimodel_summary.json`` and ``per_run_detail.csv`` next to this
script and the figure to ``docs/assets/figures/multimodel-ensemble.png``.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FIGURE = REPO / "docs/assets/figures/multimodel-ensemble.png"
NW = 32


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def fit_ensemble(n: int, threads: int) -> dict[str, np.ndarray]:
    table1 = _load(REPO / "benchmarks/reproduce_table1.py", "reproduce_table1_351")
    device = table1.resolve_device("cpu")
    binary = table1.resolve_binary(None, "v0.3.3")
    data, _ = table1.load_bundled_data()
    _, raw = table1.run_multimodel_ensemble(data, n, 100, threads, device, binary)
    np.savez(HERE / "ensemble.npz", **raw)
    return raw


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("n", nargs="?", type=int, default=20)
    p.add_argument("--from-npz", type=Path)
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    ens27 = _load(REPO / ".context/issue-27/multimodel_ensemble.py", "ens27")
    ama27 = _load(REPO / ".context/issue-27/amari_distance.py", "ama27")

    d = dict(np.load(a.from_npz)) if a.from_npz else fit_ensemble(a.n, a.threads)
    Fs, Gs, F_ll, G_ll = d["Fs"], d["Gs"], d["F_ll"], d["G_ll"]
    n = len(Fs)

    within_F = ens27.pairwise(Fs, Fs, True)
    within_G = ens27.pairwise(Gs, Gs, True)
    between = ens27.pairwise(Gs, Fs, False)
    diff, p_perm = ens27.perm_test_not_worse(Fs, Gs)
    ks = float(stats.ks_2samp(G_ll, F_ll).pvalue)

    summary: dict = {
        "source": str(a.from_npz) if a.from_npz else "fit here",
        "n_fortran": int(len(Fs)),
        "n_pamica": int(len(Gs)),
        "ll": {
            "fortran_mean": float(F_ll.mean()),
            "fortran_sd": float(F_ll.std()),
            "pamica_mean": float(G_ll.mean()),
            "pamica_sd": float(G_ll.std()),
            "pamica_minus_fortran": float(G_ll.mean() - F_ll.mean()),
            "ks_p": ks,
            "mann_whitney_p": float(stats.mannwhitneyu(G_ll, F_ll).pvalue),
        },
    }
    for name, metric, higher_is_worse in (
        ("corr", ama27.xcorr, False),
        ("amari", ama27.model_amari, True),
    ):
        wF = ama27.pairwise(Fs, Fs, True, metric)
        wG = ama27.pairwise(Gs, Gs, True, metric)
        bt = ama27.pairwise(Gs, Fs, False, metric)
        summary[name] = {
            "within_Fortran": {"mean": float(wF.mean()), "sd": float(wF.std())},
            "within_pamica": {"mean": float(wG.mean()), "sd": float(wG.std())},
            "between": {"mean": float(bt.mean()), "sd": float(bt.std())},
            "between_minus_within_Fortran": float(bt.mean() - wF.mean()),
            "perm_p_not_worse": float(
                ama27.perm_test_not_worse(Fs, Gs, metric, higher_is_worse)
            ),
            "range_within_Fortran": [float(wF.min()), float(wF.max())],
            "range_within_pamica": [float(wG.min()), float(wG.max())],
            "range_between": [float(bt.min()), float(bt.max())],
        }
    (HERE / "multimodel_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    corr_detail = {
        (r["implementation"], r["run"]): r
        for r in ama27.per_run_detail(Fs, Gs, ama27.xcorr)
    }
    with open(HERE / "per_run_detail.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "implementation", "run", "final_ll", "corr_within", "corr_between",
                "amari_within", "amari_between",
            ]
        )  # fmt: skip
        for row in ama27.per_run_detail(Fs, Gs, ama27.model_amari):
            key = (row["implementation"], row["run"])
            c = corr_detail[key]
            ll = (F_ll if row["implementation"] == "Fortran" else G_ll)[row["run"]]
            w.writerow(
                [
                    row["implementation"], row["run"], f"{ll:.4f}",
                    f"{c['within']:.4f}", f"{c['between']:.4f}",
                    f"{row['within']:.4f}", f"{row['between']:.4f}",
                ]
            )  # fmt: skip

    ens27.figure(within_F, within_G, between, F_ll, G_ll, diff, p_perm, ks, HERE)
    shutil.copyfile(HERE / "multimodel_ensemble_distributions.png", FIGURE)
    print(f"figure -> {FIGURE} (n={n} each)")


if __name__ == "__main__":
    main()
