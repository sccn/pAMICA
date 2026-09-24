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

import matplotlib
import numpy as np
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

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


# The figure of `.context/issue-27/multimodel_ensemble.py`, with a line style
# for each distribution besides its color (the #27 record keeps its own copy).
C_FORT, C_NG, C_BET = "#0072B2", "#E69F00", "#009E73"  # Okabe-Ito


def figure(within_F, within_G, between, F_ll, G_ll, diff, p_perm, ks, out):
    # Sized to its ACTUAL print footprint, not a big on-screen canvas: paper.md
    # embeds this at width=100% of a ~5.36in single-column page (measured from the
    # compiled paper.pdf), so figsize is set to that width directly -- LaTeX then
    # displays it near 1:1 instead of shrinking a much larger canvas down to fit,
    # which previously collapsed every font to an unreadable ~3pt in the printed
    # PDF even though it looked fine on screen (figure-qa print-scale finding).
    plt.rcParams.update(
        {"font.size": 7, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(5.36, 4.2))
    bins = np.linspace(0.5, 1.0, 26)
    for arr, col, ls, lab in [
        (within_F, C_FORT, "-", "within-Fortran"),
        (within_G, C_NG, "--", "within-pamica"),
        (between, C_BET, ":", "between (pamica-Fortran)"),
    ]:
        axA.hist(arr, bins=bins, density=True, color=col, alpha=0.25)
        axA.hist(
            arr, bins=bins, density=True, histtype="step", color=col, ls=ls, lw=2,
            label=lab,
        )  # fmt: skip
        axA.axvline(arr.mean(), color=col, ls=ls, lw=1.2)
    axA.set_xlabel("Hungarian-matched cross-correlation\n(stacked 2x32 components)")
    axA.set_ylabel("density")
    axA.set_title(
        "A  Partition-agreement distributions",
        loc="left",
        fontweight="bold",
        fontsize=8,
    )
    bins_ll = np.linspace(
        min(F_ll.min(), G_ll.min()) - 0.005, max(F_ll.max(), G_ll.max()) + 0.005, 24
    )
    for arr, col, ls, lab in [
        (F_ll, C_FORT, "-", "Fortran"),
        (G_ll, C_NG, "--", "pamica"),
    ]:
        axB.hist(arr, bins=bins_ll, density=True, color=col, alpha=0.25)
        axB.hist(
            arr, bins=bins_ll, density=True, histtype="step", color=col, ls=ls, lw=2,
            label=lab,
        )  # fmt: skip
        axB.axvline(arr.mean(), color=col, ls=ls, lw=1.2)
    axB.set_xlabel("final log-likelihood\n(mean per sample-channel)")
    axB.set_ylabel("density")
    axB.set_title(
        "B  Likelihood distributions", loc="left", fontweight="bold", fontsize=8
    )

    # Both the legend and the stats box previously sat inside the axes and ended up
    # overlapping the histogram bars (and each other) no matter where they were
    # anchored -- with 3 distributions filling most of the plotted range, there was
    # no empty pocket big enough to hold either. Reserve a fixed bottom margin for
    # both instead and place them with figure-fraction (not axes-fraction)
    # coordinates: axes-fraction anchors turned out to depend on the final axes
    # height that tight_layout picks, which isn't known in advance and caused the
    # legend/text to collide with the xlabel above it. get_position() gives each
    # axes' true horizontal center after the layout below is fixed, so this keeps
    # each panel's legend/text under its own histogram, not bleeding into the
    # other panel.
    fig.subplots_adjust(top=0.80, bottom=0.42, left=0.11, right=0.97, wspace=0.45)
    cx_a = sum(axA.get_position().intervalx) / 2
    cx_b = sum(axB.get_position().intervalx) / 2

    handles_a, labels_a = axA.get_legend_handles_labels()
    fig.legend(
        handles_a, labels_a, frameon=False, fontsize=6,
        loc="upper center", bbox_to_anchor=(cx_a, 0.31),
    )  # fmt: skip
    fig.text(
        cx_a,
        0.20,
        f"mean corr. (run pairs)\n"
        f"within-Fortran: {within_F.mean():.3f} (n={len(within_F)})\n"
        f"within-pamica: {within_G.mean():.3f} (n={len(within_G)})\n"
        f"between: {between.mean():.3f} (n={len(between)})\n"
        f"diff: {diff:+.3f} (margin +/-0.05)\n"
        f"perm. p={p_perm:.2f}",
        ha="center",
        va="top",
        fontsize=5.5,
        bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.95),
    )

    handles_b, labels_b = axB.get_legend_handles_labels()
    fig.legend(
        handles_b, labels_b, frameon=False, fontsize=6,
        loc="upper center", bbox_to_anchor=(cx_b, 0.31),
    )  # fmt: skip
    fig.text(
        cx_b,
        0.23,
        f"mean final LL\n"
        f"Fortran: {F_ll.mean():.4f} (sd {F_ll.std():.3f})\n"
        f"pamica: {G_ll.mean():.4f} (sd {G_ll.std():.3f})\n"
        f"gap: {abs(F_ll.mean() - G_ll.mean()):.4f} (100-iter budget)\n"
        f"KS p={ks:.2f}"
        if ks >= 0.01
        else f"KS p={ks:.0e}",
        ha="center",
        va="top",
        fontsize=5.5,
        bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9),
    )

    fig.text(
        0.5,
        0.90,
        "Vertical lines mark each distribution's mean, in its line style.",
        ha="center",
        fontsize=6.5,
        style="italic",
    )
    fig.suptitle(
        "Multi-model AMICA (n_models=2): pamica vs Fortran ensembles, real sample EEG",
        fontweight="bold",
        fontsize=8.5,
        y=0.97,
    )
    fig.savefig(
        out / "multimodel_ensemble_distributions.png", bbox_inches="tight", dpi=300
    )
    fig.savefig(out / "multimodel_ensemble_distributions.pdf", bbox_inches="tight")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("n", nargs="?", type=int, default=20)
    p.add_argument("--from-npz", type=Path)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--tag", default="", help="suffix for the output file names")
    p.add_argument("--no-figure", action="store_true")
    a = p.parse_args()
    sfx = f"_{a.tag}" if a.tag else ""
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
    (HERE / f"multimodel_summary{sfx}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    corr_detail = {
        (r["implementation"], r["run"]): r
        for r in ama27.per_run_detail(Fs, Gs, ama27.xcorr)
    }
    with open(HERE / f"per_run_detail{sfx}.csv", "w", newline="") as f:
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

    if a.no_figure:
        return
    figure(within_F, within_G, between, F_ll, G_ll, diff, p_perm, ks, HERE)
    shutil.copyfile(HERE / "multimodel_ensemble_distributions.png", FIGURE)
    print(f"figure -> {FIGURE} (n={n} each)")


if __name__ == "__main__":
    main()
