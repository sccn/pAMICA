"""Why the harness's 100-iteration log-likelihood gap moved (issue #351).

``validate_implementations.py --backend all`` runs each pamica backend and the
reference binary from their OWN initializations with the same seed number
(numpy's ``RandomState(42)`` for pamica, gfortran's ``random_number`` seeded
with 42 for the binary), so the two start from different points. Its
log-likelihood difference after 100 iterations mixes two things: how far apart
the two starting points leave the fits at iteration 100, and any difference in
the update rule. This script separates them on the harness's own settings.

1. Same start: pamica's seed-42 initialization is written into the binary's
   ``load_*`` files (``pamica/tests/native_oracle.py``) and both sides run the
   harness's 100 iterations from it, single-threaded. What remains is the
   update rule's difference.
2. Start-to-start spread: the binary and pamica each run from several seeds.
   The spread of their final log-likelihoods is what the harness's
   independent-start difference is drawn from.

``ROOT`` is a pamica source tree: the worktree, or a checkout of an older
commit (run it from the worktree's environment; the tree at ``ROOT`` is
imported in place of the installed package, and ``native_oracle`` is read from
this worktree so an old tree without it still works)::

    uv run python .context/issue-351/harness_gap.py ROOT OUT.json [--spread 1,2,3]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKTREE = HERE.parents[1]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--spread", default="", help="comma-separated extra seeds")
    p.add_argument(
        "--binary",
        type=Path,
        default=Path.home() / ".cache/pamica/bin/v0.3.3/amica15-macos-arm64",
    )
    a = p.parse_args()
    root = a.root.resolve()
    sys.path.insert(0, str(root))

    import numpy as np  # noqa: E402

    import pamica  # noqa: E402

    assert pamica.__file__.startswith(str(root)), pamica.__file__
    from pamica.torch_impl import AMICATorchNG  # noqa: E402

    vi = _load(root / "validate_implementations.py", "vi_under_test")
    oracle = _load(WORKTREE / "pamica/tests/native_oracle.py", "native_oracle_351")

    data, params = vi.load_sample_data()
    params = dict(params)
    params["max_iter"] = 100
    X = data.astype(np.float64)
    nw, n = X.shape
    fdt = root / "pamica/sample_data/eeglab_data.fdt"
    report: dict = {"root": str(root), "binary": str(a.binary), "seed": a.seed}

    # --- 1. same start --------------------------------------------------------
    t0 = time.time()
    fit = vi.run_pytorch_amica(data, params, Path(tempfile.mkdtemp()), a.seed)
    torch_dt = time.time() - t0
    init = AMICATorchNG(
        n_channels=nw,
        n_models=1,
        n_mix=params.get("num_mix", 3),
        rho0=params["rho0"],
        seed=a.seed,
        device="cpu",
    )
    init._preprocess(X)
    init._initialize_parameters()
    state = oracle.seed_from_torch(init)
    # The harness's reference settings: the input.param it writes, minus the
    # keys the seeded runner sets itself.
    with tempfile.TemporaryDirectory() as td:
        ref_free = vi.run_fortran_amica(data, params, Path(td), a.seed, a.binary)
        written = (Path(td) / "fortran_run" / "input.param").read_text().splitlines()
    skip = {
        "files", "outdir", "indir", "seed", "max_threads", "data_dim",
        "field_dim", "pcakeep", "max_iter", "num_models", "num_mix_comps",
        "writestep", "write_LLt", "num_comps", "load_W", "doPCA",
        "field_blocksize",
    }  # fmt: skip
    ref_params = {}
    for line in written:
        parts = line.split()
        if (
            len(parts) >= 2
            and parts[0] not in skip
            and not parts[0].startswith("load_")
        ):
            ref_params[parts[0]] = parts[1]
    with tempfile.TemporaryDirectory() as td:
        ref = oracle.run_seeded_reference(
            state, fdt, Path(td), n_samples=n, max_iter=100, threads=1,
            binary=a.binary, **ref_params,
        )  # fmt: skip
    W_ref = np.linalg.inv(ref.A)  # reference A: component k in column k
    ref_ll = ref.LL[ref.LL != 0]
    same = {
        "pamica_final_ll": float(fit["final_ll"]),
        "reference_from_pamica_init_final_ll": float(ref_ll[-1]),
        "abs_ll_difference": abs(float(fit["final_ll"]) - float(ref_ll[-1])),
        "mean_matched_corr": None,
        "min_matched_corr": None,
        "amari": float(vi.amari_distance(W_ref, fit["W"])),
        "pamica_runtime_s": torch_dt,
    }
    cmp = vi.compare_results(
        {"final_ll": ref_ll[-1], "W": W_ref, "final_iter": 100}, fit
    )
    same["mean_matched_corr"] = float(cmp["mean_correlation"])
    same["min_matched_corr"] = float(cmp.get("min_correlation", np.nan))
    free = vi.compare_results(ref_free, fit)
    report["independent_start"] = {
        "reference_final_ll": float(ref_free["final_ll"]),
        "pamica_final_ll": float(fit["final_ll"]),
        "abs_ll_difference": float(free["ll_difference"]),
        "mean_matched_corr": float(free["mean_correlation"]),
        "amari": float(vi.amari_distance(ref_free["W"], fit["W"])),
    }
    report["same_start"] = same

    # --- 2. start-to-start spread ----------------------------------------------
    seeds = [a.seed] + [int(s) for s in a.spread.split(",") if s]
    if len(seeds) > 1:
        F, G = {}, {}
        for s in seeds:
            with tempfile.TemporaryDirectory() as td:
                F[s] = vi.run_fortran_amica(data, params, Path(td), s, a.binary)
            G[s] = (
                fit
                if s == a.seed
                else vi.run_pytorch_amica(data, params, Path(tempfile.mkdtemp()), s)
            )
        f_ll = np.array([F[s]["final_ll"] for s in seeds])
        g_ll = np.array([G[s]["final_ll"] for s in seeds])
        gaps = np.array([abs(g - f) for g in g_ll for f in f_ll])
        corr_fg = [
            vi.compare_results(F[fs], G[gs])["mean_correlation"]
            for fs in seeds
            for gs in seeds
        ]
        corr_ff = [
            vi.compare_results(F[x], F[y])["mean_correlation"]
            for i, x in enumerate(seeds)
            for y in seeds[i + 1 :]
        ]
        report["spread"] = {
            "seeds": seeds,
            "reference_final_ll": f_ll.tolist(),
            "pamica_final_ll": g_ll.tolist(),
            "reference_ll_sd": float(f_ll.std()),
            "pamica_ll_sd": float(g_ll.std()),
            "reference_ll_range": float(np.ptp(f_ll)),
            "pamica_ll_range": float(np.ptp(g_ll)),
            "mean_ll_difference_of_means": float(abs(g_ll.mean() - f_ll.mean())),
            "all_pairs_abs_gap_median": float(np.median(gaps)),
            "all_pairs_abs_gap_min_max": [float(gaps.min()), float(gaps.max())],
            "corr_pamica_vs_reference_mean": float(np.mean(corr_fg)),
            "corr_reference_vs_reference_mean": float(np.mean(corr_ff)),
        }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
