"""Basins of the bundled tier's single-model comparison (issue #351 item 2).

``benchmarks/reproduce_table1.py --tier bundled`` compares five pamica fits
(seeds 301-305) with five clock-seeded reference fits, one pair per seed, on
the bundled sample (32 channels, k ~ 30), Newton off, 2000 iterations, and
reports only the means. This script repeats that single-model protocol with
the unmixing matrices kept, so every pair can be compared: the same pamica
seeds (deterministic, so they reproduce the tier's pamica fits) and seeded
reference runs (``seed`` 1..N, reproducible), with the tier's keyword sets.

    uv run python .context/issue-351/bundled_single_basins.py OUT.npz [N_FORTRAN] [--threads T]
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def _table1():
    spec = importlib.util.spec_from_file_location(
        "reproduce_table1_basins", REPO / "benchmarks/reproduce_table1.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["reproduce_table1_basins"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("n_fortran", nargs="?", type=int, default=10)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--pamica-seeds", default="301,302,303,304,305")
    a = p.parse_args()
    t1 = _table1()
    from pamica import AMICA, AMICANative

    data, _ = t1.load_bundled_data()
    binary = t1.resolve_binary(None, "v0.3.3")
    runs: dict[str, dict] = {}
    for s in range(1, a.n_fortran + 1):
        eng = AMICANative(
            binary=binary, threads=a.threads, max_threads=a.threads, timeout=3600,
            n_models=1, n_mix=3, max_iter=2000, do_newton=0, use_min_dll=0,
            use_grad_norm=0, block_size=512, seed=s,
        )  # fmt: skip
        eng.fit(data)
        assert eng.output_ is not None
        runs[f"fortran_seed{s}"] = {
            "W": eng.output_.W[:, :, 0],
            "ll": float(eng.output_.LL[-1]),
        }
        print(f"fortran seed {s}: LL={runs[f'fortran_seed{s}']['ll']:.6f}", flush=True)
    for s in [int(x) for x in a.pamica_seeds.split(",") if x]:
        model = AMICA(n_models=1, n_mix=3, device="cpu", verbose=False)
        model.fit(
            data, max_iter=2000, lrate=0.05, do_mean=True, do_sphere=True,
            do_approx_sphere=True, do_newton=False, seed=s, block_size=512,
            minlrate=1e-8, lratefact=0.5, maxdecs=3, newt_start=50, newt_ramp=10,
            newtrate=1.0, rho0=1.5, minrho=1.0, maxrho=2.0, rholrate=0.05,
            rholratefact=0.5, invsigmin=0.0, invsigmax=100.0, doscaling=True,
            scalestep=1,
        )  # fmt: skip
        runs[f"pamica_seed{s}"] = {
            "W": model.get_unmixing_matrix(0),
            "ll": float(model.final_ll_),
            "iters": len(model.ll_history_),
            "stop": str(model.stop_reason_),
        }
        print(
            f"pamica seed {s}: LL={model.final_ll_:.6f} iters={len(model.ll_history_)} "
            f"stop={model.stop_reason_}",
            flush=True,
        )
    names = list(runs)
    np.savez(
        a.out,
        names=np.array(names),
        W=np.stack([runs[k]["W"] for k in names]),
        ll=np.array([runs[k]["ll"] for k in names]),
    )
    rep: dict = {"ll": {k: runs[k]["ll"] for k in names}, "pairs": {}}
    for x, y in itertools.combinations(names, 2):
        c = t1.xcorr(runs[x]["W"], runs[y]["W"])
        rep["pairs"][f"{x} vs {y}"] = {
            "mean_corr": float(c.mean()),
            "min_corr": float(c.min()),
            "amari": float(t1.amari_distance(runs[x]["W"], runs[y]["W"])),
        }
    for k in ("iters", "stop"):
        rep[k] = {n: runs[n][k] for n in names if k in runs[n]}
    a.out.with_suffix(".json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep["ll"], indent=2))


if __name__ == "__main__":
    main()
