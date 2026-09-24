"""Check that pinning every reference setting explicitly is a no-op today
(issue #351): each native-binary call of ``benchmarks/reproduce_table1.py`` and
of this directory's scripts writes a byte-identical ``input.param`` with the
explicit settings and with the keywords it passed before the pin (which relied
on ``AMICANative``'s defaults, then the bundled ``input.param`` values).

The binary is never run: ``subprocess.run`` in the engine and the seeded runner
of ``pamica/tests/native_oracle.py`` are replaced by functions that record the
parameter file and stop.

    uv run python .context/issue-351/pin_check.py OUT.json

With ``--dump PATH`` it also saves the pinned calls' parameter files, and with
``--against PATH`` it compares them with a saved dump instead, so a change of
``AMICANative``'s defaults (epic #324 Phase 17) can be checked to leave every
pinned call's ``input.param`` unchanged.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))


class _Stop(Exception):
    pass


def native_param(**kw) -> str:
    """The input.param ``AMICANative(**kw).fit`` would run the binary with."""
    import pamica.native.engine as engine

    captured: dict[str, str] = {}

    def fake_run(cmd, cwd=None, **_):
        captured["param"] = (Path(cwd) / "input.param").read_text()
        raise _Stop

    real = engine.subprocess.run
    engine.subprocess.run = fake_run
    try:
        engine.AMICANative(binary=sys.executable, **kw).fit(
            np.zeros((32, 64), dtype=np.float64)
        )
    except _Stop:
        pass
    finally:
        engine.subprocess.run = real
    return captured["param"]


def seeded_param(state_models: int = 1, **kw) -> str:
    """The parameter dict ``run_seeded_reference(state, ..., **kw)`` would run."""
    import pamica.tests.native_oracle as oracle

    captured: dict[str, str] = {}

    def fake_run_reference(param, workdir, **_):
        from pamica.native.engine import _render_param

        captured["param"] = _render_param(param)
        raise _Stop

    nw, nmix = 32, 3
    num_models = state_models
    state = oracle.SeedState(
        A=np.eye(nw * num_models)[:nw].copy() if num_models == 1 else np.ones((nw, nw * num_models)),
        mean=np.zeros(nw),
        mu=np.zeros((nmix, nw * num_models)),
        sbeta=np.ones((nmix, nw * num_models)),
        rho=np.ones((nmix, nw * num_models)),
        alpha=np.ones((nmix, nw * num_models)) / nmix,
        gm=np.ones(num_models) / num_models,
        c=np.zeros((nw, num_models)),
    )  # fmt: skip
    real = oracle._run_reference
    oracle._run_reference = fake_run_reference
    try:
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            oracle.run_seeded_reference(
                state, REPO / "pamica/sample_data/eeglab_data.fdt", Path(td),
                n_samples=64, max_iter=kw.pop("max_iter", 5), threads=1,
                binary=Path(sys.executable), **kw,
            )  # fmt: skip
    except _Stop:
        pass
    finally:
        oracle._run_reference = real
    return captured["param"]


def main() -> None:
    import _reference_settings as rs
    import bundled_single_basins as basins
    import keep_best_ensemble as kb
    import newton_seeds as ns
    from pamica.tests.test_component_rows import _REF_OPT

    t1 = rs._table1()
    checks = {
        # (before: the keywords each call passed before the pin, after: now)
        "reproduce_table1 single-model sweep": (
            dict(threads=4, max_threads=4, timeout=3600, n_models=1, n_mix=3,
                 max_iter=2000, do_newton=0, use_min_dll=0, use_grad_norm=0,
                 block_size=512),
            dict(threads=4, max_threads=4, timeout=3600,
                 **t1.single_model_reference_kwargs(2000)),
        ),
        "reproduce_table1 multi-model ensemble": (
            dict(threads=4, max_threads=4, timeout=1800, n_models=2, n_mix=3,
                 max_iter=100),
            dict(threads=4, max_threads=4, timeout=1800,
                 **t1.multimodel_reference_kwargs(100)),
        ),
        "keep_best_ensemble reference": (
            dict(threads=1, max_threads=1, timeout=3600, n_models=2, n_mix=3,
                 max_iter=300, seed=3),
            kb.fortran_kwargs(300, 3),
        ),
        "bundled_single_basins reference": (
            dict(threads=3, max_threads=3, timeout=3600, n_models=1, n_mix=3,
                 max_iter=2000, do_newton=0, use_min_dll=0, use_grad_norm=0,
                 block_size=512, seed=4),
            basins.fortran_kwargs(t1, 4, 3),
        ),
        "newton_seeds reference": (
            dict(threads=10, max_threads=10, timeout=6 * 3600, seed=1,
                 num_models=1, num_mix_comps=3, max_iter=2000, do_newton=1,
                 use_min_dll=0, use_grad_norm=0, block_size=512),
            dict(threads=10, max_threads=10, timeout=6 * 3600, seed=1,
                 **ns.fortran_kwargs(2000)),
        ),
    }  # fmt: skip
    out = {}
    pinned: dict[str, str] = {}
    for name, (before, after) in checks.items():
        a, b = native_param(**before), native_param(**after)
        pinned[name] = b
        out[name] = {"identical": a == b, "lines": a.count("\n")}
    seeded = {
        "newton_seeds same-start reference": (
            dict(max_iter=2000, do_newton=1, use_min_dll=0, use_grad_norm=0,
                 block_size=512),
            dict(max_iter=2000, **ns.fixinit_kwargs()),
        ),
        "orientation_check": (
            dict(max_iter=1, block_size=512, do_newton=1, use_min_dll=0,
                 use_grad_norm=0),
            dict(max_iter=1, **{**rs.REFERENCE_SETTINGS, "block_size": 512,
                                "do_newton": 1, "use_min_dll": 0,
                                "use_grad_norm": 0}),
        ),
        "share_comps_counts part c": (
            dict(max_iter=8, num_models=2,
                 **{**_REF_OPT, "share_start": 8, "share_iter": 100, "share_comps": 0}),
            dict(max_iter=8, num_models=2,
                 **{**rs.REFERENCE_SETTINGS, **_REF_OPT, "share_start": 8,
                    "share_iter": 100, "share_comps": 0}),
        ),
    }  # fmt: skip
    for name, (before, after) in seeded.items():
        nm = before.get("num_models", 1)
        a = seeded_param(state_models=nm, **before)
        b = seeded_param(state_models=nm, **after)
        pinned[name] = b
        out[name] = {"identical": a == b, "lines": a.count("\n")}
    args = sys.argv[1:]
    if "--dump" in args:
        Path(args[args.index("--dump") + 1]).write_text(json.dumps(pinned, indent=2))
    if "--against" in args:
        saved = json.loads(Path(args[args.index("--against") + 1]).read_text())
        out = {
            name: {
                "pinned_identical_to_saved": pinned[name] == saved[name],
                "unpinned_identical_to_pinned": out[name]["identical"],
            }
            for name in pinned
        }
        ok = all(v["pinned_identical_to_saved"] for v in out.values())
    else:
        ok = all(v["identical"] for v in out.values())
    Path(args[0]).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    if not ok:
        raise SystemExit("a pinned call writes a different input.param")


if __name__ == "__main__":
    main()
