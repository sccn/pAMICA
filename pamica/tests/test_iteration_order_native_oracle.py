"""The iteration order against the native reference binary (issues #339, #345).

Opt-in (``AMICA_RUN_FORTRAN=1``) like the other binary-driven tests. The binary
is the pinned v0.3.3 release asset (:mod:`pamica.tests.native_oracle`), seeded
with pamica's own initialization through its ``load_*`` inputs and run with
``do_history`` on, so its mixing matrix after every iteration can be compared
with PyTorch's and NumPy's, iteration by iteration, together with the
log-likelihood trajectory.

The comparison bound is the reference's own round-off noise in the same setup,
measured in the same test: the binary is run a second time with 4 threads
instead of 1, which only changes the order of its reductions. On this
recording that noise grows quickly (one mixture shape sits at ``rho=1``, where
the location update divides by ``|y|``), so it is the honest floor; each backend
must stay within 10 times it. Measured (maximum over the trajectory, PyTorch /
NumPy against the binary, then the binary against itself):

=====================  ===============================  ==================
configuration          LL deviation                     ``A`` deviation
=====================  ===============================  ==================
decreases, 30 its      2.7e-5 / 8.2e-5 (floor 1.9e-5)   8.1e-4 / 6.8e-4 (floor 6.0e-4)
maxdecs ratchet        6.6e-6 / 8.6e-6 (floor 4.2e-6)   3.0e-4 / 2.7e-4 (floor 2.3e-4)
A-freeze, doscaling    6.6e-7 / 4.4e-7 (floor 4.3e-7)   5.0e-6 / 5.6e-6 (floor 4.3e-6)
A-freeze, no scaling   4.0e-7 / 8.0e-7 (floor 6.1e-7)   1.3e-5 / 1.2e-5 (floor 1.2e-5)
=====================  ===============================  ==================

Before the fix the same comparisons were 1.4e-2 / 2.2e-1 (decreases),
1.3e-2 / 1.4e-1 (ratchet) and 6.1e-3 / 1.9e-1 (A-freeze), hundreds to tens of
thousands of times the floor; each test replays the pre-change PyTorch backend
(through :mod:`pamica.tests.pre_change`) as a control that the bound
discriminates.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest
import torch

from pamica.numpy_impl.core import AMICA as AMICA_NumPy
from pamica.torch_impl.core import AMICATorchNG
from pamica.torch_impl.utils import load_eeglab_data

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_data"
DATA_FILE = SAMPLE_DIR / "eeglab_data.fdt"
NW = 32
FIELD = 30504
NMIX = 3
SEED = 42
# The epic branch before issue #339: the order and freeze this change replaces.
PRE_CHANGE_COMMIT = "5b6ae4f69eacf6aa18002ce336ed904f73438516"
# Each backend within this many times the reference's own round-off noise.
NOISE_FACTOR = 10.0
# The pre-change code at least this many times the noise, so the bound above
# discriminates.
CONTROL_FACTOR = 100.0

pytestmark = [
    pytest.mark.skipif(
        os.environ.get("AMICA_RUN_FORTRAN") != "1",
        reason="opt-in Fortran-binary integration test (set AMICA_RUN_FORTRAN=1)",
    ),
    pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing"),
]

# The reference's input.param optimizer, no Newton, block size pinned on both
# sides; pamica keyword -> reference keyword where the names differ.
_OPT: Dict[str, Any] = dict(
    block_size=512,
    lrate=0.05,
    lratefact=0.5,
    rholrate=0.05,
    rholratefact=0.5,
    rho0=1.5,
    minrho=1.0,
    maxrho=2.0,
    invsigmin=0.0,
    invsigmax=100.0,
    do_newton=False,
    newt_start=50,
    newtrate=1.0,
    newt_ramp=10,
    maxdecs=5,
    doscaling=True,
)
_CONFIGS: Dict[str, Dict[str, Any]] = {
    # lrate 0.5 overshoots from iteration 7 on, 8 decreases in 30 iterations.
    "decreases": dict(lrate=0.5),
    # maxdecs=2 completes cycles that ratchet the lrate ceiling, and with
    # newt_start=1 the rho-rate ceiling too (amica15.f90:1066-1068).
    "ratchet": dict(lrate=0.5, maxdecs=2, newt_start=1),
    # share_comps off: A is held on iterations 3-5 and 10-15 (amica15.f90:1803).
    "freeze": dict(share_start=3, share_iter=10),
    "freeze-noscaling": dict(share_start=3, share_iter=10, doscaling=False),
}


@pytest.fixture(scope="module")
def X() -> np.ndarray:
    return load_eeglab_data(str(DATA_FILE), data_dim=NW, field_dim=FIELD).astype(
        np.float64
    )


@pytest.fixture(scope="module")
def pre(tmp_path_factory):
    """The package before this change, imported beside the live one."""
    from pamica.tests.pre_change import load_pre_change_package

    return load_pre_change_package(
        PRE_CHANGE_COMMIT, "pamica_pre339", tmp_path_factory.mktemp("pre339")
    )


def _cfg(name: str) -> Dict[str, Any]:
    return {**_OPT, **_CONFIGS[name]}


def _reference_params(cfg: Dict[str, Any]) -> Dict[str, Any]:
    params = {k: v for k, v in cfg.items() if k not in ("maxdecs", "doscaling")}
    params.update(
        max_decs=cfg["maxdecs"],
        do_newton=int(cfg["do_newton"]),
        doscaling=int(cfg["doscaling"]),
        do_opt_block=0,
        do_reject=0,
        share_comps=0,
        scalestep=1,
        do_history=1,
        histstep=1,
    )
    return params


def _torch(cls: Any, cfg: Dict[str, Any]) -> Any:
    return cls(
        n_channels=NW,
        n_mix=NMIX,
        seed=SEED,
        device="cpu",
        dtype=torch.float64,
        keep_best=False,
        **cfg,
    )


def _with_history(model: Any, get_a) -> List[np.ndarray]:
    """``A`` in the reference layout after every real ``_update_parameters``
    call (a call-recording spy that calls straight through)."""
    out: List[np.ndarray] = []
    original = model._update_parameters

    def spy(*args):
        result = original(*args)
        out.append(get_a())
        return result

    model._update_parameters = spy
    return out


def _deviation(
    ll: List[float], a: List[np.ndarray], ref_ll: np.ndarray, ref_a: Dict[int, Any]
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-iteration |LL| and max |A| deviation over the common iterations."""
    n = min(len(ll), len(ref_ll))
    dll = np.abs(np.asarray(ll[:n]) - ref_ll[:n])
    da = np.array(
        [np.abs(a[i] - ref_a[i + 1]).max() for i in range(len(a)) if i + 1 in ref_a]
    )
    return dll, da


def _decreases(ll) -> List[int]:
    return [i for i in range(1, len(ll)) if ll[i] < ll[i - 1]]


def _run(name: str, X: np.ndarray, workdir: Path, max_iter: int, pre: Any) -> dict:
    """The reference (1 and 4 threads), PyTorch, NumPy and the pre-change
    PyTorch, all from one seeded state."""
    from pamica.tests.native_oracle import (
        history_mixing,
        reference_mixing,
        run_seeded_reference,
        seed_from_torch,
    )

    cfg = _cfg(name)
    init = _torch(AMICATorchNG, cfg)
    init._preprocess(X)
    init._initialize_parameters()
    state = seed_from_torch(init)
    ref: Dict[int, Any] = {}
    for threads in (1, 4):
        wd = workdir / f"t{threads}"
        out = run_seeded_reference(
            state,
            DATA_FILE,
            wd,
            n_samples=FIELD,
            max_iter=max_iter,
            threads=threads,
            **_reference_params(cfg),
        )
        n = int((out.LL != 0).sum())
        ref[threads] = (out.LL[:n], history_mixing(wd, NW, NW))
        assert np.abs(out.S - init.sphere.numpy()).max() < 1e-12

    fits = {}
    t = _torch(AMICATorchNG, cfg)
    t_a = _with_history(t, lambda: reference_mixing(t.A.numpy()))
    t.fit(X, max_iter=max_iter, verbose=False)
    fits["torch"] = (list(t.ll_history), t_a)

    numpy_cfg = {k: v for k, v in cfg.items() if k not in ("maxdecs", "share_iter")}
    if "share_iter" in cfg:
        numpy_cfg["share_int"] = cfg["share_iter"]
    n = AMICA_NumPy(
        num_models=1,
        num_mix=NMIX,
        seed=SEED,
        max_iter=max_iter,
        use_tqdm=False,
        do_opt_block=False,
        writestep=10**7,
        outdir=str(workdir / "np"),
        max_decs=cfg["maxdecs"],
        **numpy_cfg,
    )
    # The NumPy backend draws its own initialization from a different
    # generator and keeps any parameter already set, so seed it the same way.
    for attr in ("A", "mu", "beta", "rho", "alpha", "gm", "c"):
        setattr(n, attr, getattr(init, attr).numpy().copy())

    def numpy_mixing() -> np.ndarray:
        assert n.A is not None, "the NumPy backend was seeded with A above"
        return reference_mixing(n.A)

    n_a = _with_history(n, numpy_mixing)
    n.fit(X)
    fits["numpy"] = ([float(v) for v in n.ll], n_a)

    old = _torch(pre.torch_impl.core.AMICATorchNG, cfg)
    old_a = _with_history(old, lambda: reference_mixing(old.A.numpy()))
    old.fit(X, max_iter=max_iter, verbose=False)
    fits["pre-change"] = (list(old.ll_history), old_a)
    return {"ref": ref, "fits": fits}


def _floor(run: dict) -> Tuple[float, float]:
    """The reference's own round-off noise: 1 thread against 4."""
    (ll1, a1), (ll4, a4) = run["ref"][1], run["ref"][4]
    dll, da = _deviation(list(ll4), [a4[i] for i in sorted(a4)], ll1, a1)
    return float(dll.max()), float(da.max())


def _check_against_floor(run: dict) -> None:
    ref_ll, ref_a = run["ref"][1]
    floor_ll, floor_a = _floor(run)
    assert 0.0 < floor_ll and 0.0 < floor_a, "the reference noise was not measured"
    for backend in ("torch", "numpy"):
        ll, a = run["fits"][backend]
        assert len(ll) == len(ref_ll), f"{backend} ran {len(ll)} iterations"
        dll, da = _deviation(ll, a, ref_ll, ref_a)
        assert dll.max() <= NOISE_FACTOR * floor_ll, (
            f"{backend}: LL off by {dll.max():.2e}, reference noise {floor_ll:.2e}"
        )
        assert da.max() <= NOISE_FACTOR * floor_a, (
            f"{backend}: A off by {da.max():.2e}, reference noise {floor_a:.2e}"
        )
    ll, a = run["fits"]["pre-change"]
    dll, da = _deviation(ll, a, ref_ll, ref_a)
    assert max(dll.max() / floor_ll, da.max() / floor_a) > CONTROL_FACTOR, (
        "the pre-change code also sits at the noise floor: the configuration "
        "does not exercise this change"
    )


# --- #339: the decrease response before the update --------------------------


@pytest.mark.parametrize("name", ["decreases", "ratchet"])
def test_decrease_response_matches_the_reference(name, X, pre, tmp_path):
    """With the likelihood-decrease response ahead of the update, PyTorch and
    NumPy see the reference's decreases on the same iterations and track its
    likelihood and mixing matrix at its own round-off noise through 30
    iterations of overshooting (issue #339)."""
    run = _run(name, X, tmp_path, 30, pre)
    ref_ll, _ = run["ref"][1]
    ref_decreases = _decreases(ref_ll)
    assert len(ref_decreases) >= 3, "the configuration no longer overshoots"
    if name == "ratchet":
        # A maxdecs=2 cycle completes past newt_start=1: both ceilings ratchet.
        numdecs, cycles = 0, 0
        for i in ref_decreases:
            numdecs += 1
            if numdecs >= 2:
                cycles, numdecs = cycles + 1, 0
        assert cycles >= 1
    for backend in ("torch", "numpy"):
        ll, _ = run["fits"][backend]
        assert _decreases(ll) == ref_decreases, backend
    _check_against_floor(run)


# --- #345: the A-freeze without sharing ---------------------------------------


@pytest.mark.parametrize("name", ["freeze", "freeze-noscaling"])
def test_a_freeze_without_sharing_matches_the_reference(name, X, pre, tmp_path):
    """With ``share_comps`` off, the reference holds ``A`` on iterations 3-5
    and 10-15 (``share_start=3``, ``share_iter=10``), and so do PyTorch and
    NumPy, tracking it at its own round-off noise (issue #345).

    Read straight off the binary's own history first: its ``A`` is unchanged
    on exactly those iterations (to the rescale's round-off with
    ``doscaling`` on), which is the reference behavior this change adopts.
    """
    k = 16
    run = _run(name, X, tmp_path, k, pre)
    expected = [i for i in range(1, k + 1) if i >= 3 and i % 10 <= 5]
    assert expected == [3, 4, 5, 10, 11, 12, 13, 14, 15]
    scaling = _cfg(name)["doscaling"]

    def held(steps: List[float]) -> List[int]:
        # With doscaling on, a held iteration still renormalizes A: the
        # reference's own A moves by up to 3.3e-16 there.
        bound = 8 * np.finfo(np.float64).eps if scaling else 0.0
        return [i + 2 for i, step in enumerate(steps) if step <= bound]

    _, ref_a = run["ref"][1]
    ref_steps = [np.abs(ref_a[i + 1] - ref_a[i]).max() for i in range(1, k)]
    assert held(ref_steps) == expected
    for backend in ("torch", "numpy"):
        _, a = run["fits"][backend]
        steps = [np.abs(a[i + 1] - a[i]).max() for i in range(k - 1)]
        assert held(steps) == expected, backend
        if not scaling:
            for itf in expected:
                np.testing.assert_array_equal(a[itf - 1], a[itf - 2])
    _check_against_floor(run)
