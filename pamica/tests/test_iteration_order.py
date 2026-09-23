"""Each iteration follows the reference's order (issues #339 and #345).

The reference's main loop (amica15.f90:949-1142) runs, per iteration, the
E-step (``LL(iter)``, the sufficient statistics and the step ``dAk`` with its
norm ``ndtmpsum``), then the likelihood-decrease response and the stopping
checks, exits on a stop before any parameter moves (:1111), and only then calls
``update_params`` (:1122) with the rates the response just set. Inside
``update_params`` it holds the mixing-matrix update, with its lrate ramp and the
reset of the working rho rate, whenever ``iter >= share_start`` and
``mod(iter, share_iter) <= 5`` (:1803), whether or not ``share_comps`` is on.

Every claim is observed through real fits on the bundled sample EEG, with no
mocks (``.rules/testing.md``): the only instrumentation is a call-recording spy
on the real ``_update_parameters``, which calls straight through and never
substitutes a result. Cross-backend by design (``.rules/backend_parity.md``):
PyTorch and NumPy always run; MLX skips per test (``pytest.importorskip`` plus
an Apple-GPU guard), never the module. Expected iterations are computed here
from the reference's own arithmetic, not read back from :mod:`pamica.schedule`.
The native-binary oracles for the same behavior are
``test_iteration_order_native_oracle.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

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
BACKENDS = ["torch", "numpy", "mlx"]

pytestmark = pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing")


@pytest.fixture(scope="module")
def X() -> np.ndarray:
    return load_eeglab_data(str(DATA_FILE), data_dim=NW, field_dim=FIELD).astype(
        np.float64
    )


def _mlx_class():
    """``AMICAMLXNG``, or skip this test (never the whole module)."""
    mlx_core = pytest.importorskip(
        "pamica.mlx_impl.core", reason="MLX not installed (Apple Silicon only)"
    )
    mx = mlx_core.mx
    if mx.default_device().type != mx.DeviceType.gpu:
        pytest.skip("no Apple GPU")
    return mlx_core.AMICAMLXNG


def _model(backend: str, max_iter: int, tmp_path: Path, **cfg: Any) -> Any:
    """An unfitted one-model backend; ``cfg`` uses the torch/MLX names."""
    if backend == "numpy":
        # NumPy's spellings; it has no keep_best (it always returns the last
        # iterate, which is what keep_best=False asks of the others).
        renames = {"share_iter": "share_int", "maxdecs": "max_decs"}
        params = {renames.get(k, k): v for k, v in cfg.items() if k != "keep_best"}
        return AMICA_NumPy(
            num_models=1,
            num_mix=NMIX,
            seed=SEED,
            max_iter=max_iter,
            use_tqdm=False,
            do_opt_block=False,
            writestep=10**7,
            outdir=str(tmp_path / f"np{len(list(tmp_path.iterdir()))}"),
            **params,
        )
    if backend == "torch":
        return AMICATorchNG(
            n_channels=NW,
            n_mix=NMIX,
            seed=SEED,
            device="cpu",
            dtype=torch.float64,
            **cfg,
        )
    return _mlx_class()(n_channels=NW, n_mix=NMIX, seed=SEED, **cfg)


def _fit(model: Any, backend: str, x: np.ndarray, max_iter: int) -> List[float]:
    """Fit and return the per-iteration log-likelihood trajectory."""
    if backend == "numpy":
        model.fit(x)
        return [float(v) for v in model.ll]
    model.fit(x, max_iter=max_iter, verbose=False)
    return [float(v) for v in model.ll_history]


def _as_np(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().copy()
    return np.array(value, copy=True)


def _iteration(model: Any, backend: str) -> int:
    return model.iter if backend == "numpy" else model.iteration


def _record_updates(model: Any, backend: str, rescaled: bool = False) -> List[Dict]:
    """Record every call of the real ``_update_parameters``.

    A call-recording spy (``.rules/testing.md``): it stores ``A`` and ``lrate``
    before and after, the step it was handed and the iteration, and calls the
    real method with the same arguments. With ``rescaled``, it also stores what
    the backend's own ``doscaling`` rescale alone would make of the pre-update
    ``A``, computed on the live model and then undone, so nothing it touches
    outlives the probe.
    """
    records: List[Dict] = []
    original = model._update_parameters

    def spy(*args):
        rec: Dict[str, Any] = {
            "it": _iteration(model, backend),
            "A_before": _as_np(model.A),
            "lrate_before": float(model.lrate),
            "dAk": _as_np(args[-1].dAk),
        }
        if rescaled:
            saved = (model.A, model.mu, model.beta)
            if backend == "torch":
                model.A, model.mu, model.beta = (t.clone() for t in saved)
            model._rescale_components()
            rec["A_rescaled"] = _as_np(model.A)
            model.A, model.mu, model.beta = saved
        result = original(*args)
        rec["A_after"] = _as_np(model.A)
        rec["lrate_after"] = float(model.lrate)
        records.append(rec)
        return result

    model._update_parameters = spy
    return records


def _reference_holds(itf: int, share_start: int, share_iter: int) -> bool:
    """amica15.f90:1803: A is updated only if ``iter < share_start`` or
    ``mod(iter, share_iter) > 5``, over the reference's 1-based ``iter``."""
    return not (itf < share_start or itf % share_iter > 5)


# --- #345: the A-freeze without sharing ------------------------------------


# share_start=3, share_iter=10: iterations 3-5 and 10-15 are held. The stops
# are off so every scheduled iteration runs.
_FREEZE = dict(share_start=3, share_iter=10, use_min_dll=False, use_grad_norm=False)
_FREEZE_ITERS = 16
_FREEZE_FRAMES = 4096


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_is_held_on_exactly_the_reference_iterations_without_sharing(
    backend, X, tmp_path
):
    """With ``share_comps`` off and ``doscaling`` off, ``A`` is bit for bit
    unchanged across exactly the iterations the reference holds, and moves on
    every other one (issue #345). Before the fix it moved on all of them."""
    model = _model(backend, _FREEZE_ITERS, tmp_path, doscaling=False, **_FREEZE)
    records = _record_updates(model, backend)
    ll = _fit(model, backend, X[:, :_FREEZE_FRAMES], _FREEZE_ITERS)
    assert len(ll) == len(records) == _FREEZE_ITERS, "the fit did not run through"

    held = [r["it"] + 1 for r in records if np.array_equal(r["A_after"], r["A_before"])]
    expected = [
        itf for itf in range(1, _FREEZE_ITERS + 1) if _reference_holds(itf, 3, 10)
    ]
    assert expected == [3, 4, 5, 10, 11, 12, 13, 14, 15]
    assert held == expected
    for r in records:
        if r["it"] + 1 not in expected:
            assert np.abs(r["A_after"] - r["A_before"]).max() > 1e-4
        else:
            # The ramp and the step are held together (amica15.f90:1804-1816).
            assert r["lrate_after"] == r["lrate_before"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_held_iteration_only_rescales_with_doscaling_on(backend, X, tmp_path):
    """With ``doscaling`` on (the default), a held iteration still renormalizes
    each component, as the reference's rescale runs outside the guarded branch
    (amica15.f90:1843): ``A`` then ends exactly where the backend's own rescale
    alone takes the pre-update ``A``, a change of round-off size (the
    reference's own ``A`` moves by up to 3.3e-16 on such iterations), and the
    other iterations still take their step."""
    model = _model(backend, _FREEZE_ITERS, tmp_path, doscaling=True, **_FREEZE)
    records = _record_updates(model, backend, rescaled=True)
    ll = _fit(model, backend, X[:, :_FREEZE_FRAMES], _FREEZE_ITERS)
    assert len(ll) == len(records) == _FREEZE_ITERS, "the fit did not run through"

    for r in records:
        itf = r["it"] + 1
        if _reference_holds(itf, 3, 10):
            np.testing.assert_array_equal(r["A_after"], r["A_rescaled"])
            eps = np.finfo(r["A_after"].dtype).eps
            assert np.abs(r["A_after"] - r["A_before"]).max() <= 8 * eps
        else:
            assert np.abs(r["A_after"] - r["A_rescaled"]).max() > 1e-4


# The shipped defaults, share_start = share_iter = 100: every fit of 100 or more
# iterations is held on 100-105. A short slice keeps it cheap; the stops are off
# so the fit reaches the window.
_DEFAULT_WINDOW_ITERS = 107
_DEFAULT_WINDOW_FRAMES = 2048


@pytest.mark.parametrize("backend", BACKENDS)
def test_default_fits_hold_a_on_iterations_100_to_105(backend, X, tmp_path):
    """The defaults hold ``A`` on iterations 100-105 of every fit, sharing off,
    as the reference does (the bundled ``amicaout`` fixture shows it: its
    ``log|det W|`` stays constant on iterations 101-106 of ``out.txt``)."""
    model = _model(
        backend,
        _DEFAULT_WINDOW_ITERS,
        tmp_path,
        doscaling=False,
        use_min_dll=False,
        use_grad_norm=False,
    )
    assert (model.share_start, _share_iter(model, backend)) == (100, 100)
    records = _record_updates(model, backend)
    ll = _fit(model, backend, X[:, :_DEFAULT_WINDOW_FRAMES], _DEFAULT_WINDOW_ITERS)
    assert len(ll) == len(records) == _DEFAULT_WINDOW_ITERS
    held = [r["it"] + 1 for r in records if np.array_equal(r["A_after"], r["A_before"])]
    assert held == [100, 101, 102, 103, 104, 105]


def _share_iter(model: Any, backend: str) -> int:
    return model.share_int if backend == "numpy" else model.share_iter


@pytest.mark.parametrize("share_comps", [False, True])
@pytest.mark.parametrize("bad", [6, 1, 0, -3, 2.5, True])
@pytest.mark.parametrize("backend", BACKENDS)
def test_every_backend_rejects_a_share_iter_below_7(
    backend, bad, share_comps, tmp_path
):
    """``share_iter <= 6`` would hold ``A`` on every iteration from
    ``share_start`` on (every remainder is 0-5), and the freeze applies with
    sharing off too, so every constructor rejects it either way, with one
    message that says why. 7 is accepted."""
    name = "share_int" if backend == "numpy" else "share_iter"
    with pytest.raises(ValueError, match=rf"{name} must be an integer >= 7") as err:
        _model(backend, 1, tmp_path, share_comps=share_comps, share_iter=bad)
    assert "would never update A again" in str(err.value)
    assert "amica15.f90:1803" in str(err.value)
    ok = _model(backend, 1, tmp_path, share_comps=share_comps, share_iter=7)
    assert _share_iter(ok, backend) == 7


# --- #339: the decrease response acts on the same iteration's update ---------


# A learning rate high enough that the likelihood falls within the first
# iterations: on 4096 samples the first decrease is ll index 7, by 8.8e-3, on
# PyTorch and MLX alike (PyTorch and NumPy share the trajectory). doscaling off
# and one model make each update exactly A <- A - rate * dAk, so the rate each
# update used can be read off A itself.
_DECREASE = dict(
    lrate=0.6, lratefact=0.5, newt_ramp=10, doscaling=False, keep_best=False
)
_DECREASE_ITERS = 10
_DECREASE_FRAMES = 4096


def _ramp(rate: float, cap: float, newt_ramp: int) -> float:
    """The natural-gradient ramp inside the A branch (amica15.f90:1812)."""
    return min(cap, rate + min(1.0 / newt_ramp, rate))


@pytest.mark.parametrize("backend", BACKENDS)
def test_the_step_after_a_decrease_already_uses_the_halved_rate(backend, X, tmp_path):
    """On the iteration whose likelihood fell, the reference halves ``lrate``
    before that iteration's update, so the step it takes is ramped from the
    halved rate (amica15.f90:1061-1062, then :1812-1814). Read off the real
    step: the rate that took ``A`` to its new value is ``ramp(r / 2)``, where
    ``r`` is the previous step's rate, not ``ramp(r)``, which is what the
    previous order (halving after the update) took. Before issue #339 every
    backend took ``ramp(r)`` here."""
    model = _model(backend, _DECREASE_ITERS, tmp_path, **_DECREASE)
    records = _record_updates(model, backend)
    ll = _fit(model, backend, X[:, :_DECREASE_FRAMES], _DECREASE_ITERS)
    assert len(ll) == len(records) == _DECREASE_ITERS

    decreases = [i for i in range(1, len(ll)) if ll[i] < ll[i - 1]]
    assert decreases, "the likelihood never fell: the configuration exposes nothing"
    d = decreases[0]
    assert d + 1 < _DECREASE_ITERS

    def rate(r: Dict) -> float:
        # A_after = A_before - rate * dAk, so the rate is the projection.
        step = r["A_before"] - r["A_after"]
        return float(np.sum(step * r["dAk"]) / np.sum(r["dAk"] * r["dAk"]))

    rtol = 1e-4 if backend == "mlx" else 1e-12
    previous, taken = rate(records[d - 1]), rate(records[d])
    # The recorded lrate is the one the step used.
    assert taken == pytest.approx(records[d]["lrate_after"], rel=rtol)
    cap = _DECREASE["lrate"]
    halved_first = _ramp(previous * _DECREASE["lratefact"], cap, 10)
    halved_after = _ramp(previous, cap, 10)
    assert abs(halved_first - halved_after) > 0.1 * halved_after, (
        "the two orders would take the same step here; nothing is discriminated"
    )
    assert taken == pytest.approx(halved_first, rel=rtol)
    # Before the decrease the rate had ramped to its ceiling and stayed there.
    assert previous == pytest.approx(cap, rel=rtol)


# --- #339: a stop returns the parameters its final likelihood describes -------


# A min_dll stop within a few iterations: gains below 1e-3 for more than 2
# iterations in a row.
_STOP = dict(use_min_dll=True, min_dll=1e-3, maxincs=2, keep_best=False)
_STOP_ITERS = 60
_STOP_FRAMES = 4096


def _recomputed_ll(model: Any, backend: str, x: np.ndarray) -> float:
    """The log-likelihood of the model's current parameters on ``x``, computed
    by the backend's own E-step exactly as its fit loop computes it."""
    if backend == "numpy":
        return float(model._get_updates_and_likelihood()["ll"])
    x_t = model._preprocess(x)
    acc = model._accumulate_blocks(x_t)
    n = x_t.shape[1]
    return float((acc["ll"] / (n * model.n_channels)).item())


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_stopped_fit_returns_the_parameters_its_final_ll_describes(
    backend, X, tmp_path
):
    """A convergence stop exits before its iteration's update, as the reference
    does (amica15.f90:1111), so the returned parameters reproduce the recorded
    final log-likelihood exactly. Before issue #339 the stopping iteration
    still took its update, so they were one update past it.

    A fit that runs the same iterations to ``max_iter`` does take its last
    update, as the reference's does, so there the recomputed value moves on:
    the equality is not one any parameters near the end would pass."""
    x = X[:, :_STOP_FRAMES]
    model = _model(backend, _STOP_ITERS, tmp_path, **_STOP)
    ll = _fit(model, backend, x, _STOP_ITERS)
    if backend == "numpy":
        assert model.stop_reason == "Converged: small likelihood increase"
        final = float(model.ll[-1])
    else:
        assert model.stop_reason == "min_dll"
        final = float(model.final_ll_)
        assert final == ll[-1]
        assert model.iteration == len(ll) - 1
    assert len(ll) < _STOP_ITERS
    assert _recomputed_ll(model, backend, x) == final

    ran = _model(backend, len(ll), tmp_path, **dict(_STOP, use_min_dll=False))
    ll_ran = _fit(ran, backend, x, len(ll))
    assert ll_ran == ll  # the same trajectory, without the stop
    assert _recomputed_ll(ran, backend, x) != ll_ran[-1]


# The best-iterate safeguard (PyTorch and MLX; NumPy has none): two models,
# lrate 0.5 on one 4096-sample block (test_component_rows.py's
# m2-keepbest-restores recipe), whose only decrease is on the last of six
# iterations, by 2.4e-2, so the fit ends that far below its peak and restores it.
_RESTORE_ITERS = 6


@pytest.mark.parametrize("backend", ["torch", "mlx"])
def test_a_keep_best_restore_returns_the_parameters_of_its_ll(backend, X, tmp_path):
    """The snapshot is taken right after the E-step, before any check or update
    moves the parameters, so a restored model's own log-likelihood is exactly
    the ``final_ll_`` it reports."""
    x = X[:, :4096]
    if backend == "torch":
        model = AMICATorchNG(
            n_channels=NW,
            n_models=2,
            n_mix=NMIX,
            seed=SEED,
            block_size=4096,
            device="cpu",
            dtype=torch.float64,
            keep_best=True,
            lrate=0.5,
        )
    else:
        model = _mlx_class()(
            n_channels=NW,
            n_models=2,
            n_mix=NMIX,
            seed=SEED,
            block_size=4096,
            keep_best=True,
            lrate=0.5,
        )
    model.fit(x, max_iter=_RESTORE_ITERS, verbose=False)
    ll = model.ll_history
    assert model.stop_reason == "max_iter"
    assert model.final_ll_ != ll[-1], "keep_best restored nothing"
    assert model.final_ll_ == max(ll)
    assert _recomputed_ll(model, backend, x) == model.final_ll_


# --- #339: the rho-rate ceiling is saved with the model -----------------------


@pytest.mark.parametrize("backend", ["torch", "mlx"])
def test_the_rho_rate_ceiling_survives_a_save(backend, X, tmp_path):
    """The rho rate is two values since issue #339, the working ``rholrate``
    and its ceiling ``rholrate_cap`` (the reference's ``rholrate0``). A save
    round-trips both exactly; a save written before the split has only
    ``rholrate``, which held the ceiling then, so it loads as the ceiling.
    ``maxdecs=1`` with ``newt_start=0`` ratchets the ceiling on the first
    decrease, so it no longer equals the pristine ``rholrate0``."""
    cfg = dict(_DECREASE, maxdecs=1, newt_start=0)
    model = _model(backend, _DECREASE_ITERS, tmp_path, **cfg)
    _fit(model, backend, X[:, :_DECREASE_FRAMES], _DECREASE_ITERS)
    assert model.rholrate_cap < model.rholrate0, "the ceiling never ratcheted"

    def load(state: Dict[str, Any]) -> Any:
        if backend == "torch":
            return AMICATorchNG.from_state_dict(state, device="cpu")
        return type(model).from_state_dict(state)

    state = model.state_dict()
    restored = load(state)
    assert restored.rholrate_cap == model.rholrate_cap
    assert restored.rholrate == model.rholrate

    del state["extra"]["rholrate_cap"]
    legacy = load(state)
    assert legacy.rholrate_cap == state["extra"]["rholrate"]
