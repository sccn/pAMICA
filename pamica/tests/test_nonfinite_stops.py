"""Every backend stops the same way on a non-finite value (issue #339 review).

Three places a fit can meet a non-finite value, each with its own degenerate
``stop_reason``, the same in PyTorch and MLX and named in the NumPy backend's
prose vocabulary:

* the E-step's log-likelihood (``nan_ll``): the fit stops before recording it,
  so every backend's history is the finite trajectory of the parameters the fit
  visited, with the same length for the same event;
* the update direction or its norm ``ndtmpsum`` (``nan_direction``): a NaN
  passes both ``<= min_nd`` checks (the comparison is False), so without a
  check it would be applied; the fit stops before the update instead;
* the parameters right after an update (``nan_params``): without a check, a
  corruption on the last iteration would end as ``max_iter`` with non-finite
  parameters.

A non-finite value is not something the bundled recording produces on demand at
a chosen iteration, so each test uses the sanctioned error-injection pattern
(``.rules/testing.md``, as ``test_schedule_gates.py``'s ``_NaNOnIteration``): a
subclass that runs the real E-step, direction and update on real data and then
corrupts one value on one iteration. Everything else, the stops included, is
the real code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List

import numpy as np
import pytest
import torch

from pamica.amica import AMICA
from pamica.numpy_impl.core import AMICA as AMICA_NumPy
from pamica.torch_impl.core import AMICATorchNG
from pamica.torch_impl.utils import load_eeglab_data

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_data"
DATA_FILE = SAMPLE_DIR / "eeglab_data.fdt"
NW = 32
FIELD = 30504
NMIX = 3
SEED = 42
FRAMES = 4096
BACKENDS = ["torch", "numpy", "mlx"]

# The NumPy backend's prose names for the PyTorch/MLX stop reasons.
_NUMPY_REASON = {
    "nan_ll": AMICA_NumPy._NONFINITE_LL_REASON,
    "nan_direction": AMICA_NumPy._NONFINITE_DIRECTION_REASON,
    "nan_params": AMICA_NumPy._NONFINITE_PARAMS_REASON,
}

pytestmark = pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing")


@pytest.fixture(scope="module")
def X() -> np.ndarray:
    return load_eeglab_data(str(DATA_FILE), data_dim=NW, field_dim=FIELD).astype(
        np.float64
    )[:, :FRAMES]


def _nan_at_00(value: Any) -> np.ndarray:
    assert value is not None, "injected before the parameter exists"
    out = np.array(value, dtype=np.float64, copy=True)
    out[0, 0] = np.nan
    return out


class _TorchInject(AMICATorchNG):
    """``AMICATorchNG`` that corrupts one value on iteration ``at``."""

    kind = ""
    at = -1

    def _accumulate_blocks(self, X, stash_llt=False):
        acc = super()._accumulate_blocks(X, stash_llt=stash_llt)
        if self.kind == "ll" and self.iteration == self.at:
            acc["ll"] = acc["ll"] * float("nan")
        return acc

    def _update_direction(self, acc):
        step = super()._update_direction(acc)
        if self.iteration == self.at:
            if self.kind == "dAk":
                dAk = step.dAk.clone()
                dAk[0, 0] = float("nan")
                step = step._replace(dAk=dAk)
            elif self.kind == "ndtmpsum":
                self._ndtmpsum = float("nan")
        return step

    def _update_parameters(self, acc, n_samples, step=None):
        super()._update_parameters(acc, n_samples, step)
        if self.kind == "params" and self.iteration == self.at:
            assert self.mu is not None
            mu = self.mu.clone()
            mu[0, 0] = float("nan")
            self.mu = mu


class _NumPyInject(AMICA_NumPy):
    """The NumPy backend, corrupting one value on iteration ``at``."""

    kind = ""
    at = -1

    def _get_updates_and_likelihood(self):
        updates = super()._get_updates_and_likelihood()
        if self.kind == "ll" and self.iter == self.at:
            updates["ll"] = float("nan")
        return updates

    def _update_direction(self, updates):
        step = super()._update_direction(updates)
        if self.iter == self.at:
            if self.kind == "dAk":
                step = step._replace(dAk=_nan_at_00(step.dAk))
            elif self.kind == "ndtmpsum":
                step = step._replace(nd=float("nan"))
        return step

    def _update_parameters(self, updates, step=None):
        super()._update_parameters(updates, step)
        if self.kind == "params" and self.iter == self.at:
            self.mu = _nan_at_00(self.mu)
        elif self.kind == "params_A" and self.iter == self.at:
            # W is derived from A inside the update, so rebuild it the way the
            # update does: a corrupted A reaches W, as it would for real.
            self.A = _nan_at_00(self.A)
            self._update_unmixing_matrices()


def _mlx_inject_class() -> Any:
    mlx_core = pytest.importorskip(
        "pamica.mlx_impl.core", reason="MLX not installed (Apple Silicon only)"
    )
    mx = mlx_core.mx
    if mx.default_device().type != mx.DeviceType.gpu:
        pytest.skip("no Apple GPU")

    class _MLXInject(mlx_core.AMICAMLXNG):
        """``AMICAMLXNG`` that corrupts one value on iteration ``at``."""

        kind = ""
        at = -1

        def _accumulate_blocks(self, X, stash_llt=False):
            acc = super()._accumulate_blocks(X, stash_llt=stash_llt)
            if self.kind == "ll" and self.iteration == self.at:
                acc["ll"] = acc["ll"] * float("nan")
            return acc

        def _update_direction(self, acc):
            step = super()._update_direction(acc)
            if self.iteration == self.at:
                if self.kind == "dAk":
                    step = step._replace(
                        dAk=mx.array(_nan_at_00(np.array(step.dAk)), dtype=mx.float32)
                    )
                elif self.kind == "ndtmpsum":
                    self._nd_arr = mx.array(float("nan"), dtype=mx.float32)
            return step

        def _update_parameters(self, acc, n_samples, step=None):
            super()._update_parameters(acc, n_samples, step)
            if self.kind == "params" and self.iteration == self.at:
                self.mu = mx.array(_nan_at_00(np.array(self.mu)), dtype=mx.float32)

    return _MLXInject


def _fit(backend: str, x: np.ndarray, kind: str, at: int, max_iter: int, tmp_path):
    """A one-model fit that corrupts ``kind`` on 0-based iteration ``at``."""
    if backend == "numpy":
        numpy_model = _NumPyInject(
            num_models=1,
            num_mix=NMIX,
            seed=SEED,
            max_iter=max_iter,
            use_tqdm=False,
            do_opt_block=False,
            writestep=10**7,
            # The one backend with a restart-on-NaN recovery: off here, so the
            # stops compare like for like (the recovery has its own test below).
            restartiter=0,
            outdir=str(tmp_path / "np"),
        )
        numpy_model.kind = kind
        numpy_model.at = at
        return numpy_model.fit(x)
    if backend == "torch":
        model: Any = _TorchInject(
            n_channels=NW, n_mix=NMIX, seed=SEED, device="cpu", dtype=torch.float64
        )
    else:
        model = _mlx_inject_class()(n_channels=NW, n_mix=NMIX, seed=SEED)
    model.kind = kind
    model.at = at
    model.fit(x, max_iter=max_iter, verbose=False)
    return model


def _history(model: Any, backend: str) -> List[float]:
    return [float(v) for v in (model.ll if backend == "numpy" else model.ll_history)]


def _assert_degenerate(model: Any, backend: str, reason: str) -> None:
    """The stop is ``reason`` and every output path refuses the model."""
    if backend == "numpy":
        # The parameter reason carries the offenders' names after a colon.
        assert model.stop_reason is not None
        assert model.stop_reason.split(": ")[0] == _NUMPY_REASON[reason]
        assert type(model)._is_degenerate_stop(model.stop_reason)
        assert model.converged is False
        with pytest.raises(RuntimeError, match="degenerate"):
            model.get_weights()
        return
    assert model.stop_reason == reason
    assert reason in type(model)._DEGENERATE_STOP_REASONS
    assert np.isnan(model.final_ll_)
    with pytest.raises(RuntimeError, match="degenerate"):
        model.state_dict()
    # The AMICA wrapper classifies the fitted backend on the same path its own
    # fit() and load() take.
    wrapper = AMICA(n_models=1, n_mix=NMIX, backend=backend)
    wrapper.model_ = model
    wrapper._mirror_backend()
    assert wrapper.converged_ is False and wrapper.is_fitted_ is False
    with pytest.raises(RuntimeError, match="degenerate"):
        wrapper.get_unmixing_matrix()


# --- 1. a non-finite likelihood is never recorded -----------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_nonfinite_likelihood_leaves_the_same_finite_history(backend, X, tmp_path):
    """A NaN likelihood on iteration 3 stops every backend there, with the
    three likelihoods before it and nothing else in the history. Before the
    issue #339 review the NumPy backend recorded the NaN as a fourth entry."""
    at = 3
    model = _fit(backend, X, "ll", at, max_iter=6, tmp_path=tmp_path)
    history = _history(model, backend)
    assert len(history) == at
    assert np.all(np.isfinite(history))
    _assert_degenerate(model, backend, "nan_ll")


# --- 2. a non-finite direction is never applied -------------------------------


@pytest.mark.parametrize("which", ["dAk", "ndtmpsum"])
@pytest.mark.parametrize("backend", BACKENDS)
def test_a_nonfinite_direction_stops_before_the_update(backend, which, X, tmp_path):
    """A NaN in the step, or in its norm, on the last iteration stops the fit
    before the update: the finite likelihood of that iteration is recorded, the
    returned parameters are the finite ones it describes, and the stop is the
    degenerate ``nan_direction``. Before, a NaN norm passed both ``<= min_nd``
    checks and the step was applied, so a NaN step left non-finite parameters
    behind a ``max_iter`` stop."""
    max_iter = 4
    at = max_iter - 1
    model = _fit(backend, X, which, at, max_iter=max_iter, tmp_path=tmp_path)
    history = _history(model, backend)
    assert len(history) == at + 1
    assert np.all(np.isfinite(history))
    assert model._nonfinite_params() == []
    _assert_degenerate(model, backend, "nan_direction")


# --- 3. non-finite parameters after an update stop the fit --------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_nonfinite_parameters_after_the_last_update_stop_the_fit(backend, X, tmp_path):
    """A NaN in ``mu`` right after the last iteration's update ends the fit as
    the degenerate ``nan_params``, not ``max_iter``, on every backend, with the
    finite likelihood of that iteration recorded. Before, PyTorch returned
    ``max_iter`` holding the NaN (MLX already stopped; NumPy caught it only at
    exit, under a different reason)."""
    max_iter = 4
    at = max_iter - 1
    model = _fit(backend, X, "params", at, max_iter=max_iter, tmp_path=tmp_path)
    history = _history(model, backend)
    assert len(history) == at + 1
    assert np.all(np.isfinite(history))
    assert "mu" in model._nonfinite_params()
    _assert_degenerate(model, backend, "nan_params")


@pytest.mark.parametrize(("kind", "restarted"), [("params_A", True), ("params", False)])
def test_numpy_restart_window_takes_only_what_a_restart_repairs(
    kind, restarted, X, tmp_path
):
    """The NumPy backend's restart-on-NaN keeps working where it can: a
    restart redraws A (and so W) and keeps the mixture parameters, as the
    reference's does (amica15.f90:1026-1046). So a non-finite A after an update
    whose next iteration is inside the restart window goes on to that
    iteration, whose non-finite likelihood restarts the fit, which then
    finishes normally; a non-finite ``mu`` there, which no restart would
    repair, stops as ``nan_params`` at once."""
    model = _NumPyInject(
        num_models=1,
        num_mix=NMIX,
        seed=SEED,
        max_iter=6,
        use_tqdm=False,
        do_opt_block=False,
        writestep=10**7,
        restartiter=10,
        maxrestarts=3,
        outdir=str(tmp_path / "np"),
    )
    model.kind = kind
    model.at = 1
    model.fit(X)
    assert np.all(np.isfinite(model.ll))
    if restarted:
        assert model.numrestarts == 1
        assert model.converged is True, model.stop_reason
    else:
        assert model.numrestarts == 0
        assert model.stop_reason == f"{_NUMPY_REASON['nan_params']}: mu"
        assert len(model.ll) == 2
