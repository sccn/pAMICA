"""Cross-backend guard tests for degenerate-fit output and bad-shape input
(issue #306, epic #324 Phase 5).

Three gaps existed identically in ``AMICATorchNG``, ``AMICAMLXNG`` and the
legacy NumPy ``AMICA`` backend, so per ``.rules/backend_parity.md`` they are
fixed and tested together in one module rather than three:

1. Fitted-output accessors (``transform``, ``get_mixing_matrix``,
   ``get_unmixing_matrix``, ``get_sensor_mixing_matrix``, ``get_rho``,
   ``variance_order``, ``model_loglik``, ``model_probability``, ``mir``,
   ``pmi`` on torch/MLX; ``transform``, ``get_weights``,
   ``get_sensor_mixing_matrix`` on NumPy) used to return numeric-looking
   output from a fit the backend itself classified as degenerate, with no
   warning. They now raise ``RuntimeError``, mirroring the two-layer pattern
   ``state_dict()``/``write_amica_output()`` already used: a ``stop_reason``
   gate, then a defense-in-depth isfinite sweep over the fitted parameters.
2. The data-taking accessors among those (``transform``, ``model_loglik``,
   ``model_probability``, ``mir``, ``pmi`` on torch/MLX; ``transform`` on
   NumPy) did not validate ``X``/``data`` the way ``fit()`` does, so a
   wrong-shaped array failed with a raw matmul/broadcast error instead of the
   same named ``ValueError`` ``fit()`` gives for the identical mistake.
3. ``from_state_dict``'s ``cls(**config)`` surfaced a malformed (missing or
   unexpected key) config as a bare ``TypeError`` instead of a named
   ``ValueError`` naming the payload, like every neighboring validation step
   in that method.

Real bundled sample EEG only (32 channels x 30504 frames), sliced to a few
thousand samples and 2 iterations to keep fits fast -- these are guard/
validation tests, not convergence tests (``.rules/testing.md``). Degenerate
state is produced the established way (``pamica/tests/torch_tests/
test_ng_backend.py`` ~1078-1113, ``test_ng_exception_consistency.py``
~127-130): fit for real, then set ``stop_reason``/``converged`` directly or
poison one parameter entry with NaN, rather than trying to drive a real fit
into divergence.
"""

from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import pytest
import torch

from pamica.numpy_impl.core import AMICA as AMICA_NumPy
from pamica.torch_impl import AMICATorchNG
from pamica.torch_impl.utils import load_eeglab_data

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_data"
DATA_FILE = SAMPLE_DIR / "eeglab_data.fdt"
NW = 32
FIELD = 30504
NMIX = 3
SEED = 42
# Sliced for speed (module docstring): guard/validation tests, not
# convergence tests.
N_SAMPLES = 2048
PCAKEEP = 20

pytestmark = pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing")


def _mlx_core():
    """The MLX backend module, or skip this test (never the whole module)."""
    return pytest.importorskip(
        "pamica.mlx_impl.core", reason="MLX not installed (Apple Silicon only)"
    )


@pytest.fixture(scope="module")
def real_data() -> np.ndarray:
    return load_eeglab_data(str(DATA_FILE), data_dim=NW, field_dim=FIELD).astype(
        np.float64
    )[:, :N_SAMPLES]


def _fit_torch(data: np.ndarray, **kwargs) -> AMICATorchNG:
    m = AMICATorchNG(
        n_channels=NW,
        n_mix=NMIX,
        seed=SEED,
        device="cpu",
        dtype=torch.float64,
        block_size=256,
        **kwargs,
    )
    m.fit(data, max_iter=2, verbose=False)
    return m


def _fit_mlx(data: np.ndarray, **kwargs):
    AMICAMLXNG = _mlx_core().AMICAMLXNG
    m = AMICAMLXNG(n_channels=NW, n_mix=NMIX, seed=SEED, block_size=256, **kwargs)
    m.fit(data.astype(np.float32), max_iter=2, verbose=False)
    return m


def _fit_numpy(data: np.ndarray, tmp_path: Path, **kwargs) -> AMICA_NumPy:
    m = AMICA_NumPy(
        num_models=1,
        num_mix=NMIX,
        max_iter=2,
        seed=SEED,
        block_size=256,
        do_newton=False,
        use_tqdm=False,
        outdir=str(tmp_path / "numpy_out"),
        **kwargs,
    )
    m.fit(data)
    return m


# (name, fn(model, X) -> result, is_data_accessor). ``is_data_accessor`` marks
# the methods that take an ``X``/``data`` array and so are covered by item 2's
# shape validation; the rest only take ``model_idx``.
Accessor = Tuple[str, Callable, bool]

_TORCH_MLX_ACCESSORS: List[Accessor] = [
    ("transform", lambda m, X: m.transform(X), True),
    ("get_mixing_matrix", lambda m, X: m.get_mixing_matrix(), False),
    ("get_unmixing_matrix", lambda m, X: m.get_unmixing_matrix(), False),
    ("get_sensor_mixing_matrix", lambda m, X: m.get_sensor_mixing_matrix(), False),
    ("get_rho", lambda m, X: m.get_rho(), False),
    ("variance_order", lambda m, X: m.variance_order(), False),
    ("model_loglik", lambda m, X: m.model_loglik(X), True),
    ("model_probability", lambda m, X: m.model_probability(X), True),
    ("mir", lambda m, X: m.mir(X), True),
    ("pmi", lambda m, X: m.pmi(X), True),
]

_NUMPY_ACCESSORS: List[Accessor] = [
    ("transform", lambda m, X: m.transform(X), True),
    ("get_weights", lambda m, X: m.get_weights(), False),
    ("get_sensor_mixing_matrix", lambda m, X: m.get_sensor_mixing_matrix(), False),
]

_BACKENDS = ["torch", "mlx", "numpy"]


def _build(backend: str, real_data: np.ndarray, tmp_path: Path, **kwargs):
    """Fit a fresh small model on ``real_data`` and return
    ``(model, accessors)`` for the requested backend."""
    if backend == "torch":
        return _fit_torch(real_data, **kwargs), _TORCH_MLX_ACCESSORS
    if backend == "mlx":
        return _fit_mlx(real_data, **kwargs), _TORCH_MLX_ACCESSORS
    if backend == "numpy":
        return _fit_numpy(real_data, tmp_path, **kwargs), _NUMPY_ACCESSORS
    raise ValueError(backend)  # pragma: no cover - parametrize typo guard


def _poison_a00(backend: str, m) -> None:
    """NaN-poison a single entry of the fitted mixing matrix ``A`` in place,
    the sanctioned direct-poisoning pattern (``.rules/testing.md``)."""
    if backend == "mlx":
        mx = _mlx_core().mx
        m.A[0, 0] = mx.array(float("nan"))
    else:
        m.A[0, 0] = float("nan")


# ---------------------------------------------------------------------
# Item 1: degenerate-fit guard on every fitted-output accessor
# ---------------------------------------------------------------------


@pytest.mark.parametrize("backend", _BACKENDS)
def test_guarded_accessors_work_on_a_healthy_fit(backend, real_data, tmp_path):
    m, accessors = _build(backend, real_data, tmp_path)
    for _name, fn, _is_data in accessors:
        assert fn(m, real_data) is not None


@pytest.mark.parametrize("backend", _BACKENDS)
def test_guarded_accessors_refuse_degenerate_stop_reason(backend, real_data, tmp_path):
    m, accessors = _build(backend, real_data, tmp_path)

    # A real divergence sets this marker (and leaves NaNs in the params);
    # assert the guard fires on the marker deterministically, without
    # needing to induce an actual blow-up (established pattern, see module
    # docstring).
    if backend in ("torch", "mlx"):
        m.stop_reason = "nan_ll"
    else:
        # NumPy's single ``converged`` flag is its degenerate marker; a real
        # divergence sets both it and a descriptive stop_reason together.
        m.converged = False
        m.stop_reason = "Non-finite likelihood (NaN/-inf) encountered"

    for name, fn, _is_data in accessors:
        with pytest.raises(RuntimeError, match="degenerate"):
            fn(m, real_data)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_guarded_accessors_refuse_nonfinite_param(backend, real_data, tmp_path):
    """Defense-in-depth: even with a non-degenerate stop_reason/converged
    state, a non-finite parameter blocks every guarded accessor."""
    m, accessors = _build(backend, real_data, tmp_path)

    if backend in ("torch", "mlx"):
        m.stop_reason = "max_iter"  # not a degenerate marker
    else:
        m.converged = True
        m.stop_reason = "max_iter reached"
    assert m.A is not None
    _poison_a00(backend, m)

    for name, fn, _is_data in accessors:
        with pytest.raises(RuntimeError, match="non-finite"):
            fn(m, real_data)


# ---------------------------------------------------------------------
# Item 2: input-shape validation on the data-taking accessors
# ---------------------------------------------------------------------


@pytest.mark.parametrize("backend", _BACKENDS)
def test_data_accessors_reject_1d_and_wrong_channel_count(backend, real_data, tmp_path):
    m, accessors = _build(backend, real_data, tmp_path)
    data_accessors = [(name, fn) for name, fn, is_data in accessors if is_data]
    assert data_accessors, "expected at least one data-taking accessor"

    bad_1d = real_data[0]
    bad_channels = real_data[:20]  # fitted on 32 channels

    for name, fn in data_accessors:
        with pytest.raises(ValueError):
            fn(m, bad_1d)
        with pytest.raises(ValueError):
            fn(m, bad_channels)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_transform_channel_count_follows_n_channels_in_after_pca(
    backend, real_data, tmp_path
):
    """After a ``pcakeep=20`` fit on 32 channels, the model's fitted rank is
    20 but its input channel count stays 32: a 32-channel array must be
    accepted and a 20-channel one rejected (the ``n_channels_in`` rule)."""
    m, _accessors = _build(backend, real_data, tmp_path, pcakeep=PCAKEEP)

    if backend == "torch":
        assert m.n_channels_in == NW
        assert m.n_channels == PCAKEEP
        assert m.transform(real_data).shape[0] == PCAKEEP
        with pytest.raises(ValueError, match="channels"):
            m.transform(real_data[:PCAKEEP])
    elif backend == "mlx":
        assert m.n_channels_in == NW
        assert m.n_channels == PCAKEEP
        data32 = real_data.astype(np.float32)
        assert m.transform(data32).shape[0] == PCAKEEP
        with pytest.raises(ValueError, match="channels"):
            m.transform(data32[:PCAKEEP])
    else:
        assert m.data_dim_in == NW
        assert m.data_dim == PCAKEEP
        assert m.transform(real_data).shape[0] == PCAKEEP
        with pytest.raises(ValueError, match="channels"):
            m.transform(real_data[:PCAKEEP])


# ---------------------------------------------------------------------
# Item 3: from_state_dict surfaces a malformed config as a named ValueError
# ---------------------------------------------------------------------


def test_torch_from_state_dict_unexpected_key_raises_value_error(real_data):
    m = _fit_torch(real_data)
    state = m.state_dict()
    bad_state = {**state, "config": {**state["config"], "bogus_key_xyz": 123}}

    with pytest.raises(ValueError, match="malformed AMICATorchNG state") as excinfo:
        AMICATorchNG.from_state_dict(bad_state)
    assert isinstance(excinfo.value.__cause__, TypeError)


def test_torch_from_state_dict_missing_key_raises_value_error(real_data):
    m = _fit_torch(real_data)
    state = m.state_dict()
    config = dict(state["config"])
    del config["n_channels"]
    bad_state = {**state, "config": config}

    with pytest.raises(ValueError, match="malformed AMICATorchNG state") as excinfo:
        AMICATorchNG.from_state_dict(bad_state)
    assert isinstance(excinfo.value.__cause__, TypeError)


def test_mlx_from_state_dict_unexpected_key_raises_value_error(real_data):
    AMICAMLXNG = _mlx_core().AMICAMLXNG
    m = _fit_mlx(real_data)
    state = m.state_dict()
    bad_state = {**state, "config": {**state["config"], "bogus_key_xyz": 123}}

    with pytest.raises(ValueError, match="malformed AMICAMLXNG state") as excinfo:
        AMICAMLXNG.from_state_dict(bad_state)
    assert isinstance(excinfo.value.__cause__, TypeError)


def test_mlx_from_state_dict_missing_key_raises_value_error(real_data):
    AMICAMLXNG = _mlx_core().AMICAMLXNG
    m = _fit_mlx(real_data)
    state = m.state_dict()
    config = dict(state["config"])
    del config["n_channels"]
    bad_state = {**state, "config": config}

    with pytest.raises(ValueError, match="malformed AMICAMLXNG state") as excinfo:
        AMICAMLXNG.from_state_dict(bad_state)
    assert isinstance(excinfo.value.__cause__, TypeError)
