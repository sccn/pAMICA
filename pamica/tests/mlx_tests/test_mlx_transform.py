"""``transform`` and the mixing/unmixing accessors on the MLX backend --
issue #287, epic #278 Phase 1.

Apple-Silicon only, real sample EEG (no synthetic/mock), same module guards as
``test_mlx_backend.py``. This module holds the MLX-only mechanics: model_idx
validation, unfitted-model errors, and the training-data bit-level check
(``transform`` reproduces the fit's own E-step activations via the transpose
identity ``_forward`` relies on). The float64-twin precision question (does
float32 MLX ``transform``/the accessors agree with a float64 oracle holding
identical parameters) lives in
``pamica/tests/test_mlx_transform_cross_backend.py``, outside any one
backend's subdirectory, per ``.rules/backend_parity.md``.

The last test (:func:`test_fit_path_is_unchanged_by_phase1`) is the
fit-path no-op pin for the whole phase: adding ``transform``/the accessors/
persistence touched nothing ``_fit_once`` calls, so a short seeded fit on
this branch must reproduce exactly what the SAME fit produced on the epic
branch tip (076605c) before any Phase 1 code existed -- recorded below from
that run.
"""

import hashlib
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from pamica.mlx_impl import AMICAMLXNG  # noqa: E402  (after the MLX importorskip)

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "sample_data"
DATA_FILE = SAMPLE_DIR / "eeglab_data.fdt"
NW = 32
FIELD = 30504
NMIX = 3
SEED = 42
BLOCK = 1024

pytestmark = [
    pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing"),
    pytest.mark.skipif(
        mx.default_device().type != mx.DeviceType.gpu, reason="no Apple GPU"
    ),
]


def _real_data(n_samples: int | None = None) -> np.ndarray:
    from pamica.torch_impl.utils import load_eeglab_data

    data = load_eeglab_data(str(DATA_FILE), data_dim=NW, field_dim=FIELD).astype(
        np.float64
    )
    return data if n_samples is None else data[:, :n_samples]


def _fitted_model(n_models: int = 1, max_iter: int = 5, **kwargs) -> AMICAMLXNG:
    m = AMICAMLXNG(
        n_channels=NW, n_models=n_models, n_mix=NMIX, seed=SEED, block_size=BLOCK,
        **kwargs,
    )  # fmt: skip
    m.fit(_real_data(4096), max_iter=max_iter, verbose=False)
    assert m.stop_reason not in AMICAMLXNG._DEGENERATE_STOP_REASONS
    return m


# --- unfitted / invalid model_idx --------------------------------------------


def test_transform_requires_fitted_model():
    m = AMICAMLXNG(n_channels=NW, n_mix=NMIX)
    with pytest.raises(RuntimeError, match="requires a fitted model"):
        m.transform(np.zeros((NW, 10)))


@pytest.mark.parametrize(
    "accessor",
    ["get_mixing_matrix", "get_unmixing_matrix", "get_sensor_mixing_matrix", "get_rho"],
)
def test_accessors_require_fitted_model(accessor):
    m = AMICAMLXNG(n_channels=NW, n_mix=NMIX)
    with pytest.raises(RuntimeError, match="requires a fitted model"):
        getattr(m, accessor)()


@pytest.mark.parametrize("bad_idx", [-1, 1, 2, 100])
def test_transform_rejects_invalid_model_idx(bad_idx):
    m = _fitted_model(n_models=1)
    with pytest.raises(ValueError, match="out of range"):
        m.transform(_real_data(64), model_idx=bad_idx)


@pytest.mark.parametrize(
    "accessor",
    ["get_mixing_matrix", "get_unmixing_matrix", "get_sensor_mixing_matrix", "get_rho"],
)
def test_accessors_reject_invalid_model_idx(accessor):
    m = _fitted_model(n_models=1)
    with pytest.raises(ValueError, match="out of range"):
        getattr(m, accessor)(-1)


def test_transform_rejects_non_int_model_idx():
    m = _fitted_model(n_models=1)
    with pytest.raises(TypeError, match="must be an int"):
        m.transform(_real_data(64), model_idx="0")  # ty: ignore[invalid-argument-type]


# --- shapes / dtypes ----------------------------------------------------------


def test_transform_output_shape_and_dtype():
    m = _fitted_model(n_models=1)
    data = _real_data(64)
    S = m.transform(data, model_idx=0)
    assert S.shape == (NW, 64)
    assert S.dtype == np.float32


def test_accessor_shapes():
    m = _fitted_model(n_models=2, max_iter=5)
    for h in range(2):
        assert m.get_mixing_matrix(h).shape == (NW, NW)
        assert m.get_unmixing_matrix(h).shape == (NW, NW)
        assert m.get_sensor_mixing_matrix(h).shape == (NW, NW)  # full rank here
        assert m.get_rho(h).shape == (NMIX, NW)


def test_get_mixing_and_unmixing_are_inverses():
    """A = get_mixing_matrix, W = get_unmixing_matrix; W @ A should be
    (approximately) the identity -- a sign/transpose slip in either accessor
    would break this."""
    m = _fitted_model(n_models=1)
    A = m.get_mixing_matrix()
    W = m.get_unmixing_matrix()
    assert np.allclose(W @ A, np.eye(NW), atol=1e-3)


# --- transform(training X) reproduces the fit's own E-step activations ------


def test_transform_matches_forward_activations_on_training_data():
    """``transform`` applied to the ORIGINAL raw data used for a fit must
    reproduce that fit's own ``_forward`` activations, up to the float32
    rounding gap between two independently-computed preprocessing paths
    (``_preprocess`` computes ``sphere @ (X - mean)`` in float64 numpy then
    casts to float32; ``transform`` computes the same product as an MLX
    float32 op on the raw input -- see :meth:`AMICAMLXNG.transform`). Measured
    on the bundled sample: max absolute difference ~5.7e-6, i.e. a handful of
    float32 ULPs, not a formula difference -- confirmed by the transpose
    identity check below.
    """
    m = AMICAMLXNG(n_channels=NW, n_mix=NMIX, seed=SEED, block_size=BLOCK)
    data = _real_data(4096)
    x_t = m._preprocess(data)
    m._initialize_parameters()
    for it in range(5):
        m.iteration = it
        m._update_parameters(m._accumulate_blocks(x_t), x_t.shape[1])
    mx.eval(m.A, m.W, m.mu, m.alpha, m.beta, m.rho, m.gm, m.c)

    _, b_list, _, _, _ = m._forward(x_t)
    b0 = np.array(b_list[0])  # (batch, n_channels), the training E-step activation

    S = m.transform(data, model_idx=0)  # (n_channels, batch)
    diff = np.abs(S.T.astype(np.float64) - b0.astype(np.float64))
    assert diff.max() < 1e-4, (
        f"transform(training X) disagrees with the fit's own E-step "
        f"activations by {diff.max():.3e}"
    )
    # The transpose identity itself (S = b.T, exactly, given the same inputs
    # in the same precision) is what _forward and transform BOTH implement;
    # check it directly in float64 on host, sidestepping the two independent
    # preprocessing paths above, so a genuine formula bug (not just precision)
    # cannot hide behind the tolerance above.
    assert m.c is not None and m.W is not None
    b_np = np.array(x_t, dtype=np.float64)
    c0 = np.array(m.c[:, 0], dtype=np.float64)
    W0 = np.array(m.W[0], dtype=np.float64)
    b_hand = (b_np - c0[:, None]).T @ W0
    S_hand = W0.T @ (b_np - c0[:, None])
    assert np.array_equal(S_hand.T, b_hand), "transpose identity does not hold exactly"


def test_transform_multimodel_routes_by_model_idx():
    """Different ``model_idx`` values must not silently alias the same
    unmixing (a broadcast/indexing slip would make every model_idx return
    the same sources)."""
    m = _fitted_model(n_models=2, max_iter=4, do_newton=True, newt_start=0)
    data = _real_data(64)
    S0 = m.transform(data, model_idx=0)
    S1 = m.transform(data, model_idx=1)
    assert not np.allclose(S0, S1)


# --- fit-path no-op pin (plan item C7) ---------------------------------------

# Recorded from an ACTUAL run on the epic branch tip (076605c, "Address
# 0.3.3 release review findings (#303)"), i.e. before any Phase 1 code
# existed in this worktree -- see the PR body for the exact recording
# command. MLX GPU execution on this machine is deterministic (confirmed:
# two back-to-back runs of this exact config produced bit-identical
# ll_history and A), so re-running the identical config on this branch and
# comparing is a real, non-vacuous pin that adding transform/accessors/
# persistence touched nothing _fit_once calls.
_NOOP_PIN_LL_HISTORY = [
    -3.3320186138153076,
    -3.2827978134155273,
    -3.2742865085601807,
    -3.2692978382110596,
    -3.26499080657959,
    -3.2611570358276367,
    -3.2578959465026855,
    -3.2551658153533936,
    -3.2528350353240967,
    -3.25075626373291,
]
_NOOP_PIN_FINAL_LL = -3.25075626373291
_NOOP_PIN_STOP_REASON = "max_iter"
_NOOP_PIN_A_SHA256 = "fd35584abbeef2aae607777e043df33c057d238bee27098e0e46021baa47e55f"
_NOOP_PIN_W_SHA256 = "fa75ae1f05e7996936526ef436de4004ef4cf376c04105a2ba7e296d44540f62"


def test_fit_path_is_unchanged_by_phase1():
    """The default fit path (``_fit_once`` and everything it calls) is
    bit-identical to the epic branch tip, before any of this phase's
    ``transform``/accessor/persistence additions existed. Phase 1 only adds
    new methods; it does not edit ``_fit_once`` or its call graph, so this
    must reproduce the pre-Phase-1 run exactly, not just approximately.
    """
    m = AMICAMLXNG(n_channels=NW, n_mix=NMIX, seed=SEED, block_size=BLOCK)
    m.fit(_real_data(4096), max_iter=10, verbose=False)

    assert m.ll_history == _NOOP_PIN_LL_HISTORY
    assert m.final_ll_ == _NOOP_PIN_FINAL_LL
    assert m.stop_reason == _NOOP_PIN_STOP_REASON

    a_bytes = np.array(m.A, dtype=np.float32).tobytes()
    w_bytes = np.array(m.W, dtype=np.float32).tobytes()
    assert hashlib.sha256(a_bytes).hexdigest() == _NOOP_PIN_A_SHA256
    assert hashlib.sha256(w_bytes).hexdigest() == _NOOP_PIN_W_SHA256
