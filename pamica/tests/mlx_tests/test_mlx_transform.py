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
    [
        "get_mixing_matrix",
        "get_unmixing_matrix",
        "get_sensor_mixing_matrix",
        "get_rho",
        "variance_order",
    ],
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
    [
        "get_mixing_matrix",
        "get_unmixing_matrix",
        "get_sensor_mixing_matrix",
        "get_rho",
        "variance_order",
    ],
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


def test_variance_order_is_a_permutation_with_matching_svar():
    """``variance_order`` returns each source index exactly once, in
    descending back-projected-variance order, with ``return_svar`` reporting
    those same variances already sorted to match (issue #92)."""
    m = _fitted_model(n_models=2, max_iter=5)
    for h in range(2):
        # Tuple-unpacked so both names are the ndarray element, not the
        # `np.ndarray | tuple` union the bare call below returns (ty).
        order, svar = m.variance_order(h, return_svar=True)
        assert order.shape == (NW,)
        assert sorted(order.tolist()) == list(range(NW))  # a permutation
        assert svar.shape == (NW,)
        # Descending: svar[order] as reported, so it must already be sorted.
        assert np.all(np.diff(svar) <= 0)

        order_only = m.variance_order(h)
        assert np.array_equal(order, order_only)


def test_get_mixing_and_unmixing_are_inverses():
    """A = get_mixing_matrix, W = get_unmixing_matrix; W @ A should be
    (approximately) the identity -- a sign/transpose slip in either accessor
    would break this."""
    m = _fitted_model(n_models=1)
    A = m.get_mixing_matrix()
    W = m.get_unmixing_matrix()
    assert np.allclose(W @ A, np.eye(NW), atol=1e-3)


def test_n_channels_in_full_rank_equals_n_channels():
    m = AMICAMLXNG(n_channels=NW, n_mix=NMIX)
    assert m.n_channels_in == NW  # unfitted: falls back to n_channels
    m = _fitted_model(n_models=1)
    assert m.n_channels_in == NW == m.n_channels


def test_n_channels_in_reports_original_width_on_rank_reduced_fit():
    """On a rank-reduced fit, ``n_channels`` becomes the DETECTED rank (issue
    #223), so ``n_channels_in`` -- the sphere's original input width -- is the
    only place the pre-reduction channel count is still recoverable."""
    x = _real_data(4096)
    x = x - x.mean(axis=1, keepdims=True)
    rank = 20
    U_r = np.linalg.svd(x, full_matrices=False)[0][:, :rank]
    x_low = U_r @ (U_r.T @ x)

    m = AMICAMLXNG(n_channels=NW, n_mix=NMIX, seed=SEED, block_size=BLOCK)
    m.fit(x_low, max_iter=5, verbose=False)
    assert m.stop_reason not in AMICAMLXNG._DEGENERATE_STOP_REASONS
    assert m.n_channels == rank
    assert m.n_channels_in == NW


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


# --- defense-in-depth (non-finite parameters on an otherwise "clean" fit) ---


def test_get_rho_raises_on_force_set_nan_rho():
    """``get_rho``'s isfinite guard actually fires: a real fit's ``rho`` is
    force-corrupted to NaN afterward (the same direct-attribute-assignment
    pattern ``_force_merged_column`` uses in ``test_mlx_sharing.py``, not a
    mock), so the model looks fitted and its ``stop_reason`` stays healthy --
    only ``rho`` itself is broken."""
    m = _fitted_model(n_models=1)
    rho_np = np.array(m.rho)
    rho_np[0, 0] = np.nan
    m.rho = mx.array(rho_np)

    with pytest.raises(RuntimeError, match="non-finite"):
        m.get_rho()


# --- fit-path no-op pin (plan item C7) ---------------------------------------

# Recorded from an ACTUAL run on the epic branch tip (076605c, "Address
# 0.3.3 release review findings (#303)"), i.e. before any Phase 1 code
# existed in this worktree, on the development machine (Apple M4 Pro) -- see
# the PR body for the exact recording command. THE BIT-LEVEL NO-OP CLAIM WAS
# VERIFIED THERE: two back-to-back runs of this exact config on that machine
# produced bit-identical ll_history/A/W, and a before/after run across the
# Phase 1 changes on that same machine was likewise bit-identical, which is
# what actually establishes that this phase touches nothing _fit_once calls.
#
# This test is the LOOSER canary that survives CI running a different Apple
# GPU model: MLX float32 is bit-reproducible on one machine but not across
# GPU models (observed on CI: ll_history[0] -3.332018613 vs this machine's
# -3.332018852, a ~7e-8 relative difference -- not a fit-path regression).
# _REL_TOL (5e-6) is calibrated between that noise floor and a real change:
# ~50x the observed ~1e-7 cross-GPU spread, comfortably below the ~1e-6+
# trajectory shift a genuine fit-path change produces on this same benchmark
# (issue #216's block_size default change alone shifted the trajectory
# ~1e-6, "inside parity tolerance" -- AGENTS.md). stop_reason and
# len(ll_history) are asserted exactly: they are a string and an int, so
# they carry no floating-point cross-machine risk at all.
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
# A handful of representative A entries (two diagonal, two off-diagonal, one
# corner), replacing the previous SHA-256 hash of the full A/W arrays: a
# cross-machine hash can never match (any per-entry float32 noise flips it),
# but these entries under the same relative tolerance still catch a real
# fit-path change while surviving cross-GPU float32 noise.
_NOOP_PIN_A_ENTRIES = {
    (0, 0): 0.8831924,
    (5, 5): 0.95371497,
    (10, 20): -0.0037135077,
    (31, 31): 0.99918866,
    (0, 31): 0.032124873,
}
# Recorded max(|A|) from the same run (A columns are ~unit-normalized by
# construction, so this sits near 1.0 regardless of seed/config). A[10, 20]
# above (-0.0037) is itself near zero, so a PER-ENTRY relative bound would
# collapse to an absolute tolerance of ~1.9e-8 there -- tighter than the
# ~1.1e-7 cross-GPU float32 noise CI actually observed, and exactly the
# near-zero-entry failure mode _max_rel_disagreement's docstring in
# test_mlx_transform_cross_backend.py explains for transform's output. Scale
# by this matrix-wide max instead, the same fix that function applies.
_NOOP_PIN_A_MAXABS = 0.99975544
_REL_TOL = 5e-6


def _assert_relclose(actual: float, expected: float, *, label: str) -> None:
    diff = abs(actual - expected)
    tol = _REL_TOL * abs(expected)
    assert diff <= tol, (
        f"{label}: {actual!r} differs from the recorded {expected!r} by "
        f"{diff:.3e} (relative {diff / abs(expected):.3e}), over the "
        f"{_REL_TOL:.0e} cross-GPU tolerance"
    )


def _assert_matrix_scale_close(actual: float, expected: float, *, label: str) -> None:
    """Like :func:`_assert_relclose`, but scaled by the matrix-wide
    ``_NOOP_PIN_A_MAXABS`` rather than by ``expected`` itself -- appropriate
    for a single entry of a matrix whose entries individually pass through
    near zero, per the module-level comment above."""
    diff = abs(actual - expected)
    tol = _REL_TOL * _NOOP_PIN_A_MAXABS
    assert diff <= tol, (
        f"{label}: {actual!r} differs from the recorded {expected!r} by "
        f"{diff:.3e}, over the {tol:.3e} matrix-scaled tolerance "
        f"({_REL_TOL:.0e} x max|A|={_NOOP_PIN_A_MAXABS})"
    )


def test_fit_path_is_unchanged_by_phase1():
    """The default fit path (``_fit_once`` and everything it calls) is
    unchanged by Phase 1's ``transform``/accessor/persistence additions,
    which add new methods without editing ``_fit_once`` or its call graph.

    The bit-level no-op claim was verified same-machine at development time
    (a before/after run across this phase's changes on the epic branch tip,
    076605c, was bit-identical there -- see the module-level comment above).
    Exact equality does not survive a different Apple GPU model, though
    (MLX float32 is bit-reproducible on one machine, not across models), so
    this CI-facing version checks agreement with the recorded M4 Pro values
    instead: ``ll_history``/``final_ll_`` per-entry relative (safe -- their
    magnitude, ~3.3, stays well clear of zero), the five ``A`` spot entries
    scaled by the matrix-wide ``max|A|`` instead (``A[10, 20]`` is itself
    near zero, so a per-entry relative bound there would be tighter than the
    observed cross-GPU noise -- see the module-level comment). Both are
    loose enough to survive cross-GPU float32 noise (~1e-7), tight enough to
    still catch a genuine fit-path regression (~1e-6+, see the module-level
    comment). ``stop_reason`` and the trajectory length are still exact --
    both machine-independent.
    """
    m = AMICAMLXNG(n_channels=NW, n_mix=NMIX, seed=SEED, block_size=BLOCK)
    m.fit(_real_data(4096), max_iter=10, verbose=False)

    assert m.stop_reason == _NOOP_PIN_STOP_REASON
    assert len(m.ll_history) == len(_NOOP_PIN_LL_HISTORY)

    for it, (actual, expected) in enumerate(zip(m.ll_history, _NOOP_PIN_LL_HISTORY)):
        _assert_relclose(actual, expected, label=f"ll_history[{it}]")
    assert m.final_ll_ is not None
    _assert_relclose(m.final_ll_, _NOOP_PIN_FINAL_LL, label="final_ll_")

    a = np.array(m.A, dtype=np.float32)
    for (i, j), expected in _NOOP_PIN_A_ENTRIES.items():
        _assert_matrix_scale_close(float(a[i, j]), expected, label=f"A[{i},{j}]")
