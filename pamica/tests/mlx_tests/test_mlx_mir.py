"""MIR/PMI diagnostics on the MLX backend -- issue #137, epic #278 Phase
3/#289 (port of ``AMICATorchNG.mir``/``pmi``, torch_impl/core.py:2929-3036,
and the ``fit(mir_step=...)`` waypoint machinery, torch_impl/core.py's
issue #161 tests in ``test_ng_convergence.py``/``test_amica_ng_wrapper.py``).

MLX-only mechanics: the composition pin (against ``pamica.metrics.mir``
directly, order-independent of any other backend), the ``mir_step``
schedule, the failed-waypoint NaN-and-continue guard, the #300
fitted-geometry PCA guard, and ``mir_history_``'s keep_best/state_dict
exclusions. The numeric agreement pin against ``AMICATorchNG.mir()`` on
identical parameters lives in
``pamica/tests/test_mlx_reject_cross_backend.py`` alongside the do_reject
agreement test -- the one new cross-backend file this phase adds.

Apple-Silicon only, real sample EEG (no synthetic/mock); the rank-reduction
test below projects real EEG onto a lower-rank subspace via SVD (the same
technique ``test_amica_ng_wrapper.py``'s
``test_mir_raises_under_auto_detected_rank_reduction`` uses), which is a
real-data manipulation, not fabricated data.
"""

import logging
import math
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from pamica.mlx_impl import AMICAMLXNG  # noqa: E402  (after the MLX importorskip)
from pamica.metrics import mir as mir_metric  # noqa: E402
from pamica.metrics import pairwise_mi  # noqa: E402

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "sample_data"
DATA_FILE = SAMPLE_DIR / "eeglab_data.fdt"
NW = 32
FIELD = 30504
NMIX = 3
BLOCK = 1024
SEED = 42

pytestmark = [
    pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing"),
    pytest.mark.skipif(
        mx.default_device().type != mx.DeviceType.gpu, reason="no Apple GPU"
    ),
]


@pytest.fixture(scope="module")
def real_data() -> np.ndarray:
    from pamica.torch_impl.utils import load_eeglab_data

    return load_eeglab_data(str(DATA_FILE), data_dim=NW, field_dim=FIELD).astype(
        np.float64
    )[:, :4096]


def _fit(X, n_models=1, max_iter=5, mir_step=0, **kwargs) -> AMICAMLXNG:
    kwargs.setdefault("seed", SEED)
    m = AMICAMLXNG(
        n_channels=NW,
        n_models=n_models,
        n_mix=NMIX,
        block_size=BLOCK,
        **kwargs,
    )
    m.fit(X, max_iter=max_iter, verbose=False, mir_step=mir_step)
    return m


# --- mir()/pmi() composition -----------------------------------------------
def test_mir_composes_unmixing_the_documented_way(real_data):
    """model.mir(X) must match metrics.mir(get_unmixing_matrix(0) @ sphere, X)
    -- the composition issue #137 pins (mirrors
    test_amica_ng_wrapper.py::test_mir_composes_unmixing_the_documented_way)."""
    m = _fit(real_data)
    sphere = np.array(m.sphere)
    unmixing = m.get_unmixing_matrix(0) @ sphere
    expected_mir, expected_var = mir_metric(unmixing, real_data)

    actual_mir, actual_var = m.mir(real_data)
    np.testing.assert_allclose(actual_mir, expected_mir, rtol=1e-10)
    np.testing.assert_allclose(actual_var, expected_var, rtol=1e-10)


def test_pmi_matches_pairwise_mi_on_transform(real_data):
    m = _fit(real_data)
    expected = pairwise_mi(m.transform(real_data))
    actual = m.pmi(real_data)
    np.testing.assert_allclose(actual, expected, rtol=1e-10)


def test_mir_and_pmi_honour_model_idx(real_data):
    m = _fit(real_data, n_models=2, seed=7, max_iter=3)
    mir0, _ = m.mir(real_data, model_idx=0)
    mir1, _ = m.mir(real_data, model_idx=1)
    assert mir0 != mir1, "model_idx=1 returned model 0's MIR"

    pmi0 = m.pmi(real_data, model_idx=0)
    pmi1 = m.pmi(real_data, model_idx=1)
    assert not np.array_equal(pmi0, pmi1), "model_idx=1 returned model 0's PMI"

    sphere = np.array(m.sphere)
    for idx in (0, 1):
        expected, _ = mir_metric(m.get_unmixing_matrix(idx) @ sphere, real_data)
        actual, _ = m.mir(real_data, model_idx=idx)
        np.testing.assert_allclose(actual, expected, rtol=1e-10)


def test_mir_and_pmi_honour_nbins(real_data):
    m = _fit(real_data)
    default_mir, _ = m.mir(real_data)
    tuned_mir, _ = m.mir(real_data, nbins=20)
    assert default_mir != tuned_mir

    default_pmi = m.pmi(real_data)
    tuned_pmi = m.pmi(real_data, nbins=20)
    assert not np.array_equal(default_pmi, tuned_pmi)


def test_mir_real_fitted_unmixing_is_large_and_positive(real_data):
    m = _fit(real_data, max_iter=8)
    mir_nats, _ = m.mir(real_data)
    assert mir_nats > 0
    identity_mir, _ = mir_metric(np.eye(NW), real_data)
    assert mir_nats > identity_mir


def test_mir_requires_a_fitted_model():
    m = AMICAMLXNG(n_channels=NW, n_mix=NMIX, seed=SEED)
    with pytest.raises(RuntimeError, match="fitted"):
        m.mir(np.zeros((NW, 10)))


# --- mir_step waypoint schedule ---------------------------------------------
def test_mir_step_populates_history_at_right_iterations(real_data):
    m = _fit(real_data, max_iter=5, mir_step=2)
    iterations = [entry[0] for entry in m.mir_history_]
    assert iterations == [0, 2, 4]
    for _, mir_nats, variance in m.mir_history_:
        assert math.isfinite(mir_nats)
        assert math.isfinite(variance)


def test_mir_step_zero_leaves_history_empty(real_data):
    m = _fit(real_data, max_iter=3, mir_step=0)
    assert m.mir_history_ == []


def test_mir_step_negative_raises(real_data):
    m = AMICAMLXNG(n_channels=NW, n_mix=NMIX, seed=SEED, block_size=BLOCK)
    with pytest.raises(ValueError, match="mir_step"):
        m.fit(real_data, max_iter=3, verbose=False, mir_step=-1)


def test_mir_step_zero_matches_omitted_argument(real_data):
    """mir_step=0 (explicit) must leave fit() behaviour byte-for-byte
    identical to not passing mir_step at all."""
    default_m = _fit(real_data, max_iter=3, keep_best=False)
    explicit_m = AMICAMLXNG(
        n_channels=NW, n_mix=NMIX, seed=SEED, block_size=BLOCK, keep_best=False
    )
    explicit_m.fit(real_data, max_iter=3, verbose=False, mir_step=0)

    assert default_m.ll_history == explicit_m.ll_history
    assert default_m.final_ll_ == explicit_m.final_ll_
    np.testing.assert_array_equal(np.array(default_m.W), np.array(explicit_m.W))
    assert default_m.mir_history_ == explicit_m.mir_history_ == []


def test_failing_mir_waypoint_does_not_kill_the_fit(real_data, monkeypatch, caplog):
    """A diagnostic must never destroy a decomposition: a raise inside mir()
    is caught, warned, and recorded as a NaN waypoint rather than
    propagating out of fit(). Forcing the raise via monkeypatch mirrors
    test_amica_ng_wrapper.py::test_failing_mir_waypoint_does_not_kill_the_fit
    -- the real trigger is a transient that cannot be induced on demand, so
    this exercises the fit's RESPONSE to a raising waypoint on otherwise
    real data/arithmetic."""
    real_mir = AMICAMLXNG.mir
    calls = {"n": 0}

    def flaky_mir(self, X, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:  # fail on the second waypoint, mid-fit
            raise ValueError("mir(): unmixing matrix is singular or near-singular")
        return real_mir(self, X, **kwargs)

    monkeypatch.setattr(AMICAMLXNG, "mir", flaky_mir)

    m = AMICAMLXNG(n_channels=NW, n_mix=NMIX, seed=SEED, block_size=BLOCK)
    with caplog.at_level(logging.WARNING, logger="pamica.mlx_impl.core"):
        m.fit(real_data, max_iter=4, verbose=False, mir_step=1)

    assert m.stop_reason is not None
    assert m.final_ll_ is not None and math.isfinite(m.final_ll_)

    iters = [row[0] for row in m.mir_history_]
    assert iters == [0, 1, 2, 3], iters
    values = [row[1] for row in m.mir_history_]
    assert math.isnan(values[1]), "failed waypoint must be a visible NaN"
    assert all(math.isfinite(v) for i, v in enumerate(values) if i != 1)
    assert any(
        "MIR waypoint failed" in r.getMessage() and "iter 1" in r.getMessage()
        for r in caplog.records
    )


# --- the #300 fitted-geometry PCA guard -------------------------------------
def test_mir_raises_under_auto_detected_rank_reduction(real_data):
    """Rank reduction from AUTOMATIC mineig/mineig_rel numerical-rank
    detection (no explicit pcakeep/pcadb -- this backend has none) must
    trip the documented ValueError, not an opaque LinAlgError. Real EEG
    projected onto a rank-16 subspace via SVD, matching
    test_amica_ng_wrapper.py::test_mir_raises_under_auto_detected_rank_reduction."""
    x = real_data - real_data.mean(axis=1, keepdims=True)
    u16 = np.linalg.svd(x, full_matrices=False)[0][:, :16]
    x_low = u16 @ (u16.T @ x)

    m = _fit(x_low, max_iter=2)
    assert m.n_channels_in != m.n_channels, (
        "test setup: automatic rank detection did not reduce the sphere"
    )
    with pytest.raises(ValueError, match="incompatible with PCA reduction"):
        m.mir(x_low)


def test_mir_full_rank_succeeds(real_data):
    """Positive control: an ordinary full-rank fit's sphere is square, so
    mir() must not trip the PCA-reduction guard."""
    m = _fit(real_data, max_iter=3)
    assert m.n_channels_in == m.n_channels
    mir_nats, variance = m.mir(real_data)
    assert math.isfinite(mir_nats) and math.isfinite(variance)


# --- mir_history_ vs keep_best / state_dict (issue #161) --------------------
def test_mir_history_survives_keep_best_restore(real_data):
    """mir_history_ is a TRUE trajectory that a keep_best restore does NOT
    rewrite: its last entry is computed from the pre-restore, discarded
    parameters, not the restored ones fit() actually returns. Uses the
    same forced-overshoot recipe as test_mlx_llt_stash.py, with
    mir_step=1 so a waypoint lands strictly inside the truncation window a
    buggy restore would damage (mirrors test_ng_convergence.py's
    identically-named test and its mir_step=1 rationale)."""
    m = AMICAMLXNG(
        n_channels=NW,
        n_models=2,
        n_mix=NMIX,
        seed=0,
        block_size=BLOCK,
        do_newton=True,
        newt_start=1,
        lrate=0.5,
        use_min_dll=True,
        min_dll=1e-4,
        maxincs=2,
        use_grad_norm=False,
        keep_best=True,
    )
    m.fit(real_data, max_iter=60, verbose=False, mir_step=1)
    if m.stop_reason in AMICAMLXNG._DEGENERATE_STOP_REASONS:
        pytest.skip("aggressive run ended degenerate; not the case under test")
    assert m.final_ll_ is not None
    if np.isclose(m.ll_history[-1], m.final_ll_):
        pytest.skip("run was monotone; keep_best restore did not fire")

    assert m.mir_history_, "test setup: mir_step recorded nothing"
    final_it = len(m.ll_history) - 1
    best_it = m.ll_history.index(m.final_ll_)
    assert best_it < final_it, (
        "test setup: the restore must discard at least one iteration"
    )
    last_it, last_mir, _ = m.mir_history_[-1]
    assert last_it == final_it, "the post-peak waypoints were dropped"
    assert len(m.mir_history_) == final_it + 1, (
        "mir_history_ is not the full per-iteration trajectory"
    )

    mir_now, _ = m.mir(real_data)
    assert not math.isclose(mir_now, last_mir, rel_tol=1e-4)


def test_mir_history_empty_after_state_dict_round_trip(real_data):
    """mir_history_ is not persisted in state_dict() -- a diagnostic
    trajectory, not a fitted parameter -- so a round trip yields an EMPTY
    mir_history_ on the restored model."""
    m = _fit(real_data, max_iter=6, mir_step=2, keep_best=False)
    assert m.mir_history_, "test setup: mir_step recorded nothing"

    state = m.state_dict()
    assert "mir_history_" not in state["extra"]
    restored = AMICAMLXNG.from_state_dict(state)
    assert restored.mir_history_ == []
