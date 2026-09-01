"""Outlier rejection (``do_reject``) on the MLX backend -- issue #123's
``AMICATorchNG`` mechanism, epic #278 Phase 3/#289.

MLX-only mechanics: constructor validation, the good_idx schedule, the
maxrej cap, the all-rejected/none-rejected edges, the keep_best inactive
warning, and state_dict round-tripping. Cross-backend AGREEMENT (same real
data, same config, torch vs MLX reject the same sample set) lives in
``pamica/tests/test_mlx_reject_cross_backend.py`` per ``.rules/
backend_parity.md`` (a behavior-agreement pin belongs outside any one
backend's subdirectory).

Apple-Silicon only, real sample EEG (no synthetic/mock) except for the
directly-constructed ``_reject_outliers`` unit tests below, which follow
``pamica/tests/test_numpy_reject.py``'s pattern of calling the method on a
hand-built ``good_idx``/``ll_vec`` -- the sanctioned way to reach the
all-rejected edge case, which a real fit cannot trigger organically (see
``_reject_outliers``'s own docstring: for finite log-likelihoods the max
sample is always kept).
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from pamica.mlx_impl import AMICAMLXNG  # noqa: E402  (after the MLX importorskip)

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "sample_data"
DATA_FILE = SAMPLE_DIR / "eeglab_data.fdt"
NW = 32
FIELD = 30504
NMIX = 3
BLOCK = 1024

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


def _model(**kwargs: Any) -> AMICAMLXNG:
    params: dict[str, Any] = dict(n_channels=NW, n_mix=NMIX, block_size=BLOCK)
    params.update(kwargs)
    return AMICAMLXNG(**params)


# --- constructor validation --------------------------------------------
def test_reject_param_validation():
    """Matches AMICATorchNG's validation exactly (torch_impl/core.py:
    679-685): rejint<1 would ZeroDivisionError in the reject schedule,
    rejsig<=0 breaks the reject-below-mean semantics, maxrej<0 is a sanity
    guard. rejstart is NOT validated (torch does not validate it either)."""
    with pytest.raises(ValueError, match="rejint"):
        _model(do_reject=True, rejint=0)
    with pytest.raises(ValueError, match="rejsig"):
        _model(do_reject=True, rejsig=0.0)
    with pytest.raises(ValueError, match="rejsig"):
        _model(do_reject=True, rejsig=-5.0)
    with pytest.raises(ValueError, match="maxrej"):
        _model(do_reject=True, maxrej=-1)


def test_reject_defaults_match_torch():
    """do_reject=False, rejsig=3.0, rejstart=2, rejint=3, maxrej=1 -- the
    same defaults AMICATorchNG and the NumPy backend ship (Fortran's own
    amica15_header.f90 defaults)."""
    m = _model()
    assert m.do_reject is False
    assert m.rejsig == 3.0
    assert m.rejstart == 2
    assert m.rejint == 3
    assert m.maxrej == 1


def test_do_reject_false_leaves_good_idx_unset(real_data):
    """The default path never touches the rejection machinery: good_idx
    stays None and the fit is byte-for-byte unaffected (issue #24/#157
    parity paths)."""
    m = _model(seed=1, keep_best=False)
    m.fit(real_data, max_iter=3, verbose=False)
    assert m.good_idx is None
    assert m.numrej == 0


# --- the reject schedule -------------------------------------------------
def test_rejection_shrinks_good_sample_set_on_the_expected_schedule(real_data):
    """rejstart=2/rejint=3/maxrej=2: rejection fires at it=2 (unconditional)
    and it=5 (max(1,5-2)%3==0, numrej<2), then is capped -- no more passes
    at it=8/11 despite the modulo condition recurring, matching Fortran's
    schedule (amica15.f90:1142) and the torch/NumPy backends' own tests."""
    n_total = real_data.shape[1]
    m = _model(
        seed=42,
        do_reject=True,
        rejsig=2.0,
        rejstart=2,
        rejint=3,
        maxrej=2,
        keep_best=False,
    )
    m.fit(real_data, max_iter=12, verbose=False)

    assert m.good_idx is not None
    n_good = int(m.good_idx.size)
    assert n_good < n_total
    assert m.numrej == 2  # capped at maxrej, not the 4 modulo hits available
    # good_idx is a strict subset of the original indices, unique and in range.
    good = np.array(m.good_idx)
    assert good.min() >= 0 and good.max() < n_total
    assert len(set(good.tolist())) == n_good
    assert np.all(np.isfinite(np.array(m.A)))
    assert np.all(np.isfinite(np.array(m.ll_history)))


def test_maxrej_zero_disables_rejection(real_data):
    """maxrej=0 is a valid, inert configuration (matching AMICATorchNG's
    ``maxrej > 0`` gate in the schedule condition): good_idx is allocated
    (do_reject=True) but never shrinks."""
    n_total = real_data.shape[1]
    m = _model(seed=1, do_reject=True, maxrej=0, keep_best=False)
    m.fit(real_data, max_iter=6, verbose=False)
    assert m.good_idx is not None
    assert int(m.good_idx.size) == n_total
    assert m.numrej == 0


def test_none_rejected_in_a_pass_still_counts_as_a_pass(real_data):
    """A generous rejsig can make a scheduled pass fire (the schedule does
    not know in advance whether anything will cross the threshold) and drop
    NOTHING -- numrej still increments (a pass ran) but good_idx does not
    shrink, distinct from maxrej=0's "never scheduled" case above."""
    n_total = real_data.shape[1]
    m = _model(
        seed=42,
        do_reject=True,
        rejsig=100.0,
        rejstart=2,
        rejint=3,
        maxrej=2,
        keep_best=False,
    )
    m.fit(real_data, max_iter=10, verbose=False)
    assert m.numrej == 2
    assert m.good_idx is not None and int(m.good_idx.size) == n_total


def test_multimodel_rejection_keeps_gm_and_c_finite(real_data):
    """do_reject + n_models=2: a shrinking good set can drive a model's
    responsibility mass toward zero, exercising the dead-model dgm==0
    guards in the per-model bias c and gm updates (mirrors
    test_numpy_reject.py's identically-named test)."""
    m = _model(
        n_models=2,
        seed=1,
        do_reject=True,
        rejsig=2.0,
        rejstart=2,
        maxrej=1,
        keep_best=False,
    )
    m.fit(real_data, max_iter=8, verbose=False)
    assert m.good_idx is not None and int(m.good_idx.size) < real_data.shape[1]
    gm = np.array(m.gm)
    c = np.array(m.c)
    assert np.all(np.isfinite(c))
    assert np.all(np.isfinite(gm))
    assert gm.min() >= 0.0
    np.testing.assert_allclose(gm.sum(), 1.0, atol=1e-5)


# --- keep_best interaction ------------------------------------------------
def test_keep_best_inactive_under_do_reject_warns(real_data, caplog):
    """keep_best defaults on; enabling do_reject must surface the same
    inactive-safeguard warning AMICATorchNG logs, not silently drop the
    safeguard (track_best now also excludes do_reject, the Phase-2 marker
    comment's extension point)."""
    import logging

    m = _model(seed=1, do_reject=True, rejstart=2, rejint=2, maxrej=1)
    assert m.keep_best is True
    with caplog.at_level(logging.WARNING, logger="pamica.mlx_impl.core"):
        m.fit(real_data, max_iter=6, verbose=False)
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "keep_best is inactive under do_reject" in text


def test_keep_best_inactive_reason_prefers_do_reject_when_both_are_on(
    real_data, caplog
):
    """PR #311 review: when do_reject AND share_comps are BOTH on, the
    reported reason must be "do_reject", matching AMICATorchNG's exact
    precedence (torch_impl/core.py:2425, ``"do_reject" if self.do_reject
    else "share_comps"``) -- the two backends must report the same reason
    for the same configuration, not whichever flag MLX happened to check
    first."""
    import logging

    m = _model(
        n_models=2,
        seed=1,
        do_reject=True,
        rejstart=2,
        rejint=2,
        maxrej=1,
        share_comps=True,
        share_start=10,
        share_iter=10,
        comp_thresh=0.9,
    )
    assert m.keep_best is True
    with caplog.at_level(logging.WARNING, logger="pamica.mlx_impl.core"):
        m.fit(real_data, max_iter=6, verbose=False)
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "keep_best is inactive under do_reject" in text
    assert "keep_best is inactive under share_comps" not in text


def test_keep_best_restore_never_fires_under_do_reject(real_data):
    """Even the aggressive-Newton recipe known to overshoot under plain
    keep_best (test_mlx_llt_stash.py) never restores when do_reject is on:
    fit() always returns the last iterate there."""
    m = _model(
        n_models=2,
        seed=0,
        do_newton=True,
        newt_start=1,
        lrate=0.5,
        use_min_dll=True,
        min_dll=1e-4,
        maxincs=2,
        use_grad_norm=False,
        do_reject=True,
        rejsig=3.0,
        rejstart=5,
        rejint=5,
        maxrej=1,
    )
    m.fit(real_data, max_iter=30, verbose=False)
    if m.stop_reason in AMICAMLXNG._DEGENERATE_STOP_REASONS:
        pytest.skip("run ended degenerate; not the case under test")
    assert m.final_ll_ == m.ll_history[-1]


# --- interaction with best-of-N restarts and component sharing -------------
def test_do_reject_survives_best_of_n_restarts_winner_not_last(real_data):
    """PR #311 review: do_reject's good_idx/numrej must round-trip through
    the restart snapshot/restore exactly like every other fit-path state
    (they are already in _RESTART_STATE_ATTRS, added alongside do_reject
    itself), pinned explicitly for a winner that is NOT the last restart --
    the only scenario that actually exercises _apply_restart_state rather
    than leaving the live (already-correct) final restart's state in place.

    Reuses test_mlx_restarts.py's SEEDS = [54, 50, 42] (measured there,
    without do_reject, to have the winner strictly first); confirmed
    empirically to still produce winner index 0 (not the last restart)
    under this do_reject config on this data.
    """
    seeds = [54, 50, 42]
    kwargs: dict[str, Any] = dict(
        n_channels=NW,
        n_mix=NMIX,
        do_reject=True,
        rejsig=2.0,
        rejstart=2,
        rejint=3,
        maxrej=2,
    )

    solo_fits = []
    for seed in seeds:
        solo = AMICAMLXNG(seed=seed, **kwargs)
        solo.fit(real_data, max_iter=5, verbose=False)
        solo_fits.append(solo)

    best_of_n = AMICAMLXNG(seed=0, n_restarts=len(seeds), restart_seeds=seeds, **kwargs)
    best_of_n.fit(real_data, max_iter=5, verbose=False)

    winner = max(range(len(seeds)), key=lambda i: solo_fits[i].final_ll_)
    assert winner != len(seeds) - 1, (
        "test setup: the winning seed must not be the last restart, or the "
        "restore path (_apply_restart_state) goes untested"
    )
    assert best_of_n.seed == seeds[winner]

    winning_solo = solo_fits[winner]
    assert winning_solo.good_idx is not None and best_of_n.good_idx is not None
    assert best_of_n.numrej == winning_solo.numrej
    np.testing.assert_array_equal(
        np.array(best_of_n.good_idx), np.array(winning_solo.good_idx)
    )
    np.testing.assert_array_equal(np.array(best_of_n.A), np.array(winning_solo.A))
    assert best_of_n.final_ll_ == winning_solo.final_ll_


def test_do_reject_with_a_genuine_share_comps_merge(real_data):
    """do_reject x share_comps, both firing for real within one fit (not
    the standalone-_identify_shared_comps pattern test_mlx_sharing.py uses
    elsewhere): comp_thresh=0.9 is the same real-data recipe
    test_mlx_sharing.py::test_merge_on_the_final_iteration_completes uses
    to force a genuine merge (not synthetic/forced-column data), combined
    with a do_reject schedule that also fires. The fit must complete with
    finite parameters, the merged (shrunken) comp_used mask must survive
    on the returned model, and good_idx must have shrunk too -- i.e.
    neither mechanism silently no-ops or corrupts the other.
    """
    m = _model(
        n_models=2,
        seed=42,
        share_comps=True,
        share_start=8,
        share_iter=100,
        comp_thresh=0.9,
        do_reject=True,
        rejsig=2.0,
        rejstart=2,
        rejint=3,
        maxrej=2,
        keep_best=False,
    )
    m.fit(real_data, max_iter=15, verbose=False)

    assert m.stop_reason not in AMICAMLXNG._DEGENERATE_STOP_REASONS
    assert m.final_ll_ is not None and np.isfinite(m.final_ll_)
    for name in ("A", "W", "mu", "alpha", "beta", "rho", "gm", "c"):
        assert np.all(np.isfinite(np.array(getattr(m, name)))), name

    used = int(np.array(m.comp_used).sum())
    assert used < m.n_comps, "test setup: no merge fired"
    cl = np.array(m.comp_list)
    assert used == np.unique(cl).size, "comp_list disagrees with comp_used"

    assert m.good_idx is not None
    n_good = int(m.good_idx.size)
    assert n_good < real_data.shape[1], "test setup: no rejection fired"
    assert m.numrej > 0


# --- persistence -----------------------------------------------------------
def test_state_dict_round_trip_persists_numrej_and_good_idx(real_data):
    m = _model(seed=42, do_reject=True, rejsig=2.0, rejstart=2, rejint=3, maxrej=2)
    m.fit(real_data, max_iter=10, verbose=False)
    assert m.numrej > 0 and m.good_idx is not None

    state = m.state_dict()
    assert state["extra"]["numrej"] == m.numrej
    assert state["extra"]["good_idx"] == np.array(m.good_idx).astype(np.int64).tolist()

    restored = AMICAMLXNG.from_state_dict(state)
    assert restored.numrej == m.numrej
    np.testing.assert_array_equal(np.array(restored.good_idx), np.array(m.good_idx))


def test_save_load_round_trip_persists_numrej_and_good_idx(real_data, tmp_path):
    m = _model(seed=42, do_reject=True, rejsig=2.0, rejstart=2, rejint=3, maxrej=2)
    m.fit(real_data, max_iter=10, verbose=False)
    path = tmp_path / "model.npz"
    m.save(str(path))
    restored = AMICAMLXNG.load(str(path))
    assert restored.numrej == m.numrej
    np.testing.assert_array_equal(np.array(restored.good_idx), np.array(m.good_idx))


def test_phase2_era_payload_loads_with_no_rejection_state(real_data):
    """A state_dict missing numrej/good_idx (a payload saved before Phase 3
    added them) falls back additively -- the one place phase-1's
    require-all-extra-keys check is relaxed, exactly for these two keys."""
    m = _model(seed=1, keep_best=False)
    m.fit(real_data, max_iter=3, verbose=False)
    state = m.state_dict()
    assert "numrej" in state["extra"] and "good_idx" in state["extra"]
    del state["extra"]["numrej"]
    del state["extra"]["good_idx"]

    restored = AMICAMLXNG.from_state_dict(state)
    assert restored.numrej == 0
    assert restored.good_idx is None


def test_missing_core_extra_key_still_raises(real_data):
    """The relaxation is scoped to exactly numrej/good_idx: every
    phase-1-era key stays strictly required."""
    m = _model(seed=1)
    m.fit(real_data, max_iter=2, verbose=False)
    state = m.state_dict()
    del state["extra"]["sldet"]
    with pytest.raises(ValueError, match="missing extra fields"):
        AMICAMLXNG.from_state_dict(state)


# --- _reject_outliers unit tests (sanctioned direct-call pattern, mirroring
# test_numpy_reject.py's test_rejection_nonfinite_ll_raises_instability_error)
# ---------------------------------------------------------------------------
def test_reject_outliers_nonfinite_ll_raises_instability_error():
    """A non-finite per-sample LL (numerical instability upstream) makes
    rejection raise a clear error naming the real cause, not blaming
    rejsig (issue #127). Even a single NaN poisons mean/std and would drop
    every sample."""
    m = _model(do_reject=True, rejsig=3.0)
    m.good_idx = mx.arange(16)
    m._llt_logv = mx.zeros((16, 1), dtype=mx.float32)
    m._llt_ll = mx.zeros((16,), dtype=mx.float32)
    ll = np.arange(16, dtype=np.float64)
    ll[7] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        m._reject_outliers(ll)


def test_reject_outliers_all_finite_never_drops_everything():
    """Defensive: for finite log-likelihoods the max sample is always kept
    (rejsig>0 validated at construction), so a large but finite spread
    cannot trigger the "removed all samples" branch."""
    m = _model(do_reject=True, rejsig=0.001)
    m.good_idx = mx.arange(20)
    m._llt_logv = mx.zeros((20, 1), dtype=mx.float32)
    m._llt_ll = mx.zeros((20,), dtype=mx.float32)
    rng = np.random.RandomState(0)
    ll = rng.randn(20) * 5.0
    m._reject_outliers(ll)  # must not raise
    assert int(m.good_idx.size) >= 1
    assert m.numrej == 1


def test_reject_outliers_zeroes_the_llt_stash_for_dropped_samples():
    m = _model(do_reject=True, rejsig=1.0)
    m.good_idx = mx.arange(10)
    m._llt_logv = mx.arange(20, dtype=mx.float32).reshape(10, 2) + 1.0
    m._llt_ll = mx.arange(10, dtype=mx.float32) + 1.0
    ll = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, -100.0])
    m._reject_outliers(ll)
    assert int(m.good_idx.size) == 9
    assert 9 not in np.array(m.good_idx).tolist()
    logv_np = np.array(m._llt_logv)
    ll_np = np.array(m._llt_ll)
    assert np.array_equal(logv_np[9], np.zeros(2, dtype=np.float32))
    assert ll_np[9] == 0.0
    # The kept rows are untouched.
    assert not np.any(logv_np[:9] == 0.0)
