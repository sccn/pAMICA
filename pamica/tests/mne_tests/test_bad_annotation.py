"""``reject_by_annotation`` support in ``AMICAICA`` (issue #251).

Real sample EEG only: the bundled ``eeglab_data.set`` with ``bad_*``
annotations added on top -- the annotations are metadata, the data stay real.
Pinned behaviors:

1. fit-time omission matches MNE's own ``get_data(reject_by_annotation="omit")``
   selection exactly, and ``good_sample_mask_`` indexes precisely the fitted
   columns (they come from the same ``get_data`` call, so they cannot drift);
2. fitting an annotated ``Raw`` equals fitting :class:`pamica.AMICA` directly
   on the kept columns -- the wrapper adds bookkeeping, not numerics;
3. per-sample scoring stays timeline-aligned: ``NaN`` exactly on the rejected
   columns, driven by the *passed* instance's annotations;
4. ``start``/``stop`` composes with rejection (the mask is False outside the
   fitted range -- regression for the initial PR #252 draft);
5. ``Epochs`` input is unaffected (regression: ``Epochs.get_data`` has no
   ``reject_by_annotation`` kwarg).
"""

from pathlib import Path

import numpy as np
import pytest

mne = pytest.importorskip("mne")

from pamica import AMICA  # noqa: E402  (after importorskip)
from pamica.mne_compat import AMICAICA  # noqa: E402

mne.set_log_level("ERROR")

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "sample_data"
SET_FILE = SAMPLE_DIR / "eeglab_data.set"
SEED = 42
MAX_ITER = 12  # enough to move off the init; behavior, not convergence

pytestmark = pytest.mark.skipif(
    not SET_FILE.exists(), reason="sample eeglab_data.set missing"
)

# (onset_s, duration_s) of the bad spans; a non-bad "stimulus" annotation is
# added alongside to pin that only bad_* descriptions are rejected.
BAD_SPANS = [(5.0, 2.0), (100.0, 3.0), (200.0, 1.0)]


@pytest.fixture(scope="module")
def raw_annot():
    """Real continuous EEG with three bad_* spans and one non-bad annotation."""
    raw = mne.io.read_raw_eeglab(str(SET_FILE), preload=True)
    raw.set_annotations(
        mne.Annotations(
            onset=[o for o, _ in BAD_SPANS] + [150.0],
            duration=[d for _, d in BAD_SPANS] + [2.0],
            description=["bad_one", "bad_two", "bad_three", "stimulus"],
        )
    )
    return raw


@pytest.fixture(scope="module")
def picked(raw_annot):
    """The channel selection AMICAICA fits on (good data channels)."""
    return raw_annot.copy().pick("data", exclude="bads")


@pytest.fixture(scope="module")
def n_good(picked):
    """MNE's own count of samples surviving bad_* omission."""
    return picked.get_data(reject_by_annotation="omit").shape[1]


@pytest.fixture(scope="module")
def fitted_rej(raw_annot):
    """One annotated-Raw fit (default reject_by_annotation=True), reused."""
    return AMICAICA(n_mix=3, random_state=SEED, device="cpu", verbose=False).fit(
        raw_annot, max_iter=MAX_ITER
    )


def test_fit_omits_bad_annotated_samples(fitted_rej, raw_annot, n_good):
    assert n_good < raw_annot.n_times  # the annotations actually removed data
    assert fitted_rej._n_samples == n_good
    assert fitted_rej.reject_by_annotation_ is True
    mask = fitted_rej.good_sample_mask_
    assert mask is not None and mask.dtype == bool
    assert mask.shape == (raw_annot.n_times,)
    assert int(mask.sum()) == n_good


def test_mask_indexes_exactly_the_fitted_columns(fitted_rej, picked):
    """full_data[:, mask] reproduces MNE's omit selection bit-for-bit."""
    mask = fitted_rej.good_sample_mask_
    full = picked.get_data()
    omitted = picked.get_data(reject_by_annotation="omit")
    np.testing.assert_array_equal(full[:, mask], omitted)


def test_fit_equals_manual_column_drop(fitted_rej, picked):
    """The annotated-Raw fit is the plain AMICA fit on the kept columns."""
    x = np.ascontiguousarray(picked.get_data()[:, fitted_rej.good_sample_mask_])
    # All channels are one type (EEG), so the channel-type pre-whitener is a
    # single global std -- reproduce it as the wrapper computes it.
    x = x / float(np.std(x))
    manual = AMICA(n_mix=3, device="cpu", verbose=False)
    manual.fit(x, seed=SEED, max_iter=MAX_ITER)
    np.testing.assert_array_equal(
        fitted_rej.amica_.get_unmixing_matrix(0), manual.get_unmixing_matrix(0)
    )


def test_reject_false_keeps_all_samples(raw_annot):
    fitted = AMICAICA(n_mix=3, random_state=SEED, device="cpu", verbose=False).fit(
        raw_annot, max_iter=5, reject_by_annotation=False
    )
    assert fitted._n_samples == raw_annot.n_times
    assert fitted.reject_by_annotation_ is False
    mask = fitted.good_sample_mask_
    assert mask is not None and mask.all()


def test_start_stop_composes_with_rejection(raw_annot, picked):
    start, stop = 1000, 20000
    fitted = AMICAICA(n_mix=3, random_state=SEED, device="cpu", verbose=False).fit(
        raw_annot, start=start, stop=stop, max_iter=5
    )
    mask = fitted.good_sample_mask_
    assert mask is not None
    assert not mask[:start].any() and not mask[stop:].any()
    assert int(mask.sum()) == fitted._n_samples
    expected = picked.get_data(
        reject_by_annotation="omit", start=start, stop=stop
    ).shape[1]
    assert fitted._n_samples == expected


def test_model_probability_is_timeline_aligned(fitted_rej, raw_annot):
    """Full-length output, NaN exactly on the rejected columns (issue #251)."""
    mask = fitted_rej.good_sample_mask_
    prob = fitted_rej.get_model_probability(raw_annot)
    assert prob.shape == (1, raw_annot.n_times)
    assert np.isnan(prob[:, ~mask]).all()
    assert np.isfinite(prob[:, mask]).all()
    np.testing.assert_allclose(prob[:, mask], 1.0)  # single model
    # Opting out scores every sample: same shape, no NaN.
    prob_all = fitted_rej.get_model_probability(raw_annot, reject_by_annotation=False)
    assert prob_all.shape == (1, raw_annot.n_times)
    assert np.isfinite(prob_all).all()


def test_scoring_follows_the_passed_instances_annotations(fitted_rej, raw_annot):
    """Rejection is evaluation-time and per-instance, like ICA.score_sources."""
    clean = raw_annot.copy().set_annotations(None)
    prob = fitted_rej.get_model_probability(clean)
    assert prob.shape == (1, clean.n_times)
    assert np.isfinite(prob).all()


def test_mir_scores_good_samples_only(fitted_rej, raw_annot, picked):
    mask = fitted_rej.good_sample_mask_
    x_good = picked.get_data()[:, mask]
    x_good = np.ascontiguousarray(x_good) / fitted_rej.pre_whitener_
    mir_wrapper, _ = fitted_rej.mir(raw_annot)
    mir_manual, _ = fitted_rej.amica_.mir(x_good, model_idx=0)
    assert np.isclose(mir_wrapper, mir_manual)
    # Opting out changes the scored sample set without erroring.
    mir_all, _ = fitted_rej.mir(raw_annot, reject_by_annotation=False)
    assert np.isfinite(mir_all)


def test_epochs_fit_is_unaffected(raw_annot):
    """Regression: Epochs.get_data has no reject_by_annotation kwarg."""
    epochs = mne.make_fixed_length_epochs(raw_annot, duration=2.0, preload=True)
    fitted = AMICAICA(n_mix=3, random_state=SEED, device="cpu", verbose=False).fit(
        epochs, max_iter=5
    )
    assert fitted.good_sample_mask_ is None
    assert fitted.reject_by_annotation_ is False


def test_nan_inside_bad_segment_is_tolerated(raw_annot, n_good):
    """Rejection drops NaN samples inside bad spans before the finite check."""
    data = raw_annot.get_data()
    bad_start = int(BAD_SPANS[0][0] * raw_annot.info["sfreq"])
    data[0, bad_start + 10 : bad_start + 20] = np.nan
    noisy = mne.io.RawArray(data, raw_annot.info.copy(), verbose=False)
    noisy.set_annotations(raw_annot.annotations)
    fitted = AMICAICA(n_mix=3, random_state=SEED, device="cpu", verbose=False).fit(
        noisy, max_iter=5
    )
    assert fitted._n_samples == n_good
    with pytest.raises(ValueError, match="non-finite"):
        AMICAICA(n_mix=3, random_state=SEED, device="cpu", verbose=False).fit(
            noisy, max_iter=5, reject_by_annotation=False
        )


def test_fully_annotated_range_raises(raw_annot):
    covered = raw_annot.copy()
    covered.set_annotations(
        mne.Annotations(
            onset=[0.0],
            duration=[float(raw_annot.times[-1]) + 1.0],
            description=["bad_all"],
        )
    )
    with pytest.raises(ValueError, match="no samples left"):
        AMICAICA(device="cpu", verbose=False).fit(covered, max_iter=5)


def test_empty_start_stop_range_blames_the_range(raw_annot):
    """An inverted range must not be misdiagnosed as annotation rejection."""
    with pytest.raises(ValueError, match="start/stop range selects no samples"):
        AMICAICA(device="cpu", verbose=False).fit(
            raw_annot, start=1000, stop=1000, max_iter=5
        )


def test_fully_annotated_scoring_raises(fitted_rej, raw_annot):
    """All four scoring surfaces refuse an all-bad Raw with one clear error."""
    covered = raw_annot.copy()
    covered.set_annotations(
        mne.Annotations(
            onset=[0.0],
            duration=[float(raw_annot.times[-1]) + 1.0],
            description=["bad_all"],
        )
    )
    for call in (
        lambda: fitted_rej.get_model_probability(covered),
        lambda: fitted_rej.plot_model_probability(covered),
        lambda: fitted_rej.mir(covered),
        lambda: fitted_rej.pmi(covered),
    ):
        with pytest.raises(ValueError, match="nothing to score"):
            call()
    # Opting out still scores the annotated samples.
    prob = fitted_rej.get_model_probability(covered, reject_by_annotation=False)
    assert np.isfinite(prob).all()
