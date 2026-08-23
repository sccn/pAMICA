"""Annotation rejection combined with rank reduction through AMICAICA (#261).

Real sample EEG only, same convention as ``test_bad_annotation.py``: the bundled
``eeglab_data.set`` with ``bad_*`` annotations added on top. This pins the
exact production configuration the external MEG tester runs by hand
(Maxwell-filtered rank-deficient data + ``bad_*`` annotations, #261) -- fitting
``AMICAICA`` with ``reject_by_annotation=True`` *and* rank reduction, either an
explicit ``pcakeep`` or automatic numerical-rank detection (``mineig_rel``, on
by default, issue #223).

The two mechanisms are architecturally independent: the annotation mask
operates on the ``Raw`` timeline before the backend ever sees the data
(``AMICAICA.fit``'s ``_raw_data_and_mask``), while rank reduction is a property
of the backend's covariance eigendecomposition. This module pins that the
combination actually works end to end -- fit, source export, and per-sample
scoring -- not just each half in isolation.
"""

from pathlib import Path

import numpy as np
import pytest

mne = pytest.importorskip("mne")

from pamica.mne_compat import AMICAICA  # noqa: E402  (after importorskip)

mne.set_log_level("ERROR")

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "sample_data"
SET_FILE = SAMPLE_DIR / "eeglab_data.set"
SEED = 42
MAX_ITER = 12  # enough to move off the init; behavior, not convergence
RANK = 16

pytestmark = pytest.mark.skipif(
    not SET_FILE.exists(), reason="sample eeglab_data.set missing"
)

# Same bad spans as test_bad_annotation.py: three bad_* spans plus one non-bad
# "stimulus" annotation, pinning that only bad_* descriptions are rejected.
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


# --- explicit pcakeep --------------------------------------------------------


@pytest.fixture(scope="module")
def fitted_pcakeep(raw_annot):
    """Annotation rejection (default on) AND explicit PCA rank reduction."""
    return AMICAICA(n_mix=3, random_state=SEED, device="cpu", verbose=False).fit(
        raw_annot, max_iter=MAX_ITER, pcakeep=RANK
    )


def test_pcakeep_applies_with_annotation_rejection(fitted_pcakeep, raw_annot, n_good):
    """Rank reduction and annotation rejection both took effect on one fit."""
    assert fitted_pcakeep.converged_ is True
    assert fitted_pcakeep.n_components_ == RANK
    assert fitted_pcakeep._n_samples == n_good
    assert fitted_pcakeep.reject_by_annotation_ is True
    mask = fitted_pcakeep.good_sample_mask_
    assert mask is not None and mask.dtype == bool
    assert mask.shape == (raw_annot.n_times,)
    assert int(mask.sum()) == n_good


def test_pcakeep_sources_are_timeline_shaped_and_rank_sized(fitted_pcakeep, raw_annot):
    """get_sources spans the full timeline and is sized to the retained rank,
    not the channel count; it is not itself annotation-aware (no NaN gaps)."""
    s = fitted_pcakeep.get_sources(raw_annot).get_data()
    assert s.shape == (RANK, raw_annot.n_times)
    assert np.isfinite(s).all()


def test_pcakeep_scoring_is_timeline_aligned(fitted_pcakeep, raw_annot):
    """get_model_probability keeps the NaN-at-rejected-samples contract (#251)
    even when the fit is also rank-reduced (#225): the two features compose."""
    mask = fitted_pcakeep.good_sample_mask_
    prob = fitted_pcakeep.get_model_probability(raw_annot)
    assert prob.shape == (1, raw_annot.n_times)
    assert np.isnan(prob[:, ~mask]).all()
    assert np.isfinite(prob[:, mask]).all()
    np.testing.assert_allclose(prob[:, mask], 1.0)  # single model


def test_pcakeep_pmi_scores_good_samples_only(fitted_pcakeep, raw_annot, picked):
    mask = fitted_pcakeep.good_sample_mask_
    x_good = np.ascontiguousarray(picked.get_data()[:, mask])
    x_good = x_good / fitted_pcakeep.pre_whitener_
    pmi_wrapper = fitted_pcakeep.pmi(raw_annot)
    pmi_manual = fitted_pcakeep.amica_.pmi(x_good, model_idx=0)
    assert pmi_wrapper.shape == (RANK, RANK)
    np.testing.assert_allclose(pmi_wrapper, pmi_manual)


def test_pcakeep_mir_still_refuses_pca_reduction(fitted_pcakeep, raw_annot):
    """mir() is documented as incompatible with PCA reduction regardless of
    annotation rejection (the rank-deficient sphere has no log-Jacobian term);
    pin that the combination still raises that clean, intentional error rather
    than something else once annotation-aware scoring is layered on top."""
    with pytest.raises(ValueError, match="incompatible with PCA reduction"):
        fitted_pcakeep.mir(raw_annot)


def test_pcakeep_export_is_rank_sized_and_orthonormal(fitted_pcakeep):
    """The exported MNE ICA reflects the reduced rank, not the channel count."""
    ica = fitted_pcakeep.to_mne_ica()
    assert ica.pca_components_.shape == (RANK, len(fitted_pcakeep.ch_names_))
    np.testing.assert_allclose(
        ica.pca_components_ @ ica.pca_components_.T, np.eye(RANK), atol=1e-10
    )


# --- automatic numerical-rank detection (no explicit pcakeep) ---------------


@pytest.fixture(scope="module")
def raw_annot_low_rank(raw_annot, picked):
    """Real EEG linearly projected onto a rank-16 subspace, mirroring the
    Maxwell-filtered MEG rank deficiency the external tester reported (#261),
    carrying over the original bad_* annotations verbatim so the timeline and
    annotation spans match ``raw_annot``/``n_good`` exactly."""
    x = picked.get_data()
    x = x - x.mean(axis=1, keepdims=True)
    u_r = np.linalg.svd(x, full_matrices=False)[0][:, :RANK]
    x_low = u_r @ (u_r.T @ x)
    raw_low = mne.io.RawArray(x_low, picked.info.copy(), verbose=False)
    raw_low.set_annotations(raw_annot.annotations)
    return raw_low


@pytest.fixture(scope="module")
def fitted_auto_rank(raw_annot_low_rank):
    """Annotation rejection (default on) with NO explicit pcakeep: the default
    ``mineig_rel`` floor must auto-detect the rank-16 subspace on its own."""
    return AMICAICA(n_mix=3, random_state=SEED, device="cpu", verbose=False).fit(
        raw_annot_low_rank, max_iter=MAX_ITER
    )


def test_auto_rank_detection_applies_with_annotation_rejection(
    fitted_auto_rank, raw_annot_low_rank, n_good
):
    assert fitted_auto_rank.converged_ is True
    assert fitted_auto_rank.n_components_ == RANK
    assert fitted_auto_rank._n_samples == n_good
    assert fitted_auto_rank.reject_by_annotation_ is True
    mask = fitted_auto_rank.good_sample_mask_
    assert mask is not None and mask.shape == (raw_annot_low_rank.n_times,)
    assert int(mask.sum()) == n_good


def test_auto_rank_sources_are_timeline_shaped_and_rank_sized(
    fitted_auto_rank, raw_annot_low_rank
):
    s = fitted_auto_rank.get_sources(raw_annot_low_rank).get_data()
    assert s.shape == (RANK, raw_annot_low_rank.n_times)
    assert np.isfinite(s).all()


def test_auto_rank_scoring_is_timeline_aligned(fitted_auto_rank, raw_annot_low_rank):
    mask = fitted_auto_rank.good_sample_mask_
    prob = fitted_auto_rank.get_model_probability(raw_annot_low_rank)
    assert prob.shape == (1, raw_annot_low_rank.n_times)
    assert np.isnan(prob[:, ~mask]).all()
    assert np.isfinite(prob[:, mask]).all()


def test_auto_rank_pmi_scores_good_samples_only(
    fitted_auto_rank, raw_annot_low_rank, picked
):
    """pmi() has no PCA-reduction guard (it only needs `transform`, which works
    at any rank), so unlike mir() it is exercised here for real."""
    mask = fitted_auto_rank.good_sample_mask_
    picked_low = raw_annot_low_rank.copy().pick("data", exclude="bads")
    x_good = np.ascontiguousarray(picked_low.get_data()[:, mask])
    x_good = x_good / fitted_auto_rank.pre_whitener_
    pmi_wrapper = fitted_auto_rank.pmi(raw_annot_low_rank)
    pmi_manual = fitted_auto_rank.amica_.pmi(x_good, model_idx=0)
    assert pmi_wrapper.shape == (RANK, RANK)
    np.testing.assert_allclose(pmi_wrapper, pmi_manual)
