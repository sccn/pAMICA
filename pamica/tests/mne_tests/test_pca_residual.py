"""PCA-residual restoration in the AMICAICA export (issue #322).

Real sample EEG only: the bundled EEGLAB ``eeglab_data.set`` (32 channels,
30504 samples) loaded through ``mne.io.read_raw_eeglab``, as in the other
``mne_tests`` modules.

The pins below capture the export exactly as it stood before issue #322, by
recomputing it inline with the original formula (``_reference_export``), and
assert that a full-rank fit still exports it bit for bit.
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
MAX_ITER = 10  # the export is a function of the fitted state, not convergence

pytestmark = pytest.mark.skipif(
    not SET_FILE.exists(), reason="sample eeglab_data.set missing"
)


@pytest.fixture(scope="module")
def raw():
    """Real continuous EEG as an MNE Raw (32 EEG channels, 128 Hz)."""
    return mne.io.read_raw_eeglab(str(SET_FILE), preload=True)


@pytest.fixture(scope="module")
def fitted_full(raw):
    """A full-rank single-model fit (no reduction: 32 components)."""
    return AMICAICA(random_state=SEED, device="cpu", verbose=False).fit(
        raw, max_iter=MAX_ITER
    )


@pytest.fixture(scope="module")
def fitted_full_2m(raw):
    """A full-rank two-model fit, whose per-model center ``c`` is nonzero."""
    return AMICAICA(n_models=2, random_state=SEED, device="cpu", verbose=False).fit(
        raw, max_iter=MAX_ITER
    )


def _reference_export(fitted, model_idx=0):
    """The pre-#322 ``to_mne_ica`` attributes, recomputed by the original formula.

    Eigenbasis of a square sphere (``eigh``, descending variance) or the right
    singular vectors of a reduced one; ``pca_components_`` is the retained
    basis only, and the per-model center ``c`` folds into ``pca_mean_``.
    """
    backend = fitted.amica_.model_
    mean = backend.mean.cpu().numpy().ravel()
    sphere = backend.sphere.cpu().numpy()
    w_fort = fitted.amica_.get_unmixing_matrix(model_idx=model_idx)
    c = backend.c.cpu().numpy()[:, model_idx]
    n_ch, n_in = sphere.shape
    if n_ch == n_in:
        sphere_evals, evecs = np.linalg.eigh(sphere)
        cov_evals = 1.0 / sphere_evals**2
        order = np.argsort(cov_evals)[::-1]
        v = evecs[:, order]
        cov_evals = cov_evals[order]
    else:
        _, svals, vt = np.linalg.svd(sphere, full_matrices=False)
        order = np.argsort(svals)
        v = vt[order].T
        cov_evals = 1.0 / svals[order] ** 2
    pca_mean = mean + np.linalg.pinv(sphere) @ c if np.any(c) else mean
    return {
        "pca_components_": v.T,
        "pca_explained_variance_": cov_evals,
        "unmixing_matrix_": w_fort @ sphere @ v,
        "pca_mean_": pca_mean,
    }


def _reference_ica(fitted, model_idx=0):
    """An MNE ICA assembled from :func:`_reference_export` the pre-#322 way."""
    attrs = _reference_export(fitted, model_idx)
    n = attrs["pca_components_"].shape[0]
    ica = mne.preprocessing.ICA(
        n_components=n, method="infomax", max_iter="auto", random_state=SEED
    )
    ica.info = fitted.info_
    ica.ch_names = list(fitted.ch_names_)
    ica.n_components_ = n
    for name, value in attrs.items():
        setattr(ica, name, value)
    ica.pre_whitener_ = fitted.pre_whitener_
    ica.n_iter_ = MAX_ITER
    ica.reject_ = None
    ica.n_samples_ = fitted.to_mne_ica(model_idx).n_samples_
    ica._update_mixing_matrix()
    ica._update_ica_names()
    ica.current_fit = fitted._fit_kind
    return ica


# --- pin: a full-rank export is unchanged -----------------------------------
def test_full_rank_export_is_unchanged(fitted_full):
    """Every exported PCA/ICA attribute equals the pre-#322 formula bit for bit."""
    ica = fitted_full.to_mne_ica()
    ref = _reference_export(fitted_full)
    assert ica.pca_components_.shape == (32, 32)
    for name, expected in ref.items():
        np.testing.assert_array_equal(getattr(ica, name), expected, err_msg=name)


def test_full_rank_two_model_export_is_unchanged(fitted_full_2m):
    """The per-model ``c`` fold into ``pca_mean_`` is untouched, for each model."""
    c = fitted_full_2m.amica_.model_.c.cpu().numpy()
    for h in range(2):
        assert np.any(c[:, h]), f"model {h} center c should be nonzero"
        ica = fitted_full_2m.to_mne_ica(h)
        ref = _reference_export(fitted_full_2m, h)
        for name, expected in ref.items():
            np.testing.assert_array_equal(
                getattr(ica, name), expected, err_msg=f"model {h}: {name}"
            )


@pytest.mark.parametrize("exclude", [[], [0, 3]])
def test_full_rank_apply_is_unchanged(raw, fitted_full, exclude):
    """``apply`` output equals the pre-#322 export's, bit for bit."""
    out = fitted_full.apply(raw.copy(), exclude=exclude).get_data()
    ref = _reference_ica(fitted_full).apply(raw.copy(), exclude=exclude).get_data()
    np.testing.assert_array_equal(out, ref)
