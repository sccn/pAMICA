"""PCA-residual restoration in the AMICAICA export (issue #322).

Real sample EEG only: the bundled EEGLAB ``eeglab_data.set`` (32 channels,
30504 samples) loaded through ``mne.io.read_raw_eeglab``, as in the other
``mne_tests`` modules.

The pins capture the export exactly as it stood before issue #322, by
recomputing it inline with the original formula (``_reference_export``), and
assert that a full-rank fit still exports it bit for bit.

The rest covers rank-reduced fits, whose export now carries the full PCA basis:
the retained rows first (unchanged), then the residual the reduction discarded.
MNE's ``ICA.apply`` keeps rows past ``n_components_`` as residual PCA
components, so the residual is restored by default and
``apply(n_pca_components=n_components_)`` gives the reference's rank-reduced
reconstruction. Independent oracles throughout: the input itself, the
backend's own ``transform``/sphere/unmixing, and ``np.linalg.eigvalsh`` of the
fit data's covariance.
"""

import logging
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


# --- rank-reduced fits: the residual is restored ------------------------------
N_KEEP = 20
REL_TOL = 1e-10  # "reproduces"/"unchanged": measured ~1e-15 on the sample


def _rel(a, b):
    """Frobenius-norm relative difference of ``a`` from ``b``."""
    return np.linalg.norm(a - b) / np.linalg.norm(b)


def _data(inst, fitted):
    """The fitted channels of ``inst`` in original units, epochs concatenated."""
    x = inst.get_data(picks=fitted.ch_names_)
    return np.hstack(x) if x.ndim == 3 else x


def _discarded_eigenvalues(x_scaled, n_keep):
    """Oracle: the fit covariance eigenvalues past ``n_keep``, descending."""
    return np.linalg.eigvalsh(np.cov(x_scaled, bias=True))[::-1][n_keep:]


def _back_projection(fitted, x_scaled, k, model_idx=0):
    """Oracle: component ``k``'s contribution in pre-whitened sensor space.

    Built from the backend alone (``pinv(sphere) @ inv(W)`` and
    ``AMICA.transform``), not from the exported MNE matrices.
    """
    sphere = fitted.amica_.model_.sphere.cpu().numpy()
    w_fort = fitted.amica_.get_unmixing_matrix(model_idx=model_idx)
    a_k = (np.linalg.pinv(sphere) @ np.linalg.inv(w_fort))[:, k]
    s_k = fitted.amica_.transform(x_scaled, model_idx=model_idx)[k]
    return np.outer(a_k, s_k)


def _assert_exclude_removes_only_component(fitted, inst, k, model_idx=0):
    """``exclude=[k]`` subtracts exactly component ``k``'s back-projection.

    The change is compared in original units against the backend oracle, and
    its projection on the residual rows must vanish: excluding an ICA
    component cannot touch the subspace ICA never modeled.
    """
    ica = fitted.to_mne_ica(model_idx)
    x = _data(inst, fitted)
    pw = fitted.pre_whitener_
    out = _data(ica.apply(inst.copy(), exclude=[k]), fitted)
    change = out - x
    expected = -pw * _back_projection(fitted, x / pw, k, model_idx)
    assert _rel(change, expected) <= REL_TOL
    d = change / pw
    residual_rows = ica.pca_components_[ica.n_components_ :]
    assert np.linalg.norm(residual_rows @ d) <= REL_TOL * np.linalg.norm(d)


@pytest.fixture(scope="module")
def fitted_keep(raw):
    """Explicit PCA reduction to 20 of 32 dimensions."""
    return AMICAICA(random_state=SEED, device="cpu", verbose=False).fit(
        raw, max_iter=MAX_ITER, pcakeep=N_KEEP
    )


def test_fitted_basis_is_full_and_exported_unaltered(fitted_keep):
    """The fit stores the full basis; the export carries it, retained rows
    exactly as the pre-#322 formula computed them."""
    assert fitted_keep.n_components_ == N_KEEP
    assert fitted_keep.pca_components_.shape == (32, 32)
    assert fitted_keep.pca_explained_variance_.shape == (32,)
    ica = fitted_keep.to_mne_ica()
    assert ica.n_components_ == N_KEEP
    np.testing.assert_array_equal(ica.pca_components_, fitted_keep.pca_components_)
    np.testing.assert_array_equal(
        ica.pca_explained_variance_, fitted_keep.pca_explained_variance_
    )
    ref = _reference_export(fitted_keep)
    np.testing.assert_array_equal(ica.pca_components_[:N_KEEP], ref["pca_components_"])
    np.testing.assert_array_equal(
        ica.pca_explained_variance_[:N_KEEP], ref["pca_explained_variance_"]
    )
    for name in ("unmixing_matrix_", "pca_mean_"):
        np.testing.assert_array_equal(getattr(ica, name), ref[name], err_msg=name)


def test_exported_basis_is_orthonormal(fitted_keep):
    p = fitted_keep.to_mne_ica().pca_components_
    np.testing.assert_allclose(p @ p.T, np.eye(32), rtol=0, atol=1e-12)


def test_residual_variances_are_the_discarded_eigenvalues(raw, fitted_keep):
    """Residual rows carry the covariance eigenvalues the reduction dropped."""
    x = _data(raw, fitted_keep) / fitted_keep.pre_whitener_
    ev = fitted_keep.to_mne_ica().pca_explained_variance_
    np.testing.assert_allclose(
        ev[N_KEEP:], _discarded_eigenvalues(x, N_KEEP), rtol=1e-8
    )
    assert np.all(np.diff(ev) <= 0), "explained variance must be descending"


def test_apply_restores_the_residual(raw, fitted_keep):
    """No exclusions: the input comes back, residual included (issue #322)."""
    x = _data(raw, fitted_keep)
    out = _data(fitted_keep.apply(raw.copy()), fitted_keep)
    assert _rel(out, x) <= REL_TOL
    # The pre-#322 export lost the residual here (~8% of the signal), so the
    # check above is discriminative.
    before = _data(_reference_ica(fitted_keep).apply(raw.copy()), fitted_keep)
    assert _rel(before, x) > 1e-3


def test_n_pca_components_restores_the_reference_reconstruction(raw, fitted_keep):
    """``n_pca_components=n_components_`` is exactly the pre-#322 behavior."""
    ica = fitted_keep.to_mne_ica()
    reduced = ica.apply(raw.copy(), n_pca_components=ica.n_components_)
    before = _reference_ica(fitted_keep).apply(raw.copy())
    np.testing.assert_array_equal(reduced.get_data(), before.get_data())
    # MNE's persistent form of the same knob, set on a copy of the cached export.
    stored = ica.copy()
    stored.n_pca_components = stored.n_components_
    np.testing.assert_array_equal(
        stored.apply(raw.copy()).get_data(), before.get_data()
    )
    # Independent oracle: projection onto the retained subspace, around the mean.
    x = _data(raw, fitted_keep)
    pw = fitted_keep.pre_whitener_
    v = ica.pca_components_[:N_KEEP]
    mean = ica.pca_mean_[:, None]
    expected = pw * (v.T @ (v @ (x / pw - mean)) + mean)
    assert _rel(_data(reduced, fitted_keep), expected) <= REL_TOL


@pytest.mark.parametrize("k", [0, N_KEEP - 1])
def test_exclude_leaves_the_residual_untouched(raw, fitted_keep, k):
    _assert_exclude_removes_only_component(fitted_keep, raw, k)


def test_get_sources_matches_amica_transform(raw, fitted_keep):
    """The residual rows are invisible to the sources."""
    s_mne = fitted_keep.get_sources(raw).get_data()
    x = _data(raw, fitted_keep) / fitted_keep.pre_whitener_
    s_amica = fitted_keep.amica_.transform(x)
    assert s_mne.shape == (N_KEEP, raw.n_times)
    np.testing.assert_allclose(s_mne, s_amica, rtol=1e-6, atol=1e-9)


def test_residual_export_is_logged(raw, caplog):
    """to_mne_ica says at INFO when it exports a residual, and how to opt out."""
    reduced = AMICAICA(random_state=SEED, device="cpu", verbose=False).fit(
        raw, stop=4096, max_iter=2, pcakeep=N_KEEP
    )
    full = AMICAICA(random_state=SEED, device="cpu", verbose=False).fit(
        raw, stop=4096, max_iter=2
    )
    with caplog.at_level(logging.INFO, logger="pamica.mne_compat.core"):
        full.to_mne_ica()
        assert not [r for r in caplog.records if "PCA residual" in r.getMessage()]
        reduced.to_mne_ica()
    messages = [
        r.getMessage() for r in caplog.records if "PCA residual" in r.getMessage()
    ]
    assert len(messages) == 1
    assert "12-dimensional PCA residual" in messages[0]
    assert f"n_pca_components={N_KEEP}" in messages[0]


# --- multi-model -------------------------------------------------------------
@pytest.fixture(scope="module")
def fitted_keep_2m(raw):
    """Two models with PCA reduction: per-model centers ``c`` are nonzero."""
    return AMICAICA(n_models=2, random_state=SEED, device="cpu", verbose=False).fit(
        raw, max_iter=MAX_ITER, pcakeep=N_KEEP
    )


def test_every_model_restores_the_residual(raw, fitted_keep_2m):
    """Each per-model export restores the residual, and the per-model ``c``
    fold into ``pca_mean_`` stays out of the residual rows."""
    backend = fitted_keep_2m.amica_.model_
    c = backend.c.cpu().numpy()
    mean = backend.mean.cpu().numpy().ravel()
    x = _data(raw, fitted_keep_2m)
    for h in range(2):
        assert np.any(c[:, h]), f"model {h} center c should be nonzero"
        ica = fitted_keep_2m.to_mne_ica(h)
        np.testing.assert_array_equal(
            ica.pca_components_, fitted_keep_2m.pca_components_
        )
        out = _data(fitted_keep_2m.apply(raw.copy(), model_idx=h), fitted_keep_2m)
        assert _rel(out, x) <= REL_TOL, f"model {h}"
        offset = ica.pca_mean_ - mean
        residual_rows = ica.pca_components_[N_KEEP:]
        assert np.linalg.norm(residual_rows @ offset) <= REL_TOL * np.linalg.norm(
            offset
        )
        _assert_exclude_removes_only_component(fitted_keep_2m, raw, 0, model_idx=h)
        s_mne = fitted_keep_2m.get_sources(raw, model_idx=h).get_data()
        s_amica = fitted_keep_2m.amica_.transform(
            x / fitted_keep_2m.pre_whitener_, model_idx=h
        )
        np.testing.assert_allclose(s_mne, s_amica, rtol=1e-6, atol=1e-9)


# --- Epochs --------------------------------------------------------------------
def test_epochs_apply_restores_the_residual(raw):
    epochs = mne.make_fixed_length_epochs(raw, duration=2.0, preload=True)
    fitted = AMICAICA(random_state=SEED, device="cpu", verbose=False).fit(
        epochs, max_iter=MAX_ITER, pcakeep=N_KEEP
    )
    assert fitted.pca_explained_variance_ is not None
    x = _data(epochs, fitted)
    out = _data(fitted.apply(epochs.copy()), fitted)
    assert _rel(out, x) <= REL_TOL
    reduced = fitted.apply(epochs.copy(), n_pca_components=fitted.n_components_)
    assert _rel(_data(reduced, fitted), x) > 1e-3
    np.testing.assert_allclose(
        fitted.pca_explained_variance_[N_KEEP:],
        _discarded_eigenvalues(x / fitted.pre_whitener_, N_KEEP),
        rtol=1e-8,
    )
    _assert_exclude_removes_only_component(fitted, epochs, 0)


# --- save / read_ica and explained-variance selection -------------------------
def test_saved_ica_restores_the_residual(raw, fitted_keep, tmp_path):
    ica = fitted_keep.to_mne_ica()
    fname = tmp_path / "amica-residual-ica.fif"
    ica.save(fname)
    reloaded = mne.preprocessing.read_ica(fname)
    assert reloaded.n_components_ == N_KEEP
    assert reloaded.pca_components_.shape == (32, 32)
    for exclude in ([], [0]):
        np.testing.assert_array_equal(
            reloaded.apply(raw.copy(), exclude=exclude).get_data(),
            ica.apply(raw.copy(), exclude=exclude).get_data(),
        )


def test_float_n_pca_components_selects_by_explained_variance(
    raw, fitted_keep, tmp_path
):
    """A fraction keeps the residual rows needed to reach it, through MNE's own
    cumulative-variance rule, and survives a save/read round trip."""
    ica = fitted_keep.to_mne_ica()
    n_keep = N_KEEP + 6
    cvar = np.cumsum(ica.pca_explained_variance_)
    cvar /= cvar[-1]
    # MNE keeps (cvar <= frac).sum() + 1 rows; a midpoint selects exactly n_keep.
    frac = float((cvar[n_keep - 2] + cvar[n_keep - 1]) / 2)
    by_frac = ica.apply(raw.copy(), n_pca_components=frac).get_data()
    by_count = ica.apply(raw.copy(), n_pca_components=n_keep).get_data()
    np.testing.assert_array_equal(by_frac, by_count)
    x = _data(raw, fitted_keep)
    assert _rel(by_count, x) > REL_TOL  # some residual rows were dropped
    reduced = ica.apply(raw.copy(), n_pca_components=N_KEEP).get_data()
    assert _rel(by_count, x) < _rel(reduced, x)  # but fewer than all of them

    stored = ica.copy()
    stored.n_pca_components = frac
    fname = tmp_path / "amica-frac-ica.fif"
    stored.save(fname)
    reloaded = mne.preprocessing.read_ica(fname)
    assert reloaded.n_pca_components == pytest.approx(frac)
    np.testing.assert_array_equal(reloaded.apply(raw.copy()).get_data(), by_count)


# --- mixed channel types -----------------------------------------------------
@pytest.fixture(scope="module")
def mixed_type_raw(raw):
    """Real EEG relabeled as magnetometers and gradiometers on MEG unit scales.

    Built as in ``test_mne_meg.py``: the recording's own signals, with only the
    declared type and the scale changed, so the pre-whitener is non-uniform.
    """
    data = raw.get_data()[:32]
    n = data.shape[0]
    types = ["mag" if i % 2 == 0 else "grad" for i in range(n)]
    names = [f"MEG{i:03d}" for i in range(n)]
    info = mne.create_info(names, raw.info["sfreq"], ch_types=types)
    scaled = data.copy()
    scaled[::2] *= 1e-13 / np.abs(data[::2]).max()
    scaled[1::2] *= 1e-11 / np.abs(data[1::2]).max()
    return mne.io.RawArray(scaled, info)


def test_mixed_channel_types_restore_the_residual_in_original_units(
    mixed_type_raw,
):
    fitted = AMICAICA(random_state=SEED, device="cpu", verbose=False).fit(
        mixed_type_raw, max_iter=MAX_ITER, pcakeep=N_KEEP
    )
    pw = fitted.pre_whitener_
    assert pw is not None and fitted.pca_explained_variance_ is not None
    assert len(np.unique(pw)) == 2, "one scale per channel type"
    x = _data(mixed_type_raw, fitted)
    out = _data(fitted.apply(mixed_type_raw.copy()), fitted)
    # Per type, since their units differ by ~1e2: each must come back whole.
    for sel in (slice(0, None, 2), slice(1, None, 2)):
        assert _rel(out[sel], x[sel]) <= REL_TOL
    reduced = fitted.apply(mixed_type_raw.copy(), n_pca_components=N_KEEP)
    assert _rel(_data(reduced, fitted), x) > 1e-3
    np.testing.assert_allclose(
        fitted.pca_explained_variance_[N_KEEP:],
        _discarded_eigenvalues(x / pw, N_KEEP),
        rtol=1e-8,
    )
    _assert_exclude_removes_only_component(fitted, mixed_type_raw, 0)


# --- automatic rank reduction ------------------------------------------------
def test_average_reference_rank_reduction_reconstructs(raw):
    """Average referencing leaves rank 31, detected automatically (no
    ``pcakeep``). The one residual direction carries only round-off variance,
    so this guards the zero-variance edge (clipping, no NaN) rather than
    discriminating old from new behavior."""
    avg = raw.copy().set_eeg_reference("average")
    fitted = AMICAICA(random_state=SEED, device="cpu", verbose=False).fit(
        avg, max_iter=MAX_ITER
    )
    assert fitted.n_components_ == 31
    ev = fitted.pca_explained_variance_
    assert ev is not None and ev.shape == (32,)
    assert 0.0 <= ev[-1] <= 1e-12 * ev[0]
    x = _data(avg, fitted)
    out = _data(fitted.apply(avg.copy()), fitted)
    assert _rel(out, x) <= REL_TOL
    _assert_exclude_removes_only_component(fitted, avg, 0)


# --- the pcadb trigger -------------------------------------------------------
def test_pcadb_reduction_restores_the_residual(raw):
    """``pcadb`` keeps the eigenvalues within ``pcadb`` dB of the largest; at
    30 dB that is 26 of the sample's 32 dimensions, leaving a 6-dimensional
    residual (``pcadb=20`` would keep 11, ``pcadb=40`` all 32)."""
    fitted = AMICAICA(random_state=SEED, device="cpu", verbose=False).fit(
        raw, max_iter=MAX_ITER, pcadb=30
    )
    n_kept = 26
    assert fitted.n_components_ == n_kept
    assert fitted.pca_components_ is not None
    assert fitted.pca_explained_variance_ is not None
    assert fitted.pca_components_.shape == (32, 32)
    x = _data(raw, fitted)
    assert _rel(_data(fitted.apply(raw.copy()), fitted), x) <= REL_TOL
    reduced = fitted.apply(raw.copy(), n_pca_components=n_kept)
    assert _rel(_data(reduced, fitted), x) > 1e-3
    np.testing.assert_allclose(
        fitted.pca_explained_variance_[n_kept:],
        _discarded_eigenvalues(x / fitted.pre_whitener_, n_kept),
        rtol=1e-8,
    )
    _assert_exclude_removes_only_component(fitted, raw, 0)
