"""AMICAICA MNE-wrapper tests (issue #139, single-model phase 1 + multi-model #141).

Real sample EEG only (no synthetic/mock): the bundled EEGLAB ``eeglab_data.set``
(32 channels, 30504 samples) is loaded through ``mne.io.read_raw_eeglab`` -- the
same entry point a real MNE user would use. The whole module self-skips when the
optional ``mne`` extra is absent, so the base CI env (no mne) skips it; the
dedicated ``test-mne`` CI job installs ``pamica[mne]`` and runs it for real.

The load-bearing check is the round trip: the ICA that ``to_mne_ica`` builds must
reproduce ``AMICA.transform`` bit-for-bit (up to float64 eigh residual), which
pins the mean/sphere/unmixing -> pca_mean_/pca_components_/unmixing_matrix_ map.
"""

from pathlib import Path

import numpy as np
import pytest

mne = pytest.importorskip("mne")

from pamica.mne_compat import AMICAICA, PDFTYPE_NAMES  # noqa: E402  (after importorskip)

mne.set_log_level("ERROR")

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "sample_data"
SET_FILE = SAMPLE_DIR / "eeglab_data.set"
SEED = 42
MAX_ITER = 12  # enough to move off the init; parity is orientation, not convergence

pytestmark = pytest.mark.skipif(
    not SET_FILE.exists(), reason="sample eeglab_data.set missing"
)


@pytest.fixture(scope="module")
def raw():
    """Real continuous EEG as an MNE Raw (32 EEG channels, 128 Hz)."""
    return mne.io.read_raw_eeglab(str(SET_FILE), preload=True)


@pytest.fixture(scope="module")
def fitted(raw):
    """A single AMICAICA fit reused across the read-only assertions."""
    return AMICAICA(n_mix=3, random_state=SEED, device="cpu", verbose=False).fit(
        raw, max_iter=MAX_ITER
    )


def _picked_data(raw, fitted=None, picks="data"):
    """The exact array AMICAICA fits on.

    Good data channels, float64, divided by the fit's channel-type pre-whitener
    (issue #225) -- the same scaling MNE's own ICA applies before PCA. Pass the
    fitted wrapper to apply it; omit it only when the raw, unscaled array is
    what is under test.
    """
    x = raw.copy().pick(picks, exclude="bads").get_data().astype(np.float64)
    if fitted is not None:
        x = x / fitted.pre_whitener_
    return x


# --- the mapping crux -------------------------------------------------------
def test_get_sources_matches_amica_transform(raw, fitted):
    """to_mne_ica().get_sources == AMICA.transform on the same data."""
    s_mne = fitted.to_mne_ica().get_sources(raw).get_data()
    s_amica = fitted.amica_.transform(_picked_data(raw, fitted))
    assert s_mne.shape == s_amica.shape == (fitted.n_components_, raw.n_times)
    np.testing.assert_allclose(s_mne, s_amica, rtol=1e-6, atol=1e-9)


def test_get_components_are_channel_space_maps(fitted):
    """get_components() equals the channel-space mixing inv(sphere) @ inv(W)."""
    comps = fitted.get_components()
    assert comps.shape == (fitted.n_components_, fitted.n_components_)
    sphere = fitted.amica_.model_.sphere.cpu().numpy()
    w_fort = fitted.amica_.get_unmixing_matrix(0)
    maps_ref = np.linalg.inv(sphere) @ np.linalg.inv(w_fort)
    np.testing.assert_allclose(comps, maps_ref, rtol=1e-6, atol=1e-10)


def test_apply_without_exclude_reconstructs(raw, fitted):
    """apply() with no excluded components returns the input unchanged."""
    recon = fitted.apply(raw.copy())
    np.testing.assert_allclose(recon.get_data(), raw.get_data(), rtol=1e-6, atol=1e-12)


def test_apply_excludes_component(raw, fitted):
    """Excluding a component changes the data and zeroes that source."""
    ica = fitted.to_mne_ica()
    cleaned = ica.apply(raw.copy(), exclude=[0])
    # The reconstruction must differ from the input (a component was removed).
    assert not np.allclose(cleaned.get_data(), raw.get_data())
    # Re-deriving sources from the cleaned data: component 0 is gone.
    src_after = ica.get_sources(cleaned).get_data()
    assert np.abs(src_after[0]).max() < 1e-6 * np.abs(src_after).max()


# --- object validity --------------------------------------------------------
def test_to_mne_ica_returns_valid_fitted_ica(raw, fitted):
    ica = fitted.to_mne_ica()
    assert isinstance(ica, mne.preprocessing.ICA)
    assert ica.current_fit != "unfitted"
    assert ica.n_components_ == fitted.n_components_
    assert ica.ch_names == fitted.ch_names_
    # A genuine MNE ICA exposes the full component interface.
    assert ica.get_components().shape == (fitted.n_components_, fitted.n_components_)


def test_export_is_cached_and_invalidated_on_refit(raw):
    ica = AMICAICA(random_state=SEED, device="cpu", verbose=False).fit(
        raw, max_iter=MAX_ITER
    )
    first = ica.to_mne_ica()
    assert ica.to_mne_ica() is first  # cached
    ica.fit(raw, max_iter=MAX_ITER)  # refit invalidates the cache
    assert ica.to_mne_ica() is not first


# --- picks / epochs / plotting ----------------------------------------------
def test_fit_with_channel_subset(raw):
    picks = raw.ch_names[:10]
    ica = AMICAICA(random_state=SEED, device="cpu", verbose=False).fit(
        raw, picks=picks, max_iter=MAX_ITER
    )
    assert ica.n_components_ == 10
    assert ica.ch_names_ == picks
    s_mne = ica.to_mne_ica().get_sources(raw).get_data()
    x = raw.copy().pick(picks).get_data().astype(np.float64) / ica.pre_whitener_
    assert ica.amica_ is not None
    np.testing.assert_allclose(s_mne, ica.amica_.transform(x), rtol=1e-6, atol=1e-9)


def test_fit_from_epochs(raw):
    epochs = mne.make_fixed_length_epochs(raw, duration=2.0, preload=True)
    ica = AMICAICA(random_state=SEED, device="cpu", verbose=False).fit(
        epochs, max_iter=MAX_ITER
    )
    assert ica._fit_kind == "epochs"
    src = ica.to_mne_ica().get_sources(epochs).get_data()  # (n_ep, n_comp, n_time)
    x = np.hstack(epochs.copy().pick("data", exclude="bads").get_data())
    x = x / ica.pre_whitener_
    assert ica.amica_ is not None
    np.testing.assert_allclose(
        np.hstack(src), ica.amica_.transform(x), rtol=1e-6, atol=1e-9
    )


def test_plot_components_returns_figure(fitted):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.figure
    import matplotlib.pyplot as plt

    out = fitted.plot_components(picks=[0, 1], show=False)
    figs = out if isinstance(out, list) else [out]
    assert figs and all(isinstance(f, matplotlib.figure.Figure) for f in figs)
    plt.close("all")


def test_pca_components_orthonormal_and_variance_ordered(raw, fitted):
    """The eigenbasis ordering (invisible to the V-cancelling round trip) is real.

    get_sources/get_components reduce to W@sphere and inv(sphere)@inv(W) for ANY
    orthonormal V, so they cannot see a wrong eigenvector order. Pin it here:
    pca_components_ must be orthonormal and its rows ordered by descending data
    variance, with pca_explained_variance_ equal to that per-row variance.
    """
    ica = fitted.to_mne_ica()
    p = ica.pca_components_
    np.testing.assert_allclose(p @ p.T, np.eye(p.shape[0]), atol=1e-10)
    ev = ica.pca_explained_variance_
    assert np.all(np.diff(ev) <= 0), "explained variance must be descending"
    # Independent oracle: project the mean-centered data onto pca_components_;
    # each row's variance is that data-covariance eigenvalue, in the same order.
    x = _picked_data(raw, fitted)
    proj = p @ (x - ica.pca_mean_[:, None])
    np.testing.assert_allclose(proj.var(axis=1), ev, rtol=1e-5)


def test_fit_with_start_stop_subranges_raw(raw):
    stop = 5000
    ica = AMICAICA(random_state=SEED, device="cpu", verbose=False).fit(
        raw, start=100, stop=stop, max_iter=MAX_ITER
    )
    assert ica.to_mne_ica().n_samples_ == stop - 100
    # A different range yields a different decomposition (the range is honored).
    full = AMICAICA(random_state=SEED, device="cpu", verbose=False).fit(
        raw, max_iter=MAX_ITER
    )
    assert not np.allclose(ica.get_components(), full.get_components())


def test_apply_on_epochs(raw):
    epochs = mne.make_fixed_length_epochs(raw, duration=2.0, preload=True)
    ica = AMICAICA(random_state=SEED, device="cpu", verbose=False).fit(
        epochs, max_iter=MAX_ITER
    )
    recon = ica.apply(epochs.copy())
    np.testing.assert_allclose(
        recon.get_data(), epochs.get_data(), rtol=1e-6, atol=1e-12
    )
    cleaned = ica.apply(epochs.copy(), exclude=[0])
    assert not np.allclose(cleaned.get_data(), epochs.get_data())


def test_exported_ica_saves_and_reloads(fitted, raw, tmp_path):
    """reject_/n_samples_ must be set, else ICA.save() raises AttributeError."""
    ica = fitted.to_mne_ica()
    fname = tmp_path / "amica-ica.fif"
    ica.save(fname)
    reloaded = mne.preprocessing.read_ica(fname)
    np.testing.assert_allclose(
        reloaded.get_sources(raw).get_data(),
        ica.get_sources(raw).get_data(),
        rtol=1e-6,
        atol=1e-9,
    )


def test_plot_sources_returns_figure(fitted, raw):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.figure
    import matplotlib.pyplot as plt

    fig = fitted.plot_sources(raw, show=False)
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close("all")


def test_seed_determinism(raw):
    a = AMICAICA(random_state=1, device="cpu", verbose=False).fit(
        raw, max_iter=MAX_ITER
    )
    b = AMICAICA(random_state=1, device="cpu", verbose=False).fit(
        raw, max_iter=MAX_ITER
    )
    np.testing.assert_allclose(a.get_components(), b.get_components())
    c = AMICAICA(random_state=2, device="cpu", verbose=False).fit(
        raw, max_iter=MAX_ITER
    )
    assert not np.allclose(a.get_components(), c.get_components())


# --- guards -----------------------------------------------------------------
def test_unfitted_calls_raise():
    ica = AMICAICA()
    with pytest.raises(ValueError, match="must be fitted"):
        ica.to_mne_ica()


def test_fit_rejects_non_mne_input():
    with pytest.raises(TypeError, match="mne.io.Raw or mne.Epochs"):
        AMICAICA().fit(np.random.randn(8, 500))


def test_fit_rejects_start_stop_on_epochs(raw):
    epochs = mne.make_fixed_length_epochs(raw, duration=2.0, preload=True)
    with pytest.raises(ValueError, match="only supported for Raw"):
        AMICAICA().fit(epochs, stop=100)


def test_fit_rejects_out_of_range_stop(raw):
    with pytest.raises(ValueError, match="exceeds the recording length"):
        AMICAICA().fit(raw, stop=raw.n_times + 1)


def test_fit_rejects_non_finite_data(raw):
    bad = raw.copy()
    bad._data[3, 100:200] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        AMICAICA(device="cpu", verbose=False).fit(bad, max_iter=MAX_ITER)


def test_pca_reduction_is_supported(raw):
    """Formerly rejected outright; the export now builds pca_components_ from a
    non-square sphere, so an explicitly reduced fit round-trips (issue #225)."""
    fitted = AMICAICA(device="cpu", verbose=False).fit(raw, max_iter=5, pcakeep=10)
    assert fitted.amica_ is not None and fitted.ch_names_ is not None
    assert fitted.n_components_ == 10
    ica = fitted.to_mne_ica()
    assert ica.pca_components_.shape == (10, len(fitted.ch_names_))
    # Orthonormal rows are what make MNE's get_components/apply valid.
    np.testing.assert_allclose(
        ica.pca_components_ @ ica.pca_components_.T, np.eye(10), atol=1e-10
    )
    s_mne = ica.get_sources(raw).get_data()
    s_amica = fitted.amica_.transform(_picked_data(raw, fitted))
    assert s_mne.shape == (10, raw.n_times)
    np.testing.assert_allclose(s_mne, s_amica, rtol=1e-6, atol=1e-9)


def test_degenerate_fit_is_refused_and_labeled(raw):
    """A degenerate (non-converged) fit is kept for inspection but refused.

    Real divergence is platform-dependent, so mark the wrapper's own
    convergence signal (the documented contract) to exercise the guard
    deterministically: `__repr__` labels it and the consumer methods refuse it.
    """
    ica = AMICAICA(random_state=SEED, device="cpu", verbose=False).fit(
        raw, max_iter=MAX_ITER
    )
    ica.converged_ = False
    ica.stop_reason_ = "nan_ll"
    assert "degenerate" in repr(ica)
    with pytest.raises(RuntimeError, match="degenerate"):
        ica.to_mne_ica()


def test_failed_refit_leaves_prior_state_intact(raw):
    """An exception mid-refit must not corrupt a previously fitted wrapper."""
    ica = AMICAICA(random_state=SEED, device="cpu", verbose=False).fit(
        raw, max_iter=MAX_ITER
    )
    before = ica.get_components()
    with pytest.raises(ValueError):  # stop past the end aborts before publishing
        ica.fit(raw, picks=raw.ch_names[:10], stop=raw.n_times + 1, max_iter=MAX_ITER)
    assert ica.n_components_ == before.shape[0]  # still the 32-channel fit
    np.testing.assert_array_equal(ica.get_components(), before)


# --- multi-model (issue #141) ----------------------------------------------
@pytest.fixture(scope="module")
def fitted_2m(raw):
    """A two-model AMICAICA fit reused across the multi-model assertions."""
    return AMICAICA(
        n_models=2, n_mix=3, random_state=SEED, device="cpu", verbose=False
    ).fit(raw, max_iter=MAX_ITER)


def test_per_model_export_roundtrips_with_nonzero_c(raw, fitted_2m):
    """Each model exports its own ICA reproducing that model's transform.

    Unlike the single-model case, the per-model center c is nonzero, so this
    exercises the inv(sphere)@c fold into pca_mean_ that phase 1 deferred.
    """
    x = _picked_data(raw, fitted_2m)
    c = fitted_2m.amica_.model_.c.cpu().numpy()
    for h in range(2):
        assert np.any(c[:, h]), f"model {h} center c should be nonzero"
        s_mne = fitted_2m.get_sources(raw, model_idx=h).get_data()
        np.testing.assert_allclose(
            s_mne, fitted_2m.amica_.transform(x, model_idx=h), rtol=1e-6, atol=1e-9
        )


def test_per_model_export_is_distinct_and_cached(fitted_2m):
    ica0, ica1 = fitted_2m.to_mne_ica(0), fitted_2m.to_mne_ica(1)
    assert ica0 is not ica1
    assert fitted_2m.to_mne_ica(0) is ica0  # cached per model
    assert not np.allclose(ica0.unmixing_matrix_, ica1.unmixing_matrix_)


def test_per_model_apply_reconstructs(raw, fitted_2m):
    for h in range(2):
        recon = fitted_2m.apply(raw.copy(), model_idx=h)
        np.testing.assert_allclose(
            recon.get_data(), raw.get_data(), rtol=1e-6, atol=1e-12
        )


def test_per_model_apply_exclude_differs_by_model(raw, fitted_2m):
    """apply must honor model_idx: excluding a component of different models
    removes different signals (guards against model_idx being ignored)."""
    c0 = fitted_2m.apply(raw.copy(), model_idx=0, exclude=[0]).get_data()
    c1 = fitted_2m.apply(raw.copy(), model_idx=1, exclude=[0]).get_data()
    assert not np.allclose(c0, c1)


def test_per_model_get_components_and_plot(fitted_2m):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.figure
    import matplotlib.pyplot as plt

    maps0 = fitted_2m.get_components(model_idx=0)
    maps1 = fitted_2m.get_components(model_idx=1)
    assert not np.allclose(maps0, maps1)  # per-model scalp maps differ
    out = fitted_2m.plot_components(model_idx=1, picks=[0, 1], show=False)
    figs = out if isinstance(out, list) else [out]
    assert all(isinstance(f, matplotlib.figure.Figure) for f in figs)
    plt.close("all")


def test_model_probability_is_normalized(raw, fitted_2m):
    prob = fitted_2m.get_model_probability(raw)
    assert prob.shape == (2, raw.n_times)
    np.testing.assert_allclose(prob.sum(axis=0), 1.0, atol=1e-10)
    assert np.all(prob >= 0.0) and np.all(prob <= 1.0)


def test_single_model_probability_is_all_ones(raw, fitted):
    np.testing.assert_allclose(fitted.get_model_probability(raw), 1.0, atol=1e-10)


def test_model_probability_and_plot_on_epochs(raw, fitted_2m):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = mne.make_fixed_length_epochs(raw, duration=2.0, preload=True)
    n = len(epochs) * len(epochs.times)
    prob = fitted_2m.get_model_probability(epochs)
    assert prob.shape == (2, n)  # concatenated along time
    np.testing.assert_allclose(prob.sum(axis=0), 1.0, atol=1e-10)
    fig = fitted_2m.plot_model_probability(epochs)
    assert fig is not None
    plt.close("all")


def test_plot_model_probability_default_srate_gives_seconds(raw, fitted_2m):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.figure
    import matplotlib.pyplot as plt

    fig = fitted_2m.plot_model_probability(raw)
    assert isinstance(fig, matplotlib.figure.Figure)
    ax = fig.axes[1]
    assert ax.get_xlabel() == "Time (s)"
    # srate default must be the fitted recording's rate, not just any non-None:
    # the x-data must equal sample-index / sfreq (catches a wrong/stale binding).
    expected_t = np.arange(raw.n_times) / raw.info["sfreq"]
    np.testing.assert_allclose(
        np.asarray(ax.lines[0].get_xdata(), dtype=float), expected_t
    )
    plt.close("all")


def test_out_of_range_model_idx_raises(fitted_2m):
    with pytest.raises(ValueError, match="out of range"):
        fitted_2m.to_mne_ica(model_idx=2)
    with pytest.raises(ValueError, match="out of range"):
        fitted_2m.get_sources(None, model_idx=-1)
    with pytest.raises(TypeError, match="model_idx must be an int"):
        fitted_2m.to_mne_ica(model_idx=1.0)


def test_get_model_probability_rejects_non_mne_input(fitted_2m):
    with pytest.raises(TypeError, match="Raw or mne.Epochs"):
        fitted_2m.get_model_probability(np.zeros((32, 100)))


def test_degenerate_fit_refused_on_dominance_methods(raw):
    ica = AMICAICA(n_models=2, random_state=SEED, device="cpu", verbose=False).fit(
        raw, max_iter=MAX_ITER
    )
    ica.converged_ = False
    ica.stop_reason_ = "nan_ll"
    with pytest.raises(RuntimeError, match="degenerate"):
        ica.get_model_probability(raw)
    with pytest.raises(RuntimeError, match="degenerate"):
        ica.plot_model_probability(raw)


# --- pamica-specific metadata (issue #142) ----------------------------------
def test_get_pdftype_and_rho_per_model(fitted_2m):
    for h in range(2):
        pdf = fitted_2m.get_pdftype(model_idx=h)
        assert pdf.shape == (fitted_2m.n_components_,)
        assert np.array_equal(np.unique(pdf), [0])  # default generalized Gaussian
        assert PDFTYPE_NAMES[int(pdf[0])] == "generalized_gaussian"
        rho = fitted_2m.get_rho(model_idx=h)
        assert rho.shape == (fitted_2m.n_mix, fitted_2m.n_components_)
        assert np.all(np.isfinite(rho))


def test_shared_components_empty_by_default(fitted_2m):
    assert fitted_2m.shared_components() == []


def test_metadata_respects_model_idx_bounds(fitted_2m):
    with pytest.raises(ValueError, match="out of range"):
        fitted_2m.get_pdftype(model_idx=2)
    with pytest.raises(ValueError, match="out of range"):
        fitted_2m.get_rho(model_idx=5)


def test_metadata_requires_fit():
    ica = AMICAICA()
    with pytest.raises(ValueError, match="must be fitted"):
        ica.get_pdftype()
    with pytest.raises(ValueError, match="must be fitted"):
        ica.shared_components()


# --- separation-quality metrics (issue #143) --------------------------------
def test_mir_matches_amica_per_model(raw, fitted_2m):
    x = _picked_data(raw, fitted_2m)
    mirs = []
    for h in range(2):
        m = fitted_2m.mir(raw, model_idx=h)
        np.testing.assert_allclose(m, fitted_2m.amica_.mir(x, model_idx=h))
        mirs.append(m[0])
    assert not np.isclose(mirs[0], mirs[1])  # model_idx honored, not hardcoded to 0


def test_pmi_matches_amica_per_model(raw, fitted_2m):
    x = _picked_data(raw, fitted_2m)
    pmis = []
    for h in range(2):
        pmi_w = fitted_2m.pmi(raw, model_idx=h)
        assert pmi_w.shape == (fitted_2m.n_components_, fitted_2m.n_components_)
        np.testing.assert_allclose(pmi_w, fitted_2m.amica_.pmi(x, model_idx=h))
        pmis.append(pmi_w)
    assert not np.allclose(pmis[0], pmis[1])  # per-model, not hardcoded to 0


def test_mir_pmi_on_epochs(raw, fitted_2m):
    epochs = mne.make_fixed_length_epochs(raw, duration=2.0, preload=True)
    x = np.hstack(epochs.copy().pick("data", exclude="bads").get_data())
    x = x / fitted_2m.pre_whitener_
    np.testing.assert_allclose(fitted_2m.mir(epochs), fitted_2m.amica_.mir(x))
    np.testing.assert_allclose(fitted_2m.pmi(epochs), fitted_2m.amica_.pmi(x))


def test_mir_pmi_nbins_passthrough(raw, fitted_2m):
    """nbins must reach the metric: a non-default nbins matches AMICA with the
    same nbins (a dropped forward would silently fall back to the default)."""
    x = _picked_data(raw, fitted_2m)
    np.testing.assert_allclose(
        fitted_2m.mir(raw, nbins=50), fitted_2m.amica_.mir(x, nbins=50)
    )
    np.testing.assert_allclose(
        fitted_2m.pmi(raw, nbins=50), fitted_2m.amica_.pmi(x, nbins=50)
    )


def test_mir_pmi_respect_model_idx_bounds(fitted_2m, raw):
    with pytest.raises(ValueError, match="out of range"):
        fitted_2m.mir(raw, model_idx=2)
    with pytest.raises(ValueError, match="out of range"):
        fitted_2m.pmi(raw, model_idx=-1)


def test_mir_pmi_require_fit(raw):
    ica = AMICAICA()
    with pytest.raises(ValueError, match="must be fitted"):
        ica.mir(raw)
    with pytest.raises(ValueError, match="must be fitted"):
        ica.pmi(raw)


def test_mir_pmi_refuse_degenerate(raw):
    ica = AMICAICA(n_models=2, random_state=SEED, device="cpu", verbose=False).fit(
        raw, max_iter=MAX_ITER
    )
    ica.converged_ = False
    ica.stop_reason_ = "nan_ll"
    with pytest.raises(RuntimeError, match="degenerate"):
        ica.mir(raw)
    with pytest.raises(RuntimeError, match="degenerate"):
        ica.pmi(raw)
