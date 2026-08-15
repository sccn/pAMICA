"""Mixed-channel-type and rank-deficient handling in the MNE wrapper (issue #225).

Signals are the real bundled EEG throughout. What is constructed is the
*channel-type metadata* and the unit scaling -- the two properties of Elekta/MEGIN
MEG that break a naive fit, and which the bundled EEG cannot supply on its own
(pamica ships no MEG recording). The measurements below are therefore about the
scaling and rank machinery, not about MEG source separation quality; no claim is
made against a MEG oracle.

Reported in #221: 306 channels (102 magnetometers + 204 gradiometers) at
numerical rank ~70 after Maxwell filtering.
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
MAX_ITER = 8

pytestmark = pytest.mark.skipif(
    not SET_FILE.exists(), reason="sample eeglab_data.set missing"
)


@pytest.fixture(scope="module")
def raw():
    return mne.io.read_raw_eeglab(str(SET_FILE), preload=True)


@pytest.fixture(scope="module")
def mixed_type_raw(raw):
    """Real EEG relabelled as two channel types on MEG-like unit scales.

    Half the channels become magnetometers (~1e-13 T), half gradiometers
    (~1e-11 T/m). The signals are the recording's own; only the declared type and
    the scale differ, which is exactly the condition that makes an unscaled fit
    follow one channel type.
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


def test_pre_whitener_is_one_scale_per_channel_type(mixed_type_raw):
    """MNE's convention: one std per type, broadcast across that type."""
    fitted = AMICAICA(random_state=SEED, device="cpu", verbose=False).fit(
        mixed_type_raw, max_iter=MAX_ITER
    )
    assert fitted.amica_ is not None
    pw = fitted.pre_whitener_
    assert pw is not None
    assert pw.shape == (32, 1)
    mags, grads = pw[::2, 0], pw[1::2, 0]
    assert len(set(mags)) == 1, "one scale for all magnetometers"
    assert len(set(grads)) == 1, "one scale for all gradiometers"
    # The two types differ by ~1e2 in these units; without scaling the
    # gradiometers would dominate the covariance entirely.
    assert grads[0] / mags[0] > 10


def test_scaling_equalizes_channel_type_variance(mixed_type_raw):
    """After the pre-whitener the two types contribute comparable variance."""
    fitted = AMICAICA(random_state=SEED, device="cpu", verbose=False).fit(
        mixed_type_raw, max_iter=MAX_ITER
    )
    assert fitted.amica_ is not None
    x = mixed_type_raw.get_data() / fitted.pre_whitener_
    ratio = np.std(x[1::2]) / np.std(x[::2])
    assert 0.1 < ratio < 10, f"types still unbalanced after scaling: {ratio:.3g}"


def test_roundtrip_holds_for_mixed_channel_types(mixed_type_raw):
    """The load-bearing check, with a non-unit pre-whitener in the path."""
    fitted = AMICAICA(random_state=SEED, device="cpu", verbose=False).fit(
        mixed_type_raw, max_iter=MAX_ITER
    )
    assert fitted.amica_ is not None
    s_mne = fitted.to_mne_ica().get_sources(mixed_type_raw).get_data()
    x = mixed_type_raw.get_data().astype(np.float64) / fitted.pre_whitener_
    np.testing.assert_allclose(s_mne, fitted.amica_.transform(x), rtol=1e-6, atol=1e-9)


def test_eeg_only_scaling_is_absorbed_by_the_sphere(raw):
    """A single channel type means a global rescale, which sphering absorbs.

    For ``X/s`` the covariance scales by ``1/s^2`` and the sphere by ``s``, so the
    sphered data -- everything AMICA sees after preprocessing -- is unchanged.
    That exact claim is asserted on the sphere itself (the ratio between the two
    fits' spheres is a single constant to ~1e-15).

    The resulting sources agree to ~1e-5 relative rather than exactly: the
    eigendecomposition round-off differs between the two scales and EM amplifies
    it over the iterations. Hence a tolerance, not a bit-for-bit assertion.
    """
    from pamica import AMICA

    fitted = AMICAICA(random_state=SEED, device="cpu", verbose=False).fit(
        raw, max_iter=MAX_ITER
    )
    assert fitted.amica_ is not None
    x_raw = raw.copy().pick("data", exclude="bads").get_data().astype(np.float64)

    direct = AMICA(device="cpu", verbose=False)
    direct.fit(x_raw, max_iter=MAX_ITER, seed=SEED)
    assert direct.model_ is not None

    # The exact claim: sphere_scaled == s * sphere_unscaled, so their ratio is
    # one constant across every entry.
    assert fitted.amica_.model_ is not None
    assert fitted.amica_.model_.sphere is not None
    assert direct.model_.sphere is not None
    ratio = (
        fitted.amica_.model_.sphere.cpu().numpy() / direct.model_.sphere.cpu().numpy()
    )
    assert ratio.std() < 1e-12, "sphering did not absorb the rescale"

    scaled = fitted.amica_.transform(x_raw / fitted.pre_whitener_)
    unscaled = direct.transform(x_raw)
    rel = np.abs(scaled - unscaled).max() / np.abs(unscaled).max()
    assert rel < 1e-4, f"sources diverged beyond round-off: {rel:.3e}"
    corr = min(
        abs(np.corrcoef(scaled[i], unscaled[i])[0, 1]) for i in range(scaled.shape[0])
    )
    assert corr > 0.9999, f"worst per-source correlation {corr:.6f}"


# --- rank-deficient input (the #221 case) -----------------------------------
@pytest.fixture(scope="module")
def rank_deficient_raw(raw):
    """Real EEG projected onto its top-20 subspace, as SSS does to MEG."""
    data = raw.get_data()[:32]
    centered = data - data.mean(axis=1, keepdims=True)
    U = np.linalg.svd(centered, full_matrices=False)[0][:, :20]
    info = mne.create_info(
        [f"EEG{i:03d}" for i in range(32)], raw.info["sfreq"], ch_types="eeg"
    )
    return mne.io.RawArray(U @ (U.T @ centered), info)


def test_rank_deficient_fit_exports_and_roundtrips(rank_deficient_raw):
    """Formerly impossible twice over: the fit went degenerate, and the export
    called eigh on a non-square sphere."""
    fitted = AMICAICA(random_state=SEED, device="cpu", verbose=False).fit(
        rank_deficient_raw, max_iter=MAX_ITER
    )
    assert fitted.amica_ is not None
    assert fitted.converged_, fitted.stop_reason_
    assert fitted.n_components_ == 20

    ica = fitted.to_mne_ica()
    assert ica.pca_components_.shape == (20, 32)
    np.testing.assert_allclose(
        ica.pca_components_ @ ica.pca_components_.T, np.eye(20), atol=1e-10
    )
    assert np.all(np.diff(ica.pca_explained_variance_) <= 0)

    s_mne = ica.get_sources(rank_deficient_raw).get_data()
    x = rank_deficient_raw.get_data().astype(np.float64) / fitted.pre_whitener_
    assert s_mne.shape == (20, rank_deficient_raw.n_times)
    np.testing.assert_allclose(s_mne, fitted.amica_.transform(x), rtol=1e-6, atol=1e-9)


def test_rank_deficient_components_are_sensor_shaped(rank_deficient_raw):
    """Scalp maps stay (n_channels, n_components) so MNE can plot them."""
    fitted = AMICAICA(random_state=SEED, device="cpu", verbose=False).fit(
        rank_deficient_raw, max_iter=MAX_ITER
    )
    assert fitted.amica_ is not None
    assert fitted.get_components().shape == (32, 20)


def test_rank_deficient_apply_reconstructs(rank_deficient_raw):
    """apply() with nothing excluded must return the input, through the
    reduced-rank PCA basis."""
    fitted = AMICAICA(random_state=SEED, device="cpu", verbose=False).fit(
        rank_deficient_raw, max_iter=MAX_ITER
    )
    assert fitted.amica_ is not None
    out = fitted.apply(rank_deficient_raw.copy())
    original = rank_deficient_raw.get_data()
    rel = np.abs(out.get_data() - original).max() / np.abs(original).max()
    assert rel < 1e-6, f"reconstruction error {rel:.3e}"
