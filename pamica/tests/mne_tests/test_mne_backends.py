"""``AMICAICA`` on the MLX backend (issue #313, epic #324 Phase 4).

``AMICAICA(backend="mlx")`` fits through ``AMICA(backend="mlx")`` and exports
through the same backend-agnostic float64 accessors as the PyTorch default
(``get_sphere``/``get_mean``/``get_model_center``). MLX computes in float32, so
the export is float32-consistent rather than float64-parity: sources and
component back-projections agree with the backend's own float32 ``transform``
to float32 tolerance, while reconstruction (``apply`` with nothing excluded,
PCA residual included) is float64 round-off, because MNE's mixing is the
float64 pseudo-inverse of the exported unmixing and the PCA basis is
orthonormal. These tests pin both.

Real sample EEG only: the bundled EEGLAB ``eeglab_data.set`` (32 channels,
30504 samples) through ``mne.io.read_raw_eeglab``, as in the other
``mne_tests`` modules. MLX tests skip individually without MLX or an Apple GPU;
the construction checks run everywhere.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

mne = pytest.importorskip("mne")

from pamica.mne_compat import AMICAICA  # noqa: E402  (after importorskip)

mne.set_log_level("ERROR")

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "sample_data"
SET_FILE = SAMPLE_DIR / "eeglab_data.set"
SEED = 42
MAX_ITER = 10  # the export is a function of the fitted state, not convergence
N_KEEP = 20
NW = 32

# Reconstruction and anything ICA never touches: float64 round-off (measured
# 1.4e-15 for apply(), 1.8e-15 for the residual leak of an exclusion).
REL_TOL = 1e-10
# What passes through MLX's float32 fit: sources against the float32
# transform (measured 2.0e-7 reduced, 3.0e-7 full rank) and an exclusion
# against its float32 oracle (measured 1.9e-7).
FLOAT32_REL_TOL = 1e-5

pytestmark = pytest.mark.skipif(
    not SET_FILE.exists(), reason="sample eeglab_data.set missing"
)


def _require_mlx():
    mlx_core = pytest.importorskip(
        "pamica.mlx_impl.core", reason="MLX not installed (Apple Silicon only)"
    )
    mx = mlx_core.mx
    if mx.default_device().type != mx.DeviceType.gpu:
        pytest.skip("no Apple GPU")
    return mlx_core.AMICAMLXNG


def _rel(a, b):
    """Frobenius-norm relative difference of ``a`` from ``b``."""
    return np.linalg.norm(a - b) / np.linalg.norm(b)


@pytest.fixture(scope="module")
def raw():
    """Real continuous EEG as an MNE Raw (32 EEG channels, 128 Hz)."""
    return mne.io.read_raw_eeglab(str(SET_FILE), preload=True)


@pytest.fixture(scope="module")
def mlx_keep(raw):
    """An MLX fit reduced to 20 of 32 dimensions by an explicit pcakeep."""
    _require_mlx()
    return AMICAICA(random_state=SEED, verbose=False, backend="mlx").fit(
        raw, max_iter=MAX_ITER, pcakeep=N_KEEP
    )


@pytest.fixture(scope="module")
def mlx_full(raw):
    _require_mlx()
    return AMICAICA(random_state=SEED, verbose=False, backend="mlx").fit(
        raw, max_iter=MAX_ITER
    )


def _data(inst, fitted):
    """The fitted channels of ``inst`` in original units."""
    return inst.get_data(picks=fitted.ch_names_)


# --- construction ---------------------------------------------------------------
def test_backend_is_validated_like_amica():
    with pytest.raises(ValueError, match="backend must be one of 'torch', 'mlx'"):
        AMICAICA(backend="jax")
    with pytest.raises(ValueError, match="applies only to backend='torch'"):
        AMICAICA(backend="mlx", device="cpu")
    assert AMICAICA().backend == "torch"


def test_mlx_without_mlx_raises_import_error(monkeypatch):
    """``sys.modules[name] = None`` is Python's marker for an unimportable
    module, so the real availability check sees what a machine without MLX
    shows it."""
    monkeypatch.setitem(sys.modules, "mlx", None)
    with pytest.raises(ImportError, match="uv pip install mlx"):
        AMICAICA(backend="mlx")


def test_mlx_rejects_a_dtype(raw):
    _require_mlx()
    ica = AMICAICA(random_state=SEED, verbose=False, backend="mlx")
    with pytest.raises(ValueError, match="apply only to backend='torch'"):
        ica.fit(raw, stop=2048, max_iter=1, dtype=torch.float32)
    assert ica.amica_ is None


# --- the MLX fit and its export ---------------------------------------------------
def test_mlx_fit_builds_the_mlx_backend(mlx_keep):
    AMICAMLXNG = _require_mlx()
    assert mlx_keep.backend == "mlx"
    assert mlx_keep.amica_ is not None and mlx_keep.amica_.backend == "mlx"
    assert type(mlx_keep.amica_.model_) is AMICAMLXNG
    assert mlx_keep.converged_ is True
    assert mlx_keep.n_components_ == N_KEEP
    assert mlx_keep.pca_components_ is not None
    assert mlx_keep.pca_components_.shape == (NW, NW)
    ica = mlx_keep.to_mne_ica()
    assert ica.n_components_ == N_KEEP
    assert ica.get_components().shape == (NW, N_KEEP)


@pytest.mark.parametrize("fixture", ["mlx_keep", "mlx_full"])
def test_get_sources_matches_the_mlx_transform(raw, fixture, request):
    fitted = request.getfixturevalue(fixture)
    s_mne = fitted.get_sources(raw).get_data()
    s_mlx = fitted.amica_.transform(_data(raw, fitted) / fitted.pre_whitener_)
    assert s_mne.shape == s_mlx.shape == (fitted.n_components_, raw.n_times)
    assert _rel(s_mne, s_mlx) <= FLOAT32_REL_TOL


@pytest.mark.parametrize("fixture", ["mlx_keep", "mlx_full"])
def test_apply_reproduces_the_input(raw, fixture, request):
    """Nothing excluded: the input comes back to float64 round-off, the
    discarded residual included, although MLX's mean and unmixing are float32."""
    fitted = request.getfixturevalue(fixture)
    x = _data(raw, fitted)
    out = _data(fitted.apply(raw.copy()), fitted)
    assert _rel(out, x) <= REL_TOL


def test_exclude_removes_only_that_component(raw, mlx_keep):
    """``exclude=[0]`` subtracts component 0's back-projection (oracle: the
    backend's sphere and unmixing with its float32 transform) and leaves the
    residual subspace untouched."""
    amica = mlx_keep.amica_
    ica = mlx_keep.to_mne_ica()
    x = _data(raw, mlx_keep)
    pw = mlx_keep.pre_whitener_
    change = _data(ica.apply(raw.copy(), exclude=[0]), mlx_keep) - x
    w = amica.get_unmixing_matrix().astype(np.float64)
    a_0 = (np.linalg.pinv(amica.get_sphere()) @ np.linalg.inv(w))[:, 0]
    s_0 = amica.transform(x / pw)[0].astype(np.float64)
    assert _rel(change, -pw * np.outer(a_0, s_0)) <= FLOAT32_REL_TOL
    d = change / pw
    residual_rows = ica.pca_components_[ica.n_components_ :]
    assert np.linalg.norm(residual_rows @ d) <= REL_TOL * np.linalg.norm(d)


def test_saved_ica_applies_identically(raw, mlx_keep, tmp_path):
    ica = mlx_keep.to_mne_ica()
    fname = tmp_path / "amica-mlx-ica.fif"
    ica.save(fname)
    reloaded = mne.preprocessing.read_ica(fname)
    assert reloaded.n_components_ == N_KEEP
    for exclude in ([], [0]):
        np.testing.assert_array_equal(
            reloaded.apply(raw.copy(), exclude=exclude).get_data(),
            ica.apply(raw.copy(), exclude=exclude).get_data(),
        )


def test_mlx_and_torch_export_the_same_basis(raw, mlx_keep):
    """Both backends build the PCA basis from their float64 sphere, so the
    basis, the variances and the component count agree across backends."""
    torch_keep = AMICAICA(random_state=SEED, device="cpu", verbose=False).fit(
        raw, max_iter=2, pcakeep=N_KEEP
    )
    assert torch_keep.n_components_ == mlx_keep.n_components_ == N_KEEP
    ev_t = torch_keep.pca_explained_variance_
    p_m, p_t = mlx_keep.pca_components_, torch_keep.pca_components_
    assert ev_t is not None and p_m is not None and p_t is not None
    np.testing.assert_allclose(mlx_keep.pca_explained_variance_, ev_t, rtol=1e-8)
    # Rows are unique up to sign; compare the projectors onto each subspace.
    for rows in (slice(0, N_KEEP), slice(N_KEEP, NW)):
        np.testing.assert_allclose(
            p_m[rows].T @ p_m[rows], p_t[rows].T @ p_t[rows], atol=1e-10
        )
