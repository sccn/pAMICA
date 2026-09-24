"""The workflow we run ourselves, end to end, on each backend (issue #315).

Epic #324's first users fit AMICA per session on average-referenced EEG,
keeping ``pcakeep = n_channels - 1`` components (average referencing removes
one dimension), then clean the recording in MNE and hand the same fit to
EEGLAB. This module runs that pipeline on the bundled real EEG through the
public API only, once per backend (the PyTorch default and
``backend="mlx"``), and checks each hand-off:

1. ``AMICAICA`` fit: 31 components, ``get_sources`` equal to the wrapper's
   ``transform``;
2. ``apply`` with one component excluded: only that component's
   back-projection is removed, and the discarded residual is kept;
3. the EEGLAB export reloads through ``loadmodout`` with the reduced rank
   handled as the reference writes it;
4. ``AMICA.save``/``AMICA.load`` gives identical sources;
5. a fit driven by the bundled Fortran ``input.param``;
6. the two backends find the same components.

It is the trust gate for the whole pipeline, so it stays fast (a few seconds
per backend) and is not marked slow: CI's macOS job runs it on both backends.
MLX tests skip individually without MLX or an Apple GPU, like the other MLX
suites; the PyTorch ones run everywhere the MNE extra is installed.
"""

from pathlib import Path

import numpy as np
import pytest

mne = pytest.importorskip("mne")

from pamica import AMICA  # noqa: E402  (after importorskip)
from pamica.mne_compat import AMICAICA  # noqa: E402
from pamica.numpy_impl.load import loadmodout  # noqa: E402
from pamica.tests.test_pca_reduction_cross_backend import (  # noqa: E402
    _matched_abs_corr,
)

mne.set_log_level("ERROR")

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "sample_data"
SET_FILE = SAMPLE_DIR / "eeglab_data.set"
PARAM_FILE = SAMPLE_DIR / "input.param"
SEED = 42
MAX_ITER = 10  # the hand-offs are functions of the fitted state, not convergence
NW = 32
N_KEEP = NW - 1  # pcakeep = n_channels - 1 after the average reference
BACKENDS = ["torch", "mlx"]

# Anything computed in float64 from the fitted state: apply(), the residual
# leak of an exclusion (measured 4e-16 to 2e-15 on both backends).
REL_TOL = 1e-10
# What passes through a backend's own transform, the sources and a component's
# back-projection: float64 for PyTorch (measured 3-6e-15), float32 for MLX
# (measured 1.9-2.2e-7).
TRANSFORM_TOL = {"torch": 1e-10, "mlx": 1e-5}

pytestmark = pytest.mark.skipif(
    not SET_FILE.exists(), reason="sample eeglab_data.set missing"
)


def _require_backend(backend: str) -> None:
    """Skip this test (never the module) when ``backend`` cannot run here."""
    if backend == "mlx":
        mlx_core = pytest.importorskip(
            "mlx.core", reason="MLX not installed (Apple Silicon only)"
        )
        if mlx_core.default_device().type != mlx_core.DeviceType.gpu:
            pytest.skip("no Apple GPU")


def _rel(a, b) -> float:
    """Frobenius-norm relative difference of ``a`` from ``b``."""
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


@pytest.fixture(scope="module")
def raw() -> "mne.io.BaseRaw":
    """The bundled EEG, average-referenced as in our own preprocessing."""
    raw = mne.io.read_raw_eeglab(str(SET_FILE), preload=True)
    raw.set_eeg_reference("average")
    return raw


@pytest.fixture(scope="module")
def fitted(raw):
    """One ``AMICAICA`` fit per backend, shared by the checks below."""
    cache: dict = {}

    def get(backend: str) -> AMICAICA:
        _require_backend(backend)
        if backend not in cache:
            n_eeg = len(mne.pick_types(raw.info, eeg=True))
            ica = AMICAICA(backend=backend, random_state=SEED, verbose=False)
            ica.fit(raw, picks="eeg", pcakeep=n_eeg - 1, max_iter=MAX_ITER)
            cache[backend] = ica
        return cache[backend]

    return get


def _fit_data(raw, ica: AMICAICA) -> np.ndarray:
    """The data the backend was fitted on: the picked channels, pre-whitened."""
    return raw.get_data(picks=ica.ch_names_) / ica.pre_whitener_


def test_average_reference_leaves_rank_n_minus_one(raw):
    """The premise of ``pcakeep = n_channels - 1``: after the average
    reference the recording has one dimension fewer than it has channels."""
    x = raw.get_data(picks="eeg")
    s = np.linalg.svd(x, compute_uv=False)
    assert x.shape[0] == NW
    assert int((s > 1e-10 * s[0]).sum()) == N_KEEP


# --- 1. fit ---------------------------------------------------------------------
@pytest.mark.parametrize("backend", BACKENDS)
def test_fit_keeps_n_minus_one_components(raw, fitted, backend):
    ica = fitted(backend)
    assert ica.converged_ is True
    assert ica.amica_ is not None and ica.amica_.backend == backend
    assert ica.n_components_ == N_KEEP
    assert ica.amica_.get_sphere().shape == (N_KEEP, NW)
    s_mne = ica.get_sources(raw).get_data()
    s_amica = ica.amica_.transform(_fit_data(raw, ica))
    assert s_mne.shape == s_amica.shape == (N_KEEP, raw.n_times)
    assert _rel(s_mne, s_amica) <= TRANSFORM_TOL[backend]


# --- 2. clean one component -------------------------------------------------------
@pytest.mark.parametrize("backend", BACKENDS)
def test_excluding_a_component_removes_only_its_back_projection(raw, fitted, backend):
    """``apply(exclude=[k])`` keeps every channel and changes the data by
    exactly minus component ``k``'s back-projection (the scalp map times the
    source, computed from the wrapper's own sphere, unmixing and transform),
    with nothing leaking into the discarded residual subspace."""
    ica = fitted(backend)
    amica = ica.amica_
    assert amica is not None
    k = 0
    x = raw.get_data(picks=ica.ch_names_)
    out = ica.apply(raw.copy(), exclude=[k]).get_data(picks=ica.ch_names_)
    assert out.shape == x.shape == (NW, raw.n_times)

    pw = ica.pre_whitener_
    a_k = (
        np.linalg.pinv(amica.get_sphere()) @ np.linalg.inv(amica.get_unmixing_matrix())
    )[:, k]
    s_k = amica.transform(x / pw)[k].astype(np.float64)
    change = out - x
    assert _rel(change, -pw * np.outer(a_k, s_k)) <= TRANSFORM_TOL[backend]

    # Exactly one dimension is removed: the residual the fit discarded is
    # restored, not dropped with the component (issue #322). Ranks are of the
    # time-centered data, since the removed back-projection is of centered
    # sources (it adds a constant offset per channel).
    residual_rows = ica.pca_components_[N_KEEP:]
    d = change / pw
    assert np.linalg.norm(residual_rows @ d) <= REL_TOL * np.linalg.norm(d)

    def rank(data):
        centered = data - data.mean(axis=1, keepdims=True)
        s = np.linalg.svd(centered, compute_uv=False)
        return int((s > 1e-8 * s[0]).sum())

    assert rank(x) == N_KEEP
    assert rank(out) == N_KEEP - 1

    # Nothing excluded gives the input back, residual included.
    same = ica.apply(raw.copy()).get_data(picks=ica.ch_names_)
    assert _rel(same, x) <= REL_TOL


# --- 3. EEGLAB export ---------------------------------------------------------------
@pytest.mark.parametrize("backend", BACKENDS)
def test_eeglab_export_reloads_with_the_reduced_rank(raw, fitted, backend, tmp_path):
    """``write_amica_output`` through the wrapper reloads with ``loadmodout``
    (the NumPy port of EEGLAB's ``loadmodout15``): a 31-dimensional model in a
    32-channel space, the sphere padded with zero rows to the reference's
    ``nx x nx`` record, and ``W``, ``S`` and ``A`` consistent with each other
    and with the live model."""
    ica = fitted(backend)
    amica = ica.amica_
    assert amica is not None
    outdir = tmp_path / "amicaout"
    amica.write_amica_output(str(outdir))
    out = loadmodout(outdir)

    assert (out.num_pcs, out.data_dim, out.num_models) == (N_KEEP, NW, 1)
    assert out.W.shape == (N_KEEP, N_KEEP, 1)
    assert out.S.shape == (NW, NW)
    assert out.A.shape == (NW, N_KEEP, 1)
    np.testing.assert_array_equal(out.S[N_KEEP:], 0.0)
    unmix = out.W[:, :, 0] @ out.S[:N_KEEP]
    np.testing.assert_allclose(unmix @ out.A[:, :, 0], np.eye(N_KEEP), atol=1e-8)

    # Up to loadmodout's variance order and per-component rescaling, the
    # reloaded sources are the live ones.
    x = _fit_data(raw, ica)
    order = np.asarray(out.origord).ravel()
    assert sorted(order.tolist()) == list(range(N_KEEP))
    loaded = out.sources(x)
    live = amica.transform(x).astype(np.float64)[order]
    scale = (loaded * live).sum(axis=1) / (live**2).sum(axis=1)
    residual = loaded - scale[:, None] * live
    assert np.linalg.norm(residual) <= TRANSFORM_TOL[backend] * np.linalg.norm(loaded)


# --- 4. save and load -----------------------------------------------------------------
@pytest.mark.parametrize("backend", BACKENDS)
def test_save_load_gives_identical_sources(raw, fitted, backend, tmp_path):
    ica = fitted(backend)
    amica = ica.amica_
    assert amica is not None
    path = tmp_path / "model.pt"
    amica.save(str(path))
    loaded = AMICA.load(str(path))
    assert loaded.backend == backend
    x = _fit_data(raw, ica)
    np.testing.assert_array_equal(loaded.transform(x), amica.transform(x))


# --- 5. a params-file driven run --------------------------------------------------------
@pytest.mark.parametrize("backend", BACKENDS)
def test_input_param_drives_the_fit(raw, backend):
    """The bundled Fortran ``input.param`` configures the fit on either
    backend: its settings reach the backend (``block_size 512``, Newton on,
    three mixtures), and its ``pcakeep 32`` is capped by the data's rank 31."""
    _require_backend(backend)
    model = AMICA.from_params_file(str(PARAM_FILE), backend=backend, verbose=False)
    model.fit(raw.get_data(picks="eeg"), max_iter=3, seed=SEED)
    assert model.converged_ is True
    backend_model = model.model_
    assert backend_model is not None
    assert backend_model.block_size == 512
    assert backend_model.do_newton is True
    assert backend_model.n_mix == 3
    assert backend_model.n_channels == N_KEEP
    assert len(model.ll_history_) == 3


# --- 6. the two backends agree ------------------------------------------------------------
def test_torch_and_mlx_find_the_same_components(raw, fitted):
    """Same data, seed and settings on both backends: the same number of
    components and the same sources (Hungarian-matched |corr|, the bar of the
    other torch-vs-MLX suites; measured min 0.999999997 at the default lrate,
    0.1 since issue #354)."""
    t, m = fitted("torch"), fitted("mlx")
    assert t.n_components_ == m.n_components_ == N_KEEP
    matched, _ = _matched_abs_corr(
        t.get_sources(raw).get_data(), m.get_sources(raw).get_data()
    )
    assert matched.min() >= 0.999, f"min matched |corr| {matched.min():.6f}"
