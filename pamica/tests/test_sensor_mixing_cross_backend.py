"""``get_sensor_mixing_matrix`` agrees across backends (epic #324 Phase 6).

The sensor-space mixing matrix is what scalp maps, dipole fits and the EEGLAB
``icawinv`` are built from, so it has to invert the spatial filter the sources
come from on every backend: ``unmixing @ sphere @ M == I`` (the sphere may be
non-square after rank reduction, hence ``pinv``). The legacy NumPy backend
returned ``pinv(sphere) @ stored_A`` without the transpose the other two apply
(its stored ``W`` is the unmixing transposed, issue #24's convention), so its
maps were the rows of the true mixing matrix instead of its columns: in this
module's fits, a 10.5% (full rank) and 9.0% (``pcakeep=20``) relative
difference from the PyTorch backend's maps, against 4.6e-12 and 9.5e-11 with
the transpose, which also brings ``get_weights() @ sphere @ M`` to the
identity within 2.4e-15.

PyTorch and NumPy always run, so a divergence between them cannot land; MLX
uses ``pytest.importorskip`` plus an Apple-GPU guard (``.rules/backend_parity.md``).
Real bundled sample EEG only, a few iterations per fit: this checks the
accessor on a fitted state, not convergence.
"""

from pathlib import Path

import numpy as np
import pytest

from pamica import AMICA_NumPy
from pamica.torch_impl.core import AMICATorchNG
from pamica.torch_impl.utils import load_eeglab_data

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_data"
DATA_FILE = SAMPLE_DIR / "eeglab_data.fdt"
NW = 32
FIELD = 30504
SEED = 42
N_FRAMES = 8192
MAX_ITER = 5
BLOCK_SIZE = 4096
PCAKEEP = 20
CONFIGS = {"full": None, "reduced": PCAKEEP}

pytestmark = pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing")


@pytest.fixture(scope="module")
def X() -> np.ndarray:
    data = load_eeglab_data(str(DATA_FILE), data_dim=NW, field_dim=FIELD)
    return data.astype(np.float64)[:, :N_FRAMES]


def _mlx_class():
    mlx_core = pytest.importorskip(
        "pamica.mlx_impl.core", reason="MLX not installed (Apple Silicon only)"
    )
    if mlx_core.mx.default_device().type != mlx_core.mx.DeviceType.gpu:
        pytest.skip("no Apple GPU")
    return mlx_core.AMICAMLXNG


def _fit(backend: str, X: np.ndarray, pcakeep, outdir: Path):
    """A short fit of ``backend`` on ``X``; the same seed and settings on all."""
    if backend == "numpy":
        model = AMICA_NumPy(
            use_tqdm=False,
            seed=SEED,
            max_iter=MAX_ITER,
            block_size=BLOCK_SIZE,
            pcakeep=pcakeep,
            outdir=str(outdir),
        )
        return model.fit(X)
    cls = AMICATorchNG if backend == "torch" else _mlx_class()
    kwargs = {"device": "cpu"} if backend == "torch" else {}
    model = cls(
        n_channels=NW, seed=SEED, block_size=BLOCK_SIZE, pcakeep=pcakeep, **kwargs
    )
    data = X if backend == "torch" else X.astype(np.float32)
    return model.fit(data, max_iter=MAX_ITER, verbose=False)


def _filter(backend: str, model) -> np.ndarray:
    """The full spatial filter ``unmixing @ sphere`` the sources come from."""
    if backend == "numpy":
        return model.get_weights() @ model.sphere
    return model.get_unmixing_matrix().astype(np.float64) @ model.get_sphere()


@pytest.fixture(scope="module")
def fits(X, tmp_path_factory):
    cache: dict = {}

    def get(backend: str, config: str):
        if (backend, config) not in cache:
            outdir = tmp_path_factory.mktemp(f"{backend}_{config}")
            cache[backend, config] = _fit(backend, X, CONFIGS[config], outdir)
        return cache[backend, config]

    return get


@pytest.mark.parametrize("config", list(CONFIGS))
@pytest.mark.parametrize("backend", ["torch", "numpy", "mlx"])
def test_sensor_mixing_inverts_the_spatial_filter(fits, backend, config):
    """``unmixing @ sphere @ M == I``: each column of ``M`` is the sensor map
    of the source with the same index."""
    if backend == "mlx":
        _mlx_class()
    model = fits(backend, config)
    n = NW if CONFIGS[config] is None else PCAKEEP
    M = model.get_sensor_mixing_matrix()
    assert M.shape == (NW, n)
    atol = 1e-5 if backend == "mlx" else 1e-10
    np.testing.assert_allclose(_filter(backend, model) @ M, np.eye(n), atol=atol)


@pytest.mark.parametrize("config", list(CONFIGS))
def test_numpy_sensor_mixing_matches_torch(fits, config):
    """The float64 backends follow the same trajectory, so their sensor maps
    agree to round-off (measured 4.6e-12 full rank, 9.5e-11 reduced); before
    the fix they differed by about 10%."""
    m_torch = fits("torch", config).get_sensor_mixing_matrix()
    m_numpy = fits("numpy", config).get_sensor_mixing_matrix()
    err = np.linalg.norm(m_numpy - m_torch) / np.linalg.norm(m_torch)
    assert err < 1e-9, f"NumPy vs PyTorch sensor mixing differs by {err:.3e}"


@pytest.mark.parametrize("config", list(CONFIGS))
def test_mlx_sensor_mixing_matches_torch(fits, config):
    """MLX computes in float32, so after a few iterations its sensor maps agree
    with the float64 PyTorch ones to float32-trajectory tolerance (measured
    1.5e-5 full rank, 4.4e-5 reduced), under the bar
    ``test_mlx_transform_cross_backend.py`` uses for the same accessor."""
    _mlx_class()
    m_torch = fits("torch", config).get_sensor_mixing_matrix()
    m_mlx = fits("mlx", config).get_sensor_mixing_matrix()
    err = np.linalg.norm(m_mlx - m_torch) / np.linalg.norm(m_torch)
    assert err < 1e-3, f"MLX vs PyTorch sensor mixing differs by {err:.3e}"
