"""``get_sphere``/``get_mean``/``get_model_center`` on every backend (issue #313).

The three accessors let a consumer that composes the transform itself (the MNE
export in ``pamica.mne_compat``) read the preprocessing of any backend the same
way: same names, same shapes, float64 whatever the backend computes in. This is
the anti-drift guard required by ``.rules/backend_parity.md``: the PyTorch
checks always run, and the MLX ones (``pytest.importorskip`` plus an Apple-GPU
guard) compare MLX against PyTorch on the same data, seed and configuration.

Real bundled sample EEG only (32 channels x 30504 frames), no synthetic data or
mocks (``.rules/testing.md``).
"""

from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np
import pytest
import torch

from pamica.torch_impl.core import AMICATorchNG
from pamica.torch_impl.utils import load_eeglab_data

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_data"
DATA_FILE = SAMPLE_DIR / "eeglab_data.fdt"
NW = 32
FIELD = 30504
SEED = 42
PCAKEEP = 20
N_FRAMES = 8192

# The configurations compared: single-model full rank, single-model reduced
# (a non-square sphere), and two models (a nonzero per-model center c). The
# two-model fit runs one iteration, because multi-model trajectories separate
# quickly between float64 torch and float32 MLX (measured: the c columns
# agree to 4.7e-6 of their largest entry after one iteration, and only to
# 5.0e-3 after three), and this compares preprocessing plus one M-step, not
# convergence.
CONFIGS: Dict[str, Tuple[dict, int]] = {
    "full": ({}, 3),
    "reduced": ({"pcakeep": PCAKEEP}, 3),
    "two_models": ({"n_models": 2}, 1),
}
N_KEPT = {"full": NW, "reduced": PCAKEEP, "two_models": NW}

# MLX's get_sphere is its float64 host sphere, built by the same float64
# eigendecomposition as torch's with a different LAPACK path (measured
# 1.4e-14 relative, 3.0e-14 reduced): the tolerances of
# test_pca_reduction_cross_backend.py's host-sphere comparison.
SPHERE_RTOL = 1e-10
SPHERE_ATOL = 1e-13
# MLX stores mean and c in float32, so they agree to float32 rounding for the
# mean (measured 4.0e-8 relative to its largest entry) and to one float32
# M-step for c (measured 4.7e-6).
MEAN_TOL = 1e-6
CENTER_TOL = 5e-5

pytestmark = pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing")


@pytest.fixture(scope="module")
def X() -> np.ndarray:
    return load_eeglab_data(str(DATA_FILE), data_dim=NW, field_dim=FIELD).astype(
        np.float64
    )[:, :N_FRAMES]


def _mlx_core():
    """The MLX backend module, or skip this test (never the whole module)."""
    mlx_core = pytest.importorskip(
        "pamica.mlx_impl.core", reason="MLX not installed (Apple Silicon only)"
    )
    mx = mlx_core.mx
    if mx.default_device().type != mx.DeviceType.gpu:
        pytest.skip("no Apple GPU")
    return mlx_core


def _torch(**kwargs) -> AMICATorchNG:
    return AMICATorchNG(
        n_channels=NW, seed=SEED, device="cpu", dtype=torch.float64, **kwargs
    )


def _mlx(**kwargs):
    return _mlx_core().AMICAMLXNG(n_channels=NW, seed=SEED, **kwargs)


BUILDERS: Dict[str, Callable] = {"torch": _torch, "mlx": _mlx}


@pytest.fixture(scope="module")
def fits(X) -> Callable[[str, str], object]:
    """Fit each (backend, config) pair once per module and share it."""
    cache: Dict[Tuple[str, str], object] = {}

    def get(backend: str, config: str):
        if backend == "mlx":
            _mlx_core()  # skip before fitting, not after
        key = (backend, config)
        if key not in cache:
            kwargs, max_iter = CONFIGS[config]
            model = BUILDERS[backend](**kwargs)
            model.fit(X, max_iter=max_iter, verbose=False)
            assert model.stop_reason not in model._DEGENERATE_STOP_REASONS
            cache[key] = model
        return cache[key]

    return get


def _as_float64(values) -> np.ndarray:
    """A backend array (torch tensor or MLX array) as a float64 numpy array."""
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    return np.array(values, dtype=np.float64)


# --- each backend: contract --------------------------------------------------
@pytest.mark.parametrize("backend", ["torch", "mlx"])
def test_unfitted_accessors_raise(backend):
    model = BUILDERS[backend]()
    for accessor in (model.get_sphere, model.get_mean, model.get_model_center):
        with pytest.raises(RuntimeError, match="requires a fitted model"):
            accessor()


@pytest.mark.parametrize("backend", ["torch", "mlx"])
@pytest.mark.parametrize("config", list(CONFIGS))
def test_accessors_return_the_fitted_state_as_float64(fits, backend, config):
    """Shapes, dtype and values: each accessor returns exactly the stored
    array (MLX's sphere is its float64 host copy, the others float32 casts)."""
    model = fits(backend, config)
    n_kept = N_KEPT[config]
    sphere = model.get_sphere()
    mean = model.get_mean()
    assert sphere.shape == (n_kept, NW) and sphere.dtype == np.float64
    assert mean.shape == (NW,) and mean.dtype == np.float64
    stored_sphere = model._sphere_np if backend == "mlx" else model.sphere
    np.testing.assert_array_equal(sphere, _as_float64(stored_sphere))
    np.testing.assert_array_equal(mean, _as_float64(model.mean).ravel())
    c = _as_float64(model.c)
    for h in range(model.n_models):
        center = model.get_model_center(h)
        assert center.shape == (n_kept,) and center.dtype == np.float64
        np.testing.assert_array_equal(center, c[:, h])
    # numpy integer indices are valid model indices, as for every accessor.
    np.testing.assert_array_equal(model.get_model_center(np.int64(0)), c[:, 0])


@pytest.mark.parametrize("backend", ["torch", "mlx"])
def test_accessors_return_independent_copies(fits, backend):
    """Writing into a returned array must not reach the fitted model."""
    model = fits(backend, "two_models")
    for accessor in (model.get_sphere, model.get_mean, model.get_model_center):
        before = accessor().copy()
        accessor()[...] = np.nan
        np.testing.assert_array_equal(accessor(), before)


@pytest.mark.parametrize("backend", ["torch", "mlx"])
def test_model_center_validates_the_index(fits, backend):
    model = fits(backend, "two_models")
    with pytest.raises(ValueError, match="out of range"):
        model.get_model_center(2)
    with pytest.raises(ValueError, match="out of range"):
        model.get_model_center(-1)
    with pytest.raises(TypeError, match="model_idx must be an int"):
        model.get_model_center(0.0)


def test_single_model_center_is_zero(fits):
    """The c update is gated to n_models > 1, so this pins the documented zero."""
    np.testing.assert_array_equal(
        fits("torch", "full").get_model_center(), np.zeros(NW)
    )


def test_float32_torch_fit_still_returns_float64(X):
    model = AMICATorchNG(n_channels=NW, seed=SEED, device="cpu", dtype=torch.float32)
    model.fit(X, max_iter=2, verbose=False)
    for value in (model.get_sphere(), model.get_mean(), model.get_model_center()):
        assert value.dtype == np.float64


# --- anti-drift: MLX agrees with PyTorch ---------------------------------------
@pytest.mark.parametrize("config", list(CONFIGS))
def test_mlx_accessors_agree_with_torch(fits, config):
    t = fits("torch", config)
    m = fits("mlx", config)
    np.testing.assert_allclose(
        m.get_sphere(), t.get_sphere(), rtol=SPHERE_RTOL, atol=SPHERE_ATOL
    )
    t_mean = t.get_mean()
    assert np.abs(m.get_mean() - t_mean).max() <= MEAN_TOL * np.abs(t_mean).max()
    for h in range(t.n_models):
        t_c = t.get_model_center(h)
        m_c = m.get_model_center(h)
        assert m_c.shape == t_c.shape
        if t.n_models == 1:
            np.testing.assert_array_equal(m_c, t_c)  # both identically zero
            continue
        scale = np.abs(t_c).max()
        assert scale > 0, f"model {h}: c should be nonzero for two models"
        assert np.abs(m_c - t_c).max() <= CENTER_TOL * scale
