"""Cross-backend round trip for the EEGLAB sphere export (issue #336).

``write_amicaout`` writes the sphere matrix ``S`` column-major (Fortran
layout), matching the reference binary (amica15.f90 near :2423) and both
readers (EEGLAB's ``loadmodout15.m`` and pamica's ``loadmodout``). For the
default symmetric zero-phase component analysis (ZCA) sphere this was
invisible even under the old, wrong C-order write, because that sphere is its
own transpose to about 1e-17; with ``do_approx_sphere=False`` the sphere is
genuinely asymmetric, and a C-order write came back exactly transposed
(measured on the bundled sample, torch, 3 iterations: ``max|S_loaded - S| =
0.51``, ``max|S_loaded - S.T| = 0.0``).

This module fits each backend briefly on the bundled real sample EEG, with
``do_approx_sphere`` both on and off and with and without rank reduction
(``pcakeep``), and checks that ``loadmodout`` (and, for the NumPy backend,
``load_results`` on its own output directory) returns exactly the sphere the
backend itself reports through ``get_sphere()``/``.sphere``.

Real bundled sample EEG only, short fits (``.rules/testing.md``: no mocks, no
synthetic data). MLX is exercised through ``pytest.importorskip`` plus the
Apple-GPU guard, following ``pamica/tests/test_sensor_mixing_cross_backend.py``.
"""

from pathlib import Path

import numpy as np
import pytest

from pamica import AMICA_NumPy
from pamica.numpy_impl.data import load_results
from pamica.numpy_impl.load import loadmodout
from pamica.torch_impl.core import AMICATorchNG
from pamica.torch_impl.utils import load_eeglab_data

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_data"
DATA_FILE = SAMPLE_DIR / "eeglab_data.fdt"
NW = 32
FIELD = 30504
SEED = 42
N_FRAMES = 8192
MAX_ITER = 2
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


def _fit_and_write(
    backend: str, X: np.ndarray, do_approx_sphere: bool, pcakeep, outdir: Path
):
    """Fit a short ``backend`` model and write its EEGLAB output to
    ``outdir``. Returns ``(model, expected_sphere)``, where ``expected_sphere``
    is what the backend itself reports as the fitted sphere."""
    if backend == "numpy":
        model = AMICA_NumPy(
            use_tqdm=False,
            seed=SEED,
            max_iter=MAX_ITER,
            block_size=BLOCK_SIZE,
            pcakeep=pcakeep,
            do_approx_sphere=do_approx_sphere,
            outdir=str(outdir),
        )
        model.fit(X)
        assert model.converged, "fixture expects a converged (finite-LL) fit"
        return model, model.sphere

    cls = AMICATorchNG if backend == "torch" else _mlx_class()
    kwargs = {"device": "cpu"} if backend == "torch" else {}
    data = X if backend == "torch" else X.astype(np.float32)
    model = cls(
        n_channels=NW,
        seed=SEED,
        block_size=BLOCK_SIZE,
        pcakeep=pcakeep,
        do_approx_sphere=do_approx_sphere,
        **kwargs,
    )
    model.fit(data, max_iter=MAX_ITER, verbose=False)
    model.write_amica_output(outdir)
    return model, model.get_sphere()


@pytest.fixture(scope="module")
def fits(X, tmp_path_factory):
    cache: dict = {}

    def get(backend: str, do_approx_sphere: bool, config: str):
        key = (backend, do_approx_sphere, config)
        if key not in cache:
            outdir = tmp_path_factory.mktemp(f"{backend}_{do_approx_sphere}_{config}")
            cache[key] = _fit_and_write(
                backend, X, do_approx_sphere, CONFIGS[config], outdir
            ) + (outdir,)
        return cache[key]

    return get


@pytest.mark.parametrize("config", list(CONFIGS))
@pytest.mark.parametrize("do_approx_sphere", [True, False])
@pytest.mark.parametrize("backend", ["torch", "numpy", "mlx"])
def test_loadmodout_sphere_matches_backend(fits, backend, do_approx_sphere, config):
    if backend == "mlx":
        _mlx_class()
    model, expected, outdir = fits(backend, do_approx_sphere, config)

    n = NW if CONFIGS[config] is None else PCAKEEP
    assert expected.shape == (n, NW)

    if not do_approx_sphere and config == "full":
        # Full-rank exact sphering is genuinely asymmetric -- exactly the case
        # the old C-order write got wrong (it agreed with a symmetric sphere
        # by coincidence). Guard that this test would actually catch a
        # regression back to C order rather than passing vacuously.
        assert np.abs(expected - expected.T).max() > 1e-3, (
            "sphere is unexpectedly symmetric here; this test would not guard anything"
        )

    out = loadmodout(outdir)
    loaded = out.S[:n]
    if backend == "mlx":
        # write_amica_output exports self.sphere, the float32-cast sphere
        # (mlx_impl/core.py _preprocess); get_sphere() returns the un-rounded
        # float64 sphere (_sphere_np) computed just before that cast. The two
        # differ by a ~1e-7 relative float32-rounding gap independent of the
        # column-major fix this module guards (present equally for both
        # do_approx_sphere values), so this is not a tolerance loosened to
        # hide the order bug: a wrong-order read is off by O(1), not O(1e-7).
        np.testing.assert_allclose(loaded, expected, rtol=1e-6, atol=1e-7)
    else:
        np.testing.assert_array_equal(loaded, expected)

    if backend == "numpy":
        r = load_results(outdir)
        np.testing.assert_array_equal(r["sphere"], expected)
