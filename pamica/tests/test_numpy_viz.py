"""The legacy NumPy plotting helpers plot the model's own maps and sources.

``pamica.numpy_impl.viz`` reads a results directory through ``load_results``,
which returns the stored arrays: the unmixing ``W`` and the mixing ``A``
transposed (issue #24 convention). The helpers used to plot those as if they
were the true matrices, and to form activations from the raw data with no
mean removal, no sphere and no transpose, so the "mixing vectors" were rows of
the mixing matrix and the activations were not the sources. They now plot what
``AMICA_NumPy.get_sensor_mixing_matrix`` and ``AMICA_NumPy.transform`` return
for the fitted model, which these tests check against the live accessors,
both through the shared data-preparation helper and read back from the
plotted figures.

Real bundled sample EEG only: a short NumPy fit written to ``tmp_path`` and
read back, full rank and with ``pcakeep`` (a non-square, zero-padded sphere).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from pamica import AMICA_NumPy  # noqa: E402
from pamica.numpy_impl import viz  # noqa: E402
from pamica.numpy_impl.data import load_results  # noqa: E402
from pamica.torch_impl.utils import load_eeglab_data  # noqa: E402

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_data"
DATA_FILE = SAMPLE_DIR / "eeglab_data.fdt"
NW = 32
N_FRAMES = 4096
PCAKEEP = 20
CONFIGS = {"full": None, "reduced": PCAKEEP}
# Written and read back in float64, so everything agrees to round-off.
TOL = 1e-10

pytestmark = pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing")


@pytest.fixture(scope="module")
def X() -> np.ndarray:
    data = load_eeglab_data(str(DATA_FILE), data_dim=NW, field_dim=30504)
    return data.astype(np.float64)[:, :N_FRAMES]


@pytest.fixture(scope="module", params=list(CONFIGS))
def fitted(request, X, tmp_path_factory):
    """A short NumPy fit and the directory it wrote."""
    outdir = tmp_path_factory.mktemp(f"viz_{request.param}")
    model = AMICA_NumPy(
        use_tqdm=False,
        seed=0,
        max_iter=5,
        block_size=N_FRAMES,
        pcakeep=CONFIGS[request.param],
        outdir=str(outdir),
    )
    model.fit(X)
    return model, outdir


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _live(model: AMICA_NumPy, X: np.ndarray):
    """The fitted model's sensor maps and model-0 sources, from its accessors."""
    assert model.comp_list is not None
    maps = model.get_sensor_mixing_matrix()
    sources = model.transform(X)[model.comp_list[:, 0], :, 0]
    return maps, sources


def _rel(a, b) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def _bar_heights(ax) -> list:
    """The bar heights of the histogram drawn on ``ax``."""
    return [p.get_height() for p in ax.patches if isinstance(p, Rectangle)]


def test_maps_and_sources_match_the_accessors(fitted, X):
    model, outdir = fitted
    maps, sources = viz._component_maps_and_sources(load_results(outdir), X)
    live_maps, live_sources = _live(model, X)
    n = model.data_dim
    assert maps.shape == (NW, n) and sources is not None
    assert sources.shape == (n, N_FRAMES)
    assert _rel(maps, live_maps) <= TOL
    assert _rel(sources, live_sources) <= TOL


def test_load_results_reads_a_reduced_sphere(fitted):
    """A rank-reduced fit writes its sphere zero-padded to (nx, nx); the
    reader returns the fitted (nw, nx) sphere."""
    model, outdir = fitted
    np.testing.assert_array_equal(load_results(outdir)["sphere"], model.sphere)


def test_plot_components_draws_the_maps_and_the_sources(fitted, X):
    model, outdir = fitted
    live_maps, live_sources = _live(model, X)
    viz.plot_components(outdir, data=X, max_comps=3)
    axes = np.array(plt.gcf().axes).reshape(3, 2)
    for i in range(3):
        drawn_map = axes[i, 0].lines[0].get_ydata()
        assert _rel(drawn_map, live_maps[:, i]) <= TOL
        heights = _bar_heights(axes[i, 1])
        expected, _ = np.histogram(live_sources[i], bins=50, density=True)
        np.testing.assert_allclose(heights, expected, rtol=1e-8)


def test_plot_pdf_fits_histograms_the_sources(fitted, X):
    model, outdir = fitted
    _, live_sources = _live(model, X)
    viz.plot_pdf_fits(outdir, X, max_comps=2)
    for i, ax in enumerate(plt.gcf().axes):
        heights = _bar_heights(ax)
        expected, _ = np.histogram(live_sources[i], bins=50, density=True)
        np.testing.assert_allclose(heights, expected, rtol=1e-8)


def test_plot_model_comparison_draws_the_reconstruction(fitted, X):
    """The red line is the model's reconstruction from its own maps and
    sources, plus the mean it removed; for a full-rank fit that is the data."""
    model, outdir = fitted
    live_maps, live_sources = _live(model, X)
    assert model.mean is not None
    reconstruction = live_maps @ live_sources + model.mean
    viz.plot_model_comparison(outdir, X, n_examples=2)
    axes = np.array(plt.gcf().axes).reshape(2, 2)
    rng = np.random.RandomState(42)  # the helper's own choice of time points
    times = rng.choice(X.shape[1], 2, replace=False)
    for i, t in enumerate(times):
        drawn = axes[i, 1].lines[0].get_ydata()
        assert _rel(drawn, reconstruction[:, t]) <= TOL
        if model.data_dim == NW:
            assert _rel(drawn, X[:, t]) <= TOL
