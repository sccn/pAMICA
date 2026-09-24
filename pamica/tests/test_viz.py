"""Tests for ``pamica.viz`` (issue #136, Phase 4 visualizations).

Per the NO-MOCK policy (`.rules/testing.md`), every test exercises the real
bundled sample EEG data and a real (short but genuine) AMICA fit; none use
synthetic/random data as ground truth.

Assertions target structure (axes count, labels, line count, tick labels,
masked diagonal, error messages, probabilities summing to 1, the fitted-PDF
values themselves), matching the precedent in `test_metrics_pmi.py` /
`test_metrics_mir.py`, not pixel comparisons and not merely "did not crash".
"""

import dataclasses
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

from pamica.amica import AMICA
from pamica.metrics import block_diagonal_order, pairwise_mi
from pamica.numpy_impl.load import (
    AmicaOutput,
    loadmodout,
    read_eeglab_set_metadata,
)
from pamica.viz import plot_model_probability, plot_pmi_heatmap

SAMPLE_DIR = Path(__file__).resolve().parents[1] / "sample_data"
DATA_FILE = SAMPLE_DIR / "eeglab_data.fdt"
SET_FILE = SAMPLE_DIR / "eeglab_data.set"

NW = 32
FIELD = 30504


@pytest.fixture(scope="module")
def real_data() -> np.ndarray:
    if not DATA_FILE.exists():
        pytest.skip("sample data missing")
    from pamica.torch_impl.utils import load_eeglab_data

    return load_eeglab_data(str(DATA_FILE), data_dim=NW, field_dim=FIELD).astype(
        np.float64
    )


@pytest.fixture(scope="module")
def data_slice(real_data) -> np.ndarray:
    return real_data[:, :4096]


@pytest.fixture(scope="module")
def eeglab_metadata() -> dict:
    if not SET_FILE.exists():
        pytest.skip("eeglab_data.set missing")
    return read_eeglab_set_metadata(SET_FILE)


@pytest.fixture(scope="module")
def two_model_output(data_slice, tmp_path_factory) -> AmicaOutput:
    """A short, real 2-model NG fit, written and read back via loadmodout, so
    `out.Lht`/`out.A`/`out.alpha` etc. are genuine fitted values (not
    fabricated). Reused (read-only) across the tests in this module."""
    model = AMICA(n_models=2, n_mix=3, device="cpu", verbose=False)
    model.fit(data_slice, max_iter=5, block_size=1024, seed=7)

    outdir = tmp_path_factory.mktemp("amicaout_viz")
    model.write_amica_output(str(outdir))
    return loadmodout(outdir)


@pytest.fixture(scope="module")
def small_mi_matrix(data_slice) -> np.ndarray:
    return pairwise_mi(data_slice[:6])


@pytest.fixture(autouse=True)
def _close_figures():
    """Every test creates at least one Figure; close them so the suite doesn't
    trip matplotlib's max-open-figures warning."""
    yield
    plt.close("all")


def test_read_eeglab_set_metadata_tolerates_unlocalized_channels(tmp_path):
    """An unlocalized channel must not block the sample rate.

    This reader originally rejected any file with a channel missing X/Y/Z,
    because its only consumer was a scalp-topography plot that genuinely could
    not use them. That plot was cut (issue #159), so the sole remaining consumer
    wants `srate` and nothing else. Refusing the whole file over a field nobody
    reads would block the sample rate for any dataset with an unlocalized
    channel, and EOG channels commonly are unlocalized.

    Built by blanking one channel's coordinates in a copy of the REAL bundled
    .set, matching this repo's existing precedent for degenerate edge cases
    (see test_amari_distance.py); the file, its srate and its other 31 channels
    stay real.
    """
    from scipy.io import loadmat, savemat

    mat = loadmat(str(SET_FILE), struct_as_record=False, squeeze_me=True)
    eeg = mat["EEG"]
    for axis in ("X", "Y", "Z"):
        setattr(eeg.chanlocs[1], axis, np.array([]))  # EOG1: unlocalized
    doctored = tmp_path / "unlocalized.set"
    savemat(str(doctored), {"EEG": eeg})

    meta = read_eeglab_set_metadata(doctored)

    # The sample rate, the point of the call, still comes back.
    assert meta["srate"] == 128.0
    assert len(meta["labels"]) == NW
    # The unlocalized channel is a visible NaN, not a silent zero.
    assert np.all(np.isnan(meta["positions"][1]))
    # Every other channel keeps its real coordinates.
    others = np.delete(meta["positions"], 1, axis=0)
    assert np.all(np.isfinite(others))


# --- plot_pmi_heatmap -------------------------------------------------------


def test_plot_pmi_heatmap_returns_figure(small_mi_matrix):
    fig = plot_pmi_heatmap(small_mi_matrix)
    assert isinstance(fig, Figure)
    assert len(fig.axes) >= 1


def test_plot_pmi_heatmap_masks_diagonal_by_default(small_mi_matrix):
    fig = plot_pmi_heatmap(small_mi_matrix)
    im = fig.axes[0].images[0]
    arr = im.get_array()
    assert arr is not None
    mask = np.ma.getmaskarray(arr)
    n = small_mi_matrix.shape[0]
    assert np.all(mask[np.eye(n, dtype=bool)])
    assert not np.any(mask[~np.eye(n, dtype=bool)])


def test_plot_pmi_heatmap_mask_diagonal_false_leaves_diagonal_visible(
    small_mi_matrix,
):
    fig = plot_pmi_heatmap(small_mi_matrix, mask_diagonal=False)
    im = fig.axes[0].images[0]
    arr = im.get_array()
    assert arr is not None
    mask = np.ma.getmaskarray(arr)
    assert not np.any(mask)


def test_plot_pmi_heatmap_default_order_matches_block_diagonal_order(
    small_mi_matrix,
):
    expected_order = block_diagonal_order(small_mi_matrix)
    fig = plot_pmi_heatmap(small_mi_matrix)
    tick_labels = [t.get_text() for t in fig.axes[0].get_xticklabels()]
    assert tick_labels == [str(int(i)) for i in expected_order]


def test_plot_pmi_heatmap_custom_order_is_honored(small_mi_matrix):
    n = small_mi_matrix.shape[0]
    custom_order = np.arange(n)[::-1]
    fig = plot_pmi_heatmap(small_mi_matrix, order=custom_order)
    tick_labels = [t.get_text() for t in fig.axes[0].get_xticklabels()]
    assert tick_labels == [str(int(i)) for i in custom_order]


@pytest.mark.parametrize("use_custom_order", [False, True])
def test_plot_pmi_heatmap_pixels_are_actually_reordered(
    small_mi_matrix, use_custom_order
):
    """The DRAWN values must be the permuted matrix, not just the tick labels.

    The tick labels are generated straight from `order`, independently of the
    array handed to `imshow`, so a heatmap whose axes advertise a reordering
    while the pixels underneath stay in the original order would satisfy every
    other test here. Verified: deleting the `mi_matrix[np.ix_(order, order)]`
    permutation entirely left all nine other heatmap tests passing.

    That is the same failure class as issue #136's trap 1, where MATLAB's
    `mInfoMatrix` turned out to be stored ALREADY reordered and comparing it
    raw gave r=-0.13 -- labels and data disagreeing, silently, while looking
    entirely plausible. Assert the pixels directly.
    """
    n = small_mi_matrix.shape[0]
    order = (
        np.arange(n)[::-1]
        if use_custom_order
        else block_diagonal_order(small_mi_matrix)
    )
    fig = plot_pmi_heatmap(
        small_mi_matrix,
        order=order if use_custom_order else None,
        mask_diagonal=False,
    )
    drawn = fig.axes[0].images[0].get_array()
    assert drawn is not None
    np.testing.assert_array_equal(
        np.asarray(drawn), small_mi_matrix[np.ix_(order, order)]
    )


def test_plot_pmi_heatmap_labels_param_indexed_by_order(small_mi_matrix):
    n = small_mi_matrix.shape[0]
    labels = [f"ch{i}" for i in range(n)]
    order = np.arange(n)[::-1]
    fig = plot_pmi_heatmap(small_mi_matrix, order=order, labels=labels)
    tick_labels = [t.get_text() for t in fig.axes[0].get_xticklabels()]
    assert tick_labels == [labels[i] for i in order]


def test_plot_pmi_heatmap_model_title(small_mi_matrix):
    fig_default = plot_pmi_heatmap(small_mi_matrix)
    assert fig_default.axes[0].get_title() == "Pairwise MI"

    fig_model = plot_pmi_heatmap(small_mi_matrix, model=1)
    assert fig_model.axes[0].get_title() == "Model 2"


def test_plot_pmi_heatmap_raises_on_non_square():
    with pytest.raises(ValueError, match="square"):
        plot_pmi_heatmap(np.zeros((3, 5)))


@pytest.mark.parametrize("bad", [np.nan, np.inf])
@pytest.mark.parametrize("use_custom_order", [False, True])
def test_plot_pmi_heatmap_raises_on_non_finite(small_mi_matrix, bad, use_custom_order):
    """Non-finite entries must raise on BOTH order paths.

    Validation used to ride along inside `block_diagonal_order`, so passing an
    explicit `order` skipped it and imshow rendered NaN as a blank cell -- which
    reads as "no dependency here" rather than "the computation failed". Real
    data reaches this via a constant or non-finite source upstream.
    """
    mi = small_mi_matrix.copy()
    mi[0, 1] = mi[1, 0] = bad
    n = mi.shape[0]
    order = np.arange(n)[::-1] if use_custom_order else None
    with pytest.raises(ValueError, match="non-finite"):
        plot_pmi_heatmap(mi, order=order)


def test_plot_pmi_heatmap_accepts_existing_ax(small_mi_matrix):
    fig, ax = plt.subplots()
    returned = plot_pmi_heatmap(small_mi_matrix, ax=ax)
    assert returned is fig
    assert len(ax.images) == 1


# --- plot_model_probability --------------------------------------------------


def test_plot_model_probability_returns_two_axes(two_model_output):
    fig = plot_model_probability(two_model_output)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2


def test_plot_model_probability_one_line_per_model(two_model_output):
    fig = plot_model_probability(two_model_output)
    ax_top = fig.axes[0]
    assert len(ax_top.lines) == two_model_output.num_models == 2


def test_plot_model_probability_probabilities_sum_to_one(two_model_output):
    fig = plot_model_probability(two_model_output)
    probs = np.array([line.get_ydata() for line in fig.axes[0].lines])
    np.testing.assert_allclose(probs.sum(axis=0), 1.0, atol=1e-10)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)


def test_plot_model_probability_bottom_panel_is_best_model_ll(two_model_output):
    fig = plot_model_probability(two_model_output)
    ax_bottom = fig.axes[1]
    assert len(ax_bottom.lines) == 1
    expected = two_model_output.Lht.max(axis=0)
    np.testing.assert_array_equal(ax_bottom.lines[0].get_ydata(), expected)


def test_plot_model_probability_xaxis_samples_without_srate(two_model_output):
    fig = plot_model_probability(two_model_output)
    ax_bottom = fig.axes[1]
    assert ax_bottom.get_xlabel() == "Sample"
    n_samples = two_model_output.Lht.shape[1]
    np.testing.assert_array_equal(ax_bottom.lines[0].get_xdata(), np.arange(n_samples))


def test_plot_model_probability_xaxis_seconds_with_srate(
    two_model_output, eeglab_metadata
):
    srate = eeglab_metadata["srate"]
    fig = plot_model_probability(two_model_output, srate=srate)
    ax_bottom = fig.axes[1]
    assert ax_bottom.get_xlabel() == "Time (s)"
    n_samples = two_model_output.Lht.shape[1]
    expected_t = np.arange(n_samples) / srate
    got_t = np.asarray(ax_bottom.lines[0].get_xdata(), dtype=float)
    np.testing.assert_allclose(got_t, expected_t)


def test_plot_model_probability_smooth_sec_requires_srate(two_model_output):
    with pytest.raises(ValueError, match="srate"):
        plot_model_probability(two_model_output, smooth_sec=1.0)


def test_plot_model_probability_window_sec_requires_srate(two_model_output):
    with pytest.raises(ValueError, match="srate"):
        plot_model_probability(two_model_output, window_sec=5.0)


def test_plot_model_probability_smoothing_preserves_normalization(
    two_model_output, eeglab_metadata
):
    srate = eeglab_metadata["srate"]
    fig = plot_model_probability(two_model_output, srate=srate, smooth_sec=1.0)
    probs = np.array([line.get_ydata() for line in fig.axes[0].lines])
    np.testing.assert_allclose(probs.sum(axis=0), 1.0, atol=1e-8)


def test_plot_model_probability_smoothing_edge_corrected_not_dragged_to_zero(
    two_model_output, eeglab_metadata
):
    """A naive convolve(..., mode="same") zero-pads beyond the data; since Lht
    sits around -100 (nowhere near 0), that would drag the boundary samples
    violently toward zero. The edge-corrected smoothing must keep the first
    and last smoothed samples close to their neighbors instead (issue #136
    MATLAB-oracle finding)."""
    srate = eeglab_metadata["srate"]
    fig = plot_model_probability(two_model_output, srate=srate, smooth_sec=1.0)
    ax_bottom = fig.axes[1]
    ll = np.asarray(ax_bottom.lines[0].get_ydata(), dtype=float)
    raw = two_model_output.Lht.max(axis=0)

    # The naive (uncorrected) approach would put ll[0] far from raw[0] (tens
    # of units away, per the MATLAB-oracle measurement); edge-corrected
    # smoothing keeps it within a small multiple of the local variation.
    local_scale = np.std(raw[:200]) + 1e-9
    assert abs(ll[0] - raw[0]) < 10 * local_scale
    assert abs(ll[-1] - raw[-1]) < 10 * local_scale


def test_plot_model_probability_smooth_sec_too_small_raises(
    two_model_output, eeglab_metadata
):
    srate = eeglab_metadata["srate"]
    with pytest.raises(ValueError, match="smooth_sec"):
        plot_model_probability(two_model_output, srate=srate, smooth_sec=0.001)


def test_plot_model_probability_smooth_sec_too_large_raises(
    two_model_output, eeglab_metadata
):
    srate = eeglab_metadata["srate"]
    n_samples = two_model_output.Lht.shape[1]
    too_long_sec = (n_samples + 1) / srate
    with pytest.raises(ValueError, match="smooth_sec"):
        plot_model_probability(two_model_output, srate=srate, smooth_sec=too_long_sec)


def test_plot_model_probability_window_sec_sets_xlim(two_model_output, eeglab_metadata):
    srate = eeglab_metadata["srate"]
    fig = plot_model_probability(two_model_output, srate=srate, window_sec=5.0)
    assert fig.axes[1].get_xlim() == (0.0, 5.0)


def test_plot_model_probability_legend_labels(two_model_output):
    fig = plot_model_probability(two_model_output)
    legend = fig.axes[0].get_legend()
    assert legend is not None
    legend_labels = [t.get_text() for t in legend.get_texts()]
    assert legend_labels == ["Model 1", "Model 2"]


def test_plot_model_probability_accepts_existing_axes(two_model_output):
    fig, axes = plt.subplots(2, 1)
    returned = plot_model_probability(two_model_output, axes=axes)
    assert returned is fig


def test_plot_model_probability_raises_without_lht(two_model_output):
    out_no_lht = dataclasses.replace(two_model_output, Lht=None)
    with pytest.raises(ValueError, match="Lht"):
        plot_model_probability(out_no_lht)


# --- live `lht` array path (issue #141) --------------------------------------
def test_plot_model_probability_lht_array_matches_out(two_model_output):
    """Passing a raw Lht array reproduces the AmicaOutput path exactly."""
    lht = np.asarray(two_model_output.Lht, dtype=np.float64)
    fig_out = plot_model_probability(two_model_output)
    fig_lht = plot_model_probability(lht=lht)
    for line_out, line_lht in zip(fig_out.axes[0].lines, fig_lht.axes[0].lines):
        np.testing.assert_array_equal(line_out.get_ydata(), line_lht.get_ydata())
    np.testing.assert_array_equal(
        fig_out.axes[1].lines[0].get_ydata(), fig_lht.axes[1].lines[0].get_ydata()
    )


def test_plot_model_probability_requires_exactly_one_source(two_model_output):
    lht = np.asarray(two_model_output.Lht, dtype=np.float64)
    with pytest.raises(ValueError, match="exactly one"):
        plot_model_probability(two_model_output, lht=lht)  # both
    with pytest.raises(ValueError, match="provide"):
        plot_model_probability()  # neither


def test_plot_model_probability_rejects_non_2d_lht():
    with pytest.raises(ValueError, match="2-D"):
        plot_model_probability(lht=np.zeros(10))


def test_plot_model_probability_lht_with_smoothing(two_model_output, eeglab_metadata):
    """The srate/smoothing kwargs (passed through by AMICAICA) work on lht=."""
    lht = np.asarray(two_model_output.Lht, dtype=np.float64)
    srate = eeglab_metadata["srate"]
    fig = plot_model_probability(lht=lht, srate=srate, smooth_sec=1.0)
    probs = np.array([line.get_ydata() for line in fig.axes[0].lines])
    np.testing.assert_allclose(probs.sum(axis=0), 1.0, atol=1e-8)
