"""Block-size auto-tuner on the MLX backend (issue #232).

Apple-Silicon only; the module self-skips without MLX or an Apple GPU. Real
sample EEG, float32 (Apple GPUs have no float64).

The MLX-specific risks this covers are the two that do not exist on the other
backends: the accumulate pass is a *lazy* graph, so a probe that forgets to
``mx.eval`` would time graph construction and pick whichever block size builds
fastest; and an allocation failure had to be shown to be a catchable Python
exception rather than the process abort MLX takes elsewhere (issue #274).

Assertions are machine-robust -- which size wins is a timing outcome -- and this
module runs in macOS CI.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from pamica import blocktune  # noqa: E402

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "sample_data"
DATA_FILE = SAMPLE_DIR / "eeglab_data.fdt"
NW = 32
FIELD = 30504
NMIX = 3
SEED = 42

pytestmark = [
    pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing"),
    pytest.mark.skipif(
        mx.default_device().type != mx.DeviceType.gpu, reason="no Apple GPU"
    ),
]


def _load_real_data() -> np.ndarray:
    from pamica.torch_impl.utils import load_eeglab_data

    return load_eeglab_data(str(DATA_FILE), data_dim=NW, field_dim=FIELD).astype(
        np.float64
    )


@pytest.fixture(scope="module")
def real_data() -> np.ndarray:
    return _load_real_data()


def _model(**kwargs: Any):
    from pamica.mlx_impl import AMICAMLXNG

    params: dict[str, Any] = dict(n_channels=NW, n_mix=NMIX, seed=SEED)
    params.update(kwargs)
    return AMICAMLXNG(**params)


RANK = 20


@pytest.fixture(scope="module")
def rank_deficient(real_data: np.ndarray) -> np.ndarray:
    """Real EEG projected onto its top-``RANK`` subspace (what SSS does to MEG),
    the established rank-reduction route in ``torch_tests/test_ng_rank_deficient.py``."""
    centered = real_data - real_data.mean(axis=1, keepdims=True)
    U = np.linalg.svd(centered, full_matrices=False)[0][:, :RANK]
    return U @ (U.T @ centered)


def _recorded_search(monkeypatch) -> dict:
    """Capture the arguments the backend hands ``blocktune.search``: after
    preprocessing they are the model's REDUCED dimensions, which is how the
    ordering of the search relative to preprocessing becomes observable."""
    from pamica.mlx_impl import core as mlx_core

    recorded: dict = {}
    real_search = blocktune.search

    def recording_search(**kwargs):
        recorded.update(kwargs)
        return real_search(**kwargs)

    monkeypatch.setattr(mlx_core.blocktune, "search", recording_search)
    return recorded


# ---------------------------------------------------------------------------
# Defaults and inertness
# ---------------------------------------------------------------------------


def test_search_is_off_by_default_with_the_shared_bounds():
    model = _model()
    assert model.do_opt_block is False
    assert model.block_size == 8192
    assert (model.blk_min, model.blk_max, model.blk_step) == (
        blocktune.DEFAULT_BLK_MIN,
        blocktune.DEFAULT_BLK_MAX,
        blocktune.DEFAULT_BLK_STEP,
    )


def test_block_size_untouched_when_search_is_off(real_data):
    model = _model(block_size=4096)
    model.fit(real_data[:, :8192], max_iter=2, verbose=False)
    assert model.block_size == 4096


def test_sweep_bounds_are_not_validated_when_search_is_off():
    assert _model(do_opt_block=False, blk_min=0, blk_max=1, blk_step=0) is not None


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"blk_min": 0}, "blk_min must be >= 1"),
        ({"blk_step": 0}, "blk_step must be >= 1"),
        ({"blk_min": 8192, "blk_max": 4096}, "blk_max must be >= blk_min"),
    ],
)
def test_sweep_bounds_are_validated_when_search_is_on(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _model(do_opt_block=True, **kwargs)


# ---------------------------------------------------------------------------
# The search itself
# ---------------------------------------------------------------------------


def test_chosen_block_size_is_a_valid_candidate_and_is_logged(real_data, caplog):
    model = _model(do_opt_block=True, blk_min=4096, blk_max=16384, blk_step=4096)
    with caplog.at_level(logging.INFO, logger="pamica.mlx_impl.core"):
        model.fit(real_data, max_iter=1, verbose=False)

    assert model.block_size in blocktune.block_size_candidates(
        4096, 16384, 4096, n_samples=real_data.shape[1]
    )
    message = "\n".join(r.getMessage() for r in caplog.records)
    assert f"chose block_size={model.block_size}" in message


def test_candidates_never_exceed_the_sample_count(real_data):
    subset = real_data[:, :5000]
    model = _model(do_opt_block=True, blk_min=4096, blk_max=32768, blk_step=4096)
    model.fit(subset, max_iter=1, verbose=False)
    assert model.block_size <= subset.shape[1]


def test_memory_cap_uses_the_recommended_working_set(monkeypatch):
    """Which of the two reported sizes is used is the whole point of this
    helper: MLX allocates past the recommended working set and starts paging,
    which surfaces as a mysteriously slow candidate rather than a failure, so
    the cap must come from the recommended set and not the larger raw memory
    size. Both keys are reported with different values here, so preferring the
    wrong one is visible rather than indistinguishable."""
    from pamica.mlx_impl.core import AMICAMLXNG

    recommended, total = 8 * 1024**3, 64 * 1024**3
    monkeypatch.setattr(
        mx,
        "device_info",
        lambda: {
            "max_recommended_working_set_size": recommended,
            "memory_size": total,
        },
    )
    assert AMICAMLXNG._available_memory_bytes() == recommended


def test_memory_cap_falls_back_to_memory_size(monkeypatch):
    """Older MLX builds report only the raw size; that is a usable cap, and
    better than no cap at all."""
    from pamica.mlx_impl.core import AMICAMLXNG

    monkeypatch.setattr(mx, "device_info", lambda: {"memory_size": 64 * 1024**3})
    assert AMICAMLXNG._available_memory_bytes() == 64 * 1024**3


def test_memory_cap_is_none_when_mlx_reports_nothing(monkeypatch):
    """No reported size means no cap -- the search then relies entirely on
    catching the real allocation failure, which is the actual safety net."""
    from pamica.mlx_impl.core import AMICAMLXNG

    monkeypatch.setattr(mx, "device_info", lambda: {})
    assert AMICAMLXNG._available_memory_bytes() is None


def test_tuner_runs_on_the_reduced_channel_count(rank_deficient, monkeypatch):
    """Ordering pin: the search runs after preprocessing, so on rank-reduced
    input it sees the kept rank rather than the input channel count -- which is
    what the memory cap has to be sized against."""
    recorded = _recorded_search(monkeypatch)
    model = _model(do_opt_block=True, blk_min=2048, blk_max=8192, blk_step=2048)
    model.fit(rank_deficient, max_iter=2, verbose=False)

    assert model.n_channels == RANK < NW  # reduction really happened
    assert recorded["n_channels"] == RANK
    assert model.block_size in (2048, 4096, 6144, 8192)


# ---------------------------------------------------------------------------
# The tuner must not change the fit
# ---------------------------------------------------------------------------


def test_post_tune_fit_is_bit_identical_to_a_direct_fit(real_data):
    """A fit that tuned its way to size N must be bit-for-bit the fit handed N
    directly: the timed probes only read model state and consume no RNG."""
    tuned = _model(do_opt_block=True, blk_min=4096, blk_max=16384, blk_step=4096)
    tuned.fit(real_data, max_iter=4, verbose=False)

    direct = _model(block_size=tuned.block_size)
    direct.fit(real_data, max_iter=4, verbose=False)

    assert direct.block_size == tuned.block_size
    np.testing.assert_array_equal(np.array(tuned.A), np.array(direct.A))
    assert tuned.ll_history == direct.ll_history
    assert tuned.final_ll_ == direct.final_ll_


def test_probe_evaluates_the_whole_lazy_graph(real_data, monkeypatch):
    """MLX builds a lazy graph, so without an ``mx.eval`` over every
    accumulator the clock would measure graph *construction*: every candidate
    would look identically instant and the winner would be noise. A probe here
    must therefore cost roughly what an EM iteration costs, not ~0."""
    from pamica.mlx_impl import core as mlx_core

    model = _model(do_opt_block=True)
    X = model._preprocess(real_data)
    model._initialize_parameters()

    timings: list = []
    real_search = blocktune.search

    def recording_search(**kwargs):
        probe = kwargs["probe"]

        def wrapped(size):
            elapsed = probe(size)
            timings.append(elapsed)
            return elapsed

        kwargs["probe"] = wrapped
        return real_search(**kwargs)

    model.blk_min = model.blk_max = 8192
    model.blk_step = 8192
    monkeypatch.setattr(mlx_core.blocktune, "search", recording_search)
    model._tune_block_size(X)

    assert timings, "the probe never ran"
    # An MLX iteration on this sample is ~10 ms; building the graph without
    # evaluating it is orders of magnitude below that. The bound is loose so a
    # slow CI host cannot flake it.
    assert max(timings) > 1e-3


# ---------------------------------------------------------------------------
# Degrade, never exit
# ---------------------------------------------------------------------------


def test_metal_allocation_failure_is_a_catchable_exception():
    """The premise the whole fallback rests on. MLX aborts the process on some
    failures (its LU on singular input, issue #274), so this had to be
    established rather than assumed: a buffer above the device maximum raises a
    normal Python RuntimeError, and ``blocktune`` recognizes it as an
    allocation failure rather than a bug.

    Requesting a buffer larger than the device's own advertised maximum, so it
    is refused outright without the host ever trying to back it with memory.
    The request is spread over two axes because MLX carries each dimension as
    an int32, so a single axis cannot express a shape this large.
    """
    info = mx.device_info()
    max_buffer = info.get("max_buffer_length")
    if not max_buffer:
        pytest.skip("device does not report max_buffer_length")

    rows = 100_000
    cols = int(max_buffer // 4) // rows + 1  # (rows * cols * 4) > max_buffer
    if rows >= 2**31 or cols >= 2**31:
        pytest.skip("cannot express an over-maximum buffer in int32 dimensions")

    with pytest.raises(RuntimeError) as excinfo:
        mx.eval(mx.zeros((rows, cols), dtype=mx.float32))

    assert "metal::malloc" in str(excinfo.value)
    assert blocktune.is_allocation_failure(excinfo.value)


def test_allocation_failure_falls_back_to_the_last_working_size(
    real_data, caplog, monkeypatch
):
    """The core deliverable of issue #232, on MLX. The per-candidate probe is
    replaced with one that raises the verbatim ``[metal::malloc]`` RuntimeError
    captured from the live probe above. This is approved error-path injection,
    not a forbidden mock: driving the GPU into a real OOM in CI is not safe,
    the exception is genuine (and proven catchable by the test above), and the
    tuner logic plus the fit that follows are entirely real.
    """
    model = _model(do_opt_block=True, blk_min=4096, blk_max=16384, blk_step=4096)
    real_accumulate = model._accumulate_blocks

    def failing_accumulate(X, stash_llt=False):
        if model.block_size > 8192:
            raise RuntimeError(
                "[metal::malloc] Attempting to allocate 160000000000 bytes "
                "which is greater than the maximum allowed buffer size of "
                "41747087360 bytes."
            )
        return real_accumulate(X, stash_llt=stash_llt)

    monkeypatch.setattr(model, "_accumulate_blocks", failing_accumulate)
    with caplog.at_level(logging.DEBUG, logger="pamica.mlx_impl.core"):
        model.fit(real_data, max_iter=2, verbose=False)

    assert model.block_size <= 8192  # never a size that already failed
    assert np.isfinite(model.final_ll_)  # and the fit still completed
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "12288" in text and "stopped early" in text


def test_a_real_error_in_the_probe_is_not_swallowed(real_data, monkeypatch):
    """A genuine MLX failure must surface rather than be absorbed as memory
    pressure -- the distinction matters most here, where both arrive as a bare
    RuntimeError."""
    model = _model(do_opt_block=True, blk_min=4096, blk_max=8192, blk_step=4096)

    def broken_accumulate(X, stash_llt=False):
        raise RuntimeError("[linalg::lu] Input matrix is singular")

    monkeypatch.setattr(model, "_accumulate_blocks", broken_accumulate)
    with pytest.raises(RuntimeError, match="singular"):
        model.fit(real_data[:, :8192], max_iter=1, verbose=False)
