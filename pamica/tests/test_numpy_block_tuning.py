"""Block-size auto-tuner on the NumPy backend (issue #232).

Real sample EEG only. This backend is where the feature has history: it already
exposed ``do_opt_block``/``blk_min``/``blk_max``/``blk_step`` and shipped a
``determine_block_size`` helper that timed a bare ``X.T @ X`` (the shape of no
work AMICA actually does) over Fortran's 128-1024 range, on by default, with no
fallback when an allocation failed. Those four names are kept; everything behind
them is replaced by the shared :mod:`pamica.blocktune` search.

Assertions are machine-robust: which block size wins is a timing outcome, so
nothing here asserts a particular winner, only that it is a valid candidate and
that having chosen it changed nothing about the fit.
"""

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pamica import AMICA_NumPy as AMICA
from pamica import blocktune
from pamica.numpy_impl.data import load_data_file

_FDT = Path(__file__).resolve().parent.parent / "sample_data" / "eeglab_data.fdt"

pytestmark = pytest.mark.skipif(not _FDT.exists(), reason="sample data missing")


class _Collector(logging.Handler):
    """Collects records off the backend's own logger.

    This backend configures a private ``"AMICA"`` logger with ``propagate =
    False``, so pytest's ``caplog`` (which listens on the root logger) never
    sees its output; the handler has to be attached directly.
    """

    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.messages: list = []

    def emit(self, record):
        self.messages.append(record.getMessage())


@contextmanager
def _captured(model: AMICA):
    handler = _Collector()
    previous_level = model.logger.level
    model.logger.setLevel(logging.DEBUG)
    model.logger.addHandler(handler)
    try:
        yield handler
    finally:
        model.logger.removeHandler(handler)
        model.logger.setLevel(previous_level)


def _real_data(n_samples: int) -> np.ndarray:
    """A slice of the committed sample EEG (32 channels), float64."""
    data = load_data_file(str(_FDT), 32, 30504, dtype=np.float32)
    return data[:, :n_samples].astype(np.float64)


def _model(**kwargs: Any) -> AMICA:
    params: dict[str, Any] = dict(
        num_models=1, num_mix=3, max_iter=2, seed=42, use_tqdm=False
    )
    params.update(kwargs)
    return AMICA(**params)


RANK = 20


def _rank_deficient(n_samples: int) -> np.ndarray:
    """Real EEG projected onto its top-``RANK`` subspace (what SSS does to MEG),
    the established rank-reduction route in ``torch_tests/test_ng_rank_deficient.py``."""
    data = _real_data(n_samples)
    centered = data - data.mean(axis=1, keepdims=True)
    U = np.linalg.svd(centered, full_matrices=False)[0][:, :RANK]
    return U @ (U.T @ centered)


def _recorded_search(monkeypatch) -> dict:
    """Capture the arguments the backend hands ``blocktune.search``: after
    preprocessing they are the model's REDUCED dimensions, which is how the
    ordering of the search relative to preprocessing becomes observable."""
    from pamica.numpy_impl import core as np_core

    recorded: dict = {}
    real_search = blocktune.search

    def recording_search(**kwargs):
        recorded.update(kwargs)
        return real_search(**kwargs)

    monkeypatch.setattr(np_core.blocktune, "search", recording_search)
    return recorded


# ---------------------------------------------------------------------------
# Defaults and inertness
# ---------------------------------------------------------------------------


def test_search_is_now_off_by_default():
    """The default flipped with issue #232. It used to be on (Fortran's header
    default), which meant every NumPy fit silently re-tuned itself over
    128-1024 and ignored the block_size it was given -- and made a parity run
    depend on the host it ran on."""
    model = _model()
    assert model.do_opt_block is False
    assert model.block_size == 8192


def test_bare_default_fit_runs_at_the_shipped_block_size():
    """Direct pin on the shipped default, on the object a user actually gets.
    ``params.json`` had pinned block_size to 128, which went unnoticed only
    because the always-on tuner overrode it every time; with the tuner off that
    would have shipped as a silent ~9x slowdown. Nothing here passes
    ``block_size`` or ``do_opt_block`` -- ``max_iter`` only keeps the test
    short, since the bare default is 2000 iterations."""
    model = AMICA(use_tqdm=False, max_iter=2)
    assert model.block_size == 8192
    assert model.do_opt_block is False

    model.fit(_real_data(4000))
    assert model.block_size == 8192


def test_tuner_runs_on_the_reduced_channel_count(monkeypatch):
    """Ordering pin: the search runs after preprocessing, so on rank-reduced
    input it sees the kept rank rather than the input channel count -- which is
    what the memory cap has to be sized against."""
    recorded = _recorded_search(monkeypatch)
    model = _model(do_opt_block=True, blk_min=2048, blk_max=8192, blk_step=2048)
    model.fit(_rank_deficient(9000))

    assert model.data_dim == RANK < 32  # reduction really happened
    assert recorded["n_channels"] == RANK
    assert model.block_size in (2048, 4096, 6144, 8192)


def test_default_sweep_bounds_match_the_other_backends():
    model = _model()
    assert (model.blk_min, model.blk_max, model.blk_step) == (
        blocktune.DEFAULT_BLK_MIN,
        blocktune.DEFAULT_BLK_MAX,
        blocktune.DEFAULT_BLK_STEP,
    )


def test_block_size_untouched_when_search_is_off():
    model = _model(block_size=1024)
    model.fit(_real_data(4000))
    assert model.block_size == 1024


def test_sweep_bounds_are_not_validated_when_search_is_off():
    """A Fortran input.param can carry blk_* alongside do_opt_block=0; that file
    must stay loadable, since nothing reads the values."""
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


def test_naive_helper_is_gone_not_deprecated():
    """Replaced, not shimmed: the old helper timed the wrong operation over the
    wrong range and could not fall back at all."""
    from pamica.numpy_impl import utils

    assert not hasattr(utils, "determine_block_size")


# ---------------------------------------------------------------------------
# The search itself
# ---------------------------------------------------------------------------


def test_chosen_block_size_is_a_valid_candidate_and_is_logged():
    data = _real_data(12000)
    model = _model(do_opt_block=True, blk_min=2048, blk_max=8192, blk_step=2048)
    with _captured(model) as log:
        model.fit(data)

    assert model.block_size in blocktune.block_size_candidates(
        2048, 8192, 2048, n_samples=data.shape[1]
    )
    message = "\n".join(log.messages)
    assert f"chose block_size={model.block_size}" in message


def test_candidates_never_exceed_the_sample_count():
    """A block larger than the data is useless, and Fortran silently NaNs when
    block_size exceeds the frames available per thread (issue #292)."""
    data = _real_data(3000)
    model = _model(do_opt_block=True, blk_min=4096, blk_max=32768, blk_step=4096)
    model.fit(data)
    assert model.block_size <= data.shape[1]


# ---------------------------------------------------------------------------
# The tuner must not change the fit
# ---------------------------------------------------------------------------


def test_post_tune_fit_is_bit_identical_to_a_direct_fit():
    """The tuner's contract: its timed passes are throwaway work. A fit that
    tuned its way to size N must be bit-for-bit the fit handed N directly. On
    this backend the probe also writes ``_last_ll_samples`` under do_reject, so
    the restore in ``_tune_block_size`` is what this pins down."""
    data = _real_data(12000)
    tuned = _model(do_opt_block=True, blk_min=2048, blk_max=8192, blk_step=2048)
    tuned.fit(data.copy())

    direct = _model(block_size=tuned.block_size)
    direct.fit(data.copy())

    np.testing.assert_array_equal(tuned.A, direct.A)
    np.testing.assert_array_equal(tuned.W, direct.W)
    assert tuned.ll == direct.ll


def test_post_tune_fit_is_bit_identical_under_do_reject():
    """Same contract with rejection on, the one path where the probe writes
    model state (``_last_ll_samples``) that must not survive it."""
    data = _real_data(9000)
    common = dict(max_iter=4, do_reject=True, rejstart=1, rejint=1, maxrej=2, seed=42)
    tuned = _model(
        do_opt_block=True, blk_min=2048, blk_max=4096, blk_step=2048, **common
    )
    tuned.fit(data.copy())

    direct = _model(block_size=tuned.block_size, **common)
    direct.fit(data.copy())

    np.testing.assert_array_equal(tuned.A, direct.A)
    assert tuned.ll == direct.ll
    assert tuned.num_good_samples == direct.num_good_samples


# ---------------------------------------------------------------------------
# Degrade, never exit
# ---------------------------------------------------------------------------


def test_allocation_failure_falls_back_to_the_last_working_size(monkeypatch):
    """The core deliverable of issue #232: Fortran's ``determine_block_size``
    allocates with no ``stat=`` and aborts the run when a candidate does not
    fit. Here the failing candidate is skipped and the fit continues at the
    largest size that ran.

    The per-candidate probe is replaced with one that raises the genuine NumPy
    ``MemoryError`` above a threshold. This is approved error-path injection,
    not a forbidden mock: a real OOM cannot be provoked safely in CI, the
    exception is the verbatim article NumPy raises, and all of the tuner logic
    under test -- the ascending walk, the fallback, the state the model is left
    in -- plus the fit that follows are entirely real.
    """
    data = _real_data(12000)
    model = _model(do_opt_block=True, blk_min=2048, blk_max=8192, blk_step=2048)
    real_pass = model._get_updates_and_likelihood

    def failing_pass():
        if model.block_size > 4096:
            raise MemoryError(
                "Unable to allocate 64.0 GiB for an array with shape "
                "(8192, 32, 3) and data type float64"
            )
        return real_pass()

    monkeypatch.setattr(model, "_get_updates_and_likelihood", failing_pass)
    with _captured(model) as log:
        model.fit(data)

    assert model.block_size <= 4096  # never a size that already failed
    assert np.isfinite(model.ll[-1])  # and the fit still completed
    text = "\n".join(log.messages)
    assert "6144" in text and "stopped early" in text


def test_a_real_error_in_the_probe_is_not_swallowed(monkeypatch):
    """A genuine bug during a probe must surface rather than be absorbed as
    memory pressure and reported as a tuned block size."""
    model = _model(do_opt_block=True, blk_min=2048, blk_max=4096, blk_step=2048)

    def broken_pass():
        raise RuntimeError("singular matrix in the E-step")

    monkeypatch.setattr(model, "_get_updates_and_likelihood", broken_pass)
    with pytest.raises(RuntimeError, match="singular matrix"):
        model.fit(_real_data(6000))
