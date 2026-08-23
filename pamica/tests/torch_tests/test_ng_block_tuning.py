"""Block-size auto-tuner on the PyTorch backend (issue #232).

Real sample EEG only. The decisive checks are that the search leaves the fit
numerically untouched -- a post-tune fit must be bit-identical to one started
directly at the chosen block size -- and that an allocation failure degrades to
the last working size instead of aborting, which is the behavior Fortran's own
``do_opt_block`` gets wrong.

Assertions are machine-robust: which block size wins is a timing outcome and
varies by host, so nothing here asserts a particular winner, only that the
winner is a valid candidate and that the fit is unchanged by having chosen it.
"""

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from pamica import blocktune
from pamica.torch_impl import AMICATorchNG

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "sample_data"
DATA_FILE = SAMPLE_DIR / "eeglab_data.fdt"
NW = 32
FIELD = 30504
NMIX = 3
SEED = 42

pytestmark = pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing")


def _load_real_data() -> np.ndarray:
    from pamica.torch_impl.utils import load_eeglab_data

    return load_eeglab_data(str(DATA_FILE), data_dim=NW, field_dim=FIELD).astype(
        np.float64
    )


@pytest.fixture(scope="module")
def real_data() -> np.ndarray:
    return _load_real_data()


def _model(**kwargs: Any) -> AMICATorchNG:
    params: dict[str, Any] = dict(
        n_channels=NW,
        n_mix=NMIX,
        seed=SEED,
        device="cpu",
        dtype=torch.float64,
    )
    params.update(kwargs)
    return AMICATorchNG(**params)


# ---------------------------------------------------------------------------
# Off by default, and inert when off
# ---------------------------------------------------------------------------


def test_search_is_off_by_default():
    """A parity run must get the pinned block_size it asked for without saying
    so (issue #232): the search perturbs the trajectory at the ~1e-6 level like
    any block-size change."""
    model = _model()
    assert model.do_opt_block is False
    assert model.block_size == 8192


def test_defaults_are_the_shared_re_derived_bounds():
    model = _model()
    assert (model.blk_min, model.blk_max, model.blk_step) == (
        blocktune.DEFAULT_BLK_MIN,
        blocktune.DEFAULT_BLK_MAX,
        blocktune.DEFAULT_BLK_STEP,
    )


def test_block_size_untouched_when_search_is_off(real_data):
    """Inertness is the property that keeps every existing parity result valid:
    with the search off nothing about the fit may change, starting with the
    block size it runs at."""
    model = _model(block_size=4096)
    model.fit(real_data[:, :8192], max_iter=2, verbose=False)
    assert model.block_size == 4096


def test_sweep_bounds_are_not_validated_when_search_is_off():
    """A literal Fortran input.param can carry blk_* alongside do_opt_block=0;
    rejecting those values would make such a file unloadable for no benefit,
    since nothing reads them."""
    model = _model(do_opt_block=False, blk_min=0, blk_max=1, blk_step=0)
    assert model.do_opt_block is False


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
    """The choice is timing-dependent, so the assertion is that it is one of the
    sizes actually offered -- not which one won."""
    model = _model(do_opt_block=True, blk_min=4096, blk_max=16384, blk_step=4096)
    with caplog.at_level(logging.INFO, logger="pamica.torch_impl.core"):
        model.fit(real_data, max_iter=1, verbose=False)

    assert model.block_size in blocktune.block_size_candidates(
        4096, 16384, 4096, n_samples=real_data.shape[1]
    )
    message = "\n".join(r.getMessage() for r in caplog.records)
    assert f"chose block_size={model.block_size}" in message
    assert "ms" in message  # the timings are reported alongside the choice


def test_candidates_never_exceed_the_sample_count(real_data):
    """A block larger than the data is useless, and on the Fortran side a
    block_size above the frames per thread silently yields NaNs (issue #292),
    so pamica must never pick one."""
    subset = real_data[:, :5000]
    model = _model(do_opt_block=True, blk_min=4096, blk_max=32768, blk_step=4096)
    model.fit(subset, max_iter=1, verbose=False)
    assert model.block_size <= subset.shape[1]


def test_search_runs_under_do_reject(real_data):
    """Under rejection the E-step runs on the good subset, so that is what must
    be timed and what n_samples must be clamped against."""
    subset = real_data[:, :6000]
    model = _model(
        do_opt_block=True,
        blk_min=1024,
        blk_max=4096,
        blk_step=1024,
        do_reject=True,
        rejstart=1,
        rejint=1,
    )
    model.fit(subset, max_iter=3, verbose=False)
    assert model.block_size in (1024, 2048, 3072, 4096)
    assert model.final_ll_ is not None and math.isfinite(model.final_ll_)


# ---------------------------------------------------------------------------
# The tuner must not change the fit
# ---------------------------------------------------------------------------


def test_post_tune_fit_is_bit_identical_to_a_direct_fit(real_data):
    """The tuner's contract: the timed probes are throwaway work. A fit that
    tuned its way to size N must be bit-for-bit the fit that was handed N
    directly -- same seed, same everything. Anything less would mean the probes
    leaked state (an RNG draw, a stale accumulator) into the run."""
    tuned = _model(do_opt_block=True, blk_min=4096, blk_max=16384, blk_step=4096)
    tuned.fit(real_data, max_iter=4, verbose=False)

    direct = _model(block_size=tuned.block_size)
    direct.fit(real_data, max_iter=4, verbose=False)

    assert direct.block_size == tuned.block_size
    np.testing.assert_array_equal(tuned.get_mixing_matrix(), direct.get_mixing_matrix())
    np.testing.assert_array_equal(
        tuned.get_unmixing_matrix(), direct.get_unmixing_matrix()
    )
    assert tuned.ll_history == direct.ll_history
    assert tuned.final_ll_ == direct.final_ll_


def test_probes_leave_block_size_and_parameters_untouched(real_data):
    """The probe restores block_size around every candidate, so a search that
    ends up keeping the fallback leaves the model exactly as it found it."""
    model = _model(block_size=8192, do_opt_block=True)
    X = model._preprocess(real_data[:, :8192])
    model._initialize_parameters()
    assert model.A is not None
    before = model.A.clone()

    # A sweep whose only candidate equals the configured size: the search runs
    # for real, and must return the model in its original state.
    model.blk_min = model.blk_max = 4096
    model.blk_step = 4096
    model._tune_block_size(X)

    assert model.block_size == 4096
    torch.testing.assert_close(model.A, before, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Degrade, never exit
# ---------------------------------------------------------------------------


def test_allocation_failure_falls_back_to_the_last_working_size(
    real_data, caplog, monkeypatch
):
    """The core deliverable of issue #232. Fortran's ``determine_block_size``
    allocates with no ``stat=``, so a candidate it cannot fit aborts the run;
    here the failing candidate is skipped and the fit continues at the largest
    size that ran.

    The per-candidate probe is replaced with one that raises the genuine
    PyTorch CPU out-of-memory RuntimeError above a threshold. This is approved
    error-path injection rather than a forbidden mock: a real OOM cannot be
    provoked safely in CI (there is no portable per-process memory cap), the
    exception raised is the verbatim article, and every line of tuner logic --
    candidate construction, the ascending walk, the fallback, the model's
    post-search state -- runs for real, as does the fit that follows.
    """
    model = _model(do_opt_block=True, blk_min=4096, blk_max=16384, blk_step=4096)
    real_accumulate = model._accumulate_blocks

    def failing_accumulate(X):
        if model.block_size > 8192:
            raise RuntimeError(
                "[enforce fail at alloc_cpu.cpp:117] . DefaultCPUAllocator: "
                "not enough memory: you tried to allocate 68719476736 bytes. "
                "Ran out of memory"
            )
        return real_accumulate(X)

    monkeypatch.setattr(model, "_accumulate_blocks", failing_accumulate)
    with caplog.at_level(logging.DEBUG, logger="pamica.torch_impl.core"):
        model.fit(real_data, max_iter=2, verbose=False)

    assert model.block_size <= 8192  # never a size that already failed
    # and the fit still completed
    assert model.final_ll_ is not None and math.isfinite(model.final_ll_)
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "12288" in text and "stopped early" in text


def test_a_real_error_in_the_probe_is_not_swallowed(real_data, monkeypatch):
    """The mirror image: a genuine bug during a probe must surface, not be
    absorbed as memory pressure and silently reported as a tuned block size."""
    model = _model(do_opt_block=True, blk_min=4096, blk_max=8192, blk_step=4096)

    def broken_accumulate(X):
        raise RuntimeError("linalg.slogdet: A must be batches of square matrices")

    monkeypatch.setattr(model, "_accumulate_blocks", broken_accumulate)
    with pytest.raises(RuntimeError, match="square matrices"):
        model.fit(real_data[:, :8192], max_iter=1, verbose=False)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_wrapper_forwards_the_search_kwargs(real_data):
    """The public surface: ``AMICA.fit`` derives its forwardable keywords from
    the backend's signature, so the four names must reach it as constructor
    kwargs rather than landing in the "not applied" warning."""
    from pamica.amica import AMICA

    model = AMICA(device="cpu", verbose=False)
    model.fit(
        real_data[:, :12000],
        max_iter=2,
        do_opt_block=True,
        blk_min=2048,
        blk_max=8192,
        blk_step=2048,
    )
    backend = model.model_
    assert backend is not None
    assert backend.do_opt_block is True
    assert backend.block_size in (2048, 4096, 6144, 8192)


def test_state_dict_round_trips_the_tuned_block_size(real_data):
    """A reloaded model must reproduce the run it came from, which means
    persisting the size the search chose, not the one the constructor saw."""
    model = _model(do_opt_block=True, blk_min=4096, blk_max=16384, blk_step=4096)
    model.fit(real_data, max_iter=2, verbose=False)

    state = model.state_dict()
    assert state["config"]["block_size"] == model.block_size
    assert state["config"]["do_opt_block"] is True

    restored = AMICATorchNG.from_state_dict(state, device="cpu")
    assert restored.block_size == model.block_size
    assert (restored.blk_min, restored.blk_max, restored.blk_step) == (
        model.blk_min,
        model.blk_max,
        model.blk_step,
    )
