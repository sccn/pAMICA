"""Unit tests for the shared block-size search (``pamica.blocktune``, issue #232).

The backend-level tests (``torch_tests/test_ng_block_tuning.py``,
``test_numpy_block_tuning.py``, ``mlx_tests/test_mlx_blocktune.py``) drive this
module through a real fit on real sample EEG. What is tested here is the search
logic itself -- candidate construction, clamping, and above all the OOM
degrade-never-exit contract -- which needs no data at all, only a callable that
returns a duration.

The probes here are plain timing callables, not stand-ins for AMICA: this module
is *defined* as "run this callable per candidate and keep the fastest", so a
callable is its real input, not a mock of one. Where a probe raises, it raises
the allocation error the backend genuinely raises (verified strings, see
``blocktune._OOM_TEXT``); the tuner logic under test is entirely real.
"""

import logging

import pytest

from pamica import blocktune


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "blk_min, blk_max, blk_step, match",
    [
        (0, 1024, 128, "blk_min must be >= 1"),
        (-4096, 1024, 128, "blk_min must be >= 1"),
        (128, 1024, 0, "blk_step must be >= 1"),
        (128, 1024, -8, "blk_step must be >= 1"),
        (2048, 1024, 128, "blk_max must be >= blk_min"),
    ],
)
def test_validate_rejects_unusable_sweeps(blk_min, blk_max, blk_step, match):
    """A sweep that cannot produce a candidate is rejected up front rather than
    silently leaving block_size at its default, which would look exactly like a
    search that had run and chosen the default."""
    with pytest.raises(ValueError, match=match):
        blocktune.validate_block_tune_params(blk_min, blk_max, blk_step)


def test_validate_accepts_fortran_and_pamica_defaults():
    """Both the pamica defaults and Fortran's own 128/1024/128 are valid: a
    literal input.param must stay loadable."""
    blocktune.validate_block_tune_params(
        blocktune.DEFAULT_BLK_MIN, blocktune.DEFAULT_BLK_MAX, blocktune.DEFAULT_BLK_STEP
    )
    blocktune.validate_block_tune_params(128, 1024, 128)


# ---------------------------------------------------------------------------
# Candidate construction
# ---------------------------------------------------------------------------


def test_candidates_use_fortran_arithmetic_stepping():
    """blk_step keeps Fortran's additive meaning (amica15.f90:2084), so a
    param file carrying it means the same thing on both sides."""
    assert blocktune.block_size_candidates(128, 1024, 128, n_samples=10**6) == [
        128,
        256,
        384,
        512,
        640,
        768,
        896,
        1024,
    ]


def test_default_sweep_brackets_the_measured_optimum():
    """The re-derived defaults span the measured CPU optimum (16384-32768,
    issue #216) and include the shipped 8192 default, instead of Fortran's
    128-1024, which is entirely below where any pamica backend peaks."""
    cands = blocktune.block_size_candidates(
        blocktune.DEFAULT_BLK_MIN,
        blocktune.DEFAULT_BLK_MAX,
        blocktune.DEFAULT_BLK_STEP,
        n_samples=10**6,
    )
    assert cands == [4096, 8192, 12288, 16384, 20480, 24576, 28672, 32768]


def test_candidates_are_clamped_to_n_samples():
    """No candidate may exceed the data: a larger block is never useful, and on
    the Fortran side a block_size above the frames per thread silently produces
    NaNs (issue #292). The over-range candidates collapse onto n_samples rather
    than being dropped, so the largest usable size is still timed."""
    cands = blocktune.block_size_candidates(4096, 32768, 4096, n_samples=10000)
    assert max(cands) == 10000
    assert cands == [4096, 8192, 10000]


def test_candidates_clamped_when_floor_exceeds_data():
    """A sweep entirely above the data collapses to one candidate at
    n_samples, not to an empty list (which would leave nothing to time)."""
    assert blocktune.block_size_candidates(4096, 32768, 4096, n_samples=500) == [500]


def test_candidates_are_capped_by_the_memory_estimate():
    """The search is bounded by available memory, not only by the fixed range
    (issue #232), so it does not have to walk into an allocation failure to
    find its ceiling."""
    cands = blocktune.block_size_candidates(
        4096, 32768, 4096, n_samples=10**6, max_block=9000
    )
    assert max(cands) == 9000
    assert cands == [4096, 8192, 9000]


def test_candidates_reject_empty_data():
    with pytest.raises(ValueError, match="n_samples must be >= 1"):
        blocktune.block_size_candidates(128, 1024, 128, n_samples=0)


def test_memory_cap_shrinks_as_channels_and_models_rise():
    """The useful maximum falls as channels and models rise (issue #232), which
    is exactly why a single fixed default cannot be right everywhere."""
    small = blocktune.memory_capped_block_size(
        8 * 1024**3, n_channels=32, n_mix=3, n_models=1, itemsize=8
    )
    many_channels = blocktune.memory_capped_block_size(
        8 * 1024**3, n_channels=256, n_mix=3, n_models=1, itemsize=8
    )
    many_models = blocktune.memory_capped_block_size(
        8 * 1024**3, n_channels=32, n_mix=3, n_models=4, itemsize=8
    )
    assert small is not None and many_channels is not None
    assert many_models is not None
    assert small > many_channels
    assert small > many_models
    assert many_channels >= 1


def test_memory_cap_is_none_when_memory_is_unknown():
    """No reported memory means no cap, not a cap of zero."""
    assert blocktune.memory_capped_block_size(None, 32, 3, 1, 8) is None
    assert blocktune.memory_capped_block_size(0, 32, 3, 1, 8) is None


def test_host_memory_is_positive_or_unavailable():
    mem = blocktune.host_memory_bytes()
    assert mem is None or mem > 0


# ---------------------------------------------------------------------------
# The OOM predicate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exc",
    [
        MemoryError("Unable to allocate 4.00 GiB for an array"),
        RuntimeError(
            "[metal::malloc] Attempting to allocate 160000000000 bytes which is "
            "greater than the maximum allowed buffer size of 41747087360 bytes."
        ),
        RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"),
        RuntimeError("MPS backend out of memory (MPS allocated: 9.00 GB)"),
        RuntimeError("[enforce fail at alloc_cpu.cpp:117] . Ran out of memory"),
        RuntimeError("std::bad_alloc"),
        # Not a verified wording; the narrow "allocate <n> <unit>" arm exists
        # for messages this project has not seen.
        RuntimeError("failed to allocate 68719476736 bytes"),
    ],
)
def test_recognizes_every_backend_allocation_failure(exc):
    """The verbatim messages each backend raises. The MLX one was captured from
    a live probe on an M4 Pro under mlx 0.32: MLX raises a catchable Python
    RuntimeError here, it does not abort the process the way its LU does on
    singular input (issue #274)."""
    assert blocktune.is_allocation_failure(exc)


@pytest.mark.parametrize(
    "exc",
    [
        RuntimeError("mat1 and mat2 shapes cannot be multiplied"),
        RuntimeError("linalg.inv: The diagonal element 3 is zero"),
        ValueError("block_size must be positive"),
        ZeroDivisionError("division by zero"),
        # "Allocate" alone does not mean memory: a smaller block size fixes
        # none of these, so treating them as memory pressure would silently
        # degrade the fit instead of surfacing the real failure.
        RuntimeError("failed to allocate device id 3"),
        RuntimeError("cannot allocate a file handle for the output stream"),
        RuntimeError("unable to allocate port 8080"),
    ],
)
def test_does_not_mistake_a_real_bug_for_memory_pressure(exc):
    """The whole reason the predicate matches on message: on torch and MLX an
    allocation failure is a plain RuntimeError, the same class a genuine bug
    raises. Treating every RuntimeError as OOM would silently swallow bugs."""
    assert not blocktune.is_allocation_failure(exc)


# ---------------------------------------------------------------------------
# The search itself
# ---------------------------------------------------------------------------


def test_picks_the_fastest_candidate():
    timings = {4096: 0.05, 8192: 0.02, 16384: 0.03}
    assert (
        blocktune.tune_block_size(
            lambda size: timings[size], [4096, 8192, 16384], fallback=512
        )
        == 8192
    )


def test_logs_the_choice_and_every_timing_at_info(caplog):
    """The choice is machine-dependent, so a run has to be able to say which
    size it picked and what it measured (issue #232 policy)."""
    timings = {4096: 0.05, 8192: 0.02, 16384: 0.03}
    with caplog.at_level(logging.INFO, logger="pamica.blocktune"):
        blocktune.tune_block_size(
            lambda size: timings[size], [4096, 8192, 16384], fallback=512
        )
    message = "\n".join(r.getMessage() for r in caplog.records)
    assert "chose block_size=8192" in message
    for size in timings:
        assert str(size) in message


def test_scores_each_candidate_on_its_fastest_pass():
    """Every candidate's first pass pays a one-off cost that belongs to no
    candidate in particular (on Metal, shader compilation for a new block
    shape, measured at ~4x on an M4 Pro). Scoring on the minimum discards it.
    Here the genuinely-faster size is charged 10x on its first touch and must
    still win, which it cannot do if the first measurement is the one kept."""
    calls: list = []

    def probe(size: int) -> float:
        calls.append(size)
        steady = 0.01 if size == 8192 else 0.02
        return steady * 10 if calls.count(size) == 1 else steady

    assert blocktune.tune_block_size(probe, [4096, 8192], fallback=512) == 8192
    assert calls == [4096, 4096, 8192, 8192]  # REPEATS passes per candidate


def test_repeats_must_be_at_least_one():
    with pytest.raises(ValueError, match="repeats must be >= 1"):
        blocktune.tune_block_size(lambda size: 0.0, [4096], fallback=512, repeats=0)


def test_falls_back_to_last_working_size_on_allocation_failure(caplog):
    """The core deliverable (issue #232): the reference implementation aborts
    the whole run when a candidate cannot allocate. Here the failing candidate
    is skipped, the upward walk stops (every larger candidate would fail too),
    and the fit continues at the largest size that actually ran.

    The probe raises the exact MemoryError NumPy raises. This is approved
    error-path injection, not a forbidden mock: a real OOM cannot be provoked
    safely in CI (MLX's memory limit is documented as a soft guideline and
    NumPy/torch offer no portable cap), the raised exception is the genuine
    article, and every line of search logic under test runs for real.
    """
    fine = {4096: 0.05, 8192: 0.02}

    def probe(size: int) -> float:
        if size >= 16384:
            raise MemoryError(
                f"Unable to allocate {size} bytes for an array with shape (32, 3)"
            )
        return fine[size]

    with caplog.at_level(logging.DEBUG, logger="pamica.blocktune"):
        chosen = blocktune.tune_block_size(
            probe, [4096, 8192, 16384, 32768], fallback=512
        )

    assert chosen == 8192  # fastest of the two that ran, never a failed size
    messages = [r.getMessage() for r in caplog.records]
    joined = "\n".join(messages)
    # The failed candidate is an expected outcome of searching upward, so it is
    # reported at DEBUG rather than as a defect-looking warning...
    assert any(
        "16384" in m and r.levelno == logging.DEBUG
        for m, r in zip(messages, caplog.records)
    )
    # ...while the truncation itself gets exactly one summary warning.
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "stopped early" in joined
    assert "not a failure" in joined


def test_does_not_try_larger_candidates_after_a_failure():
    """Memory demand grows monotonically with block size, so once one candidate
    fails every larger one would too; trying them wastes time and risks pushing
    an already-pressured host further."""
    attempted: list = []

    def probe(size: int) -> float:
        attempted.append(size)
        if size >= 8192:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return 0.01

    assert blocktune.tune_block_size(probe, [4096, 8192, 16384], fallback=512) == 4096
    assert 16384 not in attempted


def test_keeps_the_static_default_when_nothing_can_be_timed(caplog):
    """Even the smallest candidate failing must not abort: the fit continues at
    the configured block_size, which is the size the caller already asked for."""

    def probe(size: int) -> float:
        raise MemoryError("Unable to allocate 4.00 GiB for an array")

    with caplog.at_level(logging.WARNING, logger="pamica.blocktune"):
        assert blocktune.tune_block_size(probe, [4096, 8192], fallback=8192) == 8192
    assert "keeping the configured block_size=8192 untested" in "\n".join(
        r.getMessage() for r in caplog.records
    )


def test_first_candidate_failure_does_not_claim_a_size_ran(caplog):
    """Every other failure test fails at a LATER candidate, where a working
    size really is in hand. When the first (or only) candidate fails there is
    none: `best_size` is still the untouched fallback, so the truncation
    wording ("continuing at N, the fastest size that ran") would assert that a
    size ran when nothing did, and would fire a second, contradictory warning
    alongside it. One warning, and it must not claim a measurement."""

    def probe(size: int) -> float:
        raise MemoryError("Unable to allocate 4.00 GiB for an array")

    with caplog.at_level(logging.DEBUG, logger="pamica.blocktune"):
        assert blocktune.tune_block_size(probe, [4096], fallback=8192) == 8192

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "the first candidate (4096) could not be allocated" in message
    assert "keeping the configured block_size=8192 untested" in message
    assert "the fastest size that ran" not in message


def test_a_non_allocation_error_propagates():
    """A bug in an E-step must not be quietly absorbed as memory pressure and
    turned into a slightly different block size -- the one failure mode a
    permissive `except Exception` here would create."""

    def probe(size: int) -> float:
        raise RuntimeError("linalg.inv: The diagonal element 3 is zero")

    with pytest.raises(RuntimeError, match="diagonal element"):
        blocktune.tune_block_size(probe, [4096, 8192], fallback=512)


def test_a_non_allocation_error_stops_on_the_very_first_pass():
    """Same contract on the first pass of the first candidate: nothing is
    retried in the hope that the second attempt behaves."""
    calls: list = []

    def probe(size: int) -> float:
        calls.append(size)
        raise ValueError("something is actually wrong")

    with pytest.raises(ValueError, match="actually wrong"):
        blocktune.tune_block_size(probe, [4096], fallback=512)
    assert calls == [4096]


def test_empty_candidate_list_is_a_programming_error():
    with pytest.raises(ValueError, match="must not be empty"):
        blocktune.tune_block_size(lambda size: 0.0, [], fallback=512)


def test_search_composes_the_cap_and_the_sweep():
    """`search` is the single call each backend makes: it must apply both the
    memory cap and the n_samples clamp before timing anything."""
    seen: list = []

    def probe(size: int) -> float:
        seen.append(size)
        return 1.0 / size  # bigger is faster, so the cap decides the winner

    chosen = blocktune.search(
        probe=probe,
        fallback=512,
        blk_min=4096,
        blk_max=32768,
        blk_step=4096,
        n_samples=20000,
        n_channels=32,
        n_mix=3,
        n_models=1,
        itemsize=8,
        available_bytes=32 * 1024**2,  # tiny budget: the cap, not n_samples, binds
    )
    cap = blocktune.memory_capped_block_size(32 * 1024**2, 32, 3, 1, 8)
    assert cap is not None and cap < 20000
    assert chosen == max(seen) <= cap
