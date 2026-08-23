"""Host- and data-aware ``block_size`` search, shared by every backend (issue #232).

``block_size`` is pamica's single largest throughput knob (issue #216): every
backend is dispatch-bound at small blocks, and the measured optimum moves with
the host, the device and the data (16384-32768 on CPU for the bundled 32-channel
sample, a single block on Apple GPUs, much smaller at high channel counts). The
shipped default of 8192 is a compromise that has to stay safe at high channel
counts, so it necessarily leaves 16-60% on the table depending on backend.

Fortran has the same idea (``do_opt_block``: time ``blk_min``..``blk_max`` step
``blk_step`` and keep the fastest, ``amica15.f90:2078``) and pamica keeps its
four parameter names, but **not** its failure mode. Fortran's
``determine_block_size`` calls ``allocate_blocks`` with no ``stat=``, so a
candidate that cannot allocate aborts the whole run -- and the search walks
*upward* into larger blocks, which is exactly where memory runs out. That is the
behavior this module exists to fix: a candidate that fails to allocate is
skipped, the search stops walking upward, and the fit continues at the largest
size that actually ran (never at a size already known to fail, and never by
aborting).

Two differences from Fortran are deliberate:

* **Bounds.** Fortran's 128-1024 range sits far below where any pamica backend
  peaks, so the pamica *defaults* are re-derived (:data:`DEFAULT_BLK_MIN` ..
  :data:`DEFAULT_BLK_MAX` step :data:`DEFAULT_BLK_STEP`). The *stepping* stays
  Fortran's arithmetic ``range(blk_min, blk_max + 1, blk_step)``, so a literal
  Fortran ``input.param`` carrying ``blk_min``/``blk_max``/``blk_step`` means the
  same thing on both sides.
* **A memory cap.** Candidates are clamped by an estimate of what one block
  actually costs (:func:`memory_capped_block_size`), so the search does not rely
  solely on catching an allocation failure. Every candidate is also clamped to
  ``n_samples``: a block larger than the data is never useful, and on the Fortran
  side a ``block_size`` above the frames available per thread silently produces
  NaNs (issue #292).

The search is **timing-based and therefore machine-dependent**: two hosts can
pick different sizes and their trajectories then differ at the ~1e-6 level that
any ``block_size`` change produces. It is off by default for that reason, and a
run being compared bit-for-bit against the reference binary must pin
``block_size`` and leave ``do_opt_block`` off. See
``docs/guides/amica-differences.md``.

It also is not free: the sweep costs ``REPEATS * len(candidates)`` accumulate
passes, i.e. about 16 EM iterations' worth under the defaults (measured at 0.54 s
on torch-CPU and 2.8 s on NumPy for the bundled 32-channel sample). That pays for
itself over a normal multi-hundred-iteration fit and does not over a very short
one, which is the other half of why this is opt-in.
"""

from __future__ import annotations

import logging
import math
import os
import re
from typing import Callable, Dict, Optional, Sequence

logger = logging.getLogger(__name__)

# Re-derived sweep bounds (see the module docstring): the arithmetic sequence
# 4096, 8192, ..., 32768 brackets the measured CPU optimum (16384-32768, issue
# #216) and includes the shipped 8192 default, in the same 8 candidates Fortran
# uses for its own 128..1024 sweep. Fortran's defaults (128/1024/128) are kept
# only when a parameter file supplies them explicitly.
DEFAULT_BLK_MIN = 4096
DEFAULT_BLK_MAX = 32768
DEFAULT_BLK_STEP = 4096

# Fraction of the reported available memory one block may be estimated to need.
# Deliberately conservative: the estimate below counts live block tensors, not
# the parameter/accumulator state, the framework's caching allocator, or
# anything else sharing the device, and being wrong in the low direction only
# costs throughput while being wrong in the high direction costs the fit.
MEMORY_BUDGET_FRACTION = 0.25

# Timed passes per candidate; each candidate is scored on its fastest one.
# 2 rather than 1 because the first pass at a given block size pays costs that
# belong to no candidate in particular -- on Metal a new block shape triggers
# shader compilation, which inflated single measurements 4x on an M4 Pro and
# made a cold search pick a block size a warm search beat by 1.4x. 2 rather
# than more because the whole search costs `2 * len(candidates)` passes, which
# has to stay small against the fit it is tuning (16 passes against the 100+
# EM iterations of any real run).
REPEATS = 2

# Live ``(block, n_channels, n_mix)``-shaped tensors at the peak of one block's
# E/M pass, over and above the three (``y``, ``z``, ``|y|^rho``) each model
# keeps for the whole block. Counted off AMICATorchNG._forward /
# _get_block_updates (the MLX and NumPy passes carry the same intermediates):
# the score ``fp``, ``u = v*z``, ``ufp``, ``ufp/y``, ``|y|``, ``rho*ln|y|``,
# and the two Newton curvature terms. An order-of-magnitude guard, not an exact
# accounting -- catching the allocation failure is the real safety net.
_BLOCK_TENSOR_SLOTS = 8

# An allocation failure reaches Python as a different type and message on every
# backend, and on torch/MLX it arrives as a plain RuntimeError -- the same class
# a genuine bug raises. Matching the message is what keeps this module from
# swallowing real errors: anything that is not recognizably an allocation
# failure is re-raised (see :func:`tune_block_size`).
#
# Verified message sources: NumPy raises MemoryError (``_ArrayMemoryError``);
# PyTorch raises ``torch.cuda.OutOfMemoryError`` ("CUDA out of memory") or
# RuntimeError ("MPS backend out of memory", "[enforce fail ...] ... Ran out of
# memory"); MLX raises RuntimeError ("[metal::malloc] Attempting to allocate N
# bytes which is greater than the maximum allowed buffer size of M bytes"),
# confirmed by probe on an M4 Pro under mlx 0.32 -- it is a catchable Python
# exception, not the process abort that MLX's LU decomposition takes (issue
# #274).
_OOM_TEXT = re.compile(
    r"out of memory"
    r"|cannot allocate"
    r"|can't allocate"
    r"|unable to allocate"
    r"|failed to allocate"
    r"|metal::malloc"
    r"|maximum allowed buffer size"
    r"|bad_alloc",
    re.IGNORECASE,
)


def validate_block_tune_params(blk_min: int, blk_max: int, blk_step: int) -> None:
    """Reject a block-size sweep that could not produce a usable candidate.

    Raises ``ValueError`` for a non-positive floor or step (an empty or
    infinite sweep) and for ``blk_max < blk_min`` (an empty range, which would
    otherwise silently leave ``block_size`` untouched and look like the tuner
    had simply chosen the default).
    """
    if blk_min < 1:
        raise ValueError(f"blk_min must be >= 1, got {blk_min}")
    if blk_step < 1:
        raise ValueError(f"blk_step must be >= 1, got {blk_step}")
    if blk_max < blk_min:
        raise ValueError(
            f"blk_max must be >= blk_min, got blk_max={blk_max}, blk_min={blk_min}"
        )


def host_memory_bytes() -> Optional[int]:
    """Total host RAM in bytes, or ``None`` where it cannot be determined.

    ``sysconf`` covers Linux and macOS; anywhere it is missing (Windows) the
    caller simply gets no memory cap and relies on catching the allocation
    failure instead.
    """
    try:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, ValueError, OSError):
        return None


def memory_capped_block_size(
    available_bytes: Optional[int],
    n_channels: int,
    n_mix: int,
    n_models: int,
    itemsize: int,
) -> Optional[int]:
    """Largest block whose estimated peak fits the memory budget.

    One block's peak is estimated as::

        block_size * n_channels * n_mix * itemsize * (3 * n_models + 8)

    i.e. the three ``(block, n_channels, n_mix)`` tensors each model holds for
    the whole block plus :data:`_BLOCK_TENSOR_SLOTS` live intermediates, capped
    at :data:`MEMORY_BUDGET_FRACTION` of ``available_bytes``. Returns ``None``
    when the available memory is unknown, meaning "no cap".
    """
    if not available_bytes or available_bytes <= 0:
        return None
    per_sample = n_channels * n_mix * itemsize * (3 * n_models + _BLOCK_TENSOR_SLOTS)
    if per_sample <= 0:
        return None
    return max(1, int(available_bytes * MEMORY_BUDGET_FRACTION) // per_sample)


def block_size_candidates(
    blk_min: int,
    blk_max: int,
    blk_step: int,
    n_samples: int,
    max_block: Optional[int] = None,
) -> list:
    """Ascending, de-duplicated candidate block sizes for the sweep.

    Fortran's arithmetic stepping (``blk_min``, ``blk_min + blk_step``, ...,
    ``<= blk_max``), with every candidate clamped to ``min(n_samples,
    max_block)`` -- so a sweep whose top end exceeds the data (or the memory
    budget) still *times* the largest usable size instead of dropping it. Never
    empty, and never returns a size above ``n_samples`` (issue #292).
    """
    validate_block_tune_params(blk_min, blk_max, blk_step)
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")
    hard_max = n_samples if max_block is None else min(n_samples, max_block)
    hard_max = max(1, hard_max)
    return sorted(
        {min(size, hard_max) for size in range(blk_min, blk_max + 1, blk_step)}
    )


def is_allocation_failure(exc: BaseException) -> bool:
    """Whether ``exc`` is an out-of-memory failure rather than a genuine bug.

    ``MemoryError`` by type (NumPy's ``_ArrayMemoryError`` subclasses it);
    ``RuntimeError`` by message, because that is the only thing separating
    PyTorch's and MLX's allocation failures from every other RuntimeError they
    raise. ``torch.cuda.OutOfMemoryError`` is a ``RuntimeError`` subclass whose
    message contains "CUDA out of memory", so it needs no special case and no
    import of torch here.
    """
    if isinstance(exc, MemoryError):
        return True
    if isinstance(exc, RuntimeError):
        return bool(_OOM_TEXT.search(str(exc)))
    return False


def tune_block_size(
    probe: Callable[[int], float],
    candidates: Sequence[int],
    *,
    fallback: int,
    repeats: int = REPEATS,
    log: Optional[logging.Logger] = None,
) -> int:
    """Time each candidate with ``probe`` and return the fastest that ran.

    ``probe(block_size)`` runs one representative pass at that block size and
    returns its elapsed seconds. It must leave no state behind: the timed
    passes are throwaway work, and the fit that follows has to be bit-identical
    to one started directly at the chosen size.

    Each candidate is probed ``repeats`` times and scored on its *fastest*
    pass. One pass each is not enough: the first pass at a given block size
    pays one-off costs that belong to no candidate in particular -- most
    sharply on Metal, where a new block shape triggers shader compilation and
    can inflate a single measurement several-fold. Taking the minimum discards
    that, at the cost of ``repeats`` passes per candidate; see
    :data:`REPEATS`.

    The sweep walks ``candidates`` in ascending order so a working size is
    always in hand before a larger one is attempted. On an allocation failure
    it stops walking upward -- every larger candidate would fail too -- and
    returns the fastest size that completed, or ``fallback`` if none did. A
    failed candidate is an expected outcome of searching upward, so it is logged
    at DEBUG; the truncation itself gets one summary WARNING, since silently
    tuning within a range the caller did not get is worth one line.

    An exception that is not recognizably an allocation failure
    (:func:`is_allocation_failure`) is re-raised: this must not become a place
    where a real bug in an E-step is mistaken for memory pressure.
    """
    log = log or logger
    if not candidates:
        raise ValueError("candidates must not be empty")
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")

    timings: Dict[int, float] = {}
    best_size = fallback
    best_time = math.inf
    exhausted_at: Optional[int] = None

    for size in candidates:
        try:
            elapsed = min(probe(size) for _ in range(repeats))
        except Exception as exc:
            if not is_allocation_failure(exc):
                raise
            log.debug(
                "Block-size search: candidate %d could not allocate (%s: %s); "
                "not searching further upward.",
                size,
                type(exc).__name__,
                exc,
            )
            exhausted_at = size
            break
        timings[size] = elapsed
        if elapsed < best_time:
            best_time, best_size = elapsed, size

    if exhausted_at is not None:
        log.warning(
            "Block-size search stopped early: block_size=%d could not be "
            "allocated, so candidates above it were not tried. Continuing at "
            "block_size=%d, the fastest size that ran. This is the expected "
            "outcome on a memory-constrained device, not a failure.",
            exhausted_at,
            best_size,
        )

    if not timings:
        log.warning(
            "Block-size search timed no candidate; keeping block_size=%d.",
            fallback,
        )
        return fallback

    log.info(
        "Block-size search: chose block_size=%d (%.1f ms/pass) from %s",
        best_size,
        best_time * 1e3,
        ", ".join(f"{size}:{sec * 1e3:.1f}ms" for size, sec in sorted(timings.items())),
    )
    return best_size


def search(
    *,
    probe: Callable[[int], float],
    fallback: int,
    blk_min: int,
    blk_max: int,
    blk_step: int,
    n_samples: int,
    n_channels: int,
    n_mix: int,
    n_models: int,
    itemsize: int,
    available_bytes: Optional[int],
    log: Optional[logging.Logger] = None,
) -> int:
    """Build the candidate list and run the sweep: the one call each backend makes.

    ``fallback`` is the statically configured ``block_size``, returned unchanged
    if nothing could be timed. ``itemsize`` and ``available_bytes`` are the only
    backend-specific inputs to the memory cap (element width, and whatever the
    device reports as available; ``None`` disables the cap).
    """
    cap = memory_capped_block_size(
        available_bytes, n_channels, n_mix, n_models, itemsize
    )
    candidates = block_size_candidates(
        blk_min, blk_max, blk_step, n_samples, max_block=cap
    )
    return tune_block_size(probe, candidates, fallback=fallback, log=log)
