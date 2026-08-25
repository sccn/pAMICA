"""Best-of-N random restarts, shared by every array backend (issue #198).

AMICA's natural-gradient EM is a local optimizer, so which basin a fit lands in
is decided by the random initialization. On well-determined components that does
not matter -- issue #145 showed ``AMICATorchNG`` and the Fortran binary agree to
0.9974 mean correlation from an *identical* init -- but the weakest,
under-determined components of a high-channel fit are basin-sensitive: seed 42 on
the full 70-channel/k=152 recording put 7-10 of them in a different
(equal-or-higher-likelihood) basin than the reference reached from its own init.
Running the same data from several seeds and keeping the highest-likelihood fit
is the standard answer, and it extends issue #51's ``keep_best`` (best iterate
*within* a run) across runs.

**This is a pamica extension, not a parity feature.** The Fortran reference has
no restart loop of this kind: its ``maxrestarts``/``restartiter`` machinery
(amica15.f90:1022-1052, ported to the NumPy backend as ``numrestarts``) is a
*recovery* path that redraws the mixing matrix after an early non-finite
likelihood, not a search over seeds, and it never compares two completed fits.
``n_restarts=1`` is therefore the default in every backend, and on that path the
restart machinery does nothing at all: no extra RNG draw, no state copy, no
change of any kind to the trajectory. See ``docs/guides/amica-differences.md``.

Only the *policy* lives here -- seed derivation, the selection rule, and the log
lines -- so the three backends cannot drift apart (``.rules/backend_parity.md``).
Each backend keeps its own fit loop and its own state-copy list, because what
counts as per-fit state is genuinely backend-specific (the PyTorch backend has an
LLt stash and MIR waypoints, the MLX backend a lazy-graph gradient norm, the
NumPy backend Newton buffers and an on-disk write cadence).
"""

from __future__ import annotations

import copy
import math
from typing import Any, List, Optional, Sequence

import numpy as np

# Parity-preserving default: one restart is exactly the pre-#198 fit.
DEFAULT_N_RESTARTS = 1

# The records every backend leaves on a fitted model: index-aligned lists of the
# seed each restart ran from, the log-likelihood it returned (NaN where it ended
# degenerate) and why it stopped. Always populated, single-restart fits included,
# so a caller never has to special-case which mode produced the model. These
# describe the *search*, not any one restart, so they are written once after the
# loop and are never part of a restart snapshot.
RECORD_ATTRS = ("restart_seeds_", "restart_lls_", "restart_stop_reasons_")

# ``stop_reason`` for a restart that RAISED instead of stopping (issue #198
# review). A truly singular mixing matrix makes the unmixing inversion raise --
# ``torch.linalg.LinAlgError`` (a ``RuntimeError``), ``numpy.linalg.LinAlgError``
# (a ``ValueError``), or the MLX backend's own condition-number guard
# (``RuntimeError``, issue #274) -- rather than produce the non-finite
# likelihood the in-loop guards catch. Inside a best-of-N search that must not
# discard the restarts that already succeeded, so each restart's fit is wrapped,
# the failure is recorded under this reason, and the search continues.
#
# It is a DEGENERATE stop reason on every backend (it is in each backend's
# ``_DEGENERATE_STOP_REASONS``; the NumPy backend also sets ``converged=False``),
# so a crashed restart is excluded from selection, and a search in which every
# restart crashed leaves a model the degenerate-fit contract (issue #50) refuses
# to transform or persist -- exactly like a search in which every restart
# diverged. Distinct from ``singular_ll`` on purpose: that one means "the fit
# ran and reported a non-finite likelihood", this one means "the fit could not
# run to completion at all", and a user reading ``restart_stop_reasons_``
# deserves to see which happened.
#
# Only the multi-restart path catches. A single fit (``n_restarts=1``) still
# raises exactly as it did before this feature existed -- bit-identity includes
# error behavior.
ERROR_STOP_REASON = "restart_error"


def resolve_seeds(
    n_restarts: int,
    restart_seeds: Optional[Sequence[Optional[int]]],
    seed: Optional[int],
) -> List[Optional[int]]:
    """Validate a restart configuration and return one seed per restart.

    Called from every backend's constructor, so a bad configuration fails before
    any data is touched rather than mid-fit.

    Parameters
    ----------
    n_restarts : int
        Number of independent fits to run. Must be >= 1.
    restart_seeds : sequence of int, optional
        Explicit per-restart seeds. Must have exactly ``n_restarts`` entries.
    seed : int, optional
        The model's base seed, used to derive the seeds when ``restart_seeds``
        is not given: ``seed, seed + 1, ..., seed + n_restarts - 1``.

    Returns
    -------
    list
        The seed each restart will initialize from. For ``n_restarts == 1``
        without explicit seeds this is ``[seed]`` unchanged -- including
        ``[None]``, which is what keeps the single-restart path bit-identical to
        a fit that never heard of restarts.

    Raises
    ------
    ValueError
        On ``n_restarts < 1``, a ``restart_seeds`` length mismatch, or
        ``n_restarts > 1`` with neither a base ``seed`` nor explicit seeds --
        the last because best-of-N is only meaningful if the winning fit can be
        reproduced, which requires knowing the seed it ran from. Deriving one
        from the clock or from OS entropy would make the winner
        irreproducible, so this is refused loudly instead.
    """
    if isinstance(n_restarts, bool) or not isinstance(n_restarts, (int, np.integer)):
        raise TypeError(f"n_restarts must be an int, got {type(n_restarts).__name__}")
    n_restarts = int(n_restarts)
    if n_restarts < 1:
        raise ValueError(f"n_restarts must be >= 1, got {n_restarts}")

    if restart_seeds is not None:
        seeds = list(restart_seeds)
        if len(seeds) != n_restarts:
            raise ValueError(
                f"restart_seeds has {len(seeds)} entries but n_restarts is "
                f"{n_restarts}; pass exactly one seed per restart (or omit "
                f"restart_seeds to derive them from seed)."
            )
        resolved: List[Optional[int]] = []
        for value in seeds:
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError(
                    f"restart_seeds entries must be ints, got {type(value).__name__}"
                )
            resolved.append(int(value))
        return resolved

    if n_restarts == 1:
        return [seed]

    if seed is None:
        raise ValueError(
            "n_restarts > 1 requires a reproducible starting point: pass "
            "seed=<int> (restarts then run seed, seed+1, ...) or "
            "restart_seeds=[...]. Without one, the winning restart could not "
            "be reproduced, and pamica will not seed itself from the clock."
        )
    return [int(seed) + offset for offset in range(n_restarts)]


def select_best(
    lls: Sequence[Optional[float]], degenerate: Sequence[bool]
) -> Optional[int]:
    """Index of the winning restart: the highest finite log-likelihood among the
    non-degenerate restarts, or ``None`` if every restart was degenerate.

    ``lls`` holds each restart's *returned-iterate* log-likelihood (the PyTorch
    and MLX backends' ``final_ll_``, the NumPy backend's ``ll[-1]``), so under
    issue #51's ``keep_best`` the comparison is between the iterates the fits
    actually return, not the last iterates they passed through. Ties keep the
    earlier restart, so the choice is deterministic.

    A degenerate restart (non-finite likelihood or non-finite parameters) is
    excluded from selection but still recorded by the caller: it is a fact about
    the seed, not something to hide.
    """
    if len(lls) != len(degenerate):
        raise ValueError(
            f"lls has {len(lls)} entries but degenerate has {len(degenerate)}; "
            "the per-restart records must stay aligned."
        )
    best: Optional[int] = None
    best_ll = -math.inf
    for index, (ll, is_degenerate) in enumerate(zip(lls, degenerate)):
        if is_degenerate or ll is None or not math.isfinite(ll):
            continue
        if ll > best_ll:
            best_ll = ll
            best = index
    return best


def copy_state_value(value: Any) -> Any:
    """Independent copy of one piece of per-fit state.

    Handles what the array backends actually hold: ``None``, Python scalars,
    lists (``ll_history``/``mir_history_``), NumPy arrays, ``RandomState`` (the
    NumPy backend's ``rng``) and anything with a ``clone()`` (PyTorch tensors).
    Unknown types raise rather than being aliased through -- an aliased snapshot
    would silently roll forward with the next restart's in-place updates, which
    is exactly the failure this copy exists to prevent. The MLX backend wraps
    this with its own ``mx.array`` case.
    """
    if value is None or isinstance(
        value, (int, float, bool, str, np.integer, np.floating)
    ):
        return value
    if isinstance(value, list):
        return list(value)
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, np.random.RandomState):
        return copy.deepcopy(value)
    clone = getattr(value, "clone", None)
    if callable(clone):
        return clone()
    raise TypeError(
        f"restarts.copy_state_value cannot copy {type(value).__name__}; add an "
        f"explicit case rather than letting a restart snapshot alias live state."
    )


def progress_message(
    index: int,
    n_restarts: int,
    seed: Optional[int],
    ll: Optional[float],
    stop_reason: Optional[str],
    degenerate: bool,
) -> str:
    """One line per completed restart (INFO), identical on every backend."""
    ll_text = "n/a" if ll is None else f"{ll:.6f}"
    return (
        f"Restart {index + 1}/{n_restarts} (seed={seed}): final LL {ll_text}, "
        f"stop_reason={stop_reason!r}"
        + (" -- DEGENERATE, excluded from selection" if degenerate else "")
    )


def winner_message(
    index: int, n_restarts: int, seed: Optional[int], ll: Optional[float]
) -> str:
    """The one line naming the winner (INFO), identical on every backend."""
    ll_text = "n/a" if ll is None else f"{ll:.6f}"
    return (
        f"Best-of-{n_restarts} restarts: restart {index + 1} (seed={seed}) wins "
        f"with final LL {ll_text}; its parameters are the fitted model (issue #198)."
    )


def error_message(
    index: int, n_restarts: int, seed: Optional[int], exc: BaseException
) -> str:
    """One line for a restart that raised (WARNING), identical on every backend.

    Names the exception type and message, so a caught failure is as diagnosable
    in the log as an uncaught one would have been in a traceback.
    """
    return (
        f"Restart {index + 1}/{n_restarts} (seed={seed}) raised "
        f"{type(exc).__name__}: {exc}; recorded as {ERROR_STOP_REASON!r} "
        f"(degenerate, excluded from selection) and continuing with the next "
        f"seed. A single-restart fit would have raised this instead."
    )


def all_degenerate_message(n_restarts: int, stop_reasons: Sequence[Any]) -> str:
    """Emitted when no restart produced a usable fit.

    Logged at WARNING on every backend (the caller convention, unified in the
    #198 review): the run still returns a model and the degenerate-fit contract
    is what refuses it downstream, so this is the same severity as the
    single-fit degenerate warnings it sits alongside -- not an ERROR on one
    backend and a WARNING on the others.
    """
    return (
        f"All {n_restarts} restarts ended degenerate ({list(stop_reasons)}); "
        f"returning the last restart, which the degenerate-fit contract "
        f"(issue #50) refuses to transform or persist. Lower lrate, disable "
        f"Newton, or check data conditioning."
    )
