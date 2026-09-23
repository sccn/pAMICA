"""Iteration-schedule gates shared by every pamica backend (issue #335).

The Fortran reference counts its iterations from 1 (``iter = 1`` before the main
loop, amica15.f90:949) and states every schedule against that counter:
``iter .ge. newt_start`` switches Newton on, ``iter == rejstart`` fires the
first rejection, ``mod(iter, writestep) == 0`` writes a checkpoint. pamica's fit
loops count from 0 (``for it in range(max_iter)``), and that 0-based index is
public (``iteration``, the ``mir_history_`` entries, log lines), so it stays as
it is. Every gate instead converts it through :func:`reference_iteration` here,
in one place, so the three backends cannot drift apart and no gate can compare
the 0-based index with a 1-based setting again: the Newton switch did exactly
that until issue #335, and started one iteration late in all three backends.

Only the *decisions* live here: each backend still runs its own loop and calls
these predicates with its own counters (``.rules/backend_parity.md``). So does
the validation of the settings they read (:func:`validate_iteration_setting`),
which every backend constructor calls so the three reject the same values with
the same message.
"""

from __future__ import annotations

import numbers

# The reference holds the A-update for the merge iteration and the 5 after it
# (``mod(iter, share_iter) > 5``, amica15.f90:1803).
_SHARE_FREEZE_ITERS = 6


def reference_iteration(index: int) -> int:
    """The reference's 1-based iteration number of a 0-based loop index."""
    return index + 1


def validate_iteration_setting(name: str, value: object, minimum: int) -> None:
    """Raise ``ValueError`` unless ``value`` is an integer of at least ``minimum``.

    For the settings that name an iteration or an interval (``newt_start``,
    ``rejstart``, ``restartiter``, ``maxrestarts``, ``histstep``,
    ``scalestep``): an integer in
    the ``numbers.Integral`` sense, so numpy integers qualify, while ``bool``
    (which subclasses ``int``) and floats are rejected rather than coerced, as
    :func:`pamica.rank.validate_pca_reduction` does for ``pcakeep``.
    """
    if (
        isinstance(value, bool)
        or not isinstance(value, numbers.Integral)
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")


def _check_interval(where: str, name: str, interval: int) -> None:
    # A modulo by a non-positive interval is a bare ZeroDivisionError (0) or a
    # silently wrong cadence (negative). The constructors validate the settings
    # that reach these helpers; this is the backstop for one that did not.
    if interval < 1:
        raise ValueError(
            f"{where}: {name} must be >= 1 (a schedule interval), got {interval!r}"
        )


def newton_active(do_newton: bool, index: int, newt_start: int) -> bool:
    """Whether this iteration takes the Newton step (``iter .ge. newt_start``,
    amica15.f90:1498/1651/1666/1719/1739/1804).

    ``newt_start`` counts from 1, as the reference's does: ``newt_start=50``
    takes the first Newton step on the 50th iteration, and ``newt_start=1`` on
    the first. ``newt_start=0`` fits exactly as ``1`` does: Newton is active
    from the first iteration either way, and the two differ only in gates that
    cannot act on iteration 1 (:func:`newton_switches_on` clears a decrease
    counter that is still 0 there, and no ``maxdecs`` ratchet can precede the
    first likelihood comparison).
    """
    return do_newton and reference_iteration(index) >= newt_start


def newton_switches_on(do_newton: bool, index: int, newt_start: int) -> bool:
    """Whether this is the iteration Newton switches on, where the reference
    clears the likelihood-decrease counter (``iter == newt_start``,
    amica15.f90:1099-1102)."""
    return do_newton and reference_iteration(index) == newt_start


def past_newton_start(index: int, newt_start: int) -> bool:
    """Whether a ``maxdecs`` ratchet on this iteration also tightens the rho-rate
    ceiling and, under Newton, ``newtrate`` (``iter > newt_start``,
    amica15.f90:1067/1070). The rho-rate half applies whether or not Newton is
    on; callers add ``do_newton`` for the ``newtrate`` half."""
    return reference_iteration(index) > newt_start


def rejection_due(
    do_reject: bool, index: int, rejstart: int, rejint: int, numrej: int, maxrej: int
) -> bool:
    """Whether outlier rejection fires after this iteration's update
    (amica15.f90:1136).

    The ``max(1, ...)`` clamp is the reference's: without it, a non-negative
    modulo would make ``iter - rejstart`` hit 0 before ``rejstart`` and fire
    early. The ``iter == rejstart`` arm fires regardless of ``numrej``, as in
    the reference.
    """
    if not do_reject or maxrej <= 0:
        return False
    itf = reference_iteration(index)
    return itf == rejstart or (max(1, itf - rejstart) % rejint == 0 and numrej < maxrej)


def periodic_due(index: int, start: int, interval: int) -> bool:
    """Whether a schedule that runs every ``interval`` iterations from ``start``
    fires on this iteration: the share-merge pass (``iter .ge. share_start``
    and ``mod(iter - share_start, share_iter) == 0``, amica15.f90:1856), and the
    extended-Infomax kurtosis switch on its ``kurt_start``/``kurt_int`` schedule
    (a pamica port with no runnable reference, whose ``do_choose_pdfs`` is dead
    code, so it reads the same way)."""
    _check_interval("periodic_due", "interval", interval)
    itf = reference_iteration(index)
    return itf >= start and (itf - start) % interval == 0


def share_freeze(index: int, share_start: int, share_iter: int) -> bool:
    """Whether ``share_comps`` holds the A-update on this iteration: the merge
    iteration and the 5 after it, in every ``share_iter`` cycle from
    ``share_start``.

    Anchored on ``share_start`` so the window stays aligned with the merge
    schedule (:func:`periodic_due`). The reference's literal
    ``mod(iter, share_iter) > 5`` (amica15.f90:1803) misaligns unless
    ``share_start`` is a multiple of ``share_iter``; that path is dead in the
    reference, so this is a documented pamica decision
    (``docs/guides/amica-differences.md``), not a parity constraint. Callers
    gate it on ``share_comps`` with more than one model.
    """
    itf = reference_iteration(index)
    if itf < share_start:
        return False
    return (itf - share_start) % share_iter < _SHARE_FREEZE_ITERS


def every(index: int, step: int) -> bool:
    """Whether a ``mod(iter, step) == 0`` cadence fires on this iteration
    (``writestep``/``histstep``, amica15.f90:1124/1130; also pamica's
    ``scalestep``, which the reference parses but never reads): the first time
    at iteration ``step``, not after the first iteration."""
    _check_interval("every", "step", step)
    return reference_iteration(index) % step == 0


def within_restart_window(index: int, restartiter: int) -> bool:
    """Whether a non-finite likelihood on this iteration may still restart from
    a fresh draw (``iter .le. restartiter``, amica15.f90:1022): the first
    ``restartiter`` iterations, so ``restartiter=0`` disables the recovery."""
    return reference_iteration(index) <= restartiter
