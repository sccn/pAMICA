"""Numerical-rank policy shared by every pamica backend (issue #223).

The Fortran reference detects the numerical rank of the data covariance and
sizes the model to it::

    numeigs = min(pcakeep, count(eigs > mineig))    ! amica15.f90:413
    nw = numeigs                                    ! amica15.f90:563

Rank-deficient input is ordinary in this field -- Maxwell-filtered MEG, average
referencing, channel interpolation all produce it -- so every backend needs the
same answer for "how many dimensions are real?". That decision lives here rather
than being reimplemented per backend, so the backends cannot drift apart
(``.rules/backend_parity.md``).

Only the *decision* is shared. Each backend builds its own sphering matrix in its
own array library, because the PyTorch path is bit-exact against Fortran and must
not be routed through a different eigensolver.

The explicit PCA-reduction request (``pcakeep``/``pcadb``) is part of the same
decision, so its validation and its "is a reduction requested?" predicate live
here too (issue #323): every backend constructor calls
:func:`validate_pca_reduction` and :func:`log_ignored_pca_request`,
:func:`numerical_rank` re-checks at fit time, and the upfront ``mir_step`` gates
ask :func:`pca_reduction_requested`. ``pcadb`` is a pamica extension: the
reference parses it (amica15.f90:3459-3461) but never reads it again, so only
``pcakeep`` enters its ``numeigs``.
"""

import logging
import math
import numbers
from typing import Optional

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

# Fortran's absolute covariance-eigenvalue floor (amica15_header.f90:66).
MINEIG = 1e-15

# pamica's relative floor, applied as ``mineig_rel * largest_eigenvalue``. On by
# default, unlike Fortran, which has no relative option; see ADR 0004 and
# ``docs/guides/amica-differences.md``.
MINEIG_REL = 1e-12


def validate_pca_reduction(pcakeep: Optional[int], pcadb: Optional[float]) -> None:
    """Reject an explicit PCA-reduction request that cannot mean anything.

    Called by every backend constructor, so a bad value fails before any data
    is touched, and again at fit time by :func:`numerical_rank` and
    :func:`pca_reduction_requested`, which catch an attribute reassigned after
    construction. Validation only: it never logs, so the fit-time re-checks
    stay silent (the constructors log through :func:`log_ignored_pca_request`).

    Parameters
    ----------
    pcakeep : int, optional
        Number of principal dimensions to keep. Must be ``None`` or an integer
        (``numbers.Integral``, so numpy integers qualify) of at least 1.
        ``bool`` and ``numpy.bool_`` are rejected (``bool`` subclasses
        ``int``), and a float is rejected rather than truncated. Unvalidated,
        ``pcakeep=-3`` sliced from the end and fitted 29 of 32 dimensions,
        ``pcakeep=2.7`` silently kept 2, and ``pcakeep=0`` built an empty model
        that ended in ``nan_ll``.
    pcadb : float, optional
        Keep the dimensions whose eigenvalue lies within ``pcadb`` dB of the
        largest. Must be ``None`` or a finite real number (``numbers.Real``, so
        numpy floats qualify; ``bool`` and ``numpy.bool_`` rejected) greater
        than 0; ``pcadb <= 0`` kept nothing and ended in ``nan_ll``.

    Raises
    ------
    ValueError
        Naming the parameter and the offending value.

    Notes
    -----
    Setting both is allowed: ``pcakeep`` takes precedence and ``pcadb`` is
    ignored, the order :func:`numerical_rank` applies them in. This matches
    the reference, which parses ``pcadb`` (amica15.f90:3459-3461) but never
    uses it, so a Fortran ``input.param`` that sets both (as both bundled
    parameter files do) means its ``pcakeep``.
    """
    if pcakeep is not None and (
        isinstance(pcakeep, bool)
        or not isinstance(pcakeep, numbers.Integral)
        or pcakeep < 1
    ):
        raise ValueError(f"pcakeep must be None or an integer >= 1, got {pcakeep!r}")
    if pcadb is not None and (
        isinstance(pcadb, bool)
        or not isinstance(pcadb, numbers.Real)
        or not math.isfinite(pcadb)
        or pcadb <= 0
    ):
        raise ValueError(
            f"pcadb must be None or a finite real number > 0, got {pcadb!r}"
        )


def log_ignored_pca_request(
    pcakeep: Optional[int], pcadb: Optional[float], do_sphere: bool
) -> None:
    """Log the part of a valid ``pcakeep``/``pcadb`` request that a fit will
    ignore. Called once by every backend constructor, after
    :func:`validate_pca_reduction`, so each note appears once per model.

    * ``do_sphere=False``: one WARNING that both are ignored. Reduction only
      happens while sphering, as in the reference, whose no-sphere branch keeps
      every dimension (``numeigs = nx``, amica15.f90:527).
    * Otherwise, both set: one INFO line that ``pcadb`` is ignored because
      ``pcakeep`` takes precedence (see :func:`validate_pca_reduction`).

    Parameters
    ----------
    pcakeep, pcadb : int, float, optional
        The validated request.
    do_sphere : bool
        The backend's ``do_sphere`` setting.
    """
    requested = [
        f"{name}={value!r}"
        for name, value in (("pcakeep", pcakeep), ("pcadb", pcadb))
        if value is not None
    ]
    if not requested:
        return
    if not do_sphere:
        logger.warning(
            "%s ignored because do_sphere=False: PCA reduction happens only "
            "while sphering, as in the reference, which keeps every dimension "
            "when it does not sphere (numeigs = nx, amica15.f90:527).",
            " and ".join(requested),
        )
    elif pcakeep is not None and pcadb is not None:
        logger.info(
            "pcakeep=%d and pcadb=%g are both set; pcadb is ignored because "
            "pcakeep takes precedence (as in the reference, which parses pcadb "
            "but never uses it).",
            pcakeep,
            pcadb,
        )


def pca_reduction_requested(
    pcakeep: Optional[int],
    pcadb: Optional[float],
    n_channels: int,
    do_sphere: bool,
) -> bool:
    """Whether an explicit ``pcakeep``/``pcadb`` asks to fit fewer than
    ``n_channels`` dimensions.

    Config-only: it runs before any eigenvalue exists, for the upfront
    ``mir_step`` gates, so it answers what the *request* implies, not what
    the fit will find. ``pcakeep >= n_channels`` is not a request (the
    reference's ``min(pcakeep, ...)`` keeps every dimension the data have,
    e.g. the bundled ``input.param``'s ``pcakeep 32`` on 32 channels); any
    ``pcadb`` is one, because whether it cuts anything depends on the
    eigenvalues; and ``pcakeep`` wins when both are set, as in
    :func:`numerical_rank`. Without sphering nothing is a request, because no
    backend reduces then (see :func:`log_ignored_pca_request`). Rank reduction
    from automatic detection (``mineig``/``mineig_rel``) is not a request and
    cannot be known here; the backends catch it once the fitted sphere exists.

    Validates first, so a ``pcakeep``/``pcadb`` reassigned after construction
    raises the validator's ``ValueError`` here rather than a ``TypeError``.

    Parameters
    ----------
    pcakeep, pcadb : int, float, optional
        As :func:`validate_pca_reduction`.
    n_channels : int
        Channel count of the data being fitted.
    do_sphere : bool
        The backend's ``do_sphere`` setting.
    """
    validate_pca_reduction(pcakeep, pcadb)
    if not do_sphere:
        return False
    if pcakeep is not None:
        return pcakeep < n_channels
    return pcadb is not None


def numerical_rank(
    evals: npt.ArrayLike,
    *,
    mineig: float = MINEIG,
    mineig_rel: Optional[float] = MINEIG_REL,
    pcakeep: Optional[int] = None,
    pcadb: Optional[float] = None,
) -> int:
    """Number of data-covariance eigen-directions to keep.

    Parameters
    ----------
    evals : array_like
        Covariance eigenvalues in **descending** order.
    mineig : float
        Absolute eigenvalue floor (Fortran ``mineig``). Used only when
        ``mineig_rel`` is ``None``.
    mineig_rel : float or None
        Relative floor, as a fraction of the largest eigenvalue. When set (the
        default) it *replaces* ``mineig`` rather than combining with it: a
        relative floor for MEG in Tesla lands near 1e-35 in absolute terms, so
        taking the larger of the two would silently discard it. Pass ``None`` to
        reproduce Fortran's absolute-only behavior exactly.
    pcakeep, pcadb : int or float, optional
        Explicit PCA reduction, capped by the detected rank
        (Fortran ``min(pcakeep, ...)``). Validated first by
        :func:`validate_pca_reduction`; ``pcakeep`` takes precedence and
        ``pcadb`` is ignored when both are set.

    Returns
    -------
    int
        Dimensions to keep; never greater than ``len(evals)``.

    Raises
    ------
    ValueError
        If ``pcakeep``/``pcadb`` is invalid (see
        :func:`validate_pca_reduction`), or if no eigenvalue clears the
        threshold, so there is nothing to decompose. Fortran would compute
        ``numeigs = 0`` and carry on into undefined behavior.
    """
    # The fit-time choke point every backend passes through, so a pcakeep or
    # pcadb reassigned after construction cannot slip past the constructors'
    # checks. Validation only: the constructor already logged any notes.
    validate_pca_reduction(pcakeep, pcadb)
    ev = np.asarray(evals, dtype=np.float64)
    if ev.ndim != 1 or ev.size == 0:
        raise ValueError(
            f"evals must be a non-empty 1-D sequence, got shape {ev.shape}"
        )

    if not np.isfinite(ev).all():
        # NaN/Inf in the data. Rank detection is meaningless, and `nan > thresh`
        # is False, which would masquerade as rank zero. Keep every dimension and
        # let the caller's degenerate-fit handling report the real problem
        # (issue #50: `nan_ll`, model marked unusable) rather than raising a
        # different error from preprocessing.
        return int(ev.size)

    thresh = mineig if mineig_rel is None else mineig_rel * float(ev[0])
    n_rank = int((ev > thresh).sum())
    if n_rank < 1:
        raise ValueError(
            f"No data covariance eigenvalue exceeds the rank threshold "
            f"{thresh:g} (largest is {float(ev[0]):g}), so the numerical rank is "
            "zero and there is nothing to decompose. With mineig_rel=None the "
            "threshold is the absolute Fortran floor, which is unit-dependent: "
            "MEG in Tesla gives eigenvalues ~1e-26 and falls below it. Rescale "
            "the data, or set mineig/mineig_rel to suit its units."
        )

    if pcakeep is not None:
        return min(int(pcakeep), n_rank)
    if pcadb is not None:
        db = 10.0 * np.log10(ev / ev[0])
        return min(int((db > -pcadb).sum()), n_rank)
    return n_rank
