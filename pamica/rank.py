"""Numerical-rank policy shared by every pamica backend (issue #223).

The Fortran reference detects the numerical rank of the data covariance and
sizes the model to it::

    numeigs = min(pcakeep, count(eigs > mineig))    ! amica15.f90:395
    nw = numeigs                                    ! amica15.f90:545

Rank-deficient input is ordinary in this field -- Maxwell-filtered MEG, average
referencing, channel interpolation all produce it -- so every backend needs the
same answer for "how many dimensions are real?". That decision lives here rather
than being reimplemented per backend, so the backends cannot drift apart
(``.rules/backend_parity.md``).

Only the *decision* is shared. Each backend builds its own sphering matrix in its
own array library, because the PyTorch path is bit-exact against Fortran and must
not be routed through a different eigensolver.
"""

from typing import Optional

import numpy as np
import numpy.typing as npt

# Fortran's absolute covariance-eigenvalue floor (amica15_header.f90:66).
MINEIG = 1e-15

# pamica's relative floor, applied as ``mineig_rel * largest_eigenvalue``. On by
# default, unlike Fortran, which has no relative option; see ADR 0004 and
# ``docs/guides/amica-differences.md``.
MINEIG_REL = 1e-12


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
        (Fortran ``min(pcakeep, ...)``).

    Returns
    -------
    int
        Dimensions to keep; never greater than ``len(evals)``.

    Raises
    ------
    ValueError
        If no eigenvalue clears the threshold, so there is nothing to
        decompose. Fortran would compute ``numeigs = 0`` and carry on into
        undefined behavior.
    """
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
