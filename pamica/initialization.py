"""The reference's initial mixing matrix, shared by every array backend
(issue #341, epic #324 Phase 12).

The reference draws each model's ``nw x nw`` block of ``A`` uniformly around
zero, sets the block's diagonal to one and divides every component by its
Euclidean norm (amica15.f90:805-823); its restart after a non-finite likelihood
redraws the same way (:1026-1044). With ``fix_init`` it starts from the
identity instead, whose components already have unit norm. A loaded ``A``
(``load_A``, :793-802) is used as is, without normalization.

Every array backend (PyTorch, NumPy, MLX) draws its initial ``A`` here, in
float64, so the three agree bit for bit (MLX casts the result to float32), and
each draws from its NumPy generator in the order it always did: one ``n x n``
uniform block per model, before ``mu`` and ``beta``. A backend that honors a
supplied ``A`` (the NumPy backend initializes a parameter only while it is
``None``) never calls this for it, so a supplied ``A`` is not normalized either.

The draw itself cannot match the reference's: the binary draws from gfortran's
``random_number``, pamica from ``numpy.random.RandomState``. Both fill component
``i`` from the ``i``-th run of ``n`` values in the stream (the reference's
``Wtmp`` is column-major and its columns are components; pamica's draw is
row-major and its rows are components, ADR 0007).
"""

from __future__ import annotations

import numpy as np

# The reference's off-diagonal draw is 0.01 * (0.5 - u), u uniform on [0, 1):
# every off-diagonal entry of a block lies in (-0.005, 0.005] before the
# normalization (amica15.f90:814).
OFF_DIAGONAL_SCALE = 0.01


def draw_initial_block(rng: np.random.RandomState, n: int) -> np.ndarray:
    """One model's ``n x n`` block before normalization.

    ``0.01 * (0.5 - u)`` from one ``rng.rand(n, n)`` call, with the diagonal
    then set to exactly one (amica15.f90:812-816). Row ``i`` is component ``i``.
    """
    block = OFF_DIAGONAL_SCALE * (0.5 - rng.rand(n, n))
    block[np.diag_indices(n)] = 1.0
    return block


def normalize_components(block: np.ndarray) -> np.ndarray:
    """``block`` with every row (component) divided by its Euclidean norm,
    ``A(:,k) / sqrt(sum(A(:,k)*A(:,k)))`` in the reference (amica15.f90:818-819).

    The reference has no zero-norm guard here, and needs none: a block from
    :func:`draw_initial_block` has a unit diagonal, so every norm is at least 1.
    A direct call on any other block could still divide by a zero or
    non-finite norm, so that raises instead of returning NaN rows; it cannot
    happen on the reference's path, so the arithmetic there is unchanged.

    Raises
    ------
    ValueError
        If a row's norm is zero or not finite.
    """
    norm = np.sqrt(np.sum(block * block, axis=1))
    bad = np.flatnonzero(~(np.isfinite(norm) & (norm > 0)))
    if bad.size:
        raise ValueError(
            f"cannot normalize component row(s) {bad.tolist()}: their norm is "
            "zero or not finite"
        )
    return block / norm[:, None]


def initial_mixing(
    rng: np.random.RandomState, n: int, n_models: int, *, fix_init: bool = False
) -> np.ndarray:
    """The initial component-row ``A`` of shape ``(n_models * n, n)``.

    Model ``h``'s block is rows ``h*n`` to ``(h+1)*n - 1``, the default
    ``comp_list``. For each model in turn (amica15.f90:805-823): with
    ``fix_init`` the identity, drawing nothing; otherwise
    :func:`draw_initial_block` normalized by :func:`normalize_components`.

    Parameters
    ----------
    rng : numpy.random.RandomState
        The generator to draw from; it advances by ``n * n`` values per model
        (none with ``fix_init``).
    n : int
        Components per model (the kept channel count).
    n_models : int
        Number of models.
    fix_init : bool, default False
        Start every model from the identity, as the reference's ``fix_init``.

    Returns
    -------
    ndarray of shape (n_models * n, n), float64
        Every row has unit norm.
    """
    A = np.zeros((n_models * n, n))
    for h in range(n_models):
        rows = slice(h * n, (h + 1) * n)
        if fix_init:
            A[rows, :] = np.eye(n)
        else:
            A[rows, :] = normalize_components(draw_initial_block(rng, n))
    return A
