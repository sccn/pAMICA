"""Component-row storage of the mixing matrix, shared by every array backend
(issue #334, epic #324 Phase 8, ADR 0007).

Every array backend (PyTorch, NumPy, MLX) stores the mixing matrix ``A`` with
shape ``(n_comps, n)``, one ROW per component: ``A[k, :]`` is component ``k``'s
mixing vector in the sphered space, the reference's column ``A(:, k)``
(amica15.f90), and its density lives at ``mu[:, k]``, ``beta[:, k]`` (the
reference's ``sbeta``), ``alpha[:, k]`` and ``rho[:, k]``. ``comp_list[i, h]``
names the component source ``i`` of model ``h`` uses, so model ``h``'s ``n x n``
block is ``A[comp_list[:, h], :]``, whose row ``i`` is that source's mixing
vector. The backends invert that block for ``W`` (issue #24 convention: the
block is the transpose of the reference's ``A(:, comp_list(:, h))``). Globally,
``A`` is the reference's ``A`` transposed.

Before issue #334 the backends stored ``A`` with shape ``(n, n_comps)`` and took
model ``h``'s block as ``A[:, comp_list[:, h]]``: the same ``n x n`` matrix, but
with the component ids indexing stored COLUMNS. A stored column held one
sphered channel's loadings across a model's components, not a component, so
``share_comps`` compared and tied the wrong vectors. For an unmerged
``comp_list`` (every component id used exactly once) that layout holds exactly
the same per-model blocks, so :func:`rows_from_legacy_columns` converts it
without loss. A merged one was computed under the wrong semantics and has no
counterpart in this layout; it is refused with a message to refit.

Only the conversion policy lives here (``.rules/backend_parity.md``): the
PyTorch and MLX backends both call it when they read a pre-#334 save, so they
convert and refuse identically. The NumPy backend has no saved-model format of
its own.
"""

from __future__ import annotations

import numpy as np


def rows_from_legacy_columns(
    A_legacy: np.ndarray, comp_list: np.ndarray, *, owner: str
) -> np.ndarray:
    """Convert a pre-#334 ``A`` of shape ``(n, n_comps)`` to component rows.

    Model ``h``'s block is carried over element for element:
    ``A_rows[comp_list[:, h], :] == A_legacy[:, comp_list[:, h]]``, so every
    unmixing matrix, source and log-likelihood of the converted model is
    exactly the saved one's.

    Parameters
    ----------
    A_legacy : ndarray of shape (n, n_comps)
        The mixing matrix as a pre-#334 ``state_dict`` stored it.
    comp_list : ndarray of int, shape (n, n_models)
        The saved component assignments; ``n_comps == n * n_models``.
    owner : str
        The class name the error messages name (``"AMICATorchNG"``, ...).

    Returns
    -------
    ndarray of shape (n_comps, n)
        The component-row ``A``, in ``A_legacy``'s dtype.

    Raises
    ------
    ValueError
        If the shapes disagree, a component id is out of range or repeated
        within one model (a malformed payload), or a component id is shared by
        two or more models: a ``share_comps`` merge computed on stored columns
        (issue #334), which cannot be converted and needs a refit.
    """
    A_legacy = np.asarray(A_legacy)
    comp_list = np.asarray(comp_list)
    if comp_list.ndim != 2:
        raise ValueError(
            f"malformed {owner} state: comp_list has shape {comp_list.shape}, "
            "expected (n_channels, n_models)"
        )
    n, n_models = comp_list.shape
    n_comps = n * n_models
    if A_legacy.shape != (n, n_comps):
        raise ValueError(
            f"malformed {owner} state: the pre-component-row A has shape "
            f"{A_legacy.shape}, expected {(n, n_comps)} for comp_list of shape "
            f"{comp_list.shape}"
        )
    if comp_list.min() < 0 or comp_list.max() >= n_comps:
        raise ValueError(
            f"malformed {owner} state: comp_list holds ids outside [0, {n_comps})"
        )
    for h in range(n_models):
        if np.unique(comp_list[:, h]).size != n:
            raise ValueError(
                f"malformed {owner} state: model {h} uses a component id twice "
                "in comp_list"
            )
    ids, counts = np.unique(comp_list, return_counts=True)
    shared = ids[counts > 1]
    if shared.size:
        raise ValueError(
            f"This {owner} model was saved by a pamica version that stored the "
            "mixing matrix with components as columns, after share_comps had "
            f"merged {shared.size} component(s) across models. Those merges "
            "compared and tied stored mixing columns, which are not components "
            "(issue #334), so the fitted model cannot be converted. Refit it "
            "with this version of pamica."
        )
    A_rows = np.empty((n_comps, n), dtype=A_legacy.dtype)
    for h in range(n_models):
        idx = comp_list[:, h]
        A_rows[idx, :] = A_legacy[:, idx]
    return A_rows
