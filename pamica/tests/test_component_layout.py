"""Validation of :func:`pamica.component_layout.rows_from_legacy_columns`.

The conversion that loads a pre-#334 save (components as stored columns) into
the component-row layout. Its round trips on real fitted models are in
``test_component_rows_persistence.py``; these are the input checks, on small
arrays: every malformed payload and every merged ``comp_list`` must raise the
module's own ``ValueError`` instead of converting, or failing later with an
indexing error.
"""

from __future__ import annotations

import numpy as np
import pytest

from pamica.component_layout import rows_from_legacy_columns

N = 3
# Two models, unmerged: model 0 uses components 0-2, model 1 uses 3-5, in a
# permuted order so the conversion cannot pass by treating ids as positions.
COMP_LIST = np.array([[0, 5], [2, 3], [1, 4]])
A_LEGACY = np.arange(N * 2 * N, dtype=np.float64).reshape(N, 2 * N) / 7.0


def _convert(A: np.ndarray = A_LEGACY, comp_list: np.ndarray = COMP_LIST):
    return rows_from_legacy_columns(A, comp_list, owner="Probe")


def test_an_unmerged_payload_converts_block_for_block():
    A_rows = _convert()
    assert A_rows.shape == (2 * N, N)
    assert A_rows.dtype == A_LEGACY.dtype
    for h in range(2):
        idx = COMP_LIST[:, h]
        np.testing.assert_array_equal(A_rows[idx, :], A_LEGACY[:, idx])


def test_a_one_dimensional_comp_list_is_refused():
    with pytest.raises(ValueError, match=r"malformed Probe state: comp_list has shape"):
        _convert(comp_list=COMP_LIST[:, 0])


@pytest.mark.parametrize(
    "comp_list",
    [COMP_LIST.astype(np.float64), np.where(COMP_LIST == 4, np.nan, COMP_LIST)],
    ids=["float-ids", "nan-id"],
)
def test_a_non_integer_comp_list_is_refused(comp_list):
    with pytest.raises(ValueError, match=r"comp_list has dtype float64"):
        _convert(comp_list=comp_list)


@pytest.mark.parametrize(
    "A",
    [A_LEGACY.T, A_LEGACY[:, :-1], A_LEGACY[None]],
    ids=["transposed", "one-column-short", "three-dimensional"],
)
def test_a_mismatched_A_shape_is_refused(A):
    with pytest.raises(ValueError, match=r"the pre-component-row A has shape"):
        _convert(A=A)


@pytest.mark.parametrize("bad", [-1, 2 * N], ids=["negative", "past-the-end"])
def test_an_out_of_range_id_is_refused(bad):
    comp_list = COMP_LIST.copy()
    comp_list[1, 1] = bad
    with pytest.raises(ValueError, match=r"comp_list holds ids outside \[0, 6\)"):
        _convert(comp_list=comp_list)


def test_an_id_repeated_within_a_model_is_refused():
    comp_list = COMP_LIST.copy()
    comp_list[2, 1] = comp_list[0, 1]
    with pytest.raises(ValueError, match=r"model 1 uses a component id twice"):
        _convert(comp_list=comp_list)


def test_an_id_shared_across_models_asks_for_a_refit():
    comp_list = COMP_LIST.copy()
    comp_list[0, 1] = comp_list[0, 0]  # a share_comps merge: 5 folded into 0
    with pytest.raises(ValueError, match=r"merged 1 component\(s\).*Refit it"):
        _convert(comp_list=comp_list)
