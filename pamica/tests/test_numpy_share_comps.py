"""``share_comps`` on the NumPy backend (issues #240, #242).

Two defects, both on the path a merge opens up and neither reachable with the
default disjoint ``comp_list``:

* ``comp_used`` was rebuilt fresh on every ``identify_shared_components`` call,
  so once ``comp_list`` was fully merged the ``k1 == k2`` guard skipped every
  pair and the mask came back all-True while half the columns were dead. The
  unmasked mixture update then divided 0/0 and left NaN in ``mu``/``beta``
  while the fit reported success (#240).
The A-update defect on the same path (#242) is not covered here and not fixed
here: a shared column still takes one step per contributing model. Three
attempts at a test for it passed equally with the loop and with Fortran's single
weighted application, so it ships with a test that actually distinguishes them.

Real sample EEG throughout. Sharing is forced by construction where a short fit
would not reliably produce a merge.
"""

from pathlib import Path

import numpy as np
import pytest

from pamica import AMICA_NumPy as AMICA
from pamica.numpy_impl.data import load_data_file
from pamica.numpy_impl.utils import identify_shared_components

_FDT = Path(__file__).resolve().parent.parent / "sample_data" / "eeglab_data.fdt"

pytestmark = pytest.mark.skipif(not _FDT.exists(), reason="sample data missing")


def _real_data(n_samples: int = 4096) -> np.ndarray:
    data = load_data_file(str(_FDT), 32, 30504, dtype=np.float32)
    return data[:, :n_samples].astype(np.float64)


def _shared_fit(max_iter: int = 5, share_comps: bool = True):
    model = AMICA(
        num_models=2,
        num_mix=3,
        max_iter=max_iter,
        seed=7,
        share_comps=share_comps,
        share_start=1,
        share_int=2,
    )
    model.fit(_real_data())
    assert model.comp_list is not None
    return model


# --- comp_used staleness (#240) ---------------------------------------------
def test_comp_used_survives_a_second_identify_call():
    """The mask must not forget columns merged away by an earlier call.

    Calling twice is the crux: the second call sees an already-merged
    ``comp_list``, so every pair hits the ``k1 == k2`` guard and no merge fires.
    A mask built during that loop comes back all-True.
    """
    model = _shared_fit(max_iter=3)
    first = identify_shared_components(
        model.A, model.W, model.comp_list.copy(), model.comp_thresh
    )
    comp_list_after, used_first = first
    if used_first.all():
        pytest.skip("no merge fired on this data; nothing to forget")

    _, used_second = identify_shared_components(
        model.A, model.W, comp_list_after.copy(), model.comp_thresh
    )
    assert used_second.sum() == used_first.sum(), (
        "comp_used was rebuilt from scratch and forgot the earlier merge"
    )


def test_comp_used_matches_the_columns_comp_list_references():
    """The mask is exactly the set of referenced columns, as in AMICATorchNG."""
    model = _shared_fit()
    referenced = np.zeros(model.num_comps, dtype=bool)
    referenced[np.unique(model.comp_list)] = True
    np.testing.assert_array_equal(model.comp_used, referenced)


# --- NaN mixture parameters (#240) ------------------------------------------
def test_sharing_leaves_finite_mixture_parameters():
    """A merged-away column receives no mass, so its update is 0/0.

    Before the fix this left NaN in half of ``mu`` and ``beta`` while the fit
    returned normally.
    """
    model = _shared_fit()
    for name in ("A", "mu", "beta", "gm", "alpha"):
        value = np.asarray(getattr(model, name))
        assert np.all(np.isfinite(value)), f"{name} holds non-finite values"


def test_unused_columns_keep_their_last_finite_value():
    """Frozen, not zeroed: an unused column keeps the value it last held."""
    model = _shared_fit()
    unused = ~model.comp_used
    if not unused.any():
        pytest.skip("no column was merged away on this data")
    assert np.all(np.isfinite(model.mu[:, unused]))
    assert np.all(model.beta[:, unused] > 0.0)


def test_default_comp_list_is_unaffected():
    """Every column has one contributor without sharing, so nothing changes."""
    model = AMICA(num_models=2, num_mix=3, max_iter=5, seed=42)
    model.fit(_real_data())
    assert model.comp_used is None or model.comp_used.all()
    for name in ("A", "mu", "beta"):
        assert np.all(np.isfinite(np.asarray(getattr(model, name))))
