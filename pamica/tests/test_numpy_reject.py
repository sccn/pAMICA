"""Outlier rejection (do_reject) on the NumPy backend (issue #123).

Analogous to the torch backend's do_reject tests (test_ng_backend.py): the
mechanism (good_idx shrinks on the maxrej schedule, gm normalizes by the good
count, the fit stays finite, degenerate LL raises) is behavior-validated against
the real sample EEG, never synthetic data. The NumPy port mirrors
AMICATorchNG.good_idx / _reject_outliers.
"""

from pathlib import Path

import numpy as np
import pytest

from pamica import AMICA_NumPy as AMICA
from pamica.numpy_impl.data import load_data_file

_FDT = Path(__file__).resolve().parent.parent / "sample_data" / "eeglab_data.fdt"


def _real_data(n_samples: int) -> np.ndarray:
    """A slice of the committed sample EEG (32 channels), float64."""
    data = load_data_file(str(_FDT), 32, 30504, dtype=np.float32)
    return data[:, :n_samples].astype(np.float64)


def test_reject_param_validation():
    """Invalid rejection parameters are refused at construction (matching
    AMICATorchNG), guarding the otherwise-opaque downstream failures: rejint<1
    -> ZeroDivisionError in the reject schedule; rejsig<=0 -> reject-below-mean
    semantics break (at 0 the threshold is the mean, dropping ~half every pass;
    negative inverts it)."""
    with pytest.raises(ValueError, match="rejint"):
        AMICA(do_reject=True, rejint=0)
    with pytest.raises(ValueError, match="rejsig"):
        AMICA(do_reject=True, rejsig=0.0)
    with pytest.raises(ValueError, match="rejsig"):
        AMICA(do_reject=True, rejsig=-5.0)
    with pytest.raises(ValueError, match="rejstart"):
        AMICA(do_reject=True, rejstart=-1)


def test_do_reject_false_leaves_good_idx_unset():
    """The default path never touches the rejection machinery: good_idx stays
    None and the fit is unaffected (issue #24 parity path)."""
    model = AMICA(num_models=1, num_mix=3, max_iter=3, do_newton=False)
    model.fit(_real_data(2000))
    assert model.good_idx is None


@pytest.mark.skipif(not _FDT.exists(), reason="sample data missing")
def test_rejection_shrinks_good_sample_set():
    """With do_reject enabled, low-log-likelihood samples are permanently
    dropped: the good-sample set shrinks, exactly maxrej passes run (numrej is
    capped), and the fit stays finite (Fortran-style reject_data, mirroring the
    torch backend)."""
    data = _real_data(3000)
    n_total = data.shape[1]
    model = AMICA(
        num_models=1,
        num_mix=3,
        max_iter=12,
        do_newton=False,
        do_reject=True,
        rejsig=2.0,
        rejstart=2,
        rejint=3,
        maxrej=2,
    )
    model.fit(data)

    assert model.good_idx is not None
    n_good = int(model.good_idx.size)
    assert n_good < n_total  # some samples were rejected
    assert n_good == model.num_good_samples  # count tracks the index array
    # Rejection ran, and no more than the maxrej budget of passes (the schedule
    # fires at rejstart=2 and rejstart+rejint=5, both within the 12-iter budget).
    assert 1 <= model.numrej <= 2
    # good_idx is a strict subset of the original indices, unique and in range.
    assert model.good_idx.min() >= 0 and model.good_idx.max() < n_total
    assert len(set(model.good_idx.tolist())) == n_good
    # The fit stayed finite through the rejection.
    assert model.W is not None
    assert np.all(np.isfinite(model.W))
    assert np.all(np.isfinite(model.ll))


@pytest.mark.skipif(not _FDT.exists(), reason="sample data missing")
def test_rejection_does_not_spuriously_stop_fit():
    """self.ll is normalized by the good-sample count (issue #212), and the
    reject iteration shrinks that count. Dropping the most-negative samples
    raises the mean of what remains regardless of normalization (removing
    below-average points can only raise a mean), so the fit keeps ascending
    across the rejection rather than stopping early on a spurious LL
    'decrease'."""
    data = _real_data(3000)
    model = AMICA(
        num_models=1,
        num_mix=3,
        max_iter=12,
        do_newton=False,
        do_reject=True,
        rejsig=2.0,
        rejstart=2,
        rejint=3,
        maxrej=1,
    )
    model.fit(data)
    ll = np.asarray(model.ll)
    # The fit did not stop at/just after the rejection: it ran well past rejstart.
    assert len(ll) > model.rejstart + 2
    # LL after the rejection is not below LL at the rejection iteration (dropping
    # below-average samples raises the mean of what remains; no spurious descent).
    assert ll[-1] >= ll[model.rejstart]


@pytest.mark.skipif(not _FDT.exists(), reason="sample data missing")
def test_multimodel_rejection_keeps_gm_and_c_finite():
    """do_reject + n_models=2: a shrinking good set can drive a model's
    responsibility mass toward zero, exercising the dead-model dgm==0 guards in
    the per-model bias c and gm updates. Both must stay finite (no 0/0 NaN), and
    gm must remain a valid distribution over models."""
    data = _real_data(3000)
    model = AMICA(
        num_models=2,
        num_mix=3,
        max_iter=10,
        do_newton=False,
        do_reject=True,
        rejsig=2.0,
        rejstart=2,
        maxrej=1,
    )
    model.fit(data)
    assert model.good_idx is not None and int(model.good_idx.size) < data.shape[1]
    assert model.gm is not None and model.c is not None
    assert np.all(np.isfinite(model.c))  # per-model bias survives rejection
    assert np.all(np.isfinite(model.gm))
    assert model.gm.min() >= 0.0
    np.testing.assert_allclose(model.gm.sum(), 1.0, atol=1e-8)


def test_rejection_nonfinite_ll_raises_instability_error():
    """A non-finite per-sample LL (numerical instability upstream) makes
    rejection raise a clear error naming the real cause, not blaming rejsig
    (issue #127). Even a single NaN poisons mean/std and would drop every
    sample, so the guard fires and the message must be accurate."""
    model = AMICA(do_reject=True, rejsig=3.0)
    model.good_idx = np.arange(16)
    ll = np.arange(16, dtype=np.float64)
    ll[7] = np.nan  # one non-finite entry poisons the whole reject decision
    model._last_ll_samples = ll
    with pytest.raises(ValueError, match="non-finite"):
        model._reject_outliers()
