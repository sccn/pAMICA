"""Weight-gradient norm (Fortran ndtmpsum) on the NumPy backend (issue #212).

The bug this guards against: `self.nd` was built from the raw `dWtmp` block sum,
never divided by `dgm[h]`, never `+I`, never premultiplied by `A`, and never
masked by `comp_used` -- so it did not use `directions[h]`, the quantity two
lines above that drives the actual A update. It reported ~5.4e3 on this sample
where Fortran reports ~5.7e-2, and sat flat rather than tracking convergence,
which made `min_nd`/`use_grad_norm` unreachable in this backend.

Deliberately no hardcoded Fortran values here. The iteration at which any
convergence quantity reaches a given magnitude varies by more than 3x across
BLAS implementations (measured 326 / 412 / 1076 for the same `min_dll` stop on
macOS-arm64, Linux-x86_64 and the CI runner, PR #213), so an exact-value
assertion would be a snapshot of one machine. These assert the two properties
the bug actually violated: the scale, and that the quantity tracks the fit.
"""

from pathlib import Path

import numpy as np

from pamica import AMICA_NumPy as AMICA
from pamica.numpy_impl.data import load_data_file

_FDT = Path(__file__).resolve().parent.parent / "sample_data" / "eeglab_data.fdt"


def _real_data(n_samples: int) -> np.ndarray:
    """A slice of the committed sample EEG (32 channels), float64."""
    data = load_data_file(str(_FDT), 32, 30504, dtype=np.float32)
    return data[:, :n_samples].astype(np.float64)


def test_grad_norm_is_on_fortran_scale():
    """ndtmpsum is a normalized per-element quantity, order 1e-2 early in a fit,
    not the order-1e3 raw sum the pre-fix code produced. The bound is loose by
    five orders of magnitude relative to the bug, so it cannot be tripped by
    ordinary numerical variation."""
    model = AMICA(num_models=1, num_mix=3, max_iter=10, seed=42)
    model.fit(_real_data(8192))

    nd = np.asarray(model.nd)
    assert nd.size == 10
    assert np.all(np.isfinite(nd))
    assert np.all(nd > 0.0)
    # The pre-fix implementation reported ~5.4e3 here.
    assert nd[0] < 1.0


def test_grad_norm_tracks_convergence():
    """The pre-fix quantity was flat across iterations (it was not a function of
    the step actually being taken), so it carried no convergence information.
    The corrected one shrinks as the fit approaches a stationary point."""
    model = AMICA(num_models=1, num_mix=3, max_iter=30, seed=42)
    model.fit(_real_data(8192))

    nd = np.asarray(model.nd)
    # Compare windows rather than adjacent iterations: the natural-gradient
    # trajectory is not monotone, but the trend over a fit this short is clear.
    assert nd[-5:].mean() < nd[:5].mean() / 2.0


# Not tested here: that the norm is computed BEFORE the A step and the doscaling
# rescale, matching Fortran's ordering (amica15.f90:1731-1743 precedes :1789). A
# doscaling on/off comparison was tried and rejected: it passes against the
# pre-fix implementation too, because rescaling changes A, which changes the next
# iteration's sufficient statistics, so the trajectories differ either way. It
# would have asserted nothing. The ordering is enforced by the placement itself
# and documented at the call site.
