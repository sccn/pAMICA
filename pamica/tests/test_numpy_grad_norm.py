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


def test_grad_norm_uses_the_pre_step_mixing_matrix():
    """Fortran builds dAk inside accum_updates_and_likelihood (amica15.f90:1731-
    1743) strictly before update_params applies the step (:1789), so the norm
    describes the gradient at the CURRENT A, not the updated one.

    Snapshot A immediately before an M-step, recompute the Fortran formula from
    that snapshot independently of the implementation, and require the recorded
    value to match. A regression that moved the computation back after the A
    update or after the doscaling rescale would use a different A and fail here.

    A doscaling on/off comparison was tried first and rejected: it passes against
    the pre-fix implementation too, because rescaling changes A, which changes the
    next iteration's sufficient statistics, so the trajectories differ either way.
    It asserted nothing.

    The recomputation needs no access to the implementation's internals. For a
    single model with rescaling disabled, the applied step IS ``lrate * dAk``
    (``A -= lrate * dir.T @ A``, and dAk is that same product with zeta == gm ==
    1), so ``dAk == (A_before - A_after) / lrate`` exactly. Deriving the expected
    norm from the step that was actually taken is independent of how the
    implementation computed it.
    """
    model = AMICA(num_models=1, num_mix=3, max_iter=3, seed=42, doscaling=False)
    model.fit(_real_data(4096))

    updates = model._get_updates_and_likelihood()
    assert model.A is not None  # set by fit(); narrows Optional for the checker
    a_before = np.asarray(model.A).copy()
    model._update_parameters(updates)

    dAk = (a_before - np.asarray(model.A)) / model.lrate
    expected = np.sqrt(np.sum(dAk**2) / (model.data_dim * model.num_comps))
    assert np.isclose(model.nd[-1], expected, rtol=1e-8)


# --- gm ordering under sharing (issue #219) ---------------------------------
def _shared_two_model_fit(max_iter=3):
    """A fitted 2-model NumPy model whose comp_list genuinely shares a column.

    Sharing is forced directly rather than waited for: `share_comps` reaches a
    shared column only after a merge fires, and a short fit on this sample is
    not guaranteed to produce one.
    """
    model = AMICA(num_models=2, num_mix=3, max_iter=max_iter, seed=42)
    model.fit(_real_data(4096))
    assert model.comp_list is not None
    model.comp_list[0, 1] = model.comp_list[0, 0]
    return model


def _nd_for_pre_update_gm(model, updates, gm_pre):
    """Run one M-step with `gm` preset to `gm_pre`, return the resulting nd.

    `updates` is held fixed across calls, so the *post*-update gm
    (`updates["dgm"] / num_samples`) is identical whichever `gm_pre` is passed.
    Any difference in nd therefore comes from the pre-update value.
    """
    saved = {
        name: np.array(getattr(model, name), copy=True)
        for name in ("A", "mu", "beta", "rho", "alpha", "gm", "c")
        if getattr(model, name, None) is not None
    }
    model.gm = np.array(gm_pre, dtype=float)
    model.nd = []
    model._update_parameters(updates)
    nd = float(np.asarray(model.nd)[-1])
    for name, value in saved.items():
        setattr(model, name, value)
    return nd


def test_nd_uses_pre_update_model_weights():
    """nd is weighted by the model weights from before the mixture update.

    Fortran builds dAk in accum_updates_and_likelihood (amica15.f90:1753) and
    only reassigns gm in update_params (:1788), so its ndtmpsum never sees the
    current iteration's weights.

    The two M-steps below share one `updates` dict, so they agree on the
    post-update gm exactly. They differ only in the gm present when dAk is
    built. Weighting by the post-update value would make them identical; the
    assertion is that they are not.
    """
    model = _shared_two_model_fit()
    updates = model._get_updates_and_likelihood()

    nd_a = _nd_for_pre_update_gm(model, updates, [0.9, 0.1])
    nd_b = _nd_for_pre_update_gm(model, updates, [0.1, 0.9])

    assert np.isfinite(nd_a) and np.isfinite(nd_b)
    assert nd_a != nd_b, (
        "nd is identical under opposite pre-update model weights, so it is "
        "being built from the post-update gm"
    )


def test_nd_ignores_the_post_update_model_weights():
    """The complement: holding the pre-update gm fixed pins nd.

    Guards the other direction, so the test above cannot be satisfied by nd
    simply being sensitive to unrelated state.
    """
    model = _shared_two_model_fit()
    updates = model._get_updates_and_likelihood()

    first = _nd_for_pre_update_gm(model, updates, [0.7, 0.3])
    second = _nd_for_pre_update_gm(model, updates, [0.7, 0.3])
    assert first == second


def test_shared_column_keeps_grad_norm_finite():
    """A column shared across models divides by `zeta = sum_h gm[h]` over both.

    The guard that matters is that a shared column cannot produce a 0/0: `zeta`
    accumulates a positive weight per contributing model, so the division stays
    finite even when one model's weight collapses.
    """
    model = _shared_two_model_fit()
    updates = model._get_updates_and_likelihood()
    nd = _nd_for_pre_update_gm(model, updates, [1.0, 0.0])
    assert np.isfinite(nd) and nd > 0.0
