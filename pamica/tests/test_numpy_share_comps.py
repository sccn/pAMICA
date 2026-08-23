"""``share_comps`` on the NumPy backend (issues #240, #242).

Three defects, all on the path a merge opens up and none reachable with the
default disjoint ``comp_list``:

* ``comp_used`` was rebuilt fresh on every ``identify_shared_components`` call,
  so once ``comp_list`` was fully merged the ``k1 == k2`` guard skipped every
  pair and the mask came back all-True while half the columns were dead. The
  unmasked mixture update then divided 0/0 and left NaN in ``mu``/``beta``
  while the fit reported success (#240).
* The mixture update masked those 0/0 results with ``np.where`` behind an
  ``np.errstate`` suppression, so a genuine 0/0 in a LIVE column went silent
  too. The dead columns are now indexed out instead, and a fit that ends
  holding non-finite parameters reports ``converged=False`` (#240).
* The A-update was a per-model loop, so a column shared by two models took one
  step per contributing model -- the second against an already-stepped ``A`` --
  instead of Fortran's single ``gm``-weighted average applied once (#242).

Real sample EEG throughout. Sharing is forced by construction where a short fit
would not reliably produce a merge.
"""

import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

from pamica import AMICA_NumPy as AMICA
from pamica.numpy_impl.data import load_data_file
from pamica.numpy_impl.load import loadmodout
from pamica.numpy_impl.utils import identify_shared_components
from pamica.torch_impl.core import AMICATorchNG

_FDT = Path(__file__).resolve().parent.parent / "sample_data" / "eeglab_data.fdt"

pytestmark = pytest.mark.skipif(not _FDT.exists(), reason="sample data missing")

_BLOCK = 1024


def _real_data(n_samples: int = 4096) -> np.ndarray:
    data = load_data_file(str(_FDT), 32, 30504, dtype=np.float32)
    return data[:, :n_samples].astype(np.float64)


def _shared_fit(
    max_iter: int = 10,
    share_comps: bool = True,
    share_start: int = 2,
    share_int: int = 8,
    **kwargs,
):
    """A 2-model fit with sharing on: merge at iteration ``share_start``, the
    6-iteration A-freeze, then iterations where A moves again."""
    model = AMICA(
        num_models=2,
        num_mix=3,
        max_iter=max_iter,
        seed=7,
        share_comps=share_comps,
        share_start=share_start,
        share_int=share_int,
        use_tqdm=False,
        do_opt_block=False,
        block_size=_BLOCK,
        **kwargs,
    )
    model.fit(_real_data())
    assert model.comp_list is not None
    return model


def _force_merged_column(model):
    """Merge model 1's first component into model 0's, as a fit-time merge does.

    Returns ``(kept, dead)``: the shared column index and the merged-away one.
    Deterministic on purpose -- keying a test on whether a merge happened to
    occur would let it skip under the very bugs it guards.
    """
    kept = int(model.comp_list[0, 0])
    dead = int(model.comp_list[0, 1])
    model.A[:, dead] = model.A[:, kept]
    model.comp_list[model.comp_list == dead] = kept
    model.comp_used = np.zeros(model.num_comps, dtype=bool)
    model.comp_used[np.unique(model.comp_list)] = True
    model._update_unmixing_matrices()
    return kept, dead


# --- configuration validation ------------------------------------------------
@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(share_start=0), "share_start"),
        (dict(share_int=6), "share_int"),
        (dict(share_int=2), "share_int"),  # the issue #240 reproduction's config
        (dict(comp_thresh=0.0), "comp_thresh"),
        (dict(comp_thresh=1.5), "comp_thresh"),
    ],
)
def test_share_constructor_validation(kwargs, match):
    """Rejected up front, and for the same reasons as AMICATorchNG.

    ``share_int <= 6`` is the one that bites: the post-merge A-freeze window is
    6 iterations long, so a shorter cycle would hold A frozen on every iteration
    of every cycle and the fit would silently stop moving its mixing matrix.
    """
    with pytest.raises(ValueError, match=match):
        AMICA(num_models=2, share_comps=True, use_tqdm=False, **kwargs)


# --- comp_used staleness (#240) ---------------------------------------------
def test_comp_used_survives_a_second_identify_call():
    """The mask must not forget columns merged away by an earlier call.

    Calling twice is the crux: the second call sees an already-merged
    ``comp_list``, so every pair hits the ``k1 == k2`` guard and no merge fires.
    A mask built during that loop comes back all-True.
    """
    model = _shared_fit(max_iter=3, share_comps=False)
    model.A[:, int(model.comp_list[0, 1])] = model.A[:, int(model.comp_list[0, 0])]
    atil = model._pinv_sphere() @ model.A

    comp_list_after, used_first = identify_shared_components(
        atil, model.comp_list.copy(), model.comp_thresh
    )
    assert not used_first.all(), "setup failed: the forced collinear pair did not merge"

    _, used_second = identify_shared_components(
        atil, comp_list_after.copy(), model.comp_thresh
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
    """A merged-away column receives no mass, so its update would be 0/0.

    Before the fix this left NaN in half of ``mu`` and ``beta`` while the fit
    returned normally. This is the issue #240 reproduction, at the shortest
    sharing interval the constructor accepts.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = _shared_fit(max_iter=5, share_start=1, share_int=7)

    assert int(model.comp_used.sum()) < model.num_comps, "setup failed: no merge fired"
    for name in ("A", "mu", "beta", "gm", "alpha", "rho"):
        value = np.asarray(getattr(model, name))
        assert np.all(np.isfinite(value)), f"{name} holds non-finite values"
    assert model.converged is True

    # The dead columns are indexed out of the divisions rather than divided and
    # masked, so no invalid-value warning is raised and none has to be
    # suppressed (an np.errstate blanket would also hide a live column's 0/0).
    divides = [
        w
        for w in caught
        if issubclass(w.category, RuntimeWarning)
        and ("divide" in str(w.message) or "invalid value" in str(w.message))
    ]
    assert not divides, f"divide warnings fired: {[str(w.message) for w in divides]}"


def test_unused_columns_keep_their_last_finite_value():
    """Frozen, not zeroed: an unused column keeps the value it last held.

    The merge is forced rather than hoped for, so a stale all-True mask fails
    here instead of skipping. ``doscaling`` is off so the comparison can be
    exact: the rescale pass normalizes every column, dead ones included, which
    multiplies their mu by a norm that is 1.0 only to within a ULP.
    """
    model = _shared_fit(max_iter=3, share_comps=False, doscaling=False)
    _, dead = _force_merged_column(model)
    unused = ~model.comp_used
    assert unused.any(), "setup failed: no column was merged away"
    mu_before = model.mu[:, unused].copy()
    rho_before = model.rho[:, unused].copy()

    model._update_parameters(model._get_updates_and_likelihood())

    np.testing.assert_array_equal(model.mu[:, unused], mu_before)
    np.testing.assert_array_equal(model.rho[:, unused], rho_before)
    assert np.all(model.beta[:, unused] > 0.0)
    assert np.all(np.isfinite(model.alpha[:, unused]))


def test_default_comp_list_is_unaffected():
    """Every column has one contributor without sharing, so nothing changes."""
    model = AMICA(
        num_models=2, num_mix=3, max_iter=5, seed=42, use_tqdm=False, block_size=_BLOCK
    )
    model.fit(_real_data())
    assert model.comp_used is None or model.comp_used.all()
    for name in ("A", "mu", "beta"):
        assert np.all(np.isfinite(np.asarray(getattr(model, name))))


# --- degenerate-fit reporting (#240) ----------------------------------------
class _CollapseOneComponent(AMICA):
    """Zero one LIVE column's mixture location statistics at ``collapse_iter``.

    A component whose responsibility mass collapses to exactly zero is the real
    0/0 that the guarded path deliberately does NOT hide: only merged-away
    columns are skipped. Injecting the collapse into the accumulated statistics
    (the fault-injection idiom ``test_sample_data.py`` uses for the restart
    path) drives the production update, which is what must leave the fit
    reporting failure rather than a NaN success.

    ``collapse_iter`` is a 0-indexed iteration; the default lands on the last
    one, so the reported LL stays finite and only the parameters are degenerate.
    An earlier value leaves the fit running afterwards, which is what exercises
    the mid-fit checkpoint path.
    """

    def __init__(self, *args, collapse_iter=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.collapse_iter = (
            self.max_iter - 1 if collapse_iter is None else collapse_iter
        )

    def _get_updates_and_likelihood(self):
        updates = super()._get_updates_and_likelihood()
        if self.iter == self.collapse_iter:
            updates["dmu_n"][:, 0] = 0.0
            updates["dmu_d"][:, 0] = 0.0
        return updates


def _collapsing_model(tmp_path, max_iter=3, **kwargs):
    return _CollapseOneComponent(
        num_models=1,
        num_mix=3,
        max_iter=max_iter,
        seed=7,
        use_tqdm=False,
        do_opt_block=False,
        block_size=_BLOCK,
        outdir=str(tmp_path / "out"),
        **kwargs,
    )


def test_non_finite_parameters_are_reported_as_failure(tmp_path):
    """A finite likelihood is not proof of a usable fit.

    The collapse lands on the final iteration, so the reported LL stays finite
    and only the parameters are degenerate -- exactly the shape of the silent
    failure in #240.
    """
    model = _collapsing_model(tmp_path, writestep=10_000_000)
    # The counterpart to the silence on dead columns: a LIVE column's 0/0 is
    # still numpy's own warning, because nothing suppresses it.
    with pytest.warns(RuntimeWarning, match="invalid value"):
        model.fit(_real_data(2048))

    assert np.isfinite(model.ll[-1]), "setup failed: the LL itself went non-finite"
    assert not np.all(np.isfinite(np.asarray(model.mu))), "setup failed: mu is finite"
    assert model.converged is False
    assert model.stop_reason is not None and "mu" in model.stop_reason
    # Announced in the run log, not only on the object.
    log_text = (tmp_path / "out" / "out.txt").read_text().lower()
    assert "did not converge" in log_text
    # Nothing was written: this run never reached a writestep boundary, and the
    # final write is refused, so the degenerate state reaches no file at all.
    assert not (tmp_path / "out" / "W").exists()


def test_checkpoints_never_persist_non_finite_parameters(tmp_path):
    """A mid-fit checkpoint must not write a degenerate state to disk.

    ``fit``'s final write is not the only write: ``writestep`` checkpoints run
    inside the loop, and a state that goes non-finite early is still on the
    object for every later checkpoint. Persisting it would leave a run whose
    only on-disk artifact is corrupt -- ``loadmodout`` reads NaN back without
    complaint. The collapse lands at iteration 2 of 4 with ``writestep=1``, so
    the first checkpoint is valid and every later one must be refused, loudly,
    without disturbing what the valid one wrote.
    """
    model = _collapsing_model(tmp_path, max_iter=4, collapse_iter=1, writestep=1)
    with pytest.warns(RuntimeWarning, match="invalid value"):
        model.fit(_real_data(2048))

    assert not np.all(np.isfinite(np.asarray(model.mu))), "setup failed: mu is finite"
    assert model.converged is False

    # The pre-collapse checkpoint is still there -- a refused write leaves it
    # alone rather than truncating it -- and everything in it is finite.
    assert (tmp_path / "out" / "W").exists(), "the valid checkpoint was lost"
    out = loadmodout(tmp_path / "out")
    for name in ("W", "A", "S", "mu", "alpha", "sbeta", "rho", "c"):
        value = getattr(out, name, None)
        assert value is not None, f"{name} missing from the written output"
        assert np.all(np.isfinite(np.asarray(value))), f"{name} written non-finite"

    # Refused loudly, naming the parameter. Once here rather than once per
    # remaining iteration: the NaN mu makes the next likelihood non-finite, so
    # restart-on-NaN takes over and its `continue` skips the checkpoint entirely.
    log_text = (tmp_path / "out" / "out.txt").read_text()
    assert "Skipping the results checkpoint" in log_text
    assert "non-finite mu" in log_text


# --- one gm-weighted A step per shared column (#242) ------------------------
def _rebuild_directions(model, updates):
    """The natural-gradient direction per model, as _update_parameters builds it."""
    directions = []
    for h in range(model.num_models):
        dA = -updates["dWtmp"][:, :, h] / updates["dgm"][h]
        dA[np.diag_indices_from(dA)] += 1
        directions.append(dA)
    return directions


def test_shared_column_takes_one_gm_weighted_step():
    """Fortran builds one gm-weighted averaged dAk and applies it once.

    The old per-model loop stepped the shared column twice, the second time
    against an already-stepped A. Both candidate steps are recomputed from the
    same pre-update state here, so the test fails against either mistake: it
    pins the Fortran expression AND asserts the sequential loop is a materially
    different operation (not a rounding difference).
    """
    model = _shared_fit(max_iter=3, share_comps=False, doscaling=False)
    kept, _ = _force_merged_column(model)
    A_before = model.A.copy()
    gm_before = model.gm.copy()
    lrate = model.lrate

    updates = model._get_updates_and_likelihood()
    directions = _rebuild_directions(model, updates)
    model._update_parameters(updates)

    # Fortran: one dAk, gm-weighted across contributing models, divided by
    # zeta = sum_h gm[h], applied once (amica15.f90:1749-1761 build, DAXPY at
    # :1807/:1814).
    dAk = np.zeros_like(A_before)
    zeta = np.zeros(model.num_comps)
    for h in range(model.num_models):
        zeta[model.comp_list[:, h]] += gm_before[h]
    for h in range(model.num_models):
        idx = model.comp_list[:, h]
        dAk[:, idx] += gm_before[h] * np.dot(directions[h].T, A_before[:, idx])
    nonzero = zeta > 0
    dAk[:, nonzero] /= zeta[nonzero]
    expected = A_before - lrate * dAk

    # Same arithmetic in a different association order, so the agreement is at
    # the ULP level (~1e-17 here); 1e-12 leaves five orders of margin.
    np.testing.assert_allclose(model.A, expected, rtol=0, atol=1e-12)

    # The rejected alternative: one sequential step per contributing model.
    sequential = A_before.copy()
    for h in range(model.num_models):
        idx = model.comp_list[:, h]
        sequential[:, idx] = sequential[:, idx] - lrate * np.dot(
            directions[h].T, sequential[:, idx]
        )
    assert np.abs(model.A[:, kept] - sequential[:, kept]).max() > 1e-3, (
        "the two A-update semantics are indistinguishable on this state, so the "
        "test above would pass with the per-model loop restored"
    )


def test_merged_away_column_does_not_move():
    """A column no contributor references gets exactly zero dAk, so it holds."""
    model = _shared_fit(max_iter=3, share_comps=False, doscaling=False)
    _, dead = _force_merged_column(model)
    dead_before = model.A[:, dead].copy()

    model._update_parameters(model._get_updates_and_likelihood())

    np.testing.assert_array_equal(model.A[:, dead], dead_before)


# --- post-merge A-freeze (#242) ---------------------------------------------
def test_a_frozen_window_matches_the_torch_schedule():
    """Identical window to AMICATorchNG: the merge iteration and the 5 after."""
    model = AMICA(
        num_models=2,
        share_comps=True,
        share_start=10,
        share_int=20,
        use_tqdm=False,
    )

    def frozen(itf):
        model.iter = itf - 1  # _a_frozen works in Fortran-style 1-indexed iters
        return model._a_frozen()

    assert not any(frozen(i) for i in range(1, 10))  # before share_start
    assert all(frozen(i) for i in range(10, 16))  # merge iteration + 5
    assert not any(frozen(i) for i in range(16, 30))  # A moves again
    assert all(frozen(i) for i in range(30, 36))  # next cycle


def test_a_frozen_is_off_for_a_single_model():
    """A model cannot share with itself, so sharing never freezes A there."""
    model = AMICA(
        num_models=1, share_comps=True, share_start=1, share_int=8, use_tqdm=False
    )
    model.iter = 0
    assert model._a_frozen() is False


def test_freeze_holds_A_but_still_measures_the_gradient():
    """During the settle window A does not move, but nd is still recorded.

    Fortran computes ndtmpsum in the accumulation pass, which runs whether or
    not the A step is taken, so the gradient-norm stop keeps seeing the true
    gradient magnitude (the same reason AMICATorchNG computes dAk outside its
    freeze guard).
    """
    model = _shared_fit(max_iter=3, share_comps=False, doscaling=False)
    _force_merged_column(model)
    model.share_comps = True
    model.share_start = 1
    model.share_int = 8
    model.iter = 0  # itf == 1 == share_start: inside the freeze window
    assert model._a_frozen() is True
    A_before = model.A.copy()
    lrate_before = model.lrate
    nd_count = len(model.nd)

    model._update_parameters(model._get_updates_and_likelihood())

    np.testing.assert_array_equal(model.A, A_before)
    assert model.lrate == lrate_before  # the ramp is held with the step
    assert len(model.nd) == nd_count + 1 and model.nd[-1] > 0.0


# --- cross-backend agreement (.rules/backend_parity.md) ---------------------
def test_shared_column_update_matches_the_torch_backend():
    """The same merged state must move the shared column the same way in both.

    Both backends are driven from one set of parameters and one sphered data
    matrix, so the only thing under test is the A-update itself. They run the
    same float64 arithmetic and differ only in BLAS association order, so the
    columns agree to ~1e-16; 1e-12 is the tolerance with margin.
    """
    model = _shared_fit(max_iter=3, share_comps=False)
    kept, dead = _force_merged_column(model)
    A_before = model.A.copy()

    ng = AMICATorchNG(
        n_channels=model.data_dim,
        n_models=model.num_models,
        n_mix=model.num_mix,
        device="cpu",
        block_size=_BLOCK,
        seed=7,
        pdftype=0,
    )
    ng._initialize_parameters()
    for name in ("A", "mu", "alpha", "beta", "rho", "gm", "c", "sphere"):
        setattr(ng, name, torch.from_numpy(np.asarray(getattr(model, name)).copy()))
    ng.comp_list = torch.from_numpy(model.comp_list.copy())
    ng.lrate = model.lrate
    ng.rholrate = model.rholrate
    ng.iteration = model.iter
    ng._update_unmixing_matrices()
    assert int(ng.comp_used.sum()) == int(model.comp_used.sum())

    X_t = torch.from_numpy(model.data)  # the NumPy-sphered data, shared by both
    ng._update_parameters(ng._accumulate_blocks(X_t), X_t.shape[1])
    model._update_parameters(model._get_updates_and_likelihood())

    assert ng.A is not None
    A_torch = ng.A.numpy()
    assert np.abs(model.A[:, kept] - A_before[:, kept]).max() > 1e-6, (
        "the shared column did not move, so this compares nothing"
    )
    np.testing.assert_allclose(model.A[:, kept], A_torch[:, kept], rtol=0, atol=1e-12)
    np.testing.assert_allclose(model.A, A_torch, rtol=0, atol=1e-12)
    # The dead column is frozen in both, but both still run it through their own
    # rescale, whose column norm agrees only to a ULP across array libraries.
    np.testing.assert_allclose(model.A[:, dead], A_torch[:, dead], rtol=0, atol=1e-12)


def test_forced_merge_fit_is_finite_in_both_backends():
    """End to end, a fit that merges keeps finite parameters in both backends."""
    x = _real_data()
    npm = _shared_fit(max_iter=10, share_start=2, share_int=8, comp_thresh=0.9)
    assert int(npm.comp_used.sum()) < npm.num_comps
    for name in ("A", "mu", "beta", "gm", "alpha", "rho"):
        assert np.all(np.isfinite(np.asarray(getattr(npm, name)))), name
    assert npm.converged is True

    ng = AMICATorchNG(
        n_channels=32,
        n_models=2,
        n_mix=3,
        device="cpu",
        block_size=_BLOCK,
        seed=7,
        share_comps=True,
        share_start=2,
        share_iter=8,
        comp_thresh=0.9,
    )
    ng.fit(x, max_iter=10)
    assert int(ng.comp_used.sum()) < ng.n_comps
    for name in ("A", "mu", "beta", "gm", "alpha", "rho"):
        tensor = getattr(ng, name)
        assert tensor is not None and bool(torch.isfinite(tensor).all()), name


def test_merge_on_the_final_iteration_completes():
    """A merge scheduled on the LAST iteration must still leave a usable model.

    The schedule hook runs after this iteration's likelihood is stored
    (Fortran runs identify_shared_comps after accum_updates_and_likelihood,
    amica15.f90:1856), so ``self.ll[-1]`` deliberately describes the state
    BEFORE this merge -- the merged model is never scored. That is faithful to
    the reference and matches AMICATorchNG/AMICAMLXNG (mirrors
    tests/torch_tests/test_ng_sharing.py::test_merge_on_the_final_iteration_completes
    and
    tests/mlx_tests/test_mlx_sharing.py::test_merge_on_the_final_iteration_completes);
    issue #269 tracks documenting it across backends. Here it is pinned as
    behavior: the fit completes, the merge survives on the returned model, and
    the reported LL is finite.
    """
    model = _shared_fit(
        max_iter=10,
        share_start=10,
        share_int=100,  # only one merge attempt, on the final iteration
        comp_thresh=0.9,
    )
    assert len(model.ll) == 10  # the merge did not truncate the run
    assert model.comp_used is not None and int(model.comp_used.sum()) < model.num_comps
    assert model.converged is True
    assert np.isfinite(model.ll[-1])
    for name in ("A", "mu", "beta", "gm", "alpha", "rho"):
        assert np.all(np.isfinite(np.asarray(getattr(model, name)))), name


# --- sensor-space sharing similarity (issue #258) ----------------------------
def test_numpy_merge_decision_matches_torch_backend():
    """Acceptance test: from one matched fitted state, the two backends reach
    the identical merge decision now that both compare sensor-space
    (de-sphered) mixing columns -- ``pinv(sphere) @ A`` -- mirroring
    ``AMICATorchNG._identify_shared_comps`` exactly (torch_impl/core.py:1696-
    1741) instead of numpy's former sphered-space comparison.
    """
    model = _shared_fit(max_iter=3, share_comps=False)
    # Perturb one cross-model column into near- (not exact-) collinearity, so
    # the merge decision itself is under test rather than a state whose
    # comp_list is already pre-merged.
    i0 = int(model.comp_list[0, 0])
    i1 = int(model.comp_list[0, 1])
    rng = np.random.RandomState(0)
    model.A[:, i1] = model.A[:, i0] + 1e-3 * rng.standard_normal(model.A.shape[0])
    model._update_unmixing_matrices()
    thresh = 0.9

    ng = AMICATorchNG(
        n_channels=model.data_dim,
        n_models=model.num_models,
        n_mix=model.num_mix,
        device="cpu",
        block_size=_BLOCK,
        seed=7,
        pdftype=0,
    )
    ng._initialize_parameters()
    for name in ("A", "mu", "alpha", "beta", "rho", "gm", "c", "sphere"):
        setattr(ng, name, torch.from_numpy(np.asarray(getattr(model, name)).copy()))
    ng.comp_list = torch.from_numpy(model.comp_list.copy())
    ng._sphere_pinv = None
    ng.comp_thresh = thresh

    numpy_comp_list, _ = identify_shared_components(
        model._pinv_sphere() @ model.A, model.comp_list.copy(), thresh
    )
    ng._identify_shared_comps()

    assert not np.array_equal(numpy_comp_list, model.comp_list), (
        "no merge fired; the cross-backend comparison would be vacuous"
    )
    np.testing.assert_array_equal(numpy_comp_list, ng.comp_list.numpy())


def test_rank_reduced_numpy_sharing_completes():
    """Twin of ``test_ng_sharing.py::test_rank_reduced_share_fit_completes``: a
    rank-reduced sphere ``(16, 32)`` must not block ``share_comps`` -- the
    sensor-space back-map is a pseudo-inverse, valid at any rank (issue #253's
    fix, now shared by both backends via issue #258)."""
    # do_newton stays off: Newton is unrelated to this change, and turning it
    # on here crosses into a pre-existing, separately-scoped shape bug in the
    # NumPy Newton path when a share pass collapses many components at once
    # (encountered while writing this test; not touched by issue #258 -- see
    # PR body).
    model = AMICA(
        num_models=2,
        num_mix=3,
        max_iter=25,
        seed=3,
        share_comps=True,
        share_start=8,
        share_int=10,
        comp_thresh=0.99,
        pcakeep=16,
        use_tqdm=False,
        do_opt_block=False,
        block_size=_BLOCK,
        do_newton=False,
    )
    model.fit(_real_data())
    assert model.sphere is not None
    assert model.sphere.shape == (16, 32)
    assert model.data_dim == 16 and model.data_dim_in == 32
    assert model.converged is True
    for name in ("A", "mu", "beta", "gm", "alpha", "rho"):
        assert np.all(np.isfinite(np.asarray(getattr(model, name)))), name
    assert model.comp_used is not None
    assert int(model.comp_used.sum()) < model.num_comps  # sharing really ran


def test_zero_norm_column_is_not_merged_and_raises_no_warning():
    """Pins the ``tiny`` denominator guard: a zero-norm column's cosine comes
    out a clean 0.0, never NaN, so it neither merges nor trips a NumPy divide
    warning (which an unguarded 0/0 would)."""
    model = _shared_fit(max_iter=3, share_comps=False)
    zero_idx = int(model.comp_list[0, 1])
    other_idx = int(model.comp_list[0, 0])
    model.A[:, zero_idx] = 0.0
    atil = model._pinv_sphere() @ model.A

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        comp_list_after, _ = identify_shared_components(
            atil, model.comp_list.copy(), model.comp_thresh
        )

    divides = [
        w
        for w in caught
        if issubclass(w.category, RuntimeWarning)
        and ("divide" in str(w.message) or "invalid value" in str(w.message))
    ]
    assert not divides, f"divide warnings fired: {[str(w.message) for w in divides]}"
    assert int(comp_list_after[0, 1]) == zero_idx, "the zero-norm column merged"
    assert other_idx != zero_idx


def test_non_finite_sphere_fails_loudly():
    """A degenerate (non-finite) sphere has no pseudo-inverse; the guard must
    raise loudly rather than let the per-pair isfinite check downstream
    silently decline every merge (twin of
    ``test_ng_sharing.py::test_non_finite_sphere_still_fails_loudly``)."""
    model = _shared_fit(max_iter=2, share_comps=False)
    model.sphere = np.full_like(model.sphere, np.nan)
    model._sphere_pinv = None
    with pytest.raises(RuntimeError, match="non-finite"):
        model.get_sensor_mixing_matrix()
