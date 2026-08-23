"""Component sharing (``share_comps``) on the MLX backend -- issue #263.

Apple-Silicon only, real sample EEG (no synthetic/mock), same module guards as
``test_mlx_backend.py``. The port mirrors ``AMICATorchNG`` decision for decision
(``tests/torch_tests/test_ng_sharing.py``) and reuses the NumPy merge kernel
``pamica.numpy_impl.utils.identify_shared_components``.

This module holds the MLX-only mechanics: the default-path no-op, the schedule
and its A-freeze, frozen merged-away columns, multi-model and rank-reduced
sharing fits, constructor validation and the degenerate-sphere guard. The two
tests that pin MLX against ``AMICATorchNG`` (the merge decision and the ``gm``
weighting of the A-update) live in
``pamica/tests/test_mlx_sharing_cross_backend.py``, outside any one backend's
subdirectory, per ``.rules/backend_parity.md``.

The invariant that carries the whole port: with sharing off (or not yet
scheduled) every masking and freezing step added for it is a no-op, so the
validated default trajectory is bit-identical.
"""

from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "sample_data"
DATA_FILE = SAMPLE_DIR / "eeglab_data.fdt"
NW = 32
FIELD = 30504
NMIX = 3
SEED = 3
BLOCK = 1024

pytestmark = [
    pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing"),
    pytest.mark.skipif(
        mx.default_device().type != mx.DeviceType.gpu, reason="no Apple GPU"
    ),
]


def _real_data(n_samples: int = 4096) -> np.ndarray:
    from pamica.torch_impl.utils import load_eeglab_data

    data = load_eeglab_data(str(DATA_FILE), data_dim=NW, field_dim=FIELD)
    return data[:, :n_samples].astype(np.float64)


def _warm_model(n_models: int = 2, warmup: int = 5, **kwargs):
    """A model driven ``warmup`` M-steps past init on the real sample, returned
    with its sphered data so the caller can keep stepping it.

    Hand-driving the loop (rather than calling ``fit``) is what lets the freeze
    and gm-weighting tests inspect the state between two specific iterations;
    every step still goes through the production ``_update_parameters``.
    """
    from pamica.mlx_impl import AMICAMLXNG

    model = AMICAMLXNG(
        n_channels=NW,
        n_models=n_models,
        n_mix=NMIX,
        seed=SEED,
        block_size=BLOCK,
        **kwargs,
    )
    x_t = model._preprocess(_real_data())
    model._initialize_parameters()
    for it in range(warmup):
        model.iteration = it
        model._update_parameters(model._accumulate_blocks(x_t), x_t.shape[1])
    mx.eval(model.A, model.mu, model.alpha, model.beta, model.rho, model.gm, model.c)
    return model, x_t


def _force_merged_column(model):
    """Fold model 1's first component into model 0's, as a fit-time merge does.

    Returns ``(kept, dead)``. Deterministic on purpose: keying a test on whether
    a merge happened to occur would let it skip under the very bugs it guards.
    """
    cl = np.array(model.comp_list)
    kept, dead = int(cl[0, 0]), int(cl[0, 1])
    a_np = np.array(model.A)
    a_np[:, dead] = a_np[:, kept]
    model.A = mx.array(a_np)
    cl[cl == dead] = kept
    model.comp_list = mx.array(cl)
    used = np.zeros(model.n_comps, dtype=bool)
    used[np.unique(cl)] = True
    model._comp_used_arr = mx.array(used)
    model._update_unmixing_matrices()
    mx.eval(model.A, model.W)
    return kept, dead


# --- (a) default-path parity -------------------------------------------------


@pytest.mark.parametrize("n_models", [1, 2])
def test_unscheduled_sharing_is_bit_identical(n_models):
    """Sharing that never fires must leave the trajectory bit-for-bit unchanged.

    This is the invariant the whole port rests on: the ``comp_used`` masks are
    all-True, ``nd * 1.0`` is exact, the moved lrate ramp feeds only the step it
    now sits with, and ``gm_prev`` cancels against ``zeta`` for a disjoint
    ``comp_list``. The ``n_models=2`` case is the one that would catch a gm
    slip on the unshared path; ``n_models=1`` pins the issue #24 single-model
    path. ``share_start`` is past any reachable iteration, so the only thing
    ``share_comps=True`` changes here is which code path runs.
    """
    from pamica.mlx_impl import AMICAMLXNG

    data = _real_data()
    off = AMICAMLXNG(
        n_channels=NW, n_models=n_models, n_mix=NMIX, seed=SEED, block_size=BLOCK
    )
    off.fit(data, max_iter=10, verbose=False)
    on = AMICAMLXNG(
        n_channels=NW,
        n_models=n_models,
        n_mix=NMIX,
        seed=SEED,
        block_size=BLOCK,
        share_comps=True,
        share_start=10**6,
        share_iter=100,
    )
    on.fit(data, max_iter=10, verbose=False)

    assert off.ll_history == on.ll_history
    np.testing.assert_array_equal(np.array(off.A), np.array(on.A))
    np.testing.assert_array_equal(np.array(off.mu), np.array(on.mu))
    np.testing.assert_array_equal(np.array(off.gm), np.array(on.gm))
    assert off._ndtmpsum == on._ndtmpsum
    assert bool(np.array(on.comp_used).all())


def test_default_multimodel_fit_leaves_every_component_distinct():
    """With ``share_comps`` off (the default) a 2-model fit keeps the full block
    ``comp_list``, so ``comp_used`` stays all-True and no group is shared."""
    from pamica.mlx_impl import AMICAMLXNG

    model = AMICAMLXNG(
        n_channels=NW, n_models=2, n_mix=NMIX, seed=SEED, block_size=BLOCK
    )
    model.fit(_real_data(), max_iter=8, verbose=False)
    assert int(np.array(model.comp_used).sum()) == model.n_comps
    assert model.shared_components() == []


# --- (c) end-to-end -----------------------------------------------------------


def test_two_model_share_fit_completes_and_merges():
    """A full 2-model fit with sharing runs to a non-degenerate stop, keeps every
    parameter finite, and the merges survive to the returned model."""
    from pamica.mlx_impl import AMICAMLXNG

    model = AMICAMLXNG(
        n_channels=NW,
        n_models=2,
        n_mix=NMIX,
        seed=SEED,
        block_size=BLOCK,
        share_comps=True,
        share_start=4,
        share_iter=8,
        comp_thresh=0.9,
    )
    model.fit(_real_data(), max_iter=25, verbose=False)

    assert model.stop_reason not in AMICAMLXNG._DEGENERATE_STOP_REASONS
    assert model.final_ll_ is not None and np.isfinite(model.final_ll_)
    for name in ("A", "W", "mu", "alpha", "beta", "rho", "gm", "c"):
        value = np.array(getattr(model, name))
        assert np.all(np.isfinite(value)), name
    assert model._ndtmpsum is not None and np.isfinite(model._ndtmpsum)

    cl = np.array(model.comp_list)
    used = int(np.array(model.comp_used).sum())
    assert used < model.n_comps, "no merge survived the fit"
    assert used == np.unique(cl).size  # the mask is exactly the referenced set

    groups = model.shared_components()
    assert groups, "sharing ran but shared_components() reports nothing"
    for group in groups:
        cols = {int(cl[i, h]) for h, i in group}
        assert len(cols) == 1, "a shared group must reference exactly one column"
        assert len({h for h, _ in group}) >= 2, "sharing is across models"
    # Two models: the within-model guard caps a group at one source per model,
    # so each merge folds exactly one column away into exactly one new group.
    assert len(groups) == model.n_comps - used


def _assert_share_result_consistent(model):
    """Every merged fit holds finite parameters and a comp_list that agrees with
    the comp_used mask and the shared_components() grouping (the MLX twin of
    test_ng_sharing.py's helper of the same name)."""
    from pamica.mlx_impl import AMICAMLXNG

    for name in ("A", "W", "mu", "alpha", "beta", "rho", "gm", "c"):
        value = np.array(getattr(model, name))
        assert np.all(np.isfinite(value)), name
    assert model.final_ll_ is not None and np.isfinite(model.final_ll_)
    assert model.stop_reason not in AMICAMLXNG._DEGENERATE_STOP_REASONS

    cl = np.array(model.comp_list)
    assert cl.shape == (model.n_channels, model.n_models)
    assert cl.min() >= 0 and cl.max() < model.n_comps
    assert np.array(model.A).shape == (model.n_channels, model.n_comps)
    used = int(np.array(model.comp_used).sum())
    assert used == np.unique(cl).size

    for group in model.shared_components():
        cols = {int(cl[i, h]) for h, i in group}
        assert len(cols) == 1, "a shared group must reference exactly one column"
        assert len({h for h, _ in group}) >= 2, "sharing is across models"
        for h, i in group:
            assert 0 <= h < model.n_models and 0 <= i < model.n_channels


def test_merge_on_the_final_iteration_completes():
    """A merge scheduled on the LAST iteration must still leave a usable model.

    The schedule hook runs after the iteration's likelihood has been recorded
    (Fortran runs identify_shared_comps after accum_updates_and_likelihood,
    amica15.f90:1856), so ``final_ll_`` deliberately describes the state BEFORE
    this merge -- the merged model is never scored. That is faithful to the
    reference and matches AMICATorchNG, but it is a real trap for a caller
    comparing `final_ll_` against `comp_used`; issue #269 tracks documenting it
    across backends. Here it is pinned as behavior: the fit completes, the merge
    survives on the returned model, and the reported LL is finite.
    """
    from pamica.mlx_impl import AMICAMLXNG

    model = AMICAMLXNG(
        n_channels=NW,
        n_models=2,
        n_mix=NMIX,
        seed=SEED,
        block_size=BLOCK,
        share_comps=True,
        share_start=10,
        share_iter=100,  # only one merge attempt, on the final iteration
        comp_thresh=0.9,
    )
    model.fit(_real_data(), max_iter=10, verbose=False)

    assert len(model.ll_history) == 10  # the merge did not truncate the run
    assert int(np.array(model.comp_used).sum()) < model.n_comps
    _assert_share_result_consistent(model)


def test_convergence_stop_can_fire_while_A_is_frozen():
    """The grad-norm stop must stay live inside the A-freeze window.

    ``ndtmpsum`` is computed every iteration, not only when A moves (issue #207,
    Fortran computes it in the accumulation pass), so a run whose gradient is
    already below ``min_nd`` stops during the settle window instead of burning
    the remaining budget with A held. ``min_nd`` is set to a value this real fit
    actually crosses at the start (the shipped 1e-7 default is unreachable on a
    recording this size, issue #218).
    """
    from pamica.mlx_impl import AMICAMLXNG

    model = AMICAMLXNG(
        n_channels=NW,
        n_models=2,
        n_mix=NMIX,
        seed=SEED,
        block_size=BLOCK,
        share_comps=True,
        share_start=1,
        share_iter=8,
        comp_thresh=0.9,
        min_nd=0.04,
    )
    model.fit(_real_data(), max_iter=20, verbose=False)

    assert model.stop_reason == "grad_norm"
    # It stops on the first iteration that can stop at all (two LL values are
    # required, Fortran's `if (iter > 1)`), which is itf=2 -- inside the first
    # freeze window [share_start, share_start + 5] = [1, 6].
    itf = model.iteration + 1
    assert itf == 2 and model.share_start <= itf <= model.share_start + 5
    assert model._a_frozen() is True, "the stop did not land inside the window"
    assert model._ndtmpsum is not None and model._ndtmpsum <= model.min_nd
    assert model.final_ll_ is not None and np.isfinite(model.final_ll_)
    # The merge at itf=share_start did fire, so A really was held: this is a
    # stop inside a live settle window, not merely inside the schedule.
    assert int(np.array(model.comp_used).sum()) < model.n_comps


def test_three_model_share_fit_completes():
    """Sharing with three models: the scan runs 3 model pairs with sequential
    mutation, and a column can end up referenced by all three.

    Behavioral assertions only -- how many merges a real fit makes at a loose
    ``comp_thresh`` is data-dependent, so pinning a count would pin the sample,
    not the algorithm (the exact-count case is covered on controlled matrices by
    test_ng_sharing.py::test_three_model_guard_and_merge).
    """
    from pamica.mlx_impl import AMICAMLXNG

    model = AMICAMLXNG(
        n_channels=NW,
        n_models=3,
        n_mix=NMIX,
        seed=SEED,
        block_size=BLOCK,
        share_comps=True,
        share_start=4,
        share_iter=8,
        comp_thresh=0.9,
    )
    model.fit(_real_data(), max_iter=20, verbose=False)

    assert int(np.array(model.comp_used).sum()) < model.n_comps
    _assert_share_result_consistent(model)
    # A model still never shares a component with itself: within one model every
    # source references a distinct column.
    cl = np.array(model.comp_list)
    for h in range(model.n_models):
        assert np.unique(cl[:, h]).size == model.n_channels


def test_rank_reduced_share_fit_completes():
    """The issue #221 route: real EEG projected onto a rank-20 subspace (what
    Maxwell filtering does to MEG), fitted with automatic rank detection and
    sharing on. The sensor-space back-map is a pseudo-inverse, so a non-square
    ``(20, 32)`` sphere is not an obstacle (issue #253; the MLX twin of
    test_ng_sharing.py::test_low_rank_projected_data_share_fit_completes)."""
    from pamica.mlx_impl import AMICAMLXNG

    x = _real_data()
    x = x - x.mean(axis=1, keepdims=True)
    rank = 20
    u_r = np.linalg.svd(x, full_matrices=False)[0][:, :rank]
    x_low = u_r @ (u_r.T @ x)

    model = AMICAMLXNG(
        n_channels=NW,
        n_models=2,
        n_mix=NMIX,
        seed=SEED,
        block_size=BLOCK,
        share_comps=True,
        share_start=8,
        share_iter=10,
        comp_thresh=0.9,
    )
    model.fit(x_low, max_iter=25, verbose=False)

    assert model.n_channels == rank  # rank detection reduced the model
    assert model._sphere_np is not None and model._sphere_np.shape == (rank, NW)
    assert model._pinv_sphere().shape == (NW, rank)
    assert int(np.array(model.comp_used).sum()) < model.n_comps
    _assert_share_result_consistent(model)


def test_second_identify_call_does_not_resurrect_merged_columns():
    """The mask must not forget columns merged away by an earlier cycle.

    A second scan over an already-merged ``comp_list`` hits the ``ci == cj``
    guard on every folded pair, so nothing merges -- and a mask rebuilt from
    scratch during that loop would come back all-True while half the columns are
    dead. That was issue #240 on the NumPy backend
    (test_numpy_share_comps.py::test_comp_used_survives_a_second_identify_call);
    MLX shares the same kernel, so it must hold here across share cycles too.
    """
    model, _ = _warm_model(warmup=3)
    _force_merged_column(model)
    model.comp_thresh = 0.9
    cl_before = np.array(model.comp_list)
    used_before = np.array(model.comp_used)
    assert not used_before.all(), "setup failed: no column was merged away"

    model._identify_shared_comps()
    used_after = np.array(model.comp_used)

    assert used_after.sum() <= used_before.sum(), "a merged-away column came back"
    np.testing.assert_array_equal(
        np.array(model.comp_used),
        np.isin(np.arange(model.n_comps), np.array(model.comp_list)),
    )
    # Columns folded away by the first merge are still folded away.
    for dead in np.where(~used_before)[0]:
        assert dead not in np.array(model.comp_list)
    assert np.unique(np.array(model.comp_list)).size <= np.unique(cl_before).size


# --- (d) merged-away columns are frozen --------------------------------------


def test_merged_away_columns_keep_their_last_finite_value():
    """A column no model references receives no sufficient statistics, so its
    mixture update would be 0/0 and its ``dAk`` is exactly zero. It must freeze
    at its last finite value rather than go NaN (which ``fit``'s ``nan_params``
    guard would then -- correctly -- abort on). ``doscaling`` is off so the
    comparison is exact: the rescale pass renormalizes every column, dead ones
    included, by a norm that is 1.0 only to within a ULP.
    """
    model, x_t = _warm_model(warmup=3, doscaling=False)
    _force_merged_column(model)
    dead = ~np.array(model.comp_used)
    assert dead.any(), "setup failed: no column was merged away"
    before = {
        name: np.array(getattr(model, name))[:, dead]
        for name in ("A", "mu", "alpha", "beta", "rho")
    }
    live_mu_before = np.array(model.mu)[:, ~dead]

    for it in range(3, 6):
        model.iteration = it
        model._update_parameters(model._accumulate_blocks(x_t), x_t.shape[1])
    mx.eval(model.A, model.mu, model.alpha, model.beta, model.rho)

    for name, expected in before.items():
        actual = np.array(getattr(model, name))[:, dead]
        assert np.all(np.isfinite(actual)), f"{name} went non-finite on a dead column"
        np.testing.assert_array_equal(actual, expected, err_msg=name)
    # The live columns did keep moving, so "unchanged" above means frozen, not
    # "nothing happened in these three iterations".
    assert not np.array_equal(np.array(model.mu)[:, ~dead], live_mu_before)


# --- (e) the post-merge A-freeze window ---------------------------------------


def test_a_frozen_window_matches_the_torch_schedule():
    """Identical window to AMICATorchNG: the merge iteration and the 5 after."""
    from pamica.mlx_impl import AMICAMLXNG

    model = AMICAMLXNG(
        n_channels=8,
        n_models=2,
        share_comps=True,
        share_start=10,
        share_iter=20,
    )

    def frozen(itf):
        model.iteration = itf - 1  # itf is the Fortran-style 1-indexed iteration
        return model._a_frozen()

    assert not any(frozen(i) for i in range(1, 10))  # before share_start
    assert all(frozen(i) for i in range(10, 16))  # merge + 5 (residue 0..5)
    assert not any(frozen(i) for i in range(16, 30))  # thawed rest of cycle
    assert all(frozen(i) for i in range(30, 36))  # next cycle boundary


def test_a_frozen_is_off_for_a_single_model():
    """A model cannot share with itself, so sharing never freezes A there."""
    from pamica.mlx_impl import AMICAMLXNG

    model = AMICAMLXNG(
        n_channels=8, n_models=1, share_comps=True, share_start=2, share_iter=8
    )
    model.iteration = 3
    assert model._a_frozen() is False


def test_freeze_holds_A_while_the_mixture_keeps_moving():
    """Inside the settle window A (and its lrate ramp) are held, but the mixture
    parameters and the gradient norm keep updating -- Fortran computes ``dAk``/
    ``ndtmpsum`` in the accumulation pass, which runs whether or not the A step
    is taken (issue #207). A moves again at the sixth iteration after the merge.
    """
    share_start, share_iter = 4, 20
    model, x_t = _warm_model(
        warmup=share_start - 1,  # iterations 0..2, so the next itf is share_start
        share_comps=True,
        share_start=share_start,
        share_iter=share_iter,
        doscaling=False,
    )
    # The ramp saturates at lrate_cap on the first iteration, so it can only be
    # seen moving from a rate fit() has annealed below the cap (its LL-decrease
    # branch); start the window from such a rate.
    model.lrate = 0.25 * model.lrate_cap
    a_start = np.array(model.A)
    mu_start = np.array(model.mu)
    lrate_start = model.lrate

    frozen_iters = range(share_start - 1, share_start + 5)  # itf = 4..9
    for it in frozen_iters:
        model.iteration = it
        assert model._a_frozen() is True
        model._update_parameters(model._accumulate_blocks(x_t), x_t.shape[1])
        mx.eval(model.A, model.mu)
        np.testing.assert_array_equal(np.array(model.A), a_start)
        assert model.lrate == lrate_start  # the ramp is held with the step
        assert model._ndtmpsum is not None and model._ndtmpsum > 0.0

    assert not np.array_equal(np.array(model.mu), mu_start), (
        "the mixture parameters were frozen too; only A should be held"
    )

    model.iteration = share_start + 5  # itf = share_start + 6: thawed
    assert model._a_frozen() is False
    model._update_parameters(model._accumulate_blocks(x_t), x_t.shape[1])
    mx.eval(model.A)
    assert not np.array_equal(np.array(model.A), a_start)
    assert model.lrate > lrate_start


# --- (g) constructor validation ----------------------------------------------


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(share_start=0), "share_start"),
        (dict(share_iter=6), "share_iter"),
        (dict(share_iter=1), "share_iter"),
        (dict(comp_thresh=0.0), "comp_thresh"),
        (dict(comp_thresh=1.5), "comp_thresh"),
    ],
)
def test_share_constructor_validation(kwargs, match):
    """Rejected up front, for the same reasons and with the same messages as
    ``AMICATorchNG``. ``share_iter <= 6`` is the one that bites: the post-merge
    A-freeze is 6 iterations long, so a shorter cycle would hold A frozen on
    every iteration of every cycle and the mixing matrix would stop moving."""
    from pamica.mlx_impl import AMICAMLXNG

    with pytest.raises(ValueError, match=match):
        AMICAMLXNG(n_channels=8, n_models=2, share_comps=True, **kwargs)


def test_share_settings_are_not_validated_when_sharing_is_off():
    """The validation is gated on ``share_comps``, as in AMICATorchNG, so a
    default-constructed model carries the (unused) defaults untouched."""
    from pamica.mlx_impl import AMICAMLXNG

    model = AMICAMLXNG(n_channels=8, n_models=2, share_start=0, share_iter=1)
    assert model.share_comps is False and model.share_iter == 1


def test_single_model_sharing_is_accepted_and_inert():
    """``share_comps=True`` with one model is legal and does nothing: a model
    cannot share a component with itself (torch semantics)."""
    from pamica.mlx_impl import AMICAMLXNG

    model = AMICAMLXNG(
        n_channels=NW,
        n_models=1,
        n_mix=NMIX,
        seed=SEED,
        block_size=BLOCK,
        share_comps=True,
        share_start=2,
        share_iter=8,
        comp_thresh=0.5,  # low enough to merge anything, if it ran at all
    )
    model.fit(_real_data(), max_iter=8, verbose=False)
    assert model.stop_reason not in AMICAMLXNG._DEGENERATE_STOP_REASONS
    np.testing.assert_array_equal(
        np.array(model.comp_list), np.arange(NW).reshape(NW, 1)
    )
    assert bool(np.array(model.comp_used).all())
    assert model.shared_components() == []


# --- (h) degenerate sphere ----------------------------------------------------


def test_non_finite_sphere_fails_loudly():
    """A degenerate fit's sphere has no pseudo-inverse, and that must raise
    rather than let the per-pair finiteness check inside the merge kernel
    silently decline every merge (twin of the torch and NumPy tests)."""
    model, _ = _warm_model(warmup=2)
    model._sphere_np = np.full_like(model._sphere_np, np.nan)
    model._sphere_pinv = None
    with pytest.raises(RuntimeError, match="non-finite"):
        model._identify_shared_comps()


def test_pinv_sphere_before_fit_raises():
    """The back-map needs a preprocessed model; asking for it earlier is a usage
    error, not a silently empty result."""
    from pamica.mlx_impl import AMICAMLXNG

    model = AMICAMLXNG(n_channels=NW, n_models=2, n_mix=NMIX)
    with pytest.raises(RuntimeError, match="fit"):
        model._pinv_sphere()
