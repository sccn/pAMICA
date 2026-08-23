"""Component sharing (``share_comps``) on the MLX backend -- issue #263.

Apple-Silicon only, real sample EEG (no synthetic/mock), same module guards as
``test_mlx_backend.py``. The port mirrors ``AMICATorchNG`` decision for decision
(``tests/torch_tests/test_ng_sharing.py``) and reuses the NumPy merge kernel
``pamica.numpy_impl.utils.identify_shared_components``, so the cross-backend
merge decision is an acceptance test here rather than a re-derivation.

Two invariants carry the whole port:

* with sharing off (or not yet scheduled) every new masking/weighting step is a
  no-op, so the validated default trajectory is bit-identical; and
* the A-update weights each model's direction with the PREVIOUS iteration's
  ``gm`` (Fortran builds ``dAk`` before ``update_params`` reassigns ``gm``,
  amica15.f90:1788; issue #219 for the PyTorch twin). That weighting cancels
  exactly for a disjoint ``comp_list``, so it is only observable on a SHARED
  column -- which is what ``test_gm_prev_weighting_matches_torch[True]`` drives.
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


def _torch_twin(model, x_t):
    """A float64 ``AMICATorchNG`` holding the MLX model's exact state, plus the
    same sphered data as a float64 tensor. The two then differ only in the
    arithmetic they run, so a one-iteration comparison isolates the update."""
    import torch

    from pamica.torch_impl.core import AMICATorchNG

    dtype = torch.float64
    ng = AMICATorchNG(
        n_channels=NW,
        n_models=model.n_models,
        n_mix=NMIX,
        device="cpu",
        dtype=dtype,
        do_newton=False,
        block_size=BLOCK,
        seed=SEED,
        doscaling=model.doscaling,
        scalestep=model.scalestep,
    )
    ng._initialize_parameters()
    for name in ("A", "mu", "alpha", "beta", "rho", "gm", "c"):
        value = np.array(getattr(model, name)).astype(np.float64)
        setattr(ng, name, torch.from_numpy(value).to(dtype))
    ng.comp_list = torch.from_numpy(np.array(model.comp_list).astype(np.int64))
    ng.sphere = torch.from_numpy(model._sphere_np.copy()).to(dtype)
    ng._sphere_pinv = None
    ng.lrate = model.lrate
    ng.rholrate = model.rholrate
    ng.iteration = model.iteration
    # sldet cancels in the cross-model softmax, but carry it anyway so the two
    # states are identical in every field either backend reads.
    ng.sldet = model.sldet
    ng._update_unmixing_matrices()
    x_ng = torch.from_numpy(np.array(x_t).astype(np.float64)).to(dtype)
    return ng, x_ng


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


# --- (b) cross-backend merge decision (acceptance) ---------------------------


def test_merge_decision_matches_the_torch_backend():
    """Acceptance test: from one real fitted 2-model state, MLX and
    ``AMICATorchNG`` reach the identical merge decision.

    MLX does not re-derive the scan -- it calls the same NumPy kernel the NumPy
    backend uses (issue #258) on host float64 ``pinv(sphere) @ A``, which is the
    metric ``AMICATorchNG._identify_shared_comps`` computes inline. Driving both
    from one state is what pins that equality. The threshold is read off the
    data so exactly one pair (the perturbed near-collinear one) is above it,
    making the comparison non-vacuous and its outcome exact.
    """
    model, x_t = _warm_model(warmup=3)
    cl = np.array(model.comp_list)
    i0, i1 = int(cl[0, 0]), int(cl[0, 1])
    # Perturb one cross-model column into NEAR- (not exact-) collinearity, so
    # the decision itself is under test rather than a pre-merged state.
    a_np = np.array(model.A)
    rng = np.random.RandomState(0)
    a_np[:, i1] = a_np[:, i0] + 1e-3 * rng.standard_normal(NW).astype(np.float32)
    model.A = mx.array(a_np)
    model._update_unmixing_matrices()
    mx.eval(model.A, model.W)

    # Separate the perturbed pair from the runner-up: any threshold between the
    # two merges exactly one pair, in either backend.
    atil = model._pinv_sphere() @ np.array(model.A, dtype=np.float64)
    norms = np.linalg.norm(atil, axis=0)
    cosines = sorted(
        abs(atil[:, int(cl[i, 0])] @ atil[:, int(cl[j, 1])])
        / (norms[int(cl[i, 0])] * norms[int(cl[j, 1])])
        for i in range(NW)
        for j in range(NW)
    )
    assert cosines[-1] > cosines[-2], "no separation to threshold on"
    thresh = 0.5 * (cosines[-1] + cosines[-2])
    assert cosines[-1] > thresh > cosines[-2]  # strictly between, in float64
    model.comp_thresh = thresh

    ng, _ = _torch_twin(model, x_t)
    ng.comp_thresh = thresh

    model._identify_shared_comps()
    ng._identify_shared_comps()

    merged_cl = np.array(model.comp_list)
    assert not np.array_equal(merged_cl, cl), "no merge fired; the test is vacuous"
    assert np.unique(merged_cl).size == NW * 2 - 1, "expected exactly one merge"
    np.testing.assert_array_equal(merged_cl, ng.comp_list.numpy())
    # The cached mask follows the merged comp_list, as torch's derived one does.
    np.testing.assert_array_equal(np.array(model.comp_used), ng.comp_used.cpu().numpy())


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


# --- (f) gm_prev weighting ----------------------------------------------------


@pytest.mark.parametrize("merged", [False, True])
def test_gm_prev_weighting_matches_torch(merged):
    """The A-update weights ``dAk`` with the PREVIOUS iteration's ``gm``.

    Fortran builds ``dAk`` in ``accum_updates_and_likelihood``
    (amica15.f90:1749-1761), before ``update_params`` reassigns ``gm``
    (:1788+); ``AMICATorchNG`` snapshots ``gm_prev`` for exactly that reason
    (issue #219) and MLX used the post-update ``gm``. Both backends run one
    iteration from one shared state, so only the update is compared.

    ``merged=False`` is the disjoint-``comp_list`` case: every column has a
    single contributor, so ``gm`` cancels exactly against ``zeta`` and the two
    weightings agree analytically -- reverting ``gm_prev`` there leaves the gap
    at 4.6e-8 (pure float32 noise), so that case CANNOT discriminate; it is kept
    as the plain one-iteration cross-backend pin. ``merged=True`` is where the
    weighting is observable at all: a shared column averages two models'
    directions with those weights. Measured on this state: 7.9e-7 with the fix
    against 9.9e-3 with the post-update ``gm`` restored -- four orders of
    magnitude, which is what the tolerance below discriminates on.
    """
    model, x_t = _warm_model(warmup=5, doscaling=False)
    if merged:
        _force_merged_column(model)
    ng, x_ng = _torch_twin(model, x_t)

    model.iteration = ng.iteration = 5
    model._update_parameters(model._accumulate_blocks(x_t), x_t.shape[1])
    ng._update_parameters(ng._accumulate_blocks(x_ng), x_ng.shape[1])
    mx.eval(model.A)

    assert ng.A is not None
    gap = np.abs(np.array(model.A, dtype=np.float64) - ng.A.numpy()).max()
    # 1e-4: two orders below the post-update-gm failure (9.9e-3) and two above
    # the float32-vs-float64 agreement this pair actually reaches (7.9e-7).
    assert gap < 1e-4, f"A diverged from the float64 torch update by {gap:.2e}"

    # Same state, same mask: the gradient norm must agree too. This is what pins
    # the comp_used masking of ndtmpsum, since torch divides by its own
    # count(comp_used); dropping the mask on the merged state would divide by
    # n_comps instead of n_comps-1, a 0.8% shift, far above the ~1e-6 relative
    # float32-vs-float64 agreement.
    assert model._ndtmpsum is not None and ng._ndtmpsum is not None
    assert model._ndtmpsum == pytest.approx(ng._ndtmpsum, rel=1e-4)


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
