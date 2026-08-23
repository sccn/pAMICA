"""Newton: MLX (float32) vs PyTorch (float64) agreement (issue #264).

Cross-backend by design, so it lives in ``pamica/tests/`` rather than
``pamica/tests/mlx_tests/`` (``.rules/backend_parity.md``: a test that pins two
backends against each other belongs to neither of them -- same placement as
``test_mlx_convergence_stops.py`` and ``test_mlx_sharing_cross_backend.py``). The
MLX-only mechanics of Newton -- the schedule arms, the pre-update-mu ordering,
the sharing interplay -- stay in ``mlx_tests/test_mlx_newton.py``.

The float64 ``AMICATorchNG`` is the oracle: MLX has no float64 on the Apple GPU,
so the question this phase had to answer was whether the curvature
(``sigma2``/``kappa``/``lambda``) and the 2x2 direction solve survive float32.
Every comparison here starts from ONE real fitted state copied into both
backends, so only the arithmetic differs. Measured on the bundled sample:

* the finalized curvature agrees to ~4e-7 relative, at both an early and a
  warmed state (the go/no-go's G1);
* one warmed Newton M-step moves ``A`` to within 2.4e-7 absolute of the float64
  twin's;
* the positive-definiteness decision agrees at both states, including the
  early-block state where BOTH backends reject (min ``prod-1`` = -0.96 there,
  nowhere near the guard boundary, so the shared decision is not a coincidence
  of float32 noise).

Real bundled sample EEG only, no synthetic data and no mocked curvature
(``.rules/testing.md``); the fallback path is reached by evaluating a genuinely
under-determined 256-sample block, which is how ``test_ng_backend.py`` reaches it
on the PyTorch side. MLX is an optional Apple-Silicon backend, so the module
self-skips via ``importorskip`` plus an Apple-GPU guard; PyTorch always runs.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from pamica.torch_impl.core import AMICATorchNG

mx = pytest.importorskip("mlx.core", reason="MLX not installed (Apple Silicon only)")
mlx_core = pytest.importorskip(
    "pamica.mlx_impl.core", reason="MLX not installed (Apple Silicon only)"
)
AMICAMLXNG = mlx_core.AMICAMLXNG

SAMPLE_DIR = Path(__file__).resolve().parents[1] / "sample_data"
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


def _warm_model(warmup: int, block_size: int = BLOCK, n_models: int = 1, **kwargs):
    """An MLX Newton model driven ``warmup`` M-steps past init on the real
    sample, returned with its sphered data. Hand-driving the loop (rather than
    calling ``fit``) is what lets the comparisons below start both backends from
    one identical mid-fit state; every step still goes through the production
    ``_update_parameters``."""
    model = AMICAMLXNG(
        n_channels=NW,
        n_models=n_models,
        n_mix=NMIX,
        seed=SEED,
        block_size=block_size,
        do_newton=True,
        newt_start=0,
        **kwargs,
    )
    x_t = model._preprocess(_real_data())
    model._initialize_parameters()
    for it in range(warmup):
        model.iteration = it
        model._update_parameters(model._accumulate_blocks(x_t), x_t.shape[1])
    mx.eval(model.A, model.mu, model.alpha, model.beta, model.rho, model.gm, model.c)
    model.iteration = warmup
    return model, x_t


def _torch_twin(model, x_t, block_size: int = BLOCK):
    """A float64 ``AMICATorchNG`` holding the MLX model's exact state, plus the
    same sphered data as a float64 tensor. The two then differ only in the
    arithmetic they run, so a one-iteration comparison isolates the update."""
    dtype = torch.float64
    ng = AMICATorchNG(
        n_channels=NW,
        n_models=model.n_models,
        n_mix=NMIX,
        device="cpu",
        dtype=dtype,
        do_newton=True,
        newt_start=model.newt_start,
        newtrate=model.newtrate0,
        newt_ramp=model.newt_ramp,
        block_size=block_size,
        seed=SEED,
        keep_best=False,
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
    # The whole mutable schedule state, not just lrate: the Newton ramp reads
    # lrate_cap and newtrate as well, and a mismatch there would show up as an
    # A-step difference that has nothing to do with the curvature under test.
    ng.lrate = model.lrate
    ng.lrate_cap = model.lrate_cap
    ng.newtrate = model.newtrate
    ng.rholrate = model.rholrate
    ng.iteration = model.iteration
    # sldet cancels in the cross-model softmax, but carry it anyway so the two
    # states are identical in every field either backend reads.
    ng.sldet = model.sldet
    ng._update_unmixing_matrices()
    x_ng = torch.from_numpy(np.array(x_t).astype(np.float64)).to(dtype)
    return ng, x_ng


def _relerr(a: np.ndarray, b: np.ndarray) -> float:
    """Max relative error against the float64 reference ``b``, floored so a
    near-zero reference entry cannot manufacture a huge ratio."""
    scale = np.maximum(np.abs(b), np.abs(b).max() * 1e-6)
    return float(np.max(np.abs(a - b) / scale))


@pytest.mark.parametrize("warmup", [1, 55])
def test_newton_curvature_matches_float64_twin(warmup):
    """G1: the finalized Newton curvature survives float32.

    ``sigma2``/``kappa``/``lambda`` are the whole float32 question -- they are
    sums of squares over every sample, then divided by the model mass, and they
    feed a solve whose guard is a comparison against exactly 1. Both an early
    (``warmup=1``) and a warmed (``warmup=55``) state are checked, because the
    curvature grows by an order of magnitude over a fit and a precision problem
    could appear only at one end. Measured max relative error: 4e-7 at both.
    """
    model, x_t = _warm_model(warmup)
    ng, x_ng = _torch_twin(model, x_t)

    acc = model._accumulate_blocks(x_t)
    sigma2, lambda_, kappa = model._finalize_newton_stats(acc)
    acc_ng = ng._accumulate_blocks(x_ng)
    sigma2_t, lambda_t, kappa_t = ng._finalize_newton_stats(acc_ng)

    for name, got, ref in (
        ("sigma2", sigma2, sigma2_t),
        ("lambda", lambda_, lambda_t),
        ("kappa", kappa, kappa_t),
    ):
        a = np.array(got, dtype=np.float64)  # MLX is model-major (n_models, n_ch)
        b = ref.numpy().T  # torch is (n_ch, n_models)
        assert a.shape == b.shape, f"{name} layout: {a.shape} vs {b.shape}"
        err = _relerr(a, b)
        assert err < 1e-4, f"{name} differs from the float64 twin by {err:.2e}"
        # Curvature is a responsibility-weighted sum of squares: strictly
        # positive, so a sign flip (a broadcast/transpose slip) cannot pass.
        assert np.all(a > 0), f"{name} is not strictly positive"


def test_warmed_newton_mstep_matches_float64_twin():
    """G1: one warmed Newton M-step -- posdef path -- matches the float64 twin.

    Warmed 55 iterations first so the positive-definite branch is reached
    NATURALLY (no mocked curvature): at this state the minimum off-diagonal
    ``prod - 1`` is 2.27, i.e. every source pair clears the guard with two orders
    of margin, matching the float64 conditioning measured for this data in
    ``.context/issue-145/setup_and_config.md``. The whole M-step is compared, not
    just the direction, so the ramp-to-``newtrate`` branch is covered too.
    """
    model, x_t = _warm_model(55)
    ng, x_ng = _torch_twin(model, x_t)

    model._update_parameters(model._accumulate_blocks(x_t), x_t.shape[1])
    ng._update_parameters(ng._accumulate_blocks(x_ng), x_ng.shape[1])
    mx.eval(model.A)

    # The posdef branch is what this test is about; a silent fallback on either
    # side would compare the natural gradient instead and pass vacuously.
    assert model.n_newton_fallbacks == 0, "MLX fell back; the posdef path is untested"
    assert ng.n_newton_fallbacks == 0, "torch fell back; the posdef path is untested"
    # Both ramped toward newtrate (0.5), not the natural-gradient cap (0.1).
    assert model.lrate == pytest.approx(ng.lrate)
    assert model.lrate > model.lrate_cap

    assert ng.A is not None
    gap = np.abs(np.array(model.A, dtype=np.float64) - ng.A.numpy()).max()
    # 1e-4 is two orders above the 2.4e-7 these two actually reach and two below
    # the 1e-2 scale of the Newton step itself, so a wrong direction cannot pass.
    assert gap < 1e-4, f"A diverged from the float64 Newton M-step by {gap:.2e}"


def test_early_block_newton_falls_back_like_the_float64_twin():
    """The positive-definiteness guard rejects the same state in both backends,
    and the rejected iteration takes the natural gradient.

    A single 256-sample block from init genuinely under-determines the curvature
    (measured min off-diagonal ``prod - 1`` = -0.96 across 344 of 992 pairs), so
    the fallback is reached by real data rather than by injected numbers -- the
    same way ``test_ng_backend.py::test_newton_mstep_matches_numpy_reference``
    reaches it on the PyTorch side. Being that far from the ``prod > 1`` boundary
    is also why float32 and float64 cannot disagree about the decision here.
    """
    blk = 256
    model, x_t = _warm_model(0, block_size=blk)
    model.iteration = 5  # >= newt_start, so Newton is active
    ng, x_ng = _torch_twin(model, x_t, block_size=blk)

    block = x_t[:, :blk]
    acc = model._get_block_updates(block)
    sigma2, lambda_, kappa = model._finalize_newton_stats(acc)
    eye = mx.eye(NW)
    _, posdef = model._newton_direction(
        -acc["dWtmp"][0] / acc["dgm"][0] + eye, sigma2[0], lambda_[0], kappa[0]
    )
    assert not posdef, "the early block is positive definite; the test is vacuous"

    model._update_parameters(acc, blk)
    block_ng = torch.from_numpy(np.array(block).astype(np.float64)).to(torch.float64)
    acc_ng = ng._get_block_updates(block_ng)
    ng._update_parameters(acc_ng, blk)
    mx.eval(model.A)

    # Same decision, and the rejected iteration is counted once on both sides.
    assert model.n_newton_fallbacks == 1
    assert ng.n_newton_fallbacks == 1
    # Fallback ramps toward lrate_cap, NOT newtrate.
    assert model.lrate <= model.lrate_cap
    assert model.lrate == pytest.approx(ng.lrate)

    assert ng.A is not None
    gap = np.abs(np.array(model.A, dtype=np.float64) - ng.A.numpy()).max()
    assert gap < 1e-4, f"the fallback step diverged from float64 by {gap:.2e}"

    # And the fallback really is the plain natural gradient: an identically
    # seeded do_newton=False model, stepped from the same state, lands on the
    # same A (both take the lrate_cap ramp and the same direction).
    plain = AMICAMLXNG(
        n_channels=NW, n_mix=NMIX, seed=SEED, block_size=blk, do_newton=False
    )
    x_plain = plain._preprocess(_real_data())
    plain._initialize_parameters()
    plain.iteration = 5
    plain._update_parameters(plain._get_block_updates(x_plain[:, :blk]), blk)
    mx.eval(plain.A)
    assert np.array_equal(np.array(model.A), np.array(plain.A))


def _force_merged_column(model):
    """Fold model 1's first component into model 0's, as a fit-time merge does.

    Deterministic on purpose: keying the comparison on whether a merge happened
    to occur would let it skip under the very bug it guards. Same helper as
    ``test_mlx_sharing_cross_backend.py``.
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


@pytest.mark.parametrize("merged", [False, True])
def test_multimodel_newton_mstep_matches_float64_twin(merged):
    """Two models, with and without a shared mixing column, still match the
    float64 twin through a full Newton M-step.

    Multi-model is where the curvature reduction's broadcast axis becomes
    observable (the NumPy backend's issue #267 crash), and ``merged=True`` adds
    the ``share_comps`` interaction: one column carries both models' Newton
    directions through the ``gm``-weighted ``dAk`` average, and one column is
    frozen. The curvature itself is indexed by (model, SOURCE) rather than by
    mixing column, so a merge must NOT create a 0/0 there -- this is what pins
    that.
    """
    model, x_t = _warm_model(5, n_models=2)
    if merged:
        _force_merged_column(model)
    ng, x_ng = _torch_twin(model, x_t)

    acc = model._accumulate_blocks(x_t)
    acc_ng = ng._accumulate_blocks(x_ng)
    for name, got, ref in zip(
        ("sigma2", "lambda", "kappa"),
        model._finalize_newton_stats(acc),
        ng._finalize_newton_stats(acc_ng),
    ):
        a = np.array(got, dtype=np.float64)
        b = ref.numpy().T
        assert a.shape == (2, NW), f"{name} layout {a.shape}"
        err = _relerr(a, b)
        assert err < 1e-4, f"{name} differs from the float64 twin by {err:.2e}"

    fallbacks_before = model.n_newton_fallbacks
    model._update_parameters(acc, x_t.shape[1])
    ng._update_parameters(acc_ng, x_ng.shape[1])
    mx.eval(model.A)

    assert model.n_newton_fallbacks == fallbacks_before, "MLX fell back"
    assert ng.n_newton_fallbacks == 0, "torch fell back"
    assert ng.A is not None
    gap = np.abs(np.array(model.A, dtype=np.float64) - ng.A.numpy()).max()
    assert gap < 1e-4, f"A diverged from the float64 Newton M-step by {gap:.2e}"
    # The gradient norm reads the same Newton-preconditioned dAk, under the same
    # comp_used mask, so it has to agree too.
    assert model._ndtmpsum is not None and ng._ndtmpsum is not None
    assert model._ndtmpsum == pytest.approx(ng._ndtmpsum, rel=1e-4)


@pytest.mark.slow
def test_newton_fit_ll_matches_float64_torch():
    """G3: at a matched 100-iteration budget on the full recording, the MLX
    float32 Newton fit is not materially worse than float64 PyTorch Newton.

    This is the MLX analogue of ``test_ng_float32_stability.py::
    test_float32_ll_matches_float64``, and it is the acceptance criterion the
    float32 go/no-go turned on: a float32 curvature that quietly degraded the
    Hessian would show up here as a lower converged likelihood, not as a NaN.
    The bar is relational to the in-test float64 fit, never a hardcoded LL, and
    one-sided ("not materially worse") is the load-bearing half -- the
    trajectories diverge chaotically, so the two-sided band is a sanity guard.
    """
    data = _real_data(n_samples=FIELD)
    mlx_m = AMICAMLXNG(n_channels=NW, n_mix=NMIX, seed=SEED, do_newton=True)
    mlx_m.fit(data, max_iter=100, verbose=False)

    ng = AMICATorchNG(
        n_channels=NW,
        n_models=1,
        n_mix=NMIX,
        seed=SEED,
        device="cpu",
        dtype=torch.float64,
        do_newton=True,
    )
    ng.fit(data, max_iter=100, verbose=False)

    assert mlx_m.final_ll_ is not None and ng.final_ll_ is not None
    assert np.isfinite(mlx_m.final_ll_)
    assert mlx_m.stop_reason not in AMICAMLXNG._DEGENERATE_STOP_REASONS
    assert mlx_m.final_ll_ >= ng.final_ll_ - 0.05
    assert abs(mlx_m.final_ll_ - ng.final_ll_) < 0.1
    # Newton must actually have run to completion on this well-conditioned data
    # (the PyTorch backend's own bar on the sample: zero fallbacks).
    assert mlx_m.n_newton_fallbacks == 0
