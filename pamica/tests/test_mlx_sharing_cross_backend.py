"""Component sharing: MLX vs PyTorch agreement (issue #263).

Cross-backend by design, so it lives in ``pamica/tests/`` rather than
``pamica/tests/mlx_tests/``: ``.rules/backend_parity.md`` puts a test that pins
two backends against each other outside any one backend's subdirectory, because
a divergence here is not one backend's problem (same placement as
``test_mlx_convergence_stops.py`` and ``test_rank_policy.py``). The MLX-only
mechanics of sharing -- validation, the freeze schedule, frozen dead columns --
stay in ``mlx_tests/test_mlx_sharing.py``.

Two things are pinned here, both driven from ONE real fitted state copied into
both backends so only the operation under test differs:

1. the merge DECISION (MLX calls the same
   ``pamica.numpy_impl.utils.identify_shared_components`` kernel the NumPy
   backend uses, on the same host float64 ``pinv(sphere) @ A``); and
2. the A-update's ``gm`` weighting, which Fortran takes from the PREVIOUS
   iteration (``dAk`` is accumulated at amica15.f90:1749-1761, before
   ``update_params`` reassigns ``gm`` at :1788). Issue #219 raised that ordering
   question for ``numpy_impl``'s ``ndtmpsum`` and flagged the array backends as
   follow-up; PyTorch was fixed there, MLX here.

MLX is an optional Apple-Silicon backend, so the module self-skips via
``importorskip`` plus an Apple-GPU guard; PyTorch always runs. Real bundled
sample EEG only, no synthetic data or mocks (``.rules/testing.md``).
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


def _warm_model(warmup: int = 5, **kwargs):
    """An MLX model driven ``warmup`` M-steps past init on the real sample,
    returned with its sphered data. Hand-driving the loop (rather than calling
    ``fit``) is what lets the comparison below start both backends from one
    identical mid-fit state; every step still goes through the production
    ``_update_parameters``."""
    model = AMICAMLXNG(
        n_channels=NW,
        n_models=2,
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

    Returns ``(kept, dead)``. Deterministic on purpose: keying the comparison on
    whether a merge happened to occur would let it skip under the very bug it
    guards.
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


@pytest.mark.parametrize("merged", [False, True])
def test_gm_prev_weighting_matches_torch(merged):
    """The A-update weights ``dAk`` with the PREVIOUS iteration's ``gm``.

    Fortran builds ``dAk`` in ``accum_updates_and_likelihood``
    (amica15.f90:1749-1761), before ``update_params`` reassigns ``gm``
    (:1788); ``AMICATorchNG`` snapshots ``gm_prev`` for exactly that reason and
    MLX used the post-update ``gm``. Both backends run one iteration from one
    shared state, so only the update is compared.

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


def test_comp_used_before_fit_raises_in_both_backends():
    """``comp_used`` on an unfitted model is a usage error in both backends, with
    the same message shape. The PyTorch side used a bare ``assert``, which
    ``python -O`` strips -- turning this into an obscure ``NoneType`` error."""
    with pytest.raises(RuntimeError, match="requires a fitted model"):
        _ = AMICAMLXNG(n_channels=NW, n_models=2, n_mix=NMIX).comp_used
    with pytest.raises(RuntimeError, match="requires a fitted model"):
        _ = AMICATorchNG(n_channels=NW, n_models=2, n_mix=NMIX, device="cpu").comp_used
