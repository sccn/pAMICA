"""Tests for the natural-gradient EM backend (AMICATorchNG).

Real sample EEG data only (no synthetic/mock). The decisive correctness
check is that the vectorized per-block sufficient statistics match the
validated NumPy reference (``AMICA_NumPy._get_block_updates``) to float64
precision on an identical block with identical parameters.
"""

from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pytest
import scipy.special as sp
import torch

from pamica.torch_impl import AMICATorchNG
from pamica.torch_impl.core import _KEEP_BEST_TOL
from pamica.numpy_impl.core import AMICA as AMICA_NumPy

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "sample_data"
DATA_FILE = SAMPLE_DIR / "eeglab_data.fdt"
NW = 32
FIELD = 30504
NMIX = 3
SEED = 42


def _load_real_data() -> np.ndarray:
    from pamica.torch_impl.utils import load_eeglab_data

    return load_eeglab_data(str(DATA_FILE), data_dim=NW, field_dim=FIELD).astype(
        np.float64
    )


def _fresh_ng(block_size: int = 256, **kwargs: Any) -> AMICATorchNG:
    return AMICATorchNG(
        n_channels=NW,
        n_models=1,
        n_mix=NMIX,
        seed=SEED,
        device="cpu",
        dtype=torch.float64,
        block_size=block_size,
        **kwargs,
    )


def _numpy_ref_like(ng: AMICATorchNG, blk: int, **kwargs) -> AMICA_NumPy:
    """A NumPy AMICA carrying the NG backend's exact parameters (the
    established copy-params parity pattern). Model count follows ``ng``."""
    assert (
        ng.comp_list is not None
        and ng.A is not None
        and ng.W is not None
        and ng.c is not None
        and ng.mu is not None
        and ng.alpha is not None
        and ng.beta is not None
        and ng.rho is not None
        and ng.gm is not None
    )
    npm = AMICA_NumPy(num_models=ng.n_models, num_mix=NMIX, **kwargs)
    npm.data_dim = NW
    npm.num_comps = NW * ng.n_models
    npm.num_models = ng.n_models
    npm.num_mix = NMIX
    npm.block_size = blk
    npm.sldet = ng.sldet
    npm.comp_list = ng.comp_list.cpu().numpy()
    npm.A = ng.A.cpu().numpy().copy()
    npm.W = ng.W.cpu().numpy().copy()
    npm.c = ng.c.cpu().numpy().copy()
    npm.mu = ng.mu.cpu().numpy().copy()
    npm.alpha = ng.alpha.cpu().numpy().copy()
    npm.beta = ng.beta.cpu().numpy().copy()
    npm.rho = ng.rho.cpu().numpy().copy()
    npm.gm = ng.gm.cpu().numpy().copy()
    return npm


@pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing")
def test_sufficient_stats_match_numpy_reference():
    """Decisive parity: NG per-block sufficient statistics == NumPy reference.

    Copies the NG backend's exact parameters into a NumPy AMICA instance and
    compares ``_get_block_updates`` on an identical real-data block. These
    accumulators are what drive every M-step update, so machine-precision
    agreement proves the vectorized port is faithful. (``ll`` is excluded: the
    NG module intentionally computes the log-likelihood from pre-normalization
    mixture logits, which is more correct than the NumPy reference's
    post-normalization value; LL is checked against Fortran separately.)
    """
    data = _load_real_data()
    ng = _fresh_ng()
    X_t = ng._preprocess(data)
    ng._initialize_parameters()

    blk = 256
    block = X_t[:, :blk].contiguous()
    ng_upd = ng._get_block_updates(block)

    assert (
        ng.comp_list is not None
        and ng.A is not None
        and ng.W is not None
        and ng.c is not None
        and ng.mu is not None
        and ng.alpha is not None
        and ng.beta is not None
        and ng.rho is not None
        and ng.gm is not None
    )
    npm = AMICA_NumPy(num_models=1, num_mix=NMIX, do_newton=False)
    npm.data_dim = NW
    npm.num_comps = NW
    npm.num_models = 1
    npm.num_mix = NMIX
    npm.block_size = blk
    npm.comp_list = ng.comp_list.cpu().numpy()
    npm.A = ng.A.cpu().numpy().copy()
    npm.W = ng.W.cpu().numpy().copy()
    npm.c = ng.c.cpu().numpy().copy()
    npm.mu = ng.mu.cpu().numpy().copy()
    npm.alpha = ng.alpha.cpu().numpy().copy()
    npm.beta = ng.beta.cpu().numpy().copy()
    npm.rho = ng.rho.cpu().numpy().copy()
    npm.gm = ng.gm.cpu().numpy().copy()

    np_upd = npm._get_block_updates(block.cpu().numpy())

    keys = [
        "dgm",
        "dalpha_n",
        "dmu_n",
        "dmu_d",
        "dbeta_n",
        "dbeta_d",
        "drho_n",
        "dWtmp",
        "dc_numer",
    ]
    for key in keys:
        a = np.asarray(ng_upd[key].cpu().numpy(), dtype=np.float64)
        b = np.asarray(np_upd[key], dtype=np.float64).reshape(a.shape)
        max_diff = float(np.max(np.abs(a - b)))
        assert max_diff < 1e-8, f"{key} differs from NumPy reference by {max_diff:.3e}"


@pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing")
def test_blocking_invariance():
    """Accumulated sufficient statistics are independent of block_size."""
    data = _load_real_data()[:, :4096]

    def accumulate(block_size):
        m = _fresh_ng(block_size=block_size)
        X_t = m._preprocess(data)
        m._initialize_parameters()
        return m._accumulate_blocks(X_t)

    acc_a = accumulate(256)
    acc_b = accumulate(512)
    keys = [
        "dgm",
        "dalpha_n",
        "dmu_n",
        "dmu_d",
        "dbeta_n",
        "dbeta_d",
        "drho_n",
        "dWtmp",
        "dc_numer",
        "ll",
    ]
    for key in keys:
        a = acc_a[key].cpu().numpy()
        b = acc_b[key].cpu().numpy()
        assert np.allclose(a, b, atol=1e-8), f"{key} depends on block_size"


@pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing")
def test_blocking_invariance_multimodel_bias():
    """The data-space bias accumulator ``dc_numer`` (and the resulting ``c``) are
    block-size independent for ``n_models=2``. The single-model
    ``test_blocking_invariance`` cannot catch a dc_numer accumulation bug because
    dc_numer is ~0 there regardless of block size; with two models it is nonzero
    and a per-block reset/double-count would show up."""
    data = _load_real_data()[:, :4096]

    def accumulate(block_size):
        m = AMICATorchNG(
            n_channels=NW,
            n_models=2,
            n_mix=NMIX,
            seed=SEED,
            device="cpu",
            dtype=torch.float64,
            block_size=block_size,
        )
        X_t = m._preprocess(data)
        m._initialize_parameters()
        return m._accumulate_blocks(X_t)

    acc_a = accumulate(256)
    acc_b = accumulate(512)
    da = acc_a["dc_numer"].cpu().numpy()
    db = acc_b["dc_numer"].cpu().numpy()
    assert not np.allclose(da, 0.0), "multi-model dc_numer should be nonzero"
    assert np.allclose(da, db, atol=1e-8), "dc_numer depends on block_size"


@pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing")
def test_reported_ll_includes_jacobian_near_fortran_at_init():
    """At initialization the reported per-sample-per-channel LL sits in
    Fortran's range (~ -3 to -4), which requires the log|det W| and sldet
    Jacobian terms to be present (without them it would be off by ~sldet/nw)."""
    data = _load_real_data()
    m = _fresh_ng()
    m.fit(data, max_iter=1, verbose=False)
    ll0 = m.ll_history[0]
    assert -6.0 < ll0 < -2.0, (
        f"init LL {ll0:.3f} outside Fortran range; Jacobian likely missing"
    )


@pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing")
def test_forward_ll_per_source_factorization_order():
    """`_forward` must logsumexp over mixture components *per source* and only
    then sum over sources (amica17.f90:1313-1360).

    Guards the historical "issue #11" factorization-order bug for the default
    single-model NG path with a fast, deterministic, independent NumPy
    reference (no fit, no Fortran binary). Building the reference from NG's own
    parameters, a regression that swapped the reduction order (sum over sources
    before the mixture logsumexp) would match neither NG nor Fortran; the two
    orderings are asserted to differ so the check has teeth.
    """
    data = _load_real_data()
    ng = _fresh_ng()
    X_t = ng._preprocess(data)
    ng._initialize_parameters()

    block = X_t[:, :256].contiguous()
    # NG's per-sample block LL (n_models=1 -> logsumexp over the single model
    # collapses to the model-0 logV).
    ng_ll = ng._block_sample_ll(block).cpu().numpy()

    # Independent NumPy reference from NG's own tensors.
    assert (
        ng.W is not None
        and ng.c is not None
        and ng.comp_list is not None
        and ng.mu is not None
        and ng.beta is not None
        and ng.rho is not None
        and ng.alpha is not None
        and ng.gm is not None
    )
    block_np = block.cpu().numpy()
    W = ng.W.cpu().numpy()[:, :, 0]
    c = ng.c.cpu().numpy()[:, 0]
    idx = ng.comp_list.cpu().numpy()[:, 0]
    mu = ng.mu.cpu().numpy()[:, idx]
    beta = ng.beta.cpu().numpy()[:, idx]
    rho = ng.rho.cpu().numpy()[:, idx]
    alpha = ng.alpha.cpu().numpy()[:, idx]
    gm = float(ng.gm.cpu().numpy()[0])
    sldet = float(ng.sldet)

    b = block_np.T @ W - c  # (batch, n_sources)
    n_mix, n_sources = mu.shape
    # z0[sample, source, mix] = log alpha + log beta + generalized-Gaussian
    # log-pdf of beta*(b - mu), matching _log_pdf_and_deriv's branches.
    z0 = np.empty((b.shape[0], n_sources, n_mix))
    for k in range(n_mix):
        y = beta[k][None, :] * (b - mu[k][None, :])
        abs_y = np.abs(y)
        gg = -(abs_y ** rho[k][None, :]) - np.log(2.0) - sp.gammaln(1.0 + 1.0 / rho[k])
        lap = -abs_y - np.log(2.0)
        gau = -y * y - 0.5 * np.log(np.pi)
        log_pdf = np.where(rho[k] == 2.0, gau, np.where(rho[k] == 1.0, lap, gg))
        z0[:, :, k] = np.log(alpha[k])[None, :] + np.log(beta[k])[None, :] + log_pdf

    # slogdet emits benign RuntimeWarnings on some LAPACK backends even for
    # well-conditioned matrices; silence them (the value is verified against
    # NG's torch slogdet by the assert below).
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        _, logdet_W = np.linalg.slogdet(W)
    base = np.log(gm) + logdet_W + sldet
    # Correct order: logsumexp over mixture, then sum over sources.
    ll_correct = base + sp.logsumexp(z0, axis=2).sum(axis=1)
    # Swapped (buggy) order: sum over sources first, then logsumexp over mixture.
    ll_swapped = base + sp.logsumexp(z0.sum(axis=1), axis=1)

    np.testing.assert_allclose(ng_ll, ll_correct, rtol=1e-10, atol=1e-10)
    assert np.max(np.abs(ll_correct - ll_swapped)) > 1.0, (
        "correct and swapped factorization orders should differ substantially"
    )


@pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing")
def test_fit_produces_finite_unmixing_on_real_data():
    """A short fit on real data yields a finite, correctly-shaped W (no NaN)."""
    data = _load_real_data()
    m = _fresh_ng()
    m.fit(data, max_iter=10, verbose=False)
    W = m.get_unmixing_matrix(0)
    assert W.shape == (NW, NW)
    assert np.all(np.isfinite(W))
    assert np.all(np.isfinite(m.ll_history))


@pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing")
def test_newton_stats_match_numpy_reference():
    """Newton curvature statistics (sigma2, lambda, kappa) match the NumPy
    reference to float64 precision on an identical real-data block.

    Same copy-params pattern as the non-Newton parity test, with
    ``do_newton=True``. Both backends carry the Fortran-corrected Newton math
    (score ``fp`` rather than the density derivative, ``sbeta^2`` on kappa,
    the ``mu^2`` curvature term in lambda), so the finalized curvature stats
    that drive the Newton preconditioner must agree.
    """
    data = _load_real_data()
    ng = _fresh_ng(do_newton=True, newt_start=0)
    X_t = ng._preprocess(data)
    ng._initialize_parameters()

    blk = 256
    block = X_t[:, :blk].contiguous()
    ng_upd = ng._get_block_updates(block)
    sigma2_ng, lambda_ng, kappa_ng = ng._finalize_newton_stats(ng_upd)

    npm = _numpy_ref_like(ng, blk, do_newton=True)
    np_upd = npm._get_block_updates(block.cpu().numpy())
    dgm = np_upd["dgm"][None, :]
    refs = {
        "sigma2": (sigma2_ng, np_upd["dsigma2"] / dgm),
        "kappa": (kappa_ng, np_upd["dkappa"] / dgm),
        "lambda": (lambda_ng, np_upd["dlambda"] / dgm),
    }
    for name, (a_t, b_np) in refs.items():
        a = a_t.cpu().numpy()
        max_diff = float(np.max(np.abs(a - np.asarray(b_np).reshape(a.shape))))
        assert max_diff < 1e-8, f"{name} differs from NumPy reference by {max_diff:.3e}"


@pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing")
def test_newton_mstep_matches_numpy_reference():
    """One full Newton M-step (``_update_parameters`` with Newton active)
    produces the same mixing matrix as the NumPy reference.

    On this early real-data block the curvature is not yet positive definite,
    so both backends take the natural-gradient fallback; the test verifies the
    two ports agree bit-for-bit on that path (finalization, the posdef guard,
    the fallback ramp, the A update) to float64 precision. The positive-definite
    ``H`` composition is covered separately by
    ``test_newton_posdef_mstep_matches_reference``.
    """
    data = _load_real_data()
    blk = 256
    it = 5  # >= newt_start so Newton is active

    ng = _fresh_ng(do_newton=True, newt_start=0, newtrate=0.5)
    X_t = ng._preprocess(data)
    ng._initialize_parameters()
    ng.iteration = it
    block = X_t[:, :blk].contiguous()
    acc = ng._get_block_updates(block)
    ng._update_parameters(acc, blk)
    assert (
        ng.A is not None
        and ng.mu is not None
        and ng.beta is not None
        and ng.rho is not None
        and ng.alpha is not None
        and ng.gm is not None
    )
    A_ng = ng.A.cpu().numpy().copy()

    ng2 = _fresh_ng(do_newton=True, newt_start=0, newtrate=0.5)
    ng2._preprocess(data)
    ng2._initialize_parameters()
    npm = _numpy_ref_like(
        ng2, blk, do_newton=True, newt_start=0, newtrate=0.5, lrate=ng2.lrate0
    )
    npm.num_samples = blk
    npm.iter = it
    npm.doscaling = True
    npm.scalestep = 1
    npm.nd = []
    npm.ll = []
    npm.use_grad_norm = True
    np_upd = npm._get_block_updates(block.cpu().numpy())
    npm._update_parameters(np_upd)

    # Cross-check every parameter the M-step touches, not just A: the exact-EM
    # mu/beta and digamma rho updates are independently written in the two
    # backends and could diverge without A noticing.
    assert np.max(np.abs(A_ng - npm.A)) < 1e-8, "A"
    assert np.max(np.abs(ng.mu.cpu().numpy() - npm.mu)) < 1e-8, "mu"
    assert np.max(np.abs(ng.beta.cpu().numpy() - npm.beta)) < 1e-8, "beta"
    assert np.max(np.abs(ng.rho.cpu().numpy() - npm.rho)) < 1e-8, "rho"
    assert np.max(np.abs(ng.alpha.cpu().numpy() - npm.alpha)) < 1e-8, "alpha"
    assert np.max(np.abs(ng.gm.cpu().numpy() - npm.gm)) < 1e-8, "gm"


@pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing")
def test_newton_finalize_uses_preupdate_mu():
    """Regression (issue #24 review): ``_update_parameters`` must finalize the
    Newton curvature (lambda folds in mu^2) using the PRE-update mu, before the
    exact-EM mu update moves it. The torch backend previously read ``self.mu``
    after it had already been reassigned, computing lambda with the wrong mu.
    """
    data = _load_real_data()
    ng = _fresh_ng(do_newton=True, newt_start=0)
    X_t = ng._preprocess(data)
    ng._initialize_parameters()
    ng.iteration = 5  # >= newt_start so Newton finalization runs
    block = X_t[:, :256].contiguous()
    acc = ng._get_block_updates(block)

    assert ng.mu is not None
    mu_pre = ng.mu.clone()
    captured = {}
    original = ng._finalize_newton_stats

    def spy(a):
        assert ng.mu is not None
        captured["mu"] = ng.mu.clone()  # mu as seen by the finalization
        return original(a)

    with mock.patch.object(ng, "_finalize_newton_stats", side_effect=spy):
        ng._update_parameters(acc, 256)

    assert "mu" in captured, "Newton finalization was not invoked"
    assert torch.equal(captured["mu"], mu_pre), (
        "Newton finalization saw the post-update mu (issue #24 lambda bug)"
    )
    # Guard the test itself: mu genuinely moved, so pre != post is a real check.
    assert not torch.equal(ng.mu, mu_pre)


def test_newton_direction_matches_formula():
    """``_newton_direction`` reproduces the Fortran 2x2 solve and its
    positive-definiteness guard on controlled inputs (no data needed)."""
    rng = np.random.RandomState(0)
    n = 6
    dA = torch.from_numpy(rng.randn(n, n))
    # Large sigma2/kappa so sk1*sk2 > 1 for all pairs -> positive definite.
    sigma2 = torch.from_numpy(np.abs(rng.randn(n)) + 2.0)
    kappa = torch.from_numpy(np.abs(rng.randn(n)) + 2.0)
    lambda_ = torch.from_numpy(np.abs(rng.randn(n)) + 1.0)

    ng = _fresh_ng()
    ng.n_channels = n
    H, posdef = ng._newton_direction(dA, sigma2, lambda_, kappa)
    assert posdef
    H = H.numpy()
    s, k, lam, d = (
        sigma2.numpy(),
        kappa.numpy(),
        lambda_.numpy(),
        dA.numpy(),
    )
    for i in range(n):
        assert abs(H[i, i] - d[i, i] / lam[i]) < 1e-12
        for j in range(n):
            if i != j:
                sk1, sk2 = s[i] * k[j], s[j] * k[i]
                expected = (sk1 * d[i, j] - d[j, i]) / (sk1 * sk2 - 1.0)
                assert abs(H[i, j] - expected) < 1e-12

    # Tiny sigma2/kappa so no pair satisfies sk1*sk2 > 1 -> not posdef.
    small = torch.from_numpy(np.full(n, 0.1))
    _, posdef2 = ng._newton_direction(dA, small, lambda_, small)
    assert not posdef2


@pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing")
def test_newton_posdef_mstep_composition():
    """When the curvature is positive definite, the M-step applies the Newton
    direction ``H`` (not the natural gradient) and ramps toward ``newtrate``.

    On real data the curvature is not positive definite at these iterations
    (issue #21), so the posdef branch never fires in a plain fit. Here the
    finalized stats are forced positive definite to exercise that branch, and
    the resulting ``A`` is checked against an independent recomputation of the
    exact composition (slice -> H -> ``A + lrate*(A@H)`` -> column rescale).
    """
    data = _load_real_data()
    blk = 256
    big = torch.full((NW, 1), 5.0, dtype=torch.float64)
    lam = torch.full((NW, 1), 1.5, dtype=torch.float64)

    def forced(_acc):
        return big.clone(), lam.clone(), big.clone()

    ng = _fresh_ng(do_newton=True, newt_start=0, newtrate=1.0, lrate=0.1)
    X_t = ng._preprocess(data)
    ng._initialize_parameters()
    ng.iteration = 5
    block = X_t[:, :blk].contiguous()
    acc = ng._get_block_updates(block)
    assert ng.A is not None and ng.comp_list is not None
    A_before = ng.A.clone()

    # Independent expected A-update for the (single) model, using the same
    # forced stats and the unit-tested _newton_direction. The stored A is
    # Fortran's A^T, so the step is A -= lrate*(H^T @ A) (issue #24).
    eye = torch.eye(NW, dtype=torch.float64)
    dA_h = -acc["dWtmp"][:, :, 0] / acc["dgm"][0] + eye
    H, posdef = ng._newton_direction(dA_h, big[:, 0], lam[:, 0], big[:, 0])
    assert posdef
    lrate_after = min(1.0, ng.lrate + min(1.0 / ng.newt_ramp, ng.lrate))
    idx = ng.comp_list[:, 0]
    A_expected = A_before.clone()
    A_expected[:, idx] = A_before[:, idx] - lrate_after * (H.T @ A_before[:, idx])
    scale = torch.sqrt((A_expected**2).sum(dim=0))
    nz = scale > 0
    A_expected[:, nz] = A_expected[:, nz] / scale[nz]

    with mock.patch.object(ng, "_finalize_newton_stats", side_effect=forced):
        ng._update_parameters(acc, blk)

    assert ng.A is not None
    assert ng.n_newton_fallbacks == 0  # positive-definite branch taken
    assert ng.lrate == pytest.approx(lrate_after)  # ramped toward newtrate
    assert np.max(np.abs(ng.A.cpu().numpy() - A_expected.cpu().numpy())) < 1e-10
    assert np.all(np.isfinite(ng.A.cpu().numpy()))


@pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing")
@pytest.mark.parametrize("rho_val", [1.0, 2.0])
def test_newton_stats_match_at_rho_boundaries(rho_val):
    """Newton stats match the NumPy reference when rho is at the Laplace
    (1.0) / Gaussian (2.0) special cases, which real fits reach commonly.

    ``_score`` uses distinct closed forms at these boundaries (``sign(y)``,
    ``2y``), not just the generic ``rho*sign(y)*|y|^(rho-1)`` limit, so they
    need their own parity coverage.
    """
    data = _load_real_data()
    blk = 256
    ng = _fresh_ng(do_newton=True, newt_start=0)
    X_t = ng._preprocess(data)
    ng._initialize_parameters()
    assert ng.rho is not None
    ng.rho[:] = rho_val
    block = X_t[:, :blk].contiguous()
    ng_upd = ng._get_block_updates(block)
    sigma2_ng, lambda_ng, kappa_ng = ng._finalize_newton_stats(ng_upd)

    npm = _numpy_ref_like(ng, blk, do_newton=True)
    np_upd = npm._get_block_updates(block.cpu().numpy())
    dgm = np_upd["dgm"][None, :]
    for name, a_t, b_np in [
        ("sigma2", sigma2_ng, np_upd["dsigma2"] / dgm),
        ("kappa", kappa_ng, np_upd["dkappa"] / dgm),
        ("lambda", lambda_ng, np_upd["dlambda"] / dgm),
    ]:
        a = a_t.cpu().numpy()
        assert float(np.max(np.abs(a - np.asarray(b_np).reshape(a.shape)))) < 1e-8, name


@pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing")
def test_multimodel_sufficient_stats_match_numpy_reference():
    """Multi-model (n_models=2) per-block sufficient statistics == NumPy reference.

    The decisive multi-model correctness check. Both backends now compute the
    per-model log-likelihood with the log|det W| + sldet Jacobian (issue #24), so
    the model responsibilities ``v = softmax(logV)`` -- and every v-weighted
    sufficient statistic -- must agree to float64 precision. This is the
    Fortran-free proxy for the (separately verified) machine-precision match of
    one multi-model M-step against the Fortran binary.
    """
    data = _load_real_data()
    ng = AMICATorchNG(
        n_channels=NW,
        n_models=2,
        n_mix=NMIX,
        seed=SEED,
        device="cpu",
        dtype=torch.float64,
        block_size=256,
    )
    X_t = ng._preprocess(data)
    ng._initialize_parameters()
    blk = 256
    block = X_t[:, :blk].contiguous()
    ng_upd = ng._get_block_updates(block)

    npm = _numpy_ref_like(ng, blk, do_newton=False)
    assert (
        ng.A is not None
        and ng.W is not None
        and ng.c is not None
        and ng.mu is not None
        and ng.alpha is not None
        and ng.beta is not None
        and ng.rho is not None
        and ng.gm is not None
    )
    npm.A = ng.A.cpu().numpy().copy()
    npm.W = ng.W.cpu().numpy().copy()
    npm.c = ng.c.cpu().numpy().copy()
    npm.mu = ng.mu.cpu().numpy().copy()
    npm.alpha = ng.alpha.cpu().numpy().copy()
    npm.beta = ng.beta.cpu().numpy().copy()
    npm.rho = ng.rho.cpu().numpy().copy()
    npm.gm = ng.gm.cpu().numpy().copy()
    assert ng.comp_list is not None
    npm.comp_list = ng.comp_list.cpu().numpy()
    np_upd = npm._get_block_updates(block.cpu().numpy())

    keys = [
        "dgm",
        "dalpha_n",
        "dmu_n",
        "dmu_d",
        "dbeta_n",
        "dbeta_d",
        "drho_n",
        "dWtmp",
        "dc_numer",
        "ll",
    ]
    for key in keys:
        a = np.asarray(ng_upd[key].cpu().numpy(), dtype=np.float64)
        b = np.asarray(np_upd[key], dtype=np.float64).reshape(a.shape)
        max_diff = float(np.max(np.abs(a - b)))
        assert max_diff < 1e-8, f"{key} differs from NumPy reference by {max_diff:.3e}"


@pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing")
def test_single_model_bias_c_stays_zero():
    """Regression guard for issue #24 single-model parity: the per-model bias
    ``c`` update (issue #27) is skipped for ``n_models=1``, so ``c`` must stay
    exactly zero after fitting. A nonzero ``c`` here would mean the update leaked
    a float-sum residual of the mean-removed data into the single-model
    trajectory, perturbing the machine-exact Fortran parity."""
    data = _load_real_data()[:, :2048]
    ng = _fresh_ng(block_size=1024)  # n_models=1
    ng.fit(data, max_iter=5, verbose=False)
    assert ng.c is not None
    assert torch.all(ng.c == 0.0), "single-model c must remain exactly zero"


@pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing")
def test_multimodel_bias_c_is_responsibility_weighted_data_mean():
    """The multi-model ``c`` update equals Fortran's ``update_c``: for each model
    ``h``, ``c[:,h] = sum_t v_h(t) x(:,t) / sum_t v_h(t)`` in the sphered-data
    space (amica17.f90:1423-1429/1899-1901). Verifies the one M-step result
    against the responsibility-weighted data mean computed independently from the
    pre-update E-step, on a single block (so there is no accumulation ambiguity).
    """
    data = _load_real_data()[:, :512]
    ng = AMICATorchNG(
        n_channels=NW,
        n_models=2,
        n_mix=NMIX,
        seed=SEED,
        device="cpu",
        dtype=torch.float64,
        block_size=512,
    )
    X_t = ng._preprocess(data)
    ng._initialize_parameters()
    # Model responsibilities from the pre-update E-step (c is still zero here).
    logV, *_ = ng._forward(X_t)
    v = torch.softmax(logV, dim=1)  # (batch, n_models)

    acc = ng._accumulate_blocks(X_t)
    ng._update_parameters(acc, X_t.shape[1])

    assert ng.c is not None
    for h in range(2):
        expected = (X_t * v[:, h]).sum(dim=1) / v[:, h].sum()
        assert torch.allclose(ng.c[:, h], expected, atol=1e-10), (
            f"model {h}: c does not match the responsibility-weighted data mean"
        )
    # The two models must center differently (else the update is a no-op).
    assert not torch.allclose(ng.c[:, 0], ng.c[:, 1], atol=1e-6)


@pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing")
def test_newton_multimodel_finite_and_shaped():
    """Multi-model (n_models=2) Newton runs with correct per-model shapes and
    finite output: per-model finalization (shape (n_channels, n_models), finite,
    no cross-model NaN) and a short two-model Newton fit stays finite for both
    models. (Sufficient-stat NumPy parity is covered by the test above.)
    """
    data = _load_real_data()
    blk = 256
    ng = AMICATorchNG(
        n_channels=NW,
        n_models=2,
        n_mix=NMIX,
        seed=SEED,
        device="cpu",
        dtype=torch.float64,
        block_size=blk,
        do_newton=True,
        newt_start=0,
    )
    X_t = ng._preprocess(data)
    ng._initialize_parameters()
    block = X_t[:, :blk].contiguous()
    ng_upd = ng._get_block_updates(block)
    sigma2, lambda_, kappa = ng._finalize_newton_stats(ng_upd)
    for name, stat in [("sigma2", sigma2), ("lambda", lambda_), ("kappa", kappa)]:
        assert stat.shape == (NW, 2), name
        assert torch.all(torch.isfinite(stat)), name

    ng.fit(data, max_iter=6, verbose=False)
    # The fit must actually complete all 6 iterations (a c-driven NaN would
    # break the loop early via the isfinite-LL stop) and leave a finite bias c
    # for both models -- get_unmixing_matrix returns W, which never touches c,
    # so assert isfinite(ng.c) directly to close that blind spot.
    assert len(ng.ll_history) == 6
    assert ng.c is not None
    assert torch.all(torch.isfinite(ng.c))
    for h in range(2):
        assert np.all(np.isfinite(ng.get_unmixing_matrix(h)))


@pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing")
def test_newton_three_model_finite_and_shaped():
    """3-model (n_models=3) Newton coverage (issue #272).

    The curvature arrays are per-(source, model) and the direction/fallback
    loop iterates over models, so a 2-model fit cannot distinguish a correct
    per-model index from a transposition error that happens to still line up
    for exactly two models. This is the torch counterpart of the numpy #267
    regression (``tests/test_numpy_newton_multimodel.py``), which already
    parametrizes ``num_models`` up to 3, and of the MLX
    ``test_multimodel_newton_mstep_is_finite``/``test_multimodel_newton_fit_completes``
    pair -- extended here to a third model.
    """
    data = _load_real_data()
    blk = 256
    ng = AMICATorchNG(
        n_channels=NW,
        n_models=3,
        n_mix=NMIX,
        seed=SEED,
        device="cpu",
        dtype=torch.float64,
        block_size=blk,
        do_newton=True,
        newt_start=0,
    )
    X_t = ng._preprocess(data)
    ng._initialize_parameters()
    block = X_t[:, :blk].contiguous()
    ng_upd = ng._get_block_updates(block)
    sigma2, lambda_, kappa = ng._finalize_newton_stats(ng_upd)
    for name, stat in [("sigma2", sigma2), ("lambda", lambda_), ("kappa", kappa)]:
        assert stat.shape == (NW, 3), name
        assert torch.all(torch.isfinite(stat)), name

    ng.fit(data, max_iter=8, verbose=False)
    assert len(ng.ll_history) == 8, "the fit did not complete all iterations"
    assert ng.stop_reason not in AMICATorchNG._DEGENERATE_STOP_REASONS
    for name in ("A", "W", "mu", "alpha", "beta", "rho", "gm", "c"):
        value = getattr(ng, name)
        assert value is not None and torch.all(torch.isfinite(value)), name
    for h in range(3):
        assert np.all(np.isfinite(ng.get_unmixing_matrix(h)))


@pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing")
def test_multimodel_bias_c_matches_numpy_reference():
    """Cross-backend parity of the FINALIZED bias c (not just the accumulator):
    NG's ``self.c`` after one M-step equals the NumPy reference's
    ``dc_numer / dgm``. The existing sufficient-stats test proves the numerator
    matches; this closes the accumulator-matches-but-finalize-differs failure
    mode issue #27 already hit once for other statistics."""
    data = _load_real_data()[:, :512]
    ng = AMICATorchNG(
        n_channels=NW,
        n_models=2,
        n_mix=NMIX,
        seed=SEED,
        device="cpu",
        dtype=torch.float64,
        block_size=512,
    )
    X_t = ng._preprocess(data)
    ng._initialize_parameters()
    # NumPy reference carrying NG's pre-update parameters.
    npm = _numpy_ref_like(ng, 512)
    np_upd = npm._get_block_updates(X_t.cpu().numpy())

    acc = ng._get_block_updates(X_t)
    ng._update_parameters(acc, X_t.shape[1])

    expected = np_upd["dc_numer"] / np_upd["dgm"][None, :]
    assert ng.c is not None
    assert np.allclose(ng.c.cpu().numpy(), expected, atol=1e-8)


@pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing")
def test_numpy_multimodel_bias_c_updates():
    """The NumPy backend's own multi-model c update runs on real data: after a
    short 2-model fit c is finite, nonzero, and the two models center
    differently (guards the NumPy _get_block_updates/_update_parameters/transform
    edits, which mirror the NG ones but had no direct multi-model coverage)."""
    data = _load_real_data()[:, :512]
    npm = AMICA_NumPy(
        use_tqdm=False,
        num_models=2,
        num_mix=NMIX,
        seed=SEED,
        block_size=512,
        lrate=0.05,
        max_iter=3,
        do_mean=True,
        do_sphere=True,
    )
    npm.fit(data)
    assert npm.c is not None
    assert np.all(np.isfinite(npm.c))
    assert not np.allclose(npm.c, 0.0)
    assert not np.allclose(npm.c[:, 0], npm.c[:, 1], atol=1e-6)


@pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing")
def test_multimodel_transform_applies_bias_c():
    """transform() unmixes as W(x - c) with a nonzero per-model c (issue #27).
    Every other transform test fits n_models=1 where c == 0, so ``X - c`` always
    subtracts zero; here we verify each model's sources equal the hand-computed
    W[:,:,h].T @ (sphered(x) - c[:,h]), catching a sign flip, a broadcast bug, or
    model_idx selecting the wrong c column."""
    data = _load_real_data()[:, :512]
    ng = AMICATorchNG(
        n_channels=NW,
        n_models=2,
        n_mix=NMIX,
        seed=SEED,
        device="cpu",
        dtype=torch.float64,
        block_size=512,
        do_newton=True,
        newt_start=0,
    )
    ng.fit(data, max_iter=4, verbose=False)
    assert (
        ng.c is not None
        and ng.sphere is not None
        and ng.mean is not None
        and ng.W is not None
    )
    # c must actually be nonzero for this test to mean anything.
    assert not np.allclose(ng.c.cpu().numpy(), 0.0)

    Xs = (ng.sphere @ (torch.from_numpy(data).to(ng.dtype) - ng.mean)).cpu().numpy()
    for h in range(2):
        S = ng.transform(data, model_idx=h)
        W_h = ng.W[:, :, h].cpu().numpy()
        c_h = ng.c[:, h].cpu().numpy()[:, None]
        expected = W_h.T @ (Xs - c_h)
        assert np.allclose(S, expected, atol=1e-9), f"model {h} transform mismatch"


def test_multimodel_dead_model_keeps_prior_c():
    """Containment guard: a model with zero total responsibility (dgm[h]==0)
    keeps its PRIOR bias c instead of writing 0/0 == NaN. A NaN c would poison
    the next iteration's cross-model softmax for every model (unlike
    log(gm[h])=-inf, which softmax tolerates)."""
    ng = AMICATorchNG(
        n_channels=4,
        n_models=2,
        n_mix=NMIX,
        seed=SEED,
        device="cpu",
        dtype=torch.float64,
    )
    # Minimal fitted-shape state so _update_parameters can run one step.
    ng._initialize_parameters()
    prior_c = torch.tensor(
        [[1.0, 7.0], [2.0, 8.0], [3.0, 9.0], [4.0, 10.0]], dtype=torch.float64
    )
    ng.c = prior_c.clone()
    n_mix, n_comps = ng.n_mix, ng.n_comps
    z = torch.zeros
    # Model 1 is dead: dgm[1] == 0 and dc_numer[:,1] == 0 (v_h==0 => both zero).
    acc = {
        "dgm": torch.tensor([100.0, 0.0], dtype=torch.float64),
        "dalpha_n": torch.ones(n_mix, n_comps, dtype=torch.float64),
        "dmu_n": z(n_mix, n_comps, dtype=torch.float64),
        "dmu_d": torch.ones(n_mix, n_comps, dtype=torch.float64),
        "dbeta_n": torch.ones(n_mix, n_comps, dtype=torch.float64),
        "dbeta_d": torch.ones(n_mix, n_comps, dtype=torch.float64),
        "drho_n": z(n_mix, n_comps, dtype=torch.float64),
        "dWtmp": torch.zeros(4, 4, 2, dtype=torch.float64),
        "dc_numer": torch.tensor(
            [[50.0, 0.0], [60.0, 0.0], [70.0, 0.0], [80.0, 0.0]], dtype=torch.float64
        ),
        "ll": torch.tensor(0.0, dtype=torch.float64),
    }
    ng._update_parameters(acc, 100)
    c = ng.c.cpu().numpy()
    assert np.all(np.isfinite(c)), "dead model must not introduce a NaN in c"
    # Live model 0 got the responsibility-weighted mean; dead model 1 kept prior.
    assert np.allclose(c[:, 0], np.array([0.5, 0.6, 0.7, 0.8]))
    assert np.allclose(c[:, 1], prior_c[:, 1].numpy())


def test_rejection_degenerate_ll_raises_clear_error():
    """If the per-sample log-likelihood is non-finite (numerical instability
    upstream, e.g. a diverged fit), rejection raises a clear error naming the
    real cause instead of blaming rejsig (issue #127) -- a single NaN poisons
    mean/std and would otherwise silently empty the good set."""
    m = _fresh_ng(do_reject=True)
    m.good_idx = torch.arange(16)
    m.numrej = 0
    ll = torch.arange(16, dtype=torch.float64)
    ll[7] = float("nan")  # one non-finite entry poisons the reject decision
    with pytest.raises(ValueError, match="non-finite"):
        m._reject_outliers(ll)


def test_reject_param_validation():
    """Invalid rejection parameters are rejected at construction (guarding the
    otherwise-opaque downstream crashes: rejint=0 -> ZeroDivisionError,
    rejsig<=0 -> good set collapses to a ``None`` accumulator)."""
    with pytest.raises(ValueError, match="rejint"):
        _fresh_ng(do_reject=True, rejint=0)
    with pytest.raises(ValueError, match="rejsig"):
        _fresh_ng(do_reject=True, rejsig=0.0)
    with pytest.raises(ValueError, match="rejsig"):
        _fresh_ng(do_reject=True, rejsig=-5.0)


@pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing")
def test_rejection_shrinks_good_sample_set():
    """With ``do_reject`` enabled, low-log-likelihood samples are dropped: the
    good-sample set shrinks, at most ``maxrej`` passes run, and the fit stays
    finite (Fortran-style ``reject_data``)."""
    data = _load_real_data()
    m = _fresh_ng(
        do_reject=True, rejsig=2.0, rejstart=2, rejint=3, maxrej=2, block_size=512
    )
    n_total = data.shape[1]
    m.fit(data, max_iter=12, verbose=False)

    assert m.numrej <= 2
    assert m.numrej >= 1
    assert m.good_idx is not None
    n_good = int(m.good_idx.numel())
    assert n_good < n_total  # some samples rejected
    assert np.all(np.isfinite(m.get_unmixing_matrix(0)))
    assert np.all(np.isfinite(m.ll_history))

    # The reported LL is normalized by the good-sample count, not the original
    # total. Recompute the final iteration's LL from the surviving good set and
    # confirm it divides by n_good (dividing by n_total would be off by
    # n_good/n_total, well outside tolerance here since ~hundreds are dropped).
    X_t = m._preprocess(data)
    acc = m._accumulate_blocks(X_t[:, m.good_idx])
    ll_per_good = float(acc["ll"] / (n_good * m.n_channels))
    ll_per_total = float(acc["ll"] / (n_total * m.n_channels))
    assert abs(m.ll_history[-1] - ll_per_good) < abs(m.ll_history[-1] - ll_per_total)


@pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing")
def test_multimodel_rejection_keeps_bias_c_finite():
    """do_reject + n_models=2: the per-model bias c (accumulated over the good
    set, denominator dgm[h]) must never be left silently non-finite. A shrinking
    good set can drive a model's responsibility mass toward zero, so verify the
    fit either completes with finite c for both models or halts via a
    stop_reason -- it never returns a NaN c while looking normal."""
    data = _load_real_data()
    m = AMICATorchNG(
        n_channels=NW,
        n_models=2,
        n_mix=NMIX,
        seed=SEED,
        device="cpu",
        dtype=torch.float64,
        block_size=512,
        do_reject=True,
        rejsig=2.0,
        rejstart=2,
        rejint=3,
        maxrej=2,
    )
    m.fit(data, max_iter=12, verbose=False)
    if m.stop_reason in AMICATorchNG._DEGENERATE_STOP_REASONS:
        # A degenerate stop is acceptable (surfaced, not silent); nothing more
        # to assert -- the point is it did not silently return a NaN model.
        return
    assert m.c is not None
    assert torch.all(torch.isfinite(m.c)), "bias c left non-finite without a stop"
    assert np.all(np.isfinite(m.ll_history))


def test_state_dict_requires_fit():
    """A fresh (unfitted) backend cannot be serialized: state_dict() raises
    rather than emitting a half-formed payload (no silent-failure)."""
    m = _fresh_ng()
    with pytest.raises(RuntimeError, match="fitted"):
        m.state_dict()


@pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing")
def test_state_dict_roundtrip_all_fields():
    """Backend-level round-trip asserting EVERY persisted field, with Newton and
    outlier rejection active so the mixture-PDF params, ``good_idx``, and the
    mutated optimizer state are all non-trivial (issue #36).

    The wrapper round-trip only exercises A/W/c/mean/sphere via transform(); this
    guards the fields nothing downstream in the suite reads back: mu/alpha/beta/
    rho/gm (the mixture PDF) and the do_reject ``good_idx`` tensor branch.
    """
    data = _load_real_data()
    m = _fresh_ng(
        do_reject=True,
        rejsig=2.0,
        rejstart=2,
        rejint=3,
        maxrej=2,
        block_size=512,
        do_newton=True,
        newt_start=2,
    )
    m.fit(data, max_iter=12, verbose=False)

    # Preconditions: the interesting state is actually populated.
    assert m.good_idx is not None and int(m.good_idx.numel()) < data.shape[1]
    assert m.iteration > 0

    state = m.state_dict()
    loaded = AMICATorchNG.from_state_dict(state, device="cpu")

    assert loaded.dtype == m.dtype
    for name in AMICATorchNG._PARAM_TENSORS:
        orig, new = getattr(m, name), getattr(loaded, name)
        assert new.dtype == orig.dtype, name  # comp_list stays integer
        assert torch.equal(new.cpu(), orig.cpu()), name

    assert loaded.good_idx is not None
    assert torch.equal(loaded.good_idx.cpu(), m.good_idx.cpu())

    for attr in (
        "sldet",
        "iteration",
        "stop_reason",
        "n_newton_fallbacks",
        "numrej",
        "lrate",
        "lrate_cap",
        "newtrate",
        "rholrate",
        "final_ll_",  # issue #51: the returned iterate's LL survives a round-trip
        "keep_best",
    ):
        assert getattr(loaded, attr) == getattr(m, attr), attr
    assert loaded.ll_history == m.ll_history


@pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing")
def test_state_dict_refuses_degenerate_model():
    """A fit that ended on a non-finite log-likelihood (stop_reason nan_ll/
    singular_ll) must not be serialized: state_dict() raises rather than let a
    NaN model round-trip silently and surface as all-NaN sources later."""
    data = _load_real_data()
    m = _fresh_ng(block_size=512)
    m.fit(data[:, :2048], max_iter=2, verbose=False)

    # A real divergence sets this marker (and leaves NaNs in the params); assert
    # the guard fires on the marker deterministically, without needing to induce
    # an actual blow-up.
    m.stop_reason = "nan_ll"
    with pytest.raises(RuntimeError, match="degenerate"):
        m.state_dict()


@pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing")
def test_state_dict_refuses_nonfinite_params():
    """Defense-in-depth: even with a non-degenerate stop_reason, a non-finite
    parameter tensor blocks serialization."""
    data = _load_real_data()
    m = _fresh_ng(block_size=512)
    m.fit(data[:, :2048], max_iter=2, verbose=False)

    m.stop_reason = "max_iter"  # not a degenerate marker
    assert m.A is not None
    m.A[0, 0] = float("nan")
    with pytest.raises(RuntimeError, match="non-finite"):
        m.state_dict()


@pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing")
def test_state_dict_snapshots_not_aliases():
    """state_dict() must snapshot params, not alias live tensors: fit() mutates
    A/mu/beta in place, so an aliased CPU snapshot would silently roll forward
    if captured mid-fit. Mutating the model after capture must not touch it."""
    data = _load_real_data()
    m = _fresh_ng(block_size=512)
    m.fit(data[:, :2048], max_iter=2, verbose=False)

    state = m.state_dict()
    captured = state["params"]["A"].clone()
    assert m.A is not None
    m.A[0, 0] += 1.0  # the same in-place mutation fit() does each iteration
    assert torch.equal(state["params"]["A"], captured)


@pytest.mark.slow
@pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing")
def test_end_to_end_correlation_vs_fortran():
    """Epic definition-of-done: Hungarian-matched component correlation vs the
    Fortran binary > 0.95 on the sample data.

    Passes since issue #24: the natural-gradient A-update transpose fix (plus the
    exact-EM mixture updates, the digamma rho update, and the symmetric-ZCA
    sphere) makes the backend ascend to Fortran's solution (LL ~ -3.40) with
    Newton positive-definite and firing (``m.n_newton_fallbacks == 0``), reaching
    component correlation ~0.997 and Amari distance ~0.006 (issue #116).
    """
    import sys

    root = Path(__file__).resolve().parents[2].parent
    sys.path.insert(0, str(root))
    from validate_implementations import (
        load_sample_data,
        run_fortran_amica,
        compare_results,
        amari_distance,
    )

    data, params = load_sample_data()
    params = dict(params)
    params["max_iter"] = 100
    out = root / "pamica" / "tests" / "torch_tests" / "_ng_e2e_tmp"
    out.mkdir(parents=True, exist_ok=True)
    fortran = run_fortran_amica(data, params, out, SEED)
    assert fortran is not None, "Fortran binary run failed"

    m = _fresh_ng(
        block_size=512,
        do_newton=True,
        newt_start=50,
        newtrate=1.0,
        lrate=0.05,
        invsigmin=0.0,
        invsigmax=100.0,
    )
    m.fit(data.astype(np.float64), max_iter=100, verbose=False)
    ng_results = {
        "final_ll": m.ll_history[-1],
        "final_iter": len(m.ll_history),
        "W": m.get_unmixing_matrix(0),
        "A": m.get_mixing_matrix(0),
    }
    cmp = compare_results(fortran, ng_results)
    assert cmp["mean_correlation"] > 0.95
    assert amari_distance(fortran["W"], ng_results["W"]) < 0.05
    # The docstring's headline claim: Newton is positive-definite and actually
    # firing, not silently falling back to natural gradient every iteration.
    assert m.n_newton_fallbacks == 0


@pytest.mark.slow
@pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing")
def test_end_to_end_correlation_vs_fortran_from_sample_params_json():
    """Same correctness bar as ``test_end_to_end_correlation_vs_fortran``, but
    built the way a user actually reproduces the paper's Table 1 numbers: via
    ``run_pytorch_amica``, which maps ``sample_data/sample_params.json`` onto
    ``AMICATorchNG`` kwargs (the documented ``validate_implementations.py``
    path), not via hand-picked kwargs.

    This is the regression guard the hand-picked-kwargs test above cannot be:
    a bad JSON setting with an ``AMICATorchNG`` equivalent (e.g.
    ``do_approx_sphere``) silently overrides that constructor's safe default
    when loaded through this path, but never touches ``_fresh_ng``'s explicit
    kwargs above, so it can regress here while the other test stays green.

    The end-to-end correlation/Amari checks below are too loose on their own
    to reliably catch this at a practical iteration budget: a reintroduced
    ``do_approx_sphere: false`` still clears both thresholds at ``max_iter``
    below a few hundred (issue #144 PR review measured 0.9986 correlation /
    0.0049 Amari at 100 iterations, only marginally failing by 500). The
    explicit ``do_approx_sphere`` assertion below is the actual regression
    guard -- deterministic, independent of optimization dynamics -- and the
    fit-based checks stay as a broader "does the JSON-loaded pipeline work"
    sanity check.
    """
    import sys

    root = Path(__file__).resolve().parents[2].parent
    sys.path.insert(0, str(root))
    from validate_implementations import (
        _NG_PARAMS,
        load_sample_data,
        run_fortran_amica,
        run_pytorch_amica,
        compare_results,
        amari_distance,
    )

    data, params = load_sample_data()
    params = dict(params)
    params["max_iter"] = 300

    assert "do_approx_sphere" in _NG_PARAMS, (
        "AMICATorchNG must accept do_approx_sphere for this guard to mean "
        "anything -- update this test if the constructor signature changed"
    )
    assert params.get("do_approx_sphere") is True, (
        "sample_params.json's do_approx_sphere maps straight through to "
        "AMICATorchNG's constructor kwarg of the same name (run_pytorch_amica "
        "passes through any params.json key that is in _NG_PARAMS unchanged); "
        "False here silently breaks Fortran parity (non-symmetric PCA "
        "whitening instead of symmetric ZCA sphering)"
    )

    out = root / "pamica" / "tests" / "torch_tests" / "_ng_e2e_json_tmp"
    out.mkdir(parents=True, exist_ok=True)
    fortran = run_fortran_amica(data, params, out, SEED)
    assert fortran is not None, "Fortran binary run failed"

    ng_results = run_pytorch_amica(data, params, out, SEED)
    cmp = compare_results(fortran, ng_results)
    assert cmp["mean_correlation"] > 0.95
    assert amari_distance(fortran["W"], ng_results["W"]) < 0.05


@pytest.mark.slow
@pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing")
def test_full_fit_parity_numpy_vs_ng(tmp_path):
    """Full-fit parity: independently fitting AMICA_NumPy and AMICATorchNG on
    the real sample data (single model, matched config) converges to the same
    solution. Locks the NumPy<->NG relationship end-to-end, on top of the
    per-block sufficient-stat tests that already pin the update math to 1e-8.

    Measured on the sample data (issue #37): Hungarian-matched total-spatial-
    filter correlation is 1.000000 (32/32 bijective) and the per-sample-per-
    channel LL agrees to ~1e-5. Both backends report the per-sample-per-channel
    normalized LL (NumPy since issue #212, matching NG's own (n_samples *
    n_channels) normalization), so the two are compared directly with no
    rescaling. A full fit is not bit-identical across NumPy and PyTorch (float
    reduction order / BLAS vs ATen), hence tolerances rather than exact
    equality.
    """
    from scipy.optimize import linear_sum_assignment

    data = _load_real_data()  # (NW, FIELD) float64

    cfg: dict[str, Any] = dict(
        block_size=512,
        lrate=0.05,
        lratefact=0.5,
        do_newton=True,
        newt_start=50,
        newtrate=1.0,
        rho0=1.5,
        minrho=1.0,
        maxrho=2.0,
        rholrate=0.05,
        invsigmin=0.0,
        invsigmax=100.0,
        doscaling=True,
        do_mean=True,
        do_sphere=True,
    )

    npm = AMICA_NumPy(
        use_tqdm=False,
        writestep=10000,
        num_models=1,
        num_mix=NMIX,
        seed=SEED,
        max_decs=5,
        max_iter=150,
        # AMICA_NumPy defaults do_opt_block=True, which auto-retunes block_size
        # from data-timing at fit(), overriding the shared block_size=512 (NG
        # has no such auto-tune). Disable it so both backends genuinely run the
        # matched block size.
        do_opt_block=False,
        outdir=str(tmp_path / "np_out"),
        **cfg,
    )
    npm.fit(data)
    filt_np = npm.get_weights() @ npm.sphere
    ll_np = npm.ll[-1]  # already per-sample-per-channel normalized (issue #212)

    ng = _fresh_ng(maxdecs=5, **cfg)
    ng.fit(data, max_iter=150, verbose=False)
    assert ng.sphere is not None
    filt_ng = ng.get_unmixing_matrix(0) @ ng.sphere.cpu().numpy()
    ll_ng = ng.ll_history[-1]

    a = filt_np / np.linalg.norm(filt_np, axis=1, keepdims=True)
    b = filt_ng / np.linalg.norm(filt_ng, axis=1, keepdims=True)
    corr = np.abs(a @ b.T)
    # linear_sum_assignment on a square matrix returns a full permutation, so
    # `matched` is already a bijective NumPy<->NG pairing; the correlation gate
    # below is the real check.
    row, col = linear_sum_assignment(-corr)
    matched = corr[row, col]

    assert matched.min() > 0.99, (
        f"min NumPy<->NG matched filter correlation {matched.min():.4f} <= 0.99"
    )
    assert abs(ll_np - ll_ng) < 1e-2, (
        f"normalized LL disagreement {abs(ll_np - ll_ng):.3e} (np={ll_np:.6f}, "
        f"ng={ll_ng:.6f})"
    )


@pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing")
def test_keep_best_single_model_is_bit_exact():
    """The best-iterate safeguard (issue #51) is a no-op for a monotone
    single-model fit: the final iterate already is the best, so no restore fires
    and the returned parameters are byte-for-byte identical to keep_best=False.
    This guards single-model issue #24 parity against the safeguard."""
    data = _load_real_data()
    common: dict[str, Any] = dict(
        block_size=512, lrate=0.05, do_newton=True, newt_start=50
    )
    on = _fresh_ng(keep_best=True, **common)
    on.fit(data, max_iter=100, verbose=False)
    off = _fresh_ng(keep_best=False, **common)
    off.fit(data, max_iter=100, verbose=False)

    for name in AMICATorchNG._PARAM_TENSORS:
        assert torch.equal(getattr(on, name), getattr(off, name)), name
    # No restore fired either way: both report the last iterate's LL.
    assert on.final_ll_ == on.ll_history[-1]
    assert off.final_ll_ == off.ll_history[-1]


@pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing")
def test_keep_best_snapshot_restore_roundtrip():
    """The primitives behind the safeguard (issue #51): ``_snapshot_params``
    clones (does not alias) the live tensors, and ``_restore_params`` rolls an
    in-place mutation back exactly."""
    m = _fresh_ng(block_size=512)
    m.fit(_load_real_data()[:, :2048], max_iter=3, verbose=False)

    snap = m._snapshot_params()
    assert m.A is not None
    orig = float(m.A[0, 0])
    m.A[0, 0] = orig + 5.0  # the same kind of in-place edit fit() does each step
    snap_a = snap["A"]
    assert isinstance(snap_a, torch.Tensor)
    assert float(snap_a[0, 0]) == orig  # snapshot is a clone, not an alias
    m._restore_params(snap)
    assert m.A is not None
    assert float(m.A[0, 0]) == orig  # restore reverts the mutation
    for name in AMICATorchNG._PARAM_TENSORS:
        snap_val = snap[name]
        assert isinstance(snap_val, torch.Tensor)
        assert torch.equal(getattr(m, name), snap_val), name


def _multimodel_keep_best(seed: int, keep_best: bool) -> AMICATorchNG:
    m = AMICATorchNG(
        n_channels=NW, n_models=2, n_mix=NMIX, seed=seed, device="cpu",
        dtype=torch.float64, block_size=512, lrate=0.05, maxdecs=3,
        do_newton=True, newt_start=50, newt_ramp=10, newtrate=1.0,
        keep_best=keep_best,
    )  # fmt: skip
    m.fit(_load_real_data(), max_iter=100, verbose=False)
    return m


@pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing")
def test_keep_best_returns_within_tol_of_peak():
    """On a real multi-model fit the safeguard (issue #51) returns a model whose
    log-likelihood is within tolerance of the best iterate seen and never below
    the last iterate, and ``final_ll_`` reflects the *returned* parameters (not
    the raw ``ll_history[-1]``, which stays the true trajectory). keep_best does
    not change the optimization path, only which iterate is returned, so the
    ``keep_best=False`` run has the same trajectory but returns the (lower) last
    iterate. seed 8 reaches the plateau where natural-gradient AMICA dips below
    its own peak, so the restore branch runs; the invariants also hold if a
    platform's BLAS makes the run monotone (see the explicit skip below)."""
    on = _multimodel_keep_best(8, keep_best=True)
    off = _multimodel_keep_best(8, keep_best=False)

    # keep_best does not alter the trajectory, only the returned iterate.
    assert on.ll_history == off.ll_history
    assert on.final_ll_ is not None and off.final_ll_ is not None
    # return-last returns exactly the last iterate; keep_best is never worse.
    assert off.final_ll_ == off.ll_history[-1]
    assert on.final_ll_ >= off.final_ll_

    peak = max(on.ll_history)
    # The returned LL is within tolerance of the peak and never below the last.
    assert abs(on.final_ll_ - peak) <= _KEEP_BEST_TOL
    assert on.final_ll_ >= on.ll_history[-1]
    # The returned parameters really sit at final_ll_ (recompute the E-step LL).
    data = _load_real_data()
    X_t = on._preprocess(data)
    acc = on._accumulate_blocks(X_t)
    ll_model = float(acc["ll"] / (X_t.shape[1] * NW))
    assert abs(ll_model - on.final_ll_) < 1e-9

    # Make branch coverage visible rather than silently vacuous: if this run did
    # not overshoot on this platform, the restore branch was not exercised.
    if peak - on.ll_history[-1] <= _KEEP_BEST_TOL:
        pytest.skip("seed 8 did not overshoot here; restore branch not exercised")
    # It did overshoot, so keep_best strictly beat return-last.
    assert on.final_ll_ > off.final_ll_


@pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing")
def test_keep_best_inactive_under_reject():
    """The safeguard is disabled under ``do_reject`` (the good-sample set, hence
    the LL normalization, changes across iterations, so per-iteration LLs are not
    comparable): ``final_ll_`` is exactly the last trajectory value, no restore
    fires (issue #51)."""
    data = _load_real_data()
    m = AMICATorchNG(
        n_channels=NW, n_models=2, n_mix=NMIX, seed=SEED, device="cpu",
        dtype=torch.float64, block_size=512, do_reject=True, rejsig=2.0,
        rejstart=2, rejint=3, maxrej=2, keep_best=True,
    )  # fmt: skip
    m.fit(data, max_iter=12, verbose=False)
    assert m.numrej >= 1  # rejection actually fired, so the good set changed
    assert m.final_ll_ == m.ll_history[-1]  # no best-iterate restore under reject


@pytest.mark.slow
@pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing")
def test_rholrate_ratchets_at_maxdecs_not_per_decrease():
    """Issue #193: the rho learning rate is a maxdecs-ratcheted *ceiling*, not a
    per-LL-decrease monotone decay.

    Fortran resets ``rholrate = rholrate0`` every iteration before the rho update
    (amica15.f90:1806/1813) and only tightens the ceiling at ``maxdecs``
    (amica15.f90:1068, gated on ``iter > newt_start``). torch previously decayed
    ``rholrate`` on EVERY LL decrease with no reset, collapsing it to ~1e-5 within
    a few hundred iterations and freezing rho at a stale shape.

    On the real sample data a long-enough Newton run overshoots and triggers
    several LL decreases. The surviving ``rholrate`` must have ratcheted exactly
    as often as ``newtrate`` (both gated on ``iter > newt_start`` at ``maxdecs``,
    matched 0.5 factor here), NOT once per decrease.
    """
    data = _load_real_data()
    m = _fresh_ng(
        block_size=512, do_newton=True, newt_start=50, newtrate=1.0, lrate=0.05,
        lratefact=0.5, rholrate=0.05, rholratefact=0.5, maxdecs=3,
    )  # fmt: skip
    m.fit(data, max_iter=300, verbose=False)

    ll = m.ll_history
    n_dec = sum(1 for i in range(1, len(ll)) if ll[i] < ll[i - 1])
    # The run must actually exercise the schedule (Newton overshoots past
    # newt_start), or the assertions below prove nothing.
    assert m.newtrate < m.newtrate0, "no newtrate ratchet fired; budget too short"
    assert n_dec > m.maxdecs, "too few LL decreases to distinguish the schedules"

    # rholrate and newtrate share the maxdecs ratchet schedule, so the surviving
    # rho ceiling ratcheted the same number of times as newtrate.
    assert m.rholrate == pytest.approx(m.rholrate0 * (m.newtrate / m.newtrate0))
    # The old per-decrease decay (rholrate0 * rholratefact**n_dec) sits orders of
    # magnitude below the fixed ceiling; guard against a regression to it.
    buggy = m.rholrate0 * (m.rholratefact**n_dec)
    assert m.rholrate > buggy * 10
