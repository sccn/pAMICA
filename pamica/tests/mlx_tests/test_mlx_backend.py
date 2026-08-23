"""MLX backend (AMICAMLXNG) tests -- issue #76, epic #74 Phase C.

Apple-Silicon only. Real sample EEG (no synthetic/mock). The whole module
self-skips when MLX or an Apple GPU is unavailable, so CI on ubuntu skips it.

Correctness is checked two ways: the per-block sufficient statistics match the
validated NumPy float64 reference (tight, independent oracle), and a full fit's
converged log-likelihood matches the PyTorch float32 backend (the epic's
backend-vs-backend acceptance).
"""

from pathlib import Path

from typing import Any

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "sample_data"
DATA_FILE = SAMPLE_DIR / "eeglab_data.fdt"
NW = 32
FIELD = 30504
NMIX = 3
SEED = 42

pytestmark = [
    pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing"),
    pytest.mark.skipif(
        mx.default_device().type != mx.DeviceType.gpu, reason="no Apple GPU"
    ),
]

# Sufficient-stat keys the MLX backend produces (single-model, non-Newton).
_STAT_KEYS = [
    "dgm",
    "dalpha_n",
    "dmu_n",
    "dmu_d",
    "dbeta_n",
    "dbeta_d",
    "drho_n",
    "dWtmp",
]
# The three Newton curvature accumulators (issue #264) are emitted only under
# do_newton, and cannot join _STAT_KEYS: the NumPy reference accumulates them
# already summed over the mixture index with the mu^2 term folded into lambda,
# whereas MLX (like PyTorch) keeps the mixture axis and folds mu^2 in at
# finalization. They line up at the FINALIZED level -- which is also where a
# broadcast slip would show -- so test_newton_stats_match_numpy_reference
# compares _finalize_newton_stats against dsigma2/dkappa/dlambda over dgm, the
# same construction torch_tests/test_ng_backend.py uses.
_NEWTON_STAT_KEYS = ("dsigma2", "dkappa", "dlambda")


def _load_real_data() -> np.ndarray:
    from pamica.torch_impl.utils import load_eeglab_data

    return load_eeglab_data(str(DATA_FILE), data_dim=NW, field_dim=FIELD).astype(
        np.float64
    )


def test_rejects_unsupported_config():
    """The still-deferred configs fail loudly, not silently. Multi-model (issue
    #81), component sharing (#263) and Newton (#264) ARE supported, so only the
    non-GG PDF families are rejected now."""
    from pamica.mlx_impl import AMICAMLXNG

    kwarg_sets: list[dict[str, Any]] = [
        {"pdftype": 2, "n_mix": NMIX},
        {"pdftype": 1, "n_mix": NMIX},
    ]
    for kw in kwarg_sets:
        with pytest.raises(NotImplementedError):
            AMICAMLXNG(n_channels=NW, n_mix=kw.pop("n_mix", 1), **kw)

    # Newton is accepted and honored -- not silently downgraded to the natural
    # gradient, which is how the pre-#264 backend would have "supported" it.
    assert AMICAMLXNG(n_channels=NW, n_mix=NMIX, do_newton=True).do_newton is True


def test_sufficient_stats_match_numpy_reference():
    """Tight algorithmic faithfulness: the MLX (float32) per-block sufficient
    statistics match the validated NumPy float64 reference on an identical block
    with identical (shared-seed) parameters. rtol=1e-3 accommodates float32
    vs float64 (measured max relerr ~1.6e-4 on dWtmp/dmu_d)."""
    from pamica.mlx_impl import AMICAMLXNG
    from pamica.numpy_impl.core import AMICA as AMICA_NumPy

    data = _load_real_data()
    m = AMICAMLXNG(n_channels=NW, n_mix=NMIX, seed=SEED, block_size=256)
    x_t = m._preprocess(data)
    m._initialize_parameters()

    blk = 256
    block = np.array(x_t[:, :blk])  # float32 sphered block, on host
    mlx_upd = m._get_block_updates(mx.array(block))

    npm = AMICA_NumPy(num_models=1, num_mix=NMIX, do_newton=False)
    npm.data_dim = NW
    npm.num_comps = NW
    npm.num_models = 1
    npm.num_mix = NMIX
    npm.block_size = blk
    npm.comp_list = np.arange(NW).reshape(NW, 1)
    npm.A = np.array(m.A, dtype=np.float64)
    # MLX W is (n_models, n, n); NumPy expects (n, n, n_models).
    npm.W = np.array(m.W, dtype=np.float64).transpose(1, 2, 0)
    npm.c = np.zeros((NW, 1))
    npm.mu = np.array(m.mu, dtype=np.float64)
    npm.alpha = np.array(m.alpha, dtype=np.float64)
    npm.beta = np.array(m.beta, dtype=np.float64)
    npm.rho = np.array(m.rho, dtype=np.float64)
    npm.gm = np.array(m.gm, dtype=np.float64)
    np_upd = npm._get_block_updates(block.astype(np.float64))

    for key in _STAT_KEYS:
        a = np.asarray(np.array(mlx_upd[key]), dtype=np.float64)
        b = np.asarray(np_upd[key], dtype=np.float64).reshape(a.shape)
        assert np.allclose(a, b, rtol=1e-3, atol=1e-4), (
            f"{key} differs from NumPy reference by {np.max(np.abs(a - b)):.3e}"
        )


def test_newton_stats_match_numpy_reference():
    """The Newton curvature (issue #264) matches the validated NumPy float64
    reference on an identical block with identical (shared-seed) parameters.

    Compared after ``_finalize_newton_stats`` rather than accumulator-by-
    accumulator, because the two backends split the work differently (see
    ``_NEWTON_STAT_KEYS``) -- and because finalization is where the model-mass
    broadcast lives, which is the step that was wrong in the NumPy backend until
    issue #267. rtol=1e-3 accommodates float32 vs float64, as above.
    """
    from pamica.mlx_impl import AMICAMLXNG
    from pamica.numpy_impl.core import AMICA as AMICA_NumPy

    data = _load_real_data()
    m = AMICAMLXNG(n_channels=NW, n_mix=NMIX, seed=SEED, block_size=256, do_newton=True)
    x_t = m._preprocess(data)
    m._initialize_parameters()

    blk = 256
    block = np.array(x_t[:, :blk])
    mlx_upd = m._get_block_updates(mx.array(block))
    finalized = dict(
        zip(("dsigma2", "dlambda", "dkappa"), m._finalize_newton_stats(mlx_upd))
    )

    npm = AMICA_NumPy(num_models=1, num_mix=NMIX, do_newton=True)
    npm.data_dim = NW
    npm.num_comps = NW
    npm.num_models = 1
    npm.num_mix = NMIX
    npm.block_size = blk
    npm.comp_list = np.arange(NW).reshape(NW, 1)
    npm.A = np.array(m.A, dtype=np.float64)
    npm.W = np.array(m.W, dtype=np.float64).transpose(1, 2, 0)
    npm.c = np.zeros((NW, 1))
    npm.mu = np.array(m.mu, dtype=np.float64)
    npm.alpha = np.array(m.alpha, dtype=np.float64)
    npm.beta = np.array(m.beta, dtype=np.float64)
    npm.rho = np.array(m.rho, dtype=np.float64)
    npm.gm = np.array(m.gm, dtype=np.float64)
    np_upd = npm._get_block_updates(block.astype(np.float64))

    for key in _NEWTON_STAT_KEYS:
        a = np.array(finalized[key], dtype=np.float64)  # (n_models, n_ch)
        b = (np_upd[key] / np_upd["dgm"][None, :]).T  # (num_models, data_dim)
        assert np.allclose(a, b, rtol=1e-3, atol=1e-4), (
            f"{key} differs from NumPy reference by {np.max(np.abs(a - b)):.3e}"
        )
        assert np.all(a > 0), f"{key} is not strictly positive"


def test_backend_stable_on_full_data():
    """The MLX backend fits the full 30504-sample EEG on the Apple GPU (float32)
    without diverging (the Phase A ``ufp/y`` guard is carried into MLX), and the
    log-likelihood improves over the run."""
    from pamica.mlx_impl import AMICAMLXNG

    m = AMICAMLXNG(n_channels=NW, n_mix=NMIX, seed=SEED)
    m.fit(_load_real_data(), max_iter=100, verbose=False)

    hist = np.asarray(m.ll_history, dtype=float)
    assert np.all(np.isfinite(hist))
    assert m.final_ll_ is not None
    assert np.isfinite(m.final_ll_)
    assert np.all(np.isfinite(np.array(m.A)))
    assert m.stop_reason not in AMICAMLXNG._DEGENERATE_STOP_REASONS
    assert hist[-1] > hist[0]  # ascent
    # Single-model must never touch the bias c (the n_models>1-only update); a
    # nonzero c would signal the #24 bit-exact single-model path was perturbed.
    assert np.all(np.array(m.c) == 0.0)


def test_converged_ll_matches_torch_float32():
    """Epic acceptance: the MLX backend is equivalent to the PyTorch float32
    backend. At matched settings/seed both reach the same natural-gradient GG
    fixed point; the converged per-sample-per-channel LL agrees within 1e-2
    (measured gap ~2e-6). Bit-identity is impossible (GPU-vs-CPU float32 +
    reduction order), so the tolerance is relational, never a hardcoded LL."""
    import torch

    from pamica.mlx_impl import AMICAMLXNG
    from pamica.torch_impl import AMICATorchNG

    data = _load_real_data()
    mlx_m = AMICAMLXNG(n_channels=NW, n_mix=NMIX, seed=SEED)
    mlx_m.fit(data, max_iter=100, verbose=False)

    torch_m = AMICATorchNG(
        n_channels=NW,
        n_models=1,
        n_mix=NMIX,
        seed=SEED,
        device="cpu",
        dtype=torch.float32,
        do_newton=False,
    )
    torch_m.fit(data, max_iter=100, verbose=False)

    assert mlx_m.final_ll_ is not None and torch_m.final_ll_ is not None
    assert np.isfinite(mlx_m.final_ll_)
    assert abs(mlx_m.final_ll_ - torch_m.final_ll_) < 1e-2


def test_multimodel_matches_torch_float32():
    """Multi-model (n_models=2) MLX matches the PyTorch float32 backend (issue
    #81): the one-iteration sufficient stats agree to float32 precision, and the
    converged LL matches (measured gap ~1e-5). Single-model stays byte-identical
    (covered by the tests above)."""
    import torch

    from pamica.mlx_impl import AMICAMLXNG
    from pamica.torch_impl import AMICATorchNG

    data = _load_real_data()
    m = 2

    # One iteration of sufficient stats from identical (shared-seed) init.
    mlx_m = AMICAMLXNG(n_channels=NW, n_models=m, n_mix=NMIX, seed=SEED, block_size=512)
    xt = mlx_m._preprocess(data)
    mlx_m._initialize_parameters()
    ma = mlx_m._accumulate_blocks(xt)

    ng = AMICATorchNG(
        n_channels=NW,
        n_models=m,
        n_mix=NMIX,
        seed=SEED,
        device="cpu",
        dtype=torch.float32,
        do_newton=False,
        block_size=512,
    )
    xt2 = ng._preprocess(data)
    ng._initialize_parameters()
    na = ng._accumulate_blocks(xt2)
    # Every accumulator, including dWtmp (input to the gm-weighted comp_list A
    # scatter) and the scattered mixture stats. rtol=5e-3 (looser than the 1e-3
    # single-model-vs-numpy check) because this compares two float32 backends on
    # different devices, whose reduction orders diverge more than float32-vs-f64.
    for key in (
        "dgm",
        "dalpha_n",
        "dmu_n",
        "dmu_d",
        "dbeta_d",
        "drho_n",
        "dc_numer",
        "dWtmp",
    ):
        a = np.asarray(np.array(ma[key]), dtype=np.float64)
        b = na[key].cpu().numpy().astype(np.float64)
        if key == "dWtmp":  # torch (n, n, n_models) -> MLX (n_models, n, n)
            b = np.transpose(b, (2, 0, 1))
        assert np.allclose(a, b.reshape(a.shape), rtol=5e-3, atol=1e-4), (
            f"{key}: {np.abs(a - b.reshape(a.shape)).max():.2e}"
        )

    # The per-model bias c update (the n_models>1-only formula) is the
    # responsibility-weighted data mean c[:,h] = sum_t v_h*x / sum_t v_h, and the
    # two models' c must differ (a would-be no-op would leave them equal at 0).
    mlx_m._update_parameters(ma, xt.shape[1])
    c = np.array(mlx_m.c)
    dc = np.array(ma["dc_numer"]) / np.array(ma["dgm"])[None, :]
    assert np.allclose(c, dc, rtol=1e-5, atol=1e-6)
    assert not np.allclose(c[:, 0], c[:, 1])

    # Converged LL (multi-model is not partition-identifiable, but at matched
    # init/settings both backends track the same trajectory).
    mlx_c = AMICAMLXNG(n_channels=NW, n_models=m, n_mix=NMIX, seed=SEED)
    mlx_c.fit(data, max_iter=60, verbose=False)
    ng_c = AMICATorchNG(
        n_channels=NW,
        n_models=m,
        n_mix=NMIX,
        seed=SEED,
        device="cpu",
        dtype=torch.float32,
        do_newton=False,
        keep_best=False,
    )
    ng_c.fit(data, max_iter=60, verbose=False)
    assert mlx_c.final_ll_ is not None and ng_c.final_ll_ is not None
    assert np.isfinite(mlx_c.final_ll_)
    assert mlx_c.stop_reason not in AMICAMLXNG._DEGENERATE_STOP_REASONS
    assert abs(mlx_c.final_ll_ - ng_c.final_ll_) < 1e-2


def test_degenerate_fit_stops_and_reports_nan():
    """A NaN in the data drives fit() to a degenerate stop with a NaN final_ll,
    exercising the MLX degenerate-stop machinery (the same path the new
    ``nan_params`` guard uses) rather than only the happy path. Sphering/mean are
    off so the NaN reaches the E-step (numpy eigh would otherwise raise on it)."""
    from pamica.mlx_impl import AMICAMLXNG

    data = _load_real_data()[:, :4096].copy()
    data[0, 0] = np.nan
    m = AMICAMLXNG(n_channels=NW, n_mix=NMIX, seed=SEED, do_sphere=False, do_mean=False)
    m.fit(data, max_iter=5, verbose=False)
    assert m.stop_reason in AMICAMLXNG._DEGENERATE_STOP_REASONS
    assert m.final_ll_ is not None
    assert np.isnan(m.final_ll_)


def test_init_matches_torch_seed():
    """Pin the shared-seed init contract directly: AMICAMLXNG and AMICATorchNG
    produce the same starting A/mu/alpha/beta/rho (same RNG draw order), for both
    single- and multi-model -- rather than inferring it from downstream LL."""
    import torch

    from pamica.mlx_impl import AMICAMLXNG
    from pamica.torch_impl import AMICATorchNG

    data = _load_real_data()
    for nm in (1, 2):
        mlx_m = AMICAMLXNG(n_channels=NW, n_models=nm, n_mix=NMIX, seed=SEED)
        mlx_m._preprocess(data)
        mlx_m._initialize_parameters()
        ng = AMICATorchNG(
            n_channels=NW,
            n_models=nm,
            n_mix=NMIX,
            seed=SEED,
            device="cpu",
            dtype=torch.float32,
        )
        ng._preprocess(data)
        ng._initialize_parameters()
        for attr in ("A", "mu", "alpha", "beta", "rho"):
            a = np.array(getattr(mlx_m, attr))
            b = getattr(ng, attr).cpu().numpy()
            assert np.allclose(a, b, atol=1e-6), f"{attr} init differs (n_models={nm})"


def test_multimodel_dead_model_keeps_prior_c():
    """A zero-responsibility model (dgm[h]==0) keeps its prior bias c rather than
    writing 0/0 (the dead-model guard), mirroring AMICATorchNG."""
    from pamica.mlx_impl import AMICAMLXNG

    data = _load_real_data()
    m = AMICAMLXNG(n_channels=NW, n_models=2, n_mix=NMIX, seed=SEED)
    xt = m._preprocess(data)
    m._initialize_parameters()
    acc = m._accumulate_blocks(xt)
    # Kill model 1's responsibility mass and give it a marker prior c.
    dgm = np.array(acc["dgm"], dtype=np.float32)
    dgm[1] = 0.0
    acc["dgm"] = mx.array(dgm)
    marker = np.arange(NW, dtype=np.float32) + 1.0
    m.c = mx.array(np.stack([np.zeros(NW, np.float32), marker], axis=1))
    m._update_parameters(acc, xt.shape[1])
    assert np.allclose(np.array(m.c)[:, 1], marker)  # dead model kept its prior c


def test_rholrate_ratchets_at_maxdecs_not_per_decrease():
    """Issue #195 (mirrors #193/#194 for torch/numpy): the MLX rho learning rate is
    a maxdecs-ratcheted CEILING, not a per-LL-decrease monotone decay.

    Fortran resets ``rholrate = rholrate0`` each iteration before the rho update
    (amica15.f90:1806/1813) and only tightens the ceiling at ``maxdecs``
    (amica15.f90:1068, gated on ``iter > newt_start``). MLX previously decayed
    ``rholrate`` on EVERY LL decrease with no reset, collapsing it toward ~1e-5
    within a few hundred iterations and freezing rho at a stale shape.

    An aggressive-lrate natural-gradient run on the real sample overshoots and
    triggers several LL decreases; with ``newt_start=0`` (gate always open) the
    ceiling ratchets once per ``maxdecs`` decreases, NOT once per decrease.
    """
    from pamica.mlx_impl import AMICAMLXNG

    data = _load_real_data()
    m = AMICAMLXNG(
        n_channels=NW, n_mix=NMIX, seed=SEED, block_size=256,
        lrate=0.5, lratefact=0.5, rholrate=0.05, rholratefact=0.5,
        maxdecs=3, newt_start=0,
    )  # fmt: skip
    m.fit(data, max_iter=400, verbose=False)

    ll = m.ll_history
    n_dec = sum(1 for i in range(1, len(ll)) if ll[i] < ll[i - 1])
    assert m.lrate > m.minlrate, "run hit the lrate floor; use a gentler config"
    assert n_dec >= m.maxdecs, "too few LL decreases to exercise the ratchet"

    # The ceiling ratcheted a whole number of times at the maxdecs cadence
    # (rholrate0 * rholratefact**k), far fewer steps than one-per-decrease.
    assert m.rholrate < m.rholrate0
    k = round(np.log(m.rholrate / m.rholrate0) / np.log(m.rholratefact))
    assert m.rholrate == pytest.approx(m.rholrate0 * m.rholratefact**k)
    assert k <= n_dec // m.maxdecs + 1
    # Guard against a regression to the old per-decrease decay (orders below).
    buggy = m.rholrate0 * (m.rholratefact**n_dec)
    assert m.rholrate > buggy * 10


def test_rholrate_ceiling_ratchet_gated_on_newt_start():
    """The rholrate ceiling ratchet is gated on ``iter > newt_start`` (Fortran
    amica15.f90:1067, independent of do_newton). With ``newt_start`` past the whole
    budget, LL decreases must NOT tighten the rho ceiling -- it stays at
    ``rholrate0`` exactly (only ``lrate_cap`` ratchets, which is not gated)."""
    from pamica.mlx_impl import AMICAMLXNG

    data = _load_real_data()
    m = AMICAMLXNG(
        n_channels=NW, n_mix=NMIX, seed=SEED, block_size=256,
        lrate=0.5, lratefact=0.5, rholrate=0.05, rholratefact=0.5,
        maxdecs=3, newt_start=100_000,
    )  # fmt: skip
    m.fit(data, max_iter=400, verbose=False)

    ll = m.ll_history
    n_dec = sum(1 for i in range(1, len(ll)) if ll[i] < ll[i - 1])
    assert n_dec >= m.maxdecs, "config did not exercise the decrease path"
    # Gated off for the whole run: the rho ceiling never moved.
    assert m.rholrate == m.rholrate0


def test_rholrate_ceiling_resets_at_fit_start():
    """Issue #195: the rho-rate ceiling is reset to rholrate0 at fit start
    (``_initialize_parameters``, the same call ``fit`` makes), so a previously
    ratcheted ``rholrate`` does not carry across a re-fit/restart -- parity with
    the numpy backend's ``test_reinitialize_for_restart_resets_rho_ceiling``."""
    from pamica.mlx_impl import AMICAMLXNG

    m = AMICAMLXNG(n_channels=NW, n_mix=NMIX, seed=SEED)
    # Simulate a prior fit that ratcheted both ceilings down at maxdecs.
    m.rholrate = m.rholrate0 * 0.25
    m.lrate_cap = m.lrate0 * 0.25
    m._initialize_parameters()  # fit-start reset
    assert m.rholrate == m.rholrate0
    assert m.lrate_cap == m.lrate0
