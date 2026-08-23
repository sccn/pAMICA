"""Non-GG source-density families: MLX (float32) vs PyTorch/Fortran agreement
(issue #265, porting the PyTorch backend's issue #26).

Cross-backend by design, so it lives in ``pamica/tests/`` rather than
``pamica/tests/mlx_tests/`` (``.rules/backend_parity.md``: a test that pins two
backends against each other belongs to neither of them -- same placement as
``test_mlx_sharing_cross_backend.py`` and ``test_mlx_newton_cross_backend.py``).
The MLX-only mechanics (ctor validation, the switcher, ``_logcosh`` stability,
sharing x switcher) stay in ``mlx_tests/test_mlx_pdf.py``.

Two oracles are used here:

1. The literal ``amica15.f90`` closed forms (``_fortran_z0``/``_fortran_fp``,
   ported from ``torch_tests/test_ng_pdf_families.py``) for the pure density
   functions, evaluated through MLX's float32 ``_score``/``_log_pdf``.
2. A float64 ``AMICATorchNG`` twin holding the MLX model's exact state (the
   ``_torch_twin`` pattern from ``test_mlx_sharing_cross_backend.py``,
   extended here to also construct the twin with the same ``pdftype`` and copy
   ``pdtype``/``n_kurt_done``), for the M-step and full-fit comparisons.

Real bundled sample EEG only, no synthetic data (``.rules/testing.md``). MLX is
an optional Apple-Silicon backend, so the module self-skips via
``importorskip`` plus an Apple-GPU guard; PyTorch always runs.
"""

import math
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
_score = mlx_core._score
_log_pdf = mlx_core._log_pdf

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

# Fortran log-normalizer literals (amica15.f90:1333/1346/1359/1371), matching
# torch_tests/test_ng_pdf_families.py.
_LOG4 = math.log(4.0)
_LSQ2PI = math.log(2.506628274)
_LNSUB = math.log(4.132731354)
_LNSUP = math.log(1.858073988)


def _real_data(n_samples: int = 4096) -> np.ndarray:
    from pamica.torch_impl.utils import load_eeglab_data

    data = load_eeglab_data(str(DATA_FILE), data_dim=NW, field_dim=FIELD)
    return data[:, :n_samples].astype(np.float64)


def _fortran_z0(y: np.ndarray, code: int) -> np.ndarray:
    """Literal amica15.f90 log-density (alpha=beta=1, mu=0 so y=b)."""
    if code == 2:  # Gaussian, :1333
        return -0.5 * y * y - _LSQ2PI
    if code == 3:  # logistic, :1346
        return -2.0 * np.log(np.cosh(0.5 * y)) - _LOG4
    if code == 4:  # sub-Gaussian cosh+, :1359
        return -0.5 * y * y + np.log(np.cosh(y)) - _LNSUB
    if code == 1:  # super-Gaussian cosh-, :1371
        return -0.5 * y * y - np.log(np.cosh(y)) - _LNSUP
    raise ValueError(code)


def _fortran_fp(y: np.ndarray, code: int) -> np.ndarray:
    """Literal amica15.f90 score (:1467-1491)."""
    return {
        2: y,
        3: np.tanh(y / 2.0),
        4: y - np.tanh(y),
        1: y + np.tanh(y),
    }[code]


# A grid spanning the density tails; drop exact 0 (Fortran divides by y in some
# M-step denominators, and the score is unambiguous away from 0), matching
# torch_tests/test_ng_pdf_families.py's _Y/_RHO.
_Y = np.linspace(-8.0, 8.0, 65)
_Y = _Y[np.abs(_Y) > 1e-3]
_RHO = np.full_like(_Y, 1.5)  # rho is frozen at rho0 for non-GG families

# rtol=atol=1e-6: at large |y| the relative term dominates (float32 vs the
# float64 literal reference agrees to ~1e-7 relative there); near the grid's
# smallest retained |y| (~0.25 given the 65-point spacing) code 4's
# fp = y - tanh(y) starts to catastrophically cancel in float32 -- measured
# 100% relative error at y=1e-4 -- so the absolute term is what keeps that
# region meaningful (the true value itself is ~y^3/3, tiny, so a tiny absolute
# gap there is not a real disagreement). Do NOT add a Taylor branch (plan
# policy 4): parity with AMICATorchNG's literal formula is the spec.
_RTOL = 1e-6
_ATOL = 1e-6


def _lgamma_table_for(rho: np.ndarray) -> "mx.array":
    from scipy.special import gammaln

    return mx.array(gammaln(1.0 + 1.0 / rho).astype(np.float32))


# --- (a) Fixed-family z0/fp vs the literal Fortran closed forms -------------


@pytest.mark.parametrize("code", [2, 3, 4, 1])
def test_family_log_pdf_matches_fortran(code: int):
    """z0 (log-density) reproduces the literal amica15.f90 formula through
    MLX's float32 ``_log_pdf``."""
    y32 = mx.array(_Y.astype(np.float32))
    rho32 = mx.array(_RHO.astype(np.float32))
    pdt = mx.array(np.full_like(_Y, code, dtype=np.int32))
    lgamma_table = _lgamma_table_for(_RHO)
    log_pdf, az_rho = _log_pdf(y32, rho32, lgamma_table, pdt)
    assert az_rho is None  # policy 5: not returned on the non-GG path
    lp = np.array(log_pdf, dtype=np.float64)
    ref = _fortran_z0(_Y, code)
    np.testing.assert_allclose(lp, ref, rtol=_RTOL, atol=_ATOL)


@pytest.mark.parametrize("code", [2, 3, 4, 1])
def test_family_score_matches_fortran(code: int):
    """fp (score) reproduces the literal amica15.f90 formula through MLX's
    float32 ``_score``."""
    y32 = mx.array(_Y.astype(np.float32))
    rho32 = mx.array(_RHO.astype(np.float32))
    pdt = mx.array(np.full_like(_Y, code, dtype=np.int32))
    fp = np.array(_score(y32, rho32, pdt), dtype=np.float64)
    ref = _fortran_fp(_Y, code)
    np.testing.assert_allclose(fp, ref, rtol=_RTOL, atol=_ATOL)


# --- (b) GG-path byte identity -----------------------------------------------


def test_gg_path_bit_identical_none_vs_code0():
    """The default GG path is bit-identical whether ``pdtype`` is ``None`` or
    an explicit all-zero array, so ``pdftype=0`` runs are byte-for-byte the
    pre-#265 implementation (mirrors
    ``torch_tests/test_ng_pdf_families.py::test_gg_path_bit_identical_none_vs_code0``).
    ``az_rho`` is intentionally NOT compared: by design (policy 5) it is the
    real ``|y|^rho`` on the ``None`` path but ``None`` on the explicit-array
    path (nobody needing it there, since ``self.dorho`` gates its only
    consumer) -- an asymmetry in the Python return contract, not in any
    observable density value."""
    y32 = mx.array(_Y.astype(np.float32))
    rho32 = mx.array(_RHO.astype(np.float32))
    lgamma_table = _lgamma_table_for(_RHO)

    lp_none, _ = _log_pdf(y32, rho32, lgamma_table, None)
    fp_none = _score(y32, rho32, None)

    pdt0 = mx.array(np.zeros_like(_Y, dtype=np.int32))
    lp_zero, _ = _log_pdf(y32, rho32, lgamma_table, pdt0)
    fp_zero = _score(y32, rho32, pdt0)

    mx.eval(lp_none, fp_none, lp_zero, fp_zero)
    assert np.array_equal(np.array(lp_none), np.array(lp_zero))
    assert np.array_equal(np.array(fp_none), np.array(fp_zero))


# --- (c)/(d): matched-state M-step and full-fit vs the float64 torch twin ---


def _warm_mlx_model(pdftype: int, n_mix: int, warmup: int = 3):
    """An MLX model of the given fixed family, driven ``warmup`` M-steps past
    init on the real sample, returned with its sphered data. Hand-driving the
    loop (rather than calling ``fit``) is what lets the comparison below start
    both backends from one identical mid-fit state; every step still goes
    through the production ``_update_parameters``."""
    model = AMICAMLXNG(
        n_channels=NW,
        n_mix=n_mix,
        pdftype=pdftype,
        seed=SEED,
        block_size=BLOCK,
    )
    x_t = model._preprocess(_real_data())
    model._initialize_parameters()
    for it in range(warmup):
        model.iteration = it
        model._update_parameters(model._accumulate_blocks(x_t), x_t.shape[1])
    mx.eval(
        model.A, model.mu, model.alpha, model.beta, model.rho, model.gm, model.c,
        model.pdtype,
    )  # fmt: skip
    return model, x_t


def _torch_twin(model, x_t):
    """A float64 ``AMICATorchNG`` holding the MLX model's exact state --
    including ``pdftype``/``pdtype``/``n_kurt_done`` (extending the
    ``test_mlx_sharing_cross_backend.py`` pattern for issue #265) -- plus the
    same sphered data as a float64 tensor. The two then differ only in the
    arithmetic they run, so a one-iteration comparison isolates the update."""
    dtype = torch.float64
    ng = AMICATorchNG(
        n_channels=NW,
        n_models=model.n_models,
        n_mix=model.n_mix,
        device="cpu",
        dtype=dtype,
        do_newton=False,
        block_size=model.block_size,
        seed=SEED,
        pdftype=model.pdftype,
        kurt_start=model.kurt_start,
        num_kurt=model.num_kurt,
        kurt_int=model.kurt_int,
        doscaling=model.doscaling,
        scalestep=model.scalestep,
    )
    ng._initialize_parameters()
    for name in ("A", "mu", "alpha", "beta", "rho", "gm", "c"):
        value = np.array(getattr(model, name)).astype(np.float64)
        setattr(ng, name, torch.from_numpy(value).to(dtype))
    ng.comp_list = torch.from_numpy(np.array(model.comp_list).astype(np.int64))
    assert model.pdtype is not None and ng.pdtype is not None
    ng.pdtype = torch.from_numpy(np.array(model.pdtype).astype(np.int64))
    ng.n_kurt_done = model.n_kurt_done
    ng.sphere = torch.from_numpy(model._sphere_np.copy()).to(dtype)
    ng._sphere_pinv = None
    ng.lrate = model.lrate
    ng.rholrate = model.rholrate
    ng.iteration = model.iteration
    ng.sldet = model.sldet
    ng._update_unmixing_matrices()
    x_ng = torch.from_numpy(np.array(x_t).astype(np.float64)).to(dtype)
    return ng, x_ng


@pytest.mark.parametrize("pdftype,n_mix", [(2, NMIX), (3, NMIX), (4, 1), (1, 1)])
def test_one_mstep_matches_torch_twin(pdftype: int, n_mix: int):
    """From one real fitted state, one more M-step's A/mu/beta agree with the
    float64 PyTorch twin to float32 precision, for every fixed non-GG family.

    Absolute tolerance (not elementwise-relative): A/mu/beta legitimately hold
    entries near zero (e.g. off-diagonal mixing terms), where an elementwise
    relative comparison spikes on float32 noise despite an absolute gap many
    orders below anything meaningful -- the same reasoning and the same 1e-4
    threshold as ``test_mlx_sharing_cross_backend.py::
    test_gm_prev_weighting_matches_torch``. Measured on this state: max
    absolute gap ~5e-7 across all four families, four orders below the bound.
    """
    model, x_t = _warm_mlx_model(pdftype, n_mix, warmup=3)
    ng, x_ng = _torch_twin(model, x_t)

    model.iteration = ng.iteration = 3
    model._update_parameters(model._accumulate_blocks(x_t), x_t.shape[1])
    ng._update_parameters(ng._accumulate_blocks(x_ng), x_ng.shape[1])
    mx.eval(model.A, model.mu, model.beta)

    for name in ("A", "mu", "beta"):
        a = np.array(getattr(model, name), dtype=np.float64)
        b = getattr(ng, name).numpy()
        gap = np.abs(a - b).max()
        assert gap < 1e-4, f"{name} gap {gap:.2e} for pdftype={pdftype}"


@pytest.mark.slow
@pytest.mark.parametrize("pdftype,n_mix", [(2, NMIX), (3, NMIX), (4, 1), (1, 1)])
def test_full_fit_ll_matches_torch_float64(pdftype: int, n_mix: int):
    """At a matched 100-iteration budget on the full recording, the MLX
    float32 fit for every fixed non-GG family is not materially worse than the
    float64 PyTorch fit of the same family (mirrors
    ``test_mlx_newton_cross_backend.py::test_newton_fit_ll_matches_float64_torch``).
    The bar is relational to the in-test float64 fit, never a hardcoded LL.
    """
    data = _real_data(n_samples=FIELD)
    mlx_m = AMICAMLXNG(n_channels=NW, n_mix=n_mix, pdftype=pdftype, seed=SEED)
    mlx_m.fit(data, max_iter=100, verbose=False)

    ng = AMICATorchNG(
        n_channels=NW,
        n_models=1,
        n_mix=n_mix,
        pdftype=pdftype,
        seed=SEED,
        device="cpu",
        dtype=torch.float64,
        do_newton=False,
    )
    ng.fit(data, max_iter=100, verbose=False)

    assert mlx_m.final_ll_ is not None and ng.final_ll_ is not None
    assert np.isfinite(mlx_m.final_ll_)
    assert mlx_m.stop_reason not in AMICAMLXNG._DEGENERATE_STOP_REASONS
    assert mlx_m.final_ll_ >= ng.final_ll_ - 0.05
