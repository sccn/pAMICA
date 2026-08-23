"""Tests for the pdftype source-density families (issue #26).

Real sample EEG data only (no synthetic/mock). The decisive correctness check
is that the vectorized log-density ``z0`` and score ``fp`` reproduce the literal
``amica15.f90`` expressions to float64 precision for every family; ``amica15mac``
is the reference binary, so ``amica15.f90`` is the ground-truth source. An opt-in
integration test (``AMICA_RUN_FORTRAN=1``) checks converged log-likelihood parity
against the binary itself.
"""

import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from pamica.torch_impl import AMICATorchNG
from pamica.torch_impl.core import _log_pdf_and_deriv, _log_pdf_only, _score

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "sample_data"
DATA_FILE = SAMPLE_DIR / "eeglab_data.fdt"
NW = 32
FIELD = 30504

# Fortran log-normalizer literals (amica15.f90:1333/1346/1359/1371).
_LOG4 = math.log(4.0)
_LSQ2PI = math.log(2.506628274)
_LNSUB = math.log(4.132731354)
_LNSUP = math.log(1.858073988)


def _load_real_data() -> np.ndarray:
    from pamica.torch_impl.utils import load_eeglab_data

    return load_eeglab_data(str(DATA_FILE), data_dim=NW, field_dim=FIELD).astype(
        np.float64
    )


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
# M-step denominators, and the score is unambiguous away from 0).
_Y = torch.linspace(-8.0, 8.0, 65, dtype=torch.float64)
_Y = _Y[_Y.abs() > 1e-3]
_RHO = torch.full_like(_Y, 1.5)  # rho is frozen at rho0 for non-GG families


@pytest.mark.parametrize("code", [2, 3, 4, 1])
def test_family_log_pdf_matches_fortran(code: int):
    """z0 (log-density) reproduces the literal amica15.f90 formula."""
    pdt = torch.full_like(_Y, code, dtype=torch.long)
    log_pdf, _ = _log_pdf_and_deriv(_Y, _RHO, pdt)
    ref = _fortran_z0(_Y.numpy(), code)
    assert np.max(np.abs(log_pdf.numpy() - ref)) < 1e-12


@pytest.mark.parametrize("code", [2, 3, 4, 1])
def test_family_score_matches_fortran(code: int):
    """fp (score) reproduces the literal amica15.f90 formula."""
    pdt = torch.full_like(_Y, code, dtype=torch.long)
    fp = _score(_Y, _RHO, pdt)
    ref = _fortran_fp(_Y.numpy(), code)
    assert np.max(np.abs(fp.numpy() - ref)) < 1e-12


def test_family_dpdf_is_minus_fp_times_pdf():
    """The density derivative obeys dpdf = -fp * pdf for every family."""
    for code in (0, 2, 3, 4, 1):
        pdt = torch.full_like(_Y, code, dtype=torch.long)
        log_pdf, dpdf = _log_pdf_and_deriv(_Y, _RHO, pdt)
        fp = _score(_Y, _RHO, pdt)
        expected = -fp * torch.exp(log_pdf)
        assert torch.max(torch.abs(dpdf - expected)) < 1e-12


def test_gg_path_bit_identical_none_vs_code0():
    """The default GG path is bit-identical whether pdtype is None or all-zero,
    so pdftype=0 runs are byte-for-byte the pre-#26 implementation."""
    lp_none, dp_none = _log_pdf_and_deriv(_Y, _RHO, None)
    fp_none = _score(_Y, _RHO, None)
    pdt0 = torch.zeros_like(_Y, dtype=torch.long)
    lp_zero, dp_zero = _log_pdf_and_deriv(_Y, _RHO, pdt0)
    fp_zero = _score(_Y, _RHO, pdt0)
    assert torch.equal(lp_none, lp_zero)
    assert torch.equal(dp_none, dp_zero)
    assert torch.equal(fp_none, fp_zero)


@pytest.mark.parametrize("code", [None, 0, 1, 2, 3, 4])
def test_log_pdf_only_matches_log_pdf_and_deriv(code):
    """The E-step's lean ``_log_pdf_only`` (issue #63) must return exactly the
    same ``log_pdf`` as ``_log_pdf_and_deriv`` for every family, and its second
    return must be ``|y|^rho`` -- this locks in the bit-identity the whole
    optimization rests on (families 1-4 have no other non-gated regression
    check)."""
    pdt = None if code is None else torch.full_like(_Y, code, dtype=torch.long)
    lp_full, _ = _log_pdf_and_deriv(_Y, _RHO, pdt)
    lp_only, az_rho = _log_pdf_only(_Y, _RHO, pdt)
    assert torch.equal(lp_full, lp_only)
    assert torch.equal(az_rho, _Y.abs().pow(_RHO))


def test_block_size_default_is_8192():
    """Pin the issue #216 default so a revert/typo is caught (the block-size
    invariance tests would still pass at any finite block size).

    Raised from 512, which left every backend dispatch-bound: ~6x on CPU float64
    for the bundled sample. Chosen over a larger value because peak block memory
    scales with it and 8192 stays safe at high channel counts.
    """
    assert AMICATorchNG(n_channels=NW, device="cpu").block_size == 8192


def test_pdtype_h_is_none_only_for_gg():
    """_pdtype_h returns None on the GG fast path and a tensor otherwise."""
    gg = AMICATorchNG(n_channels=NW, n_mix=3, pdftype=0, device="cpu", seed=0)
    gg._initialize_parameters()
    assert gg._pdtype_h(0) is None
    for pdftype, n_mix in [(2, 3), (3, 3), (4, 1), (1, 1)]:
        m = AMICATorchNG(
            n_channels=NW, n_mix=n_mix, pdftype=pdftype, device="cpu", seed=0
        )
        m._initialize_parameters()
        ph = m._pdtype_h(0)
        assert ph is not None and ph.shape == (1, NW, 1)


def test_pdftype_validation():
    """Construction rejects unknown pdftype and mixture single-component combos."""
    with pytest.raises(ValueError):
        AMICATorchNG(n_channels=NW, pdftype=7, device="cpu")
    for bad in (1, 4):  # single-component families require n_mix == 1
        with pytest.raises(ValueError):
            AMICATorchNG(n_channels=NW, n_mix=3, pdftype=bad, device="cpu")
    # Valid single-component construction succeeds.
    AMICATorchNG(n_channels=NW, n_mix=1, pdftype=4, device="cpu")


def test_kurt_schedule_validation():
    """The adaptive-switch schedule params are validated at construction so a
    bad value fails loudly instead of crashing deep in fit()."""
    for bad in dict(kurt_int=0, kurt_start=0, num_kurt=-1).items():
        kwargs: dict[str, Any] = dict([bad])
        with pytest.raises(ValueError):
            AMICATorchNG(n_channels=NW, n_mix=1, pdftype=1, device="cpu", **kwargs)
    # The same params are inert (unvalidated) for non-adaptive pdftype.
    AMICATorchNG(n_channels=NW, n_mix=3, pdftype=0, device="cpu", kurt_int=0)


def test_pdtype_from_kurtosis_decision():
    """The pure kurtosis->family decision: super-G(+)->1, sub-G(-)->4, and a
    non-finite / dead-model kurtosis keeps the prior pdtype (the guard). This is
    the sub-Gaussian (code 4) switch branch that real EEG rarely triggers."""
    # Single model: cover +, -, NaN (keep prior), - again.
    m = AMICATorchNG(n_channels=4, n_mix=1, pdftype=1, device="cpu", seed=0)
    m._initialize_parameters()  # sets self.pdtype to all-1 (prior)
    kurt = torch.tensor([[2.0], [-2.0], [float("nan")], [-0.5]], dtype=torch.float64)
    nsub = torch.tensor([10.0], dtype=torch.float64)
    out = m._pdtype_from_kurtosis(kurt, nsub)
    assert out.flatten().tolist() == [1, 4, 1, 4]  # NaN -> kept prior (1)

    # Two models, model 1 dead (nsub==0): its sources keep the prior (1) even
    # though their (finite) negative kurtosis would otherwise pick code 4.
    m2 = AMICATorchNG(
        n_channels=2, n_models=2, n_mix=1, pdftype=1, device="cpu", seed=0
    )
    m2._initialize_parameters()
    kurt2 = torch.full((2, 2), -2.0, dtype=torch.float64)
    nsub2 = torch.tensor([10.0, 0.0], dtype=torch.float64)
    out2 = m2._pdtype_from_kurtosis(kurt2, nsub2)
    assert out2[:, 0].tolist() == [4, 4]  # live model switched to sub-Gaussian
    assert out2[:, 1].tolist() == [1, 1]  # dead model kept prior


@pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing")
@pytest.mark.parametrize("pdftype,n_mix", [(0, 3), (2, 3), (3, 3), (4, 1), (1, 1)])
def test_family_fit_finite_and_monotone(pdftype: int, n_mix: int):
    """Every family fits real EEG to a finite LL that does not regress below its
    starting value (natural-gradient AMICA can dip mid-run, so this checks net
    non-decrease, last >= first, not strict monotonicity)."""
    data = _load_real_data()
    kw: dict[str, Any] = (
        dict(num_kurt=0) if pdftype == 1 else {}
    )  # fixed super-G, no switching
    m = AMICATorchNG(
        n_channels=NW, n_mix=n_mix, pdftype=pdftype, device="cpu", seed=0, **kw
    )
    m.fit(data, max_iter=15, verbose=False)
    ll = np.asarray(m.ll_history)
    assert np.all(np.isfinite(ll))
    assert m.W is not None
    assert np.all(np.isfinite(m.W.cpu().numpy()))
    assert ll[-1] >= ll[0] - 1e-6


@pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing")
def test_auto_switcher_runs_and_is_stable():
    """The extended-Infomax switcher runs the full schedule, keeps every source
    in a valid family, and stays finite with a net non-decreasing LL (last >=
    first) on real EEG."""
    data = _load_real_data()
    m = AMICATorchNG(
        n_channels=NW,
        n_mix=1,
        pdftype=1,
        device="cpu",
        seed=0,
        kurt_start=3,
        num_kurt=5,
        kurt_int=1,
    )
    m.fit(data, max_iter=20, verbose=False)
    assert m.n_kurt_done == 5
    assert m.pdtype is not None
    pdt = m.pdtype.cpu().numpy()
    assert set(np.unique(pdt)).issubset({1, 4})
    ll = np.asarray(m.ll_history)
    assert np.all(np.isfinite(ll))
    assert ll[-1] >= ll[0] - 1e-6


@pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing")
def test_auto_switch_noop_when_num_kurt_zero():
    """num_kurt=0 disables switching: the run is identical to the fixed
    super-Gaussian family (pdtype stays at the code-1 init)."""
    data = _load_real_data()
    fixed = AMICATorchNG(
        n_channels=NW, n_mix=1, pdftype=1, device="cpu", seed=0, num_kurt=0
    )
    fixed.fit(data, max_iter=10, verbose=False)
    assert fixed.n_kurt_done == 0
    assert fixed.pdtype is not None
    assert np.all(fixed.pdtype.cpu().numpy() == 1)


@pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing")
def test_state_dict_roundtrips_pdftype_state():
    """save/load must preserve the density family. A fixed non-GG model and the
    adaptive switcher's per-source pdtype/n_kurt_done must survive a round-trip
    (else a reloaded model silently reverts to GG)."""
    data = _load_real_data()

    # Fixed logistic family.
    m = AMICATorchNG(n_channels=NW, n_mix=3, pdftype=3, device="cpu", seed=0)
    m.fit(data, max_iter=8, verbose=False)
    loaded = AMICATorchNG.from_state_dict(m.state_dict(), device="cpu")
    assert loaded.pdftype == 3 and loaded.dorho is False
    assert loaded.pdtype is not None and m.pdtype is not None
    assert torch.equal(loaded.pdtype, m.pdtype)
    assert np.allclose(loaded.transform(data), m.transform(data))

    # Adaptive switcher: per-source pdtype and the switch counter must persist.
    ad = AMICATorchNG(
        n_channels=NW,
        n_mix=1,
        pdftype=1,
        device="cpu",
        seed=0,
        kurt_start=3,
        num_kurt=5,
        kurt_int=1,
    )
    ad.fit(data, max_iter=12, verbose=False)
    ad_loaded = AMICATorchNG.from_state_dict(ad.state_dict(), device="cpu")
    assert ad_loaded.pdftype == 1 and ad_loaded.do_choose_pdfs is True
    assert ad_loaded.n_kurt_done == ad.n_kurt_done == 5
    assert ad_loaded.pdtype is not None and ad.pdtype is not None
    assert torch.equal(ad_loaded.pdtype, ad.pdtype)


@pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing")
@pytest.mark.parametrize("pdftype,n_mix", [(2, 3), (3, 3), (4, 1)])
def test_family_fit_with_newton(pdftype: int, n_mix: int):
    """Non-GG families run with the Newton preconditioner (as Fortran does) and
    stay finite/monotone on real EEG; the cosh curvature may fall back to natural
    gradient, which is expected and must not crash."""
    data = _load_real_data()
    m = AMICATorchNG(
        n_channels=NW,
        n_mix=n_mix,
        pdftype=pdftype,
        device="cpu",
        seed=0,
        do_newton=True,
        newt_start=5,
    )
    m.fit(data, max_iter=15, verbose=False)
    ll = np.asarray(m.ll_history)
    assert np.all(np.isfinite(ll))
    assert ll[-1] >= ll[0] - 1e-6


@pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing")
def test_adaptive_switch_with_rejection():
    """The adaptive switcher runs after outlier rejection has shrunk the sample
    set (it consumes the post-rejection X_use) without crashing on real EEG."""
    data = _load_real_data()
    m = AMICATorchNG(
        n_channels=NW,
        n_mix=1,
        pdftype=1,
        device="cpu",
        seed=0,
        do_reject=True,
        rejstart=2,
        rejint=2,
        maxrej=1,
        kurt_start=3,
        num_kurt=3,
        kurt_int=1,
    )
    m.fit(data, max_iter=12, verbose=False)
    assert np.all(np.isfinite(m.ll_history))
    assert m.pdtype is not None
    assert set(np.unique(m.pdtype.cpu().numpy())).issubset({1, 4})


@pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing")
def test_multimodel_fixed_family():
    """A fixed non-GG family works with n_models>1 (exercises the per-model
    _pdtype_h / _choose_pdfs indexing path)."""
    data = _load_real_data()
    m = AMICATorchNG(
        n_channels=NW, n_models=2, n_mix=3, pdftype=2, device="cpu", seed=0
    )
    m.fit(data, max_iter=10, verbose=False)
    ll = np.asarray(m.ll_history)
    assert np.all(np.isfinite(ll))
    assert m.pdtype is not None
    assert m.pdtype.shape == (NW, 2)


@pytest.mark.skipif(
    os.environ.get("AMICA_RUN_FORTRAN") != "1",
    reason="opt-in Fortran-binary integration test (set AMICA_RUN_FORTRAN=1)",
)
@pytest.mark.parametrize("pdftype,n_mix", [(0, 3), (2, 3), (3, 3), (4, 1), (1, 1)])
def test_family_converged_ll_matches_fortran(pdftype: int, n_mix: int):
    """Converged LL parity vs amica15mac with the optimizer matched (Newton on).

    Slow (runs the binary + a full NG fit per family); gated behind
    AMICA_RUN_FORTRAN=1 so it does not slow default CI.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from validate_implementations import load_sample_data, run_fortran_amica

    data, params = load_sample_data()
    fp = dict(params)
    fp.update(pdftype=pdftype, num_mix=n_mix, max_iter=150, do_newton=True)
    run_dir = SAMPLE_DIR.parent.parent / "scratch_amica_parity" / f"pt{pdftype}"
    run_dir.mkdir(parents=True, exist_ok=True)
    fres = run_fortran_amica(data, fp, run_dir, seed=0)
    assert fres is not None and "final_ll" in fres

    kw: dict[str, Any] = dict(num_kurt=0) if pdftype == 1 else {}
    m = AMICATorchNG(
        n_channels=NW,
        n_mix=n_mix,
        pdftype=pdftype,
        device="cpu",
        seed=0,
        lrate=0.05,
        do_newton=True,
        newt_start=50,
        **kw,
    )
    m.fit(data, max_iter=150, verbose=False)
    assert abs(m.ll_history[-1] - fres["final_ll"]) < 0.02


# --- three-way: sharing x Newton x pdf family (issue #277) -------------------


@pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing")
def test_sharing_newton_and_fixed_family_fit_completes():
    """``share_comps`` + ``do_newton`` + a non-default fixed ``pdftype``
    together.

    The three features have only pairwise coverage elsewhere: sharing x Newton
    is ``test_ng_sharing.py::test_two_model_share_fit_survives_merge``, Newton
    x families is ``test_family_fit_with_newton`` above. Risk is structural
    rather than numerical -- the Newton curvature is indexed by (model,
    channel), not by ``comp_list``, and the family dispatch reads only
    ``pdtype``/``rho`` -- but nothing had run all three together before this
    test (mirrors ``mlx_tests/test_mlx_pdf.py::
    test_sharing_newton_and_fixed_family_fit_completes``). Newton fallbacks are
    PERMITTED, not asserted to zero: a rejected direction under a non-GG
    family is expected, per ``test_family_fit_with_newton``.
    """
    data = _load_real_data()[:, :8192]
    m = AMICATorchNG(
        n_channels=NW,
        n_models=2,
        n_mix=3,
        pdftype=2,
        device="cpu",
        seed=42,
        block_size=1024,
        do_newton=True,
        newt_start=5,
        share_comps=True,
        share_start=8,
        share_iter=10,
        comp_thresh=0.9,
    )
    m.fit(data, max_iter=30, verbose=False)

    ll = np.asarray(m.ll_history, dtype=float)
    assert np.all(np.isfinite(ll))
    assert m.A is not None and torch.isfinite(m.A).all()
    assert m.W is not None and torch.isfinite(m.W).all()
    assert m.pdtype is not None
    assert np.all(m.pdtype.cpu().numpy() == 2)
    used = int(m.comp_used.sum())
    assert used < m.n_comps, "no merge fired; the three-way interplay is untested"
    assert 0 <= m.n_newton_fallbacks <= len(ll)


@pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing")
def test_sharing_newton_and_adaptive_switcher_fit_completes():
    """The single-component adaptive-switcher variant of the three-way
    interplay: ``share_comps`` + ``do_newton`` + the extended-Infomax switcher
    (``pdftype=1``, ``n_mix=1``) together (mirrors ``mlx_tests/test_mlx_pdf.py::
    test_sharing_newton_and_adaptive_switcher_fit_completes``)."""
    data = _load_real_data()[:, :8192]
    m = AMICATorchNG(
        n_channels=NW,
        n_models=2,
        n_mix=1,
        pdftype=1,
        device="cpu",
        seed=42,
        block_size=1024,
        do_newton=True,
        newt_start=5,
        share_comps=True,
        share_start=8,
        share_iter=10,
        comp_thresh=0.9,
        kurt_start=3,
        num_kurt=5,
        kurt_int=1,
    )
    m.fit(data, max_iter=30, verbose=False)

    ll = np.asarray(m.ll_history, dtype=float)
    assert np.all(np.isfinite(ll))
    assert m.A is not None and torch.isfinite(m.A).all()
    assert m.W is not None and torch.isfinite(m.W).all()
    assert m.pdtype is not None
    codes = set(np.unique(m.pdtype.cpu().numpy()).tolist())
    assert codes.issubset({1, 4})
    used = int(m.comp_used.sum())
    assert used < m.n_comps, "no merge fired; the three-way interplay is untested"
    assert 0 <= m.n_newton_fallbacks <= len(ll)
