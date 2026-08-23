"""Non-GG source-density families on the MLX backend: the MLX-only mechanics
(issue #265, porting the PyTorch backend's issue #26).

What lives here is everything that is not a comparison against another
backend: constructor validation, per-family real-EEG fits, the adaptive
switcher's schedule and its interaction with ``share_comps``, the pure
kurtosis-decision helper, and the ``_logcosh`` stability pin. The float32-vs-
float64 numerical question (the literal Fortran forms and the M-step/full-fit
agreement with a float64 PyTorch twin) is the cross-backend file's job --
``pamica/tests/test_mlx_pdf_families_cross_backend.py`` -- per
``.rules/backend_parity.md``.

Apple-Silicon only; the module self-skips when MLX or an Apple GPU is
unavailable. Real bundled sample EEG throughout (no synthetic/mock data,
``.rules/testing.md``); the kurtosis-decision test is a controlled-input unit
pin on the pure decision function, torch-precedented
(``torch_tests/test_ng_pdf_families.py::test_pdtype_from_kurtosis_decision``).
"""

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from pamica.mlx_impl import AMICAMLXNG  # noqa: E402  (after the MLX importorskip)
from pamica.mlx_impl.core import _logcosh  # noqa: E402

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


def _load_real_data(n_samples: int | None = None) -> np.ndarray:
    from pamica.torch_impl.utils import load_eeglab_data

    data = load_eeglab_data(str(DATA_FILE), data_dim=NW, field_dim=FIELD).astype(
        np.float64
    )
    return data if n_samples is None else data[:, :n_samples]


# --- (e) Constructor validation ---------------------------------------------


def test_pdftype_validation():
    """Construction rejects an unknown ``pdftype`` and the single-component
    mixture mismatches; the valid single-component construction succeeds
    (mirrors ``torch_tests/test_ng_pdf_families.py::test_pdftype_validation``)."""
    with pytest.raises(ValueError):
        AMICAMLXNG(n_channels=NW, pdftype=7)
    for bad in (1, 4):  # single-component families require n_mix == 1
        with pytest.raises(ValueError):
            AMICAMLXNG(n_channels=NW, n_mix=3, pdftype=bad)
    m = AMICAMLXNG(n_channels=NW, n_mix=1, pdftype=4)
    assert m.dorho is False and m.do_choose_pdfs is False


def test_kurt_schedule_validation():
    """The adaptive-switch schedule params are validated at construction under
    ``pdftype=1`` so a bad value fails loudly instead of crashing deep in
    ``fit()``; the same params are inert (unvalidated) under ``pdftype=0``
    (mirrors ``torch_tests/test_ng_pdf_families.py::
    test_kurt_schedule_validation``)."""
    for item in dict(kurt_int=0, kurt_start=0, num_kurt=-1).items():
        bad: dict[str, Any] = dict([item])
        with pytest.raises(ValueError):
            AMICAMLXNG(n_channels=NW, n_mix=1, pdftype=1, **bad)
    AMICAMLXNG(n_channels=NW, n_mix=3, pdftype=0, kurt_int=0)


def test_pdtype_h_is_none_only_for_gg():
    """``_pdtype_h`` returns ``None`` on the GG fast path (``pdftype=0``) and
    a ``(1, NW, 1)`` array otherwise, for every other family (mirrors
    ``torch_tests/test_ng_pdf_families.py::test_pdtype_h_is_none_only_for_gg``).
    A cheap, no-fit failure-attribution pin: a regression here would otherwise
    only surface indirectly, as a slower or numerically different fit."""
    gg = AMICAMLXNG(n_channels=NW, n_mix=3, pdftype=0, seed=0)
    gg._initialize_parameters()
    assert gg._pdtype_h(0) is None
    for pdftype, n_mix in [(2, 3), (3, 3), (4, 1), (1, 1)]:
        m = AMICAMLXNG(n_channels=NW, n_mix=n_mix, pdftype=pdftype, seed=0)
        m._initialize_parameters()
        ph = m._pdtype_h(0)
        assert ph is not None and ph.shape == (1, NW, 1)


# --- (f) Per-family real-EEG fits -------------------------------------------


@pytest.mark.parametrize(
    "pdftype,n_mix", [(0, NMIX), (2, NMIX), (3, NMIX), (4, 1), (1, 1)]
)
def test_family_fit_finite_and_monotone(pdftype: int, n_mix: int):
    """Every family fits real EEG to a finite LL that does not regress below
    its starting value (natural-gradient AMICA can dip mid-run, so this checks
    net non-decrease, last >= first, not strict monotonicity), and reports the
    expected ``get_pdftype`` codes (mirrors ``torch_tests/
    test_ng_pdf_families.py::test_family_fit_finite_and_monotone``)."""
    data = _load_real_data()
    kw: dict[str, Any] = (
        dict(num_kurt=0) if pdftype == 1 else {}
    )  # fixed super-G, no switching
    m = AMICAMLXNG(n_channels=NW, n_mix=n_mix, pdftype=pdftype, seed=0, **kw)
    m.fit(data, max_iter=15, verbose=False)
    ll = np.asarray(m.ll_history)
    assert np.all(np.isfinite(ll))
    assert np.all(np.isfinite(np.array(m.A)))
    assert ll[-1] >= ll[0] - 1e-6
    assert np.all(m.get_pdftype() == pdftype)


def test_multimodel_fixed_family():
    """A fixed non-GG family works with ``n_models>1``, exercising the
    per-model ``_pdtype_h`` indexing path (mirrors ``torch_tests/
    test_ng_pdf_families.py::test_multimodel_fixed_family``).

    Mutation-tested hole this closes: before this test existed, hardcoding
    ``_pdtype_h`` to always index model 0 (instead of the ``h`` argument)
    passed the entire suite -- every other fixed-family test used
    ``n_models=1``, where model 0 is the only model.
    """
    data = _load_real_data()
    m = AMICAMLXNG(n_channels=NW, n_models=2, n_mix=NMIX, pdftype=2, seed=0)
    m.fit(data, max_iter=10, verbose=False)
    ll = np.asarray(m.ll_history)
    assert np.all(np.isfinite(ll))
    assert np.all(np.isfinite(np.array(m.A)))
    assert ll[-1] >= ll[0] - 1e-6
    assert m.pdtype is not None
    assert np.array(m.pdtype).shape == (NW, 2)


# --- (g) Adaptive switcher ---------------------------------------------------


def test_auto_switcher_runs_and_is_stable():
    """The extended-Infomax switcher runs the full schedule, keeps every
    source in a valid family, and stays finite with a net non-decreasing LL
    (last >= first) on real EEG (mirrors ``torch_tests/
    test_ng_pdf_families.py::test_auto_switcher_runs_and_is_stable``)."""
    data = _load_real_data()
    m = AMICAMLXNG(
        n_channels=NW,
        n_mix=1,
        pdftype=1,
        seed=0,
        kurt_start=3,
        num_kurt=5,
        kurt_int=1,
    )
    m.fit(data, max_iter=20, verbose=False)
    assert m.n_kurt_done == 5
    codes = np.unique(m.get_pdftype())
    assert set(codes.tolist()).issubset({1, 4})
    ll = np.asarray(m.ll_history)
    assert np.all(np.isfinite(ll))
    assert ll[-1] >= ll[0] - 1e-6


def test_auto_switch_noop_when_num_kurt_zero():
    """``num_kurt=0`` disables switching: the run is identical to the fixed
    super-Gaussian family (``pdtype`` stays at the code-1 init; mirrors
    ``torch_tests/test_ng_pdf_families.py::
    test_auto_switch_noop_when_num_kurt_zero``)."""
    data = _load_real_data()
    m = AMICAMLXNG(n_channels=NW, n_mix=1, pdftype=1, seed=0, num_kurt=0)
    m.fit(data, max_iter=10, verbose=False)
    assert m.n_kurt_done == 0
    assert np.all(m.get_pdftype() == 1)


# --- (h) Kurtosis decision unit pin -----------------------------------------


def test_pdtype_from_kurtosis_decision():
    """The pure kurtosis->family decision: super-G(+)->1, sub-G(-)->4, and a
    non-finite / dead-model kurtosis keeps the prior pdtype (the guard). This
    is the sub-Gaussian (code 4) switch branch that real EEG rarely triggers
    (mirrors ``torch_tests/test_ng_pdf_families.py::
    test_pdtype_from_kurtosis_decision``, adapted to the numpy-array
    ``_pdtype_from_kurtosis`` signature)."""
    # Single model: cover +, -, NaN (keep prior), - again.
    m = AMICAMLXNG(n_channels=4, n_mix=1, pdftype=1, seed=0)
    m._initialize_parameters()  # sets self.pdtype to all-1 (prior)
    kurt = np.array([[2.0], [-2.0], [float("nan")], [-0.5]])
    nsub = np.array([10.0])
    out = m._pdtype_from_kurtosis(kurt, nsub)
    assert out.flatten().tolist() == [1, 4, 1, 4]  # NaN -> kept prior (1)

    # Two models, model 1 dead (nsub==0): its sources keep the prior (1) even
    # though their (finite) negative kurtosis would otherwise pick code 4.
    m2 = AMICAMLXNG(n_channels=2, n_models=2, n_mix=1, pdftype=1, seed=0)
    m2._initialize_parameters()
    kurt2 = np.full((2, 2), -2.0)
    nsub2 = np.array([10.0, 0.0])
    out2 = m2._pdtype_from_kurtosis(kurt2, nsub2)
    assert out2[:, 0].tolist() == [4, 4]  # live model switched to sub-Gaussian
    assert out2[:, 1].tolist() == [1, 1]  # dead model kept prior


# --- (i) Newton x non-GG families -------------------------------------------


@pytest.mark.parametrize("pdftype,n_mix", [(2, NMIX), (3, NMIX), (4, 1)])
def test_family_fit_with_newton(pdftype: int, n_mix: int):
    """Non-GG families run with the Newton preconditioner (as Fortran does)
    and stay finite/net-non-decreasing on real EEG; the cosh curvature may
    fall back to natural gradient (ADR 0002), which is expected and must not
    crash -- so fallbacks are PERMITTED, not asserted to zero (mirrors
    ``torch_tests/test_ng_pdf_families.py::test_family_fit_with_newton``)."""
    data = _load_real_data()
    m = AMICAMLXNG(
        n_channels=NW, n_mix=n_mix, pdftype=pdftype, seed=0, do_newton=True,
        newt_start=5,
    )  # fmt: skip
    m.fit(data, max_iter=15, verbose=False)
    ll = np.asarray(m.ll_history)
    assert np.all(np.isfinite(ll))
    assert np.all(np.isfinite(np.array(m.A)))
    assert ll[-1] >= ll[0] - 1e-6
    # A meaningful (failable) bound, not the structurally-unfailable ">= 0"
    # the counter starts at and only ever increments from: it cannot count
    # more fallbacks than iterations actually ran.
    assert 0 <= m.n_newton_fallbacks <= len(ll)


# --- (j) _logcosh stability pin ----------------------------------------------


@pytest.mark.parametrize("x", [88.0, 90.0, 500.0, 1000.0])
def test_logcosh_stability(x: float):
    """``_logcosh`` stays finite out to at least 1e3, where the naive
    ``mx.log(mx.cosh(x))`` overflows float32 from ``|x| >= 90`` (``cosh``
    itself overflows first -- measured: finite at 88, inf at 90). For these
    large arguments ``log(cosh(x)) ~ x - log(2)`` to within the vanishing
    ``log1p(exp(-2x))`` correction, so that is the float64 reference."""
    val = float(_logcosh(mx.array(x, dtype=mx.float32)).item())
    assert math.isfinite(val)
    ref = x - math.log(2.0)
    assert abs(val - ref) < 1e-4, f"logcosh({x})={val}, expected ~{ref}"

    # The naive form is exactly what breaks: confirm it actually overflows
    # at 90 (and not yet at 88), so this pin is testing a real failure mode.
    naive = float(mx.log(mx.cosh(mx.array(x, dtype=mx.float32))).item())
    if x >= 90.0:
        assert math.isinf(naive)
    else:
        assert math.isfinite(naive)


# --- (k) Sharing x switcher smoke -------------------------------------------


def test_sharing_with_adaptive_switcher_smoke():
    """``share_comps`` and the adaptive switcher (``pdftype=1``) run together
    without crashing: the fit completes finite, and every source stays in a
    valid family. ``shared_components()``'s caveat (issue #265 policy 7) is
    exercised, not merely documented: a merge does NOT synchronize ``pdtype``
    across the pair, so this only asserts no crash and valid codes -- not that
    a merged pair agrees, which the caveat explicitly says it need not."""
    data = _load_real_data()
    m = AMICAMLXNG(
        n_channels=NW,
        n_models=2,
        n_mix=1,
        pdftype=1,
        seed=0,
        share_comps=True,
        share_start=5,
        share_iter=10,
        kurt_start=3,
        num_kurt=5,
        kurt_int=1,
    )
    m.fit(data, max_iter=25, verbose=False)
    ll = np.asarray(m.ll_history)
    assert np.all(np.isfinite(ll))
    for h in range(m.n_models):
        codes = np.unique(m.get_pdftype(h))
        assert set(codes.tolist()).issubset({1, 4})
    # No assertion that a shared pair's codes agree -- shared_components()'s
    # docstring says they need not, and this is the state that would show it.
    m.shared_components()


# --- get_pdftype error contract ---------------------------------------------


def test_get_pdftype_rejects_bad_model_idx():
    """A fitted 2-model call raises ``ValueError`` ("out of range") for a
    ``model_idx`` at or past ``n_models``, INCLUDING negative -- the
    docstring's own rationale: MLX's negative indexing would otherwise wrap
    silently to the wrong model -- and ``TypeError`` for a non-int (mirrors
    ``torch_tests/test_ng_metadata.py::
    test_metadata_accessors_reject_bad_model_idx``, scoped to
    ``get_pdftype`` since MLX has no ``get_rho``).

    Confirmed hole (mutation testing): deleting ``_check_model_idx``'s bounds
    check passed the entire suite before this test existed.
    """
    data = _load_real_data()
    m = AMICAMLXNG(n_channels=NW, n_models=2, n_mix=NMIX, seed=0)
    m.fit(data, max_iter=5, verbose=False)
    with pytest.raises(ValueError, match="out of range"):
        m.get_pdftype(model_idx=2)
    with pytest.raises(ValueError, match="out of range"):
        m.get_pdftype(model_idx=-1)
    with pytest.raises(TypeError):
        m.get_pdftype(model_idx="0")  # ty: ignore[invalid-argument-type]


def test_get_pdftype_requires_fit():
    """An unfitted model raises ``RuntimeError``, not an opaque
    ``AttributeError``/``TypeError`` from indexing a ``None`` ``pdtype``
    (mirrors ``torch_tests/test_ng_metadata.py::
    test_metadata_accessors_require_fit``)."""
    m = AMICAMLXNG(n_channels=NW, n_models=1, n_mix=NMIX, seed=0)
    with pytest.raises(RuntimeError, match="fitted"):
        m.get_pdftype()
