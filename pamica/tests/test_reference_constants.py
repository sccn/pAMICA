"""The reference's single-precision constants (issue #344, epic #324 Phase 13).

``amica15.f90`` writes several density normalizers as default-kind Fortran
literals widened with ``dble``, for example ``log(dble(2.506628274))``: the
compiler rounds the decimal to float32, so the binary uses that rounding, not
the decimal. ``epsdble = 1.0e-16`` (``amica15_header.f90:73``) is the same kind
of literal. pamica used the decimals' double values, which moved every mixture
at ``rho == 2`` (the default ``maxrho`` clamp) by 3.0e-8 in its log-density,
and the Gaussian and cosh families by 3.7e-10, 2.0e-8 and -2.1e-8. All three
backends now read the reference's values from :mod:`pamica.reference_constants`.

Pinned here, cross-backend per ``.rules/backend_parity.md`` (PyTorch and NumPy
always run; the MLX checks skip individually without MLX or an Apple GPU):

1. every shared constant is the float32 rounding of the literal on the cited
   reference line, computed here by exact rational arithmetic, independently of
   the module under test; the dyadic literals are exact;
2. the sweep of both reference sources, as executable documentation: the set
   of default-kind literals that are not exact in single precision, and what
   each one is (a density normalizer, an initialization scale, a sentinel, a
   default of an ``input.param`` key, or a declared variable nothing reads);
3. no module of the three backend packages keeps a copy of any of these
   constants, in log or linear form, and the scanner flags every definition
   this change removed;
4. every backend's ``rho == 2`` log-density uses the reference's normalizer,
   and the NumPy plotting helper draws the density the fit uses;
5. (opt-in, ``AMICA_RUN_FORTRAN=1``) the native reference binary, seeded from
   pamica's initialization for the Gaussian and the two cosh families, matches
   PyTorch to float64 round-off after one and three iterations, where the code
   before this change is off by exactly each literal's rounding.
   ``test_component_rows.py`` runs the ``rho == 2`` oracle from a warm
   two-model state.

Real bundled sample EEG only, no synthetic data or mocks (``.rules/testing.md``).
The density checks evaluate the formulas on a grid of activations, as
``torch_tests/test_ng_pdf_families.py`` does.
"""

from __future__ import annotations

import io
import math
import os
import re
import tokenize
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
import torch

import pamica.reference_constants as rc
from pamica.numpy_impl.core import AMICA as AMICA_NumPy
from pamica.numpy_impl.pdf import compute_pdf
from pamica.tests.pre_change import load_pre_change_package
from pamica.torch_impl.core import AMICATorchNG, _log_pdf_and_deriv, _log_pdf_only
from pamica.torch_impl.utils import load_eeglab_data

PACKAGE = Path(__file__).resolve().parents[1]
REFERENCE = PACKAGE / "amica15.f90"
REFERENCE_HEADER = PACKAGE / "amica15_header.f90"
SAMPLE_DIR = PACKAGE / "sample_data"
DATA_FILE = SAMPLE_DIR / "eeglab_data.fdt"
NW = 32
FIELD = 30504
SEED = 42

# The epic #324 head this phase is based on: the last commit with the decimal
# literals' double values.
PRE_CHANGE_COMMIT = "027cb0731dbaaa8e7a5795166af95ba7ff52c2cd"


def _float32_rounding(literal: str) -> float:
    """The float32 value nearest the decimal ``literal``, ties to even, as the
    Fortran compiler rounds a default-kind literal: exact rational arithmetic,
    so no intermediate float64 rounding and nothing shared with the module under
    test. Normal positive values only, which every literal here is."""
    x = Fraction(literal)
    assert x > 0
    e = x.numerator.bit_length() - x.denominator.bit_length()
    if Fraction(2) ** e > x:
        e -= 1
    assert Fraction(2) ** e <= x < Fraction(2) ** (e + 1)
    assert -126 <= e <= 127, "not a normal float32"
    scale = Fraction(2) ** (23 - e)  # 24 significant bits
    return float(Fraction(round(x * scale)) / scale)


def _source_line(path: Path, line: int) -> str:
    return path.read_text().splitlines()[line - 1]


# --- 1. each constant is the float32 rounding of its reference literal ---------
# (constant, reference line, the text on that line, the literal)
_LOG_NORMALIZERS = [
    ("LOG_SQRT_PI", 1313, "log(dble(1.772453851))", "1.772453851"),
    ("LOG_SQRT_2PI", 1333, "log(dble(2.506628274))", "2.506628274"),
    ("LOG_NORM_COSH_SUB", 1359, "log(dble(4.132731354))", "4.132731354"),
    ("LOG_NORM_COSH_SUP", 1371, "log(dble(1.858073988))", "1.858073988"),
]
# The same constants against the double value pamica used before this change:
# log(float32(literal)) - log(literal), and for rho == 2 against 0.5 * log(pi),
# which is what the PyTorch and NumPy backends used there.
_OFFSETS = {
    "LOG_SQRT_PI": (0.5 * math.log(math.pi), 3.0e-8),
    "LOG_SQRT_2PI": (math.log(2.506628274), 3.7e-10),
    "LOG_NORM_COSH_SUB": (math.log(4.132731354), 2.0e-8),
    "LOG_NORM_COSH_SUP": (math.log(1.858073988), -2.1e-8),
}


@pytest.mark.parametrize(
    "name,line,text,literal", _LOG_NORMALIZERS, ids=[n for n, *_ in _LOG_NORMALIZERS]
)
def test_log_normalizer_is_the_reference_literal_in_single_precision(
    name, line, text, literal
):
    assert text in _source_line(REFERENCE, line), f"amica15.f90:{line} moved"
    value = getattr(rc, name)
    assert value == math.log(_float32_rounding(literal))
    # Not the double value of the decimal: the rounding is what differs.
    assert _float32_rounding(literal) != float(literal)
    old, offset = _OFFSETS[name]
    assert value - old == pytest.approx(offset, rel=0.05)


def test_epsdble_is_the_reference_literal_in_single_precision():
    header = _source_line(REFERENCE_HEADER, 73)
    assert "epsdble = 1.0e-16" in header, "amica15_header.f90:73 moved"
    assert "where (tmpy(bstrt:bstp) < epsdble)" in _source_line(REFERENCE, 1558)
    assert rc.EPSDBLE == _float32_rounding("1.0e-16")
    assert rc.EPSDBLE != 1e-16


@pytest.mark.parametrize(
    "name,line,text,value",
    [
        ("LOG2", 1308, "log(dble(2.0))", 2.0),
        ("LOG2", 1324, "log(dble(2.0))", 2.0),
        ("LOG4", 1346, "log(dble(4.0))", 4.0),
    ],
)
def test_dyadic_normalizers_are_exact(name, line, text, value):
    """A dyadic literal is exact in single precision, so its constant is the
    exact double value."""
    assert text in _source_line(REFERENCE, line), f"amica15.f90:{line} moved"
    assert _float32_rounding(repr(value)) == value
    assert getattr(rc, name) == math.log(value)


@pytest.mark.parametrize(
    "literal", ["1.772453851", "2.506628274", "4.132731354", "1.858073988", "1.0e-16"]
)
def test_the_helper_does_not_round_twice(literal):
    """``single_precision_literal`` rounds through float64; for every literal it
    is used on, that lands on the correctly rounded float32 value."""
    assert rc.single_precision_literal(float(literal)) == _float32_rounding(literal)


# --- 2. the sweep, as executable documentation ----------------------------------
_DBLE_LITERAL = re.compile(r"dble\(\s*(-?\d+\.\d*(?:[eE][-+]?\d+)?)\s*\)")
_REAL_LITERAL = re.compile(r"(?<![\w.])(\d+\.\d*(?:[eE][-+]?\d+)?|\d+[eE][-+]?\d+)")


def _code_lines(path: Path):
    """``(line number, code)`` with comments, strings and preprocessor lines
    removed, enough for these two files."""
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if line.lstrip().startswith("#"):
            continue
        code = re.sub(r"'[^']*'|\"[^\"]*\"", "''", line).split("!", 1)[0]
        yield n, code


def _inexact(literal: str) -> bool:
    magnitude = literal.lstrip("-")
    if Fraction(magnitude) == 0:
        return False
    return _float32_rounding(magnitude) != float(magnitude)


def test_every_inexact_single_precision_literal_in_the_reference_is_accounted_for():
    """``amica15.f90`` writes every real constant in its code as ``dble(...)`` of
    a default-kind literal, never a bare or double-precision literal. The ones
    that are not exact in single precision are exactly these."""
    found: Dict[int, str] = {}
    for n, code in _code_lines(REFERENCE):
        stripped = _DBLE_LITERAL.sub("", code)
        assert not _REAL_LITERAL.search(stripped), f"amica15.f90:{n}: {code.strip()}"
        for literal in _DBLE_LITERAL.findall(code):
            if _inexact(literal):
                assert n not in found
                found[n] = literal
    assert found == {
        # Density normalizers: pamica.reference_constants.
        1313: "1.772453851",
        1333: "2.506628274",
        1359: "4.132731354",
        1371: "1.858073988",
        # Initialization scales of the drawn mu, sbeta and A (and A's redraw on
        # a restart). pamica draws from its own generator, so it cannot
        # reproduce the reference's initialization, and these are not adopted.
        756: "0.05",
        771: "0.1",
        814: "0.01",
        1035: "0.01",
        # Starting values of the rejection pass's running max and min.
        2210: "-999999999.0",
        2211: "999999999.0",
    }


def test_every_inexact_default_in_the_reference_header_is_accounted_for():
    """``amica15_header.f90`` initializes double-precision variables with
    default-kind literals. The ones that are not exact in single precision are
    defaults of ``input.param`` keys (read as double when the key is given),
    ``epsdble`` (hard-coded, now shared) or variables nothing reads."""
    found: Dict[str, str] = {}
    for n, code in _code_lines(REFERENCE_HEADER):
        if not code.lstrip().lower().startswith("double precision"):
            continue
        for name, literal in re.findall(
            r"(\w+)\s*=\s*(-?[\d.]+(?:[eE][-+]?\d+)?)", code
        ):
            if _inexact(literal):
                found[name] = literal
    read_from_input_param = {
        "mineig": "1.0e-15",
        "lrate": "0.1",
        "minlrate": "1.0e-12",
        "rholrate": "0.05",
        "rholrate0": "0.05",
        "rholratefact": "0.1",
        "invsigmin": "0.0001",
        "comp_thresh": "0.99",
        "min_dll": "1.0e-9",
        "min_nd": "1.0e-7",
    }
    never_read = {
        "mincond": "1.0e-15",
        "maxdble": "1.0e32",
        "mineigv": "1.0e-15",
        "minhess": "1.0e-5",
    }
    assert found == {**read_from_input_param, **never_read, "epsdble": "1.0e-16"}
    reference = REFERENCE.read_text()
    # variable -> (input.param key, the variable its case reads into)
    reads = {
        "lrate": ("lrate", "lrate0"),  # then lrate = lrate0
        "rholrate": ("rholrate", "rholrate0"),  # then rholrate = rholrate0
        "rholrate0": ("rholrate", "rholrate0"),
        "min_nd": ("min_grad_norm", "min_nd"),
    }
    for name in read_from_input_param:
        key, target = reads.get(name, (name, name))
        case = re.search(rf"case\('{key}'\)(.*?)(?=\n\s*case\()", reference, re.S)
        assert case is not None, key
        assert re.search(rf"read\(tmparg,'\([^)]*\)'\)\s*{target}\b", case.group(1))
        if name != target:
            assert re.search(rf"\b{name} = {target}\b", case.group(1)), name
    for name in never_read:
        uses = [
            n for n, code in _code_lines(REFERENCE) if re.search(rf"\b{name}\b", code)
        ]
        assert not uses, f"{name} is read at amica15.f90:{uses}"


# --- 3. no backend keeps its own copy --------------------------------------------
# Every Python module of the three backend packages, not just their cores (the
# NumPy plotting helper pdf.py once carried its own sqrt(pi)).
_BACKEND_PACKAGES = ("torch_impl", "numpy_impl", "mlx_impl")
_BACKEND_FILES = sorted(
    path for package in _BACKEND_PACKAGES for path in (PACKAGE / package).rglob("*.py")
)
_MOD = r"(?:math|np|numpy|torch|mx)"
_PI = rf"{_MOD}\.pi\b"
_TWO = r"2(?:\.0*)?"
# A constant written into a backend instead of imported: the decimal literals
# (and truncations of them), epsdble, and the normalizers in log or linear form.
_COPIES = re.compile(
    "|".join(
        [
            r"1\.77245|2\.50662|4\.13273|1\.85807",
            r"\b1(?:\.0*)?e-16\b",
            rf"log\(\s*{_PI}\s*\)",  # log(pi)
            rf"log\(\s*{_TWO}\s*\*\s*{_PI}",  # log(2*pi...)
            rf"log\(\s*{_PI}\s*\*\s*{_TWO}\b",  # log(pi*2)
            rf"log\(\s*(?:{_MOD}\.)?sqrt\(",  # log(sqrt(...))
            r"log\(\s*[24](?:\.0*)?\s*\)",  # log(2), log(4.0)
            rf"sqrt\(\s*{_PI}\s*\)",  # sqrt(pi)
            rf"sqrt\(\s*{_TWO}\s*\*\s*{_PI}",  # sqrt(2*pi), sqrt(2 * np.pi * rho)
            rf"sqrt\(\s*{_PI}\s*\*\s*{_TWO}\b",  # sqrt(pi*2)
            rf"{_PI}\s*\*\*\s*0?\.5",  # pi ** 0.5
        ]
    )
)
# Code that the scanner must flag: the definitions this change removed.
_OLD_DEFINITIONS = [
    "_HALF_LOG_PI = 0.5 * math.log(math.pi)",
    "_LOG_SQRT_2PI = math.log(2.506628274)",
    "_LOG_NORM_COSH_SUB = math.log(4.132731354)",
    "_LOG_NORM_COSH_SUP = math.log(1.858073988)",
    "_EPSDBLE = 1e-16",
    "_LOG2 = math.log(2.0)",
    "_LOG4 = math.log(4.0)",
    "log_pdf = -np.abs(y) - np.log(2.0)",
    "log_pdf = -y * y - 0.5 * np.log(np.pi)",
    "logab = np.where(ayrho < 1e-16, 0.0, logab)",
    "pdf = np.exp(-y * y) / np.sqrt(np.pi)",
    "pdf1 = np.exp(-0.5 * y * y) / np.sqrt(2 * np.pi)",
    "pdf2 = np.exp(-0.5 * y * y / rho) / np.sqrt(2 * np.pi * rho)",
]


def _python_code(source: str) -> list[str]:
    """``source``'s lines with comments, strings and docstrings blanked out, so
    prose that names a constant is not mistaken for a copy of it."""
    lines = [list(line) for line in source.splitlines(keepends=True)]
    prose = {tokenize.COMMENT, tokenize.STRING}
    if hasattr(tokenize, "FSTRING_MIDDLE"):
        prose.add(tokenize.FSTRING_MIDDLE)
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type not in prose:
            continue
        (row, col), (end_row, end_col) = tok.start, tok.end
        for r in range(row, end_row + 1):
            line = lines[r - 1]
            stop = end_col if r == end_row else len(line)
            for c in range(col if r == row else 0, stop):
                if line[c] != "\n":
                    line[c] = " "
    return ["".join(line) for line in lines]


@pytest.mark.parametrize("code", _OLD_DEFINITIONS)
def test_the_copy_scanner_flags_the_removed_definitions(code):
    """The scanner below is not vacuous: it flags each definition this change
    removed from the backends."""
    assert _COPIES.search(_python_code(code + "\n")[0])


@pytest.mark.parametrize(
    "path", _BACKEND_FILES, ids=[str(p.relative_to(PACKAGE)) for p in _BACKEND_FILES]
)
def test_no_backend_keeps_a_copy_of_the_constants(path):
    copies = [
        f"{n}: {line.strip()}"
        for n, line in enumerate(_python_code(path.read_text()), 1)
        if _COPIES.search(line)
    ]
    assert not copies, f"{path.relative_to(PACKAGE)} defines its own: {copies}"


@pytest.mark.parametrize("package", _BACKEND_PACKAGES)
def test_every_backend_imports_the_shared_constants(package):
    assert "reference_constants import" in (PACKAGE / package / "core.py").read_text()


# --- 4. the rho == 2 branch uses the reference's normalizer, on every backend ---
_Y = np.linspace(-8.0, 8.0, 65)
_Y = _Y[np.abs(_Y) > 1e-3]


def test_torch_and_numpy_rho2_log_density_is_the_references():
    """The exact-Gaussian branch of the generalized Gaussian (amica15.f90:1313)
    subtracts ``log(dble(1.772453851))``; PyTorch and NumPy now match that
    formula bit for bit, and each other."""
    expected = -_Y * _Y - math.log(_float32_rounding("1.772453851"))
    y = torch.from_numpy(_Y.copy())
    rho = torch.full_like(y, 2.0)
    lp_only, _ = _log_pdf_only(y, rho)
    lp_full, _ = _log_pdf_and_deriv(y, rho)
    np.testing.assert_array_equal(lp_only.numpy(), expected)
    np.testing.assert_array_equal(lp_full.numpy(), expected)
    numpy_model = AMICA_NumPy(use_tqdm=False)
    lp_numpy, _ = numpy_model._compute_log_pdf(_Y, 2.0)
    np.testing.assert_array_equal(lp_numpy, expected)


@pytest.mark.parametrize("rho", [1.0, 1.5, 2.0])
def test_the_plotted_density_is_the_fits(rho):
    """``viz.plot_pdf_fits`` draws ``compute_pdf``, which must be the density
    the fit uses, including the reference's normalizer at ``rho == 2``."""
    fit_log_pdf, _ = AMICA_NumPy(use_tqdm=False)._compute_log_pdf(_Y, rho)
    plotted, _ = compute_pdf(_Y, rho)
    np.testing.assert_allclose(plotted, np.exp(fit_log_pdf), rtol=1e-13, atol=0)


def test_mlx_rho2_normalizer_is_the_references():
    """MLX folds the generalized Gaussian's normalizer into a host-side table,
    ``log2 + table``, with ``table = lgamma(1 + 1/rho)`` (MLX has no lgamma).
    At ``rho == 2`` the table holds the reference's normalizer instead, so the
    float32 log-density is the reference's formula cast to float32."""
    mlx_core = pytest.importorskip(
        "pamica.mlx_impl.core", reason="MLX not installed (Apple Silicon only)"
    )
    mx = mlx_core.mx
    if mx.default_device().type != mx.DeviceType.gpu:
        pytest.skip("no Apple GPU")
    from scipy.special import gammaln

    m = mlx_core.AMICAMLXNG(n_channels=NW, n_mix=3)
    rho = np.array([[1.0], [1.5], [2.0]])
    m.rho = mx.array(rho.astype(np.float32))
    m._refresh_lgamma_table()
    table = np.array(m._lgamma_table)
    assert table.dtype == np.float32
    assert table[0, 0] == 0.0  # lgamma(2): the Laplace branch's log(2) alone
    assert table[1, 0] == np.float32(gammaln(1.0 + 1.0 / 1.5))
    assert table[2, 0] == np.float32(rc.LOG_SQRT_PI - rc.LOG2)
    assert table[2, 0] != np.float32(gammaln(1.5))

    y32 = mx.array(_Y.astype(np.float32))
    rho2 = mx.full(y32.shape, 2.0)
    lp, _ = mlx_core._log_pdf(y32, rho2, mx.full(y32.shape, table[2, 0]))
    expected = -_Y * _Y - math.log(_float32_rounding("1.772453851"))
    np.testing.assert_allclose(np.array(lp, dtype=np.float64), expected, rtol=1e-6)


# --- 5. seeded native reference oracle, per family (opt-in) ----------------------
# The reference's input.param optimizer, natural gradient only, with the block
# size pinned on both sides (the same as test_doscaling_rows.py's oracle).
_OPT: Dict[str, Any] = dict(
    block_size=512,
    lrate=0.05,
    lratefact=0.5,
    rholrate=0.05,
    rholratefact=0.5,
    rho0=1.5,
    minrho=1.0,
    maxrho=2.0,
    invsigmin=0.0,
    invsigmax=100.0,
    do_newton=False,
    newt_start=50,
    newtrate=1.0,
    newt_ramp=10,
)
_REF_OPT: Dict[str, Any] = {
    **{k: v for k, v in _OPT.items() if k != "do_newton"},
    "do_newton": 0,
    "do_opt_block": 0,
    "do_reject": 0,
    "share_comps": 0,
    "doscaling": 1,
    "scalestep": 1,
}
# (pdftype, n_mix, the constant whose rounding the pre-change code missed)
_FAMILIES = [
    (2, 3, "LOG_SQRT_2PI"),
    (4, 1, "LOG_NORM_COSH_SUB"),
    (1, 1, "LOG_NORM_COSH_SUP"),
]
# Per iteration count, the doscaling oracle's round-off bounds. Measured maxima
# over the three families (PyTorch against the binary):
#   1 iteration:  A 9.8e-16, mu 1.5e-14, sbeta 1.4e-14, LL 2.7e-15
#   3 iterations: A 1.9e-15, mu 1.8e-14, sbeta 1.6e-14, LL 2.7e-15
# Before this change the A, mu and sbeta errors were the same and the
# log-likelihood was off by 3.7e-10, 2.0e-8 and -2.1e-8 (pdftype 2, 4 and 1).
_ORACLE_TOL = {
    1: {"A": 1e-13, "mu": 1e-9, "sbeta": 1e-12, "LL": 1e-12},
    3: {"A": 1e-10, "mu": 1e-6, "sbeta": 1e-8, "LL": 1e-11},
}


@pytest.fixture(scope="module")
def pre344(tmp_path_factory) -> Any:
    """The pamica package at ``PRE_CHANGE_COMMIT``, imported beside the live one."""
    return load_pre_change_package(
        PRE_CHANGE_COMMIT, "pamica_pre344", tmp_path_factory.mktemp("pre344")
    )


@pytest.mark.skipif(
    os.environ.get("AMICA_RUN_FORTRAN") != "1",
    reason="opt-in Fortran-binary integration test (set AMICA_RUN_FORTRAN=1)",
)
@pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing")
@pytest.mark.parametrize(
    "pdftype,n_mix,constant", _FAMILIES, ids=[f"pdftype{p}" for p, *_ in _FAMILIES]
)
def test_family_matches_the_seeded_reference(
    pre344, pdftype, n_mix, constant, tmp_path
):
    """Seed the reference with pamica's initialization for one fixed density
    family, run one and three iterations, and compare ``A``/``mu``/``sbeta``
    and the log-likelihood. The family's normalizer shifts every sample's
    log-density by the same amount, so it moves the log-likelihood and nothing
    else: the code before this change is off by exactly the literal's rounding
    in the log-likelihood, and PyTorch now matches to round-off."""
    from pamica.tests.native_oracle import (
        reference_mixing,
        run_seeded_reference,
        seed_from_torch,
    )

    X = load_eeglab_data(str(DATA_FILE), data_dim=NW, field_dim=FIELD)
    X = X.astype(np.float64)
    # The reference never switches a family (its do_choose_pdfs is set at
    # amica15.f90:612-613 and never read), so pdftype=1 holds code 1.
    extra = dict(num_kurt=0) if pdftype == 1 else {}

    def torch_model(cls: Any) -> Any:
        return cls(
            n_channels=NW,
            n_mix=n_mix,
            pdftype=pdftype,
            seed=SEED,
            device="cpu",
            dtype=torch.float64,
            keep_best=False,
            **extra,
            **_OPT,
        )

    init = torch_model(AMICATorchNG)
    init._preprocess(X)
    init._initialize_parameters()
    state = seed_from_torch(init)
    assert init.sphere is not None
    offset = getattr(rc, constant) - _OFFSETS[constant][0]

    for k, tol in _ORACLE_TOL.items():
        ref = run_seeded_reference(
            state,
            DATA_FILE,
            tmp_path / f"k{k}",
            n_samples=FIELD,
            max_iter=k,
            pdftype=pdftype,
            **_REF_OPT,
        )
        family = re.search(r"pdf type =\s*(\d+)", ref.stdout)
        assert family is not None and int(family.group(1)) == pdftype
        assert np.abs(ref.S - init.sphere.numpy()).max() < 1e-12

        t = torch_model(AMICATorchNG)
        t.fit(X, max_iter=k, verbose=False)
        assert t.A is not None and t.mu is not None and t.beta is not None
        errs = {
            "A": np.abs(reference_mixing(t.A.numpy()) - ref.A).max(),
            "mu": np.abs(t.mu.numpy() - ref.mu).max(),
            "sbeta": np.abs(t.beta.numpy() - ref.sbeta).max(),
            "LL": np.abs(np.asarray(t.ll_history) - ref.LL).max(),
        }
        print(f"pdftype={pdftype} k={k}: {errs}")
        over = {q: e for q, e in errs.items() if not e <= tol[q]}
        assert not over, f"pdftype={pdftype}, {k} iteration(s): {over} (bounds {tol})"

        # The pre-change code: the same trajectory, its log-likelihood shifted
        # by the literal's rounding (pamica's log-density was higher by it).
        old = torch_model(pre344.torch_impl.core.AMICATorchNG)
        old.fit(X, max_iter=k, verbose=False)
        shift = np.asarray(old.ll_history) - ref.LL
        print(f"pdftype={pdftype} k={k} pre-change: LL - reference = {shift}")
        assert np.abs(shift - offset).max() <= tol["LL"]
        assert abs(offset) > 10 * tol["LL"]
