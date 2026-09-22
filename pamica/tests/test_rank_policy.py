"""Shared numerical-rank policy (issue #223, ADR 0004).

Rank detection is one decision used by three backends, so it lives in
``pamica.rank`` and is tested once here. The cross-backend test at the bottom is
the anti-drift guard required by ``.rules/backend_parity.md``: it fails if any
backend starts answering "how many dimensions are real?" differently.

The explicit ``pcakeep``/``pcadb`` request is part of the same decision (issue
#323): its validator and its "is a reduction requested?" predicate are unit-tested
here, and ``test_pca_reduction_cross_backend.py`` checks that every backend
applies them identically.
"""

import logging
import math
from pathlib import Path

import numpy as np
import pytest

from pamica.rank import (
    MINEIG,
    MINEIG_REL,
    log_ignored_pca_request,
    numerical_rank,
    pca_reduction_requested,
    validate_pca_reduction,
)
from pamica.torch_impl.utils import load_eeglab_data

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_data"
DATA_FILE = SAMPLE_DIR / "eeglab_data.fdt"
NW = 32
FIELD = 30504
RANK = 20


@pytest.fixture(scope="module")
def real_data() -> np.ndarray:
    if not DATA_FILE.exists():
        pytest.skip("sample data missing")
    X = load_eeglab_data(str(DATA_FILE), data_dim=NW, field_dim=FIELD).astype(
        np.float64
    )
    return X - X.mean(axis=1, keepdims=True)


@pytest.fixture(scope="module")
def rank_deficient(real_data: np.ndarray) -> np.ndarray:
    U = np.linalg.svd(real_data, full_matrices=False)[0][:, :RANK]
    return U @ (U.T @ real_data)


def _cov_eigenvalues(X: np.ndarray) -> np.ndarray:
    return np.linalg.eigvalsh(np.cov(X, bias=True))[::-1]


def test_full_rank_data_keeps_every_dimension(real_data: np.ndarray) -> None:
    ev = _cov_eigenvalues(real_data)
    # Real EEG is well conditioned (lambda_min/lambda_max ~ 5e-4), far above
    # either floor, so the default cannot silently reduce it.
    assert ev[-1] / ev[0] > 1e-6
    assert numerical_rank(ev) == NW
    assert numerical_rank(ev, mineig_rel=None) == NW


def test_relative_floor_finds_the_true_rank(rank_deficient: np.ndarray) -> None:
    assert numerical_rank(_cov_eigenvalues(rank_deficient)) == RANK


def test_absolute_floor_over_retains(rank_deficient: np.ndarray) -> None:
    """Fortran's absolute floor lands amid the numerical-zero eigenvalues."""
    assert numerical_rank(_cov_eigenvalues(rank_deficient), mineig_rel=None) > RANK


def test_average_reference_costs_exactly_one_dimension(
    real_data: np.ndarray,
) -> None:
    """Average referencing is rank-deficient by construction.

    This is the everyday EEG case, not an exotic one: the relative floor detects
    it, while the absolute floor sits within a factor of ~10 of the resulting
    eigenvalue and so decides it by luck.
    """
    avg_ref = real_data - real_data.mean(axis=0, keepdims=True)
    assert numerical_rank(_cov_eigenvalues(avg_ref)) == NW - 1


def test_scale_invariance_of_the_relative_floor(real_data: np.ndarray) -> None:
    """Rescaling the data must not change how many dimensions are real."""
    for scale in (1e-13, 1.0, 1e6):
        assert numerical_rank(_cov_eigenvalues(real_data * scale)) == NW


def test_absolute_floor_is_not_scale_invariant(real_data: np.ndarray) -> None:
    """The documented reason the default diverges from Fortran."""
    with pytest.raises(ValueError, match="numerical rank is zero"):
        numerical_rank(_cov_eigenvalues(real_data * 1e-13), mineig_rel=None)


def test_pcakeep_is_capped_by_the_detected_rank(
    rank_deficient: np.ndarray,
) -> None:
    ev = _cov_eigenvalues(rank_deficient)
    assert numerical_rank(ev, pcakeep=10) == 10
    # Fortran's min(): asking for more than exists still yields the real rank.
    assert numerical_rank(ev, pcakeep=NW) == RANK


def test_non_finite_eigenvalues_keep_full_dimension() -> None:
    """NaN data must reach the degenerate-fit contract, not raise here (#50)."""
    assert numerical_rank(np.array([np.nan, np.nan, np.nan])) == 3


def test_defaults_match_the_documented_constants() -> None:
    assert MINEIG == 1e-15  # Fortran amica15_header.f90:66
    assert MINEIG_REL == 1e-12


# --- explicit pcakeep/pcadb policy (issue #323) -------------------------------
@pytest.mark.parametrize(
    ("pcakeep", "pcadb"),
    [
        (None, None),
        (1, None),
        (20, None),
        (np.int64(5), None),  # numpy integers are numbers.Integral
        (None, 3.0),
        (None, 30),  # an int is a real number of dB
        (None, np.float32(3.0)),  # numpy floats are numbers.Real
        (None, np.float64(30.0)),
        (32, 30.0),  # both bundled param files set both
    ],
)
def test_validator_accepts(pcakeep, pcadb) -> None:
    validate_pca_reduction(pcakeep, pcadb)


@pytest.mark.parametrize(
    "pcakeep", [True, False, np.bool_(True), np.bool_(False), 0, -3, 2.7, "20"]
)
def test_validator_rejects_pcakeep(pcakeep) -> None:
    """``True``/``False`` are ints to Python but never a dimension count;
    ``-3`` used to slice from the end (29 of 32 kept) and ``2.7`` to truncate."""
    with pytest.raises(ValueError, match="pcakeep") as info:
        validate_pca_reduction(pcakeep, None)
    assert repr(pcakeep) in str(info.value)


@pytest.mark.parametrize(
    "pcadb", [0, -5, math.nan, math.inf, True, False, np.bool_(True), np.bool_(False)]
)
def test_validator_rejects_pcadb(pcadb) -> None:
    with pytest.raises(ValueError, match="pcadb") as info:
        validate_pca_reduction(None, pcadb)
    assert repr(pcadb) in str(info.value)


def test_validator_rejects_either_parameter_when_both_are_set() -> None:
    """A valid partner does not rescue an invalid value: pcakeep's precedence
    is about which one sizes the model, not about skipping validation."""
    with pytest.raises(ValueError, match="pcadb"):
        validate_pca_reduction(20, -5.0)
    with pytest.raises(ValueError, match="pcakeep"):
        validate_pca_reduction(0, 30.0)


def test_validator_never_logs(caplog) -> None:
    """Validation only, so the fit-time re-checks cannot repeat the
    constructor's notes."""
    with caplog.at_level(logging.DEBUG, logger="pamica.rank"):
        validate_pca_reduction(32, 30.0)
        validate_pca_reduction(20, None)
    assert not caplog.records


def test_ignored_request_logs_the_precedence_once(caplog) -> None:
    with caplog.at_level(logging.INFO, logger="pamica.rank"):
        log_ignored_pca_request(32, 30.0, True)
        log_ignored_pca_request(20, None, True)
        log_ignored_pca_request(None, 30.0, True)
        log_ignored_pca_request(None, None, True)
    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.INFO
    assert "pcadb is ignored" in caplog.records[0].getMessage()


@pytest.mark.parametrize(
    ("pcakeep", "pcadb"), [(20, None), (None, 30.0), (20, 30.0), (np.int64(20), None)]
)
def test_ignored_request_warns_once_without_sphering(caplog, pcakeep, pcadb) -> None:
    """Without sphering nothing is reduced (the reference's no-sphere branch
    sets numeigs = nx, amica15.f90:527): one WARNING, and no precedence note,
    since pcadb's precedence is moot when neither is used."""
    with caplog.at_level(logging.INFO, logger="pamica.rank"):
        log_ignored_pca_request(pcakeep, pcadb, False)
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelno == logging.WARNING
    assert "do_sphere=False" in record.getMessage()


def test_nothing_to_ignore_without_a_request(caplog) -> None:
    with caplog.at_level(logging.DEBUG, logger="pamica.rank"):
        log_ignored_pca_request(None, None, False)
    assert not caplog.records


@pytest.mark.parametrize(
    ("pcakeep", "pcadb", "n_channels", "do_sphere", "expected"),
    [
        (None, None, 32, True, False),
        (20, None, 32, True, True),
        (31, None, 32, True, True),
        (32, None, 32, True, False),  # the bundled input.param: nothing is reduced
        (64, None, 32, True, False),
        (None, 30.0, 32, True, True),  # unknowable before the eigenvalues exist
        (32, 30.0, 32, True, False),  # pcakeep wins, as in numerical_rank
        (20, 30.0, 32, True, True),
        # Without sphering no backend reduces (amica15.f90:527: numeigs = nx).
        (20, None, 32, False, False),
        (None, 30.0, 32, False, False),
        (20, 30.0, 32, False, False),
        (None, None, 32, False, False),
    ],
)
def test_reduction_predicate_truth_table(
    pcakeep, pcadb, n_channels, do_sphere, expected
):
    assert pca_reduction_requested(pcakeep, pcadb, n_channels, do_sphere) is expected


@pytest.mark.parametrize(
    ("pcakeep", "pcadb", "name"),
    [(-3, None, "pcakeep"), ("20", None, "pcakeep"), (None, -5.0, "pcadb")],
)
def test_reduction_predicate_validates_first(pcakeep, pcadb, name) -> None:
    """The gate runs at fit time, so a reassigned attribute must raise the
    validator's ValueError there, not a TypeError from comparing a string."""
    with pytest.raises(ValueError, match=name):
        pca_reduction_requested(pcakeep, pcadb, 32, True)


@pytest.mark.parametrize(
    ("pcakeep", "pcadb"),
    [(-3, None), (2.7, None), (0, None), (True, None), (None, 0.0), (None, -5.0)],
)
def test_numerical_rank_rejects_invalid_reduction(
    real_data: np.ndarray, pcakeep, pcadb
) -> None:
    """The fit-time choke point: an attribute reassigned after construction
    reaches numerical_rank unvalidated by any constructor."""
    with pytest.raises(ValueError, match="pcakeep" if pcadb is None else "pcadb"):
        numerical_rank(_cov_eigenvalues(real_data), pcakeep=pcakeep, pcadb=pcadb)


def test_numerical_rank_pcakeep_wins_over_pcadb(real_data: np.ndarray, caplog):
    """pcadb=30 alone cuts the real sample to 26 dimensions; with pcakeep set
    it is ignored, and the fit-time re-check does not repeat the INFO note."""
    ev = _cov_eigenvalues(real_data)
    assert numerical_rank(ev, pcadb=30.0) == 26
    with caplog.at_level(logging.INFO, logger="pamica.rank"):
        assert numerical_rank(ev, pcakeep=20, pcadb=30.0) == 20
        assert numerical_rank(ev, pcakeep=NW, pcadb=30.0) == NW
    assert not caplog.records


def test_all_backends_agree_on_the_rank(rank_deficient: np.ndarray) -> None:
    """Anti-drift guard: every backend must size its model identically.

    MLX is skipped when unavailable (Apple Silicon only), but PyTorch and NumPy
    always run, so a divergence between the two cannot land.
    """
    from pamica import AMICA, AMICA_NumPy

    torch_model = AMICA(verbose=False)
    torch_model.fit(rank_deficient, max_iter=3, seed=0)
    assert torch_model.model_ is not None
    assert torch_model.model_.n_channels == RANK

    numpy_model = AMICA_NumPy(num_models=1, max_iter=3, use_tqdm=False)
    numpy_model.fit(rank_deficient)
    assert numpy_model.data_dim == RANK
    assert numpy_model.data_dim_in == NW

    mlx_core = pytest.importorskip(
        "pamica.mlx_impl.core", reason="MLX not installed (Apple Silicon only)"
    )
    mlx_model = mlx_core.AMICAMLXNG(n_channels=NW, seed=0)
    mlx_model.fit(rank_deficient.astype(np.float32), max_iter=3)
    assert mlx_model.n_channels == RANK
