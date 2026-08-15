"""Shared numerical-rank policy (issue #223, ADR 0004).

Rank detection is one decision used by three backends, so it lives in
``pamica.rank`` and is tested once here. The cross-backend test at the bottom is
the anti-drift guard required by ``.rules/backend_parity.md``: it fails if any
backend starts answering "how many dimensions are real?" differently.
"""

from pathlib import Path

import numpy as np
import pytest

from pamica.rank import MINEIG, MINEIG_REL, numerical_rank
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
