"""Rank-deficient input handling (issue #223) for AMICATorchNG.

Ports the Fortran reference's numerical-rank machinery, which pamica had
dropped: ``numeigs = min(pcakeep, count(eigs > mineig))`` (amica15.f90:413),
``nw = numeigs`` sizing the model to the kept rank (amica15.f90:563), and the
sphere pseudo-inverse ``Spinv`` for mapping components back to sensor space
(amica15.f90:568-578).

Rank deficiency is produced by projecting the real sample EEG onto its own
leading subspace -- the same rank-reducing linear operator Maxwell filtering
applies to MEG. The data stay real throughout; only the rank is reduced.

Full-rank behavior must be byte-for-byte unchanged, since ``mineig`` keeps every
eigenvalue of well-conditioned data.
"""

from pathlib import Path

import numpy as np
import pytest

from pamica.amica import AMICA
from pamica.torch_impl.utils import load_eeglab_data

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "sample_data"
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
    """Real EEG projected onto its top-``RANK`` subspace (what SSS does to MEG)."""
    U = np.linalg.svd(real_data, full_matrices=False)[0][:, :RANK]
    return U @ (U.T @ real_data)


def test_full_rank_model_dimension_unchanged(real_data: np.ndarray) -> None:
    """mineig keeps every eigenvalue of full-rank data, so nothing is reduced."""
    m = AMICA(verbose=False)
    m.fit(real_data, max_iter=5, seed=0)
    assert m.model_ is not None
    assert m.model_.sphere is not None
    assert m.model_.n_channels == NW
    assert m.model_.n_channels_in == NW
    assert tuple(m.model_.sphere.shape) == (NW, NW)


def test_rank_deficient_fits_instead_of_going_degenerate(
    rank_deficient: np.ndarray,
) -> None:
    """Before #223 this stopped at ``nan_ll`` on iteration 0.

    The symmetric ZCA sphere inverts ``sqrt(lambda ~ 0)`` in the null
    directions; detecting the rank first avoids ever forming those rows.
    """
    m = AMICA(verbose=False)
    m.fit(rank_deficient, max_iter=15, seed=0)
    assert m.model_ is not None
    assert m.converged_, f"degenerate fit: {m.stop_reason_}"
    assert m.stop_reason_ != "nan_ll"
    # Model sized to the kept rank, not the input channel count.
    assert m.model_.sphere is not None
    assert m.model_.n_channels < NW
    assert m.model_.n_channels_in == NW
    assert tuple(m.model_.sphere.shape) == (m.model_.n_channels, NW)


def test_default_relative_threshold_recovers_the_exact_rank(
    rank_deficient: np.ndarray,
) -> None:
    """The default scale-free floor finds the true rank exactly (ADR 0004)."""
    m = AMICA(verbose=False)
    m.fit(rank_deficient, max_iter=10, seed=0)
    assert m.model_ is not None
    assert m.model_.n_channels == RANK


def test_fortran_absolute_floor_over_retains(rank_deficient: np.ndarray) -> None:
    """``mineig_rel=None`` restores Fortran's absolute floor, warts included.

    That floor sits amid the numerical-zero eigenvalues (~1e-16 to 1e-14), so it
    keeps directions that carry no signal. Pinned here because it is the
    documented reason pamica's default diverges from the reference.
    """
    m = AMICA(verbose=False)
    m.fit(rank_deficient, max_iter=10, seed=0, mineig_rel=None)
    assert m.model_ is not None
    assert m.model_.n_channels > RANK


def test_sensor_mixing_matrix_reconstructs_the_input(
    rank_deficient: np.ndarray,
) -> None:
    """``pinv(sphere) @ A`` maps the decomposition back to input channels."""
    m = AMICA(verbose=False)
    m.fit(rank_deficient, max_iter=15, seed=0)
    assert m.model_ is not None
    A = m.model_.get_sensor_mixing_matrix()
    S = m.transform(rank_deficient)
    assert A.shape == (NW, RANK)
    assert S.shape == (RANK, rank_deficient.shape[1])

    assert m.model_.mean is not None
    mean = m.model_.mean.cpu().numpy()
    recon = A @ S + mean
    rel = np.abs(rank_deficient - recon).max() / np.abs(rank_deficient).max()
    assert rel < 1e-10, f"reconstruction error {rel:.3e}"


def test_pcakeep_with_pca_whitening_no_longer_crashes(
    rank_deficient: np.ndarray,
) -> None:
    """Regression: ``pcakeep`` + ``do_approx_sphere=False`` raised a shape error.

    ``_preprocess`` returned ``(pcakeep, T)`` while every parameter was still
    sized to the input channel count.
    """
    m = AMICA(verbose=False)
    m.fit(rank_deficient, max_iter=5, seed=0, pcakeep=RANK, do_approx_sphere=False)
    assert m.model_ is not None
    assert m.model_.sphere is not None
    assert m.model_.n_channels == RANK
    assert tuple(m.model_.sphere.shape) == (RANK, NW)


def test_tesla_scale_data_fits_by_default(real_data: np.ndarray) -> None:
    """MEG-magnitude input works without rescaling under the default floor."""
    m = AMICA(verbose=False)
    m.fit(real_data * 1e-13, max_iter=5, seed=0)
    assert m.model_ is not None
    assert m.converged_
    assert m.model_.n_channels == NW


def test_absolute_floor_rejects_tesla_scale_data(real_data: np.ndarray) -> None:
    """Under Fortran's absolute floor every Tesla-scale eigenvalue is "zero".

    Fortran computes ``numeigs = 0`` and proceeds; we refuse with a message
    naming the cause and the fix. This is the failure mode the default relative
    floor exists to avoid.
    """
    with pytest.raises(ValueError, match="numerical rank is zero"):
        AMICA(verbose=False).fit(real_data * 1e-13, max_iter=5, seed=0, mineig_rel=None)


def test_rank_reduced_newton_fit_completes(rank_deficient: np.ndarray) -> None:
    """Newton on rank-reduced data (issue #273).

    Newton's curvature and 2x2 direction solve are per-(source, model), sized
    to the model's working dimensionality -- which rank reduction shrinks from
    ``n_channels_in`` to the detected rank (issue #223) before any Newton array
    is allocated. Nothing in ``test_newton_multimodel_finite_and_shaped``
    (``test_ng_backend.py``) or the Newton tests above exercises that combination,
    and it is exactly the Maxwell-filtered MEG route #221/#223 motivated.
    """
    m = AMICA(verbose=False)
    m.fit(rank_deficient, max_iter=15, seed=0, do_newton=True, newt_start=3)
    assert m.model_ is not None
    assert m.converged_, f"degenerate fit: {m.stop_reason_}"
    assert m.stop_reason_ != "nan_ll"
    assert m.model_.n_channels == RANK
    assert m.model_.n_channels_in == NW
    assert m.model_.A is not None
    assert np.all(np.isfinite(m.model_.A.cpu().numpy()))
    assert np.all(np.isfinite(m.get_unmixing_matrix()))


def test_nan_data_still_reaches_the_degenerate_fit_contract(
    real_data: np.ndarray,
) -> None:
    """Rank detection must not intercept unusable data (issue #50).

    A NaN makes every covariance eigenvalue non-finite, and ``nan > thresh`` is
    False, which would look like rank zero. That must not become a ``ValueError``
    from preprocessing: the documented behavior is a ``nan_ll`` stop with the
    model marked unusable.
    """
    bad = real_data[:, :4096].copy()
    bad[0, 0] = np.nan
    m = AMICA(n_models=1, device="cpu", verbose=False)
    m.fit(bad, max_iter=3, block_size=1024, seed=0)
    assert m.stop_reason_ == "nan_ll"
    assert m.converged_ is False
    assert m.is_fitted_ is False
