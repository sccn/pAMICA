import os.path as op
import numpy as np
import numpy.testing as npt
import pytest
from pathlib import Path
import tempfile
import shutil

import pamica
from pamica import AMICA_NumPy as AMICA
from pamica.numpy_impl.data import load_data_file, preprocess_data
from pamica.numpy_impl.pdf import compute_pdf

# Setup test data path
data_path = op.join(pamica.__path__[0], "data")


@pytest.fixture
def random_data():
    """Generate random test data."""
    rng = np.random.RandomState(42)
    n_channels = 64
    n_samples = 1000
    return rng.randn(n_channels, n_samples)


@pytest.fixture
def temp_dir():
    """Create and clean up a temporary directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


def test_amica_initialization():
    """Test AMICA model initialization with various parameters."""
    # Test default initialization
    model = AMICA()
    assert model.num_models == 1
    assert model.max_iter == 2000

    # Test custom parameters
    model = AMICA(num_models=2, max_iter=500, do_newton=True)
    assert model.num_models == 2
    assert model.max_iter == 500
    assert model.do_newton is True

    # Test invalid parameters
    with pytest.raises(ValueError):
        AMICA(num_models=0)
    with pytest.raises(ValueError):
        AMICA(max_iter=-1)


def test_data_preprocessing(random_data):
    """Test data preprocessing functionality."""
    # Test mean removal
    data, mean, _ = preprocess_data(random_data.copy(), do_mean=True, do_sphere=False)
    npt.assert_allclose(data.mean(axis=1), np.zeros(data.shape[0]), atol=1e-10)
    npt.assert_allclose(mean.ravel(), random_data.mean(axis=1))

    # Test sphering
    data, _, sphere = preprocess_data(random_data.copy(), do_mean=False, do_sphere=True)
    cov = np.cov(data)
    npt.assert_allclose(cov, np.eye(data.shape[0]), atol=1e-10)


def test_pdf_computation():
    """Test PDF computation for different distributions."""
    y = np.linspace(-5, 5, 100)

    # Test Laplace distribution
    pdf, dpdf = compute_pdf(y, rho=1.0)
    npt.assert_allclose(pdf, np.exp(-np.abs(y)) / 2.0)
    npt.assert_allclose(dpdf, -np.sign(y) * pdf)

    # Test Gaussian distribution
    pdf, dpdf = compute_pdf(y, rho=2.0)
    npt.assert_allclose(pdf, np.exp(-y * y) / np.sqrt(np.pi))
    npt.assert_allclose(dpdf, -2 * y * pdf)


def test_pdf_is_a_normalized_density_for_general_rho():
    """`compute_pdf` must return an actual density for every rho, not just the
    special-cased rho=1 (Laplace) and rho=2 (Gaussian).

    The generalized Gaussian p(y) = exp(-|y|^rho) / (2*Gamma(1+1/rho)) integrates
    to 1 by construction, so integrating the returned values is an oracle-free
    check that holds for any rho. This is a property, not a restatement of the
    formula: asserting `pdf == <the same expression the code uses>` cannot catch
    a wrong expression, which is exactly how a `gammaln`-for-`gamma` transcription
    bug survived here (it made the "density" negative, integrating to -8.82 at the
    default rho0=1.5, while rho=1 and rho=2 stayed correct because they are
    special-cased and were the only values tested).
    """
    y = np.linspace(-60.0, 60.0, 400001)
    for rho in (1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0):
        pdf, _ = compute_pdf(y, rho=rho)
        assert np.all(pdf >= 0.0), f"rho={rho}: density has negative values"
        integral = np.trapezoid(pdf, y)
        npt.assert_allclose(integral, 1.0, rtol=1e-6, err_msg=f"rho={rho}")


def test_pdf_general_branch_is_continuous_with_the_special_cases():
    """The general-rho branch must agree with the rho=1/rho=2 special cases in
    the limit, since they are the same density.

    `compute_pdf` dispatches on exact equality (`rho == 1.0`, `rho == 2.0`), so
    a nudge of 1e-9 takes the identical distribution down the general code path.
    Any disagreement means the two branches do not describe the same density.
    """
    y = np.linspace(-8.0, 8.0, 2001)
    for rho in (1.0, 2.0):
        special_case, _ = compute_pdf(y, rho=rho)
        general, _ = compute_pdf(y, rho=rho + 1e-9)
        npt.assert_allclose(general, special_case, rtol=1e-6, atol=1e-12)


def test_data_loading(temp_dir):
    """Test data loading functionality."""
    # Create test data
    data = np.random.randn(10, 100).astype(np.float32)
    data_file = Path(temp_dir) / "test.bin"

    # Save in Fortran format
    with open(data_file, "wb") as f:
        data.T.tofile(f)

    # Test loading
    loaded_data = load_data_file(data_file, 10, 100, dtype=np.float32)
    npt.assert_allclose(loaded_data, data)

    # Test error handling
    with pytest.raises(FileNotFoundError):
        load_data_file(Path(temp_dir) / "nonexistent.bin", 10, 100)


def test_error_handling():
    """Test error handling in various scenarios."""
    model = AMICA()

    # Test fitting with invalid data
    with pytest.raises(ValueError):
        model.fit(np.array([]))  # Empty array

    with pytest.raises(ValueError):
        model.fit(np.ones((10,)))  # 1D array

    # __init__ already rejects a non-positive max_iter (see
    # test_amica_initialization above), but max_iter is a plain public
    # attribute a caller can reassign after construction; fit() must
    # re-check it too rather than silently running the EM loop zero times
    # (PR #318 review item 5).
    bad_max_iter = AMICA(num_models=1, num_mix=3)
    bad_max_iter.max_iter = 0
    with pytest.raises(ValueError, match="max_iter"):
        bad_max_iter.fit(np.ones((4, 20)))

    # Test transform before fit
    with pytest.raises(RuntimeError):
        model.transform(np.random.randn(10, 100))
