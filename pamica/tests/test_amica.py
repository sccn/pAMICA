"""Tests for AMICA implementation."""

import shutil
import tempfile
import numpy as np
from pathlib import Path
import unittest

from pamica.numpy_impl.data import load_data_file, preprocess_data
from pamica.numpy_impl.pdf import compute_pdf


class TestAMICA(unittest.TestCase):
    """Test AMICA implementation against Fortran code."""

    @classmethod
    def setUpClass(cls):
        """Create test data."""
        # Create random test data
        rng = np.random.RandomState(42)
        cls.data_dim = 64
        cls.num_samples = 1000
        cls.data = rng.randn(cls.data_dim, cls.num_samples)

        # Save test data in Fortran format. Use a unique temp dir (not a shared
        # relative "test_data/") so parallel workers (pytest-xdist -n auto) do
        # not race on creating/removing the same path.
        cls.test_dir = Path(tempfile.mkdtemp(prefix="pyamica_test_amica_"))

        data_file = cls.test_dir / "test.bin"
        with open(data_file, "wb") as f:
            cls.data.T.astype(np.float32).tofile(f)

    def test_data_loading(self):
        """Test data loading matches Fortran."""
        data = load_data_file(
            self.test_dir / "test.bin",
            self.data_dim,
            self.num_samples,
            dtype=np.float32,
        )
        np.testing.assert_allclose(data, self.data)

    def test_preprocessing(self):
        """Test preprocessing matches Fortran."""
        # Test mean removal
        data, mean, _ = preprocess_data(self.data.copy(), do_mean=True, do_sphere=False)
        np.testing.assert_allclose(mean.ravel(), self.data.mean(axis=1))
        np.testing.assert_allclose(
            data.mean(axis=1), np.zeros(self.data_dim), atol=1e-10
        )

        # Test sphering
        data, _, sphere = preprocess_data(
            self.data.copy(), do_mean=False, do_sphere=True
        )
        cov = np.cov(data)
        np.testing.assert_allclose(cov, np.eye(self.data_dim), atol=1e-10)

    def test_pdf_computation(self):
        """Test PDF computation matches Fortran."""
        # Test Laplace distribution
        y = np.linspace(-5, 5, 100)
        pdf, dpdf = compute_pdf(y, rho=1.0)
        np.testing.assert_allclose(pdf, np.exp(-np.abs(y)) / 2.0)
        np.testing.assert_allclose(dpdf, -np.sign(y) * pdf)

        # Test Gaussian distribution
        pdf, dpdf = compute_pdf(y, rho=2.0)
        np.testing.assert_allclose(pdf, np.exp(-y * y) / np.sqrt(np.pi))
        np.testing.assert_allclose(dpdf, -2 * y * pdf)

    # NOTE: the former synthetic-data source-recovery test (test_full_amica) was
    # removed: it fabricated data (against the NO-MOCK policy) and used a broken
    # metric (corrcoef on raveled sources, no permutation/sign/scale matching, so
    # it could never pass). Real-data NumPy-vs-Fortran parity is covered by
    # tests/test_sample_data.py::test_sample_data_numpy_vs_fortran.
    #
    # NOTE: the former test_newton_direction test (issue #270) pinned the
    # standalone numpy_impl/newton.py module, which was dead pre-#21/#24 math:
    # compute_newton_parameters took the density derivative dpdf instead of the
    # score and omitted the sbeta^2 factor on kappa and the mu^2 fold in lambda,
    # and update_unmixing_matrix used the += right-multiply form that was the
    # issue #24 root cause. That module was never imported by the live numpy
    # Newton path (inlined in numpy_impl/core.py) and was removed in #270. Its
    # behavior is superseded by the live-path Newton coverage in
    # tests/torch_tests/test_ng_backend.py and
    # tests/test_numpy_newton_multimodel.py (#267).

    @classmethod
    def tearDownClass(cls):
        """Clean up test files (ignore_errors: the temp dir may already be gone)."""
        shutil.rmtree(cls.test_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
