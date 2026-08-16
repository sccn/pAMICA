"""
AMICA (Adaptive Mixture ICA) Implementation
=========================================

This module implements the Adaptive Mixture Independent Component Analysis (AMICA)
algorithm, which performs blind source separation using a mixture of adaptive
independent component analyzers.

Key Features
-----------
* Multiple Source Models: Can learn different mixing models for different parts of the data
* Flexible PDFs: Supports various source distributions including Gaussian, Laplace, and mixtures
* Component Sharing: Automatically identifies and shares similar components across models
* Outlier Rejection: Robust estimation by identifying and excluding outlier samples
* Optimization: Efficient parameter updates using natural gradient and Newton methods
* Preprocessing: Automatic mean removal and data sphering

Mathematical Background
--------------------
AMICA extends traditional ICA by:

1. Using mixture models for source PDFs:
   p(s) = Σ_k α_k p_k(s)
   where α_k are mixture weights and p_k are component PDFs

2. Learning multiple mixing models:
   x = A_m s + c_m
   where m indexes different models and c_m are bias terms

3. Optimizing model parameters via maximum likelihood:
   L = Σ_t log p(x_t)
   where p(x_t) includes all mixture components and models

Usage Example
------------
>>> import numpy as np
>>> from pamica import AMICA
>>>
>>> # Generate random data
>>> X = np.random.randn(64, 1000)  # 64 channels, 1000 samples
>>>
>>> # Initialize and fit model
>>> model = AMICA(num_models=2)  # Use 2 mixing models
>>> model.fit(X)
>>>
>>> # Get separated sources
>>> S = model.transform(X)

The algorithm automatically:
- Removes data mean if requested
- Spheres the data if requested
- Initializes model parameters
- Optimizes parameters using natural gradient
- Switches to Newton optimization if requested
- Identifies shared components across models
- Rejects outliers if requested

See Also
--------
pdf : PDF implementations
utils : Utility functions
viz : Visualization tools
cli : Command-line interface

References
----------
1. Palmer, J. A., et al. "Newton Method for the ICA Mixture Model."
   ICASSP 2008.
2. Palmer, J. A., et al. "AMICA: An Adaptive Mixture of Independent
   Component Analyzers with Shared Components." 2012.
"""

import numpy as np
from scipy import linalg
from scipy.special import digamma
import logging
import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
from tqdm import tqdm
from ..rank import MINEIG, MINEIG_REL, numerical_rank
from .utils import (
    gammaln,
    determine_block_size,
    identify_shared_components,
    get_unmixing_matrices,
)


def load_default_params(params_file: Optional[Union[str, Path]] = None) -> Dict:
    """
    Load default parameters from JSON file.

    Parameters
    ----------
    params_file : str, optional
        Path to JSON parameter file. If None, uses default params.json

    Returns
    -------
    params : dict
        Dictionary of default parameters
    """
    if params_file is None:
        params_file = Path(__file__).parent / "params.json"

    with open(params_file) as f:
        params = json.load(f)

    # Remove data-specific parameters
    data_params = {"files", "num_samples", "data_dim", "field_dim"}
    return {k: v for k, v in params.items() if k not in data_params}


class AMICA:
    """
    Adaptive Mixture ICA (AMICA) implementation.

    This class implements the AMICA algorithm for blind source separation using
    adaptive mixtures of independent component analyzers.

    The algorithm provides two progress reporting modes:
    1. A modern tqdm progress bar (default) showing overall progress and key metrics
    2. Detailed per-line progress output in the style of the original Fortran implementation
       (enabled with verbose=True or use_tqdm=False)
    """

    def __init__(
        self,
        params_file: Optional[str] = None,
        use_tqdm: bool = True,
        verbose: bool = False,
        **kwargs,
    ):
        """
        Initialize AMICA with parameters.

        Parameters
        ----------
        params_file : str, optional
            Path to JSON parameter file with default values
        use_tqdm : bool, default=True
            Whether to use tqdm progress bar (False will use per-line printing)
        verbose : bool, default=False
            Whether to enable verbose output (will use per-line printing regardless of use_tqdm)
        **kwargs : dict
            Override default parameters with these values
        """
        # Store progress bar settings
        self.use_tqdm = use_tqdm
        self.verbose = verbose
        # Load default parameters
        params = load_default_params(params_file)

        # Override with any provided parameters
        params.update(kwargs)

        # Store parameters
        self.num_models = params.get("num_models", 1)
        if self.num_models < 1:
            raise ValueError(f"num_models must be >= 1, got {self.num_models}")
        self.num_mix = params.get("num_mix", 3)
        self.max_iter = params.get("max_iter", 2000)
        if self.max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {self.max_iter}")
        self.do_newton = params.get("do_newton", False)
        self.newt_start = params.get("newt_start", 20)
        self.newt_ramp = params.get("newt_ramp", 10)
        self.newtrate = params.get("newtrate", 0.5)
        self.do_reject = params.get("do_reject", False)
        self.rejsig = params.get("rejsig", 3.0)
        self.rejstart = params.get("rejstart", 2)
        self.rejint = params.get("rejint", 3)
        self.maxrej = params.get("maxrej", 1)
        if self.do_reject:
            # Validate up front, matching AMICATorchNG, rather than letting a bad
            # value produce a nonsensical run deep in the EM loop: rejint<1 would
            # be a ZeroDivisionError in the reject schedule; rejsig<=0 breaks the
            # reject-below-the-mean semantics (at 0 the threshold is the mean, so
            # ~half the samples drop every pass, and negative values invert it);
            # maxrej<0 is a sanity guard (it would just make rejection inert via
            # the maxrej>0 schedule gate); rejstart<0 is nonsensical.
            if self.rejint < 1:
                raise ValueError(f"rejint must be >= 1, got {self.rejint}")
            if self.rejsig <= 0:
                raise ValueError(f"rejsig must be > 0, got {self.rejsig}")
            if self.maxrej < 0:
                raise ValueError(f"maxrej must be >= 0, got {self.maxrej}")
            if self.rejstart < 0:
                raise ValueError(f"rejstart must be >= 0, got {self.rejstart}")
        self.num_comps = params.get("num_comps", -1)
        self.lrate = params.get("lrate", 0.1)
        self.lrate0 = self.lrate
        self.minlrate = params.get("minlrate", 1e-12)
        self.lratefact = params.get("lratefact", 0.5)
        self.rho0 = params.get("rho0", 1.5)
        self.minrho = params.get("minrho", 1.0)
        self.maxrho = params.get("maxrho", 2.0)
        self.rholrate = params.get("rholrate", 0.05)
        self.rholrate0 = self.rholrate
        self.rholratefact = params.get("rholratefact", 0.1)
        self.invsigmax = params.get("invsigmax", 1000.0)
        self.invsigmin = params.get("invsigmin", 1e-4)
        self.do_history = params.get("do_history", False)
        self.histstep = params.get("histstep", 10)
        self.do_opt_block = params.get("do_opt_block", True)
        self.block_size = params.get("block_size", 8192)
        self.blk_min = params.get("blk_min", 128)
        self.blk_max = params.get("blk_max", 1024)
        self.blk_step = params.get("blk_step", 128)
        self.share_comps = params.get("share_comps", False)
        self.comp_thresh = params.get("comp_thresh", 0.99)
        self.share_start = params.get("share_start", 100)
        self.share_int = params.get("share_int", 100)
        self.doscaling = params.get("doscaling", True)
        self.scalestep = params.get("scalestep", 1)
        self.do_sphere = params.get("do_sphere", True)
        self.do_mean = params.get("do_mean", True)
        self.do_approx_sphere = params.get("do_approx_sphere", True)
        self.pcakeep = params.get("pcakeep")
        self.pcadb = params.get("pcadb")
        # Numerical-rank floors (issue #223); see pamica/rank.py and ADR 0004.
        self.mineig = params.get("mineig", MINEIG)
        self.mineig_rel = params.get("mineig_rel", MINEIG_REL)
        self.writestep = params.get("writestep", 100)
        self.max_decs = params.get("max_decs", 5)
        # Consecutive small-increase iterations tolerated before stopping
        # (Fortran maxincs, amica17.f90:1087).
        self.maxincs = params.get("maxincs", 5)
        # Restart-on-NaN (Fortran amica17.f90:1027-1060): if the LL goes
        # non-finite at iter <= restartiter, reinitialize and start over, up to
        # maxrestarts times; a later NaN stops the fit (Fortran exits too).
        self.restartiter = params.get("restartiter", 10)
        self.maxrestarts = params.get("maxrestarts", 3)
        self.numrestarts = 0
        # Set by fit(): whether the fit ended with a finite likelihood, and the
        # reason it stopped. converged=False signals a terminal non-finite LL
        # (diverged, no results written), which callers/CLI must surface.
        self.converged = False
        self.stop_reason = None
        self.min_dll = params.get("min_dll", 1e-9)
        self.min_grad_norm = params.get("min_grad_norm", 1e-7)
        self.use_min_dll = params.get("use_min_dll", True)
        self.use_grad_norm = params.get("use_grad_norm", True)
        self.pdftype = params.get("pdftype", 1)
        self.outdir = Path(params.get("outdir", "output"))

        # Data-source config (used by fit() when called without explicit
        # data). load_default_params() strips 'files'/'data_dim'/'field_dim'
        # from `params` (they are data-specific, not hyperparameters), so
        # read them directly from the raw params_file JSON instead.
        self._config_files = None
        self._config_data_dim = None
        self._config_field_dim = None
        if params_file is not None:
            with open(params_file) as f:
                raw_params = json.load(f)
            self._config_files = raw_params.get("files")
            self._config_data_dim = raw_params.get("data_dim")
            self._config_field_dim = raw_params.get("field_dim")

        # Initialize random state
        self.rng = np.random.RandomState(params.get("seed"))

        # Initialize model parameters
        self.A: Optional[np.ndarray] = None  # Mixing matrix
        self.W: Optional[np.ndarray] = None  # Unmixing matrix
        self.c: Optional[np.ndarray] = None  # Bias terms
        self.mu: Optional[np.ndarray] = None  # Means of mixture components
        self.alpha: Optional[np.ndarray] = None  # Mixture weights
        self.beta: Optional[np.ndarray] = None  # Scale parameters
        self.rho: Optional[np.ndarray] = None  # Shape parameters
        self.gm: Optional[np.ndarray] = None  # Model weights

        # Initialize data parameters
        self.data_dim: Optional[int] = None
        self.num_samples: Optional[int] = None
        self.mean: Optional[np.ndarray] = None
        self.sphere: Optional[np.ndarray] = None
        self.sldet = 0.0
        self.comp_list: Optional[np.ndarray] = None
        self.comp_used: Optional[np.ndarray] = None
        # Outlier rejection (do_reject), mirroring AMICATorchNG's good_idx: an
        # index array of the currently-kept samples that only ever shrinks, plus
        # its count (num_good_samples), which normalizes gm (the model weights)
        # and self.ll (see _get_updates_and_likelihood, matching Fortran's
        # LL(iter) = LLtmp2 / dble(numgoodsum*nw), amica15.f90:1752). Both
        # None/full until fit() sets them up. numrej counts rejection passes
        # (the maxrej budget). _last_ll_samples
        # holds the pre-update per-sample LL that _reject_outliers thresholds
        # (captured in the E-step, applied after the parameter update, matching
        # the torch ordering).
        self.good_idx: Optional[np.ndarray] = None
        self.num_good_samples: Optional[int] = None
        self.numrej = 0
        self._last_ll_samples: Optional[np.ndarray] = None

        # Initialize optimization state
        self.iter = 0
        self.ll = []  # Log likelihood history
        self.nd = []  # Gradient norm history

        # Initialize Newton optimization parameters
        self.sigma2: Optional[np.ndarray] = None
        self.lambda_: Optional[np.ndarray] = None
        self.kappa: Optional[np.ndarray] = None

        # Setup logging
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging configuration."""
        # Create main logger
        self.logger = logging.getLogger("AMICA")
        self.logger.setLevel(logging.INFO)

        # Ensure output directory exists
        self.outdir = Path(self.outdir)
        if not self.outdir.exists():
            self.outdir.mkdir(parents=True)

        # Remove any existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # Add console handler for stdout
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter("%(message)s")
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # Add file handler for out.txt
        self.file_path = self.outdir / "out.txt"
        file_handler = logging.FileHandler(self.file_path, mode="w")
        file_formatter = logging.Formatter("%(message)s")
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        # Prevent propagation to avoid duplicate logging
        self.logger.propagate = False

    @classmethod
    def from_json_file(cls, params_file: str, **kwargs) -> "AMICA":
        """
        Construct an AMICA model from a JSON parameter file.

        Equivalent to ``AMICA(params_file=params_file, **kwargs)``. If the
        parameter file defines ``files``/``data_dim``/``field_dim``, a
        subsequent call to :meth:`fit` with no arguments will load the data
        described there (see :meth:`fit`).

        Parameters
        ----------
        params_file : str
            Path to JSON parameter file.
        **kwargs
            Additional overrides passed through to the constructor.

        Returns
        -------
        model : AMICA
            An unfitted model configured from the parameter file.
        """
        return cls(params_file=params_file, **kwargs)

    @property
    def data_dim_in(self) -> int:
        """Input channel count, i.e. the width of the sphere.

        Differs from ``data_dim`` only when rank reduction shrank the model to
        the detected numerical rank (issue #223). Derived rather than stored, so
        it cannot drift from the sphere it describes. Mirrors
        ``AMICATorchNG.n_channels_in``.
        """
        if self.sphere is None:
            return self.data_dim if self.data_dim is not None else 0
        return int(self.sphere.shape[1])

    def get_sensor_mixing_matrix(self, model_idx: int = 0) -> np.ndarray:
        """Mixing matrix mapped back to input-channel space.

        ``pinv(sphere) @ A``, of shape ``(data_dim_in, data_dim)`` -- the Fortran
        ``Spinv`` mapping (amica15.f90:550-560), and the only way to recover
        sensor maps when rank reduction has made the sphere non-square
        (issue #223). Mirrors ``AMICATorchNG.get_sensor_mixing_matrix``.
        """
        if self.sphere is None or self.A is None or self.comp_list is None:
            raise RuntimeError("Model has not been fitted yet; call fit() first.")
        A = self.A[:, self.comp_list[:, model_idx]]
        return np.linalg.pinv(self.sphere) @ A

    def get_weights(self) -> np.ndarray:
        """
        Return the learned unmixing matrix for the first model.

        Returns
        -------
        W : ndarray of shape (n_components, n_components)
            Unmixing matrix for model 0.
        """
        if self.W is None:
            raise RuntimeError("Model has not been fitted yet; call fit() first.")
        # Internal W = inv(A) is stored transposed relative to the true unmixing
        # (the E-step forms activations as (X-c)^T @ W), so return W^T (issue #24).
        # This is the raw unmixing matrix; it does not account for the per-model
        # data-space center c (issue #27) -- use transform() for c-corrected
        # sources. Harmless for model 0 single-model fits where c == 0.
        return self.W[:, :, 0].T

    def fit(self, data: Optional[np.ndarray] = None) -> "AMICA":
        """
        Fit the AMICA model to the data.

        Parameters
        ----------
        data : ndarray of shape (n_channels, n_samples), optional
            The input data to fit the model to. If omitted, the data is
            loaded from the ``files``/``data_dim``/``field_dim`` parameters
            supplied via ``params_file`` (see :meth:`from_json_file`).

        Returns
        -------
        self : AMICA
            The fitted model.
        """
        if data is None:
            if not self._config_files:
                raise ValueError(
                    "No data provided and no 'files' configured in params_file; "
                    "either pass data explicitly or set 'files'/'data_dim'/'field_dim'."
                )
            if self._config_data_dim is None or self._config_field_dim is None:
                raise ValueError(
                    "No data provided and 'data_dim'/'field_dim' are not both "
                    "configured in params_file; either pass data explicitly or "
                    "set 'files'/'data_dim'/'field_dim'."
                )
            if len(self._config_files) != len(self._config_field_dim):
                raise ValueError(
                    f"'files' has {len(self._config_files)} entries but "
                    f"'field_dim' has {len(self._config_field_dim)}; "
                    "load_multiple_files() requires one field_dim per file "
                    "(a length mismatch would silently truncate to the "
                    "shorter list via zip())."
                )
            from .data import load_multiple_files

            data = load_multiple_files(
                self._config_files, self._config_data_dim, self._config_field_dim
            )

        if data.ndim != 2:
            raise ValueError(
                f"data must be a 2D array (n_channels, n_samples), got shape {data.shape}"
            )
        if data.size == 0:
            raise ValueError("data must not be empty")

        # Log initial message
        self.logger.info("Starting AMICA fitting...")

        # Initialize dimensions
        self.data_dim = data.shape[0]
        self.num_samples = data.shape[1]

        if self.num_comps == -1:
            self.num_comps = self.data_dim * self.num_models

        # Preprocess data
        self._preprocess_data(data)

        # Initialize parameters
        self._initialize_parameters()

        # Optimize block size if requested
        if self.do_opt_block:
            self.block_size = determine_block_size(
                self.data, self.blk_min, self.blk_max, self.blk_step
            )
            self.logger.info(f"Optimal block size: {self.block_size}")

        # Main optimization loop
        self._optimize()

        # Record the outcome: a terminal non-finite LL means the fit diverged
        # (even restart-on-NaN could not recover), which callers/CLI must be
        # able to detect rather than silently trusting model.A/W.
        self.converged = len(self.ll) > 0 and bool(np.isfinite(self.ll[-1]))
        if not self.converged:
            self.logger.error(
                "AMICA did not converge: the log-likelihood is non-finite "
                "(diverged after %d restart(s)); results were not written.",
                self.numrestarts,
            )

        # Always persist the final converged result. _write_results is otherwise
        # only called on writestep boundaries during the loop, so a run whose
        # last iteration is not a writestep multiple (or that stops early) would
        # never save the final state. Guard on a finite likelihood so a run that
        # diverged to a non-finite LL (issue #39) does not overwrite the last
        # good on-disk result with NaNs.
        if self.converged:
            self._write_results()

        return self

    def _preprocess_data(self, data: np.ndarray):
        """Preprocess the data by removing mean and sphering."""
        assert self.data_dim is not None
        # Remove mean if requested
        if self.do_mean:
            self.mean = np.mean(data, axis=1, keepdims=True)
            data = data - self.mean
        else:
            self.mean = np.zeros((self.data_dim, 1))

        # Compute sphering matrix if requested
        if self.do_sphere:
            # Population covariance (divide by N, bias=True), matching Fortran's
            # DSYRK scatter/N -- not np.cov's default /(N-1). The two differ by a
            # scalar sqrt(N/(N-1)); /(N-1) leaves a ~5e-6 sphere mismatch vs the
            # reference (issue #24).
            cov = np.cov(data, bias=True)

            # Eigenvalue decomposition
            evals, evecs = linalg.eigh(cov)

            # Sort in descending order
            idx = np.argsort(evals)[::-1]
            evals = evals[idx]
            evecs = evecs[:, idx]

            # Numerical rank + explicit PCA reduction, decided by the policy
            # shared with the PyTorch and MLX backends (pamica/rank.py, issue
            # #223) so the three cannot disagree about how many dimensions are
            # real. Fortran: numeigs = min(pcakeep, count(eigs > mineig)).
            n_comp = numerical_rank(
                evals,
                mineig=self.mineig,
                mineig_rel=self.mineig_rel,
                pcakeep=self.pcakeep,
                pcadb=self.pcadb,
            )

            V = evecs[:, :n_comp]
            inv_sqrt = np.diag(1.0 / np.sqrt(evals[:n_comp]))
            # Create sphering matrix
            if n_comp < self.data_dim:
                # Rank-reduced: the sphere is (n_comp, data_dim), so the sphered
                # data come out at the kept rank instead of staying
                # rank-deficient at full width (Fortran nw = numeigs,
                # amica15.f90:545).
                w_pca = inv_sqrt @ V.T
                if self.do_approx_sphere:
                    # Fortran symmetrizes the reduced whitening by the orthogonal
                    # polar factor of the leading n_comp block of V^T
                    # (amica15.f90:483-490).
                    U_b, _, Vt_b = np.linalg.svd(evecs.T[:n_comp, :n_comp])
                    self.sphere = (Vt_b.T @ U_b.T) @ w_pca
                else:
                    self.sphere = w_pca
            elif self.do_approx_sphere:
                # Symmetric ZCA sphere V diag(1/sqrt(eval)) V^T (Fortran
                # do_approx_sphere=True, amica17.f90:480-481) -- the parity form.
                # The old diag(1/sqrt)@V^T (PCA whitening) is a different,
                # non-symmetric transform that breaks activation parity.
                self.sphere = V @ inv_sqrt @ V.T
            else:
                # Non-symmetric PCA whitening D^-1/2 V^T (amica17.f90:495).
                self.sphere = inv_sqrt @ V.T

            # Rank reduction shrank the sphered space: size the model to the
            # kept rank before _initialize_parameters allocates against
            # data_dim (Fortran nw = numeigs, amica15.f90:545). No-op, and so
            # bit-exact, for full-rank data.
            n_kept = self.sphere.shape[0]
            if n_kept != self.data_dim:
                self.logger.info(
                    "Data covariance has numerical rank %d of %d; fitting %d "
                    "sources and mapping back via the sphere pseudo-inverse.",
                    n_kept,
                    self.data_dim,
                    n_kept,
                )
                if self.num_comps == self.data_dim * self.num_models:
                    # num_comps was derived from the channel count, so it follows
                    # the rank; an explicitly configured num_comps is left alone.
                    self.num_comps = n_kept * self.num_models
                self.data_dim = n_kept

            # Apply sphering
            data = np.dot(self.sphere, data)
            # Sphering log-determinant term of the data log-likelihood
            # (Fortran sldet, amica17.f90:474): sum over kept eigenvalues of
            # -0.5*log(eval). Required so the reported LL matches Fortran; its
            # omission was why the NumPy LL sat ~ +1.5 instead of ~ -3.4.
            self.sldet = float(-0.5 * np.sum(np.log(evals[:n_comp])))
        else:
            self.sphere = np.eye(self.data_dim)
            self.sldet = 0.0

        self.data = data

    def _initialize_parameters(self):
        """Initialize all model parameters."""
        assert self.data_dim is not None
        # Initialize mixing/unmixing matrices
        if self.A is None:
            self.A = np.zeros((self.data_dim, self.num_comps))
            for h in range(self.num_models):
                if not hasattr(self, "fix_init") or not self.fix_init:
                    self.A[:, h * self.data_dim : (h + 1) * self.data_dim] = np.eye(
                        self.data_dim
                    ) + 0.01 * (0.5 - self.rng.rand(self.data_dim, self.data_dim))
                else:
                    self.A[:, h * self.data_dim : (h + 1) * self.data_dim] = np.eye(
                        self.data_dim
                    )

        # Initialize component assignments
        self.comp_list = np.zeros((self.data_dim, self.num_models), dtype=int)
        self.comp_used = np.ones(self.num_comps, dtype=bool)
        for h in range(self.num_models):
            self.comp_list[:, h] = np.arange(h * self.data_dim, (h + 1) * self.data_dim)

        # Outlier-rejection state (do_reject): start with every sample good
        # (good_idx = all indices), mirroring AMICATorchNG. num_good_samples
        # drives the gm/LL normalization and equals num_samples until the first
        # rejection shrinks good_idx.
        assert self.num_samples is not None
        if self.do_reject:
            self.good_idx = np.arange(self.num_samples)
        self.num_good_samples = self.num_samples

        # Initialize mixture parameters
        if self.mu is None:
            self.mu = np.zeros((self.num_mix, self.num_comps))
            for k in range(self.num_comps):
                self.mu[:, k] = np.linspace(-1, 1, self.num_mix)
                if not hasattr(self, "fix_init") or not self.fix_init:
                    self.mu[:, k] += 0.05 * (1 - 2 * self.rng.rand(self.num_mix))

        if self.alpha is None:
            self.alpha = np.ones((self.num_mix, self.num_comps)) / self.num_mix

        if self.beta is None:
            self.beta = np.ones((self.num_mix, self.num_comps))
            if not hasattr(self, "fix_init") or not self.fix_init:
                self.beta += 0.1 * (0.5 - self.rng.rand(self.num_mix, self.num_comps))

        if self.rho is None:
            self.rho = self.rho0 * np.ones((self.num_mix, self.num_comps))

        if self.gm is None:
            self.gm = np.ones(self.num_models) / self.num_models

        # Initialize bias terms
        if self.c is None:
            self.c = np.zeros((self.data_dim, self.num_models))

        # Initialize Newton optimization parameters
        if self.do_newton:
            self.sigma2 = np.ones((self.data_dim, self.num_models))
            self.lambda_ = np.zeros((self.data_dim, self.num_models))
            self.kappa = np.zeros((self.data_dim, self.num_models))

        # Get initial unmixing matrices
        self._update_unmixing_matrices()

    def _reinitialize_for_restart(self):
        """Redraw the mixing matrix after a non-finite likelihood.

        Matches Fortran's restart path (amica17.f90:1032-1053): it re-draws
        *only* the mixing matrix ``A`` (from the already-advanced RNG, a new
        random basin) and recomputes ``comp_list``/``W``; the last-successful
        mixture parameters (``mu``/``alpha``/``beta``/``rho``/``gm``/``c``) are
        kept, not cold-reset. The learning rate and the LL/gradient-norm history
        are reset so the restarted run is judged from scratch; preprocessing
        (mean/sphere) and the RNG are preserved.
        """
        # Only A is nulled; _initialize_parameters redraws it (and unconditionally
        # rebuilds comp_list and W) while leaving the still-finite mixture params.
        # Preserve the outlier-rejection state across a restart: Fortran's
        # startover redraws A but never reverts already-applied rejections
        # (amica17.f90:1121-1148), so snapshot good_idx/num_good_samples around
        # the reinit (which would otherwise reset them to the full set) and
        # restore them; numrej is an instance attribute and already survives.
        saved_good_idx = self.good_idx
        saved_num_good = self.num_good_samples
        self.A = None
        self._initialize_parameters()
        if self.do_reject:
            self.good_idx = saved_good_idx
            self.num_good_samples = saved_num_good
        self.lrate = self.lrate0
        # rholrate is the (maxdecs-ratcheted) rho-rate ceiling; restore it to the
        # pristine rholrate0 so a re-fit starts fresh (issue #193).
        self.rholrate = self.rholrate0
        self.ll = []
        self.nd = []

    def _update_unmixing_matrices(self):
        """Update unmixing matrices from mixing matrix."""
        self.W = get_unmixing_matrices(self.A, self.comp_list)

    def _compute_log_pdf(
        self, y: np.ndarray, rho: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute log PDF value and its derivative for given activation.

        Parameters
        ----------
        y : ndarray
            Activation values
        rho : float
            Shape parameter

        Returns
        -------
        log_pdf : ndarray
            Log PDF values
        dpdf : ndarray
            PDF derivatives (not in log space)
        """
        if rho == 1.0:
            # Laplace distribution
            log_pdf = -np.abs(y) - np.log(2.0)
            pdf = np.exp(log_pdf)
            dpdf = -np.sign(y) * pdf
        elif rho == 2.0:
            # Gaussian distribution
            log_pdf = -y * y - 0.5 * np.log(np.pi)
            pdf = np.exp(log_pdf)
            dpdf = -2 * y * pdf
        else:
            # Generalized Gaussian distribution
            log_pdf = -np.power(np.abs(y), rho) - np.log(2.0) - gammaln(1.0 + 1.0 / rho)
            pdf = np.exp(log_pdf)
            dpdf = -rho * np.power(np.abs(y), rho - 1) * np.sign(y) * pdf

        return log_pdf, dpdf

    def _compute_score(self, y: np.ndarray, rho: float) -> np.ndarray:
        """Generalized-Gaussian score ``fp = d|y|^rho/dy`` (Fortran ``fp``,
        amica17.f90:1455-1467): ``sign(y)`` for Laplace, ``2y`` for Gaussian,
        ``rho*sign(y)*|y|^(rho-1)`` otherwise. Used by the Newton curvature
        statistics; distinct from the density derivative ``dpdf``."""
        if rho == 1.0:
            return np.sign(y)
        if rho == 2.0:
            return 2.0 * y
        return rho * np.sign(y) * np.power(np.abs(y), rho - 1.0)

    def _get_updates_and_likelihood(self) -> Dict:
        """
        Compute parameter updates and data likelihood.

        Returns
        -------
        updates : dict
            Dictionary containing parameter updates and likelihood
        """
        assert self.data_dim is not None
        # Initialize update accumulators
        updates = {
            "dgm": np.zeros(self.num_models),
            "dalpha_n": np.zeros((self.num_mix, self.num_comps)),
            "dmu_n": np.zeros((self.num_mix, self.num_comps)),
            "dmu_d": np.zeros((self.num_mix, self.num_comps)),
            "dbeta_n": np.zeros((self.num_mix, self.num_comps)),
            "dbeta_d": np.zeros((self.num_mix, self.num_comps)),
            "drho_n": np.zeros((self.num_mix, self.num_comps)),
            "dWtmp": np.zeros((self.data_dim, self.data_dim, self.num_models)),
            "dc_numer": np.zeros((self.data_dim, self.num_models)),
            "ll": 0.0,
        }

        if self.do_newton:
            updates.update(
                {
                    "dsigma2": np.zeros((self.data_dim, self.num_models)),
                    "dlambda": np.zeros((self.data_dim, self.num_models)),
                    "dkappa": np.zeros((self.data_dim, self.num_models)),
                }
            )

        # Restrict the E-step to the currently-good samples under do_reject
        # (mirrors AMICATorchNG's ``X_use = X_t[:, good_idx]``); the default path
        # uses the full array with no copy, so it stays bit-identical.
        data_use = self.data[:, self.good_idx] if self.do_reject else self.data

        # Per-sample log-likelihood of the good set, collected in good_idx order
        # so _reject_outliers' keep-mask maps back onto good_idx. Only gathered
        # under do_reject, so the default path carries no extra work.
        ll_parts = []

        # Process data in blocks
        for start in range(0, data_use.shape[1], self.block_size):
            end = min(start + self.block_size, data_use.shape[1])
            X = data_use[:, start:end]

            # Get block updates
            block_updates = self._get_block_updates(X)

            # Accumulate updates (block_updates may carry an extra "ll_samples"
            # under do_reject, which is gathered below rather than summed here).
            for key in updates:
                updates[key] += block_updates[key]

            if self.do_reject:
                ll_parts.append(block_updates["ll_samples"])

        if self.do_reject:
            self._last_ll_samples = np.concatenate(ll_parts)

        # Normalize the accumulated total LL by (good-sample count x working
        # dimensionality), matching Fortran's LL(iter) = LLtmp2 / dble(numgoodsum*nw)
        # (amica15.f90:1752) at the point the full-data sum becomes available.
        # numgoodsum is self.num_good_samples: Fortran initializes numgoodsum to
        # all_blks (all samples) and only shrinks it under do_reject (amica15.f90:234,
        # 2234), and self.num_good_samples mirrors that exactly (set to num_samples
        # in _initialize_parameters, only shrunk by _reject_outliers), so this one
        # expression is correct whether or not do_reject is set -- no branch needed.
        # nw is the per-model working dimensionality (Fortran's numeigs, amica15.f90:545);
        # self.data_dim is its NumPy analogue -- comp_list is (data_dim, num_models),
        # matching Fortran's comp_list(nw, num_models) (amica15.f90:599).
        assert self.num_good_samples is not None and self.data_dim is not None
        updates["ll"] /= self.num_good_samples * self.data_dim

        return updates

    def _forward_block(self, X: np.ndarray):
        """Shared E-step forward pass for one data block (Fortran
        amica17.f90:1280-1372): activations ``b``, the raw (unnormalized)
        mixture log-probabilities ``z``, their per-source max ``z0max``, the
        per-model log-likelihood ``logV`` (with the log|det W| + sldet
        Jacobian), and the per-sample total log-likelihood (``Vmax``/
        ``ll_samples``, log-sum-exp over models).

        Shared by ``_get_block_updates`` (the training-path M-step, which
        normalizes ``z``/``logV`` into responsibilities) and
        ``_compute_full_posterior_ll`` (the LLt write path, which only needs
        ``logV``/``ll_samples``) so a single copy carries any future fix to
        this forward pass -- this codebase has had one-sided forward-pass bugs
        before (issue #24's fp-vs-dpdf sign bug), and a duplicated copy could
        silently drift (issue #155).

        Returns
        -------
        b : ndarray of shape (batch, data_dim, num_models)
        z : ndarray of shape (batch, data_dim, num_mix, num_models)
            Raw (unnormalized) mixture log-probabilities.
        z0max : ndarray of shape (batch, data_dim, 1, num_models)
        logV : ndarray of shape (batch, num_models)
        Vmax : ndarray of shape (batch, 1)
        ll_samples : ndarray of shape (batch,)
        """
        assert (
            self.data_dim is not None
            and self.c is not None
            and self.W is not None
            and self.comp_list is not None
            and self.beta is not None
            and self.mu is not None
            and self.rho is not None
            and self.alpha is not None
            and self.gm is not None
        )
        batch_size = X.shape[1]

        # Compute activations for each model. c is the per-model data-space
        # center: b = W(x - c). Fortran subtracts wc in the E-step
        # (amica17.f90:1280-1292), where wc = W@c is precomputed in
        # get_unmixing_matrices (amica17.f90:2178). For n_models=1, c == 0, so
        # this is bit-identical to X.T @ W.
        b = np.zeros((batch_size, self.data_dim, self.num_models))
        for h in range(self.num_models):
            b[:, :, h] = np.dot((X - self.c[:, h][:, None]).T, self.W[:, :, h])

        # Compute mixture probabilities and responsibilities
        z = np.zeros((batch_size, self.data_dim, self.num_mix, self.num_models))
        for h in range(self.num_models):
            for i in range(self.data_dim):
                k = self.comp_list[i, h]
                for j in range(self.num_mix):
                    y = self.beta[j, k] * (b[:, i, h] - self.mu[j, k])

                    # Compute log PDF and its derivative
                    log_pdf, dpdf = self._compute_log_pdf(y, self.rho[j, k])

                    # Compute log probability directly in log space
                    z[:, i, j, h] = (
                        np.log(self.alpha[j, k]) + np.log(self.beta[j, k]) + log_pdf
                    )

        # Per-source log-density = logsumexp over mixtures of the pre-norm logits
        # z0, then the per-model log-likelihood logV adds the log|det W| + sldet
        # Jacobian (Fortran amica17.f90:1341-1350). This is the correct
        # pre-normalization log-likelihood; the earlier post-normalization
        # np.sum(np.exp(z_normalized)) did not recover a real log-density and
        # omitted the Jacobian (so LL was positive and ~4.9 off per channel).
        z0max = np.max(z, axis=2, keepdims=True)
        ll_src = z0max[:, :, 0, :] + np.log(
            np.sum(np.exp(z - z0max), axis=2)
        )  # (batch, data_dim, num_models)
        logV = np.zeros((batch_size, self.num_models))
        for h in range(self.num_models):
            # A near-singular W (a transient the natural gradient can pass
            # through) makes slogdet emit divide/overflow/invalid FP warnings.
            # Suppress the numpy console noise, but DON'T rely on that as the
            # guard: the explicit isfinite check below is the real diagnostic --
            # it fires for a -inf logdet (singular W) AND a NaN logdet (genuinely
            # broken W), so silencing `invalid` does not hide a NaN W. The
            # fit-loop LL check (_check_convergence) also stops on a -inf LL.
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                _, logdet_W = np.linalg.slogdet(self.W[:, :, h])
            if not np.isfinite(logdet_W):
                self.logger.warning(
                    "Non-finite logdet(W) for model %d at iter %d (logdet=%s); "
                    "W is singular or corrupt.",
                    h,
                    getattr(self, "iter", -1),
                    logdet_W,
                )
            logV[:, h] = (
                np.log(self.gm[h])
                + logdet_W
                + self.sldet
                + np.sum(ll_src[:, :, h], axis=1)
            )

        # Block log-likelihood = sum_t logsumexp_h logV (Fortran :1372).
        Vmax = np.max(logV, axis=1, keepdims=True)
        ll_samples = Vmax[:, 0] + np.log(np.sum(np.exp(logV - Vmax), axis=1))

        return b, z, z0max, logV, Vmax, ll_samples

    def _get_block_updates(self, X: np.ndarray) -> Dict:
        """
        Compute parameter updates for a data block.

        Parameters
        ----------
        X : ndarray
            Data block to process

        Returns
        -------
        updates : dict
            Parameter updates for this block
        """
        assert (
            self.data_dim is not None
            and self.c is not None
            and self.W is not None
            and self.comp_list is not None
            and self.beta is not None
            and self.mu is not None
            and self.rho is not None
            and self.alpha is not None
            and self.gm is not None
        )
        batch_size = X.shape[1]
        tiny = np.finfo(np.float64).tiny
        updates = {
            "dgm": np.zeros(self.num_models),
            "dalpha_n": np.zeros((self.num_mix, self.num_comps)),
            "dmu_n": np.zeros((self.num_mix, self.num_comps)),
            "dmu_d": np.zeros((self.num_mix, self.num_comps)),
            "dbeta_n": np.zeros((self.num_mix, self.num_comps)),
            "dbeta_d": np.zeros((self.num_mix, self.num_comps)),
            "drho_n": np.zeros((self.num_mix, self.num_comps)),
            "dWtmp": np.zeros((self.data_dim, self.data_dim, self.num_models)),
            "dc_numer": np.zeros((self.data_dim, self.num_models)),
            "ll": 0.0,
        }

        if self.do_newton:
            updates.update(
                {
                    "dsigma2": np.zeros((self.data_dim, self.num_models)),
                    "dlambda": np.zeros((self.data_dim, self.num_models)),
                    "dkappa": np.zeros((self.data_dim, self.num_models)),
                }
            )

        b, z, z0max, logV, Vmax, ll_samples = self._forward_block(X)

        updates["ll"] = np.sum(ll_samples)
        # Expose the per-sample vector for outlier rejection (issue #123); only
        # under do_reject, so the default path is unchanged.
        if self.do_reject:
            updates["ll_samples"] = ll_samples

        # Model responsibilities v = softmax(logV); mixture responsibilities z.
        v = np.exp(logV - Vmax)
        v /= np.sum(v, axis=1, keepdims=True)
        z = np.exp(z - z0max)
        z /= np.sum(z, axis=2, keepdims=True)

        for h in range(self.num_models):
            # Model weights
            updates["dgm"][h] = np.sum(v[:, h])

            for i in range(self.data_dim):
                k = self.comp_list[i, h]

                # Newton second moment sigma2 = E_v[b_i^2] is a per-source
                # quantity (no mixture index), accumulated once per (i, h)
                # (Fortran amica17.f90:1419). It must NOT sit inside the
                # mixture loop below, or it is inflated num_mix-fold.
                if self.do_newton:
                    updates["dsigma2"][i, h] += np.sum(v[:, h] * b[:, i, h] ** 2)

                # Mixture exact-EM sufficient statistics (Fortran
                # amica17.f90:1524-1578). These use the score
                # fp = rho*sign(y)*|y|^(rho-1) (Fortran fp), NOT the density
                # derivative dpdf, and produce numerator/denominator pairs for a
                # fixed-point (not first-order gradient) update. Assumes rho <= 2
                # (the maxrho default); the rho > 2 denominator branches are
                # unreachable and not implemented.
                for j in range(self.num_mix):
                    y = self.beta[j, k] * (b[:, i, h] - self.mu[j, k])
                    fp = self._compute_score(y, self.rho[j, k])
                    u = v[:, h] * z[:, i, j, h]  # model x mixture responsibility
                    ufp = u * fp

                    updates["dalpha_n"][j, k] += np.sum(u)
                    updates["dmu_n"][j, k] += np.sum(ufp)  # sum(ufp) (:1532)
                    updates["dmu_d"][j, k] += self.beta[j, k] * np.sum(
                        ufp / y
                    )  # sbeta*sum(ufp/y) (:1537)
                    updates["dbeta_n"][j, k] += np.sum(u)  # sum(u) (:1550)
                    updates["dbeta_d"][j, k] += np.sum(ufp * y)  # sum(ufp*y) (:1556)

                    # drho_numer = rho*sum(u*|y|^rho*ln|y|) (:1560-1578). Leading
                    # rho from ln(|y|^rho)=rho*ln|y| (issue #24 Bug 1); no
                    # per-component rho!=1&rho!=2 mask (Bug 2), only the
                    # per-sample underflow guard (:1570).
                    ay = np.abs(y)
                    ayrho = np.power(ay, self.rho[j, k])
                    logab = self.rho[j, k] * np.log(np.maximum(ay, tiny))
                    # Fortran zeros the term when |y|^rho < epsdble=1e-16
                    # (amica17.f90:1570 / amica17_header.f90:73), not at denormal
                    # underflow; use 1e-16 to match, not np.finfo.tiny.
                    logab = np.where(ayrho < 1e-16, 0.0, logab)
                    updates["drho_n"][j, k] += np.sum(u * ayrho * logab)

                    if self.do_newton:
                        # Newton curvature terms use the score fp (Fortran
                        # :1500-1512): kappa carries sbeta^2, lambda folds in the
                        # mu^2 curvature term so lambda=dlambda/dgm matches Fortran.
                        dkap = np.sum(u * fp**2) * self.beta[j, k] ** 2
                        updates["dkappa"][i, h] += dkap
                        updates["dlambda"][i, h] += (
                            np.sum(u * (fp * y - 1) ** 2) + dkap * self.mu[j, k] ** 2
                        )

            # Natural-gradient accumulator: g_i = sum_j sbeta*u*fp, then the
            # source-space sum dWtmp = g^T b (Fortran :1493/:1592). Uses the
            # score fp, not dpdf.
            g = np.zeros((batch_size, self.data_dim))
            for i in range(self.data_dim):
                k = self.comp_list[i, h]
                for j in range(self.num_mix):
                    y = self.beta[j, k] * (b[:, i, h] - self.mu[j, k])
                    fp = self._compute_score(y, self.rho[j, k])
                    g[:, i] += self.beta[j, k] * (v[:, h] * z[:, i, j, h]) * fp

            updates["dWtmp"][:, :, h] += np.dot(g.T, b[:, :, h])

            # Data-space bias numerator dc_numer[i,h] = sum_t v_h(t)*x(i,t)
            # (Fortran :1423-1429); denominator is dgm[h] = sum_t v_h(t). Replaces
            # the old gradient-style bias sum(g), which was accumulated but never
            # applied (c was frozen at 0); the Fortran update is the data-space
            # responsibility-weighted mean (issue #27).
            updates["dc_numer"][:, h] += np.dot(X, v[:, h])

        return updates

    def _compute_full_posterior_ll(self) -> Tuple[np.ndarray, np.ndarray]:
        """Recompute the per-model/per-sample log-likelihood over every sample
        of ``self.data``, for the Fortran ``LLt`` output (issue #155).

        Shares the ``_forward_block`` forward pass with ``_get_block_updates``
        (only ``logV``/``ll_samples`` are needed here), so this read-only
        computation cannot silently drift from the training path's E-step.
        Computed fresh from ``self``'s current parameters, not stashed
        mid-training-loop values, so it is correct even after a keep-best
        rollback (issue #51).

        Deliberate divergence from Fortran: Fortran's ``modloglik`` is filled
        during iteration i's E-step, the M-step then updates the parameters,
        and ``write_output`` writes both -- so Fortran's on-disk LLt is stale
        by one M-step relative to the parameters written alongside it. This
        method recomputes from the POST-update parameters, so pamica's LLt is
        self-consistent with the written W/A (better-behaved, not "fixed
        toward Fortran" -- do not change this to match Fortran's staleness).

        Cost note: called at every ``writestep`` checkpoint during training
        (see ``_write_results``), not just once at the end, so each
        checkpoint now pays a full O(n_samples) forward pass it did not pay
        before this method existed. Bounded and small at the default
        ``writestep`` (100; the bundled sample config uses 20); accepted for
        now since this is the legacy backend. A cheaper design (stash the
        per-sample ``logV`` computed during the training E-step instead of
        recomputing, as Fortran does by keeping ``modloglik`` permanently
        allocated) is tracked as a follow-up rather than done here.

        Under ``do_reject``, only the good set (``self.good_idx``) is scored
        -- Fortran zeroes a rejected sample's ``modloglik``/``loglik`` on write
        (amica15.f90:2211-2216) and ``load_rej`` uses that exact zero as the
        rejection sentinel (``sum(modloglik(:,i)) == 0.0``, amica15.f90:
        887-896), so rejected columns of ``Lht``/``Lt`` are left at their
        zero-initialized value rather than computed and discarded -- this also
        avoids running rejected outliers through the model for the first time
        at write time.

        Returns
        -------
        Lht : ndarray of shape (num_models, n_samples)
            Per-model log-likelihood, Fortran's ``modloglik``. Zero for
            rejected samples under ``do_reject``.
        Lt : ndarray of shape (n_samples,)
            Total log-likelihood, Fortran's ``loglik``. Zero for rejected
            samples under ``do_reject``.
        """
        assert (
            self.data_dim is not None
            and self.data is not None
            and self.W is not None
            and self.c is not None
            and self.comp_list is not None
            and self.beta is not None
            and self.mu is not None
            and self.rho is not None
            and self.alpha is not None
            and self.gm is not None
        )
        data = self.data
        n_samples = data.shape[1]
        Lht = np.zeros((self.num_models, n_samples))
        Lt = np.zeros(n_samples)

        if self.do_reject:
            assert self.good_idx is not None
            idx = self.good_idx
            data_use = data[:, idx]
        else:
            idx = np.arange(n_samples)
            data_use = data
        n_use = data_use.shape[1]

        for start in range(0, n_use, self.block_size):
            end = min(start + self.block_size, n_use)
            X = data_use[:, start:end]
            _, _, _, logV, _, ll_samples = self._forward_block(X)

            cols = idx[start:end]
            Lht[:, cols] = logV.T
            Lt[cols] = ll_samples

        return Lht, Lt

    def _update_parameters(self, updates: Dict):
        """
        Update model parameters using computed updates.

        Parameters
        ----------
        updates : dict
            Dictionary containing parameter updates
        """
        assert (
            self.data_dim is not None
            and self.num_samples is not None
            and self.c is not None
            and self.mu is not None
            and self.beta is not None
            and self.rho is not None
            and self.comp_list is not None
            and self.A is not None
        )
        # Fortran builds dAk from the model weights of the *previous* iteration:
        # gm is not reassigned until update_params (amica15.f90:1789+), which runs
        # after accum_updates_and_likelihood (:1731-1743). Snapshot it here so the
        # nd block below weights by the same gm Fortran would (issue #219).
        assert self.gm is not None
        gm_prev = self.gm.copy()

        # Update model weights, normalizing by the number of samples the E-step
        # actually summed over: the good set under do_reject, else all samples.
        if self.do_reject:
            assert self.num_good_samples is not None
            self.gm = updates["dgm"] / self.num_good_samples
        else:
            self.gm = updates["dgm"] / self.num_samples

        # Per-model data-space bias c (Fortran's `update_c` flag, amica17.f90:1423-
        # 1429 numerator / :1899-1901 division): c[i,h] = sum_t v_h*x / sum_t v_h,
        # the responsibility-weighted data mean for model h (the E-step centers
        # each model at its own mean, b = W(x - c)). Skipped for a single model to
        # keep the issue #24 parity bit-exact: with v==1 the update would add a
        # ~1e-13 float-sum residual of the (mean-removed) data. dgm[h] = sum_t v_h
        # is the denominator (Fortran `dc_denom`). A dead model (dgm[h]==0) gives
        # 0/0; keep its prior c so a NaN cannot poison the next iteration's
        # cross-model softmax for every model (mirrors the mu/beta/rho guards).
        if self.num_models > 1:
            dgm = updates["dgm"]
            live = dgm > 0.0
            new_c = (
                updates["dc_numer"]
                / np.maximum(dgm, np.finfo(np.float64).tiny)[None, :]
            )
            self.c = np.where(live[None, :], new_c, self.c)
            if not np.all(live):
                self.logger.warning(
                    "Zero-responsibility model(s) at iter %d; kept prior bias c "
                    "(dead-model guard).",
                    self.iter,
                )

        # Update mixture weights
        self.alpha = updates["dalpha_n"] / np.sum(updates["dalpha_n"], axis=0)

        # Exact-EM mixture location/scale (Fortran :1978/:1993). These are
        # fixed-point updates -- mu += dmu_n/dmu_d, beta *= sqrt(dbeta_n/dbeta_d)
        # -- NOT first-order gradient steps, so they carry no lrate.
        self.mu = self.mu + updates["dmu_n"] / updates["dmu_d"]
        self.beta = self.beta * np.sqrt(updates["dbeta_n"] / updates["dbeta_d"])
        self.beta = np.clip(self.beta, self.invsigmin, self.invsigmax)
        # Fortran keeps a live "NaN in sbeta!" canary here (amica17.f90:1996-2000);
        # the exact-EM mu/beta divisions are unguarded (matching Fortran), so
        # surface a non-finite value immediately instead of letting it propagate
        # to a later, unattributable nan-LL stop.
        if not np.all(np.isfinite(self.mu)) or not np.all(np.isfinite(self.beta)):
            self.logger.warning(
                "Non-finite mu/beta at iter %d (mixture component mass likely "
                "collapsed).",
                self.iter,
            )

        # GG shape update with the 1/psi(1+1/rho) digamma factor (Fortran
        # :2013-2014); the divisor is the per-component responsibility mass
        # dalpha_n (floored so a near-empty component cannot poison rho). A NaN
        # is reset to rho0 -- but logged first, so the reset does not silently
        # erase the failure origin.
        if not np.all(self.rho == 1.0) and not np.all(self.rho == 2.0):
            drho = updates["drho_n"] / np.maximum(updates["dalpha_n"], 1e-8)
            psi = digamma(1.0 + 1.0 / self.rho)
            new_rho = self.rho + self.rholrate * (1.0 - (self.rho / psi) * drho)
            nan_mask = np.isnan(new_rho)
            if nan_mask.any():
                self.logger.warning(
                    "NaN in rho update at iter %d for %d component(s); resetting "
                    "to rho0=%g.",
                    self.iter,
                    int(nan_mask.sum()),
                    self.rho0,
                )
                new_rho = np.where(nan_mask, self.rho0, new_rho)
            self.rho = np.clip(new_rho, self.minrho, self.maxrho)

        # Update unmixing matrices
        newton_active = self.do_newton and self.iter >= self.newt_start
        if newton_active:
            # Finalize Newton curvature statistics (Fortran amica17.f90:1762-1776).
            # The dsigma2/dkappa/dlambda accumulators already carry the sbeta^2
            # and baralpha-weighted mu^2 factors, so finalization is a plain
            # division by the model mass dgm = sum_t v_h.
            self.sigma2 = updates["dsigma2"] / updates["dgm"][:, None]
            self.lambda_ = updates["dlambda"] / updates["dgm"][:, None]
            self.kappa = updates["dkappa"] / updates["dgm"][:, None]

        # Per-model direction: Newton H if the model is positive definite,
        # otherwise natural gradient. Matching Fortran (amica17.f90:1814-1837),
        # if any off-diagonal pair fails sk1*sk2 > 1 the whole model falls
        # back to the natural gradient and the ramp targets lrate0, not newtrate.
        directions = []
        no_newt = False
        for h in range(self.num_models):
            dA = -updates["dWtmp"][:, :, h] / updates["dgm"][h]
            dA[np.diag_indices_from(dA)] += 1

            if newton_active:
                assert (
                    self.lambda_ is not None
                    and self.sigma2 is not None
                    and self.kappa is not None
                )
                H = np.zeros_like(dA)
                posdef = True
                for i in range(self.data_dim):
                    for j in range(self.data_dim):
                        if i == j:
                            H[i, i] = dA[i, i] / self.lambda_[i, h]
                        else:
                            sk1 = self.sigma2[i, h] * self.kappa[j, h]
                            sk2 = self.sigma2[j, h] * self.kappa[i, h]
                            if sk1 * sk2 > 1.0:
                                H[i, j] = (sk1 * dA[i, j] - dA[j, i]) / (
                                    sk1 * sk2 - 1.0
                                )
                            else:
                                posdef = False
                if posdef:
                    directions.append(H)
                else:
                    no_newt = True
                    directions.append(dA)
            else:
                directions.append(dA)

        if newton_active and no_newt:
            # Fortran prints this whenever a model is not positive definite
            # (amica17.f90:1911-1913); surface it rather than falling back
            # silently.
            self.logger.info(
                "Hessian not positive definite at iter %d; using natural gradient.",
                self.iter,
            )

        if newton_active and not no_newt:
            self.lrate = min(
                self.newtrate, self.lrate + min(1.0 / self.newt_ramp, self.lrate)
            )
        else:
            self.lrate = min(
                self.lrate0, self.lrate + min(1.0 / self.newt_ramp, self.lrate)
            )

        # Weight-gradient norm (Fortran ndtmpsum, amica15.f90:1731-1743). Computed
        # HERE, before the A step and before rescaling, because Fortran builds dAk
        # inside accum_updates_and_likelihood (:1731-1743) strictly before
        # update_params applies it (:1789). Using the post-update, post-rescale A
        # would measure a different quantity.
        #
        # dAk is the gm-weighted average of the per-model directions mapped through
        # A (dAk = sum_h gm[h] * dir_h^T @ A, scattered by comp_list, divided by
        # zeta = sum_h gm[h] per column). ndtmpsum is then the RMS of the used
        # columns of dAk: ||dAk[:, used]|| / sqrt(nw * n_used), with NO lrate
        # factor. Fortran measures the gradient direction before the step, not the
        # applied update lrate*dAk, and does not divide by lrate either
        # (amica15.f90:1742-1743) -- there is no missing factor here.
        #
        # Weighted by gm_prev, the pre-update model weights, because Fortran builds
        # dAk before update_params reassigns gm (issue #219). Invisible for
        # num_models=1 (gm == 1) and for the default disjoint comp_list, where the
        # gm[h] factor cancels against zeta[idx]; it bites only under share_comps
        # with a genuinely shared column, where the two weightings differ.
        #
        # Diagnostic only *here*: this dAk feeds nd_value alone. The A-update below
        # is a separate per-model loop that never reads dAk, so the ordering cannot
        # move a fitted parameter in this backend. AMICATorchNG applies the same
        # dAk to A, so there the identical fix does change fitted parameters.
        dAk = np.zeros_like(self.A)
        zeta = np.zeros(self.num_comps)
        for h in range(self.num_models):
            idx = self.comp_list[:, h]
            dAk[:, idx] += gm_prev[h] * np.dot(directions[h].T, self.A[:, idx])
            zeta[idx] += gm_prev[h]
        nonzero = zeta > 0
        dAk[:, nonzero] /= zeta[nonzero]
        # comp_used is None until fit() sets it up, and the M-step is exercised
        # directly in tests before that happens, so fall back to "all used".
        used = (
            self.comp_used
            if self.comp_used is not None
            else np.ones(self.num_comps, dtype=bool)
        )
        nd_value = float(
            np.sqrt(np.sum(dAk[:, used] ** 2) / (self.data_dim * int(used.sum())))
        )

        # A is stored as Fortran's A^T (true unmixing = W^T = inv(A)^T), so the
        # Fortran step A_fort -= lrate*A_fort @ dir becomes A -= lrate*dir^T @ A
        # (LEFT-multiply by the TRANSPOSED direction). Right-multiply by the
        # untransposed dir is invisible at the fixed point but sends the fit
        # downhill -- issue #24 root cause.
        for h in range(self.num_models):
            idx = self.comp_list[:, h]
            self.A[:, idx] = self.A[:, idx] - self.lrate * np.dot(
                directions[h].T, self.A[:, idx]
            )

        # (c was updated above, before the mixture/A updates, from dc_numer/dgm.)

        # Rescale parameters if requested
        if self.doscaling and self.iter % self.scalestep == 0:
            for k in range(self.num_comps):
                scale = np.sqrt(np.sum(self.A[:, k] ** 2))
                if scale > 0:
                    self.A[:, k] /= scale
                    self.mu[:, k] *= scale
                    self.beta[:, k] /= scale

        # Update unmixing matrices
        self._update_unmixing_matrices()

        # Store likelihood
        self.ll.append(updates["ll"])

        # The weight-gradient norm is computed above, before the A step, matching
        # Fortran's ordering. _check_convergence uses it as the gradient floor in
        # the decrease-stop condition regardless of use_grad_norm; the flag only
        # gates the separate final gradient-norm stop.
        self.nd.append(nd_value)

    def _optimize(self):
        """Main optimization loop."""
        # Log optimization start
        self.logger.info("Starting optimization...")

        # Initialize optimization variables
        numdecs = 0
        numincs = 0
        start_time = time.time()
        convergence_reason = None
        final_iter = 0
        # Per-fit counters (reset here so a refit on the same instance gets a
        # fresh budget). numrej is an instance attribute so the reject schedule
        # and tests can read the pass count; it survives a restart (which
        # preserves prior rejections) but not a fresh fit().
        self.numrestarts = 0
        self.numrej = 0

        # Determine whether to use tqdm or per-line printing
        use_tqdm_progress = self.use_tqdm and not self.verbose

        # Create iterator (with or without tqdm)
        if use_tqdm_progress:
            # Use minimal progress bar with dynamic width and ASCII characters for better compatibility
            progress_bar = tqdm(
                range(self.max_iter),
                desc="AMICA",
                unit="it",
                ncols=60,  # Smaller fixed width
                dynamic_ncols=True,  # Adapt to terminal width
                ascii=True,  # Use ASCII characters for better compatibility
                miniters=1,  # Update on every iteration
                leave=True,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",  # Simpler format
            )
            iterator = progress_bar
        else:
            iterator = range(self.max_iter)

        try:
            for iter in iterator:
                self.iter = iter
                final_iter = iter

                # Get updates and likelihood
                updates = self._get_updates_and_likelihood()

                # Update parameters
                self._update_parameters(updates)

                # Restart-on-NaN (Fortran amica17.f90:1027-1056): an early
                # non-finite LL usually means an unlucky init, so redraw A and
                # start over, up to maxrestarts times, within the first
                # restartiter iterations (Fortran's absolute `iter <= restartiter`
                # window; the iteration counter is not reset on restart here). A
                # later NaN falls through to _check_convergence, which stops
                # (Fortran exits too).
                if (
                    len(self.ll) > 0
                    and not np.isfinite(self.ll[-1])
                    and iter <= self.restartiter
                    and self.numrestarts < self.maxrestarts
                ):
                    self.numrestarts += 1
                    self.logger.warning(
                        "Non-finite LL at iter %d; reinitializing and starting "
                        "over (restart %d of %d).",
                        iter + 1,
                        self.numrestarts,
                        self.maxrestarts,
                    )
                    self._reinitialize_for_restart()
                    continue

                # Calculate metrics for logging/progress
                elapsed_time = time.time() - start_time
                seconds_per_iter = (
                    elapsed_time / (iter + 1) if iter > 0 else elapsed_time
                )
                total_seconds = seconds_per_iter * self.max_iter
                total_hours = total_seconds / 3600
                current_seconds = (
                    elapsed_time / 3600 - int(elapsed_time / 3600)
                ) * 3600

                if len(self.ll) > 1:
                    ll_diff = self.ll[-1] - self.ll[-2]

                    # Always log detailed metrics to the file logger
                    if self.use_grad_norm:
                        detailed_log = (
                            f" iter {iter + 1:5d} lrate = {self.lrate:12.10f} "
                            f"LL = {self.ll[-1]:13.10f} "
                            f"nd = {self.nd[-1]:11.10f}, "
                            f"D = {ll_diff:11.5e} {ll_diff:11.5e}  "
                            f"({current_seconds:5.2f} s, {total_hours:4.1f} h)"
                        )

                        # Always write detailed logs to the file
                        with open(self.file_path, "a") as f:
                            f.write(detailed_log + "\n")

                        # Also log to console if verbose or not using tqdm
                        if self.verbose or not self.use_tqdm:
                            self.logger.info(detailed_log)

                # Check convergence (threads numdecs/numincs back so they
                # accumulate across iterations, and ratchets the lrate ceiling).
                converged, reason, numdecs, numincs = self._check_convergence(
                    numdecs, numincs
                )
                if converged:
                    convergence_reason = reason
                    break

                # Reset the decrease counter when Newton turns on (Fortran
                # amica17.f90:1105-1108).
                if self.do_newton and iter == self.newt_start:
                    numdecs = 0

                # Reject outliers if requested (Fortran amica17.f90:1142). The
                # max(1, ...) clamp matches Fortran and AMICATorchNG: without it,
                # Python's non-negative modulo makes (iter - rejstart) % rejint
                # hit 0 for iter < rejstart, firing rejection before rejstart.
                if (
                    self.do_reject
                    and self.maxrej > 0
                    and (
                        (iter == self.rejstart)
                        or (
                            (max(1, iter - self.rejstart) % self.rejint == 0)
                            and (self.numrej < self.maxrej)
                        )
                    )
                ):
                    self._reject_outliers()
                    self.numrej += 1

                # Share components if requested
                if (
                    self.share_comps
                    and iter >= self.share_start
                    and (iter - self.share_start) % self.share_int == 0
                ):
                    self.comp_list, self.comp_used = identify_shared_components(
                        self.A, self.W, self.comp_list, self.comp_thresh
                    )

                # Write intermediate results if requested
                if self.writestep > 0 and iter % self.writestep == 0:
                    self._write_results()

                # Write history if requested
                if self.do_history and iter % self.histstep == 0:
                    self._write_history()
        finally:
            # Close the progress bar if using tqdm
            if use_tqdm_progress:
                progress_bar.close()

                # Display final metrics after progress bar is closed
                if len(self.ll) > 0 and self.use_grad_norm and len(self.nd) > 0:
                    final_metrics = (
                        f"Final LL: {self.ll[-1]:.6e}, Gradient norm: {self.nd[-1]:.6e}"
                    )
                    self.logger.info(final_metrics)
                    # Also log to file if using tqdm (since it wouldn't be logged during iterations)
                    with open(self.file_path, "a") as f:
                        f.write(final_metrics + "\n")

            # Record and log the reason the loop stopped (None if it ran to
            # max_iter). fit() uses self.converged for the terminal outcome.
            self.stop_reason = convergence_reason
            if convergence_reason:
                self.logger.info(convergence_reason)
                with open(self.file_path, "a") as f:
                    f.write(convergence_reason + "\n")

            # Log final message (only once)
            final_message = f"Optimization finished after {final_iter + 1} iterations"
            self.logger.info(final_message)

    def _check_convergence(
        self, numdecs: int, numincs: int
    ) -> Tuple[bool, Optional[str], int, int]:
        """
        Check convergence criteria and ratchet the learning rate.

        Mirrors Fortran's per-iteration convergence handling
        (amica17.f90:1062-1103). On a likelihood decrease Fortran does NOT stop
        at ``maxdecs``; it lowers the learning-rate *ceilings* (``lrate0``; the
        rho rate once ``iter > newt_start``; ``newtrate`` under Newton) and
        continues, which is what keeps a long run from oscillating and drifting
        past its converged solution (issue #41). The rho ceiling is
        ``self.rholrate`` here (reset to ``rholrate0`` each fit), ratcheted only
        at ``maxdecs`` -- never per-decrease (issue #193). The updated
        ``numdecs``/``numincs`` counters are
        returned so they accumulate across iterations (they previously did not).

        Parameters
        ----------
        numdecs : int
            Consecutive-decrease counter (Fortran ``numdecs``).
        numincs : int
            Consecutive small-increase counter (Fortran ``numincs``).

        Returns
        -------
        converged : bool
            True if optimization should stop.
        reason : str or None
            Reason for stopping, or None.
        numdecs, numincs : int
            The updated counters (must be threaded back into the loop).
        """
        if len(self.ll) == 0:
            return False, None, numdecs, numincs

        # Check for non-finite LL: a singular W makes logdet -> -inf (not NaN),
        # so guard on isfinite, not isnan alone, or a degenerate model would run
        # to max_iter undetected.
        if not np.isfinite(self.ll[-1]):
            return (
                True,
                "Non-finite likelihood (NaN/-inf) encountered",
                numdecs,
                numincs,
            )

        # The remaining checks compare consecutive iterations; skip until there
        # are two LL values -- the first iteration, or the first iteration after
        # a restart cleared the LL history (guard on len, not self.iter, which
        # keeps counting across restarts).
        if len(self.ll) < 2:
            return False, None, numdecs, numincs

        # Gradient norm is computed every iteration, so it is the decrease-stop
        # floor unconditionally (Fortran's ndtmpsum); use_grad_norm only gates
        # the separate final gradient-norm stop below.
        grad_norm = self.nd[-1] if len(self.nd) > 0 else None

        # Likelihood decrease (Fortran amica17.f90:1062-1083): reduce the current
        # lrate, and once maxdecs decreases have accrued, ratchet the ceilings
        # (lrate0; the rho rate; newtrate under Newton) down and continue --
        # NOT stop. Only a lrate/gradient floor terminates on a decrease.
        if self.ll[-1] < self.ll[-2]:
            if self.lrate <= self.minlrate or (
                grad_norm is not None and grad_norm <= self.min_grad_norm
            ):
                return (
                    True,
                    "Converged: minimum change threshold met",
                    numdecs,
                    numincs,
                )
            self.lrate *= self.lratefact
            numdecs += 1
            if numdecs >= self.max_decs:
                self.lrate0 *= self.lratefact
                if self.iter > self.newt_start:
                    # rho rate is a ceiling reset to rholrate0 each iteration
                    # (Fortran amica15.f90:1788); it ratchets ONLY here at maxdecs
                    # (amica15.f90:1050), never per LL-decrease. The old per-decrease
                    # self.rholrate *= rholratefact was a monotone decay with no
                    # reset that collapsed the rho rate and froze rho at a stale
                    # shape (issue #193).
                    self.rholrate *= self.rholratefact
                if self.do_newton and self.iter > self.newt_start:
                    self.newtrate *= self.lratefact
                numdecs = 0

        # Small likelihood increase (Fortran :1084-1096): stop after maxincs
        # consecutive tiny gains; reset on any larger gain.
        if self.use_min_dll:
            if self.ll[-1] - self.ll[-2] < self.min_dll:
                numincs += 1
                if numincs > self.maxincs:
                    return (
                        True,
                        "Converged: small likelihood increase",
                        numdecs,
                        numincs,
                    )
            else:
                numincs = 0

        # Gradient-norm floor (Fortran :1097-1103).
        if (
            self.use_grad_norm
            and grad_norm is not None
            and grad_norm <= self.min_grad_norm
        ):
            return True, "Converged: small gradient norm", numdecs, numincs

        return False, None, numdecs, numincs

    def _reject_outliers(self):
        """Permanently drop samples whose (pre-update) per-sample log-likelihood
        is a low outlier, mirroring ``AMICATorchNG._reject_outliers`` (Fortran
        ``reject_data``, amica17.f90:2380-2464): drop any currently-good sample
        with ``loglik < mean - rejsig*std`` (population std). Rejection is
        one-directional -- ``good_idx`` only shrinks -- and ``num_good_samples``
        normalizes ``gm`` and ``self.ll`` thereafter (issue #212). Dropping the
        most-negative samples raises the mean of what remains regardless of
        normalization (removing below-average points can only raise a mean), so
        the reject iteration is an LL increase and does not spuriously trip the
        convergence checks.

        ``self._last_ll_samples`` is this iteration's per-sample LL over the
        current good set (captured pre-update in ``_get_updates_and_likelihood``,
        in ``good_idx`` order), so the keep-mask indexes straight into
        ``good_idx``.
        """
        if not self.do_reject:
            return
        assert self.good_idx is not None and self._last_ll_samples is not None

        ll_vec = self._last_ll_samples
        mean = ll_vec.mean()
        # Population std (ddof=0), matching np.std's default and the torch path.
        std = np.sqrt(max(float(np.mean(ll_vec**2) - mean**2), 0.0))
        keep = ll_vec >= (mean - self.rejsig * std)

        if not bool(keep.any()):
            # For finite log-likelihoods the max sample is always >= mean >=
            # mean - rejsig*std (rejsig>0 is validated at construction), so it is
            # always kept; the only way every sample is dropped is a non-finite
            # per-sample LL (one NaN poisons mean/std, making every comparison
            # False). Report that accurately instead of blaming rejsig (issue
            # #127), which a user cannot fix by tuning rejsig. In a normal fit()
            # the earlier aggregate non-finite-LL guard (the sum is non-finite
            # iff a term is) stops the loop first, so this mainly serves direct
            # callers of _reject_outliers and is defense in depth.
            n_bad = int(np.count_nonzero(~np.isfinite(ll_vec)))
            if n_bad:
                raise ValueError(
                    f"{n_bad} of {ll_vec.size} samples have a non-finite "
                    "log-likelihood; this indicates numerical instability "
                    "upstream (singular W / overflow), not a rejsig "
                    "miscalibration. Check for rank-deficient or "
                    "average-referenced data, or reduce the learning rate."
                )
            raise ValueError(  # defensive: unreachable for finite LL, rejsig>0
                f"Outlier rejection removed all {self.good_idx.size} samples "
                f"(rejsig={self.rejsig} too aggressive for this data)."
            )

        n_before = self.good_idx.size
        self.good_idx = self.good_idx[keep]
        self.num_good_samples = int(self.good_idx.size)
        self.logger.info(
            "Rejected %d samples (%d good remaining).",
            n_before - self.num_good_samples,
            self.num_good_samples,
        )

    def _write_results(self):
        """Write current results to disk in the Fortran AMICA binary format.

        Writes raw little-endian float64 (and int32 ``comp_list``) files with no
        extension, in the layout that ``load.loadmodout`` reads (and that
        ``load_results`` reads back), so pamica output is loadable by the same
        reader as the Fortran reference (issue #30).

        This shares the ``write_amicaout`` writer with the torch backend, so its
        on-disk guarantees apply here too: single-model output is byte-identical
        to the Fortran ``amicaout`` files, and since #159 multi-model output is
        written in genuine Fortran layout as well (each array's model axis
        slowest, column-major within a model), so EEGLAB's ``loadmodout15.m``
        reads both correctly. See :func:`pamica.numpy_impl.load.write_amicaout`.

        Also writes ``LLt`` (issue #155) via ``_compute_full_posterior_ll``,
        which is called on every ``_write_results`` call -- including every
        mid-training ``writestep`` checkpoint, not just the final write -- so
        each checkpoint now pays a full O(n_samples) forward pass over the
        whole dataset. See ``_compute_full_posterior_ll``'s docstring for the
        cost/tradeoff note and the cheaper stash-during-E-step design tracked
        as a follow-up.
        """
        # A is written (Fortran output omits it; loadmodout derives A from W and
        # S) only so load_results can restore it directly for the viz helpers.
        # The Fortran 'nd' file (per-component weight-change history) is a
        # different quantity from pamica's scalar self.nd, so it is not emitted
        # (loadmodout treats 'nd' as optional).
        from .load import write_amicaout

        Lht, Lt = self._compute_full_posterior_ll()

        write_amicaout(
            self.outdir,
            gm=self.gm,
            W=self.W,
            sphere=self.sphere,
            mean=self.mean,
            c=self.c,
            alpha=self.alpha,
            mu=self.mu,
            sbeta=self.beta,  # Fortran's 'sbeta' is pamica's beta (scale)
            rho=self.rho,
            comp_list=self.comp_list,
            ll=np.asarray(self.ll),
            A=self.A,
            Lht=Lht,
            Lt=Lt,
        )

    def _write_history(self):
        """Write optimization history at current iteration."""
        if not self.do_history:
            return

        assert (
            self.A is not None
            and self.W is not None
            and self.c is not None
            and self.mu is not None
            and self.alpha is not None
            and self.beta is not None
            and self.rho is not None
            and self.gm is not None
            and self.mean is not None
            and self.sphere is not None
            and self.comp_list is not None
        )
        hist_dir = self.outdir / "history" / f"{self.iter:06d}"
        if not hist_dir.exists():
            hist_dir.mkdir(parents=True)

        # Save current state
        np.save(hist_dir / "A.npy", self.A)
        np.save(hist_dir / "W.npy", self.W)
        np.save(hist_dir / "c.npy", self.c)
        np.save(hist_dir / "mu.npy", self.mu)
        np.save(hist_dir / "alpha.npy", self.alpha)
        np.save(hist_dir / "beta.npy", self.beta)
        np.save(hist_dir / "rho.npy", self.rho)
        np.save(hist_dir / "gm.npy", self.gm)
        np.save(hist_dir / "mean.npy", self.mean)
        np.save(hist_dir / "sphere.npy", self.sphere)
        np.save(hist_dir / "comp_list.npy", self.comp_list)
        np.save(hist_dir / "ll.npy", self.ll)
        if self.use_grad_norm:
            np.save(hist_dir / "nd.npy", self.nd)

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Apply the learned unmixing matrices to new data.

        Parameters
        ----------
        data : ndarray of shape (n_channels, n_samples)
            The data to transform

        Returns
        -------
        S : ndarray of shape (n_components, n_samples, n_models)
            The unmixed sources for each model
        """
        if self.W is None or self.comp_list is None or self.c is None:
            raise RuntimeError("Model has not been fitted yet; call fit() first.")

        if self.mean is not None:
            data = data - self.mean

        if self.sphere is not None:
            data = np.dot(self.sphere, data)

        S = np.zeros((self.num_comps, data.shape[1], self.num_models))
        for h in range(self.num_models):
            idx = self.comp_list[:, h]
            # W^T is the true unmixing (issue #24 transpose convention); c is the
            # per-model data-space center, so unmix as W(x - c) (issue #27).
            S[idx, :, h] = np.dot(self.W[:, :, h].T, data - self.c[:, h][:, None])

        return S
