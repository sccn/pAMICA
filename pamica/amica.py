"""
Main AMICA interface using the PyTorch natural-gradient EM backend.

This module provides the primary :class:`AMICA` class, a scikit-learn-style
wrapper over :class:`AMICATorchNG` (the natural-gradient EM port that reaches
Fortran parity; see ``.context/decisions/0001-torch-backend-natural-gradient-em.md``).
"""

import numpy as np
import torch
from pathlib import Path
from typing import Optional, Union
import inspect
import logging

from .torch_impl import AMICATorchNG, setup_device

logger = logging.getLogger(__name__)

# AMICATorchNG's default parameter dtype, derived from its signature so the
# wrapper's MPS/float64 fallback below stays in lockstep if that default ever
# changes (rather than duplicating the literal).
_NG_DEFAULT_DTYPE = inspect.signature(AMICATorchNG).parameters["dtype"].default

# fit()'s own named parameters that a parameter-file dict can default.
_FIT_NAMED_PARAMS = {"max_iter", "lrate", "do_mean", "do_sphere", "do_newton"}

# AMICATorchNG constructor keywords eligible to be defaulted from a
# from_params_file dict via **kwargs (issue #132 review item 2).
# n_channels/n_models/n_mix/device are excluded because fit() always passes
# those positionally in the AMICATorchNG(...) call below; lrate/do_mean/
# do_sphere/do_newton are excluded because, despite also being AMICATorchNG
# constructor parameters, fit() resolves and passes those as its own named
# arguments (_FIT_NAMED_PARAMS) -- merging them into **kwargs too would raise
# "got multiple values for keyword argument".
_NG_CTOR_PARAMS = (
    set(inspect.signature(AMICATorchNG).parameters)
    - {"n_channels", "n_models", "n_mix", "device"}
    - _FIT_NAMED_PARAMS
)

# Sentinel distinguishing "caller did not pass this fit() argument" from
# "caller explicitly passed the same value as the hard default", so a
# from_params_file default can be overridden by an explicit call-site value
# without also being masked by fit()'s own hard-coded defaults.
_UNSET = object()


class AMICA:
    """
    Adaptive Mixture ICA using the PyTorch natural-gradient EM backend.

    This is the main interface for pamica, providing a scikit-learn style
    API over :class:`AMICATorchNG`, the natural-gradient EM implementation
    that matches the Fortran reference (Newton, exact-EM mixture updates,
    symmetric-ZCA sphere, Jacobian LL).

    Parameters
    ----------
    n_models : int, default=1
        Number of ICA models to learn
    n_mix : int, default=3
        Number of mixture components per source
    device : str or torch.device, optional
        Device to use ('cuda', 'mps', 'cpu', or None for auto). With ``None``
        (auto), an auto-selected MPS device is redirected to CPU because the
        backend computes in float64 for Fortran parity and MPS cannot
        represent it; pass ``dtype=torch.float32`` (with ``device="mps"``) to
        run on MPS instead.
    verbose : bool, default=True
        Whether to show progress during fitting

    Attributes
    ----------
    model_ : AMICATorchNG
        The underlying PyTorch model
    is_fitted_ : bool
        Whether a *usable* model is available. ``fit`` sets this True only when
        the fit converged normally; a degenerate fit (see ``converged_``) leaves
        it False, and ``transform``/``get_mixing_matrix``/``get_unmixing_matrix``/
        ``save`` refuse such a model (issue #50).
    converged_ : bool
        Whether the last ``fit`` ended on a usable stop rather than a degenerate
        one (``stop_reason_`` not in ``nan_ll``/``singular_ll``). A degenerate fit
        holds non-finite parameters and would produce NaN sources (issue #50).
    stop_reason_ : str or None
        Why the last ``fit`` stopped (the backend ``stop_reason``):
        ``"max_iter"``, ``"lrate_floor"``, ``"grad_norm_floor"``, ``"min_dll"``,
        ``"grad_norm"``, ``"nan_ll"``, or ``"singular_ll"``. The last five are
        Fortran-faithful convergence stops (issue #207: ``lrate_floor``/
        ``grad_norm_floor`` fire together as two halves of the same
        likelihood-decrease branch; ``min_dll``/``grad_norm`` are separate,
        unconditional per-iteration checks); only ``nan_ll``/``singular_ll``
        are degenerate (see ``converged_``). None of these checks short-
        circuits on an earlier one in the same iteration, so under the
        shipped ``use_grad_norm=True`` default ``"grad_norm"`` always takes
        precedence over ``"grad_norm_floor"`` when both would apply --
        ``"grad_norm_floor"`` only surfaces as this value when
        ``use_grad_norm=False`` (see ``AMICATorchNG``'s ``use_grad_norm``
        docstring for the full explanation).
    ll_history_ : list
        Log-likelihood history during training (the true per-iteration
        trajectory; may dip below its peak on a late overshoot)
    final_ll_ : float
        Log-likelihood of the *fitted* parameters (issue #51). Use this, not
        ``ll_history_[-1]``, as the model's log-likelihood: with the best-iterate
        safeguard the returned parameters can be an earlier, higher-LL iterate.
    mir_history_ : list
        Mutual Information Reduction (MIR) waypoint trajectory (issue #137),
        populated when ``fit`` is called
        with ``mir_step > 0``: ``(iteration, mir_nats, variance)`` tuples from
        the mid-fit ``W``/``sphere``. Like ``ll_history_``, a ``keep_best``
        restore does not rewrite it -- use :meth:`mir` on the fitted model for
        the value of the *returned* parameters, not ``mir_history_[-1]``.
        Not index-aligned with ``ll_history_``: entry ``i`` is computed after
        iteration ``i``'s update, while ``ll_history_[i]`` is the likelihood of
        the parameters before it, so the two are one update apart (issue #161).

    Examples
    --------
    >>> from pamica import AMICA
    >>> import numpy as np
    >>>
    >>> # Generate sample data
    >>> X = np.random.randn(32, 10000)  # 32 channels, 10000 samples
    >>>
    >>> # Fit AMICA model
    >>> amica = AMICA(n_models=1, n_mix=3)
    >>> amica.fit(X, max_iter=100)
    >>>
    >>> # Transform data to sources
    >>> S = amica.transform(X)
    >>>
    >>> # Get mixing matrix
    >>> A = amica.get_mixing_matrix()
    """

    def __init__(
        self,
        n_models: int = 1,
        n_mix: int = 3,
        device: Optional[Union[str, torch.device]] = None,
        verbose: bool = True,
    ):
        self.n_models = n_models
        self.n_mix = n_mix
        self.device = device
        self.verbose = verbose

        self.model_ = None
        self.is_fitted_ = False
        self.ll_history_ = []
        self.final_ll_ = None
        self.stop_reason_ = None
        self.converged_ = False
        self.mir_history_ = []
        # Set by from_params_file (issue #132 review item 2): the full
        # translated parameter-file dict, applied by fit() as per-call
        # defaults (an explicitly passed fit()/AMICATorchNG kwarg always
        # wins). None for an instance built directly via AMICA(...).
        self._file_params: Optional[dict] = None

    def _select_device(self, ng_dtype) -> Union[str, torch.device]:
        """Resolve the compute device, applying the MPS/float64 fallback.

        ``AMICATorchNG`` defaults to float64 for Fortran parity, which MPS
        cannot represent. When the device was auto-selected (the user did not
        pin one) and resolved to MPS for a float64 run, fall back to CPU so the
        default config runs instead of crashing. CUDA supports float64, so only
        MPS needs this. An explicit ``device="mps"`` is left untouched and
        surfaces ``AMICATorchNG``'s own ValueError; users wanting MPS pass
        ``dtype=torch.float32`` too.
        """
        device = setup_device() if self.device is None else self.device
        dev_type = getattr(device, "type", device)
        if self.device is None and dev_type == "mps" and ng_dtype == torch.float64:
            device = torch.device("cpu")
            msg = (
                "AMICA uses float64 for Fortran parity; MPS lacks float64 "
                "support, so falling back to CPU. Pass dtype=torch.float32 "
                "with device='mps' to run on MPS."
            )
            logger.warning(msg)
            if self.verbose:
                print(msg)
        return device

    def fit(
        self,
        X: np.ndarray,
        max_iter=_UNSET,
        lrate=_UNSET,
        do_mean=_UNSET,
        do_sphere=_UNSET,
        do_newton=_UNSET,
        mir_step: int = 0,
        **kwargs,
    ) -> "AMICA":
        """
        Fit AMICA model to data.

        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_channels, n_samples)
        max_iter : int, default=100
            Maximum number of iterations
        lrate : float, default=0.05
            Learning rate
        do_mean : bool, default=True
            Whether to remove mean from data
        do_sphere : bool, default=True
            Whether to sphere (whiten) the data
        do_newton : bool, default=False
            Whether to enable the Fortran-parity Newton preconditioner (tune
            via ``newt_start``/``newtrate`` in ``**kwargs``).
        mir_step : int, default=0
            If > 0, compute MIR every ``mir_step`` iterations during training
            and record it in ``mir_history_`` (issue #137). ``0`` (default)
            disables the waypoints; see :meth:`AMICATorchNG.fit` for details
            and the interaction with ``keep_best``.
        **kwargs
            Additional parameters passed to the :class:`AMICATorchNG`
            constructor (e.g. ``block_size``, ``rho0``, ``seed``, ``dtype``,
            ``use_min_dll``, ``min_dll``, ``maxincs``, ``use_grad_norm``,
            ``min_nd`` -- the issue #207 convergence stops, Fortran-faithful
            defaults ``True``/``1e-9``/``5``/``True``/``1e-7``) -- the
            backend's tunables are constructor arguments, not fit() kwargs.

            Rank-deficient input (Maxwell-filtered MEG, average-referenced or
            interpolated EEG) is handled by ``mineig``/``mineig_rel`` (issue
            #223): the model is sized to the detected numerical rank and
            :meth:`AMICATorchNG.get_sensor_mixing_matrix` maps components back
            to input channels. ``mineig`` is an absolute eigenvalue floor and so
            unit-dependent; pass ``mineig_rel`` for data far from unit scale.

            When the instance was built via :meth:`from_params_file` (issue
            #132), any of the parameters above -- named or in ``**kwargs`` --
            left unset here falls back to that file's translated value instead
            of the hard-coded default; an explicitly passed argument always
            wins over the file. Settings the file carries that match neither a
            named ``fit()`` parameter nor an :class:`AMICATorchNG` constructor
            keyword (data-location metadata like ``files``/``outdir``/
            ``data_dim``, or a setting with no pamica equivalent) are not
            applied and are named in a single ``logger.warning``.

        Returns
        -------
        self : AMICA
            Fitted model
        """
        # Validate input
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got shape {X.shape}")

        n_channels, n_samples = X.shape

        # Apply from_params_file's translated dict as per-call defaults
        # (issue #132 review item 2): an explicitly passed argument here
        # always wins, whether named (max_iter/lrate/do_mean/do_sphere/
        # do_newton, via the _UNSET sentinel) or in **kwargs (AMICATorchNG
        # constructor keywords, via plain dict membership). Settings the file
        # carries that apply to neither surface are named in one warning
        # rather than silently discarded.
        file_params = self._file_params or {}

        def _file_default(explicit, name, hard_default):
            if explicit is not _UNSET:
                return explicit
            return file_params.get(name, hard_default)

        max_iter = _file_default(max_iter, "max_iter", 100)
        lrate = _file_default(lrate, "lrate", 0.05)
        do_mean = _file_default(do_mean, "do_mean", True)
        do_sphere = _file_default(do_sphere, "do_sphere", True)
        do_newton = _file_default(do_newton, "do_newton", False)

        if file_params:
            for key, value in file_params.items():
                if key in _NG_CTOR_PARAMS and key not in kwargs:
                    kwargs[key] = value
            handled = _FIT_NAMED_PARAMS | _NG_CTOR_PARAMS | {"num_models", "num_mix"}
            unhandled = sorted(set(file_params) - handled)
            if unhandled:
                logger.warning(
                    "AMICA.fit: %d parameter-file setting(s) match neither a "
                    "fit()/AMICATorchNG parameter and were NOT applied "
                    "(informational only -- data-location metadata like "
                    "files/outdir/data_dim/num_comps is expected here; "
                    "anything else means pamica has no equivalent): %s",
                    len(unhandled),
                    unhandled,
                )

        if self.verbose:
            print(f"Fitting AMICA with {n_channels} channels, {n_samples} samples")
            print(f"Models: {self.n_models}, Mixture components: {self.n_mix}")

        # Setup device (with the MPS/float64 parity fallback, see _select_device).
        device = self._select_device(kwargs.get("dtype", _NG_DEFAULT_DTYPE))

        # Build and train the backend on a LOCAL reference first, and only
        # publish it to self (and derive the fitted-state attributes) once
        # fit() returns. If the backend constructor or fit() raises mid-training
        # (a numerical crash, OOM, singular sphere, interrupt, ...), self is left
        # untouched: a first fit keeps model_ is None (so the output methods
        # raise a clean "not fitted"), and a refit keeps the previous, known-good
        # model rather than a half-trained one falsely marked usable (issue #50
        # silent-failure review).
        backend = AMICATorchNG(
            n_channels=n_channels,
            n_models=self.n_models,
            n_mix=self.n_mix,
            lrate=lrate,
            do_mean=do_mean,
            do_sphere=do_sphere,
            do_newton=do_newton,
            device=device,
            **kwargs,
        )
        backend.fit(X, max_iter=max_iter, verbose=self.verbose, mir_step=mir_step)

        self.model_ = backend
        self.ll_history_ = backend.ll_history
        self.final_ll_ = backend.final_ll_
        self.stop_reason_ = backend.stop_reason
        self.mir_history_ = backend.mir_history_
        self.converged_ = self.stop_reason_ not in AMICATorchNG._DEGENERATE_STOP_REASONS
        # A degenerate fit (nan_ll/singular_ll) holds non-finite parameters and
        # would return NaN sources, so it is not a usable model: is_fitted_ stays
        # False and the output methods refuse it (issue #50). stop_reason_/
        # converged_ stay set for inspection.
        self.is_fitted_ = self.converged_
        if not self.converged_:
            logger.warning(
                "AMICA.fit ended degenerate (stop_reason=%r) at iteration %d: the "
                "model holds non-finite parameters and cannot transform. Inspect "
                "stop_reason_/ll_history_; lower lrate, disable Newton, or check "
                "data conditioning, then refit.",
                self.stop_reason_,
                backend.iteration,
            )

        return self

    def _check_usable(self, action: str) -> None:
        """Raise if the model cannot produce valid output: either never fitted,
        or the fit ended degenerate (``nan_ll``/``singular_ll``), leaving
        non-finite parameters that would yield NaN sources. This mirrors
        :meth:`AMICATorchNG.state_dict`'s refusal to serialize a degenerate model
        (issue #50): a diverged fit fails loudly here instead of silently
        returning garbage."""
        if self.model_ is None:
            raise ValueError(f"Model must be fitted before {action}.")
        if self.stop_reason_ in AMICATorchNG._DEGENERATE_STOP_REASONS:
            raise RuntimeError(
                f"Refusing to {action}: fit ended degenerate "
                f"(stop_reason={self.stop_reason_!r}), so the model holds "
                f"non-finite parameters and would produce NaN output. Lower "
                f"lrate, disable Newton, or check data conditioning, then refit."
            )

    def transform(self, X: np.ndarray, model_idx: int = 0) -> np.ndarray:
        """
        Transform data to source space.

        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_channels, n_samples)
        model_idx : int, default=0
            Which model to use for transformation

        Returns
        -------
        S : np.ndarray
            Sources of shape (n_sources, n_samples)
        """
        self._check_usable("transform")
        assert self.model_ is not None

        return self.model_.transform(X, model_idx=model_idx)

    def fit_transform(self, X: np.ndarray, **fit_params) -> np.ndarray:
        """
        Fit model and transform data.

        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_channels, n_samples)
        **fit_params
            Parameters passed to fit()

        Returns
        -------
        S : np.ndarray
            Sources of shape (n_sources, n_samples)
        """
        self.fit(X, **fit_params)
        return self.transform(X)

    def get_mixing_matrix(self, model_idx: int = 0) -> np.ndarray:
        """
        Get the mixing matrix A.

        Parameters
        ----------
        model_idx : int, default=0
            Which model's mixing matrix to return

        Returns
        -------
        A : np.ndarray
            Mixing matrix of shape (n_channels, n_sources)
        """
        self._check_usable("get the mixing matrix")
        assert self.model_ is not None

        return self.model_.get_mixing_matrix(model_idx=model_idx)

    def get_unmixing_matrix(self, model_idx: int = 0) -> np.ndarray:
        """
        Get the unmixing matrix W.

        Parameters
        ----------
        model_idx : int, default=0
            Which model's unmixing matrix to return

        Returns
        -------
        W : np.ndarray
            Unmixing matrix of shape (n_sources, n_channels)
        """
        self._check_usable("get the unmixing matrix")
        assert self.model_ is not None

        return self.model_.get_unmixing_matrix(model_idx=model_idx)

    def mir(
        self, X: np.ndarray, model_idx: int = 0, nbins: Optional[int] = None
    ) -> tuple:
        """
        Mutual Information Reduction (issue #137) of the fitted unmixing on ``X``.

        Composes the full raw-data-to-sources transform (unmixing @ sphere)
        the documented way and delegates to :func:`pamica.metrics.mir`.

        Parameters
        ----------
        X : np.ndarray
            Raw (unpreprocessed) data of shape (n_channels, n_samples)
        model_idx : int, default=0
            Which model's unmixing to use
        nbins : int, optional
            Histogram bin count; see :func:`pamica.metrics.mir`

        Returns
        -------
        mir_nats : float
            Mutual information removed, in nats.
        variance : float
            Variance of the estimate.

        Raises
        ------
        ValueError
            If the model is unfitted; or if PCA reduction (``pcakeep``/
            ``pcadb``) is active, which leaves the sphere rank-deficient so
            MIR's log-Jacobian term is undefined; or if ``X`` is non-finite or
            has a constant channel. See :meth:`AMICATorchNG.mir` and
            :func:`pamica.metrics.mir`.
        RuntimeError
            If the fit ended degenerate (issue #50), since the parameters are
            non-finite and any metric from them would be meaningless.
        """
        self._check_usable("compute MIR")
        assert self.model_ is not None

        return self.model_.mir(X, model_idx=model_idx, nbins=nbins)

    def pmi(
        self, X: np.ndarray, model_idx: int = 0, nbins: Optional[int] = None
    ) -> np.ndarray:
        """
        Pairwise Mutual Information (issue #137) between the fitted sources on ``X``.

        Delegates to :func:`pamica.metrics.pairwise_mi` on
        ``transform(X, model_idx)``.

        Parameters
        ----------
        X : np.ndarray
            Raw (unpreprocessed) data of shape (n_channels, n_samples)
        model_idx : int, default=0
            Which model's sources to use
        nbins : int, optional
            Histogram bin count; see :func:`pamica.metrics.pairwise_mi`

        Returns
        -------
        mi_matrix : np.ndarray of shape (n_sources, n_sources)
            Symmetric pairwise mutual information, in nats. The diagonal is each
            source's own entropy, not a mutual information; see
            :func:`pamica.viz.plot_pmi_heatmap`, which masks it by default.

        Raises
        ------
        ValueError
            If the model is unfitted; or if ``X`` is non-finite or has a
            constant channel; or if ``nbins`` is too large for the sample count
            (a sparse joint histogram fabricates mutual information from noise).
            See :func:`pamica.metrics.pairwise_mi`.
        RuntimeError
            If the fit ended degenerate (issue #50).
        """
        self._check_usable("compute PMI")
        assert self.model_ is not None

        return self.model_.pmi(X, model_idx=model_idx, nbins=nbins)

    def model_loglik(self, X: np.ndarray) -> np.ndarray:
        """Per-model, per-sample log-likelihood ``Lht`` on ``X`` (issue #141).

        Delegates to :meth:`AMICATorchNG.model_loglik`. For a multi-model fit
        this is the joint log-likelihood of each model at each sample, from
        which the per-sample model posterior (dominance) is
        ``softmax(Lht, axis=0)``; see :meth:`model_probability`.

        Parameters
        ----------
        X : np.ndarray of shape (n_channels, n_samples)
            Raw (unpreprocessed) data.

        Returns
        -------
        Lht : np.ndarray of shape (n_models, n_samples)

        Raises
        ------
        ValueError
            If the model is unfitted, or if ``X`` is non-finite.
        RuntimeError
            If the fit ended degenerate (issue #50).
        """
        self._check_usable("compute the model log-likelihood")
        assert self.model_ is not None

        return self.model_.model_loglik(X)

    def model_probability(self, X: np.ndarray) -> np.ndarray:
        """Per-sample posterior probability of each model (issue #141).

        Delegates to :meth:`AMICATorchNG.model_probability`: the column-wise
        ``softmax`` over models of :meth:`model_loglik`, i.e. ``P(model h |
        x_t)``. Each column sums to 1; all ones for a single model.

        Parameters
        ----------
        X : np.ndarray of shape (n_channels, n_samples)
            Raw (unpreprocessed) data.

        Returns
        -------
        prob : np.ndarray of shape (n_models, n_samples)

        Raises
        ------
        ValueError
            If the model is unfitted, if ``X`` is non-finite, or if every model
            underflows to ``-inf`` log-likelihood at some sample.
        RuntimeError
            If the fit ended degenerate (issue #50).
        """
        self._check_usable("compute the model probability")
        assert self.model_ is not None

        return self.model_.model_probability(X)

    def get_pdftype(self, model_idx: int = 0) -> np.ndarray:
        """Per-source density-family code for model ``model_idx`` (issue #142).

        Delegates to :meth:`AMICATorchNG.get_pdftype`. One integer per source
        component (0-4; see :data:`pamica.torch_impl.PDFTYPE_NAMES`).

        Returns
        -------
        np.ndarray of int, shape (n_sources,)
        """
        self._check_usable("get the density family")
        assert self.model_ is not None

        return self.model_.get_pdftype(model_idx=model_idx)

    def get_rho(self, model_idx: int = 0) -> np.ndarray:
        """Generalized-Gaussian shape ``rho`` for model ``model_idx`` (issue #142).

        Delegates to :meth:`AMICATorchNG.get_rho`.

        Returns
        -------
        np.ndarray of float, shape (n_mix, n_sources)
        """
        self._check_usable("get rho")
        assert self.model_ is not None

        return self.model_.get_rho(model_idx=model_idx)

    def shared_components(self) -> list:
        """Components shared across models by ``share_comps`` (issue #142).

        Delegates to :meth:`AMICATorchNG.shared_components`: one group of
        ``(model_idx, source_idx)`` pairs per shared column; empty when nothing
        is shared.
        """
        self._check_usable("get the shared components")
        assert self.model_ is not None

        return self.model_.shared_components()

    def variance_order(
        self, model_idx: int = 0, return_svar: bool = False
    ) -> Union[np.ndarray, tuple]:
        """
        Component order by EEGLAB back-projected variance (IC1 = highest).

        Reports the display order EEGLAB's ``loadmodout15.m`` applies on load,
        without mutating the fitted parameters. Apply it to the columns of
        :meth:`get_mixing_matrix` (or rows of :meth:`get_unmixing_matrix`) to get
        EEGLAB-ordered components in Python.

        Parameters
        ----------
        model_idx : int, default=0
            Which model's components to order.
        return_svar : bool, default=False
            If True, also return the per-source variance sorted to ``order``.

        Returns
        -------
        order : np.ndarray of int
            Source indices, highest back-projected variance first.
        """
        self._check_usable("compute the variance order")
        assert self.model_ is not None

        return self.model_.variance_order(model_idx=model_idx, return_svar=return_svar)

    def write_amica_output(self, outdir: str) -> None:
        """
        Write the fitted model as an EEGLAB-readable AMICA output directory.

        Emits the raw binary files that EEGLAB's ``loadmodout15.m`` reads (``W``,
        ``S``, ``gm``, ``mean``, ``c``, ``alpha``, ``mu``, ``sbeta``, ``rho``,
        ``comp_list``, ``LL``), so a pamica fit drops directly into an EEGLAB
        workflow (``mod = loadmodout15(outdir)``). ``loadmodout15`` applies the
        variance-ordering and normalization on load, so no manual re-ordering or
        sign-flipping is needed. Single-model output is byte-compatible with the
        Fortran reference (issue #92).

        Also writes ``LLt`` (the per-sample/per-model log-likelihood, issue
        #155) for a model that was just fit in this process; a model restored
        via :meth:`load` has no training data to recompute it from, so
        ``LLt`` is omitted for it (a warning is logged).

        Parameters
        ----------
        outdir : str
            Destination directory (created if absent).
        """
        self._check_usable("write EEGLAB output")
        assert self.model_ is not None

        self.model_.write_amica_output(outdir)

    def save(self, filepath: str) -> None:
        """
        Save the fitted model to ``filepath`` via ``torch.save``.

        Persists the underlying :class:`AMICATorchNG` state (config + fitted
        tensors) plus the wrapper's own configuration, so :meth:`load` can
        fully reconstruct a transform-ready model. Everything written is a
        tensor or plain Python primitive (see
        :meth:`AMICATorchNG.state_dict`), so it reloads with
        ``weights_only=True``.

        Parameters
        ----------
        filepath : str
            Destination path (a ``.pt`` file by convention).
        """
        self._check_usable("save")
        assert self.model_ is not None

        payload = {
            "format_version": 1,
            "wrapper": {
                "n_models": self.n_models,
                "n_mix": self.n_mix,
                "verbose": self.verbose,
            },
            "backend": self.model_.state_dict(),
        }
        torch.save(payload, filepath)

    @classmethod
    def load(
        cls, filepath: str, device: Optional[Union[str, torch.device]] = None
    ) -> "AMICA":
        """
        Load a fitted model saved by :meth:`save`.

        Parameters
        ----------
        filepath : str
            Path to a file written by :meth:`save`.
        device : str or torch.device, optional
            Device to place the restored model on. With ``None`` (auto), the
            same MPS/float64 fallback as :meth:`fit` applies so a float64
            parity model never lands on MPS.

        Returns
        -------
        amica : AMICA
            A fitted model ready for :meth:`transform` / :meth:`get_mixing_matrix`.
        """
        payload = torch.load(filepath, weights_only=True)
        version = payload.get("format_version")
        if version != 1:
            raise ValueError(
                f"unsupported AMICA save format_version: {version!r} (expected 1)"
            )
        for key in ("wrapper", "backend"):
            if key not in payload:
                raise ValueError(
                    f"malformed AMICA save file {filepath!r}: missing {key!r} "
                    f"(format_version={version}); the file may be truncated or "
                    f"corrupted."
                )

        wrapper = payload["wrapper"]
        model = cls(
            n_models=wrapper["n_models"],
            n_mix=wrapper["n_mix"],
            device=device,
            verbose=wrapper["verbose"],
        )

        # Resolve the device using the persisted backend dtype so the same
        # MPS/float64 fallback as fit() applies to an auto-selected device.
        ng_dtype = getattr(torch, payload["backend"]["config"]["dtype"])
        resolved_device = model._select_device(ng_dtype)
        model.model_ = AMICATorchNG.from_state_dict(
            payload["backend"], device=resolved_device
        )
        model.ll_history_ = model.model_.ll_history
        model.final_ll_ = model.model_.final_ll_
        # mir_history_ is not persisted in state_dict() (a diagnostic
        # trajectory, not a fitted parameter), so a loaded model's is always
        # empty; expose it anyway for attribute-surface consistency with
        # ll_history_.
        model.mir_history_ = model.model_.mir_history_
        # state_dict() refuses to serialize a degenerate model, so a loaded model
        # is always usable; carry its stop_reason through for inspection anyway.
        model.stop_reason_ = model.model_.stop_reason
        model.converged_ = (
            model.stop_reason_ not in AMICATorchNG._DEGENERATE_STOP_REASONS
        )
        model.is_fitted_ = model.converged_
        return model

    @classmethod
    def from_params_file(cls, params_file: str, **kwargs) -> "AMICA":
        """
        Create AMICA instance from a parameter file.

        Accepts two formats, auto-detected from ``params_file``'s *content*
        (issue #132): pamica's own JSON schema (``sample_data/
        sample_params.json``) and the literal Fortran ``input.param`` text
        format (``sample_data/input.param``), so the same file that drives
        the reference binary can drive pamica too. Detection always sniffs
        the content (JSON if it starts with ``{``/``[``, Fortran text
        otherwise) rather than trusting the extension -- a compact JSON file
        saved with a ``.param`` extension must still parse as JSON, not be
        silently mis-read as garbled Fortran text (issue #132 review item 3).
        See :func:`pamica.fortran_params.read_fortran_param_file` for the
        Fortran-side key-mapping table and the deliberately-unmapped keys it
        warns about rather than silently drops.

        The full translated dict (beyond the ``n_models``/``n_mix`` used to
        size the instance here) is stashed on the returned instance and
        applied by :meth:`fit` as per-call defaults -- see ``fit``'s
        docstring for the precedence rule.

        Parameters
        ----------
        params_file : str
            Path to a JSON or Fortran-format parameter file.
        **kwargs
            Additional parameters to override.

        Returns
        -------
        amica : AMICA
            Configured AMICA instance
        """
        import json

        path = Path(params_file)
        text = path.read_text()
        if text.lstrip().startswith(("{", "[")):
            params = json.loads(text)
        else:
            from .fortran_params import read_fortran_param_file

            params = read_fortran_param_file(path)

        # Extract relevant parameters
        n_models = params.get("num_models", 1)
        n_mix = params.get("num_mix", 3)

        # Override with kwargs
        n_models = kwargs.pop("n_models", n_models)
        n_mix = kwargs.pop("n_mix", n_mix)

        model = cls(n_models=n_models, n_mix=n_mix, **kwargs)
        model._file_params = params
        return model
