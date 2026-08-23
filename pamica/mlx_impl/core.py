"""MLX natural-gradient EM backend for AMICA (issue #76/#81, epic #74 Phase C/D).

``AMICAMLXNG`` is a port of :class:`pamica.torch_impl.core.AMICATorchNG`
to Apple's MLX array framework, so the per-block E/M-step runs on the Apple GPU.
It is structurally parallel to the PyTorch backend (same method names/order and
Fortran citations) so the two can be diffed method-by-method.

Design constraints (verified against MLX 0.32, see ``.context/mps_pathways.md``):

* **float32 only on the GPU** -- Apple GPUs have no FP64, and MLX raises on GPU
  float64. Full-data float32 converges thanks to the ``ufp/y`` divide-by-zero
  guard from issue #75, which is reproduced here.
* **All ``mlx.core.linalg`` is CPU-stream only** -- ``inv(A)`` and ``slogdet(W)``
  run under ``mx.stream(mx.cpu)``. They are hoisted to once per iteration (the
  PyTorch ``_forward`` recomputes ``slogdet`` inside the block loop; here it is a
  cached constant), so the GPU pipeline sees one cross-stream handoff per
  iteration, not one per block.
* **No ``lgamma``/``digamma`` in MLX** -- the GG normalizer ``lgamma(1+1/rho)``
  and the rho-update ``digamma(1+1/rho)`` depend only on the small ``rho`` array,
  so they are computed host-side with SciPy once per iteration.

Scope: single- and multi-model (``n_models >= 1``, issue #81), generalized
Gaussian (``pdftype=0``), natural gradient (``do_newton=False``), component
sharing (``share_comps``, issue #263). Newton and the other PDF families are
rejected in ``__init__`` with a clear ``NotImplementedError`` (``transform``
likewise). Outlier rejection, ``keep_best`` and save/load are simply absent (no
such parameter/method) -- all fast-follows.

Convergence stops (issue #248) are the full AMICATorchNG/Fortran set:
``use_min_dll``/``min_dll``/``maxincs``, ``use_grad_norm``/``min_nd``, and the
likelihood-decrease branch's ``ndtmpsum <= min_nd`` half -- same parameter names,
same defaults, same ``stop_reason`` strings (``"min_dll"``, ``"grad_norm"``,
``"grad_norm_floor"``), so a configuration moved from the PyTorch backend does the
same work here. The gradient norm ``ndtmpsum`` is computed every iteration and
masked by ``comp_used`` (Fortran amica15.f90:1761); without component sharing
every column is used, so the mask is all-True and drops out exactly.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

# mlx ships as a compiled extension with no type stubs, so ty cannot resolve
# it statically even when installed; scope the suppression to this one import.
import mlx.core as mx  # ty: ignore[unresolved-import]
import numpy as np
from scipy.special import digamma, gammaln

from ..numpy_impl.utils import identify_shared_components
from ..rank import MINEIG, MINEIG_REL, numerical_rank

logger = logging.getLogger(__name__)

_LOG2 = math.log(2.0)
# Fortran epsdble: zero the rho*ln|y| term when |y|^rho underflows below this
# (amica17.f90:1570), matching AMICATorchNG.
_EPSDBLE = 1e-16
# MLX linalg runs on the CPU stream only (float32-accurate); the GPU stream
# raises "not yet supported on the GPU" for inv/slogdet/eigh/solve.
_CPU = mx.cpu


def _score_gg(y: mx.array, rho: mx.array) -> mx.array:
    """GG score ``fp = rho*sign(y)*|y|^(rho-1)`` (AMICATorchNG ``_score``, GG
    branch, core.py:176-183). ``fp(0)=0`` for ``rho>=1``, which the ``ufp/y``
    guard relies on."""
    abs_y = mx.abs(y)
    sign_y = mx.sign(y)
    fp_gg = rho * sign_y * mx.power(abs_y, rho - 1.0)
    # rho is generically in (1, 2); keep the exact Laplace/Gaussian endpoints.
    return mx.where(rho == 2.0, 2.0 * y, mx.where(rho == 1.0, sign_y, fp_gg))


def _log_pdf_gg(
    y: mx.array, rho: mx.array, lgamma_table: mx.array
) -> tuple[mx.array, mx.array]:
    """GG log-density and ``|y|^rho`` (AMICATorchNG ``_log_pdf_only``, GG branch,
    core.py:216-224). ``lgamma_table = lgamma(1+1/rho)`` is precomputed host-side
    (MLX has no ``lgamma``); it makes the uniform GG form reduce to the exact
    Laplace (rho=1) and Gaussian (rho=2) log-densities."""
    abs_y = mx.abs(y)
    az_rho = mx.power(abs_y, rho)  # reused by the rho-update accumulator
    log_pdf = -az_rho - _LOG2 - lgamma_table
    return log_pdf, az_rho


class AMICAMLXNG:
    """MLX natural-gradient EM backend (GG, single- and multi-model; #76/#81).

    Parameters mirror the subset of :class:`AMICATorchNG` that is supported;
    the same ``seed`` produces the same initial parameters as the PyTorch/NumPy
    backends, so cross-backend equivalence is testable.

    The convergence-stop parameters (issue #248) carry AMICATorchNG's names,
    defaults and semantics exactly, so the two backends stop on the same
    iteration for the same reason:

    ``use_min_dll`` (True) / ``min_dll`` (1e-9) / ``maxincs`` (5)
        Stop once the per-sample-per-channel log-likelihood gain
        ``ll_history[-1] - ll_history[-2]`` stays below ``min_dll`` for more than
        ``maxincs`` *consecutive* iterations (Fortran amica15.f90:1078-1090);
        ``stop_reason="min_dll"``. The counter resets on any larger gain and a
        likelihood decrease counts as a small gain, as in Fortran.
    ``use_grad_norm`` (True) / ``min_nd`` (1e-7)
        Stop once the weight-gradient RMS norm ``ndtmpsum`` falls to or below
        ``min_nd`` (Fortran amica15.f90:1091-1097); ``stop_reason="grad_norm"``.
        The same threshold is also the second half of the likelihood-decrease
        stop (amica15.f90:1058, ``stop_reason="grad_norm_floor"``), which runs
        regardless of ``use_grad_norm``. Under the shipped defaults
        ``"grad_norm"`` shadows ``"grad_norm_floor"`` -- see AMICATorchNG's
        ``use_grad_norm`` docstring, whose precedence note applies verbatim.
        ``min_nd`` is not reachable on small recordings in any backend (issue
        #218); the Fortran-faithful default is kept rather than retuned.

    All three checks require two log-likelihood values, so none can fire on the
    first iteration (Fortran's ``if (iter > 1)``, amica15.f90:1051).
    """

    def __init__(
        self,
        n_channels: int,
        n_models: int = 1,
        n_mix: int = 3,
        block_size: int = 8192,
        lrate: float = 0.1,
        minlrate: float = 1e-12,
        lratefact: float = 0.5,
        maxdecs: int = 5,
        use_min_dll: bool = True,
        min_dll: float = 1e-9,
        maxincs: int = 5,
        use_grad_norm: bool = True,
        min_nd: float = 1e-7,
        newt_ramp: int = 10,
        newt_start: int = 20,
        do_newton: bool = False,
        rho0: float = 1.5,
        minrho: float = 1.0,
        maxrho: float = 2.0,
        rholrate: float = 0.05,
        rholratefact: float = 0.1,
        pdftype: int = 0,
        invsigmin: float = 1e-4,
        invsigmax: float = 1000.0,
        doscaling: bool = True,
        scalestep: int = 1,
        share_comps: bool = False,
        share_start: int = 100,
        share_iter: int = 100,
        comp_thresh: float = 0.99,
        do_mean: bool = True,
        do_sphere: bool = True,
        do_approx_sphere: bool = True,
        mineig: float = MINEIG,
        mineig_rel: Optional[float] = MINEIG_REL,
        seed: Optional[int] = None,
    ):
        # --- Boundaries: reject the still-deferred configurations up front. ---
        if pdftype != 0:
            raise NotImplementedError(
                "AMICAMLXNG supports the generalized-Gaussian family "
                "(pdftype=0) only; the other families are a fast-follow."
            )
        if do_newton:
            raise NotImplementedError(
                "AMICAMLXNG supports the natural gradient only "
                "(do_newton=False); Newton is a fast-follow."
            )

        self.n_channels = n_channels
        self.n_models = n_models  # multi-model (#81) + component sharing (#263)
        self.n_mix = n_mix
        self.n_comps = n_channels * n_models
        self.block_size = block_size

        self.lrate0 = lrate
        self.lrate = lrate
        self.lrate_cap = lrate
        self.minlrate = minlrate
        self.lratefact = lratefact
        self.maxdecs = maxdecs
        # Convergence stops (issue #248), same names/defaults/semantics as
        # AMICATorchNG: the small-likelihood-gain stop (use_min_dll/min_dll/
        # maxincs) and the weight-gradient-norm stop (use_grad_norm/min_nd). See
        # fit() for the per-iteration checks and _update_parameters for the
        # ndtmpsum computation they both read.
        if maxincs < 0:
            raise ValueError(f"maxincs must be >= 0, got {maxincs}")
        self.use_min_dll = use_min_dll
        self.min_dll = min_dll
        self.maxincs = maxincs
        self.use_grad_norm = use_grad_norm
        self.min_nd = min_nd
        self.newt_ramp = newt_ramp
        # Schedule threshold for the rholrate ceiling ratchet (Fortran
        # amica15.f90:1067 gates it on iter > newt_start, independent of
        # do_newton). MLX does no Newton, but keeps newt_start as this gate so the
        # rho-rate schedule matches Fortran/torch.
        self.newt_start = newt_start
        self.do_newton = False

        self.rho0 = rho0
        self.minrho = minrho
        self.maxrho = maxrho
        self.rholrate0 = rholrate
        self.rholrate = rholrate
        self.rholratefact = rholratefact
        self.dorho = True  # GG shape adapts (Fortran dorho, pdftype==0)

        self.pdftype = 0
        self.invsigmin = invsigmin
        self.invsigmax = invsigmax
        self.doscaling = doscaling
        self.scalestep = scalestep

        # Component sharing (issue #263), same names/defaults/validation as
        # AMICATorchNG (torch_impl/core.py). OFF by default and inert for
        # n_models=1 (a model cannot share a component with itself), so the
        # default trajectory is untouched.
        self.share_comps = share_comps
        self.share_start = share_start
        self.share_iter = share_iter
        self.comp_thresh = comp_thresh
        if share_comps:
            if share_start < 1:
                raise ValueError(f"share_start must be >= 1, got {share_start}")
            if share_iter <= 6:
                # The A-freeze settle window is 6 iterations; a smaller cycle
                # would freeze A permanently (never leaving room to update it).
                raise ValueError(f"share_iter must be > 6, got {share_iter}")
            if not 0.0 < comp_thresh <= 1.0:
                raise ValueError(f"comp_thresh must be in (0, 1], got {comp_thresh}")

        self.do_mean = do_mean
        self.do_sphere = do_sphere
        self.do_approx_sphere = do_approx_sphere
        # Numerical-rank floors (issue #223); see pamica/rank.py and ADR 0004.
        self.mineig = mineig
        self.mineig_rel = mineig_rel
        self.seed = seed

        self.iteration = 0
        self.ll_history: list[float] = []
        self.final_ll_: Optional[float] = None
        self.stop_reason: Optional[str] = None

        # Populated by fit()/_initialize_parameters().
        self.A: Optional[mx.array] = None
        self.W: Optional[mx.array] = None
        self.mu: Optional[mx.array] = None
        self.alpha: Optional[mx.array] = None
        self.beta: Optional[mx.array] = None
        self.rho: Optional[mx.array] = None
        self.gm: Optional[mx.array] = None
        self.c: Optional[mx.array] = None
        self.comp_list: Optional[mx.array] = None
        self.mean: Optional[mx.array] = None
        self.sphere: Optional[mx.array] = None
        # float64 host copy of the sphere, kept because the sharing metric's
        # back-map pinv(sphere) must be computed at the precision the sphere was
        # built with, not from the float32 GPU cast (issue #263). Set alongside
        # self.sphere in _preprocess, which is also where _sphere_pinv is
        # invalidated (the single point that assigns a sphere -- this backend has
        # no load path).
        self._sphere_np: Optional[np.ndarray] = None
        self._sphere_pinv: Optional[np.ndarray] = None
        # comp_used mask, CACHED here rather than derived per call; see the
        # comp_used property.
        self._comp_used_arr: Optional[mx.array] = None
        self.sldet = 0.0
        self._lgamma_table: Optional[mx.array] = (
            None  # (n_mix, n_comps): lgamma(1+1/rho)
        )
        self._logdet_W: Optional[mx.array] = (
            None  # scalar: log|det W|, refreshed per iter
        )
        # Weight-gradient norm (Fortran ndtmpsum), recomputed every iteration by
        # _update_parameters and read by fit()'s two grad-norm checks. Held as
        # the unevaluated MLX scalar rather than a Python float (AMICATorchNG's
        # eager ``_ndtmpsum`` float) so materializing it joins fit()'s single
        # per-iteration mx.eval instead of adding a second sync; the
        # ``_ndtmpsum`` property below is the float view the checks and
        # cross-backend tests read.
        self._nd_arr: Optional[mx.array] = None

    _DEGENERATE_STOP_REASONS = ("nan_ll", "singular_ll", "nan_params")

    @property
    def _ndtmpsum(self) -> Optional[float]:
        """Latest weight-gradient norm as a host float (AMICATorchNG's
        ``_ndtmpsum``), or None before the first M-step. Cheap after fit()'s
        mx.eval has materialized it; forces evaluation otherwise, so a direct
        ``_update_parameters`` call still reads the current iteration's value."""
        if self._nd_arr is None:
            return None
        return float(self._nd_arr.item())

    # ------------------------------------------------------------------
    # Preprocessing (host / numpy; mirrors AMICATorchNG._preprocess in float64)
    # ------------------------------------------------------------------
    def _preprocess(self, X: np.ndarray) -> mx.array:
        """Mean-removal + sphering in float64 on the host, then handed to MLX as
        float32. Done in numpy (not MLX) because it reuses the exact float64
        preprocessing AMICATorchNG already validates, so the sphere/sldet match
        the PyTorch backend. (MLX's CPU-stream ``eigh`` is full float64 -- only
        the GPU stream is unsupported -- so this is a code-sharing choice, not a
        precision workaround.)"""
        Xc = np.ascontiguousarray(X).astype(np.float64)
        data_dim = Xc.shape[0]

        if self.do_mean:
            mean = Xc.mean(axis=1, keepdims=True)
            Xc = Xc - mean
        else:
            mean = np.zeros((data_dim, 1))

        if self.do_sphere:
            # Population covariance (/N), matching Fortran's DSYRK scatter, not
            # numpy's default sample covariance (/(N-1)) -- see core.py:635-639.
            cov = np.cov(Xc, bias=True)
            evals, evecs = np.linalg.eigh(cov)
            order = np.argsort(evals)[::-1]
            evals = evals[order]
            evecs = evecs[:, order]
            # Numerical rank, decided by the policy shared with the PyTorch and
            # NumPy backends (pamica/rank.py, issue #223) so the three cannot
            # disagree. Fortran: numeigs = min(pcakeep, count(eigs > mineig)).
            n_comp = numerical_rank(
                evals, mineig=self.mineig, mineig_rel=self.mineig_rel
            )
            evals = evals[:n_comp]
            V = evecs[:, :n_comp]
            inv_sqrt = np.diag(1.0 / np.sqrt(evals))
            if n_comp < data_dim:
                # Rank-reduced sphere (n_comp, data_dim), so the sphered data
                # come out at the kept rank (Fortran nw = numeigs,
                # amica15.f90:563).
                w_pca = inv_sqrt @ V.T
                if self.do_approx_sphere:
                    # Fortran's orthogonal polar-factor symmetrization of the
                    # reduced whitening (amica15.f90:501-508).
                    U_b, _, Vt_b = np.linalg.svd(evecs.T[:n_comp, :n_comp])
                    sphere = (Vt_b.T @ U_b.T) @ w_pca
                else:
                    sphere = w_pca
                self.n_channels = n_comp
                self.n_comps = n_comp * self.n_models
            elif self.do_approx_sphere:
                # Symmetric ZCA sphere V diag(1/sqrt) V^T (Fortran default).
                sphere = V @ inv_sqrt @ V.T
            else:
                sphere = inv_sqrt @ V.T
            Xc = sphere @ Xc
            sldet = float(-0.5 * np.log(evals).sum())
        else:
            sphere = np.eye(data_dim)
            sldet = 0.0

        self.mean = mx.array(mean.astype(np.float32))
        self.sphere = mx.array(sphere.astype(np.float32))
        # Keep the float64 sphere for _pinv_sphere and invalidate its cached
        # pseudo-inverse here, so the cache can never describe a sphere other
        # than the current one (AMICATorchNG._preprocess does the same).
        self._sphere_np = sphere
        self._sphere_pinv = None
        self.sldet = sldet
        return mx.array(Xc.astype(np.float32))

    # ------------------------------------------------------------------
    # Initialization (identical RNG draws to AMICATorchNG for cross-backend test)
    # ------------------------------------------------------------------
    def _initialize_parameters(self):
        """Initialize parameters with the *same* ``np.random.RandomState`` draw
        order as AMICATorchNG/AMICA_NumPy (core.py:688-725), so a shared seed
        gives a bit-identical (float32-cast) starting point."""
        rng = np.random.RandomState(self.seed)
        n, m, ncomp, nmix = self.n_channels, self.n_models, self.n_comps, self.n_mix

        # Per-model mixing blocks + comp_list mapping each (channel, model) to its
        # column in A (identical RNG draw order to AMICATorchNG; for m=1 the loop
        # runs once, so single-model init stays byte-for-byte).
        A_np = np.zeros((n, ncomp))
        comp_list_np = np.zeros((n, m), dtype=np.int64)
        for h in range(m):
            A_np[:, h * n : (h + 1) * n] = np.eye(n) + 0.01 * (0.5 - rng.rand(n, n))
            comp_list_np[:, h] = np.arange(h * n, (h + 1) * n)

        mu_np = np.zeros((nmix, ncomp))
        for k in range(ncomp):
            mu_np[:, k] = np.linspace(-1, 1, nmix)
            mu_np[:, k] += 0.05 * (1 - 2 * rng.rand(nmix))

        alpha_np = np.ones((nmix, ncomp)) / nmix
        beta_np = np.ones((nmix, ncomp)) + 0.1 * (0.5 - rng.rand(nmix, ncomp))
        rho_np = self.rho0 * np.ones((nmix, ncomp))

        self.A = mx.array(A_np.astype(np.float32))
        self.comp_list = mx.array(comp_list_np)  # (n_channels, n_models) int
        # Every column is referenced by the default block comp_list; reset here
        # (not only in __init__) so a re-fit cannot inherit a merged mask.
        self._comp_used_arr = mx.array(np.ones(ncomp, dtype=bool))
        self.mu = mx.array(mu_np.astype(np.float32))
        self.alpha = mx.array(alpha_np.astype(np.float32))
        self.beta = mx.array(beta_np.astype(np.float32))
        self.rho = mx.array(rho_np.astype(np.float32))
        self.gm = mx.array((np.ones(m) / m).astype(np.float32))
        self.c = mx.array(np.zeros((n, m), dtype=np.float32))

        self.lrate = self.lrate0
        self.lrate_cap = self.lrate0
        self.rholrate = self.rholrate0
        self.iteration = 0
        self._refresh_lgamma_table()
        self._update_unmixing_matrices()

    def _refresh_lgamma_table(self):
        """Recompute ``lgamma(1+1/rho)`` host-side (MLX has no lgamma). Called at
        init and after every rho update. Cheap: rho is ``(n_mix, n_comps)``."""
        rho_np = np.array(self.rho, dtype=np.float64)
        self._lgamma_table = mx.array(gammaln(1.0 + 1.0 / rho_np).astype(np.float32))

    def _update_unmixing_matrices(self):
        """Per-model ``W_h = inv(A[:, comp_list[:, h]])`` and the LL Jacobian
        ``log|det W_h|``, on the CPU stream (MLX linalg is CPU-only), hoisted to
        once per iteration. ``W`` is ``(n_models, n, n)`` and ``_logdet_W`` is
        ``(n_models,)``. For n_models=1 this is ``inv(A)`` unchanged.

        These build lazy graph nodes; a singular ``A`` therefore raises not here
        but where the graph is materialized (the ``mx.eval`` in ``fit``), so a
        LinAlg traceback rooted in ``fit`` actually originates in this method.
        """
        assert self.A is not None and self.comp_list is not None
        ws, logdets = [], []
        for h in range(self.n_models):
            wh = mx.linalg.inv(self.A[:, self.comp_list[:, h]], stream=_CPU)
            ws.append(wh)
            logdets.append(mx.linalg.slogdet(wh, stream=_CPU)[1])
        self.W = mx.stack(ws, axis=0)  # (n_models, n, n)
        self._logdet_W = mx.stack(logdets)  # (n_models,)

    # ------------------------------------------------------------------
    # E-step
    # ------------------------------------------------------------------
    def _forward(self, Xb: mx.array):
        """E-step forward pass for one block, per model (AMICATorchNG._forward,
        core.py:762-825). ``Xb`` is ``(n_channels, batch)``. Returns ``logV``
        ``(batch, n_models)`` and per-model lists ``(b, z, y, az_rho)``. For
        n_models=1 (c=0, gm=1, comp_list=identity) this is numerically identical
        to the single-model path."""
        assert (
            self.comp_list is not None
            and self.c is not None
            and self.W is not None
            and self.mu is not None
            and self.beta is not None
            and self.rho is not None
            and self.alpha is not None
            and self._lgamma_table is not None
            and self.gm is not None
            and self._logdet_W is not None
        )
        b_list, z_list, y_list, azrho_list, logv_cols = [], [], [], [], []
        for h in range(self.n_models):
            idx = self.comp_list[:, h]
            b = (Xb - self.c[:, h][:, None]).T @ self.W[h]  # (batch, n_channels)
            mu_h = self.mu[:, idx].T[None]  # (1, n_channels, n_mix)
            beta_h = self.beta[:, idx].T[None]
            rho_h = self.rho[:, idx].T[None]
            alpha_h = self.alpha[:, idx].T[None]
            lgamma_h = self._lgamma_table[:, idx].T[None]

            y = beta_h * (b[..., None] - mu_h)  # (batch, n_channels, n_mix)
            log_pdf, az_rho = _log_pdf_gg(y, rho_h, lgamma_h)
            z0 = mx.log(alpha_h) + mx.log(beta_h) + log_pdf
            ll_i = mx.logsumexp(z0, axis=-1)  # (batch, n_channels)
            z = mx.softmax(z0, axis=-1)
            logv_cols.append(
                mx.log(self.gm[h]) + self._logdet_W[h] + self.sldet + ll_i.sum(axis=-1)
            )
            b_list.append(b)
            z_list.append(z)
            y_list.append(y)
            azrho_list.append(az_rho)
        logV = mx.stack(logv_cols, axis=1)  # (batch, n_models)
        return logV, b_list, z_list, y_list, azrho_list

    def _get_block_updates(self, Xb: mx.array) -> dict:
        """Exact-EM sufficient statistics for one block (non-Newton subset of
        AMICATorchNG._get_block_updates, core.py:833-953). Mixture stats are
        scattered into their ``comp_list`` columns; ``dWtmp``/``dgm``/``dc_numer``
        are per-model. For n_models=1 (v==1, identity comp_list) this reproduces
        the single-model accumulators exactly."""
        logV, b_list, z_list, y_list, azrho_list = self._forward(Xb)
        block_ll = mx.logsumexp(logV, axis=1).sum()
        v = mx.softmax(logV, axis=1)  # (batch, n_models) model responsibilities
        nmix, ncomp = self.n_mix, self.n_comps
        tiny = float(np.finfo(np.float32).tiny)

        def zeros():
            return mx.zeros((nmix, ncomp), dtype=mx.float32)

        dalpha_n, dmu_n, dmu_d = zeros(), zeros(), zeros()
        dbeta_n, dbeta_d, drho_n = zeros(), zeros(), zeros()
        dgm_cols, dwtmp_mods, dc_cols = [], [], []

        assert (
            self.comp_list is not None
            and self.beta is not None
            and self.rho is not None
        )
        for h in range(self.n_models):
            idx = self.comp_list[:, h]
            b, zr, y, az_rho = b_list[h], z_list[h], y_list[h], azrho_list[h]
            v_h = v[:, h]
            beta_h = self.beta[:, idx].T[None]  # (1, n_channels, n_mix)
            rho_h = self.rho[:, idx].T[None]

            fp = _score_gg(y, rho_h)
            u = v_h[:, None, None] * zr  # u = v*z, (batch, n_channels, n_mix)
            ufp = u * fp

            dgm_cols.append(v_h.sum())
            dalpha_n = dalpha_n.at[:, idx].add(u.sum(0).T)
            dmu_n = dmu_n.at[:, idx].add(ufp.sum(0).T)
            # Phase A guard: float32 can round y to exactly 0 (fp(0)=0 => ufp=0),
            # so ufp/y is 0/0=NaN; where y==0, 0/1 contributes 0 (issue #75).
            safe_y = mx.where(y == 0, mx.ones_like(y), y)
            dmu_d = dmu_d.at[:, idx].add((beta_h[0] * (ufp / safe_y).sum(0)).T)
            dbeta_n = dbeta_n.at[:, idx].add(u.sum(0).T)
            dbeta_d = dbeta_d.at[:, idx].add((ufp * y).sum(0).T)

            logab = rho_h * mx.log(mx.maximum(mx.abs(y), tiny))
            logab = mx.where(az_rho < _EPSDBLE, mx.zeros_like(logab), logab)
            drho_n = drho_n.at[:, idx].add((u * (az_rho * logab)).sum(0).T)

            g = (beta_h * ufp).sum(-1)  # (batch, n_channels)
            dwtmp_mods.append(g.T @ b)  # (n_channels, n_channels)
            dc_cols.append(Xb @ v_h)  # data-space bias numerator sum_t v_h*x

        return {
            "dgm": mx.stack(dgm_cols),  # (n_models,)
            "dalpha_n": dalpha_n,
            "dmu_n": dmu_n,
            "dmu_d": dmu_d,
            "dbeta_n": dbeta_n,
            "dbeta_d": dbeta_d,
            "drho_n": drho_n,
            "dWtmp": mx.stack(dwtmp_mods, axis=0),  # (n_models, n_ch, n_ch)
            "dc_numer": mx.stack(dc_cols, axis=1),  # (n_channels, n_models)
            "ll": block_ll,
        }

    def _accumulate_blocks(self, X: mx.array) -> dict:
        """Sum sufficient statistics over all blocks as one lazy graph (no
        per-block ``mx.eval`` -- that over-syncs 2.6x)."""
        n_samples = X.shape[1]
        acc: Optional[dict] = None
        for start in range(0, n_samples, self.block_size):
            end = min(start + self.block_size, n_samples)
            block_acc = self._get_block_updates(X[:, start:end])
            if acc is None:
                acc = block_acc
            else:
                for key in acc:
                    acc[key] = acc[key] + block_acc[key]
        assert acc is not None
        return acc

    # ------------------------------------------------------------------
    # M-step
    # ------------------------------------------------------------------
    def _update_parameters(self, acc: dict, n_samples: int):
        """Exact-EM mixture updates + natural-gradient A-update (non-Newton subset
        of AMICATorchNG._update_parameters, core.py:1049-1247)."""
        # Fortran builds dAk from the PREVIOUS iteration's model weights: gm is
        # not reassigned until update_params (amica15.f90:1788+), after
        # accum_updates_and_likelihood (:1731-1743). Snapshot before overwriting,
        # as AMICATorchNG does (issue #219); MLX arrays are immutable and gm is
        # only ever rebound, so a plain rebinding is a safe snapshot (torch
        # clones because its tensors could be written in place). Exactly gm for
        # n_models=1 (both are 1.0) and cancelling for a disjoint comp_list, so
        # the single-model and unshared multi-model paths are unchanged.
        assert self.gm is not None
        gm_prev = self.gm
        self.gm = acc["dgm"] / n_samples  # (n_models,); == 1 for single model
        tiny = float(np.finfo(np.float32).tiny)

        # Per-model data-space bias c[i,h] = sum_t v_h*x / sum_t v_h (Fortran
        # update_c, core.py:1083-1092). Skipped for n_models=1 (v==1 => c is the
        # zero data mean; the update would add a float-sum residual and break the
        # #24 bit-exact single-model path). A dead model (dgm[h]==0) keeps its
        # prior c rather than writing 0/0, and is surfaced (matching AMICATorchNG).
        if self.n_models > 1:
            dgm = acc["dgm"]
            live = dgm > 0.0
            new_c = acc["dc_numer"] / mx.maximum(dgm, tiny)[None, :]
            self.c = mx.where(live[None, :], new_c, self.c)
            if not bool(mx.all(live).item()):
                logger.warning(
                    "Zero-responsibility model(s) at iter %d; kept their prior "
                    "bias c (dead-model guard).",
                    self.iteration,
                )

        # Component sharing (#263): a component merged away by
        # _identify_shared_comps is no longer referenced by comp_list, so no
        # sufficient statistic accumulates into its column (dalpha_n/dmu_d/
        # dbeta_d == 0) and the divisions below would be 0/0 = NaN -- which the
        # nan_params guard in fit() would (correctly) abort on. Update only USED
        # columns and freeze the rest at their last finite value (Fortran carries
        # NaN there behind its comp_used mask; keeping them finite matches
        # AMICATorchNG and the NumPy backend, docs/guides/amica-differences.md
        # row 8). With the default full comp_list every column is used, so
        # ``used`` is all-True and every update below is bit-for-bit unchanged.
        assert self._comp_used_arr is not None
        used = self._comp_used_arr[None, :]  # (1, n_comps)

        self.alpha = mx.where(
            used,
            acc["dalpha_n"] / acc["dalpha_n"].sum(axis=0, keepdims=True),
            self.alpha,
        )
        self.mu = mx.where(used, self.mu + acc["dmu_n"] / acc["dmu_d"], self.mu)
        self.beta = mx.where(
            used,
            mx.clip(
                self.beta * mx.sqrt(acc["dbeta_n"] / acc["dbeta_d"]),
                self.invsigmin,
                self.invsigmax,
            ),
            self.beta,
        )

        # GG shape update with the 1/psi(1+1/rho) digamma factor (Fortran
        # :2013-2014); digamma is computed host-side (MLX has none). A NaN here
        # (e.g. from an upstream mu/beta blow-up) is reset to rho0 and surfaced,
        # matching AMICATorchNG (core.py:1140-1151), so it does not silently
        # poison the lgamma table and every subsequent E-step.
        if self.dorho:
            drho = acc["drho_n"] / mx.maximum(acc["dalpha_n"], 1e-8)
            rho_np = np.array(self.rho, dtype=np.float64)
            psi = mx.array(digamma(1.0 + 1.0 / rho_np).astype(np.float32))
            new_rho = self.rho + self.rholrate * (1.0 - (self.rho / psi) * drho)
            nan_mask = mx.isnan(new_rho)
            if bool(mx.any(nan_mask).item()):
                logger.warning(
                    "NaN in rho update at iter %d; resetting to rho0=%g.",
                    self.iteration,
                    self.rho0,
                )
                new_rho = mx.where(nan_mask, self.rho0, new_rho)
            # ``used`` freezes a merged-away column's rho, as for mu/alpha/beta.
            self.rho = mx.where(
                used, mx.clip(new_rho, self.minrho, self.maxrho), self.rho
            )

        # Natural-gradient A-update. A is stored as Fortran's A^T, so the update
        # is a LEFT-multiply by the transposed direction (core.py:1176-1184,
        # #24 root cause). Each model's direction is scattered into its mixing
        # columns as a gm-weighted average (Fortran dAk/zeta, core.py:1231-1247)
        # using the PREVIOUS iteration's gm (gm_prev, see the snapshot above):
        # for the default disjoint comp_list every column has one contributor, so
        # gm cancels and n_models=1 is byte-for-byte the old `A - lrate*(dA.T@A)`;
        # a SHARED column (#263) takes Fortran's responsibility-weighted average,
        # NOT a raw sum (a raw sum would over-step by the contributor count). A
        # merged-away column needs no special case: nothing scatters into it, so
        # its zeta is 0, dAk is 0/tiny = 0, and the rescale below leaves a
        # zero-norm column untouched.
        #
        # The direction/dAk/gradient-norm computation below runs
        # UNCONDITIONALLY, not gated on _a_frozen(): Fortran computes dAk and
        # ndtmpsum every iteration in accum_updates_and_likelihood
        # (amica15.f90:1749-1761), strictly before the separate, share-freeze
        # guarded update_A block (:1803) that steps A. Only the step itself --
        # and the lrate ramp Fortran nests inside that same guarded block -- are
        # conditional (issue #207: the grad-norm stop must see the true gradient
        # magnitude every iteration, not only when A moves). _a_frozen() is
        # always False with sharing off, so the default path is unchanged.
        eye = mx.eye(self.n_channels)
        directions = [
            -acc["dWtmp"][h] / acc["dgm"][h] + eye for h in range(self.n_models)
        ]
        assert self.A is not None and self.comp_list is not None
        dAk = mx.zeros_like(self.A)
        zeta = mx.zeros((self.n_comps,), dtype=mx.float32)
        for h in range(self.n_models):
            idx = self.comp_list[:, h]
            dAk = dAk.at[:, idx].add(gm_prev[h] * (directions[h].T @ self.A[:, idx]))
            zeta = zeta.at[idx].add(gm_prev[h] + mx.zeros((self.n_channels,)))
        dAk = dAk / mx.maximum(zeta, tiny)

        # Weight-gradient norm (Fortran ndtmpsum, amica15.f90:1760-1761):
        # ``sqrt(sum(dAk**2, mask=comp_used) / (nw*count(comp_used)))``, built
        # from the step direction BEFORE the lrate scaling and before the A step
        # applies it, exactly as Fortran does in accum_updates_and_likelihood
        # (:1749-1761) ahead of update_params' A step (:1803-1815). Read by
        # fit()'s two grad-norm checks (AMICATorchNG._update_parameters computes
        # the same quantity). The comp_used mask matters only once share_comps
        # has merged columns away: without sharing it is all-True, so the select
        # returns nd unchanged and the count is n_comps, leaving this bit-for-bit
        # the plain RMS over dAk. Kept as a lazy scalar (no .item() here) so it
        # rides fit()'s single per-iteration mx.eval.
        #
        # SELECT, not multiply-by-mask: 0*NaN is NaN, so a single non-finite
        # column would poison the whole reduction, and a NaN ndtmpsum silently
        # disables BOTH grad-norm stops (NaN <= min_nd is False), burning the
        # entire iteration budget with no diagnostic. mx.where drops the masked
        # lanes structurally instead. Unreachable today (a merged-away column's
        # dAk is exactly 0), but Phase 3's Newton direction feeds this same dAk.
        used_f = self._comp_used_arr.astype(mx.float32)
        nd = (dAk**2).sum(axis=0)  # (n_comps,)
        nd = mx.where(self._comp_used_arr, nd, mx.zeros_like(nd))
        self._nd_arr = mx.sqrt(
            nd.sum() / (self.n_channels * mx.maximum(used_f.sum(), 1.0))
        )

        # A-update. When sharing holds A this iteration (the post-merge settle
        # window, Fortran amica15.f90:1803), skip the step -- the lrate ramp and
        # the step itself.
        if not self._a_frozen():
            self.lrate = min(
                self.lrate_cap, self.lrate + min(1.0 / self.newt_ramp, self.lrate)
            )
            self.A = self.A - self.lrate * dAk

        if self.doscaling and (self.iteration % self.scalestep == 0):
            scale = mx.sqrt((self.A**2).sum(axis=0))  # (n_comps,)
            # A zero-norm (collapsed) column is left untouched, not rescaled:
            # safe_scale is 1 there, so A/beta are unchanged and mu*safe_scale
            # keeps its prior value (matching AMICATorchNG's nonzero mask,
            # core.py:1229-1234 -- using raw `scale` would zero mu instead).
            safe_scale = mx.where(scale > 0, scale, mx.ones_like(scale))
            self.A = self.A / safe_scale
            self.mu = self.mu * safe_scale
            self.beta = self.beta / safe_scale

        self._refresh_lgamma_table()
        self._update_unmixing_matrices()

    # ------------------------------------------------------------------
    # Component sharing (issue #263; AMICATorchNG's #60 port)
    # ------------------------------------------------------------------
    def _a_frozen(self) -> bool:
        """Whether the A-update (and its lrate ramp) is held this iteration.

        A is frozen for the first 6 iterations of every ``share_iter``-length
        window once ``iter >= share_start`` -- the merge iteration and the 5
        after it -- so the density parameters can settle onto any freshly merged
        component before the mixing matrix moves again (Fortran A-freeze,
        amica15.f90:1803). The window fires each cycle regardless of whether that
        cycle's :meth:`_identify_shared_comps` actually merged a pair.

        Anchored on ``(itf - share_start) % share_iter`` so it stays aligned with
        the merge schedule for any ``share_start``; the literal Fortran formula
        uses ``mod(iter, share_iter)`` (misaligned unless share_start is a
        multiple of share_iter, and a permanent freeze for ``share_iter <= 6``),
        but that path is dead in the reference (see :meth:`_identify_shared_comps`)
        so there is no parity constraint -- the constructor requires
        ``share_iter > 6`` so the window never consumes the whole cycle. Gated
        behind ``share_comps`` and ``n_models >= 2``, so with sharing off it is
        always False and the validated default trajectory is untouched.
        """
        if not self.share_comps or self.n_models < 2:
            return False
        itf = self.iteration + 1  # Fortran-style 1-indexed iteration
        if itf < self.share_start:
            return False
        return (itf - self.share_start) % self.share_iter <= 5

    def _identify_shared_comps(self) -> None:
        """Merge near-collinear mixing columns across models (Fortran
        ``identify_shared_comps``, amica15.f90:1916).

        Two components (model ``h`` source ``i`` and model ``hh`` source ``ii``,
        ``h < hh``) are identified when the angle between their mixing columns,
        measured in the original (de-sphered) data space, is below the
        ``comp_thresh`` cutoff; on a match ``cj`` is folded into ``ci``, so the
        two share one mixing column and one density.

        The decision itself is NOT reimplemented here: it runs
        :func:`pamica.numpy_impl.utils.identify_shared_components` on host
        float64 arrays, the same kernel the NumPy backend calls and the one whose
        decisions are pinned equal to ``AMICATorchNG._identify_shared_comps``
        (issue #258). A merge scan is a tiny greedy quadruple loop over
        ``n_models^2 * n_channels^2`` pairs, so nothing is gained by keeping it
        on the GPU, and a third copy of the metric is exactly the cross-backend
        divergence risk .rules/backend_parity.md forbids. ``A`` has just been
        materialized by fit's per-iteration ``mx.eval``, so the host pull is
        cheap.

        No bit-exact oracle: the reference's ``Spinv2`` metric is *declared* but
        never *allocated* in ``amica15.f90``, so invoking the routine there would
        read an unallocated array -- it is effectively unrunnable (cf. the dead
        ``do_choose_pdfs`` switch, #26). This implements the intended algorithm
        and is validated on real data, not against byte parity.
        """
        if self.n_models < 2:
            return
        assert self.A is not None and self.comp_list is not None
        # _pinv_sphere raises on a non-finite sphere, so the metric below can
        # only be garbage if A itself is (guarded per-pair inside the kernel).
        atil = self._pinv_sphere() @ np.array(self.A, dtype=np.float64)
        cl = np.array(self.comp_list)
        new_cl, new_used = identify_shared_components(atil, cl, self.comp_thresh)
        # Each fold removes exactly one column from the referenced set, so the
        # drop in unique count IS the merge count (the kernel does not report it).
        merged = int(np.unique(cl).size - np.unique(new_cl).size)
        if merged:
            self.comp_list = mx.array(new_cl)
            self._comp_used_arr = mx.array(new_used)
            logger.info(
                "Component sharing (iter %d): %d merge(s), %d unique components.",
                self.iteration,
                merged,
                int(np.unique(new_cl).size),
            )

    def _pinv_sphere(self) -> np.ndarray:
        """Cached ``pinv(sphere)``: the back-map from sphered to input-channel
        space, in host float64.

        This is the Fortran ``Spinv`` (amica15.f90:568-578), which the reference
        also builds as a pseudo-inverse, ``Spinv(nx, numeigs)``, under rank/PCA
        reduction. A pseudo-inverse rather than an inverse because reduction
        leaves the sphere non-square (issue #223) and a square sphere fitted on
        rank-deficient data is singular; for a full-rank square sphere the two
        agree to ~1e-15. Computed from ``_sphere_np``, the float64 sphere
        ``_preprocess`` builds before the float32 GPU cast, so the merge metric
        keeps the precision the PyTorch and NumPy backends use for it. Built on
        first use and invalidated per fit in :meth:`_preprocess`, so it can never
        describe a sphere other than the current one.
        """
        if self._sphere_np is None:
            raise RuntimeError(
                "AMICAMLXNG._pinv_sphere() requires a preprocessed model; call "
                "fit() first."
            )
        if self._sphere_pinv is None:
            if not np.all(np.isfinite(self._sphere_np)):
                # Only a degenerate fit (non-finite input data) gets here. Say
                # so, rather than letting LAPACK report a confusing
                # "ill-conditioned / repeated singular values" SVD failure.
                raise RuntimeError(
                    "The sphere holds non-finite values, so it has no "
                    "pseudo-inverse: the fit is degenerate. Check the input "
                    "data for NaN/inf."
                )
            self._sphere_pinv = np.linalg.pinv(self._sphere_np)
        return self._sphere_pinv

    @property
    def comp_used(self) -> mx.array:
        """Boolean mask (n_comps,) of components still referenced by comp_list.

        A component drops out of use when it is folded into another by
        :meth:`_identify_shared_comps`; unused columns receive no gradient and
        are never read by the E-step.

        CACHED (set all-True at init, rewritten by each merge) rather than
        derived from ``comp_list`` on every read, which is how
        ``AMICATorchNG.comp_used`` does it. Same semantics and same source of
        truth: the merge kernel already derives the mask host-side, from the
        merged ``comp_list``, as part of the decision it returns -- so caching
        that result costs nothing and keeps the mask fixed between merges, which
        is exactly the lifetime the M-step needs.
        """
        if self._comp_used_arr is None:
            raise RuntimeError(
                "AMICAMLXNG.comp_used requires a fitted model; call fit() first."
            )
        return self._comp_used_arr

    def shared_components(self) -> list:
        """Components shared across models by ``share_comps`` (issue #263).

        ``share_comps`` folds near-collinear components of different models onto
        one shared mixing column + density, recorded as a repeated index in
        ``comp_list``. Returns one group per shared column: a list of
        ``(model_idx, source_idx)`` pairs that all reference it. Empty when no
        component is shared across two or more models (always for one model, and
        for a default multi-model fit with ``share_comps`` off).

        Returns
        -------
        list of list of tuple(int, int)
        """
        if self.comp_list is None:
            raise RuntimeError(
                "AMICAMLXNG.shared_components() requires a fitted model; call "
                "fit() first."
            )
        cl = np.array(self.comp_list)  # (n_channels, n_models)
        groups = []
        for col in np.unique(cl):
            src, mdl = np.where(cl == col)
            if np.unique(mdl).size >= 2:
                groups.append([(int(h), int(i)) for i, h in zip(src, mdl)])
        return groups

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(
        self, X: np.ndarray, max_iter: int = 100, verbose: bool = True
    ) -> "AMICAMLXNG":
        """Fit the model. ``X`` is ``(n_channels, n_samples)``."""
        if X.ndim != 2:
            raise ValueError(f"X must be 2D (n_channels, n_samples), got {X.shape}")
        if X.shape[0] != self.n_channels:
            raise ValueError(
                f"X has {X.shape[0]} channels, model expects {self.n_channels}"
            )

        X_t = self._preprocess(X)
        n_total = X_t.shape[1]
        self._initialize_parameters()
        self.ll_history = []
        self.stop_reason = "max_iter"
        numdecs = 0
        # Consecutive-small-likelihood-gain counter for the min_dll stop (Fortran
        # numincs, amica15.f90:1079-1089). Reset here so a refit starts clean.
        numincs = 0

        rng = range(max_iter)
        if verbose:
            try:
                from tqdm import tqdm

                rng = tqdm(rng, desc="AMICA-MLX")
            except ImportError:
                pass

        for it in rng:
            self.iteration = it
            acc = self._accumulate_blocks(X_t)

            ll_arr = acc["ll"] / (n_total * self.n_channels)
            mx.eval(ll_arr)  # materialize the accumulate graph once
            ll = float(ll_arr.item())
            if not math.isfinite(ll):
                self.stop_reason = "nan_ll" if math.isnan(ll) else "singular_ll"
                logger.warning(
                    "Non-finite log-likelihood (%s) at iteration %d; stopping.", ll, it
                )
                break

            self._update_parameters(acc, n_total)
            # One eval per iteration bounds the lazy graph to a single iteration's
            # worth of ops (the updated params feed the next accumulate). gm/c are
            # included so their dependency chain is materialized each iteration too
            # (c depends on the prior iteration's c), not left to grow unbounded.
            # _nd_arr (the grad-norm stops' input) rides along here rather than
            # being materialized inside _update_parameters, so reading it below
            # costs no extra sync.
            mx.eval(
                self.A,
                self.W,
                self.mu,
                self.alpha,
                self.beta,
                self.rho,
                self.gm,
                self.c,
                self._nd_arr,
            )

            # Surface a corrupted M-step (component collapse / float32 overflow)
            # at the iteration it happens. The ll check above only catches a
            # corruption via the NEXT iteration's E-step, so a final-iteration
            # blow-up would otherwise complete as max_iter with silently NaN
            # parameters (the torch backend has state_dict as a backstop; the
            # MLX backend does not, so guard in fit()). Params are already
            # materialized by the mx.eval above, so this is a cheap read.
            # _nd_arr is included as defense in depth: a non-finite gradient norm
            # silently disables both grad-norm stops (NaN <= min_nd is False), so
            # it must not be the one quantity nothing checks.
            checked = {
                "A": self.A,
                "mu": self.mu,
                "alpha": self.alpha,
                "beta": self.beta,
                "rho": self.rho,
                "gm": self.gm,
                "c": self.c,
                "ndtmpsum": self._nd_arr,
            }
            params_finite = mx.array(True)
            for value in checked.values():
                params_finite = params_finite & mx.all(mx.isfinite(value))
            if not bool(params_finite.item()):
                # Name the offenders. Everything here is already materialized, so
                # the per-tensor reads add no mid-graph sync -- this is the MLX
                # stand-in for AMICATorchNG's inline mu/beta/alpha canary
                # (torch_impl/core.py:1461-1474), which MLX cannot afford inside
                # _update_parameters because it would sync the lazy graph.
                bad = [
                    name
                    for name, value in checked.items()
                    if not bool(mx.all(mx.isfinite(value)).item())
                ]
                logger.warning(
                    "Non-finite %s at iter %d (a mixture component likely "
                    "collapsed); stopping.",
                    ", ".join(bad),
                    it,
                )
                self.stop_reason = "nan_params"
                break

            # Component sharing (Fortran identify_shared_comps schedule,
            # amica15.f90:1856): once per share_iter cycle from share_start,
            # merge near-collinear mixing columns across models using the
            # just-updated A. Fortran runs identify_shared_comps BEFORE
            # get_unmixing_matrices (amica15.f90:1858,1863), so rebuild W from
            # the merged comp_list -- otherwise the next E-step would read a
            # stale W (pre-merge comp_list) while indexing the densities by the
            # merged comp_list. No-op when share_comps is off or n_models == 1.
            if self.share_comps:
                itf = it + 1
                if (
                    itf >= self.share_start
                    and (itf - self.share_start) % self.share_iter == 0
                ):
                    self._identify_shared_comps()
                    self._update_unmixing_matrices()

            self.ll_history.append(ll)

            # Learning-rate control (Fortran amica17.f90:1062-1108): anneal on an
            # LL decrease; ratchet the ceilings after maxdecs persistent decreases.
            #
            # rholrate is a maxdecs-ratcheted CEILING, not a per-decrease-annealed
            # working rate. Fortran resets rholrate=rholrate0 every iteration before
            # the rho update (amica15.f90:1806/1813) and only tightens the rholrate0
            # ceiling at maxdecs (amica15.f90:1068, gated on iter > newt_start), so
            # its per-decrease rholrate*=rholratefact (:1045) never reaches the rho
            # update. rho has no ramp, so self.rholrate carries that ceiling directly
            # (reset to rholrate0 at fit start, nothing re-inflates it) and must
            # ratchet ONLY at maxdecs. The previous per-decrease decay collapsed the
            # rho rate to ~1e-5 within a few hundred iterations and froze rho at a
            # stale shape (issue #195, mirroring the torch/numpy fix in #193/#194).
            #
            # have_prev mirrors Fortran's outer ``if (iter > 1)``
            # (amica15.f90:1051), which wraps the decrease branch AND the two
            # stops below, so none of the three can fire on the first iteration.
            #
            # PRECEDENCE NOTE (mirroring the same note in AMICATorchNG.fit): the
            # three blocks are independent -- none is gated on ``leave`` already
            # being True from an earlier block this same iteration, matching
            # Fortran's own structure of independent ``leave = .true.``
            # assignments with no declared precedence. Whichever block runs LAST
            # and finds its condition true wins the reported stop_reason, so with
            # this source order the standalone grad_norm block always has final
            # say; under the shipped use_grad_norm=True default that makes the
            # decrease branch's "grad_norm_floor" unreachable as the FINAL reason
            # (its condition is strictly narrower). Deliberately not restructured
            # into an explicit precedence, to keep this a direct port of
            # amica15.f90:1051-1098.
            have_prev = len(self.ll_history) > 1
            leave = False
            if have_prev and ll < self.ll_history[-2]:
                if self.lrate <= self.minlrate:
                    logger.warning(
                        "lrate floor (%g) reached at iter %d; stopping.",
                        self.minlrate,
                        it,
                    )
                    self.stop_reason = "lrate_floor"
                    leave = True
                elif self._ndtmpsum is not None and self._ndtmpsum <= self.min_nd:
                    # Fortran amica15.f90:1058's ``.or. (ndtmpsum .le. min_nd)``
                    # half of the decrease stop (issue #207 gap 3, #248 here):
                    # the same per-iteration value use_grad_norm reads below, so
                    # a run whose lrate oscillates instead of annealing still
                    # stops instead of burning the whole budget.
                    logger.warning(
                        "gradient-norm floor (%g) reached at iter %d on a "
                        "likelihood decrease; stopping.",
                        self.min_nd,
                        it,
                    )
                    self.stop_reason = "grad_norm_floor"
                    leave = True
                else:
                    self.lrate *= self.lratefact
                    numdecs += 1
                    if numdecs >= self.maxdecs:
                        self.lrate_cap *= self.lratefact
                        if it > self.newt_start:
                            self.rholrate *= self.rholratefact
                        numdecs = 0

            # Small-likelihood-increase stop (Fortran amica15.f90:1078-1090,
            # use_min_dll/min_dll/maxincs). Independent of the decrease branch
            # above: it runs every iteration once have_prev, including iterations
            # where the LL just decreased (a decrease is always "less than" a
            # positive min_dll, so it also increments numincs there, matching
            # Fortran exactly). numincs resets to 0 on any gain >= min_dll; stops
            # only after MORE than maxincs *consecutive* small gains.
            if have_prev and self.use_min_dll:
                if ll - self.ll_history[-2] < self.min_dll:
                    numincs += 1
                    if numincs > self.maxincs:
                        logger.warning(
                            "likelihood increasing by less than %g for more than "
                            "%d iterations; stopping at iter %d.",
                            self.min_dll,
                            self.maxincs,
                            it,
                        )
                        self.stop_reason = "min_dll"
                        leave = True
                else:
                    numincs = 0

            # Weight-gradient-norm stop (Fortran amica15.f90:1091-1097,
            # use_grad_norm/min_nd). Also independent of the decrease branch:
            # this is the unconditional every-iteration check, as opposed to the
            # decrease-gated grad_norm_floor above.
            if (
                have_prev
                and self.use_grad_norm
                and self._ndtmpsum is not None
                and self._ndtmpsum <= self.min_nd
            ):
                logger.warning(
                    "norm of weight gradient <= %g at iter %d; stopping.",
                    self.min_nd,
                    it,
                )
                self.stop_reason = "grad_norm"
                leave = True

            if leave:
                break

        if self.stop_reason in self._DEGENERATE_STOP_REASONS:
            self.final_ll_ = float("nan")
        else:
            self.final_ll_ = self.ll_history[-1] if self.ll_history else float("nan")
        return self

    def transform(self, X: np.ndarray, model_idx: int = 0) -> np.ndarray:
        """Not yet implemented -- fail with a clear boundary rather than a bare
        AttributeError. Use ``AMICATorchNG.transform`` for source extraction; the
        MLX backend validates via ``final_ll_``/``ll_history``."""
        raise NotImplementedError(
            "AMICAMLXNG does not implement transform yet; it is a fast-follow. "
            "Use AMICATorchNG for source extraction."
        )
