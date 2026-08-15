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
Gaussian (``pdftype=0``), natural gradient (``do_newton=False``). Newton and the
other PDF families are rejected in ``__init__`` with a clear
``NotImplementedError`` (``transform`` likewise). Component sharing, outlier
rejection, ``keep_best`` and save/load are simply absent (no such
parameter/method) -- all fast-follows.
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
        self.n_models = n_models  # multi-model supported (issue #81); no sharing yet
        self.n_mix = n_mix
        self.n_comps = n_channels * n_models
        self.block_size = block_size

        self.lrate0 = lrate
        self.lrate = lrate
        self.lrate_cap = lrate
        self.minlrate = minlrate
        self.lratefact = lratefact
        self.maxdecs = maxdecs
        self.newt_ramp = newt_ramp
        # Schedule threshold for the rholrate ceiling ratchet (Fortran
        # amica15.f90:1049 gates it on iter > newt_start, independent of
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
        self.sldet = 0.0
        self._lgamma_table: Optional[mx.array] = (
            None  # (n_mix, n_comps): lgamma(1+1/rho)
        )
        self._logdet_W: Optional[mx.array] = (
            None  # scalar: log|det W|, refreshed per iter
        )

    _DEGENERATE_STOP_REASONS = ("nan_ll", "singular_ll", "nan_params")

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
                # amica15.f90:545).
                w_pca = inv_sqrt @ V.T
                if self.do_approx_sphere:
                    # Fortran's orthogonal polar-factor symmetrization of the
                    # reduced whitening (amica15.f90:483-490).
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

        self.alpha = acc["dalpha_n"] / acc["dalpha_n"].sum(axis=0, keepdims=True)
        self.mu = self.mu + acc["dmu_n"] / acc["dmu_d"]
        self.beta = mx.clip(
            self.beta * mx.sqrt(acc["dbeta_n"] / acc["dbeta_d"]),
            self.invsigmin,
            self.invsigmax,
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
            self.rho = mx.clip(new_rho, self.minrho, self.maxrho)

        # Natural-gradient A-update. A is stored as Fortran's A^T, so the update
        # is a LEFT-multiply by the transposed direction (core.py:1176-1184,
        # #24 root cause). Each model's direction is scattered into its mixing
        # columns as a gm-weighted average (Fortran dAk/zeta, core.py:1231-1247):
        # for the default disjoint comp_list every column has one contributor, so
        # gm cancels and n_models=1 is byte-for-byte the old `A - lrate*(dA.T@A)`.
        eye = mx.eye(self.n_channels)
        directions = [
            -acc["dWtmp"][h] / acc["dgm"][h] + eye for h in range(self.n_models)
        ]
        self.lrate = min(
            self.lrate_cap, self.lrate + min(1.0 / self.newt_ramp, self.lrate)
        )
        assert self.A is not None and self.comp_list is not None and self.gm is not None
        dAk = mx.zeros_like(self.A)
        zeta = mx.zeros((self.n_comps,), dtype=mx.float32)
        for h in range(self.n_models):
            idx = self.comp_list[:, h]
            dAk = dAk.at[:, idx].add(self.gm[h] * (directions[h].T @ self.A[:, idx]))
            zeta = zeta.at[idx].add(self.gm[h] + mx.zeros((self.n_channels,)))
        dAk = dAk / mx.maximum(zeta, tiny)
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
            mx.eval(
                self.A,
                self.W,
                self.mu,
                self.alpha,
                self.beta,
                self.rho,
                self.gm,
                self.c,
            )

            # Surface a corrupted M-step (component collapse / float32 overflow)
            # at the iteration it happens. The ll check above only catches a
            # corruption via the NEXT iteration's E-step, so a final-iteration
            # blow-up would otherwise complete as max_iter with silently NaN
            # parameters (the torch backend has state_dict as a backstop; the
            # MLX backend does not, so guard in fit()). Params are already
            # materialized by the mx.eval above, so this is a cheap read.
            params_finite = (
                mx.all(mx.isfinite(self.A))
                & mx.all(mx.isfinite(self.mu))
                & mx.all(mx.isfinite(self.alpha))
                & mx.all(mx.isfinite(self.beta))
                & mx.all(mx.isfinite(self.rho))
                & mx.all(mx.isfinite(self.gm))
                & mx.all(mx.isfinite(self.c))
            )
            if not bool(params_finite.item()):
                logger.warning(
                    "Non-finite parameters at iter %d (a mixture component "
                    "likely collapsed); stopping.",
                    it,
                )
                self.stop_reason = "nan_params"
                break

            self.ll_history.append(ll)

            # Learning-rate control (Fortran amica17.f90:1062-1108): anneal on an
            # LL decrease; ratchet the ceilings after maxdecs persistent decreases.
            #
            # rholrate is a maxdecs-ratcheted CEILING, not a per-decrease-annealed
            # working rate. Fortran resets rholrate=rholrate0 every iteration before
            # the rho update (amica15.f90:1788) and only tightens the rholrate0
            # ceiling at maxdecs (amica15.f90:1050, gated on iter > newt_start), so
            # its per-decrease rholrate*=rholratefact (:1045) never reaches the rho
            # update. rho has no ramp, so self.rholrate carries that ceiling directly
            # (reset to rholrate0 at fit start, nothing re-inflates it) and must
            # ratchet ONLY at maxdecs. The previous per-decrease decay collapsed the
            # rho rate to ~1e-5 within a few hundred iterations and froze rho at a
            # stale shape (issue #195, mirroring the torch/numpy fix in #193/#194).
            if len(self.ll_history) > 1 and ll < self.ll_history[-2]:
                if self.lrate <= self.minlrate:
                    logger.warning(
                        "lrate floor (%g) reached at iter %d; stopping.",
                        self.minlrate,
                        it,
                    )
                    self.stop_reason = "lrate_floor"
                    break
                self.lrate *= self.lratefact
                numdecs += 1
                if numdecs >= self.maxdecs:
                    self.lrate_cap *= self.lratefact
                    if it > self.newt_start:
                        self.rholrate *= self.rholratefact
                    numdecs = 0

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
