"""
Natural-gradient EM PyTorch backend for AMICA (ADR 0001).

Rather than reframing AMICA as "minimize negative log-likelihood with Adam
over reparameterized tensors" (the approach of the earlier Adam/autograd
backends, since removed in issue #32), this module is a direct, vectorized
port of the closed-form E-step/M-step fixed-point updates used by the Fortran
reference (``amica17.f90``) and the legacy NumPy implementation
(``pamica.numpy_impl.core.AMICA._get_block_updates`` / ``_update_parameters``, which
is this module's line-by-line spec). There is
no autograd and no Adam: every parameter update is a closed-form function of
the E-step responsibilities and simple moments, matching the natural-gradient
EM fixed point instead of a different, Adam-driven trajectory.

Key design points (see ``.context/decisions/0001-torch-backend-natural-gradient-em.md``):

* ``W`` (and ``A``) are stored and mutated directly; ``W`` is recomputed from
  ``A`` once per iteration via a batched ``torch.linalg.inv`` (matching
  ``numpy_impl.utils.get_unmixing_matrices``), never via ``pinv`` in the hot path.
* The E-step is vectorized over ``(model, mix, source)`` via broadcasting;
  the only Python loops are over models (typically 1-3) and over blocks.
* Samples are processed in blocks and sufficient statistics are accumulated
  across blocks, so peak memory scales with ``block_size``, not with the
  total number of samples.
* Parameters default to float64 for numerical parity with Fortran's
  double-precision arithmetic; float32 is available for speed and for MPS
  (which has no float64), stabilized on full-size data by the mu-denominator
  divide-by-zero guard in ``_get_block_updates`` (issue #75).

Log-likelihood: this module computes the per-source log-likelihood from the
pre-normalization mixture logits via ``logsumexp`` plus the ``log|det W|`` +
``sldet`` Jacobian (matching ``amica17.f90:1341-1350``), the mathematically
correct per-source log-density required to hit the Fortran-normalized LL target
(~-3.4/sample-channel). As of issue #24 the legacy NumPy port
(``pamica.numpy_impl.core.AMICA``) computes it the same way; both backends now converge
to the Fortran solution (component correlation > 0.95).

Source-density families (issue #26): the default GG path cites ``amica17.f90``,
but the reference *binary* is ``amica15mac`` = ``amica15.f90``, which (unlike the
GG-only ``amica17.f90``) implements the ``pdtype`` density families. The
``pdftype`` machinery therefore cites ``amica15.f90``; the two Fortran sources
are not interchangeable. See ``.context/decisions/0002-adaptive-pdf-families.md``.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm

from ..metrics import mir as mir_metric
from ..metrics import pairwise_mi
from ..rank import MINEIG, MINEIG_REL, numerical_rank
from .utils import setup_device

logger = logging.getLogger(__name__)

# Human-readable names for the ``pdftype``/``pdtype`` source-density family codes
# (issue #26; amica15.f90). Exposed alongside the numeric codes so a fitted
# model's per-source density family is inspectable (issue #142).
PDFTYPE_NAMES = {
    0: "generalized_gaussian",
    1: "super_gaussian_cosh",
    2: "gaussian",
    3: "logistic",
    4: "sub_gaussian_cosh",
}

_LOG2 = math.log(2.0)
_LOG4 = math.log(4.0)  # logistic-family normalizer (amica15.f90:1346)
_HALF_LOG_PI = 0.5 * math.log(math.pi)
# Log-normalizers for the non-GG density families, using Fortran's exact literal
# constants (amica15.f90:1333/1359/1371) so the log-density matches the reference
# binary bit-for-bit: 2.506628274 = sqrt(2*pi) (Gaussian, pdtype 2); 4.132731354 /
# 1.858073988 = the sub-/super-Gaussian cosh normalizers (pdtype 4 / 1).
_LOG_SQRT_2PI = math.log(2.506628274)
_LOG_NORM_COSH_SUB = math.log(4.132731354)
_LOG_NORM_COSH_SUP = math.log(1.858073988)
# Fortran's epsdble (amica17_header.f90:73): the drho-numerator underflow guard
# zeros the rho*ln|y| term when |y|^rho falls below this, matching amica17.f90:1570.
_EPSDBLE = 1e-16

# Best-iterate safeguard (issue #51). The lrate schedule is deliberately
# non-monotone: both NG and Fortran anneal the rate only *after* an LL decrease,
# so a late Newton fallback can overshoot and a run can end below a peak it
# already reached (on the sample EEG this is the sole driver of NG's inflated
# multi-model LL variance -- one seed peaked at -3.357 then crashed to -3.545 in
# its last iterations). fit() therefore tracks the highest-LL iterate and
# restores it when the final LL falls more than this tolerance below that peak.
# Units: mean log-likelihood per sample-channel (the same scale as Fortran's
# min_dll comparison -- verified against BOTH Fortran sources, not just one:
# amica15.f90:1770 and amica17.f90:1866 both compute LL(iter) = LLtmp2 /
# dble(numgoodsum*nw) and then compare consecutive LL(iter) values against
# min_dll (amica15.f90:1079 / amica17.f90:1085), so this is not an amica17-only
# quirk -- amica15, the actual reference binary's source, does the identical
# normalization. The legacy NumPy backend compared un-normalized summed LL until
# issue #212, which made its own min_dll unreachable; it now normalizes the same
# way, so all three agree on this scale.
# 1e-9 reads as "numerical noise, not a real overshoot" on this normalized
# scale. This module's own ``ll_history`` is already stored normalized the
# same way (``ll = acc["ll"] / (n_use * n_channels)``, see ``fit``), so the
# min_dll stop (issue #207) compares ``ll_history[-1] - ll_history[-2]``
# directly with no extra scaling needed. The threshold also keeps a monotone
# single-model run (issue #24 parity) a bit-exact no-op: its final iterate
# already IS the best, the gap is 0 < tol, and no restore fires.
_KEEP_BEST_TOL = 1e-9


def _logcosh(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable ``log cosh(x) = |x| - log2 + log1p(exp(-2|x|))``."""
    ax = x.abs()
    return ax - _LOG2 + torch.log1p(torch.exp(-2.0 * ax))


def _log_pdf_and_deriv(
    y: torch.Tensor, rho: torch.Tensor, pdtype: Optional[torch.Tensor] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized source-density log-density and density derivative.

    Elementwise port of ``pamica.numpy_impl.core.AMICA._compute_log_pdf``: branches via
    ``torch.where`` instead of Python control flow so it runs over full
    ``(block, source, mixture)`` tensors with no source/mixture loop. ``y``,
    ``rho`` and ``pdtype`` must be broadcastable to a common shape.

    When ``pdtype is None`` (the default ``pdftype=0`` path) this computes only
    the generalized-Gaussian (GG) family, branching on ``rho``
    (Laplace/Gaussian/GG), and is bit-identical to the pre-#26 implementation.
    When ``pdtype`` is given it additionally selects, per source, among the
    fixed density families of ``amica15.f90`` (codes 0/2/3/4/1): GG, Gaussian,
    logistic, sub-Gaussian cosh+, super-Gaussian cosh-. The density derivative
    obeys ``dpdf = -fp * pdf`` for every family (``fp`` = the score from
    ``_score``), which reproduces the GG ``dpdf`` exactly.
    """
    abs_y = y.abs()
    sign_y = torch.sign(y)

    log_pdf_lap = -abs_y - _LOG2
    dpdf_lap = -sign_y * torch.exp(log_pdf_lap)

    log_pdf_gau = -y * y - _HALF_LOG_PI
    dpdf_gau = -2.0 * y * torch.exp(log_pdf_gau)

    log_pdf_gg = -abs_y.pow(rho) - _LOG2 - torch.lgamma(1.0 + 1.0 / rho)
    dpdf_gg = -rho * abs_y.pow(rho - 1.0) * sign_y * torch.exp(log_pdf_gg)

    is_lap = rho == 1.0
    is_gau = rho == 2.0

    log_pdf = torch.where(
        is_gau, log_pdf_gau, torch.where(is_lap, log_pdf_lap, log_pdf_gg)
    )
    dpdf = torch.where(is_gau, dpdf_gau, torch.where(is_lap, dpdf_lap, dpdf_gg))
    if pdtype is None:
        return log_pdf, dpdf

    # Non-GG families (amica15.f90:1327-1371). Each is `-cost - log_norm`, and
    # dpdf = -fp * exp(log_pdf).
    log_pdf_2 = -0.5 * y * y - _LOG_SQRT_2PI  # Gaussian
    log_pdf_3 = -2.0 * _logcosh(0.5 * y) - _LOG4  # logistic (sech^2)
    lc = _logcosh(y)
    log_pdf_4 = -0.5 * y * y + lc - _LOG_NORM_COSH_SUB  # sub-Gaussian cosh+
    log_pdf_1 = -0.5 * y * y - lc - _LOG_NORM_COSH_SUP  # super-Gaussian cosh-

    log_pdf = torch.where(
        pdtype == 2,
        log_pdf_2,
        torch.where(
            pdtype == 3,
            log_pdf_3,
            torch.where(
                pdtype == 4, log_pdf_4, torch.where(pdtype == 1, log_pdf_1, log_pdf)
            ),
        ),
    )
    fp = _score(y, rho, pdtype)
    dpdf = torch.where(pdtype == 0, dpdf, -fp * torch.exp(log_pdf))
    return log_pdf, dpdf


def _score(
    y: torch.Tensor, rho: torch.Tensor, pdtype: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Source-density score ``fp = -d(log pdf)/dy`` (Fortran ``fp``).

    For the GG family this is ``fp(y) = rho*sign(y)*|y|^(rho-1)`` (``sign(y)``
    for Laplace, ``2y`` for Gaussian), used by the exact-EM and Newton
    sufficient statistics (``amica15.f90:1467-1491``). It is distinct from the
    density derivative ``dpdf`` (which carries an extra ``pdf`` factor).

    With ``pdtype is None`` only the GG score is computed (bit-identical to the
    pre-#26 path). With ``pdtype`` given it selects per source among the fixed
    families: 2 Gaussian ``y``; 3 logistic ``tanh(y/2)``; 4 sub-Gaussian
    ``y - tanh(y)``; 1 super-Gaussian ``y + tanh(y)``.
    """
    abs_y = y.abs()
    sign_y = torch.sign(y)
    fp_lap = sign_y
    fp_gau = 2.0 * y
    fp_gg = rho * sign_y * abs_y.pow(rho - 1.0)
    is_lap = rho == 1.0
    is_gau = rho == 2.0
    fp = torch.where(is_gau, fp_gau, torch.where(is_lap, fp_lap, fp_gg))
    if pdtype is None:
        return fp

    tanh_half = torch.tanh(0.5 * y)
    tanh_y = torch.tanh(y)
    return torch.where(
        pdtype == 2,
        y,
        torch.where(
            pdtype == 3,
            tanh_half,
            torch.where(
                pdtype == 4, y - tanh_y, torch.where(pdtype == 1, y + tanh_y, fp)
            ),
        ),
    )


def _log_pdf_only(
    y: torch.Tensor, rho: torch.Tensor, pdtype: Optional[torch.Tensor] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Log-density only, plus ``|y|^rho`` for reuse -- the E-step's hot path.

    The forward pass needs only ``log_pdf`` (for ``z0``/responsibilities); the
    density derivative ``dpdf`` that :func:`_log_pdf_and_deriv` also computes is
    never consumed (the exact-EM M-step uses the score ``fp`` instead), so
    computing it -- a ``|y|^(rho-1)`` power and three ``exp``s per block -- is
    dead work (issue #63). This returns ``log_pdf`` bit-identically to
    :func:`_log_pdf_and_deriv` and the ``|y|^rho`` power, which the caller reuses
    for the ``rho`` update (dropping the duplicate ``|y|^rho`` that
    ``_get_block_updates`` previously recomputed for that accumulator).
    """
    abs_y = y.abs()
    az_rho = abs_y.pow(rho)  # |y|^rho, reused by the rho-update accumulator

    log_pdf_lap = -abs_y - _LOG2
    log_pdf_gau = -y * y - _HALF_LOG_PI
    log_pdf_gg = -az_rho - _LOG2 - torch.lgamma(1.0 + 1.0 / rho)
    log_pdf = torch.where(
        rho == 2.0, log_pdf_gau, torch.where(rho == 1.0, log_pdf_lap, log_pdf_gg)
    )
    if pdtype is None:
        return log_pdf, az_rho

    log_pdf_2 = -0.5 * y * y - _LOG_SQRT_2PI  # Gaussian
    log_pdf_3 = -2.0 * _logcosh(0.5 * y) - _LOG4  # logistic (sech^2)
    lc = _logcosh(y)
    log_pdf_4 = -0.5 * y * y + lc - _LOG_NORM_COSH_SUB  # sub-Gaussian cosh+
    log_pdf_1 = -0.5 * y * y - lc - _LOG_NORM_COSH_SUP  # super-Gaussian cosh-
    log_pdf = torch.where(
        pdtype == 2,
        log_pdf_2,
        torch.where(
            pdtype == 3,
            log_pdf_3,
            torch.where(
                pdtype == 4, log_pdf_4, torch.where(pdtype == 1, log_pdf_1, log_pdf)
            ),
        ),
    )
    return log_pdf, az_rho


class AMICATorchNG:
    """
    Natural-gradient EM AMICA, ported from ``pamica.numpy_impl.core.AMICA``.

    Not an ``nn.Module``: there are no learnable ``nn.Parameter``s and no
    autograd. Parameters (``A``, ``W``, ``c``, ``mu``, ``alpha``, ``beta``,
    ``rho``, ``gm``) are plain tensors mutated in place by closed-form
    E-step/M-step updates each iteration, mirroring
    ``pamica.AMICA._get_block_updates``/``_update_parameters``.

    Parameters
    ----------
    n_channels : int
        Number of input channels (``data_dim`` in the NumPy/Fortran code).
    n_models : int, default=1
        Number of ICA mixture models.
    n_mix : int, default=3
        Number of mixture components per source.
    block_size : int, default=8192
        Number of samples processed per accumulation block. Peak memory
        during the E-step scales with this, not with the total sample count.
        Larger blocks give bigger tensor ops (less Python/dispatch overhead,
        better threading/GPU utilization) at higher memory. Every backend is
        dispatch-bound at small blocks, making this the largest throughput knob:
        raising it from 512 to 8192 is ~6x on CPU float64 for the bundled sample
        (issue #216). 8192 rather than larger because peak block memory scales
        with it and 8192 stays near 240 MB even at 256 channels.

        Fortran pins no comparable value (header default 128, auto-tuned over
        128-1024 via ``do_opt_block``). Per-iteration sufficient statistics are
        block-size-independent to ~1e-8 (``test_blocking_invariance``); the
        multi-iteration trajectory shifts ~1e-6, inside parity tolerance but
        enough that a bit-for-bit Fortran comparison must match ``block_size`` on
        both sides (the bundled ``input.param`` uses 512).
    lrate : float, default=0.1
        Initial/maximum natural-gradient learning rate (``lrate0`` in NumPy).
    minlrate : float, default=1e-12
        Hard learning-rate floor: once ``lrate`` anneals to it, ``fit`` stops
        (``stop_reason="lrate_floor"``).
    lratefact : float, default=0.5
        Factor by which ``lrate`` (and the ceiling ``lrate_cap``/``newtrate``)
        are annealed when the log-likelihood decreases; see ``fit`` for the
        Fortran-style ``numdecs``/``maxdecs`` ratchet.
    maxdecs : int, default=5
        Number of consecutive log-likelihood decreases after which the
        learning-rate *ceiling* is ratcheted down (Fortran ``maxdecs``).
    use_min_dll : bool, default=True
        Enable the small-likelihood-increase stop (Fortran ``use_min_dll``,
        amica15_header.f90:24/109; amica15.f90:1078-1090): once the per-
        sample-channel log-likelihood gain ``ll_history[-1] - ll_history[-2]``
        falls below ``min_dll`` for more than ``maxincs`` *consecutive*
        iterations, ``fit`` stops (``stop_reason="min_dll"``). The counter
        resets to 0 on any iteration with a larger gain (including a
        likelihood decrease, which is always "less than" a positive
        ``min_dll``, so it also increments the counter). Checked every
        iteration once two log-likelihood values exist (never on the first).
    min_dll : float, default=1e-9
        Threshold for ``use_min_dll``, on the log-likelihood's own scale
        (mean log-likelihood per sample-channel, matching ``ll_history`` --
        see ``amica15.f90:1770``, which normalizes ``LL(iter)`` by
        ``numgoodsum*nw`` before this comparison in the reference).
    maxincs : int, default=5
        Number of consecutive small-gain iterations tolerated before
        ``use_min_dll`` stops the fit (Fortran ``maxincs``, not itself
        configurable from the Fortran param file -- fixed at its header
        default).
    use_grad_norm : bool, default=True
        Enable the weight-gradient-norm stop (Fortran ``use_grad_norm``,
        amica15_header.f90:24/74; amica15.f90:1091-1097): once the RMS
        weight-update norm ``ndtmpsum`` (see ``min_nd``) falls to or below
        ``min_nd``, ``fit`` stops (``stop_reason="grad_norm"``). This is
        independent of ``use_min_dll`` and of whether the log-likelihood
        just decreased; it is also folded into the likelihood-decrease
        branch unconditionally (``stop_reason="grad_norm_floor"``, Fortran
        amica15.f90:1058's ``.or. (ndtmpsum .le. min_nd)``, alongside the
        existing ``lrate <= minlrate`` check) -- this decrease-branch half is
        what fixes the reported CUDA/``do_newton=True`` case where ``lrate``
        sits at ``newtrate`` and oscillates instead of annealing, so the old
        ``lrate_floor``-only check never fired and ``max_iter`` was the only
        stop (issue #207). Checked every iteration once two log-likelihood
        values exist (never on the first).

        CAUTION: with ``use_grad_norm`` at this default (``True``), the
        stop that actually surfaces for the fixed CUDA scenario is
        ``"grad_norm"``, not ``"grad_norm_floor"``. The standalone check
        above runs every iteration regardless of LL direction and (in
        ``fit``'s per-iteration ordering) is evaluated after the
        likelihood-decrease branch, with no ``elif``/``leave`` gate between
        them; whichever iteration first satisfies ``ndtmpsum <= min_nd``
        also satisfies the standalone check that same iteration, so it always
        overwrites ``stop_reason`` before a decrease-gated
        ``"grad_norm_floor"`` could be the value ``fit`` finally reports.
        ``"grad_norm_floor"`` is therefore only distinctly reachable as the
        *final* ``stop_reason`` when ``use_grad_norm=False`` (isolating the
        decrease-branch half, as ``test_grad_norm_floor_fires_on_likelihood_decrease``
        does); ``test_grad_norm_shadows_grad_norm_floor_under_shipped_defaults``
        (same setup, ``use_grad_norm`` left at its default) confirms the
        shadowing directly. ``"min_dll"`` can likewise be shadowed by
        ``"grad_norm"`` if both conditions happen to hold in the same
        iteration -- Fortran has this same structure (independent
        ``leave=.true.`` assignments with no declared precedence among them),
        so this is not a fidelity bug, just a reporting nuance worth knowing
        before reading ``stop_reason`` as a precise diagnosis.
    min_nd : float, default=1e-7
        Threshold for ``use_grad_norm`` (and the decrease-branch grad-norm
        check). Matches Fortran's ``ndtmpsum`` (amica15.f90:1760-1761): the
        RMS, over ``comp_used`` components only, of the per-iteration
        weight-update direction ``dAk`` (the natural-gradient/Newton step
        before the ``lrate`` scaling and before ``share_comps``'s A-freeze
        may discard it) -- ``sqrt(sum(dAk**2, axis=0)[comp_used].sum() /
        (n_channels * comp_used.sum()))``. The ``comp_used`` mask only
        differs from all-True when ``share_comps`` has merged/frozen columns
        (issue #60); it is a no-op otherwise. Computed every iteration
        regardless of ``use_grad_norm``/``use_min_dll`` (both stops read the
        same per-iteration value; Fortran computes ``ndtmpsum`` unconditionally
        too, in ``accum_updates_and_likelihood``, before either check runs).

        Not reachable on small recordings, in any implementation: the reference
        binary's own gradient norm plateaus at 1.0-1.65e-5 on the bundled
        32-channel sample, two orders above this threshold, so the stop never
        fires there and ``min_dll`` is what ends the fit. The default is kept
        Fortran-faithful rather than retuned; see the convergence-criteria
        section of ``docs/guides/validation.md`` (issue #218).
    newt_ramp : int, default=10
        Denominator of the per-iteration learning-rate ramp toward the current
        ceiling: ``lrate = min(ceiling, lrate + min(1/newt_ramp, lrate))``
        (ceiling is ``lrate_cap`` for natural gradient, ``newtrate`` for
        Newton).
    do_newton : bool, default=False
        Enable the Newton preconditioner for the ``A``/``W`` update once
        ``iteration >= newt_start``. Ported from the Fortran reference
        (``amica17.f90``): natural gradient alone plateaus well short of the
        Fortran solution, and the Newton step (a per-source-pair 2x2 solve
        preconditioning the natural gradient by an approximate Hessian) is
        what closes the gap.
    newt_start : int, default=20
        Iteration at which the Newton step switches on (natural gradient is
        used before it, letting the mixture parameters settle first).
    newtrate : float, default=0.5
        Maximum learning rate the ramp climbs to while Newton is active
        (the natural-gradient phase is capped at ``lrate``/``lrate0``).
    do_reject : bool, default=False
        Enable Fortran-style outlier rejection: after the parameter update,
        samples whose total log-likelihood falls below
        ``mean - rejsig*std`` are permanently excluded from subsequent
        sufficient-statistic accumulation and from the sample count used to
        normalize ``gm`` and the reported log-likelihood.
    rejsig : float, default=3.0
        Rejection threshold in standard deviations of the per-sample
        log-likelihood.
    rejstart, rejint, maxrej : int
        First rejection iteration, interval between rejections, and maximum
        number of rejection passes (matching ``amica17.f90:1141-1146``).
    rho0, minrho, maxrho, rholrate : float
        Generalized-Gaussian shape-parameter initialization, clamp bounds,
        and learning rate.
    keep_best : bool, default=True
        Return the highest-log-likelihood iterate instead of the last one
        (issue #51). The lrate schedule is non-monotone (it anneals only after
        an LL *decrease*), so a late Newton-fallback overshoot can leave the
        final iterate below a peak the run already reached. When the final LL
        falls more than a small tolerance below that peak, ``fit`` restores the
        peak's parameters. A monotone single-model run (issue #24 parity) is a
        bit-exact no-op. Automatically inactive under ``do_reject`` (the
        good-sample set, and thus the LL normalization, changes across
        iterations, making per-iteration LLs incomparable).
    pdftype : int, default=0
        Source-density family (issue #26), matching Fortran ``amica15.f90``'s
        ``pdtype`` codes: 0 generalized Gaussian (default; rho adapts), 2
        Gaussian, 3 logistic, 4 sub-Gaussian cosh+. ``pdftype=1`` enables the
        extended-Infomax adaptive switcher, which flips each source between the
        super-Gaussian (code 1) and sub-Gaussian (code 4) cosh densities by
        kurtosis sign. For every non-GG family the GG shape update is frozen
        (Fortran ``dorho=.false.``); the single-component families 1/4 (and the
        adaptive mode) require ``n_mix=1``. ``pdftype=0`` is byte-for-byte the
        pre-#26 implementation.
    kurt_start, num_kurt, kurt_int : int
        Adaptive-switch schedule (only used when ``pdftype=1``): first iteration
        to re-estimate kurtosis, number of switch passes, and the iteration
        interval between them. ``num_kurt=0`` disables switching (the family
        stays at its super-Gaussian init).
    invsigmin, invsigmax : float
        Clamp bounds for the mixture scale parameter ``beta``.
    doscaling, scalestep : bool, int
        Whether/how often to rescale ``A`` columns to unit norm each
        iteration (with matching ``mu``/``beta`` rescale).
    share_comps : bool, default=False
        Enable multi-model component sharing (Fortran ``share_comps`` /
        ``identify_shared_comps``, amica15.f90:1916): components that are
        near-collinear across different models are merged so they share one
        mixing column and one density. Requires ``n_models >= 2`` (a model
        cannot share with itself); a no-op otherwise. OFF by default, so
        single-model (#24) and default multi-model (#27) results are unchanged.
        There is no bit-exact oracle -- the reference's similarity metric is
        never initialized (like ``do_choose_pdfs``, #26) -- so this implements
        the intended algorithm, validated by real-data behavior.
    share_start, share_iter : int
        Sharing schedule: first iteration to attempt merges and the interval
        between attempts (Fortran ``share_start``/``share_iter``). The A-update
        is held for the first 6 iterations of every cycle (independent of whether
        a merge fired) so densities can settle; ``share_iter`` must be ``> 6`` so
        that window never consumes the whole cycle.
    comp_thresh : float, default=0.99
        Cosine-similarity cutoff (in the de-sphered/sensor-space metric) above
        which two mixing columns are identified and merged. The de-sphering uses
        ``pinv(sphere)``, so sharing also works on rank-reduced and
        rank-deficient fits (issues #253, #221); see
        :meth:`_identify_shared_comps`.
    do_mean, do_sphere, do_approx_sphere : bool
        Preprocessing options, matching ``pamica.AMICA._preprocess_data``.
    pcakeep, pcadb : int, float, optional
        PCA dimensionality-reduction options (rarely used; see
        ``pamica.AMICA._preprocess_data``). Both are capped by the detected
        numerical rank (``mineig``), matching Fortran's
        ``numeigs = min(pcakeep, count(eigs > mineig))``.
    mineig : float, default=1e-15
        Absolute floor on data-covariance eigenvalues used to detect the
        numerical rank (Fortran ``mineig``, amica15.f90:413 and
        amica15_header.f90:66). Eigen-directions at or below it are dropped, the
        model is sized to the surviving rank, and sensor-space maps come from
        :meth:`get_sensor_mixing_matrix`. Full-rank data keep every eigenvalue,
        so this is a no-op there and single-model parity is byte-for-byte.

        Being absolute, it is unit-dependent: EEG in microvolts gives
        eigenvalues of order 1-100 and the default behaves, but MEG in Tesla
        gives ~1e-26 and every eigenvalue falls below it, which Fortran would
        turn into ``numeigs = 0``. pamica raises instead of fitting an empty
        model. Use ``mineig_rel`` (or rescale) for such data.
    mineig_rel : float, optional
        Scale-free alternative to ``mineig``: when set, the threshold becomes
        ``mineig_rel * largest_eigenvalue`` and ``mineig`` is ignored. Off by
        default so rank detection stays Fortran-exact. It is also the more
        accurate detector -- the absolute floor sits amid the numerical-zero
        eigenvalues of rank-deficient data and over-retains, while a relative
        floor recovers the true rank (issue #223).
    seed : int, optional
        Seed for parameter initialization. Uses ``numpy.random.RandomState``
        internally (not ``torch``'s RNG) with the exact same draw order as
        ``pamica.AMICA._initialize_parameters``, so the same seed produces
        bit-identical starting parameters to the NumPy reference.
    device : str or torch.device, optional
        Compute device for the block loop. Preprocessing (mean/cov/eigh) is
        always done in float64 on CPU regardless of device, since eigh is
        not reliably supported on MPS.
    dtype : torch.dtype, default=torch.float64
        Parameter/computation dtype. float64 is the parity default (Fortran
        bit-parity) and ~4.5x on CUDA over CPU (issue #63). float32 converges on
        full-size data across seeds (issue #75 guarded the one float32-only
        divide-by-zero -- a sample rounding an activation to exactly 0 gave
        ``0/0`` in the mu denominator) and is the required precision on MPS,
        which has no float64. float32 is NOT bit-parity with float64 (~7
        significant digits), so use float64 for Fortran-parity runs and float32
        for speed / Apple-GPU.
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
        do_newton: bool = False,
        newt_start: int = 20,
        newtrate: float = 0.5,
        do_reject: bool = False,
        rejsig: float = 3.0,
        rejstart: int = 2,
        rejint: int = 3,
        maxrej: int = 1,
        rho0: float = 1.5,
        minrho: float = 1.0,
        maxrho: float = 2.0,
        rholrate: float = 0.05,
        rholratefact: float = 0.1,
        keep_best: bool = True,
        pdftype: int = 0,
        kurt_start: int = 3,
        num_kurt: int = 5,
        kurt_int: int = 1,
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
        pcakeep: Optional[int] = None,
        pcadb: Optional[float] = None,
        mineig: float = MINEIG,
        mineig_rel: Optional[float] = MINEIG_REL,
        seed: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float64,
    ):
        self.n_channels = n_channels
        self.n_models = n_models
        self.n_mix = n_mix
        self.n_comps = n_channels * n_models
        self.mineig = mineig
        self.mineig_rel = mineig_rel
        self.block_size = block_size

        self.lrate0 = lrate
        self.lrate = lrate
        self.minlrate = minlrate
        self.lratefact = lratefact
        self.maxdecs = maxdecs

        # Convergence stops (issue #207), Fortran-faithful defaults (both
        # amica15_header.f90:24/74/109 flags default True): the small-
        # likelihood-increase stop (use_min_dll/min_dll/maxincs) and the
        # weight-gradient-norm stop (use_grad_norm/min_nd). See fit() for the
        # per-iteration checks and _update_parameters for the ndtmpsum
        # computation these both read.
        if maxincs < 0:
            raise ValueError(f"maxincs must be >= 0, got {maxincs}")
        self.use_min_dll = use_min_dll
        self.min_dll = min_dll
        self.maxincs = maxincs
        self.use_grad_norm = use_grad_norm
        self.min_nd = min_nd

        self.newt_ramp = newt_ramp

        self.do_newton = do_newton
        self.newt_start = newt_start
        self.newtrate = newtrate
        self.newtrate0 = newtrate

        self.do_reject = do_reject
        self.rejsig = rejsig
        self.rejstart = rejstart
        self.rejint = rejint
        self.maxrej = maxrej
        if do_reject:
            if rejint < 1:
                raise ValueError(f"rejint must be >= 1, got {rejint}")
            if rejsig <= 0:
                raise ValueError(f"rejsig must be > 0, got {rejsig}")
            if maxrej < 0:
                raise ValueError(f"maxrej must be >= 0, got {maxrej}")

        self.rho0 = rho0
        self.minrho = minrho
        self.maxrho = maxrho
        self.rholrate = rholrate
        self.rholrate0 = rholrate
        self.rholratefact = rholratefact

        # Best-iterate safeguard (issue #51). When True, fit() restores the
        # highest-log-likelihood iterate if the run ends more than _KEEP_BEST_TOL
        # below it (a late Newton-fallback overshoot). Disabled automatically
        # under do_reject, where the good-sample set (and the LL normalization)
        # changes across iterations, so per-iteration LLs are not comparable.
        self.keep_best = keep_best

        # Source-density family selection (Fortran ``pdftype``, amica15.f90). Values
        # match Fortran's per-source ``pdtype`` codes: 0 generalized Gaussian (the
        # default, GG-mixture with adaptive rho), 2 Gaussian mixture, 3 logistic
        # (sech^2) mixture, 4 sub-Gaussian cosh+ (single component). pdftype=1 enables
        # the extended-Infomax adaptive switcher (Fortran's do_choose_pdfs trigger),
        # which flips each source between the super-Gaussian (code 1) and sub-Gaussian
        # (code 4) cosh densities by kurtosis sign on the kurt_start/num_kurt/kurt_int
        # schedule. Families 1 and 4 are single-component (no alpha mixture).
        if pdftype not in (0, 1, 2, 3, 4):
            raise ValueError(f"pdftype must be one of 0,1,2,3,4; got {pdftype}")
        self.pdftype = pdftype
        # Fortran freezes the GG shape update for every non-GG family (amica15.f90:
        # `if (pdftype /= 0) dorho = .false.`, lines 3704-3705).
        self.dorho = pdftype == 0
        # pdftype==1 is Fortran's adaptive trigger (amica15.f90:612).
        self.do_choose_pdfs = pdftype == 1
        self.kurt_start = kurt_start
        self.num_kurt = num_kurt
        self.kurt_int = kurt_int
        # Families 1/4 (and the adaptive mode, which uses only codes 1 and 4) are
        # single-component densities: Fortran's z0 references only mixture component
        # j=1 and omits log(alpha). They are meaningful only with n_mix == 1.
        if pdftype in (1, 4) and n_mix != 1:
            raise ValueError(
                f"pdftype={pdftype} is a single-component density (adaptive mode "
                f"uses codes 1 and 4); it requires n_mix=1, got n_mix={n_mix}."
            )
        # Validate the adaptive-switch schedule up front (mirrors the do_reject
        # checks below): kurt_int==0 would otherwise raise a bare ZeroDivisionError
        # deep in fit(), and a negative kurt_int silently changes the schedule.
        if self.do_choose_pdfs:
            if kurt_int < 1:
                raise ValueError(f"kurt_int must be >= 1, got {kurt_int}")
            if kurt_start < 1:
                raise ValueError(f"kurt_start must be >= 1, got {kurt_start}")
            if num_kurt < 0:
                raise ValueError(f"num_kurt must be >= 0, got {num_kurt}")

        self.invsigmin = invsigmin
        self.invsigmax = invsigmax

        self.doscaling = doscaling
        self.scalestep = scalestep

        # Component sharing (Fortran share_comps / identify_shared_comps trigger
        # amica15.f90:1856, subroutine :1916-1963): periodically merge mixing
        # columns near-collinear across DIFFERENT models so they share one
        # density and one mixing column. Multi-model only (a model cannot share
        # with itself); OFF by default so single-model (#24) and default
        # multi-model (#27) parity stay byte-for-byte. No bit-exact oracle -- the
        # reference's Spinv2 metric is declared but never allocated, so its
        # reassignment is unrunnable (the do_choose_pdfs situation, #26); this is
        # the intended algorithm, validated by real-data behavior.
        self.share_comps = share_comps
        self.share_start = share_start
        self.share_iter = share_iter
        self.comp_thresh = comp_thresh
        # Cached sphere pseudo-inverse (issues #223, #253): the sensor-space
        # back-map, shared by get_sensor_mixing_matrix and the sharing metric.
        self._sphere_pinv = None
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
        self.pcakeep = pcakeep
        self.pcadb = pcadb

        self.seed = seed

        if device is None:
            device = setup_device()
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.dtype = dtype

        if self.device.type == "mps" and self.dtype == torch.float64:
            raise ValueError(
                "MPS does not support float64. Use dtype=torch.float32 for "
                "device='mps', or device='cpu'/'cuda' for float64 parity runs."
            )

        self.iteration = 0
        self.ll_history: list[float] = []
        # Log-likelihood of the *returned* parameters (issue #51). With
        # keep_best, ``ll_history`` stays the true per-iteration trajectory
        # (which can include a late overshoot), while ``final_ll_`` is the LL of
        # the iterate fit() actually kept -- use this, not ``ll_history[-1]``, as
        # the model's fitted log-likelihood. Set by fit().
        self.final_ll_: Optional[float] = None
        # Mutual Information Reduction (MIR) waypoint trajectory (issue #137),
        # populated by fit() when
        # mir_step > 0: (iteration, mir_nats, variance) tuples from the
        # CURRENT (mid-fit) W/sphere. Like ll_history, this is a true
        # trajectory that a keep_best restore does NOT rewrite -- the
        # fit-end MIR is mir() on the returned parameters, not
        # mir_history_[-1]. Not part of state_dict(): it's a diagnostic,
        # not a fitted parameter.
        #
        # Not index-aligned with ll_history: the entry for iteration i is
        # computed AFTER that iteration's _update_parameters, while
        # ll_history[i] is the likelihood of the parameters BEFORE it (the
        # E-step accumulator that produced the update). The two therefore
        # describe states one update apart, so zipping them by index compares
        # different parameters (issue #161).
        self.mir_history_: list[tuple[int, float, float]] = []

        # Outlier-rejection bookkeeping (set up in fit()).
        self.numrej = 0
        self.good_idx: Optional[torch.Tensor] = None

        # Set by fit(): why fitting stopped ("max_iter", "nan_ll", "lrate_floor",
        # "grad_norm_floor", "min_dll", "grad_norm" -- issue #207 added the last
        # three) and how many iterations reverted Newton to natural gradient
        # (Fortran prints this; here it is exposed for parity debugging, see
        # issue #21).
        self.stop_reason: Optional[str] = None
        self.n_newton_fallbacks = 0

        # Weight-gradient-norm (Fortran ndtmpsum), recomputed every iteration by
        # _update_parameters and read by fit()'s convergence checks (issue #207).
        # None before the first _update_parameters call.
        self._ndtmpsum: Optional[float] = None

        # Populated by fit()/_initialize_parameters().
        self.A: Optional[torch.Tensor] = None
        self.W: Optional[torch.Tensor] = None
        self.c: Optional[torch.Tensor] = None
        self.mu: Optional[torch.Tensor] = None
        self.alpha: Optional[torch.Tensor] = None
        self.beta: Optional[torch.Tensor] = None
        self.rho: Optional[torch.Tensor] = None
        # Per-source density-family codes (n_channels, n_models); set in
        # _initialize_parameters and mutated by the adaptive switcher.
        self.pdtype: Optional[torch.Tensor] = None
        # Number of adaptive-switch passes already performed (Fortran numchpdf).
        self.n_kurt_done = 0
        self.gm: Optional[torch.Tensor] = None
        self.comp_list: Optional[torch.Tensor] = None
        self.mean: Optional[torch.Tensor] = None
        self.sphere: Optional[torch.Tensor] = None
        self.sldet = 0.0

        # Full-dataset per-sample/per-model log-likelihood (Fortran's LLt,
        # issue #155), computed ONCE at the end of fit() -- after any
        # keep-best restore (issue #51), so it reflects the parameters
        # actually exported, never a mid-training-loop value -- and stored as
        # compact numpy arrays (not the full sphered dataset, which would pin
        # n_channels x N x 8 bytes on the model, on GPU too). Not a fitted
        # parameter (absent from state_dict()/_PARAM_TENSORS): a model
        # restored via from_state_dict() has neither, so write_amica_output
        # writes no LLt for it.
        self._llt_lht: Optional[np.ndarray] = None
        self._llt_lt: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------
    def _preprocess(self, X: np.ndarray) -> torch.Tensor:
        """Mean-removal + sphering, matching ``pamica.AMICA._preprocess_data``.

        Done in float64 on CPU (eigh is not reliably supported on MPS and
        this is a one-time O(n_channels^3) cost, not the per-block hot
        path), then cast/moved to ``self.device``/``self.dtype``.
        """
        X_cpu = torch.from_numpy(np.ascontiguousarray(X)).to(torch.float64)
        data_dim = X_cpu.shape[0]

        if self.do_mean:
            mean = X_cpu.mean(dim=1, keepdim=True)
            X_cpu = X_cpu - mean
        else:
            mean = torch.zeros(data_dim, 1, dtype=torch.float64)

        if self.do_sphere:
            # Population covariance (divide by N), matching Fortran's DSYRK
            # scatter/N -- NOT torch.cov's default sample covariance (/(N-1)).
            # The two differ by a pure scalar sqrt(N/(N-1)); using /(N-1) leaves
            # a ~5e-6 sphere mismatch vs the reference (issue #24, check [1] of
            # .context/issue-24/root_cause_Aupdate.py).
            cov = torch.cov(X_cpu, correction=0)
            evals, evecs = torch.linalg.eigh(cov)
            order = torch.argsort(evals, descending=True)
            evals = evals[order]
            evecs = evecs[:, order]

            # Numerical-rank detection (Fortran amica15.f90:413). The policy is
            # shared with the NumPy and MLX backends so they cannot drift
            # (pamica/rank.py); only the eigenvalues cross the boundary, as a
            # read-only copy, so the sphere below stays bit-exact.
            n_comp = numerical_rank(
                evals.cpu().numpy(),
                mineig=self.mineig,
                mineig_rel=self.mineig_rel,
                pcakeep=self.pcakeep,
                pcadb=self.pcadb,
            )

            V = evecs[:, :n_comp]
            inv_sqrt = torch.diag(1.0 / torch.sqrt(evals[:n_comp]))
            if n_comp < data_dim:
                # Rank-reduced sphere: (n_comp, data_dim), so the sphered data
                # come out at the kept rank rather than staying rank-deficient
                # at data_dim rows (Fortran nw = numeigs, amica15.f90:563).
                # Vt rows are eigenvectors in descending-eigenvalue order,
                # matching Fortran's reversed Stmp2 (amica15.f90:473-479).
                w_pca = inv_sqrt @ V.T
                if self.do_approx_sphere:
                    # Fortran amica15.f90:501-508 symmetrizes the reduced
                    # whitening by the orthogonal polar factor of the leading
                    # n_comp x n_comp block of V^T:
                    #   B = (V^T)[:n, :n] = U_b S_b Vt_b
                    #   S = (V_b U_b^T) @ w_pca
                    B = evecs.T[:n_comp, :n_comp]
                    U_b, _, Vt_b = torch.linalg.svd(B)
                    sphere = (Vt_b.T @ U_b.T) @ w_pca
                else:
                    sphere = w_pca
            elif self.do_approx_sphere:
                # Symmetric ZCA sphere V diag(1/sqrt(eval)) V^T (Fortran
                # do_approx_sphere=True, amica17.f90:480-481). This is the
                # Fortran default and the parity-validated form; the old
                # diag(1/sqrt)@V^T (PCA whitening) is a different, non-symmetric
                # transform that breaks activation parity.
                sphere = V @ inv_sqrt @ V.T
            else:
                # Non-symmetric PCA whitening D^-1/2 V^T (Fortran
                # do_approx_sphere=False path, amica17.f90:495).
                sphere = inv_sqrt @ V.T

            X_cpu = sphere @ X_cpu
            # Sphering log-determinant term of the data log-likelihood
            # (Fortran ``sldet``, amica17.f90:474): sum over the kept
            # eigenvalues of -0.5*log(eval). For the PCA-reduced-rank case
            # this is a pseudo-determinant, matching Fortran which sums over
            # numeigs kept eigenvalues regardless of full rank.
            sldet = float(-0.5 * torch.log(evals[:n_comp]).sum().item())
        else:
            sphere = torch.eye(data_dim, dtype=torch.float64)
            sldet = 0.0

        self.mean = mean.to(device=self.device, dtype=self.dtype)
        self.sphere = sphere.to(device=self.device, dtype=self.dtype)
        self.sldet = sldet

        # Rank reduction shrank the sphered space, so size the model to the kept
        # rank before _initialize_parameters allocates against n_channels
        # (Fortran ``nw = numeigs``, amica15.f90:563). No-op, and therefore
        # bit-exact, whenever the data are full rank.
        n_kept = sphere.shape[0]
        if n_kept != self.n_channels:
            logger.info(
                "Data covariance has numerical rank %d of %d; fitting %d "
                "sources and mapping back to %d channels via the sphere "
                "pseudo-inverse.",
                n_kept,
                data_dim,
                n_kept,
                data_dim,
            )
            self.n_channels = n_kept
            self.n_comps = n_kept * self.n_models
        self._sphere_pinv = None  # rebuilt on demand for this fit's sphere

        return X_cpu.to(device=self.device, dtype=self.dtype)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def _initialize_parameters(self):
        """Initialize parameters, mirroring ``pamica.AMICA._initialize_parameters``
        exactly (same RNG draws, same order) so the same seed gives
        bit-identical starting parameters to the NumPy reference.
        """
        rng = np.random.RandomState(self.seed)
        n, m, ncomp, nmix = self.n_channels, self.n_models, self.n_comps, self.n_mix

        A_np = np.zeros((n, ncomp), dtype=np.float64)
        for h in range(m):
            A_np[:, h * n : (h + 1) * n] = np.eye(n) + 0.01 * (0.5 - rng.rand(n, n))

        comp_list_np = np.zeros((n, m), dtype=np.int64)
        for h in range(m):
            comp_list_np[:, h] = np.arange(h * n, (h + 1) * n)

        mu_np = np.zeros((nmix, ncomp), dtype=np.float64)
        for k in range(ncomp):
            mu_np[:, k] = np.linspace(-1, 1, nmix)
            mu_np[:, k] += 0.05 * (1 - 2 * rng.rand(nmix))

        alpha_np = np.ones((nmix, ncomp), dtype=np.float64) / nmix

        beta_np = np.ones((nmix, ncomp), dtype=np.float64)
        beta_np += 0.1 * (0.5 - rng.rand(nmix, ncomp))

        rho_np = self.rho0 * np.ones((nmix, ncomp), dtype=np.float64)
        gm_np = np.ones(m, dtype=np.float64) / m
        c_np = np.zeros((n, m), dtype=np.float64)

        self.A = torch.from_numpy(A_np).to(self.device, self.dtype)
        self.comp_list = torch.from_numpy(comp_list_np).to(self.device)
        self.mu = torch.from_numpy(mu_np).to(self.device, self.dtype)
        self.alpha = torch.from_numpy(alpha_np).to(self.device, self.dtype)
        self.beta = torch.from_numpy(beta_np).to(self.device, self.dtype)
        self.rho = torch.from_numpy(rho_np).to(self.device, self.dtype)
        self.gm = torch.from_numpy(gm_np).to(self.device, self.dtype)
        self.c = torch.from_numpy(c_np).to(self.device, self.dtype)

        # Per-source density-family codes, Fortran ``pdtype = pdftype`` (amica15.f90:
        # 611). In adaptive mode (pdftype==1) every source starts as the
        # super-Gaussian code 1 and the switcher may flip it to 4.
        self.pdtype = torch.full(
            (n, m), self.pdftype, dtype=torch.long, device=self.device
        )
        self.n_kurt_done = 0

        # Reset the mutable optimization state to the pristine constructor
        # values (lrate_cap, newtrate, rholrate are ratcheted down during
        # fit; restore them so a re-fit starts fresh).
        self.lrate = self.lrate0
        self.lrate_cap = self.lrate0
        self.newtrate = self.newtrate0
        self.rholrate = self.rholrate0
        self.iteration = 0
        self._update_unmixing_matrices()

    def _update_unmixing_matrices(self):
        """Recompute W from A via direct (batched) inversion -- never pinv."""
        assert self.A is not None and self.comp_list is not None
        A_stack = torch.stack(
            [self.A[:, self.comp_list[:, h]] for h in range(self.n_models)], dim=0
        )
        W_stack = torch.linalg.inv(A_stack)
        self.W = W_stack.permute(1, 2, 0).contiguous()

    def _pdtype_h(self, h: int) -> Optional[torch.Tensor]:
        """Per-source density-family codes for model ``h``, shaped for
        broadcasting against ``(batch, n_channels, num_mix)`` tensors, or
        ``None`` on the default ``pdftype=0`` (GG-only) fast path so the E-step
        stays bit-identical to the pre-#26 implementation.
        """
        if self.pdftype == 0:
            return None
        assert self.pdtype is not None
        return self.pdtype[:, h].view(1, -1, 1)

    # ------------------------------------------------------------------
    # E-step / M-step sufficient statistics (the hot path)
    # ------------------------------------------------------------------
    def _forward(self, X: torch.Tensor):
        """Run the E-step forward pass for one data block.

        Computes, for every model ``h``, the activations ``b``, scaled
        activations ``y``, normalized mixture responsibilities ``z``, the power
        ``|y|^rho`` (reused by the rho update), and the per-sample per-model
        log-likelihood ``logV`` (including the ``log|det W|`` and ``sldet``
        Jacobian terms, matching Fortran's ``Ptmp`` seed, amica17.f90:1273).
        Shared by ``_get_block_updates`` (which reduces it into sufficient
        statistics) and ``_block_sample_ll`` (which only needs ``logV``).

        Returns
        -------
        logV : torch.Tensor of shape (batch, n_models)
        b_list, z_list, y_list, azrho_list : lists (one entry per model) of
            per-model tensors (``b``: (batch, n_channels); ``z``/``y``/``azrho``:
            (batch, n_channels, n_mix)).
        """
        assert (
            self.comp_list is not None
            and self.c is not None
            and self.W is not None
            and self.mu is not None
            and self.beta is not None
            and self.rho is not None
            and self.alpha is not None
            and self.gm is not None
        )
        batch_size = X.shape[1]
        num_models = self.n_models
        b_list, z_list, y_list, azrho_list = [], [], [], []
        logV = torch.empty(batch_size, num_models, dtype=self.dtype, device=self.device)

        for h in range(num_models):
            idx = self.comp_list[:, h]
            # Activation b = W(x - c): c is the per-model data-space center.
            # Fortran subtracts wc in the E-step (amica17.f90:1280-1292), where
            # wc = W@c is precomputed in get_unmixing_matrices (amica17.f90:2178).
            # Subtracting c in data space before W is equivalent and keeps c's
            # semantics identical to Fortran's. For n_models=1, c == 0, so this is
            # bit-identical to the old X.T @ W.
            b = (X - self.c[:, h].unsqueeze(1)).T @ self.W[:, :, h]  # (batch, n_ch)

            mu_h = self.mu[:, idx].T.unsqueeze(0)  # (1, n_channels, num_mix)
            beta_h = self.beta[:, idx].T.unsqueeze(0)
            rho_h = self.rho[:, idx].T.unsqueeze(0)
            alpha_h = self.alpha[:, idx].T.unsqueeze(0)

            y = beta_h * (b.unsqueeze(-1) - mu_h)  # (batch, n_channels, num_mix)
            # Only log_pdf is needed here; the score fp (and drho's |y|^rho) are
            # reused in _get_block_updates. az_rho = |y|^rho is threaded through
            # so the rho-update does not recompute it (issue #63).
            log_pdf, az_rho = _log_pdf_only(y, rho_h, self._pdtype_h(h))

            # z0 = log(alpha) + log(beta) + log_pdf. For the single-component
            # families (codes 1/4) n_mix==1 so alpha==1 and log(alpha)==0, which
            # reproduces Fortran's alpha-free z0 (amica15.f90:1358/1370).
            z0 = torch.log(alpha_h) + torch.log(beta_h) + log_pdf
            ll_i = torch.logsumexp(
                z0, dim=-1
            )  # (batch, n_channels) -- per-source log-density
            z = torch.softmax(z0, dim=-1)  # normalized responsibilities

            logdet_W = torch.linalg.slogdet(self.W[:, :, h])[1]
            logV[:, h] = (
                torch.log(self.gm[h]) + logdet_W + self.sldet + ll_i.sum(dim=-1)
            )

            b_list.append(b)
            z_list.append(z)
            y_list.append(y)
            azrho_list.append(az_rho)

        return logV, b_list, z_list, y_list, azrho_list

    def _block_sample_ll(self, X: torch.Tensor) -> torch.Tensor:
        """Per-sample total log-likelihood for a data block (the rejection
        statistic; Fortran ``P``/``loglik``, amica17.f90:1372)."""
        logV, *_ = self._forward(X)
        return torch.logsumexp(logV, dim=1)  # (batch,)

    def _compute_full_posterior_ll(
        self, X_t: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Recompute the per-model/per-sample log-likelihood over every sample
        of ``X_t`` (the sphered training data), for the Fortran ``LLt`` output
        (issue #155). Called once from ``fit()`` (after any keep-best restore)
        with the full dataset; not retained on ``self`` afterward.

        Reuses ``_forward``'s ``logV`` (Fortran's ``modloglik``: already
        includes the ``log|det W|`` + ``sldet`` Jacobian terms) rather than any
        value accumulated during training, so this reflects ``self``'s current
        parameters -- correct even after a keep-best rollback (issue #51) to an
        earlier iterate than the training loop's last.

        Deliberate divergence from Fortran: Fortran's ``modloglik`` is filled
        during iteration i's E-step, the M-step then updates the parameters,
        and ``write_output`` writes both -- so Fortran's on-disk LLt is stale
        by one M-step relative to the parameters written alongside it. This
        method recomputes from the POST-update (and, here, post-keep-best-
        restore) parameters, so pamica's LLt is self-consistent with the
        written W/A (better-behaved, not "fixed toward Fortran" -- do not
        change this to match Fortran's staleness).

        Under ``do_reject``, only the good set (``self.good_idx``) is scored --
        Fortran zeroes a rejected sample's ``modloglik``/``loglik`` on write
        (amica15.f90:2231-2234) and ``load_rej`` uses that exact zero as the
        rejection sentinel (``sum(modloglik(:,i)) == 0.0``, amica15.f90:
        907), so rejected columns of the returned arrays are left at their
        zero-initialized value rather than computed and discarded -- this also
        avoids running rejected outliers through the model for the first time
        at write time.

        Returns
        -------
        Lht : ndarray of shape (n_models, n_samples). Zero for rejected
            samples under ``do_reject``.
        Lt : ndarray of shape (n_samples,). Zero for rejected samples under
            ``do_reject``.
        """
        n_samples = X_t.shape[1]
        Lht = np.zeros((self.n_models, n_samples))
        Lt = np.zeros(n_samples)

        if self.do_reject:
            assert self.good_idx is not None
            idx = self.good_idx.detach().cpu().numpy()
            X_use = X_t[:, self.good_idx]
        else:
            idx = np.arange(n_samples)
            X_use = X_t
        n_use = X_use.shape[1]

        for start in range(0, n_use, self.block_size):
            end = min(start + self.block_size, n_use)
            logV, *_ = self._forward(X_use[:, start:end])
            lt_block = torch.logsumexp(logV, dim=1)
            cols = idx[start:end]
            Lht[:, cols] = logV.T.detach().cpu().numpy()
            Lt[cols] = lt_block.detach().cpu().numpy()

        return Lht, Lt

    def _get_block_updates(self, X: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute sufficient-statistic accumulators for one data block.

        Fortran-faithful exact-EM statistics (amica17.f90:1437-1592), validated
        against the reference binary to machine precision (issue #24). Unlike a
        first-order gradient M-step, the mixture updates use exact-EM numerator/
        denominator pairs and the score ``fp = rho*sign(y)*|y|^(rho-1)`` (``_score``,
        Fortran ``fp``) rather than the density derivative ``dpdf``:

        * ``dmu_n = sum(u*fp)``, ``dmu_d = sbeta*sum(u*fp/y)``   (mu += dmu_n/dmu_d)
        * ``dbeta_n = sum(u)``, ``dbeta_d = sum(u*fp*y)``        (beta *= sqrt(n/d))
        * ``drho_n = rho*sum(u*|y|^rho*ln|y|)``                  (rho digamma update)
        * ``dWtmp = g^T b`` with ``g = sum_j sbeta*u*fp``        (natural gradient)

        where ``u = v*z`` (model x mixture responsibility). ``ll`` is the correct
        pre-normalization ``logsumexp`` (see module docstring).

        Assumes ``rho <= 2`` (the ``maxrho`` default); the ``rho > 2`` denominator
        branches of Fortran (:1539/:1551) are unreachable and not implemented.

        Returns
        -------
        updates : dict with ``dgm`` (n_models,), ``dalpha_n``/``dmu_n``/``dmu_d``/
            ``dbeta_n``/``dbeta_d``/``drho_n`` (n_mix, n_comps), ``dWtmp``
            (n_channels, n_channels, n_models), ``dc_numer`` (n_channels,
            n_models; the data-space bias numerator ``sum_t v_h*x``, issue #27),
            ``ll`` (scalar), and -- when ``do_newton`` -- ``dsigma2_numer``,
            ``dkappa_numer``, ``dlambda_numer`` (see ``_finalize_newton_stats``).
        """
        assert (
            self.comp_list is not None
            and self.beta is not None
            and self.rho is not None
        )
        num_mix, num_models = self.n_mix, self.n_models
        dev, dt = self.device, self.dtype

        logV, b_list, z_list, y_list, azrho_list = self._forward(X)
        block_ll = torch.logsumexp(logV, dim=1).sum()
        v = torch.softmax(logV, dim=1)  # (batch, num_models)

        def zeros(*shape):
            return torch.zeros(*shape, dtype=dt, device=dev)

        dgm = zeros(num_models)
        dalpha_n = zeros(num_mix, self.n_comps)
        dmu_n = zeros(num_mix, self.n_comps)
        dmu_d = zeros(num_mix, self.n_comps)
        dbeta_n = zeros(num_mix, self.n_comps)
        dbeta_d = zeros(num_mix, self.n_comps)
        drho_n = zeros(num_mix, self.n_comps)
        dWtmp = zeros(self.n_channels, self.n_channels, num_models)
        dc_numer = zeros(self.n_channels, num_models)
        do_newton = self.do_newton
        if do_newton:
            dsigma2_numer = zeros(self.n_channels, num_models)
            dkappa_numer = zeros(num_mix, self.n_channels, num_models)
            dlambda_numer = zeros(num_mix, self.n_channels, num_models)
        tiny = torch.finfo(dt).tiny

        for h in range(num_models):
            idx = self.comp_list[:, h]
            b, zr, y = b_list[h], z_list[h], y_list[h]
            v_h = v[:, h]
            beta_h = self.beta[:, idx].T.unsqueeze(0)  # sbeta, (1, n_ch, num_mix)
            rho_h = self.rho[:, idx].T  # (n_ch, num_mix)
            # score fp; the family select-case is amica15.f90:1467-1491 (amica17
            # is GG-only, so cite the binary's source explicitly here).
            fp = _score(y, rho_h.unsqueeze(0), self._pdtype_h(h))
            u = v_h.unsqueeze(-1).unsqueeze(-1) * zr  # u = v*z (:1439)
            ufp = u * fp  # (:1485)

            dgm[h] = v_h.sum()
            dalpha_n.index_add_(1, idx, u.sum(0).T)  # sum(u) (:1524)
            dmu_n.index_add_(1, idx, ufp.sum(0).T)  # sum(ufp) (:1532)
            # mu denominator sbeta*sum(ufp/y) (:1537). In float32 a sample sitting
            # on a mixture mean can round y to *exactly* 0; the score fp(0)=0 (for
            # the supported rho>=1), so the raw ufp/y is 0/0 = NaN -- the sole
            # trigger of the full-data float32 divergence (issue #75; NOT a
            # summation-precision problem, so compensated accumulation does not
            # help). The true term ufp/y = u*rho*|y|^(rho-2) is NOT 0 in the limit:
            # a nonzero constant at rho==2, and an integrable singularity that
            # diverges as y->0 for rho<2 -- so once y underflows to exactly 0 the
            # real contribution is unrepresentable. Substituting 0 (ufp==0 there,
            # so 0/1) drops that one sample instead of poisoning all of dmu_d with a
            # NaN: a bounded, empirically negligible bias (it fires <=1 sample per
            # iteration on the sample EEG, and float32 still matches the float64 LL
            # to ~5 sig digits). float64 never rounds y to exactly 0, so the guard
            # is a bit-identical no-op there (single-model #24 parity preserved),
            # and it needs no float64, so it also stabilizes the MPS/float32 path.
            safe_y = torch.where(y == 0, torch.ones_like(y), y)
            dmu_d.index_add_(
                1, idx, (beta_h.squeeze(0) * (ufp / safe_y).sum(0)).T
            )  # (:1537)
            dbeta_n.index_add_(1, idx, u.sum(0).T)  # sum(u) (:1550)
            dbeta_d.index_add_(1, idx, (ufp * y).sum(0).T)  # sum(ufp*y) (:1556)

            # drho_numer = rho * sum(u*|y|^rho*ln|y|)  (:1560-1578). The leading
            # rho comes from ln(|y|^rho)=rho*ln|y| in the Fortran logab chain
            # (issue #24 Bug 1). Guard only the per-sample underflow (:1570) --
            # no per-component (rho!=1&rho!=2) mask (Bug 2): |y|^rho*ln|y| is 0 at
            # y=0, and clamping the log input makes the product collapse there.
            ay = y.abs()
            ayrho = azrho_list[h]  # |y|^rho reused from _forward (issue #63)
            logab = rho_h.unsqueeze(0) * torch.log(ay.clamp_min(tiny))  # rho*ln|y|
            logab = torch.where(ayrho < _EPSDBLE, torch.zeros_like(logab), logab)
            drho_n.index_add_(1, idx, (u * (ayrho * logab)).sum(0).T)

            g = (beta_h * ufp).sum(-1)  # g_i = sum_j sbeta*ufp (:1493)
            dWtmp[:, :, h] = g.T @ b  # source-space sum g_t b_t^T (:1592)
            # Data-space bias accumulator: dc_numer[i,h] = sum_t v_h(t)*x(i,t)
            # (Fortran :1423-1429). The denominator is dgm[h] = sum_t v_h(t).
            # NOTE: this replaces the old gradient-style bias g.sum(0), which was
            # accumulated but never applied (c was frozen at 0); the Fortran
            # update is the data-space responsibility-weighted mean (issue #27).
            dc_numer[:, h] = X @ v_h

            if do_newton:
                # Newton curvature accumulators (Fortran amica17.f90:1419,
                # 1500-1514), in terms of the score fp (not dpdf).
                dsigma2_numer[:, h] = (v_h.unsqueeze(-1) * b.pow(2)).sum(0)  # (:1419)
                dkappa_numer[:, :, h] = (
                    (u * fp.pow(2)).sum(0) * beta_h.squeeze(0).pow(2)
                ).T  # (:1500)
                dlambda_numer[:, :, h] = (u * (fp * y - 1.0).pow(2)).sum(0).T  # (:1511)

        updates = {
            "dgm": dgm,
            "dalpha_n": dalpha_n,
            "dmu_n": dmu_n,
            "dmu_d": dmu_d,
            "dbeta_n": dbeta_n,
            "dbeta_d": dbeta_d,
            "drho_n": drho_n,
            "dWtmp": dWtmp,
            "dc_numer": dc_numer,
            "ll": block_ll,
        }
        if do_newton:
            updates["dsigma2_numer"] = dsigma2_numer
            updates["dkappa_numer"] = dkappa_numer
            updates["dlambda_numer"] = dlambda_numer
        return updates

    def _accumulate_blocks(self, X: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Sum sufficient statistics over all blocks of ``X``.

        Peak memory scales with ``block_size`` (each block's intermediates
        are freed once accumulated), not with ``X.shape[1]``.
        """
        n_samples = X.shape[1]
        acc: Optional[Dict[str, torch.Tensor]] = None
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
    # M-step parameter update
    # ------------------------------------------------------------------
    def _finalize_newton_stats(self, acc: Dict[str, torch.Tensor]):
        """Reduce the Newton block accumulators into ``(sigma2, lambda, kappa)``.

        Ports the Fortran finalization (amica17.f90:1762-1776). The Fortran
        ``baralpha``/``dkappa_denom``/``dlambda_denom`` responsibility masses
        all cancel algebraically against the per-mixture ``dalpha`` weighting,
        leaving simply (with ``dgm = sum_t v_h`` the raw model mass):

            sigma2[i,h] = dsigma2_numer[i,h] / dgm[h]
            kappa[i,h]  = sum_j dkappa_numer[j,i,h] / dgm[h]
            lambda[i,h] = sum_j (dlambda_numer[j,i,h]
                                 + dkappa_numer[j,i,h] * mu[j,comp(i,h)]^2) / dgm[h]

        Returns (sigma2, lambda_, kappa), each (n_channels, n_models).
        """
        assert self.mu is not None and self.comp_list is not None
        dgm = acc["dgm"].unsqueeze(0)  # (1, n_models)
        sigma2 = acc["dsigma2_numer"] / dgm
        kappa = acc["dkappa_numer"].sum(dim=0) / dgm
        # mu at each source's component: mu[j, comp_list[i,h]] -> (n_mix, n_ch, n_models)
        mu_at = self.mu[:, self.comp_list]
        lambda_ = (acc["dlambda_numer"] + acc["dkappa_numer"] * mu_at.pow(2)).sum(
            dim=0
        ) / dgm
        return sigma2, lambda_, kappa

    def _newton_direction(self, dA_h, sigma2_h, lambda_h, kappa_h):
        """Per-model Newton direction ``H`` from the natural gradient ``dA_h``.

        Vectorized port of the per-source-pair 2x2 solve (amica17.f90:1817-1832,
        pamica.py:802-813):

            H[i,i] = dA_h[i,i] / lambda[i]
            sk1 = sigma2[i]*kappa[k];  sk2 = sigma2[k]*kappa[i]   (i != k)
            H[i,k] = (sk1*dA_h[i,k] - dA_h[k,i]) / (sk1*sk2 - 1)  if sk1*sk2 > 1

        Returns ``(H, posdef)``. ``posdef`` is False if any off-diagonal pair
        fails ``sk1*sk2 > 1`` (the positive-definiteness guard); the caller
        then falls back to the natural gradient for this model.
        """
        n = self.n_channels
        sk1 = sigma2_h.unsqueeze(1) * kappa_h.unsqueeze(0)  # [i,k] = sigma2[i]*kappa[k]
        sk2 = sigma2_h.unsqueeze(0) * kappa_h.unsqueeze(1)  # [i,k] = sigma2[k]*kappa[i]
        prod = sk1 * sk2
        valid = prod > 1.0
        denom = torch.where(valid, prod - 1.0, torch.ones_like(prod))
        h_off = (sk1 * dA_h - dA_h.T) / denom
        H = torch.where(valid, h_off, torch.zeros_like(h_off))
        # Diagonal overrides (uses lambda, not the off-diagonal formula).
        diag = torch.diagonal(dA_h) / lambda_h
        H = H - torch.diag(torch.diagonal(H)) + torch.diag(diag)
        # Positive-definite iff every off-diagonal pair passed the guard.
        offdiag = ~torch.eye(n, dtype=torch.bool, device=dA_h.device)
        posdef = bool(valid[offdiag].all().item())
        return H, posdef

    def _update_parameters(self, acc: Dict[str, torch.Tensor], n_samples: int):
        """Apply the M-step parameter update, matching
        ``pamica.AMICA._update_parameters`` (natural-gradient and Newton).

        ``n_samples`` is the number of samples that fed the accumulators (the
        good-sample count when ``do_reject`` is active), so ``gm`` and the
        reported log-likelihood are normalized by the effective sample count.

        The mixture parameters use exact-EM fixed-point updates (no ``lrate``);
        only the ``A``/``W`` step is scaled by ``lrate`` (Fortran amica17.f90:
        1890-2035). The per-model data-space bias ``c`` uses Fortran's exact-EM
        ``update_c`` (amica17.f90:1423-1429/1899-1901): ``c[i,h] = sum_t v_h*x /
        sum_t v_h``, the responsibility-weighted data mean for model ``h``. For
        ``n_models=1`` on mean-removed data ``v == 1`` so ``c`` collapses to the
        (zero) data mean; the update is skipped there so single-model parity stays
        bit-exact (issue #24). For ``n_models>1`` the per-model ``v`` is
        non-uniform and ``c`` moves each iteration (issue #27).
        """
        assert (
            self.c is not None
            and self.alpha is not None
            and self.mu is not None
            and self.beta is not None
            and self.rho is not None
            and self.A is not None
            and self.comp_list is not None
        )
        # Fortran builds dAk from the previous iteration's model weights: gm is
        # not reassigned until update_params (amica15.f90:1788+), after
        # accum_updates_and_likelihood (:1731-1743). Snapshot before overwriting so
        # dAk -- which both drives the A-update below and reports ndtmpsum, unlike
        # numpy_impl where it is only the diagnostic -- weights the way Fortran
        # does (issue #219). Cloned rather than aliased: gm is only ever rebound
        # today, but an in-place write elsewhere would silently corrupt this.
        assert self.gm is not None
        gm_prev = self.gm.clone()
        self.gm = acc["dgm"] / n_samples

        # Per-model data-space bias (Fortran's `update_c` flag, amica17.f90:1423-
        # 1429 numerator / :1899-1901 division). Skipped for a single model to keep
        # the issue #24 parity bit-exact: with v==1 the update would add a ~1e-13
        # float-sum residual of the (mean-removed) data, perturbing the
        # otherwise-exact single-model trajectory. dgm[h] = sum_t v_h(t) is the
        # denominator (Fortran `dc_denom`). A fully-dead model (dgm[h]==0 => v_h==0
        # for all t, so dc_numer[:,h]==0 too) gives 0/0; keep its PRIOR c rather
        # than write a NaN. A NaN c would poison the NEXT iteration's cross-model
        # softmax for EVERY model (unlike log(gm[h])=-inf, which softmax tolerates,
        # so a dead model was previously inert) -- this containment mirrors the
        # mu/beta/rho non-finite guards below. `dgm>0` is also False for a NaN dgm
        # from upstream corruption, so that is contained too.
        if self.n_models > 1:
            dgm = acc["dgm"]
            live = dgm > 0.0
            new_c = acc["dc_numer"] / dgm.clamp_min(torch.finfo(self.dtype).tiny)
            self.c = torch.where(live.unsqueeze(0), new_c, self.c)
            if not bool(live.all()):
                logger.warning(
                    "Zero-responsibility model(s) at iter %d; kept their prior "
                    "bias c (dead-model guard).",
                    self.iteration,
                )

        # Component sharing (#60): a component that was merged away is no longer
        # referenced by comp_list, so no sufficient statistic accumulates into
        # its column (dalpha_n/dmu_d/dbeta_d == 0) and the divisions below would
        # be 0/0 = NaN. Update only USED columns and freeze the rest at their
        # last finite value (Fortran carries NaN there harmlessly behind its
        # comp_used mask; we keep them finite so save/the degenerate guard are
        # not tripped). With the default full comp_list every column is used, so
        # ``used`` is all-True and every update below is byte-for-byte unchanged.
        used = self.comp_used.unsqueeze(0)  # (1, n_comps)

        self.alpha = torch.where(
            used, acc["dalpha_n"] / acc["dalpha_n"].sum(dim=0, keepdim=True), self.alpha
        )

        # Finalize the Newton curvature with the PRE-update mu. Fortran folds the
        # mu^2 term into lambda during E-step accumulation, before the M-step
        # moves mu (amica17.f90:1762-1774), and the NumPy port bakes it in at
        # accumulation time. Do it here, before self.mu is reassigned below, so
        # lambda uses this iteration's mu rather than the updated one.
        newton_active = self.do_newton and self.iteration >= self.newt_start
        if newton_active:
            sigma2, lambda_, kappa = self._finalize_newton_stats(acc)

        # Exact-EM mixture location/scale (Fortran :1978/:1993). No lrate.
        # ``used`` masks merged-away columns (no-op for the default comp_list).
        self.mu = torch.where(used, self.mu + acc["dmu_n"] / acc["dmu_d"], self.mu)
        self.beta = torch.where(
            used,
            torch.clamp(
                self.beta * torch.sqrt(acc["dbeta_n"] / acc["dbeta_d"]),
                self.invsigmin,
                self.invsigmax,
            ),
            self.beta,
        )
        # Fortran keeps a live "NaN in sbeta!" canary here (amica17.f90:1996-2000).
        # The exact-EM mu/beta divisions are unguarded (matching Fortran, whose own
        # mu/beta guard is commented out), so surface a non-finite value here
        # instead of letting it propagate to a later, unattributable nan-LL stop.
        if (
            not torch.isfinite(self.mu).all()
            or not torch.isfinite(self.beta).all()
            or not torch.isfinite(self.alpha).all()
        ):
            logger.warning(
                "Non-finite mu/beta/alpha at iter %d (a mixture component's mass "
                "likely collapsed).",
                self.iteration,
            )

        # GG shape update with the 1/psi(1+1/rho) digamma factor (Fortran
        # :2013-2014); the divisor is the per-component responsibility mass
        # dalpha_n (floored so a near-empty component cannot poison rho). A NaN
        # here (e.g. from upstream mu/beta corruption) is reset to rho0 -- but
        # logged first, so the reset does not silently erase the failure origin.
        # Skipped for every non-GG family: Fortran sets dorho=.false. when
        # pdftype/=0 (amica15.f90:3704), freezing rho at rho0.
        if (
            self.dorho
            and not torch.all(self.rho == 1.0)
            and not torch.all(self.rho == 2.0)
        ):
            drho = acc["drho_n"] / acc["dalpha_n"].clamp_min(1e-8)
            psi = torch.special.digamma(1.0 + 1.0 / self.rho)
            new_rho = self.rho + self.rholrate * (1.0 - (self.rho / psi) * drho)
            nan_mask = torch.isnan(new_rho)
            if nan_mask.any():
                logger.warning(
                    "NaN in rho update at iter %d for %d component(s); resetting "
                    "to rho0=%g.",
                    self.iteration,
                    int(nan_mask.sum()),
                    self.rho0,
                )
                new_rho = torch.where(
                    nan_mask, torch.full_like(new_rho, self.rho0), new_rho
                )
            self.rho = torch.where(
                used, torch.clamp(new_rho, self.minrho, self.maxrho), self.rho
            )

        # --- A / W update: natural gradient, optionally Newton-preconditioned.
        # A is stored as Fortran's A^T (the true unmixing is W^T = inv(A)^T), so
        # Fortran's A_fort -= lrate*A_fort @ dir becomes, transposed,
        # A -= lrate*dir^T @ A (LEFT-multiply by the TRANSPOSED direction). The
        # direction ``dir`` (natural gradient I - <g b^T>/dgm, or its Newton
        # precondition) is built in Fortran's untransposed convention. Getting
        # this wrong (right-multiply by the untransposed dir) is invisible at the
        # fixed point but sends the free-running fit downhill -- issue #24 root
        # cause (.context/issue-24/root_cause_Aupdate.py, machine-exact check).
        # (newton_active / sigma2 / lambda_ / kappa were finalized above.)
        #
        # The direction/dAk/gradient-norm computation below runs UNCONDITIONALLY,
        # not gated on _a_frozen(): Fortran computes dAk and ndtmpsum every
        # iteration in accum_updates_and_likelihood (amica15.f90:1749-1761),
        # strictly before the LATER, separate update_A block (amica15.f90:1803)
        # that actually steps A and is guarded by the share-freeze window. Only
        # the step itself -- and the lrate ramp / Newton-fallback bookkeeping
        # that Fortran nests inside that same guarded block -- are conditional on
        # ``not self._a_frozen()`` (issue #207: the grad-norm stop needs
        # ndtmpsum to reflect the true gradient magnitude every iteration, not
        # just the iterations where A actually moves). _a_frozen() is always
        # False when sharing is off, so the default path recomputes exactly what
        # it always did, just with the gate narrowed.
        eye = torch.eye(self.n_channels, dtype=self.dtype, device=self.device)
        directions = []
        no_newt = False
        for h in range(self.n_models):
            dA_h = -acc["dWtmp"][:, :, h] / acc["dgm"][h] + eye  # I - <g b^T>/dgm
            if newton_active:
                H, posdef = self._newton_direction(
                    dA_h, sigma2[:, h], lambda_[:, h], kappa[:, h]
                )
                if posdef:
                    directions.append(H)
                else:
                    no_newt = True
                    directions.append(dA_h)  # fall back to natural gradient
            else:
                directions.append(dA_h)

        # Accumulate each model's natural-gradient/Newton contribution per
        # mixing COLUMN as a gm-WEIGHTED AVERAGE (Fortran dAk/zeta,
        # amica15.f90:1749-1761): dAk = sum_h gm[h]*dir_h scattered by
        # comp_list, zeta = sum_h gm[h] per column, then dAk /= zeta. For the
        # default disjoint comp_list every column has exactly one contributor,
        # so gm cancels (dAk = dir) and single-model (gm=[1]) is byte-for-byte
        # unchanged; for a SHARED column (issue #60) the step is Fortran's
        # responsibility-weighted average, NOT a raw sum (a raw sum would
        # over-step by the contributor count and destabilize the fit).
        dAk = torch.zeros_like(self.A)
        zeta = torch.zeros(self.n_comps, dtype=self.dtype, device=self.device)
        for h in range(self.n_models):
            idx = self.comp_list[:, h]
            dAk.index_add_(1, idx, gm_prev[h] * (directions[h].T @ self.A[:, idx]))
            zeta.index_add_(0, idx, gm_prev[h].expand(idx.shape[0]))
        dAk = dAk / zeta.clamp_min(torch.finfo(self.dtype).tiny)

        # Weight-gradient norm (Fortran ndtmpsum, amica15.f90:1760-1761):
        # ``sqrt(sum(dAk**2, mask=comp_used) / (nw*count(comp_used)))``. Read by
        # fit()'s convergence checks (issue #207); the comp_used mask matters
        # only when share_comps has merged/frozen columns (all-True otherwise,
        # so ``comp_used_mask`` covers every column and this is a plain RMS over
        # dAk). Named distinctly from the ``used`` (1, n_comps) broadcast mask
        # above (alpha/mu/beta/rho updates) to avoid shadowing it.
        comp_used_mask = self.comp_used
        n_used = int(comp_used_mask.sum().item())
        nd = (dAk**2).sum(dim=0)  # (n_comps,)
        self._ndtmpsum = float(
            torch.sqrt(
                nd[comp_used_mask].sum() / (self.n_channels * max(n_used, 1))
            ).item()
        )

        # A-update. When sharing holds A this iteration (the post-merge settle
        # window, Fortran amica15.f90:1803), skip the step -- lrate ramp,
        # Newton-fallback bookkeeping, and the DAXPY itself -- so a discarded
        # Newton direction cannot pollute the fallback counter.
        if not self._a_frozen():
            if newton_active and no_newt:
                # Fortran prints "Hessian not positive definite, using natural
                # gradient" (amica15.f90:1809-1811). Surface the same signal so an
                # all-fallback run (issue #21) is visible without re-instrumenting.
                self.n_newton_fallbacks += 1
                logger.warning(
                    "Newton not positive definite at iter %d; using natural gradient.",
                    self.iteration,
                )

            # Learning-rate ramp: toward newtrate while Newton is active and
            # stable, otherwise toward lrate0 (Fortran amica15.f90:1804-1815).
            # Ramped after mu/beta/rho (exact-EM, lrate-free) and before A.
            if newton_active and not no_newt:
                self.lrate = min(
                    self.newtrate, self.lrate + min(1.0 / self.newt_ramp, self.lrate)
                )
            else:
                self.lrate = min(
                    self.lrate_cap, self.lrate + min(1.0 / self.newt_ramp, self.lrate)
                )

            self.A = self.A - self.lrate * dAk

        if self.doscaling and (self.iteration % self.scalestep == 0):
            assert self.A is not None and self.mu is not None and self.beta is not None
            scale = torch.sqrt((self.A**2).sum(dim=0))  # (n_comps,)
            nonzero = scale > 0
            self.A[:, nonzero] = self.A[:, nonzero] / scale[nonzero]
            self.mu[:, nonzero] = self.mu[:, nonzero] * scale[nonzero]
            self.beta[:, nonzero] = self.beta[:, nonzero] / scale[nonzero]

        self._update_unmixing_matrices()

    def _a_frozen(self) -> bool:
        """Whether the A-update (and its lrate ramp) is held this iteration.

        A is frozen for the first 6 iterations of every ``share_iter``-length
        window once ``iter >= share_start`` -- i.e. the merge iteration and the 5
        after it -- so the density parameters can settle onto any freshly merged
        component before the mixing matrix moves again (Fortran A-freeze,
        amica15.f90:1803). The window fires each cycle regardless of whether that
        cycle's ``_identify_shared_comps`` actually merged a pair.

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
        ``comp_thresh`` cutoff::

            t0 = |a . b| / (||a|| ||b||),   a = Spinv A[:,ci], b = Spinv A[:,cj]

        where ``Spinv = pinv(sphere)`` de-spheres the columns back to
        input-channel (sensor) space, so the similarity compares scalp maps.

        The pseudo-inverse -- not a true inverse -- is the faithful back-map:
        the reference carries exactly this, ``Spinv(nx, numeigs)``, whenever
        rank/PCA reduction is active (amica15.f90:568-578). Invertibility was
        never a mathematical requirement of the merge metric, only of the way it
        used to be computed (issue #253). Two consequences:

        * Full rank, square sphere: ``pinv == inv`` to ~1e-15, orders of
          magnitude below any ``comp_thresh`` (~0.99) decision boundary, so
          merge decisions on well-conditioned data are unchanged.
        * Rank-reduced sphere ``(n_kept, n_channels)`` (issue #223), or a square
          sphere fitted on rank-deficient data (Maxwell-filtered MEG,
          average-referenced EEG): ``pinv`` maps each column into the retained
          sensor subspace instead of failing. For the reduced PCA sphere
          ``S = D^-1/2 V_r^T`` this is ``pinv(S) = V_r D^1/2``, i.e. a
          de-whitening followed by the orthonormal ``U_r = V_r`` embedding
          proposed in issue #221; the embedding leaves the cosine untouched, so
          this is the same comparison the full-rank path makes, evaluated in the
          subspace the data actually occupy.

        On a match, ``cj`` is folded into ``ci``: every ``comp_list`` entry equal
        to ``cj`` is reassigned to ``ci``, so the two now share one mixing column
        and one density (the M-step already accumulates every sufficient
        statistic through ``comp_list`` via index_add, so shared components sum
        automatically).

        Greedy and order-dependent, matching the reference's quadruple loop.
        Skips a pair already merged, or one whose two columns coexist in some
        single model (a model cannot share a component with itself).

        No bit-exact oracle: ``Spinv2`` is *declared* in the reference headers
        but never *allocated* anywhere in ``amica15.f90``/``amica17.f90`` (unlike
        ``Spinv``, allocated at :551), so invoking ``identify_shared_comps`` with
        ``share_comps=.true.`` would read an unallocated array through ``DGEMV``
        -- undefined behavior, most likely a crash, not a benign no-op. The
        routine is effectively unrunnable in the reference (cf. the also-dead
        ``do_choose_pdfs`` switch, #26), so this implements the intended
        algorithm and validates it on real data, not against byte parity.
        """
        if self.n_models < 2:
            return
        assert self.A is not None and self.comp_list is not None
        # _pinv_sphere raises on a non-finite sphere, so the metric below can
        # only be garbage if A itself is (guarded per-pair in the scan).
        spinv = self._pinv_sphere()
        # De-sphered mixing columns in sensor space, on CPU for the small greedy
        # scan (n_models^2 * n_channels^2 pairs; avoids per-element GPU syncs).
        atil = (spinv @ self.A).detach().cpu().numpy()
        norms = np.linalg.norm(atil, axis=0)
        cl = self.comp_list.detach().cpu().numpy().copy()  # (nw, n_models)
        nw, m = cl.shape
        tiny = np.finfo(atil.dtype).tiny
        merged = 0
        for h in range(m):
            for hh in range(h + 1, m):
                for i in range(nw):
                    for ii in range(nw):
                        ci, cj = int(cl[i, h]), int(cl[ii, hh])
                        if ci == cj:
                            continue
                        t0 = abs(atil[:, ci] @ atil[:, cj]) / (
                            norms[ci] * norms[cj] + tiny
                        )
                        # NaN t0 (e.g. a zero-norm column) must NOT merge:
                        # `NaN < thresh` is False, so guard finiteness explicitly.
                        if not np.isfinite(t0) or t0 < self.comp_thresh:
                            continue
                        # A model cannot share a component with itself: skip if
                        # any single model already uses both columns.
                        if any(
                            (cl[:, k] == ci).any() and (cl[:, k] == cj).any()
                            for k in range(m)
                        ):
                            continue
                        cl[cl == cj] = ci  # fold cj into ci everywhere
                        merged += 1
        if merged:
            self.comp_list = torch.from_numpy(cl).to(self.comp_list.device)
            logger.info(
                "Component sharing (iter %d): %d merge(s), %d unique components.",
                self.iteration,
                merged,
                int(np.unique(cl).size),
            )

    def _pinv_sphere(self) -> torch.Tensor:
        """Cached ``pinv(sphere)``: the back-map from sphered to input-channel space.

        This is the Fortran ``Spinv`` (amica15.f90:568-578), which the reference
        also builds as a pseudo-inverse, ``Spinv(nx, numeigs)``, under rank/PCA
        reduction. A pseudo-inverse rather than an inverse because reduction
        leaves the sphere non-square (issue #223) and a square sphere fitted on
        rank-deficient data is singular; for a full-rank square sphere the two
        agree to ~1e-15. Built on first use and invalidated per fit in
        :meth:`_preprocess` (and on :meth:`_load_params`), so it can never
        describe a sphere other than the current one.
        """
        assert self.sphere is not None
        if self._sphere_pinv is None:
            if not torch.isfinite(self.sphere).all():
                # Only a degenerate fit (non-finite input data) gets here. Say
                # so, rather than letting LAPACK report a confusing
                # "ill-conditioned / repeated singular values" SVD failure.
                raise RuntimeError(
                    "The sphere holds non-finite values, so it has no "
                    "pseudo-inverse: the fit is degenerate. Check the input "
                    "data for NaN/inf."
                )
            self._sphere_pinv = torch.linalg.pinv(self.sphere)
        return self._sphere_pinv

    @property
    def comp_used(self) -> torch.Tensor:
        """Boolean mask (n_comps,) of components still referenced by comp_list.

        A component drops out of use when it is folded into another by
        :meth:`_identify_shared_comps`; unused columns receive no gradient and
        are never read by the E-step. Derived from ``comp_list`` (not stored).
        """
        assert self.comp_list is not None
        used = torch.zeros(self.n_comps, dtype=torch.bool, device=self.comp_list.device)
        used[self.comp_list.reshape(-1)] = True
        return used

    def _choose_pdfs(self, X: torch.Tensor) -> None:
        """Extended-Infomax adaptive PDF switch (Fortran ``do_choose_pdfs``).

        Re-estimates each source's kurtosis from the current model activations
        and sets its density family to the super-Gaussian (code 1) or
        sub-Gaussian (code 4) cosh density by kurtosis sign. This is the
        extended-Infomax rule that ``runamica15.m`` documents for the
        ``kurt_start``/``num_kurt``/``kurt_int`` schedule (the super/sub-Gaussian
        scores ``y +/- tanh(y)`` are exactly the two families 1/4). The
        reference binary declares this (``pdftype==1`` sets ``do_choose_pdfs``,
        amica15.f90:612) but never runs the switch (``m2sum``/``m4sum`` are
        never accumulated), so there is no bit-exact oracle; validated by
        real-data log-likelihood (must not decrease vs the fixed GG default).
        """
        n_ch, n_models = self.n_channels, self.n_models
        m2 = torch.zeros(n_ch, n_models, dtype=self.dtype, device=self.device)
        m4 = torch.zeros_like(m2)
        nsub = torch.zeros(n_models, dtype=self.dtype, device=self.device)
        n_samples = X.shape[1]
        for start in range(0, n_samples, self.block_size):
            block = X[:, start : start + self.block_size]
            logV, b_list, *_ = self._forward(block)
            v = torch.softmax(logV, dim=1)  # (batch, n_models)
            for h in range(n_models):
                b = b_list[h]  # (batch, n_ch)
                vh = v[:, h].unsqueeze(1)
                m2[:, h] += (vh * b.pow(2)).sum(0)
                m4[:, h] += (vh * b.pow(4)).sum(0)
                nsub[h] += v[:, h].sum()

        # Kurtosis = E[b^4]/E[b^2]^2 - 3 = nsub * m4 / m2^2 - 3, per (source, model).
        tiny = torch.finfo(self.dtype).tiny
        kurt = nsub.unsqueeze(0) * m4 / m2.pow(2).clamp_min(tiny) - 3.0
        self.pdtype = self._pdtype_from_kurtosis(kurt, nsub)

    def _pdtype_from_kurtosis(
        self, kurt: torch.Tensor, nsub: torch.Tensor
    ) -> torch.Tensor:
        """Map per-source excess kurtosis to a density-family code (pure).

        Super-Gaussian (positive kurtosis) -> code 1; sub-Gaussian -> code 4.
        Only sources with a meaningful signal switch: a dead model
        (``nsub[h]==0`` => ``kurt==-3.0``, finite) or a numerically blown-up
        source (``kurt`` NaN, and ``NaN>0`` is False) would otherwise be silently
        assigned code 4 with no diagnostic, so those keep their prior ``pdtype``
        and are logged -- mirroring the dead-model / non-finite guards in
        ``_update_parameters``. Split out from ``_choose_pdfs`` so the decision
        (including the sub-Gaussian branch, which real EEG rarely triggers) is
        unit-testable on a constructed ``kurt`` tensor.
        """
        assert self.pdtype is not None
        ones = torch.ones_like(self.pdtype)
        new_pdtype = torch.where(kurt > 0.0, ones, ones * 4)
        valid = torch.isfinite(kurt) & (nsub.unsqueeze(0) > 0.0)
        result = torch.where(valid, new_pdtype, self.pdtype)
        if not bool(valid.all()):
            logger.warning(
                "Non-finite or zero-mass kurtosis for %d source/model pair(s) at "
                "iter %d; kept their prior pdtype (adaptive-switch guard).",
                int((~valid).sum()),
                self.iteration,
            )
        return result

    def _snapshot_params(self) -> Dict[str, object]:
        """Snapshot the fitted state for the best-iterate safeguard (issue #51).

        Clones each ``_PARAM_TENSORS`` tensor (not an alias) so the live in-place
        M-step updates do not roll the snapshot forward; the constant
        preprocessing tensors (``mean``/``sphere``) are included so a restore is a
        total rollback. Also captures the scalar ``n_kurt_done`` (the adaptive-PDF
        switch counter that gates ``pdtype``) so a restored model's switch count
        stays consistent with its rolled-back ``pdtype`` -- otherwise a switch
        applied after the peak iterate would leave the two out of sync in a saved
        model (silent-failure review)."""
        snap: Dict[str, object] = {
            name: getattr(self, name).clone() for name in self._PARAM_TENSORS
        }
        snap["n_kurt_done"] = self.n_kurt_done
        return snap

    def _restore_params(self, snapshot: Dict[str, object]) -> None:
        """Restore the state captured by :meth:`_snapshot_params`."""
        for name, value in snapshot.items():
            setattr(self, name, value)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(
        self,
        X: np.ndarray,
        max_iter: int = 100,
        verbose: bool = True,
        mir_step: int = 0,
    ) -> "AMICATorchNG":
        """Fit the model to data.

        Parameters
        ----------
        X : np.ndarray of shape (n_channels, n_samples)
            Input data.
        max_iter : int, default=100
            Number of natural-gradient EM iterations.
        verbose : bool, default=True
            Show a tqdm progress bar.
        mir_step : int, default=0
            If > 0, compute MIR (issue #137) from the current ``W``/``sphere``
            every ``mir_step`` iterations and append it to ``mir_history_`` as
            ``(iteration, mir_nats, variance)``. ``0`` (default) disables the
            waypoints and leaves fit behaviour byte-for-byte unchanged.
            ``mir_history_`` is a true trajectory like ``ll_history``: a
            ``keep_best`` (issue #51) restore does not rewrite it, so the
            fit-end MIR is ``self.mir(X)`` on the returned parameters, not
            ``mir_history_[-1]``. Incompatible with PCA reduction
            (``pcakeep``/``pcadb``), same as :meth:`mir` itself.

        Returns
        -------
        self : AMICATorchNG
        """
        if X.ndim != 2:
            raise ValueError(
                f"X must be a 2D array (n_channels, n_samples), got shape {X.shape}"
            )
        if X.shape[0] != self.n_channels:
            raise ValueError(
                f"X has {X.shape[0]} channels, model expects {self.n_channels}"
            )
        if mir_step < 0:
            raise ValueError(f"mir_step must be >= 0, got {mir_step}")
        if mir_step > 0 and self._pca_reduced():
            raise ValueError(
                "mir_step > 0 is incompatible with PCA reduction "
                "(pcakeep/pcadb): the sphere is rank-deficient, so MIR's "
                "log-Jacobian term is undefined. Rejected up front rather "
                "than failing mid-fit at the first waypoint."
            )

        X_t = self._preprocess(X)
        n_total = X_t.shape[1]

        self._initialize_parameters()
        self.ll_history = []
        self.mir_history_ = []
        self.numrej = 0
        self.n_newton_fallbacks = 0
        self.stop_reason = "max_iter"
        self.good_idx = (
            torch.arange(n_total, device=self.device) if self.do_reject else None
        )
        numdecs = 0
        # Consecutive-small-likelihood-gain counter for the min_dll stop (Fortran
        # numincs, amica15.f90:1079-1089; issue #207). Reset here so a refit on
        # the same instance gets a fresh count, matching numdecs.
        numincs = 0

        # Best-iterate safeguard (issue #51): track the highest-LL iterate so a
        # late Newton-fallback overshoot cannot leave the returned model below a
        # peak it already reached. Inactive under do_reject (the good set, and so
        # the LL normalization, changes across iterations) and under share_comps
        # (a merge drops parameters, so pre- and post-merge LLs are not
        # comparable AND the snapshot's comp_list would revert the merge -- the
        # returned model would silently be unmerged; #60). In both cases fit()
        # returns the last iterate, matching Fortran.
        track_best = self.keep_best and not self.do_reject and not self.share_comps
        best_ll = -math.inf
        best_snapshot: Optional[Dict[str, object]] = None
        if self.keep_best and (self.do_reject or self.share_comps):
            # keep_best defaults on, so a user enabling rejection/sharing would
            # otherwise silently lose the safeguard; surface it once.
            reason = "do_reject" if self.do_reject else "share_comps"
            logger.warning(
                "keep_best is inactive under %s: best-iterate selection by LL is "
                "not well-defined (%s), so fit() returns the last iterate.",
                reason,
                "the good-sample set / LL normalization changes across iterations"
                if self.do_reject
                else "a merge changes the parameter count and reverting to an "
                "earlier snapshot would undo the merge",
            )

        iterator = tqdm(range(max_iter), desc="AMICA-NG", disable=not verbose)
        for it in iterator:
            self.iteration = it

            X_use = X_t[:, self.good_idx] if self.do_reject else X_t
            n_use = X_use.shape[1]
            acc = self._accumulate_blocks(X_use)

            # Log-likelihood of the CURRENT (pre-update) parameters: acc["ll"] is
            # this iteration's E-step total, computed before _update_parameters
            # moves the parameters. A singular W makes logdet -> -inf (not NaN),
            # so guard on isfinite, not isnan alone: a -inf LL would otherwise
            # sail past as a mere "decrease" and the run would "complete"
            # (stop_reason=max_iter) on a degenerate model. Checking here, before
            # the update, stops on the last finite parameters instead of
            # overwriting them with a garbage update first.
            ll = (acc["ll"] / (n_use * self.n_channels)).item()
            if not math.isfinite(ll):
                self.stop_reason = "nan_ll" if math.isnan(ll) else "singular_ll"
                logger.warning(
                    "Non-finite log-likelihood (%s) at iteration %d; stopping.",
                    ll,
                    it,
                )
                break

            # Best-iterate safeguard (issue #51): remember the parameters that
            # produced this LL when it is the best seen, so a later overshoot
            # does not leave the returned model below this peak.
            if track_best and ll > best_ll:
                best_ll = ll
                best_snapshot = self._snapshot_params()

            # Whether rejection fires this iteration (Fortran schedule,
            # amica17.f90:1141-1146). Fortran rejects using the per-sample
            # log-likelihood from THIS iteration's E-step, i.e. the PRE-update
            # parameters (loglik is stored in get_updates_and_likelihood before
            # update_params runs). Capture it here, before _update_parameters,
            # to match that ordering.
            will_reject = (
                self.do_reject
                and self.maxrej > 0
                and (
                    it == self.rejstart
                    or (
                        max(1, it - self.rejstart) % self.rejint == 0
                        and self.numrej < self.maxrej
                    )
                )
            )
            if will_reject:
                assert self.good_idx is not None
                reject_ll = self._sample_ll(self.good_idx, X_t)
            else:
                reject_ll = None

            self._update_parameters(acc, n_use)

            # Extended-Infomax adaptive PDF switch (Fortran do_choose_pdfs). Runs
            # on the kurt_start/num_kurt/kurt_int schedule using the just-updated
            # W; the new per-source families take effect from the next E-step.
            # itf is the Fortran-style 1-indexed iteration. num_kurt=0 disables
            # switching (the family stays at its pdftype=1 super-Gaussian init).
            if self.do_choose_pdfs and self.n_kurt_done < self.num_kurt:
                itf = it + 1
                if (
                    itf >= self.kurt_start
                    and (itf - self.kurt_start) % self.kurt_int == 0
                ):
                    self._choose_pdfs(X_use)
                    self.n_kurt_done += 1

            # Component sharing (Fortran identify_shared_comps schedule,
            # amica15.f90:1856): once per share_iter cycle from share_start,
            # merge near-collinear mixing columns across models using the
            # just-updated A. Fortran runs identify_shared_comps BEFORE
            # get_unmixing_matrices (amica15.f90:1858,1863), so rebuild W from the
            # merged comp_list -- otherwise the next E-step would read a stale W
            # (pre-merge comp_list) while indexing the densities by the merged
            # comp_list. No-op when share_comps is off or n_models == 1.
            if self.share_comps:
                itf = it + 1
                if (
                    itf >= self.share_start
                    and (itf - self.share_start) % self.share_iter == 0
                ):
                    self._identify_shared_comps()
                    self._update_unmixing_matrices()

            self.ll_history.append(ll)

            # MIR waypoint (issue #137), following the NumPy backend's
            # writestep/histstep idiom (numpy_impl/core.py). Computed from the
            # CURRENT W/sphere (just rebuilt above by _update_parameters /
            # the share_comps block) against the raw, un-preprocessed X.
            #
            # A failed waypoint must never kill the fit. `metrics.mir` raises on
            # a near-singular unmixing, and a near-singular W mid-fit is a
            # transient the natural gradient can pass through (the same
            # condition is only a warning on the training path, see
            # numpy_impl/core.py's logdet_W check). Letting that propagate would
            # let a purely diagnostic flag destroy an otherwise-recoverable
            # decomposition. Warn and record NaN instead: the gap stays visible
            # in mir_history_ rather than being silently absent, so a plotted
            # trajectory shows a hole exactly where the transient was.
            if mir_step > 0 and it % mir_step == 0:
                try:
                    mir_nats, mir_var = self.mir(X)
                except (ValueError, np.linalg.LinAlgError) as exc:
                    logger.warning(
                        "MIR waypoint failed at iter %d (%s: %s); recording NaN "
                        "and continuing. The fit itself is unaffected.",
                        it,
                        type(exc).__name__,
                        exc,
                    )
                    mir_nats = mir_var = float("nan")
                self.mir_history_.append((it, mir_nats, mir_var))

            # Learning-rate control, ported from Fortran (amica17.f90:1062-1108).
            # Natural-gradient/Newton ascent is not monotonic at a fixed rate:
            # when the log-likelihood decreases, anneal the working lrate. If
            # decreases persist for maxdecs iterations, ratchet the *ceilings*
            # down (lrate_cap; newtrate once Newton is running; and the rho rate)
            # so the per-iteration ramp can no longer re-inflate lrate back to the
            # overshooting value -- without this the ramp and a one-shot halving
            # just oscillate and the LL drifts down.
            #
            # rholrate is a CEILING here, not a per-decrease-annealed working
            # rate. Fortran resets rholrate=rholrate0 each iteration before the
            # rho update (amica15.f90:1806/1813) and only tightens the rholrate0
            # ceiling at maxdecs (amica15.f90:1068, gated on iter>newt_start), so
            # its per-decrease rholrate*=rholratefact (:1045) is always overwritten
            # by the reset and never reaches the rho update. rho has no ramp, so
            # self.rholrate carries that ceiling directly (reset to rholrate0 each
            # fit, nothing re-inflates it) and must ratchet ONLY at maxdecs. The
            # previous per-decrease self.rholrate*=rholratefact was a monotone
            # decay with no reset that collapsed the rho rate to ~1e-5 within a few
            # hundred iterations and froze rho at a stale shape (issue #193).
            # have_prev mirrors Fortran's outer ``if (iter > 1)`` (amica15.f90:1051),
            # which wraps the decrease branch AND the two stops below: none of
            # the three checks can fire on the first iteration (no LL(iter-1)
            # yet) or before ll_history has two entries after a restart.
            #
            # PRECEDENCE NOTE (PR #213 review, issue #207): the three blocks
            # below (decrease branch; min_dll; grad_norm) are independent --
            # none is gated on ``leave`` already being True from an earlier
            # block this same iteration, matching Fortran's own structure of
            # independent ``leave=.true.`` assignments with no declared
            # precedence. Whichever block runs LAST and finds its own
            # condition true wins (its ``self.stop_reason =`` is what
            # ``fit`` ultimately reports), so with this fixed source order
            # (decrease branch, then min_dll, then grad_norm) the standalone
            # grad_norm block always has final say when its condition holds.
            # In particular, under the shipped ``use_grad_norm=True`` default
            # this makes the decrease-branch's ``"grad_norm_floor"`` outcome
            # unreachable: its condition (``ndtmpsum <= min_nd`` during a
            # decrease) is strictly narrower than the standalone block's
            # (``ndtmpsum <= min_nd``, any iteration), so whenever
            # ``"grad_norm_floor"`` would fire, the standalone block fires
            # too, that same iteration, and overwrites it with
            # ``"grad_norm"``. See ``use_grad_norm``'s docstring above and
            # ``test_grad_norm_shadows_grad_norm_floor_under_shipped_defaults``.
            # This is a reporting nuance, not a behavior change -- deliberately
            # NOT restructured into an explicit precedence, to keep this
            # section a direct, reviewable port of amica15.f90:1051-1098.
            have_prev = len(self.ll_history) > 1
            leave = False
            if have_prev and ll < self.ll_history[-2]:
                # ndtmpsum is the SAME per-iteration value use_grad_norm reads
                # below (amica15.f90:1058's ``.or. (ndtmpsum .le. min_nd)``,
                # issue #207 gap 3): this is what makes lrate stopping robust
                # under do_newton, where lrate sits at newtrate/oscillates
                # instead of annealing toward minlrate, so the old
                # lrate<=minlrate-only check could never fire (the reported bug).
                if self.lrate <= self.minlrate:
                    logger.warning(
                        "lrate floor (%g) reached at iter %d; stopping.",
                        self.minlrate,
                        it,
                    )
                    self.stop_reason = "lrate_floor"
                    leave = True
                elif self._ndtmpsum is not None and self._ndtmpsum <= self.min_nd:
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
                        if self.do_newton and it > self.newt_start:
                            self.newtrate *= self.lratefact
                        numdecs = 0

            # Small-likelihood-increase stop (Fortran amica15.f90:1078-1090,
            # use_min_dll/min_dll/maxincs -- issue #207 gap 1). Independent of
            # the decrease branch above: it runs every iteration once have_prev,
            # including iterations where the LL just decreased (a decrease is
            # always "less than" a positive min_dll, so it also increments
            # numincs there, matching Fortran exactly). numincs resets to 0 on
            # any gain >= min_dll; stops only after MORE than maxincs
            # *consecutive* small gains.
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
            # use_grad_norm/min_nd -- issue #207 gap 2). Also independent of the
            # decrease branch: this is the unconditional every-iteration check
            # (as opposed to the decrease-branch's grad_norm_floor above, which
            # only applies alongside a likelihood decrease).
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

            if self.do_newton and it == self.newt_start:
                numdecs = 0

            if leave:
                break

            # Outlier rejection, after the parameter update (Fortran order,
            # amica17.f90:1141-1146) but using the pre-update per-sample LL
            # captured above.
            if will_reject:
                assert reject_ll is not None
                self._reject_outliers(reject_ll)

            iterator.set_postfix({"LL": f"{ll:.4f}", "lrate": f"{self.lrate:.4g}"})

        # Log-likelihood of the parameters fit() returns. A degenerate stop
        # leaves the model on the diverged parameters, whose LL is NOT the last
        # finite ll_history value (the guard breaks before appending), so report
        # NaN there rather than a stale healthy-looking number (silent-failure
        # review). Otherwise it is the last trajectory value, overwritten with the
        # best iterate's LL below if the safeguard restores it.
        if self.stop_reason in self._DEGENERATE_STOP_REASONS:
            self.final_ll_ = float("nan")
        else:
            self.final_ll_ = self.ll_history[-1] if self.ll_history else float("nan")

        # Restore the best iterate if the run ended materially below it (issue
        # #51). Skipped for a degenerate stop -- not because the parameters are
        # necessarily non-finite (a singular_ll stop leaves A/W finite but
        # singular) but because salvaging a diverged run here would pre-empt issue
        # #50's degenerate-fit contract; state_dict() already refuses to persist
        # any model whose stop_reason is degenerate. Also skipped when the final
        # LL is within _KEEP_BEST_TOL of the best -- a monotone single-model run
        # has final == best, so no restore fires and issue #24 parity stays
        # bit-exact.
        if (
            track_best
            and best_snapshot is not None
            and self.stop_reason not in self._DEGENERATE_STOP_REASONS
            and self.ll_history
            and best_ll - self.ll_history[-1] > _KEEP_BEST_TOL
        ):
            logger.info(
                "Restoring best iterate (LL %.6f) over final LL %.6f "
                "(issue #51 best-iterate safeguard).",
                best_ll,
                self.ll_history[-1],
            )
            self._restore_params(best_snapshot)
            self.final_ll_ = best_ll

        # LLt (Fortran's per-sample/per-model log-likelihood, issue #155):
        # computed ONCE here, strictly after the keep-best restore above, so
        # the stored arrays reflect the parameters actually being returned/
        # exported by fit() -- never a mid-training-loop value. Stored as
        # compact numpy arrays rather than retaining the full sphered dataset.
        self._llt_lht, self._llt_lt = self._compute_full_posterior_ll(X_t)

        return self

    def _sample_ll(self, good_idx: torch.Tensor, X_t: torch.Tensor) -> torch.Tensor:
        """Per-sample total log-likelihood over ``good_idx``, block by block, in
        ``good_idx`` order (so a keep-mask over the result maps back correctly)."""
        parts = [
            self._block_sample_ll(X_t[:, good_idx[start : start + self.block_size]])
            for start in range(0, int(good_idx.numel()), self.block_size)
        ]
        return torch.cat(parts)

    def _reject_outliers(self, ll_vec: torch.Tensor):
        """Permanently drop samples whose (pre-update) log-likelihood is a low
        outlier.

        Fortran ``reject_data`` (amica17.f90:2380-2464): reject any currently-good
        sample with ``loglik < mean - rejsig*std`` (population std). The rejection
        is one-directional; ``good_idx`` only ever shrinks, and the good-sample
        count drives the ``gm``/LL normalization thereafter. ``ll_vec`` is the
        per-sample log-likelihood over the current good set, in ``good_idx`` order.
        """
        assert self.good_idx is not None
        good = self.good_idx
        mean = ll_vec.mean()
        std = torch.sqrt((ll_vec.pow(2).mean() - mean.pow(2)).clamp_min(0.0))
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
            n_bad = int((~torch.isfinite(ll_vec)).sum())
            if n_bad:
                raise ValueError(
                    f"{n_bad} of {ll_vec.numel()} samples have a non-finite "
                    "log-likelihood; this indicates numerical instability "
                    "upstream (singular W / overflow), not a rejsig "
                    "miscalibration. Check for rank-deficient or "
                    "average-referenced data, or reduce the learning rate."
                )
            raise ValueError(  # defensive: unreachable for finite LL, rejsig>0
                f"Outlier rejection removed all {good.numel()} samples "
                f"(rejsig={self.rejsig} too aggressive for this data)."
            )

        self.good_idx = good[keep]
        self.numrej += 1
        n_rejected = int(good.numel() - self.good_idx.numel())
        logger.info(
            "Rejection %d at iter %d: dropped %d samples (%d good remaining).",
            self.numrej,
            self.iteration,
            n_rejected,
            int(self.good_idx.numel()),
        )

    def _check_model_idx(self, model_idx: int) -> None:
        """Validate a model index against the fitted ``n_models``.

        Raises a clear ``ValueError`` (rejecting negatives, which torch's
        negative indexing would otherwise turn into a silent wrong-model result)
        instead of an opaque tensor ``IndexError``.
        """
        if not isinstance(model_idx, (int, np.integer)):
            raise TypeError(
                f"model_idx must be an int, got {type(model_idx).__name__}."
            )
        if not (0 <= model_idx < self.n_models):
            raise ValueError(
                f"model_idx={model_idx} out of range for a {self.n_models}-model "
                f"fit (valid: 0..{self.n_models - 1})."
            )

    def transform(self, X: np.ndarray, model_idx: int = 0) -> np.ndarray:
        """Apply the learned unmixing matrix to (new) data.

        The internal ``W = inv(A)`` is stored transposed relative to the true
        unmixing (the E-step forms activations as ``(X-c)^T @ W``, see
        ``_forward``), so the unmixing applied here is ``W^T`` (issue #24
        transpose convention) with the per-model data-space center ``c``
        subtracted first (issue #27).
        """
        if self.sphere is None or self.mean is None or self.W is None or self.c is None:
            raise RuntimeError(
                "AMICATorchNG.transform() requires a fitted model; call fit() first."
            )
        self._check_model_idx(model_idx)
        X_t = torch.from_numpy(np.ascontiguousarray(X)).to(self.device, self.dtype)
        X_t = self.sphere @ (X_t - self.mean)
        # c is the per-model data-space center: unmix as W(x - c) (issue #27).
        S = self.W[:, :, model_idx].T @ (X_t - self.c[:, model_idx : model_idx + 1])
        return S.cpu().numpy()

    def get_mixing_matrix(self, model_idx: int = 0) -> np.ndarray:
        """True mixing matrix ``A_fort`` = (stored A)^T (issue #24 convention)."""
        if self.A is None or self.comp_list is None:
            raise RuntimeError(
                "AMICATorchNG.get_mixing_matrix() requires a fitted model; call "
                "fit() first."
            )
        self._check_model_idx(model_idx)
        return self.A[:, self.comp_list[:, model_idx]].T.cpu().numpy()

    @property
    def n_channels_in(self) -> int:
        """Input channel count, i.e. the width of the sphere.

        Differs from ``n_channels`` only when rank reduction shrank the model to
        the detected numerical rank (issue #223); equal to it for full-rank data
        and before :meth:`fit`. Derived rather than stored, so it cannot drift
        from the sphere it describes.
        """
        return self.n_channels if self.sphere is None else int(self.sphere.shape[1])

    def get_sensor_mixing_matrix(self, model_idx: int = 0) -> np.ndarray:
        """Mixing matrix mapped back to input-channel space.

        :meth:`get_mixing_matrix` returns ``A`` in the *sphered* space. These are
        the corresponding sensor-space maps (EEGLAB/MNE scalp maps),
        ``pinv(sphere) @ A``, of shape ``(n_channels_in, n_channels)``. This is
        the Fortran ``Spinv`` mapping (amica15.f90:568-578), and it is the only
        way to recover sensor maps when rank reduction is active, since the
        sphere is then non-square (issue #223).
        """
        if self.sphere is None:
            raise RuntimeError(
                "AMICATorchNG.get_sensor_mixing_matrix() requires a fitted "
                "model; call fit() first."
            )
        if self.A is None or self.comp_list is None:
            raise RuntimeError(
                "AMICATorchNG.get_sensor_mixing_matrix() requires a fitted "
                "model; call fit() first."
            )
        self._check_model_idx(model_idx)
        A = self.A[:, self.comp_list[:, model_idx]].T
        return (self._pinv_sphere() @ A).cpu().numpy()

    def get_unmixing_matrix(self, model_idx: int = 0) -> np.ndarray:
        """True unmixing matrix ``W_fort`` = (stored W)^T (issue #24 convention)."""
        if self.W is None:
            raise RuntimeError(
                "AMICATorchNG.get_unmixing_matrix() requires a fitted model; call "
                "fit() first."
            )
        self._check_model_idx(model_idx)
        return self.W[:, :, model_idx].T.cpu().numpy()

    def _pca_reduced(self) -> bool:
        """Whether PCA reduction is active (``pcakeep``/``pcadb``), which leaves
        the sphere rank-deficient."""
        return self.pcakeep is not None or self.pcadb is not None

    def mir(
        self, X: np.ndarray, *, model_idx: int = 0, nbins: Optional[int] = None
    ) -> Tuple[float, float]:
        """Mutual Information Reduction (issue #137) of this model's unmixing on ``X``.

        Composes the full raw-data-to-sources transform ``W_fort @ sphere`` --
        i.e. ``get_unmixing_matrix(model_idx) @ sphere`` -- and delegates to
        :func:`pamica.metrics.mir`. MIR is shift-invariant, so the data-space
        mean/``c`` centering ``transform`` applies is irrelevant here.

        Parameters
        ----------
        X : np.ndarray of shape (n_channels, n_samples)
            Raw (unpreprocessed) data.
        model_idx : int, default=0
            Which model's unmixing to use.
        nbins : int, optional
            Histogram bin count; see :func:`pamica.metrics.mir`.

        Returns
        -------
        mir_nats : float
        variance : float

        Raises
        ------
        RuntimeError
            If the model is unfitted.
        ValueError
            If PCA reduction (``pcakeep``/``pcadb``) is active: it leaves the
            sphere rank-deficient, so MIR's log-Jacobian term is undefined.
        """
        if self.A is None or self.W is None or self.sphere is None:
            raise RuntimeError(
                "AMICATorchNG.mir() requires a fitted model; call fit() first."
            )
        self._check_model_idx(model_idx)
        if self._pca_reduced():
            raise ValueError(
                "mir() is incompatible with PCA reduction (pcakeep/pcadb): "
                "the sphere is rank-deficient, so MIR's log-Jacobian term is "
                "undefined for the resulting non-square/non-invertible "
                "unmixing."
            )
        unmixing = (self.W[:, :, model_idx].T @ self.sphere).cpu().numpy()
        return mir_metric(unmixing, X, nbins)

    def pmi(
        self, X: np.ndarray, *, model_idx: int = 0, nbins: Optional[int] = None
    ) -> np.ndarray:
        """Pairwise Mutual Information (issue #137) between this model's sources on ``X``.

        Delegates to :func:`pamica.metrics.pairwise_mi` on
        ``transform(X, model_idx)``.

        Parameters
        ----------
        X : np.ndarray of shape (n_channels, n_samples)
            Raw (unpreprocessed) data.
        model_idx : int, default=0
            Which model's sources to use.
        nbins : int, optional
            Histogram bin count; see :func:`pamica.metrics.pairwise_mi`.

        Returns
        -------
        mi_matrix : np.ndarray of shape (n_sources, n_sources)

        Raises
        ------
        RuntimeError
            If the model is unfitted (via ``transform``).
        """
        return pairwise_mi(self.transform(X, model_idx=model_idx), nbins)

    # ------------------------------------------------------------------
    # Multi-model posterior (issue #141)
    # ------------------------------------------------------------------
    def model_loglik(self, X: np.ndarray) -> np.ndarray:
        """Per-model, per-sample log-likelihood ``Lht`` on (new) data.

        For each model ``h`` and sample ``t`` this is the joint log-likelihood
        ``log(gm[h]) + log|det W_h| + sldet + sum_i log p_h(s_i)`` (Fortran's
        ``Lht``/``modloglik``), evaluated on arbitrary raw data via the STORED
        sphere/mean -- never re-preprocessing, which would overwrite them. The
        per-sample posterior over models (model dominance) is
        ``softmax(Lht, axis=0)``; see :meth:`model_probability`.

        This does not replicate a training-time ``do_reject`` mask: it scores
        every sample of ``X``. On a ``do_reject`` fit's own training data it
        therefore returns real values where the stored ``_llt_lht`` carries
        Fortran's sentinel zeros for rejected samples (issue #155), so the two
        agree bit-for-bit only when the fit did not use ``do_reject``. Like
        :meth:`transform`, it assumes a usable (non-degenerate) fit; the
        :class:`~pamica.AMICA` wrapper enforces that via ``_check_usable``.

        Parameters
        ----------
        X : np.ndarray of shape (n_channels, n_samples)
            Raw (unpreprocessed) data.

        Returns
        -------
        Lht : np.ndarray of shape (n_models, n_samples)

        Raises
        ------
        RuntimeError
            If the model is unfitted.
        ValueError
            If ``X`` contains non-finite (NaN/Inf) values.
        """
        if self.sphere is None or self.mean is None or self.W is None:
            raise RuntimeError(
                "AMICATorchNG.model_loglik() requires a fitted model; call fit() first."
            )
        X = np.ascontiguousarray(X)
        if not np.isfinite(X).all():
            bad = np.flatnonzero(~np.isfinite(X).all(axis=1))
            raise ValueError(
                "AMICATorchNG.model_loglik(): input contains non-finite (NaN/Inf) "
                f"values in {bad.size} channel(s) {bad.tolist()}; clean bad "
                "segments before scoring."
            )
        X_t = torch.from_numpy(X).to(self.device, self.dtype)
        X_t = self.sphere @ (X_t - self.mean)
        n_samples = X_t.shape[1]
        Lht = np.zeros((self.n_models, n_samples))
        for start in range(0, n_samples, self.block_size):
            end = min(start + self.block_size, n_samples)
            logV, *_ = self._forward(X_t[:, start:end])
            Lht[:, start:end] = logV.T.detach().cpu().numpy()
        return Lht

    def model_probability(self, X: np.ndarray) -> np.ndarray:
        """Per-sample posterior probability of each model (model dominance).

        The column-wise ``softmax`` over models of :meth:`model_loglik`, i.e.
        ``P(model h | x_t)``; each column sums to 1. For a single model this is
        all ones.

        Parameters
        ----------
        X : np.ndarray of shape (n_channels, n_samples)
            Raw (unpreprocessed) data.

        Returns
        -------
        prob : np.ndarray of shape (n_models, n_samples)

        Raises
        ------
        RuntimeError
            If the model is unfitted.
        ValueError
            If ``X`` is non-finite, or if every model underflows to ``-inf``
            log-likelihood at some sample (the posterior is undefined there).
        """
        Lht = self.model_loglik(X)
        col_max = Lht.max(axis=0, keepdims=True)
        if not np.isfinite(col_max).all():
            n_bad = int((~np.isfinite(col_max)).sum())
            raise ValueError(
                f"AMICATorchNG.model_probability(): every model has -inf "
                f"log-likelihood at {n_bad} sample(s), so the posterior is "
                "undefined there (an extreme outlier under a tight source "
                "density)."
            )
        ex = np.exp(Lht - col_max)
        return ex / ex.sum(axis=0, keepdims=True)

    # ------------------------------------------------------------------
    # Fitted-parameter metadata (issue #142)
    # ------------------------------------------------------------------
    def get_pdftype(self, model_idx: int = 0) -> np.ndarray:
        """Per-source density-family code for model ``model_idx``.

        One integer per source component (0-4; see
        :data:`pamica.torch_impl.PDFTYPE_NAMES`): 0 generalized Gaussian, 1
        super-Gaussian cosh, 2 Gaussian, 3 logistic, 4 sub-Gaussian cosh. All
        sources share ``pdftype`` unless the adaptive switcher (``pdftype=1``)
        moved them individually (issue #26).

        Returns
        -------
        np.ndarray of int, shape (n_sources,)
        """
        if self.pdtype is None:
            raise RuntimeError(
                "AMICATorchNG.get_pdftype() requires a fitted model; call fit() first."
            )
        self._check_model_idx(model_idx)
        return self.pdtype[:, model_idx].detach().cpu().numpy()

    def get_rho(self, model_idx: int = 0) -> np.ndarray:
        """Generalized-Gaussian shape parameter ``rho`` for model ``model_idx``.

        One value per (mixture component, source): ``rho == 2`` is Gaussian-
        shaped, ``rho == 1`` Laplacian, ``rho < 1`` heavier-tailed. Only the
        generalized-Gaussian family (``pdftype=0``) updates ``rho``; for every
        non-zero code (1-4, the fixed and adaptive cosh families) it stays frozen
        at ``rho0`` and does not describe the fitted density.

        Returns
        -------
        np.ndarray of float, shape (n_mix, n_sources)
        """
        if self.rho is None or self.comp_list is None:
            raise RuntimeError(
                "AMICATorchNG.get_rho() requires a fitted model; call fit() first."
            )
        self._check_model_idx(model_idx)
        # Defense-in-depth, matching state_dict()'s isfinite sweep: a degenerate
        # multi-model fit can leave one model's rho non-finite without the
        # aggregate LL tripping nan_ll, and _check_usable only inspects
        # stop_reason. Refuse rather than return a silent NaN.
        if not torch.isfinite(self.rho).all():
            raise RuntimeError(
                "AMICATorchNG.get_rho(): rho holds non-finite values (a "
                "degenerate fit); inspect stop_reason and refit."
            )
        idx = self.comp_list[:, model_idx]
        return self.rho[:, idx].detach().cpu().numpy()

    def shared_components(self) -> list:
        """Components shared across models by ``share_comps`` (issue #60).

        ``share_comps`` folds near-collinear components of different models onto
        one shared mixing column + density, recorded as a repeated index in
        ``comp_list``. Returns one group per shared column: a list of
        ``(model_idx, source_idx)`` pairs that all reference it. Empty when no
        component is shared across two or more models (always for one model, and
        for a default multi-model fit with ``share_comps`` off).

        Note that a merge synchronizes only the mixture parameters routed through
        ``comp_list`` (``mu``/``alpha``/``beta``/``rho``); the per-source density
        *family* code ``pdtype`` is a separate tensor and is not synchronized, so
        under the adaptive switcher (``pdftype=1``) a shared pair can still report
        different :meth:`get_pdftype` codes.

        Returns
        -------
        list of list of tuple(int, int)
        """
        if self.comp_list is None:
            raise RuntimeError(
                "AMICATorchNG.shared_components() requires a fitted model; call "
                "fit() first."
            )
        cl = self.comp_list.detach().cpu().numpy()  # (n_sources, n_models)
        groups = []
        for col in np.unique(cl):
            src, mdl = np.where(cl == col)
            if np.unique(mdl).size >= 2:
                groups.append([(int(h), int(i)) for i, h in zip(src, mdl)])
        return groups

    # ------------------------------------------------------------------
    # EEGLAB drop-in output (issue #92)
    # ------------------------------------------------------------------
    def variance_order(
        self, model_idx: int = 0, return_svar: bool = False
    ) -> Union[np.ndarray, tuple]:
        """EEGLAB back-projected-variance component order (IC1 = highest variance).

        Returns the source indices sorted by descending back-projected variance,
        the ordering EEGLAB's ``loadmodout15.m`` applies on load (so ``order[0]``
        is IC1). The de-sphered sensor-space mixing column ``a_i = pinv(W S)[:, i]``
        contributes ``||a_i||^2 * sum_k alpha_ki (mu_ki^2 + r_ki / sbeta_ki^2)``
        with ``r_ki = gamma(3/rho_ki)/gamma(1/rho_ki)`` (the source's mixture
        variance), matching ``loadmodout15`` exactly. Non-mutating: the stored
        parameters keep their fit order; this only reports the display order.

        Parameters
        ----------
        model_idx : int, default=0
            Which model's components to order.
        return_svar : bool, default=False
            If True, also return the per-source variance sorted to ``order``.

        Returns
        -------
        order : np.ndarray of int, shape (n_sources,)
            Source indices, highest back-projected variance first.
        svar : np.ndarray, optional
            Present only when ``return_svar``; the sorted variances.
        """
        from scipy.special import gamma

        if (
            self.comp_list is None
            or self.alpha is None
            or self.mu is None
            or self.beta is None
            or self.rho is None
            or self.W is None
            or self.sphere is None
        ):
            raise RuntimeError(
                "AMICATorchNG.variance_order() requires a fitted model; call "
                "fit() first."
            )
        self._check_model_idx(model_idx)
        cl = self.comp_list[:, model_idx].cpu().numpy()
        alpha = self.alpha[:, cl].cpu().numpy()
        mu = self.mu[:, cl].cpu().numpy()
        sbeta = self.beta[:, cl].cpu().numpy()
        rho = self.rho[:, cl].cpu().numpy()
        # source mixture variance (sum over the mixture components); unused
        # mixtures carry alpha == 0 and drop out, matching loadmodout15.
        ratio = gamma(3.0 / rho) / gamma(1.0 / rho)
        mix_var = (alpha * (mu**2 + ratio / sbeta**2)).sum(axis=0)
        # de-sphered sensor-space mixing: A = pinv(W_fort @ S), columns = maps.
        w_fort = self.W[:, :, model_idx].T.cpu().numpy()
        sphere = self.sphere.cpu().numpy()
        a_sensor = np.linalg.pinv(w_fort @ sphere)
        svar = mix_var * (a_sensor**2).sum(axis=0)
        order = np.argsort(-svar)
        if return_svar:
            return order, svar[order]
        return order

    def write_amica_output(self, outdir) -> None:
        """Write this fitted model as the Fortran/EEGLAB AMICA output directory.

        Produces the raw binary files that EEGLAB's ``loadmodout15.m`` (and the
        Python port :func:`pamica.numpy_impl.load.loadmodout`) read: ``gm``,
        ``W``, ``S``, ``mean``, ``c``, ``alpha``, ``mu``, ``sbeta``, ``rho``,
        ``comp_list``, ``LL``, so a PyTorch NG fit drops directly into an
        EEGLAB workflow (issue #92). ``loadmodout15`` performs the
        variance-ordering and unit-norm normalization on load, so the on-disk
        parameters are written in fit order. Single-model output is
        byte-compatible with the Fortran reference.

        Also writes ``LLt`` (the per-sample/per-model log-likelihood, issue
        #155) for a model that was just ``fit()`` in this process. A model
        restored via :meth:`from_state_dict` has no training data to recompute
        it from, so ``LLt`` is omitted for it (a warning is logged) -- the
        rest of the output is unaffected.

        Parameters
        ----------
        outdir : str or path-like
            Destination directory (created if absent).
        """
        if self.A is None:
            raise RuntimeError(
                "write_amica_output requires a fitted model; call fit() first."
            )

        from ..numpy_impl.load import write_amicaout

        def _np(t):
            return t.detach().cpu().numpy()

        # The exported parameters are the fit()-kept iterate (LL == final_ll_).
        # Under the keep_best safeguard (#51) that can be an earlier iterate than
        # the last, so end the written LL trajectory at that iterate rather than
        # at a later, discarded overshoot -- otherwise LL[-1] would not match the
        # model just written. Monotone runs keep the full trajectory unchanged.
        ll = np.asarray(self.ll_history, dtype=np.float64)
        if (
            self.final_ll_ is not None
            and np.isfinite(self.final_ll_)
            and ll.size
            and not np.isclose(ll[-1], self.final_ll_)
        ):
            ll = ll[: int(np.argmax(ll)) + 1]

        # LLt (Fortran's per-sample/per-model log-likelihood, issue #155):
        # computed once at the end of fit() (after any keep-best restore) and
        # stored compactly on self. A model restored via from_state_dict()
        # never ran fit() in this process, so it has neither -- warn rather
        # than silently omitting the file (silent-failure review).
        if self._llt_lht is not None and self._llt_lt is not None:
            Lht, Lt = self._llt_lht, self._llt_lt
        else:
            logger.warning(
                "No LLt data available (model was restored via "
                "from_state_dict(), not freshly fit()); writing output "
                "without the LLt file."
            )
            Lht = Lt = None

        write_amicaout(
            outdir,
            gm=_np(self.gm),
            W=_np(self.W),
            sphere=_np(self.sphere),
            mean=_np(self.mean),
            c=_np(self.c),
            alpha=_np(self.alpha),
            mu=_np(self.mu),
            sbeta=_np(self.beta),  # Fortran's 'sbeta' is pamica's beta (scale)
            rho=_np(self.rho),
            comp_list=_np(self.comp_list),
            ll=ll,
            A=_np(self.A),
            Lht=Lht,
            Lt=Lt,
        )

    # ------------------------------------------------------------------
    # Persistence (issue #36)
    # ------------------------------------------------------------------
    # Full fitted-parameter snapshot. A/W/c/comp_list/mean/sphere are what
    # transform()/get_*matrix() read back; mu/alpha/beta/rho/gm are the
    # mixture-PDF EM state, included for a complete snapshot (and for parity/
    # continued-analysis) even though no public method currently reads them.
    # pdtype is the per-source density-family code (issue #26): a non-default
    # pdftype model, or the adaptive switcher's chosen 1/4 assignments, would
    # otherwise silently revert to GG on reload. comp_list and pdtype are integer
    # tensors (dtype preserved on load); the rest follow self.dtype.
    _PARAM_TENSORS = (
        "A", "W", "c", "mu", "alpha", "beta", "rho", "gm",
        "comp_list", "mean", "sphere", "pdtype",
    )  # fmt: skip
    # Integer tensors in _PARAM_TENSORS: keep their dtype on load, only move device.
    _INT_PARAM_TENSORS = ("comp_list", "pdtype")

    # Stop reasons that mark a fit as degenerate (non-finite log-likelihood).
    # Such a model yields NaN sources, so state_dict() refuses to persist it
    # rather than let it round-trip silently (silent-failure review, PR #44).
    _DEGENERATE_STOP_REASONS = ("nan_ll", "singular_ll")

    def state_dict(self) -> dict:
        """Serialize the fitted model to a plain, device-agnostic dict.

        The returned dict has three parts: ``config`` (the constructor
        arguments needed to rebuild the object), ``params`` (the fitted
        tensors, moved to CPU), and ``extra`` (scalar/schedule state, plus the
        optional ``good_idx`` index tensor). Every value is a tensor or a plain
        Python primitive, so the dict round-trips through
        ``torch.save``/``torch.load`` with ``weights_only=True`` (no custom
        classes or ``torch.dtype`` objects: dtype is stored by name). Rebuild
        with :meth:`from_state_dict`.

        Raises if the model is unfitted or degenerate (a fit that ended on a
        non-finite log-likelihood): a NaN model must not be persisted silently.
        """
        if self.A is None:
            raise RuntimeError(
                "AMICATorchNG.state_dict() requires a fitted model; call fit() first."
            )
        if self.stop_reason in self._DEGENERATE_STOP_REASONS:
            raise RuntimeError(
                f"Refusing to serialize a degenerate model (stop_reason="
                f"{self.stop_reason!r}): fit() hit a non-finite log-likelihood at "
                f"iteration {self.iteration}. Fix the instability (lower lrate, "
                f"disable Newton, or check data conditioning) before saving."
            )
        # Defense-in-depth: catch a non-finite parameter even if stop_reason
        # bookkeeping ever misses it (the codebase has known NaN-suppression
        # risks). isfinite on the integer comp_list is trivially all-True.
        nonfinite = [
            name
            for name in self._PARAM_TENSORS
            if not torch.isfinite(getattr(self, name)).all()
        ]
        if nonfinite:
            raise RuntimeError(
                f"Refusing to serialize a model with non-finite parameters "
                f"{nonfinite} (stop_reason={self.stop_reason!r})."
            )
        config = {
            "n_channels": self.n_channels,
            "n_models": self.n_models,
            "n_mix": self.n_mix,
            "block_size": self.block_size,
            # lrate/newtrate/rholrate are annealed during fit; persist the
            # original constructor values (lrate0/newtrate0/rholrate0) and
            # restore the mutated ones from ``extra`` below.
            "lrate": self.lrate0,
            "minlrate": self.minlrate,
            "lratefact": self.lratefact,
            "maxdecs": self.maxdecs,
            # Convergence stops (issue #207); fixed hyperparameters, not
            # annealed during fit, so no mutated counterpart in ``extra``
            # (unlike lrate/newtrate/rholrate) is needed.
            "use_min_dll": self.use_min_dll,
            "min_dll": self.min_dll,
            "maxincs": self.maxincs,
            "use_grad_norm": self.use_grad_norm,
            "min_nd": self.min_nd,
            "newt_ramp": self.newt_ramp,
            "do_newton": self.do_newton,
            "newt_start": self.newt_start,
            "newtrate": self.newtrate0,
            "do_reject": self.do_reject,
            "rejsig": self.rejsig,
            "rejstart": self.rejstart,
            "rejint": self.rejint,
            "maxrej": self.maxrej,
            "rho0": self.rho0,
            "minrho": self.minrho,
            "maxrho": self.maxrho,
            "rholrate": self.rholrate0,
            "rholratefact": self.rholratefact,
            # Best-iterate safeguard flag (issue #51); only affects a re-fit, but
            # persisted so a reloaded model reconstructs its exact configuration.
            "keep_best": self.keep_best,
            # Density-family selection (issue #26): needed so a reloaded model
            # rebuilds with the right pdftype/dorho/do_choose_pdfs and switch
            # schedule instead of the GG default.
            "pdftype": self.pdftype,
            "kurt_start": self.kurt_start,
            "num_kurt": self.num_kurt,
            "kurt_int": self.kurt_int,
            "invsigmin": self.invsigmin,
            "invsigmax": self.invsigmax,
            "doscaling": self.doscaling,
            "scalestep": self.scalestep,
            # Component sharing (issue #60): persisted so a reloaded multi-model
            # run keeps its schedule; the merged comp_list itself is in params.
            "share_comps": self.share_comps,
            "share_start": self.share_start,
            "share_iter": self.share_iter,
            "comp_thresh": self.comp_thresh,
            "do_mean": self.do_mean,
            "do_sphere": self.do_sphere,
            "do_approx_sphere": self.do_approx_sphere,
            "pcakeep": self.pcakeep,
            "pcadb": self.pcadb,
            "mineig": self.mineig,
            "mineig_rel": self.mineig_rel,
            "seed": self.seed,
            # Store dtype by name (e.g. "float64") to keep the payload
            # weights_only-safe; rebuilt via getattr(torch, ...) on load.
            "dtype": str(self.dtype).split(".")[-1],
        }
        # .clone() forces an independent copy even when self.device is already
        # CPU (where .cpu() would alias): fit() mutates A/mu/beta in place each
        # iteration, so an aliased snapshot would silently roll forward if
        # state_dict() were ever called mid-fit (e.g. best-so-far checkpointing).
        params = {
            name: getattr(self, name).detach().cpu().clone()
            for name in self._PARAM_TENSORS
        }
        extra = {
            "sldet": float(self.sldet),
            "iteration": int(self.iteration),
            "ll_history": [float(v) for v in self.ll_history],
            "final_ll": None if self.final_ll_ is None else float(self.final_ll_),
            "stop_reason": self.stop_reason,
            "n_newton_fallbacks": int(self.n_newton_fallbacks),
            "n_kurt_done": int(self.n_kurt_done),
            "numrej": int(self.numrej),
            "good_idx": None
            if self.good_idx is None
            else self.good_idx.detach().cpu().clone(),
            "lrate": float(self.lrate),
            "lrate_cap": float(self.lrate_cap),
            "newtrate": float(self.newtrate),
            "rholrate": float(self.rholrate),
        }
        return {
            "format_version": 3,
            "config": config,
            "params": params,
            "extra": extra,
        }

    @classmethod
    def from_state_dict(
        cls, state: dict, device: Optional[Union[str, torch.device]] = None
    ) -> "AMICATorchNG":
        """Rebuild a fitted :class:`AMICATorchNG` from :meth:`state_dict` output.

        ``device`` overrides where the restored tensors live (the constructor
        picks a default when ``None``); ``dtype`` always comes from the saved
        ``config``.
        """
        # format_version stays 3 here -- deliberately NOT bumped for issue
        # #207, unlike PR #52's 1->2 (adaptive PDF) and PR #53's 2->3
        # (keep_best). The check below is strict equality, so bumping would
        # break loading genuinely older (pre-#53) files for no reason: the
        # five new config keys (use_min_dll/min_dll/maxincs/use_grad_norm/
        # min_nd) are additive-only, and a payload saved before #207 simply
        # lacks them in its ``config`` dict, so ``cls(device=device,
        # **config)`` below falls back to the constructor's own
        # Fortran-faithful defaults for whichever keys are missing -- see
        # test_missing_convergence_keys_fall_back_to_fortran_defaults in
        # test_ng_convergence.py.
        version = state.get("format_version")
        if version != 3:
            raise ValueError(
                f"unsupported AMICATorchNG state format_version: {version!r} "
                "(expected 3)"
            )
        for section in ("config", "params", "extra"):
            if section not in state:
                raise ValueError(
                    f"malformed AMICATorchNG state: missing {section!r} section "
                    f"(format_version={version}); the payload may be truncated."
                )
        config = dict(state["config"])
        config["dtype"] = getattr(torch, config["dtype"])
        obj = cls(device=device, **config)
        obj._load_params(state)
        return obj

    def _load_params(self, state: dict) -> None:
        """Restore fitted tensors/scalars from :meth:`state_dict` output onto
        this instance's device/dtype."""
        params = state["params"]
        missing = [name for name in self._PARAM_TENSORS if name not in params]
        if missing:
            raise ValueError(f"malformed AMICATorchNG state: missing params {missing}")
        # Guard against config/params drift: A and comp_list must match the
        # dimensions the constructor just derived, or transform()/the E-step
        # would fail later with a confusing matmul error far from load().
        if tuple(params["A"].shape) != (self.n_channels, self.n_comps):
            raise ValueError(
                f"restored A has shape {tuple(params['A'].shape)}, expected "
                f"{(self.n_channels, self.n_comps)} for n_channels="
                f"{self.n_channels}, n_models={self.n_models}"
            )
        if tuple(params["comp_list"].shape) != (self.n_channels, self.n_models):
            raise ValueError(
                f"restored comp_list has shape {tuple(params['comp_list'].shape)}, "
                f"expected {(self.n_channels, self.n_models)}"
            )
        for name in self._PARAM_TENSORS:
            tensor = params[name]
            # comp_list/pdtype hold integer indices/codes; preserve their dtype
            # and only move devices. The float parameters follow self.dtype.
            if name in self._INT_PARAM_TENSORS:
                setattr(self, name, tensor.to(self.device))
            else:
                setattr(self, name, tensor.to(self.device, self.dtype))
        # sphere was just replaced, so any cached back-map describes the old one.
        self._sphere_pinv = None

        extra = state["extra"]
        self.sldet = extra["sldet"]
        self.iteration = extra["iteration"]
        self.ll_history = list(extra["ll_history"])
        self.final_ll_ = extra["final_ll"]
        self.stop_reason = extra["stop_reason"]
        self.n_newton_fallbacks = extra["n_newton_fallbacks"]
        self.n_kurt_done = extra["n_kurt_done"]
        self.numrej = extra["numrej"]
        good_idx = extra["good_idx"]
        self.good_idx = None if good_idx is None else good_idx.to(self.device)
        self.lrate = extra["lrate"]
        self.lrate_cap = extra["lrate_cap"]
        self.newtrate = extra["newtrate"]
        self.rholrate = extra["rholrate"]
