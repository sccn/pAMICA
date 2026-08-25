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

Scope: single- and multi-model (``n_models >= 1``, issue #81), all five
``amica15.f90`` source-density families (``pdftype`` 0/1/2/3/4, issue #265,
porting the PyTorch backend's issue #26), natural gradient and the Newton
preconditioner (``do_newton``, issue #264), component sharing
(``share_comps``, issue #263). Source extraction (``transform`` and the
``get_mixing_matrix``/``get_unmixing_matrix``/``get_sensor_mixing_matrix``/
``get_rho`` accessors) and persistence (``state_dict``/``from_state_dict`` and
``.npz`` ``save``/``load``) are implemented (epic #278 Phase 1, issue #287).
Outlier rejection and ``keep_best`` are simply absent (no such
parameter/method) -- epic #278 Phases 2 and 3.

Newton (issue #264) runs entirely in float32 on the GPU stream: the curvature
accumulators ride the existing E-step locals, and the direction is Fortran's
closed-form per-source-pair 2x2 solve (no linear algebra, so no CPU-stream
handoff). The one host sync it adds is the ``posdef`` flag, a scalar boolean
that has to reach Python because it selects the learning-rate ramp target and
drives the fallback counter. It is read once per MODEL per Newton iteration
(``_newton_direction`` is called inside the per-model loop), so a single-model
fit adds one scalar sync per iteration and an ``n_models``-model fit adds
``n_models`` -- unlike the M-step's existing dead-model and rho-NaN canaries,
which are one array-wide reduction each. Validated against a float64 PyTorch
twin on real EEG; see ``.context/issue-264/newton_findings.md``.

The non-GG source-density families (issue #265) are a fast E-step dispatch, not
a separate code path: ``_score``/``_log_pdf`` take an optional per-source
``pdtype`` array and select among the fixed Fortran families (2 Gaussian, 3
logistic, 4 sub-Gaussian cosh+, 1 super-Gaussian cosh-) via nested ``mx.where``;
``pdtype is None`` (the ``pdftype=0`` default) skips that chain entirely and
runs the pre-#265 GG-only body unchanged, so default fits stay bit-identical.
The GG shape parameter ``rho`` is frozen for every non-GG family
(``self.dorho = pdftype == 0``, Fortran ``dorho=.false.``), which also gates
the ``drho_n`` accumulation and the per-iteration lgamma-table refresh here.
AMICATorchNG already gates its digamma pull behind the same ``self.dorho``
flag (core.py:1483-1489), so that is not a divergence; its genuine dead work
for a non-GG fit is the ``drho_n`` accumulation, which it computes
unconditionally in ``_get_block_updates`` (no ``dorho`` gate there), and the
inline ``torch.lgamma(1+1/rho)`` term ``_log_pdf_only`` recomputes on every
call to build the (dead, for non-GG) GG-fallthrough branch. This backend
skips both -- ``drho_n`` via the gate above, and the lgamma term by reusing a
cached ``_lgamma_table`` refreshed only when ``dorho`` -- so this is a
deliberate MLX-only WORK divergence, never a numeric one. ``pdftype=1``
enables the extended-Infomax adaptive switcher (``_choose_pdfs``); it
accumulates its kurtosis moments in numpy float64 on the host (a knife-edge
sign decision, done off the lossy float32 GPU graph) and has no bit-exact
oracle -- Fortran declares the switch but never runs it (``m2sum``/``m4sum``
allocated, never accumulated) -- so it is behavior-validated on real EEG,
exactly as in the PyTorch backend (ADR 0002).

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

import json
import logging
import math
import time
from typing import List, Optional, Sequence

# mlx ships as a compiled extension with no type stubs, so ty cannot resolve
# it statically even when installed; scope the suppression to this one import.
import mlx.core as mx  # ty: ignore[unresolved-import]
import numpy as np
from scipy.special import digamma, gammaln

from .. import blocktune
from .. import restarts
from ..numpy_impl.utils import identify_shared_components
from ..rank import MINEIG, MINEIG_REL, numerical_rank

logger = logging.getLogger(__name__)

# Human-readable names for the ``pdftype``/``pdtype`` source-density family
# codes (issue #265, mirroring AMICATorchNG's PDFTYPE_NAMES, torch core.py:
# 65-71 -- duplicated rather than imported so this module keeps no torch
# dependency). Exposed alongside the numeric codes so a fitted model's
# per-source density family is inspectable (issue #142).
PDFTYPE_NAMES = {
    0: "generalized_gaussian",
    1: "super_gaussian_cosh",
    2: "gaussian",
    3: "logistic",
    4: "sub_gaussian_cosh",
}

_LOG2 = math.log(2.0)
_LOG4 = math.log(4.0)  # logistic-family normalizer (amica15.f90:1346)
# Log-normalizers for the non-GG density families, using Fortran's exact literal
# constants (amica15.f90:1333/1359/1371) so the log-density matches the reference
# binary bit-for-bit: 2.506628274 = sqrt(2*pi) (Gaussian, pdtype 2); 4.132731354 /
# 1.858073988 = the sub-/super-Gaussian cosh normalizers (pdtype 4 / 1). Ported
# verbatim from AMICATorchNG (core.py:76-82, policy 1).
_LOG_SQRT_2PI = math.log(2.506628274)
_LOG_NORM_COSH_SUB = math.log(4.132731354)
_LOG_NORM_COSH_SUP = math.log(1.858073988)
# Fortran epsdble: zero the rho*ln|y| term when |y|^rho underflows below this
# (amica17.f90:1570), matching AMICATorchNG.
_EPSDBLE = 1e-16
# MLX linalg runs on the CPU stream only (float32-accurate); the GPU stream
# raises "not yet supported on the GPU" for inv/slogdet/eigh/solve.
_CPU = mx.cpu
# Guard threshold for _update_unmixing_matrices (issue #274): MLX 0.32's
# CPU-stream mx.linalg.inv aborts the whole process (uncatchable LAPACK LU
# failure) instead of raising on a singular matrix, so a per-model condition
# check must reject before that call.
#
# This is NOT set from float32's ~1/eps precision-loss point (~8-17e6,
# depending on the "machine epsilon" convention): that theoretical value
# turns out to be far too low in practice. Measured by isolated-subprocess
# bisection (a real LU failure aborts the process, so each trial has to be
# disposable): whether a given near-singular float32 matrix makes MLX's LU
# actually abort is NOT a clean function of condition number alone -- it
# depends on the specific numerical cancellation pattern during pivoting.
# Across 5 random near-duplicate-column 32x32 matrices, the abort onset
# ranged from cond~9e8 up to beyond cond~5e10 with no abort at all short of
# an exact duplicate; conversely, matrices built by scaling one column toward
# (but not to) zero never aborted even past cond~1e16, returning a huge but
# finite inverse instead. Separately, the existing (legitimate, unrelated to
# this guard) test suite already exercises cond up to ~4.4e9 without ever
# hitting the abort: pamica/tests/mlx_tests/test_mlx_newton.py::
# test_fallback_ramps_toward_lrate_cap_and_counts repeatedly steps A from the
# same deliberately under-determined 256-sample block, so its conditioning
# compounds across 30 iterations. A threshold near the textbook ~1e7 value
# would raise on that legitimate scenario (confirmed: it did, before this
# constant was recalibrated).
#
# So the threshold is set empirically instead: comfortably above every cond
# observed anywhere in the full MLX test suite (~4.4e9, ~225x margin), and
# comfortably below where a genuinely singular A -- the issue's literal
# example, a duplicated component column -- actually lands (~1e15-1e17,
# effectively double-precision-computed infinity). It cannot guarantee
# catching every conceivable near-singular matrix (no scalar cond threshold
# can, given the above), but it reliably catches the realistic failure mode:
# a collapsed or duplicated component, not a merely ill-conditioned one.
#
# A complete mechanism -- running mx.linalg.inv itself in a disposable
# isolated subprocess, so a real LU abort kills only that subprocess -- was
# considered and rejected: it would trade an occasional missed abort for a
# per-iteration subprocess spawn on the hot path, which is a far larger and
# less predictable cost than one host-side np.linalg.cond call, for a defect
# this guard already makes rare in practice.
_INV_COND_THRESHOLD = 1e12


def _logcosh(x: mx.array) -> mx.array:
    """Numerically stable ``log cosh(x) = |x| - log2 + log1p(exp(-2|x|))``
    (AMICATorchNG ``_logcosh``, core.py:113-116). Naive ``mx.log(mx.cosh(x))``
    overflows to inf in float32 by ``|x| == 90`` (measured crossover: finite at
    89.0, inf at 89.5; ``cosh`` itself overflows first) -- reachable, since
    ``beta`` clips at ``invsigmax=1000`` -- while this form stays within float32
    precision (~1e-4 absolute; measured 2.9e-5 at ``x=1000``) out to at least
    1e3 (policy 3; no ``mlx.nn`` import)."""
    ax = mx.abs(x)
    return ax - _LOG2 + mx.log1p(mx.exp(-2.0 * ax))


def _score(y: mx.array, rho: mx.array, pdtype: Optional[mx.array] = None) -> mx.array:
    """Source-density score ``fp = -d(log pdf)/dy`` (AMICATorchNG ``_score``,
    core.py:184-222).

    ``pdtype is None`` (the ``pdftype=0`` fast path): GG score only -- exactly
    the pre-#265 ``_score_gg`` body, so it adds ZERO extra graph nodes relative
    to before this phase (policy 2). ``fp(0)=0`` for ``rho>=1``, which the
    ``ufp/y`` guard in ``_get_block_updates`` relies on.

    Otherwise selects per source among the fixed families (Fortran ``fp``
    select, amica15.f90:1467-1491): 2 Gaussian ``y``; 3 logistic
    ``tanh(y/2)``; 4 sub-Gaussian ``y - tanh(y)``; 1 super-Gaussian
    ``y + tanh(y)``; any other code falls through to the GG form (structurally
    dead here -- a non-None ``pdtype`` is always uniformly 1/2/3/4, never mixed
    with 0 -- kept only so this mirrors AMICATorchNG's nesting exactly).
    """
    abs_y = mx.abs(y)
    sign_y = mx.sign(y)
    fp_gg = rho * sign_y * mx.power(abs_y, rho - 1.0)
    # rho is generically in (1, 2); keep the exact Laplace/Gaussian endpoints.
    fp_gg = mx.where(rho == 2.0, 2.0 * y, mx.where(rho == 1.0, sign_y, fp_gg))
    if pdtype is None:
        return fp_gg

    tanh_half = mx.tanh(0.5 * y)
    tanh_y = mx.tanh(y)
    return mx.where(
        pdtype == 2,
        y,
        mx.where(
            pdtype == 3,
            tanh_half,
            mx.where(pdtype == 4, y - tanh_y, mx.where(pdtype == 1, y + tanh_y, fp_gg)),
        ),
    )


def _log_pdf(
    y: mx.array,
    rho: mx.array,
    lgamma_table: mx.array,
    pdtype: Optional[mx.array] = None,
) -> tuple[mx.array, Optional[mx.array]]:
    """GG log-density and ``|y|^rho``, extended with the fixed non-GG families
    (AMICATorchNG ``_log_pdf_only``, core.py:225-267).

    ``pdtype is None`` (the ``pdftype=0`` fast path): byte-for-byte the
    pre-#265 ``_log_pdf_gg`` body -- ``lgamma_table = lgamma(1+1/rho)``
    (precomputed host-side; MLX has no ``lgamma``) makes the uniform GG form
    reduce to the exact Laplace (rho=1) and Gaussian (rho=2) log-densities, and
    ``az_rho`` is returned for the rho-update accumulator (policy 2).

    Otherwise selects per source among the fixed families (Fortran ``z0``
    select, amica15.f90:1333/1346/1359/1371), in AMICATorchNG's nesting order
    (2, 3, 4, 1, GG-fallthrough -- see ``_score`` for why the fallthrough is
    structurally dead here). ``az_rho`` is returned as ``None`` in this branch
    (policy 5): ``rho`` is frozen for every non-GG family (``self.dorho`` is
    False), so no caller needs ``|y|^rho`` there, and returning ``None`` instead
    of the true (unused) value makes an accidental read fail loudly rather than
    silently accepting a quantity nobody validated for these families.
    """
    abs_y = mx.abs(y)
    az_rho = mx.power(abs_y, rho)  # reused by the rho-update accumulator (GG only)
    log_pdf_gg = -az_rho - _LOG2 - lgamma_table
    if pdtype is None:
        return log_pdf_gg, az_rho

    log_pdf_2 = -0.5 * y * y - _LOG_SQRT_2PI  # Gaussian
    log_pdf_3 = -2.0 * _logcosh(0.5 * y) - _LOG4  # logistic (sech^2)
    lc = _logcosh(y)
    log_pdf_4 = -0.5 * y * y + lc - _LOG_NORM_COSH_SUB  # sub-Gaussian cosh+
    log_pdf_1 = -0.5 * y * y - lc - _LOG_NORM_COSH_SUP  # super-Gaussian cosh-
    log_pdf = mx.where(
        pdtype == 2,
        log_pdf_2,
        mx.where(
            pdtype == 3,
            log_pdf_3,
            mx.where(
                pdtype == 4, log_pdf_4, mx.where(pdtype == 1, log_pdf_1, log_pdf_gg)
            ),
        ),
    )
    return log_pdf, None


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

    The component-sharing parameters (issue #263) likewise carry AMICATorchNG's
    names, defaults, validation and semantics:

    ``share_comps`` (False)
        Enable multi-model component sharing (Fortran ``share_comps`` /
        ``identify_shared_comps``, amica15.f90:1916): components that are
        near-collinear across different models are merged so they share one
        mixing column and one density. Requires ``n_models >= 2`` (a model
        cannot share with itself); accepted but inert otherwise. OFF by default,
        so default fits are unchanged. There is no bit-exact oracle -- the
        reference's similarity metric is never initialized (like the dead
        ``do_choose_pdfs``, #26) -- so this implements the intended algorithm,
        validated by real-data behavior and against the PyTorch backend. A
        merge that fires on the LAST fit iteration is reflected in the
        returned model but trails in ``final_ll_``; see that attribute's
        comment (issue #269).
    ``share_start`` (100) / ``share_iter`` (100)
        Sharing schedule: first iteration to attempt merges and the interval
        between attempts. The A-update is held for the first 6 iterations of
        every cycle (whether or not a merge fired) so the densities can settle;
        ``share_iter`` must be ``> 6`` so that window never consumes the whole
        cycle, and ``share_start`` must be ``>= 1``.
    ``comp_thresh`` (0.99)
        Cosine-similarity cutoff, in the de-sphered (sensor-space) metric, above
        which two mixing columns are identified and merged. Must be in
        ``(0, 1]``. The de-sphering uses ``pinv(sphere)``, so sharing also works
        on rank-reduced and rank-deficient fits (issues #253, #221); see
        :meth:`_identify_shared_comps`.

    The Newton parameters (issue #264) likewise carry AMICATorchNG's names,
    defaults and semantics:

    ``do_newton`` (False)
        Precondition the ``A``/``W`` natural gradient with the approximate
        Hessian once ``iteration >= newt_start`` (Fortran ``do_newton``: the
        2x2 solve and its positive-definiteness guard at amica15.f90:1718-1741,
        the ramp and fallback at :1803-1816). Natural gradient alone plateaus
        short of the Fortran solution; the Newton step is what closes the gap.
        OFF by default, and every accumulator it needs is gated on it, so a
        default fit is bit-for-bit what it was before #264.
    ``newt_start`` (20)
        Iteration at which the Newton step switches on (natural gradient runs
        before it, letting the mixture parameters settle first). It also gates
        the ``rholrate`` ceiling ratchet, independently of ``do_newton``
        (Fortran amica15.f90:1067).
    ``newtrate`` (0.5)
        Maximum learning rate the ramp climbs to while Newton is active and
        positive definite; the natural-gradient phase (and any fallback
        iteration) is capped at ``lrate_cap`` instead.
    ``newt_ramp`` (10)
        Denominator of the per-iteration learning-rate ramp toward the current
        ceiling: ``lrate = min(ceiling, lrate + min(1/newt_ramp, lrate))``.

    Whenever any source pair fails the positive-definiteness guard the whole
    model falls back to the natural gradient for that iteration and
    ``n_newton_fallbacks`` counts it (as AMICATorchNG does), so an all-fallback
    run is visible without re-instrumenting.

    The source-density family parameters (issue #265, porting AMICATorchNG's
    issue #26) likewise carry AMICATorchNG's names, defaults and semantics:

    ``pdftype`` (0)
        Per-source density family (Fortran ``amica15.f90`` ``pdtype`` codes): 0
        generalized Gaussian (default; rho adapts), 2 Gaussian, 3 logistic, 4
        sub-Gaussian cosh+. ``pdftype=1`` enables the extended-Infomax adaptive
        switcher, which flips each source between the super-Gaussian (code 1)
        and sub-Gaussian (code 4) cosh densities by kurtosis sign (see
        :meth:`_choose_pdfs`). The GG shape update is frozen for every non-GG
        family (Fortran ``dorho=.false.``, ``self.dorho = pdftype == 0``); the
        single-component families 1/4 (and the adaptive mode) require
        ``n_mix=1``. ``pdftype=0`` stays byte-for-byte the pre-#265
        implementation (the ``_pdtype_h`` ``None`` fast path, policy 2 --
        verified by a before/after bit-identity check, see
        ``.context/issue-265/pdf_family_findings.md``). ``rho`` does not
        describe the fitted density for codes 1-4: it stays frozen at ``rho0``
        and is only ever meaningful for the generalized-Gaussian family
        (code 0).
    ``kurt_start`` (3) / ``num_kurt`` (5) / ``kurt_int`` (1)
        Adaptive-switch schedule (only used when ``pdftype=1``): first
        iteration to re-estimate kurtosis, number of switch passes, and the
        iteration interval between them. ``num_kurt=0`` disables switching (the
        family stays at its super-Gaussian init). No bit-exact oracle -- the
        reference's own switch is dead code (``do_choose_pdfs`` is set but
        ``m2sum``/``m4sum`` are never accumulated, amica15.f90:608-615) -- so
        this is behavior-validated on real data (ADR 0002).

    The block-size search parameters (issue #232) likewise carry
    AMICATorchNG's names, defaults and semantics:

    ``do_opt_block`` (False)
        Time candidate block sizes on the real data and GPU at the start of
        ``fit`` and keep the fastest, instead of using ``block_size`` as given
        (Fortran ``do_opt_block``). MLX is the least block-size-sensitive
        backend measured (2.6x from 512 to a single block, against 32x for
        PyTorch-MPS, issue #216), so there is less here to win than on the
        other backends. The choice is timing-based and therefore
        machine-dependent, so it is OFF by default and a run compared against
        the reference binary must leave it off and pin ``block_size``.
    ``blk_min`` (4096) / ``blk_max`` (32768) / ``blk_step`` (4096)
        Candidate sweep, Fortran's arithmetic stepping, clamped to
        ``n_samples`` and to a conservative estimate of what fits in the GPU's
        recommended working set. A candidate MLX cannot allocate raises a
        catchable ``RuntimeError`` (``[metal::malloc] ...`` -- not the process
        abort MLX's LU takes on singular input, issue #274), so it is skipped
        and the fit continues at the largest size that ran.

    The best-of-N restart parameters (issue #198) likewise carry
    AMICATorchNG's names, defaults and semantics:

    ``n_restarts`` (1)
        Number of independent fits to run from different seeds, keeping the one
        with the highest ``final_ll_``. ``1`` (the default) bypasses the restart
        machinery entirely, so a default fit is bit-for-bit what it was before
        #198. ``n_restarts > 1`` requires a base ``seed`` (or explicit
        ``restart_seeds``) so the winner can be reproduced, and costs
        ``n_restarts`` times as long (restarts run serially). This is a pamica
        extension: Fortran has no search over seeds. See
        :mod:`pamica.restarts` and ``docs/guides/amica-differences.md``.
    ``restart_seeds`` (None)
        Explicit per-restart seeds, exactly ``n_restarts`` of them; otherwise
        ``seed, seed + 1, ...``.
    """

    def __init__(
        self,
        n_channels: int,
        n_models: int = 1,
        n_mix: int = 3,
        block_size: int = 8192,
        do_opt_block: bool = False,
        blk_min: int = blocktune.DEFAULT_BLK_MIN,
        blk_max: int = blocktune.DEFAULT_BLK_MAX,
        blk_step: int = blocktune.DEFAULT_BLK_STEP,
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
        newtrate: float = 0.5,
        do_newton: bool = False,
        rho0: float = 1.5,
        minrho: float = 1.0,
        maxrho: float = 2.0,
        rholrate: float = 0.05,
        rholratefact: float = 0.1,
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
        mineig: float = MINEIG,
        mineig_rel: Optional[float] = MINEIG_REL,
        seed: Optional[int] = None,
        n_restarts: int = restarts.DEFAULT_N_RESTARTS,
        restart_seeds: Optional[Sequence[int]] = None,
    ):
        self.n_channels = n_channels
        self.n_models = n_models  # multi-model (#81) + component sharing (#263)
        self.n_mix = n_mix
        self.n_comps = n_channels * n_models
        self.block_size = block_size
        self.do_opt_block = do_opt_block
        self.blk_min = blk_min
        self.blk_max = blk_max
        self.blk_step = blk_step
        if do_opt_block:
            # Validated only when the search is on, matching share_comps and
            # AMICATorchNG: inert otherwise, and a Fortran input.param carrying
            # these alongside do_opt_block=0 must stay loadable (issue #232).
            blocktune.validate_block_tune_params(blk_min, blk_max, blk_step)

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
        # Newton schedule (issue #264), same names/defaults/semantics as
        # AMICATorchNG. newt_start doubles as the gate on the rholrate ceiling
        # ratchet, which Fortran conditions on iter > newt_start independently of
        # do_newton (amica15.f90:1067) -- so it stays meaningful for a
        # natural-gradient fit too. newtrate is a CEILING that ratchets down at
        # maxdecs during fit, so keep the constructor value for the per-fit reset
        # in _initialize_parameters (as lrate0/rholrate0 do).
        self.newt_start = newt_start
        self.newtrate = newtrate
        self.newtrate0 = newtrate
        self.do_newton = do_newton
        # Iterations on which the Newton direction was rejected as not positive
        # definite and the natural gradient used instead (AMICATorchNG's counter
        # of the same name); reset per fit in _initialize_parameters.
        self.n_newton_fallbacks = 0

        self.rho0 = rho0
        self.minrho = minrho
        self.maxrho = maxrho
        self.rholrate0 = rholrate
        self.rholrate = rholrate
        self.rholratefact = rholratefact

        # Source-density family selection (issue #265, porting AMICATorchNG's
        # issue #26 -- see torch_impl/core.py:640-676 for the identical block).
        # Values match Fortran's per-source pdtype codes: 0 generalized Gaussian
        # (the default, GG-mixture with adaptive rho), 2 Gaussian mixture, 3
        # logistic (sech^2) mixture, 4 sub-Gaussian cosh+ (single component).
        # pdftype=1 enables the extended-Infomax adaptive switcher (Fortran's
        # do_choose_pdfs trigger), which flips each source between the
        # super-Gaussian (code 1) and sub-Gaussian (code 4) cosh densities by
        # kurtosis sign on the kurt_start/num_kurt/kurt_int schedule. Families 1
        # and 4 are single-component (no alpha mixture).
        if pdftype not in (0, 1, 2, 3, 4):
            raise ValueError(f"pdftype must be one of 0,1,2,3,4; got {pdftype}")
        self.pdftype = pdftype
        # Fortran freezes the GG shape update for every non-GG family
        # (amica15.f90: `if (pdftype /= 0) dorho = .false.`, lines 3704-3705).
        self.dorho = pdftype == 0
        # pdftype==1 is Fortran's adaptive trigger (amica15.f90:612).
        self.do_choose_pdfs = pdftype == 1
        self.kurt_start = kurt_start
        self.num_kurt = num_kurt
        self.kurt_int = kurt_int
        # Families 1/4 (and the adaptive mode, which uses only codes 1 and 4) are
        # single-component densities: Fortran's z0 references only mixture
        # component j=1 and omits log(alpha). They are meaningful only with
        # n_mix == 1.
        if pdftype in (1, 4) and n_mix != 1:
            raise ValueError(
                f"pdftype={pdftype} is a single-component density (adaptive mode "
                f"uses codes 1 and 4); it requires n_mix=1, got n_mix={n_mix}."
            )
        # Validate the adaptive-switch schedule up front (mirrors the
        # share_comps checks below): kurt_int==0 would otherwise raise a bare
        # ZeroDivisionError deep in fit(), and a negative kurt_int silently
        # changes the schedule.
        if self.do_choose_pdfs:
            if kurt_int < 1:
                raise ValueError(f"kurt_int must be >= 1, got {kurt_int}")
            if kurt_start < 1:
                raise ValueError(f"kurt_start must be >= 1, got {kurt_start}")
            if num_kurt < 0:
                raise ValueError(f"num_kurt must be >= 0, got {num_kurt}")
        # Adaptive-switch counter (Fortran-1-indexed schedule check in fit());
        # reset per fit in _initialize_parameters.
        self.n_kurt_done = 0

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

        # Best-of-N restarts (issue #198), a pamica extension: Fortran has no
        # search over seeds. Resolved here so a bad configuration fails before
        # any data is touched, and derived from the CONSTRUCTOR seed so that a
        # second fit() on the same instance repeats the same seeds even though
        # fit() leaves self.seed on the winning restart.
        self._restart_seeds = restarts.resolve_seeds(n_restarts, restart_seeds, seed)
        self.n_restarts = int(n_restarts)
        self.restart_seeds = None if restart_seeds is None else list(restart_seeds)
        # Per-restart records, set by fit(): index-aligned lists of the seed each
        # restart ran from, the log-likelihood it returned (NaN for a degenerate
        # restart) and why it stopped. A degenerate restart is excluded from
        # selection but kept here -- it is a fact about that seed.
        self.restart_seeds_: List[Optional[int]] = []
        self.restart_lls_: List[float] = []
        self.restart_stop_reasons_: List[Optional[str]] = []

        self.iteration = 0
        self.ll_history: list[float] = []
        # Log-likelihood of the returned parameters, set by fit() to
        # ll_history[-1] (there is no keep_best restore on this backend, see
        # the module docstring). Under share_comps, if a merge fires on the
        # LAST fit iteration, the returned A/W/comp_list are already
        # post-merge but final_ll_ still reports the pre-merge
        # log-likelihood -- the merge runs after that iteration's LL is
        # recorded, so its effect on the LL only shows up in the next
        # iteration's E-step, which never runs. This matches the reference
        # ordering (Fortran identify_shared_comps runs after the iteration's
        # LL accumulation, amica15.f90:1856-1858) and AMICATorchNG's ordering, so
        # it is documented behavior, not a bug (issue #269).
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
        # Per-source density-family codes (issue #265), (n_channels, n_models)
        # int32, allocated in _initialize_parameters. None only before the first
        # fit -- once allocated it is always a full pdftype-filled tensor, even
        # for pdftype=0 (the _pdtype_h None fast path is a SEPARATE, cheaper
        # check on the scalar self.pdftype, not on this being None).
        self.pdtype: Optional[mx.array] = None
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

    # The last entry is only reachable under best-of-N restarts (issue #198): a
    # restart whose fit raised rather than stopping, recorded as degenerate so a
    # search in which every restart crashed still reports an unusable model.
    _DEGENERATE_STOP_REASONS = (
        "nan_ll",
        "singular_ll",
        "nan_params",
        restarts.ERROR_STOP_REASON,
    )

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
            # numpy's default sample covariance (/(N-1)) -- the same choice, and
            # the reasoning for it, at torch_impl/core.py:823-827.
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
        order as AMICATorchNG/AMICA_NumPy (core.py:918-973), so a shared seed
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

        # Per-source density-family codes, Fortran `pdtype = pdftype`
        # (amica15.f90:611; AMICATorchNG core.py:960-963). In adaptive mode
        # (pdftype==1) every source starts as the super-Gaussian code (1),
        # since self.pdftype IS 1 there -- no special-case fill needed.
        self.pdtype = mx.array(np.full((n, m), self.pdftype, dtype=np.int32))
        self.n_kurt_done = 0

        # Reset the mutable optimization state to the pristine constructor values
        # (lrate_cap, newtrate and rholrate are ratcheted down during fit, and
        # n_newton_fallbacks counts one fit), so a re-fit starts fresh --
        # AMICATorchNG does the same at core.py:966-971/:1936.
        self.lrate = self.lrate0
        self.lrate_cap = self.lrate0
        self.newtrate = self.newtrate0
        self.rholrate = self.rholrate0
        self.n_newton_fallbacks = 0
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

        These build lazy graph nodes, so on a HEALTHY matrix the ``inv``/
        ``slogdet`` calls below do not themselves materialize -- the graph is
        realized later where ``mx.eval`` runs (in ``fit``). A singular ``A``
        is different: MLX 0.32's CPU-stream ``mx.linalg.inv`` does not raise a
        catchable Python exception on one. LAPACK's LU failure aborts the
        whole process (``libc++abi: ... [Inverse::eval_cpu] LU factorization
        failed``), which no ``try``/``except`` around ``fit`` can catch
        (issue #274). Near-singular-but-finite float32 matrices either invert
        to large finite values or hit this same abort -- there is no route to
        a catchable float32 overflow instead, which is why ``fit``'s
        ``nan_params`` guard treats a non-finite ``W``/``_logdet_W`` as
        defense in depth rather than a reachable case on its own.
        Consequently, each per-model matrix is condition-checked host-side,
        eagerly, immediately before its ``inv`` call: see
        ``_INV_COND_THRESHOLD``. This eager host read (via ``np.array``)
        forces the SAME materialization the CPU-stream ``inv``/``slogdet``
        below would have forced anyway, so the guard adds no new cross-stream
        handoff -- only a small ``np.linalg.cond`` on a matrix already on the
        host. Measured on the bundled sample: negligible relative to
        per-iteration time (see the issue #274 PR body for the number). A
        condition number above the threshold raises ``RuntimeError`` naming
        the model index, iteration, and value, in place of the uncatchable
        abort.

        A matrix with non-finite entries needs its own handling, verified by
        isolated-subprocess reproduction: a matrix that is ONLY non-finite
        (no other defect) never aborts -- ``inv`` propagates NaN/inf into
        ``W``, caught downstream by ``nan_params``. But a matrix that is
        BOTH non-finite AND structurally singular elsewhere (an exact
        duplicate column plus one unrelated NaN, confirmed to reach and
        abort the process) is not covered by "only non-finite" reasoning --
        skipping the check on any non-finite entry, as an earlier version of
        this guard did, lets that combination through unguarded. The check
        below therefore 0-fills non-finite entries (a no-op when already
        finite) before computing the condition number, UNLESS every entry is
        non-finite (the observed shape of a dead-model corruption -- a
        zero-responsibility model dividing by ``dgm==0`` -- which carries no
        structural signal to check and is left to flow to ``inv``/
        ``nan_params`` exactly as before).

        Caveat, carried from ``_INV_COND_THRESHOLD``: no scalar condition
        number, on the sanitized matrix or otherwise, can guarantee catching
        every conceivable abort-capable matrix -- the empirically observed
        LU-abort onset (cond~9e8 to beyond cond~5e10, matrix-dependent) sits
        below the 1e12 threshold, so a believed-rare residual window remains
        between "passes this check" and "would actually abort".
        """
        assert self.A is not None and self.comp_list is not None
        ws, logdets = [], []
        for h in range(self.n_models):
            A_h = self.A[:, self.comp_list[:, h]]
            a_h_np = np.array(A_h, dtype=np.float32, copy=False)
            finite_mask = np.isfinite(a_h_np)
            # A matrix with ZERO finite entries carries no signal to check --
            # this is exactly the observed shape of a dead-model corruption
            # (a zero-responsibility model dividing by dgm==0 propagates
            # NaN/inf through the WHOLE per-model direction matrix, so all
            # comp_list columns for that model go non-finite together, not
            # just one entry -- confirmed on a real fitted 2-model dead-model
            # state). Skipping here reproduces the pre-guard behavior exactly:
            # mx.linalg.inv on a wholly non-finite A does not abort -- it
            # propagates NaN/inf into W, which fit()'s existing nan_params
            # guard already catches.
            #
            # Otherwise (fully finite, OR a MINORITY of entries non-finite),
            # sanitize any non-finite entries to 0.0 before computing cond.
            # This is a no-op when already fully finite. When partially
            # non-finite, 0.0 is a neutral fill at the same natural scale as
            # A's entries (near-unit-norm columns) -- unlike a huge/extreme
            # sentinel, which was tried and rejected: it makes ANY non-finite
            # entry look "infinitely far" from the rest of the matrix via pure
            # scale mismatch, so a well-conditioned matrix with one stray NaN
            # and a genuinely singular one both come back cond=inf, which
            # cannot distinguish them. The neutral fill can (verified: a
            # well-conditioned real matrix with one injected NaN reads
            # cond~35 after 0-fill; the same matrix with an EXACT DUPLICATE
            # column plus that same stray NaN -- the reviewer-reported killer
            # combination, confirmed by isolated-subprocess reproduction to
            # abort the process under the OLD skip-on-any-non-finite logic --
            # reads cond~3.7e16, comfortably over the threshold). A raise here
            # is unconditional on the underlying non-finite entries (the
            # sanitized cond is a probe of the surrounding structure, not a
            # claim about what the unknown entries "really" are), so it also
            # still catches an exact-duplicate-column A with no non-finite
            # entries at all, unchanged from before.
            if np.any(finite_mask):
                sentinel = np.where(finite_mask, a_h_np, np.float32(0.0)).astype(
                    np.float32
                )
                cond = float(np.linalg.cond(sentinel))
                if not math.isfinite(cond) or cond > _INV_COND_THRESHOLD:
                    n_bad = int(a_h_np.size - finite_mask.sum())
                    nonfinite_note = (
                        f" ({n_bad} of {a_h_np.size} entries were already "
                        "non-finite and were 0-filled for this check.)"
                        if n_bad
                        else ""
                    )
                    raise RuntimeError(
                        f"Singular unmixing matrix for model {h} at iteration "
                        f"{self.iteration}: cond(A[:, comp_list[:, {h}]]) = "
                        f"{cond:.3e} exceeds the float32 threshold "
                        f"{_INV_COND_THRESHOLD:.1e} (MLX's CPU-stream inv "
                        "would otherwise abort the process instead of "
                        "raising; #274). Likely a component collapse -- "
                        "consider re-seeding or a lower lrate."
                        f"{nonfinite_note}"
                    )
            wh = mx.linalg.inv(A_h, stream=_CPU)
            ws.append(wh)
            logdets.append(mx.linalg.slogdet(wh, stream=_CPU)[1])
        self.W = mx.stack(ws, axis=0)  # (n_models, n, n)
        self._logdet_W = mx.stack(logdets)  # (n_models,)

    def _pdtype_h(self, h: int) -> Optional[mx.array]:
        """Per-source density-family codes for model ``h``, shaped for
        broadcasting against ``(batch, n_channels, n_mix)`` arrays (AMICATorchNG
        ``_pdtype_h``, core.py:984-993), or ``None`` on the default
        ``pdftype=0`` (GG-only) fast path so the E-step stays bit-identical to
        the pre-#265 implementation (policy 2).
        """
        if self.pdftype == 0:
            return None
        assert self.pdtype is not None
        return self.pdtype[:, h][None, :, None]  # (1, n_channels, 1)

    # ------------------------------------------------------------------
    # E-step
    # ------------------------------------------------------------------
    def _forward(self, Xb: mx.array):
        """E-step forward pass for one block, per model (AMICATorchNG._forward,
        core.py:998-1071). ``Xb`` is ``(n_channels, batch)``. Returns ``logV``
        ``(batch, n_models)`` and per-model lists ``(b, z, y, az_rho)``. For
        n_models=1 (c=0, gm=1, comp_list=identity) this is numerically identical
        to the single-model path. Each model's ``az_rho`` entry is ``None`` when
        that model's family is non-GG (``_pdtype_h`` not None; policy 5) -- see
        ``_log_pdf``."""
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
            log_pdf, az_rho = _log_pdf(y, rho_h, lgamma_h, self._pdtype_h(h))
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
        """Exact-EM sufficient statistics for one block (AMICATorchNG.
        _get_block_updates, core.py:1141-1283). Mixture stats are scattered into
        their ``comp_list`` columns; ``dWtmp``/``dgm``/``dc_numer`` are
        per-model. For n_models=1 (v==1, identity comp_list) this reproduces the
        single-model accumulators exactly.

        Under ``do_newton`` the three Newton curvature accumulators
        (``dsigma2_numer``, ``dkappa_numer``, ``dlambda_numer``; see
        :meth:`_finalize_newton_stats`) are emitted as well. They are gated on
        ``do_newton`` rather than always computed, matching AMICATorchNG: the key
        is then either present in every block of a fit or absent from all of
        them, so ``_accumulate_blocks``' generic key loop sums them with no
        special case, and a natural-gradient fit does none of the work.

        ``drho_n`` (the rho digamma-update numerator) is likewise gated on
        ``self.dorho`` (issue #265, policy 5): rho is frozen for every non-GG
        family (Fortran ``dorho=.false.``), so accumulating it there is dead
        work AMICATorchNG still pays (it always accumulates and discards) --
        this is a deliberate WORK-only divergence, not a numeric one. Gating
        uniformly on ``self.dorho`` (fixed for the whole model, not per-block)
        keeps the key consistently present-or-absent across every block of a
        fit, exactly like the Newton keys above.
        """
        logV, b_list, z_list, y_list, azrho_list = self._forward(Xb)
        block_ll = mx.logsumexp(logV, axis=1).sum()
        v = mx.softmax(logV, axis=1)  # (batch, n_models) model responsibilities
        nmix, ncomp = self.n_mix, self.n_comps
        tiny = float(np.finfo(np.float32).tiny)

        def zeros():
            return mx.zeros((nmix, ncomp), dtype=mx.float32)

        dalpha_n, dmu_n, dmu_d = zeros(), zeros(), zeros()
        dbeta_n, dbeta_d = zeros(), zeros()
        if self.dorho:
            drho_n = zeros()
        dgm_cols, dwtmp_mods, dc_cols = [], [], []
        # Newton curvature, model-major like dWtmp: one entry per model, stacked
        # below (the MLX convention -- MLX has no in-place slice assignment, so
        # per-model lists + mx.stack replace torch's `dsigma2_numer[:, h] = ...`).
        dsigma2_mods, dkappa_mods, dlambda_mods = [], [], []

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
            pdtype_h = self._pdtype_h(h)

            fp = _score(y, rho_h, pdtype_h)
            u = v_h[:, None, None] * zr  # u = v*z, (batch, n_channels, n_mix)
            ufp = u * fp

            dgm_cols.append(v_h.sum())
            dalpha_n = dalpha_n.at[:, idx].add(u.sum(0).T)
            dmu_n = dmu_n.at[:, idx].add(ufp.sum(0).T)
            # Phase A guard: float32 can round y to exactly 0 (fp(0)=0 => ufp=0),
            # so ufp/y is 0/0=NaN; where y==0, 0/1 contributes 0 (issue #75).
            # torch's safe_y substitution (core.py:1231) is mirrored exactly for
            # every family here, even though the true fp/y limit at y->0 is a
            # finite nonzero constant for codes 2/3/1 (fp'(0): 1 Gaussian
            # (fp=y), 0.5 logistic (fp=tanh(y/2)), 2 super-Gaussian
            # (fp=y+tanh(y))) and 0 for code 4 (fp=y-tanh(y), whose Taylor
            # expansion is O(y^3)) -- parity with AMICATorchNG is the spec, not
            # the true limit; see the PR body for the measured per-family
            # y==0 frequency.
            safe_y = mx.where(y == 0, mx.ones_like(y), y)
            dmu_d = dmu_d.at[:, idx].add((beta_h[0] * (ufp / safe_y).sum(0)).T)
            dbeta_n = dbeta_n.at[:, idx].add(u.sum(0).T)
            dbeta_d = dbeta_d.at[:, idx].add((ufp * y).sum(0).T)

            if self.dorho:
                logab = rho_h * mx.log(mx.maximum(mx.abs(y), tiny))
                logab = mx.where(az_rho < _EPSDBLE, mx.zeros_like(logab), logab)
                drho_n = drho_n.at[:, idx].add((u * (az_rho * logab)).sum(0).T)

            g = (beta_h * ufp).sum(-1)  # (batch, n_channels)
            dwtmp_mods.append(g.T @ b)  # (n_channels, n_channels)
            dc_cols.append(Xb @ v_h)  # data-space bias numerator sum_t v_h*x

            if self.do_newton:
                # Newton curvature accumulators (Fortran amica15.f90:1439-1446,
                # 1496-1513), in terms of the score fp -- not the density
                # derivative dpdf -- and reusing this block's live E-step locals,
                # so Newton adds no extra pass over the data.
                dsigma2_mods.append((v_h[:, None] * b**2).sum(0))  # (n_ch,)
                dkappa_mods.append(
                    ((u * fp**2).sum(0) * beta_h[0] ** 2).T
                )  # (n_mix, n_ch)
                dlambda_mods.append((u * (fp * y - 1.0) ** 2).sum(0).T)  # (n_mix, n_ch)

        updates = {
            "dgm": mx.stack(dgm_cols),  # (n_models,)
            "dalpha_n": dalpha_n,
            "dmu_n": dmu_n,
            "dmu_d": dmu_d,
            "dbeta_n": dbeta_n,
            "dbeta_d": dbeta_d,
            "dWtmp": mx.stack(dwtmp_mods, axis=0),  # (n_models, n_ch, n_ch)
            "dc_numer": mx.stack(dc_cols, axis=1),  # (n_channels, n_models)
            "ll": block_ll,
        }
        if self.dorho:
            updates["drho_n"] = drho_n
        if self.do_newton:
            updates["dsigma2_numer"] = mx.stack(dsigma2_mods)  # (n_models, n_ch)
            updates["dkappa_numer"] = mx.stack(dkappa_mods)  # (n_models, n_mix, n_ch)
            updates["dlambda_numer"] = mx.stack(dlambda_mods)  # (n_models, n_mix, n_ch)
        return updates

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

    @staticmethod
    def _available_memory_bytes() -> Optional[int]:
        """What the Apple GPU reports it can comfortably work with.

        ``max_recommended_working_set_size`` rather than the raw memory size:
        MLX will happily allocate past the recommended set and start paging,
        which shows up as a mysteriously slow candidate rather than a failure,
        so the cap is applied against the number the driver actually
        recommends. ``None`` (no cap) if MLX cannot report it.

        Both keys report CAPACITY, not currently-free memory (unlike CUDA's
        ``mem_get_info``): neither subtracts what is already allocated, here or
        by anything else sharing the unified memory. The cap is therefore an
        upper bound on what the GPU could ever give, which is why it is only a
        first filter -- catching the real ``[metal::malloc]`` failure is what
        makes the search safe.
        """
        try:
            info = mx.device_info()
        except (AttributeError, RuntimeError) as exc:
            logger.debug(
                "could not query MLX device memory (%s: %s); block-size "
                "search runs without a memory cap",
                type(exc).__name__,
                exc,
            )
            return None
        size = info.get("max_recommended_working_set_size") or info.get("memory_size")
        return int(size) if size else None

    def _tune_block_size(self, X: mx.array) -> None:
        """Set ``self.block_size`` to the fastest timed candidate (issue #232).

        The probe is one ``_accumulate_blocks`` pass, evaluated in full: MLX
        builds a lazy graph, so without ``mx.eval`` over every accumulator the
        clock would measure graph *construction* and pick the block size that
        builds fastest rather than the one that runs fastest. The pass only
        reads model state and consumes no RNG, and ``block_size`` is restored
        around every probe, so the fit that follows is bit-identical to one
        started directly at the chosen size.
        """
        saved = self.block_size

        def probe(size: int) -> float:
            self.block_size = size
            try:
                start = time.perf_counter()
                acc = self._accumulate_blocks(X)
                mx.eval(list(acc.values()))
                return time.perf_counter() - start
            finally:
                # Never leave the model holding a candidate -- least of all one
                # that just failed to allocate (issue #232).
                self.block_size = saved

        self.block_size = blocktune.search(
            probe=probe,
            fallback=saved,
            blk_min=self.blk_min,
            blk_max=self.blk_max,
            blk_step=self.blk_step,
            n_samples=int(X.shape[1]),
            n_channels=self.n_channels,
            n_mix=self.n_mix,
            n_models=self.n_models,
            # The MLX backend is float32 throughout (Apple GPUs have no
            # float64); see the module docstring.
            itemsize=4,
            available_bytes=self._available_memory_bytes(),
            log=logger,
        )

    # ------------------------------------------------------------------
    # M-step
    # ------------------------------------------------------------------
    def _finalize_newton_stats(self, acc: dict):
        """Reduce the Newton block accumulators into ``(sigma2, lambda_, kappa)``
        (AMICATorchNG._finalize_newton_stats, core.py:1307-1331; Fortran
        amica15.f90:1666-1680).

        The Fortran ``baralpha``/``dkappa_denom``/``dlambda_denom``
        responsibility masses all cancel algebraically against the per-mixture
        ``dalpha`` weighting, leaving (with ``dgm = sum_t v_h`` the raw model
        mass):

            sigma2[h,i] = dsigma2_numer[h,i] / dgm[h]
            kappa[h,i]  = sum_j dkappa_numer[h,j,i] / dgm[h]
            lambda[h,i] = sum_j (dlambda_numer[h,j,i]
                                 + dkappa_numer[h,j,i] * mu[j,comp(i,h)]^2) / dgm[h]

        MUST be called with the PRE-update ``mu``: lambda folds ``mu^2`` in, and
        Fortran does that during E-step accumulation, before the M-step moves mu
        (see the call site in :meth:`_update_parameters`).

        ``dgm`` is ``(n_models,)`` and everything else is model-major, so the
        model mass broadcasts on axis 0 (``dgm[:, None]``). Getting that axis
        wrong is the NumPy backend's issue #267 crash -- there the layout is
        model-MINOR, so the same reduction needs ``dgm[None, :]``.

        Returns ``(sigma2, lambda_, kappa)``, each ``(n_models, n_channels)``.
        """
        assert self.mu is not None and self.comp_list is not None
        dgm = acc["dgm"][:, None]  # (n_models, 1)
        sigma2 = acc["dsigma2_numer"] / dgm
        kappa = acc["dkappa_numer"].sum(axis=1) / dgm
        # mu at each source's component: mu[j, comp_list[i,h]]. MLX 2-D advanced
        # indexing matches NumPy's, giving (n_mix, n_ch, n_models); transpose to
        # the model-major layout the accumulators use.
        mu_at = self.mu[:, self.comp_list].transpose(2, 0, 1)
        lambda_ = (acc["dlambda_numer"] + acc["dkappa_numer"] * mu_at**2).sum(
            axis=1
        ) / dgm
        return sigma2, lambda_, kappa

    def _newton_direction(self, dA_h, sigma2_h, lambda_h, kappa_h):
        """Per-model Newton direction ``H`` from the natural gradient ``dA_h``
        (AMICATorchNG._newton_direction, core.py:1333-1361).

        Vectorized port of the per-source-pair 2x2 solve (Fortran
        amica15.f90:1718-1741):

            H[i,i] = dA_h[i,i] / lambda[i]
            sk1 = sigma2[i]*kappa[k];  sk2 = sigma2[k]*kappa[i]   (i != k)
            H[i,k] = (sk1*dA_h[i,k] - dA_h[k,i]) / (sk1*sk2 - 1)  if sk1*sk2 > 1

        Closed form, so this needs no linear algebra and stays on the GPU stream
        (unlike ``inv``/``slogdet``, which MLX runs CPU-only). The ``prod > 1.0``
        test and the ``diagonal(dA_h)/lambda_h`` divide are deliberately raw --
        no epsilon margin, no guard -- matching the PyTorch and NumPy backends;
        a non-finite result is contained downstream by the ``nan_params`` abort
        in :meth:`fit` and by the masked ``ndtmpsum`` reduction.

        Returns ``(H, posdef)``. ``posdef`` is False if any off-diagonal pair
        fails ``sk1*sk2 > 1`` (the positive-definiteness guard); the caller then
        falls back to the natural gradient for this model. Reading it costs one
        host sync of a scalar per Newton iteration per model -- accepted (it
        selects the lrate ramp target and drives the fallback counter, neither of
        which can stay on the device), alongside the M-step's existing
        dead-model and rho-NaN scalar syncs.
        """
        n = self.n_channels
        sk1 = sigma2_h[:, None] * kappa_h[None, :]  # [i,k] = sigma2[i]*kappa[k]
        sk2 = sigma2_h[None, :] * kappa_h[:, None]  # [i,k] = sigma2[k]*kappa[i]
        prod = sk1 * sk2
        valid = prod > 1.0
        denom = mx.where(valid, prod - 1.0, mx.ones_like(prod))
        h_off = (sk1 * dA_h - dA_h.T) / denom
        H = mx.where(valid, h_off, mx.zeros_like(h_off))
        # Diagonal overrides (uses lambda, not the off-diagonal formula).
        diag = mx.diagonal(dA_h) / lambda_h
        H = H - mx.diag(mx.diagonal(H)) + mx.diag(diag)
        # Positive-definite iff every OFF-diagonal pair passed the guard. MLX has
        # no boolean-mask indexing (torch's ``valid[offdiag].all()``), so force
        # the diagonal True instead -- same reduction, no gather.
        eye_bool = mx.eye(n, dtype=mx.bool_)
        posdef = bool(mx.all(mx.logical_or(valid, eye_bool)).item())
        return H, posdef

    def _update_parameters(self, acc: dict, n_samples: int):
        """Exact-EM mixture updates + natural-gradient A-update, optionally
        Newton-preconditioned (AMICATorchNG._update_parameters,
        core.py:1363-1616)."""
        # Fortran builds dAk from the PREVIOUS iteration's model weights: gm is
        # not reassigned until update_params (amica15.f90:1788+), after the
        # dAk/zeta accumulation in accum_updates_and_likelihood (:1749-1761).
        # Snapshot before overwriting, as AMICATorchNG does (the ordering
        # question issue #219 raised, fixed there and now here); MLX arrays are
        # immutable and gm is
        # only ever rebound, so a plain rebinding is a safe snapshot (torch
        # clones because its tensors could be written in place). Exactly gm for
        # n_models=1 (both are 1.0) and cancelling for a disjoint comp_list, so
        # the single-model and unshared multi-model paths are unchanged.
        assert self.gm is not None
        gm_prev = self.gm
        self.gm = acc["dgm"] / n_samples  # (n_models,); == 1 for single model
        tiny = float(np.finfo(np.float32).tiny)

        # Per-model data-space bias c[i,h] = sum_t v_h*x / sum_t v_h (Fortran
        # update_c, core.py:1401-1423). Skipped for n_models=1 (v==1 => c is the
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

        # Finalize the Newton curvature with the PRE-update mu. Fortran folds the
        # mu^2 term into lambda during E-step accumulation, before the M-step
        # moves mu (amica15.f90:1666-1680); doing it here -- between the alpha
        # update and the mu reassignment below -- is what reproduces that. Move
        # it one line later and lambda silently uses the updated mu: no error, no
        # NaN, just a subtly wrong Hessian (the torch backend's issue #24 bug,
        # pinned there and here by test_newton_finalize_uses_preupdate_mu).
        newton_active = self.do_newton and self.iteration >= self.newt_start
        if newton_active:
            sigma2, lambda_, kappa = self._finalize_newton_stats(acc)

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
        # matching AMICATorchNG (core.py:1483-1504), so it does not silently
        # poison the lgamma table and every subsequent E-step.
        # Deliberate divergence from AMICATorchNG (core.py:1483-1487), which also
        # skips the update when rho is pinned to a boundary (all 1.0 or all 2.0):
        # that early-exit needs a host sync on a (n_mix, n_comps) reduction over
        # rho every iteration. This backend does make a few scalar host syncs per
        # iteration -- the dead-model check above, the rho-NaN canary below, and
        # the Newton posdef flag -- each because a Python branch genuinely
        # depends on the value; the rho-boundary exit is not one of those, since
        # mx.clip clamps straight back to the boundary and the results are
        # identical either way. So it would buy nothing but a sync -- unrelated to
        # (and not fixed by) issue #265: pdftype != 0 is now supported, and for
        # those families this whole block is skipped by the outer ``self.dorho``
        # gate below (rho is frozen, Fortran ``dorho=.false.``), which is the
        # real reason the digamma work does not run there -- not an MLX
        # limitation. See ``_get_block_updates`` for the matching ``drho_n``
        # accumulator gate (a work-only divergence from AMICATorchNG, which
        # always accumulates and discards it).
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
        # is a LEFT-multiply by the transposed direction (core.py:1506-1514,
        # #24 root cause). Each model's direction is scattered into its mixing
        # columns as a gm-weighted average (Fortran dAk/zeta, core.py:1546-1561)
        # using the PREVIOUS iteration's gm (gm_prev, see the snapshot above):
        # for the default disjoint comp_list every column has one contributor, so
        # gm cancels and n_models=1 is byte-for-byte the old `A - lrate*(dA.T@A)`;
        # a SHARED column (#263) takes Fortran's responsibility-weighted average,
        # NOT a raw sum (a raw sum would over-step by the contributor count). A
        # merged-away column needs no special case: nothing scatters into it, so
        # its zeta is 0 and its dAk is 0/tiny = 0, i.e. it takes no step. It is
        # NOT a zero-norm column, and the rescale below does renormalize it like
        # any other -- but by its own retained (already ~unit) norm, so that is a
        # near-identity that perturbs it only at ULP scale. Hence
        # test_merged_away_columns_keep_their_last_finite_value disables
        # doscaling, to compare a frozen column exactly.
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
        # Newton only swaps out the per-model DIRECTION; the dAk/zeta scatter,
        # the gradient norm and the freeze structure below are untouched by it
        # (AMICATorchNG core.py:1531-1545). A model whose curvature fails the
        # positive-definiteness guard falls back to its natural gradient for this
        # iteration, and -- as in Fortran -- ANY model falling back also sends
        # the lrate ramp to lrate_cap instead of newtrate.
        eye = mx.eye(self.n_channels)
        directions = []
        no_newt = False
        for h in range(self.n_models):
            dA_h = -acc["dWtmp"][h] / acc["dgm"][h] + eye  # I - <g b^T>/dgm
            if newton_active:
                H, posdef = self._newton_direction(
                    dA_h, sigma2[h], lambda_[h], kappa[h]
                )
                if posdef:
                    directions.append(H)
                else:
                    no_newt = True
                    directions.append(dA_h)  # fall back to natural gradient
            else:
                directions.append(dA_h)

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
        # window, Fortran amica15.f90:1803), skip the step -- the lrate ramp, the
        # Newton-fallback bookkeeping, and the step itself -- so a discarded
        # Newton direction cannot pollute the fallback counter.
        if not self._a_frozen():
            if newton_active and no_newt:
                # Fortran prints "Hessian not positive definite, using natural
                # gradient" (amica15.f90:1809-1811). Surface the same signal so
                # an all-fallback run is visible without re-instrumenting.
                self.n_newton_fallbacks += 1
                logger.warning(
                    "Newton not positive definite at iter %d; using natural gradient.",
                    self.iteration,
                )

            # Learning-rate ramp: toward newtrate while Newton is active and
            # stable, otherwise toward lrate_cap (Fortran amica15.f90:1803-1816).
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
            scale = mx.sqrt((self.A**2).sum(axis=0))  # (n_comps,)
            # A zero-norm (collapsed) column is left untouched, not rescaled:
            # safe_scale is 1 there, so A/beta are unchanged and mu*safe_scale
            # keeps its prior value (matching AMICATorchNG's nonzero mask,
            # core.py:1608-1614 -- using raw `scale` would zero mu instead).
            safe_scale = mx.where(scale > 0, scale, mx.ones_like(scale))
            self.A = self.A / safe_scale
            self.mu = self.mu * safe_scale
            self.beta = self.beta / safe_scale

        # rho is frozen for every non-GG family (self.dorho is False), so the
        # table it feeds (used only on the GG _log_pdf path) cannot have
        # changed; refreshing it every iteration there would be dead host-CPU
        # work (issue #265, policy 5 -- gated like the drho_n accumulation and
        # the digamma pull above). Still built unconditionally once at init
        # (_initialize_parameters): _log_pdf's non-GG branches structurally
        # need a valid lgamma_table for their (dead-code) GG-fallthrough term,
        # even though its value is never selected there.
        if self.dorho:
            self._refresh_lgamma_table()
        self._update_unmixing_matrices()

    # ------------------------------------------------------------------
    # Adaptive PDF switch (issue #265; AMICATorchNG's #26 port)
    # ------------------------------------------------------------------
    def _choose_pdfs(self, X: mx.array) -> None:
        """Extended-Infomax adaptive PDF switch (Fortran ``do_choose_pdfs``,
        AMICATorchNG ``_choose_pdfs``, core.py:1790-1823).

        Re-estimates each source's kurtosis from the current model activations
        and sets its density family to the super-Gaussian (code 1) or
        sub-Gaussian (code 4) cosh density by kurtosis sign. The reference
        binary declares this (``pdftype==1`` sets ``do_choose_pdfs``,
        amica15.f90:612) but never runs the switch (``m2sum``/``m4sum`` are
        never accumulated, :608-609), so there is no bit-exact oracle;
        validated by real-data log-likelihood (must not decrease vs the fixed
        GG default).

        Mechanism difference from AMICATorchNG (MLX-motivated, not a decision
        difference): the second moments are accumulated block-by-block on the
        GPU in float32, but each block's small ``(n_channels,)``/scalar partial
        sums are pulled to the host and accumulated in numpy float64, and the
        kurtosis + validity guard + 1/4 decision run in numpy rather than on
        the MLX graph. The kurt>0 sign test is a knife-edge decision with no
        oracle, and ``m4`` loses float32 precision long before it would
        overflow, so the accumulation itself is done at float64 host precision;
        the host pulls happen on at most ``num_kurt`` iterations of a fit, so
        this costs nothing on the hot per-block path. The decision semantics
        (kurtosis formula, validity guard, super/sub-Gaussian mapping) are
        identical to AMICATorchNG's -- see :meth:`_pdtype_from_kurtosis`.
        """
        n_ch, n_models = self.n_channels, self.n_models
        m2 = np.zeros((n_ch, n_models), dtype=np.float64)
        m4 = np.zeros((n_ch, n_models), dtype=np.float64)
        nsub = np.zeros(n_models, dtype=np.float64)
        n_samples = X.shape[1]
        for start in range(0, n_samples, self.block_size):
            block = X[:, start : start + self.block_size]
            logV, b_list, *_ = self._forward(block)
            v = mx.softmax(logV, axis=1)  # (batch, n_models)
            for h in range(n_models):
                b = b_list[h]  # (batch, n_ch)
                vh = v[:, h][:, None]
                m2[:, h] += np.array((vh * b**2).sum(0), dtype=np.float64)
                m4[:, h] += np.array((vh * b**4).sum(0), dtype=np.float64)
                nsub[h] += float(v[:, h].sum().item())

        # Kurtosis = E[b^4]/E[b^2]^2 - 3 = nsub * m4 / m2^2 - 3, per (source,
        # model), in numpy float64 (policy 6).
        tiny = np.finfo(np.float64).tiny
        kurt = nsub[None, :] * m4 / np.maximum(m2**2, tiny) - 3.0
        new_pdtype = self._pdtype_from_kurtosis(kurt, nsub)
        # Silent-failure guard: the adaptive switcher only ever assigns codes 1
        # (super-Gaussian) or 4 (sub-Gaussian), or keeps the prior value (which
        # started as the ctor's all-1 fill and can therefore only ever BE 1 or
        # 4 itself). Currently unreachable -- a bug elsewhere would have to
        # write a different code first -- but an out-of-range code here would
        # otherwise fall through the _score/_log_pdf mx.where chain to a stale
        # GG density evaluated against a rho frozen by self.dorho, silently.
        bad = set(np.unique(new_pdtype).tolist()) - {1, 4}
        if bad:
            raise RuntimeError(
                f"_choose_pdfs produced pdtype code(s) outside {{1, 4}}: "
                f"{sorted(bad)} (adaptive-switcher invariant violated)."
            )
        self.pdtype = mx.array(new_pdtype.astype(np.int32))

    def _pdtype_from_kurtosis(self, kurt: np.ndarray, nsub: np.ndarray) -> np.ndarray:
        """Map per-source excess kurtosis to a density-family code (pure numpy;
        AMICATorchNG ``_pdtype_from_kurtosis``, core.py:1825-1852).

        Super-Gaussian (positive kurtosis) -> code 1; sub-Gaussian -> code 4.
        Only sources with a meaningful signal switch: a dead model
        (``nsub[h]==0`` => ``kurt==-3.0``, finite) or a numerically blown-up
        source (``kurt`` NaN, and ``NaN>0`` is False) would otherwise be
        silently assigned code 4 with no diagnostic, so those keep their prior
        ``self.pdtype`` and are logged -- mirroring the dead-model / non-finite
        guards in ``_update_parameters``. Split out from ``_choose_pdfs`` so
        the decision (including the sub-Gaussian branch, which real EEG rarely
        triggers) is unit-testable on constructed ``kurt``/``nsub`` arrays,
        matching AMICATorchNG's ``test_pdtype_from_kurtosis_decision``.
        """
        assert self.pdtype is not None
        prior = np.array(self.pdtype, dtype=np.int64)
        new_pdtype = np.where(kurt > 0.0, 1, 4)
        valid = np.isfinite(kurt) & (nsub[None, :] > 0.0)
        result = np.where(valid, new_pdtype, prior)
        if not bool(valid.all()):
            logger.warning(
                "Non-finite or zero-mass kurtosis for %d source/model pair(s) "
                "at iter %d; kept their prior pdtype (adaptive-switch guard).",
                int((~valid).sum()),
                self.iteration,
            )
        return result

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

        Note that a merge synchronizes only the mixture parameters routed
        through ``comp_list`` (``mu``/``alpha``/``beta``/``rho``); the
        per-source density *family* code ``pdtype`` is a separate array and is
        not synchronized (issue #265, matching AMICATorchNG core.py:2669-2673),
        so under the adaptive switcher (``pdftype=1``) a shared pair can still
        report different :meth:`get_pdftype` codes.

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
    # ------------------------------------------------------------------
    # Best-of-N restarts (issue #198); mirrors AMICATorchNG's implementation
    # ------------------------------------------------------------------
    # Everything a fit writes, and therefore everything a restart snapshot must
    # copy for the winning restart to be indistinguishable from a single fit
    # from that seed. Together with the invariants below this must account for
    # every ``self.x =`` in the fit path, which ``test_restart_policy.py``
    # enforces by parsing this module -- so a field added to a fit-path method
    # fails the suite until it is classified here.
    _RESTART_STATE_ATTRS = (
        # Fitted parameters and the derived per-iteration arrays ...
        "A", "W", "c", "mu", "alpha", "beta", "rho", "gm", "comp_list", "pdtype",
        "_comp_used_arr", "_lgamma_table", "_logdet_W", "_nd_arr",
        # ... the schedule/counters a fit mutates ...
        "iteration", "ll_history", "final_ll_", "stop_reason",
        "n_newton_fallbacks", "n_kurt_done",
        "lrate", "lrate_cap", "newtrate", "rholrate",
        # ... the tuned block size (do_opt_block re-times per restart) and the
        # seed the winning restart ran from.
        "block_size", "seed",
    )  # fmt: skip
    # Written by the fit path but identical across the restarts of one fit()
    # call, because they are functions of the data alone.
    _RESTART_INVARIANT_ATTRS = (
        "mean", "sphere", "_sphere_np", "sldet", "_sphere_pinv",
        "n_channels", "n_comps",
    )  # fmt: skip

    @staticmethod
    def _copy_state_value(value):
        """:func:`pamica.restarts.copy_state_value` plus the MLX case.

        ``mx.array(value)`` is a real copy (MLX arrays support item assignment,
        so aliasing one into a snapshot would not be safe), and it preserves
        dtype for the int/bool arrays here (``comp_list``, ``pdtype``,
        ``_comp_used_arr``).
        """
        if isinstance(value, mx.array):
            return mx.array(value)
        return restarts.copy_state_value(value)

    def _capture_restart_state(self) -> dict:
        """Independent copy of every attribute the fit path writes."""
        return {
            name: self._copy_state_value(getattr(self, name))
            for name in self._RESTART_STATE_ATTRS
        }

    def _apply_restart_state(self, state: dict) -> None:
        """Restore the state captured by :meth:`_capture_restart_state`."""
        for name, value in state.items():
            setattr(self, name, value)

    def fit(
        self, X: np.ndarray, max_iter: int = 100, verbose: bool = True
    ) -> "AMICAMLXNG":
        """Fit the model, running ``n_restarts`` fits and keeping the best.

        ``X`` is ``(n_channels, n_samples)``. With the default ``n_restarts=1``
        this is exactly :meth:`_fit_once` -- the restart machinery draws
        nothing, copies nothing and changes nothing, so the trajectory is
        bit-identical to a pre-issue-#198 fit. With ``n_restarts > 1`` the model
        is fit once per seed in ``restart_seeds`` (serially) and the returned
        model holds the highest-``final_ll_`` non-degenerate restart's complete
        state, exactly as a single fit from that seed would have left it.

        Records (index-aligned, always populated): ``restart_seeds_``,
        ``restart_lls_`` (NaN where a restart ended degenerate) and
        ``restart_stop_reasons_``; the winner is named in one INFO log line. A
        degenerate restart (``nan_ll``/``singular_ll``/``nan_params``) is
        excluded from selection but recorded; if every restart is degenerate the
        model is left holding the last one.
        """
        seeds = self._restart_seeds
        if len(seeds) == 1:
            # Single-restart path: seeds[0] IS self.seed unless the caller
            # passed an explicit one-element restart_seeds, so nothing here
            # perturbs the pre-#198 fit.
            self.seed = seeds[0]
            self._fit_once(X, max_iter=max_iter, verbose=verbose)
            self.restart_seeds_ = list(seeds)
            self.restart_lls_ = [
                float("nan") if self.final_ll_ is None else float(self.final_ll_)
            ]
            self.restart_stop_reasons_ = [self.stop_reason]
            return self

        lls: List[float] = []
        degenerate: List[bool] = []
        stop_reasons: List[Optional[str]] = []
        states: dict = {}
        for index, seed in enumerate(seeds):
            self.seed = seed
            try:
                self._fit_once(X, max_iter=max_iter, verbose=verbose)
            except RuntimeError as exc:
                # An ill-conditioned A makes _update_unmixing_matrices raise
                # (the issue #274 condition-number guard, which replaced MLX's
                # process abort with a catchable RuntimeError). That guard keeps
                # the process alive; this keeps the *search* alive, so one bad
                # basin cannot discard the restarts that already succeeded.
                # Mirrors AMICATorchNG._fit_restarts exactly, including catching
                # only RuntimeError so a ValueError from _fit_once's argument
                # checks still propagates.
                self.stop_reason = restarts.ERROR_STOP_REASON
                self.final_ll_ = float("nan")
                logger.warning(
                    "%s", restarts.error_message(index, len(seeds), seed, exc)
                )
            ll = float("nan") if self.final_ll_ is None else float(self.final_ll_)
            is_degenerate = self.stop_reason in self._DEGENERATE_STOP_REASONS
            lls.append(ll)
            degenerate.append(is_degenerate)
            stop_reasons.append(self.stop_reason)
            logger.info(
                "%s",
                restarts.progress_message(
                    index, len(seeds), seed, ll, self.stop_reason, is_degenerate
                ),
            )
            # Keep only the best state seen so far: one copy at a time.
            if restarts.select_best(lls, degenerate) == index:
                states = {index: self._capture_restart_state()}

        winner = restarts.select_best(lls, degenerate)
        if winner is None:
            logger.warning(
                "%s", restarts.all_degenerate_message(len(seeds), stop_reasons)
            )
        else:
            logger.info(
                "%s",
                restarts.winner_message(winner, len(seeds), seeds[winner], lls[winner]),
            )
            if winner != len(seeds) - 1:
                self._apply_restart_state(states[winner])

        self.restart_seeds_ = list(seeds)
        self.restart_lls_ = lls
        self.restart_stop_reasons_ = stop_reasons
        return self

    def _fit_once(
        self, X: np.ndarray, max_iter: int = 100, verbose: bool = True
    ) -> "AMICAMLXNG":
        """Run one fit (one initialization, one EM loop) -- what :meth:`fit`
        calls once per restart. ``X`` is ``(n_channels, n_samples)``.

        Under ``share_comps``, if a merge fires on the LAST iteration, the
        returned ``A``/``W``/``comp_list`` are already post-merge but
        ``final_ll_`` still reports the pre-merge log-likelihood; see that
        attribute's comment (issue #269).
        """
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

        # Block-size search (issue #232): after preprocessing and parameter
        # initialization, before the first EM iteration, so it times the real
        # data on the real device with the parameters the fit starts from. A
        # no-op when off, and its probes leave no state behind, so a fit with
        # the search off is byte-for-byte what it was before this existed.
        if self.do_opt_block:
            self._tune_block_size(X_t)

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
                self._logdet_W,
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
                # W and its log-determinant are DERIVED from A by
                # mx.linalg.inv/slogdet, so a non-finite value can reach the
                # caller while A itself is still finite -- and nothing else
                # would catch it on the LAST iteration, where there is no next
                # E-step to turn it into a nan_ll stop. The fit would then
                # return stop_reason="max_iter" with a healthy-looking final_ll_
                # (computed from the PREVIOUS iteration's W) and a silently
                # non-finite unmixing matrix, which is precisely the outcome
                # this guard exists to prevent. Verified by injection: with
                # W/_logdet_W excluded, that state passes every other entry
                # here.
                #
                # Defense in depth rather than a route known to be reachable:
                # the obvious candidate, a near-singular A whose inverse
                # overflows float32, is NOT reachable -- _update_unmixing_matrices
                # now raises RuntimeError on such an A before calling inv at all
                # (issue #274's condition-number guard), and before #274 it was
                # unreachable for a different reason (MLX's LU aborted the whole
                # process first, which this guard replaces with a catchable
                # error). Cheap enough to keep regardless -- both are already
                # materialized above.
                "W": self.W,
                "logdet_W": self._logdet_W,
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

            # Extended-Infomax adaptive PDF switch (Fortran do_choose_pdfs,
            # AMICATorchNG core.py:2030-2042). Runs on the
            # kurt_start/num_kurt/kurt_int schedule using the just-updated W;
            # the new per-source families take effect from the next E-step.
            # itf is the Fortran-style 1-indexed iteration. num_kurt=0 disables
            # switching (the family stays at its pdftype=1 super-Gaussian
            # init). Placed BEFORE the sharing hook below, matching
            # AMICATorchNG's source order -- component sharing does not
            # synchronize pdtype across merged columns (see
            # shared_components()), so running the switch first means a
            # just-merged pair still gets independently re-evaluated kurtosis
            # this same iteration. This ordering is documentation, not a
            # regression-tested contract: no test here pins the hooks' relative
            # order (both are no-ops for most configurations, and share_comps
            # x pdftype=1 has no bit-exact oracle either way to pin against), so
            # a future accidental swap would not be caught by the suite.
            if self.do_choose_pdfs and self.n_kurt_done < self.num_kurt:
                itf = it + 1
                if (
                    itf >= self.kurt_start
                    and (itf - self.kurt_start) % self.kurt_int == 0
                ):
                    self._choose_pdfs(X_t)
                    self.n_kurt_done += 1

            # Component sharing (Fortran identify_shared_comps schedule,
            # amica15.f90:1856): once per share_iter cycle from share_start,
            # merge near-collinear mixing columns across models using the
            # just-updated A. Fortran runs identify_shared_comps BEFORE
            # get_unmixing_matrices (amica15.f90:1858,1863), so rebuild W from
            # the merged comp_list -- otherwise the next E-step would read a
            # stale W (pre-merge comp_list) while indexing the densities by the
            # merged comp_list. No-op when share_comps is off or n_models == 1.
            #
            # This runs AFTER ``ll`` (this iteration's LL) was captured above,
            # so a merge on the final iteration lands in the returned
            # A/W/comp_list but not in the ``ll_history``/``final_ll_`` value
            # appended just below -- see final_ll_'s comment (issue #269).
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
                        if self.do_newton and it > self.newt_start:
                            # The Newton ceiling ratchets on the same maxdecs
                            # cadence as lrate_cap/rholrate (Fortran
                            # amica15.f90:1056-1077), so a run that keeps
                            # overshooting at newtrate anneals instead of
                            # oscillating there.
                            self.newtrate *= self.lratefact
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

            # Switching Newton on changes the step direction, so the decrease
            # counter accumulated during the natural-gradient phase no longer
            # describes the schedule now running: Fortran clears it on the
            # switch-on iteration (amica15.f90:1099-1102, AMICATorchNG
            # core.py:2218-2219).
            if self.do_newton and it == self.newt_start:
                numdecs = 0

            if leave:
                break

        if self.stop_reason in self._DEGENERATE_STOP_REASONS:
            self.final_ll_ = float("nan")
        else:
            self.final_ll_ = self.ll_history[-1] if self.ll_history else float("nan")
        return self

    def transform(self, X: np.ndarray, model_idx: int = 0) -> np.ndarray:
        """Apply the learned unmixing matrix to (new) data (issue #287, port of
        ``AMICATorchNG.transform``, torch_impl/core.py:2853-2869).

        Sources are ``S = W[model_idx]^T @ (sphere @ (X - mean) - c[:,
        model_idx])`` (issue #24 transpose convention, issue #27 per-model
        center) -- the exact composition ``_forward`` uses to build its ``b``
        activation, just laid out as ``(n_channels, n_samples)`` rather than
        ``_forward``'s ``(batch, n_channels)``: ``_forward`` computes ``b = (Xb
        - c[:, h]).T @ W[h]``, and ``S = W[h].T @ (Xb - c[:, h])`` is exactly
        ``b.T`` by the transpose identity ``(W^T v)^T = v^T W``. CAUTION: MLX's
        ``W`` is ``(n_models, n, n)`` (``_update_unmixing_matrices`` stacks on
        axis 0), NOT torch's ``(n, n, n_models)`` -- so the per-model slice
        here is ``W[model_idx]``, not torch's ``W[:, :, model_idx]``.

        Accepts any float ``np.ndarray``; computed in float32 (this backend's
        only precision) and returned as a float32 ``np.ndarray``.
        """
        if self.sphere is None or self.mean is None or self.W is None or self.c is None:
            raise RuntimeError(
                "AMICAMLXNG.transform() requires a fitted model; call fit() first."
            )
        self._check_model_idx(model_idx)
        X_arr = mx.array(np.ascontiguousarray(X).astype(np.float32))
        X_t = self.sphere @ (X_arr - self.mean)
        S = self.W[model_idx].T @ (X_t - self.c[:, model_idx : model_idx + 1])
        return np.array(S)

    # ------------------------------------------------------------------
    # Fitted-parameter metadata (issue #265; AMICATorchNG's #142 port)
    # ------------------------------------------------------------------
    def _check_model_idx(self, model_idx: int) -> None:
        """Validate a model index against the fitted ``n_models`` (AMICATorchNG
        ``_check_model_idx``, core.py:2338-2353). Raises a clear ``ValueError``
        (rejecting negatives, which MLX's negative indexing would otherwise turn
        into a silent wrong-model result) instead of an opaque array error."""
        if not isinstance(model_idx, (int, np.integer)):
            raise TypeError(
                f"model_idx must be an int, got {type(model_idx).__name__}."
            )
        if not (0 <= model_idx < self.n_models):
            raise ValueError(
                f"model_idx={model_idx} out of range for a {self.n_models}-model "
                f"fit (valid: 0..{self.n_models - 1})."
            )

    def get_pdftype(self, model_idx: int = 0) -> np.ndarray:
        """Per-source density-family code for model ``model_idx`` (AMICATorchNG
        ``get_pdftype``, core.py:2609-2627).

        One integer per source component (0-4; 0 generalized Gaussian, 1
        super-Gaussian cosh, 2 Gaussian, 3 logistic, 4 sub-Gaussian cosh). All
        sources share ``pdftype`` unless the adaptive switcher (``pdftype=1``)
        moved them individually (issue #265). ``rho`` does not describe the
        fitted density for codes 1-4 (it is frozen at ``rho0`` and only ever
        meaningful for the generalized-Gaussian family, code 0).

        Returns
        -------
        np.ndarray of int, shape (n_sources,)
        """
        if self.pdtype is None:
            raise RuntimeError(
                "AMICAMLXNG.get_pdftype() requires a fitted model; call fit() first."
            )
        self._check_model_idx(model_idx)
        codes = np.array(self.pdtype[:, model_idx], dtype=np.int64)
        # Silent-failure guard: an out-of-range stored code would otherwise
        # fall through the _score/_log_pdf mx.where chain to a stale GG
        # density (with rho frozen by self.dorho) with no diagnostic at all --
        # surface it here instead, at the point a caller reads it.
        bad = set(np.unique(codes).tolist()) - set(PDFTYPE_NAMES)
        if bad:
            raise RuntimeError(
                f"AMICAMLXNG.get_pdftype(): stored pdtype has code(s) outside "
                f"the valid set {sorted(PDFTYPE_NAMES)}: {sorted(bad)}."
            )
        return codes

    def get_mixing_matrix(self, model_idx: int = 0) -> np.ndarray:
        """True mixing matrix ``A_fort`` = (stored A)^T (issue #24 convention;
        issue #287 port of ``AMICATorchNG.get_mixing_matrix``, torch_impl/
        core.py:2872-2880)."""
        if self.A is None or self.comp_list is None:
            raise RuntimeError(
                "AMICAMLXNG.get_mixing_matrix() requires a fitted model; call "
                "fit() first."
            )
        self._check_model_idx(model_idx)
        return np.array(self.A[:, self.comp_list[:, model_idx]].T)

    def get_sensor_mixing_matrix(self, model_idx: int = 0) -> np.ndarray:
        """Mixing matrix mapped back to input-channel space (issue #287 port of
        ``AMICATorchNG.get_sensor_mixing_matrix``, torch_impl/core.py:
        2893-2917): ``pinv(sphere) @ A``, via :meth:`_pinv_sphere` -- the only
        correct back-map when rank reduction has left the sphere non-square
        (issue #223).
        """
        if self.sphere is None:
            raise RuntimeError(
                "AMICAMLXNG.get_sensor_mixing_matrix() requires a fitted "
                "model; call fit() first."
            )
        if self.A is None or self.comp_list is None:
            raise RuntimeError(
                "AMICAMLXNG.get_sensor_mixing_matrix() requires a fitted "
                "model; call fit() first."
            )
        self._check_model_idx(model_idx)
        A = np.array(self.A[:, self.comp_list[:, model_idx]].T, dtype=np.float64)
        return self._pinv_sphere() @ A

    def get_unmixing_matrix(self, model_idx: int = 0) -> np.ndarray:
        """True unmixing matrix ``W_fort`` = (stored W)^T (issue #24
        convention; issue #287 port of ``AMICATorchNG.get_unmixing_matrix``,
        torch_impl/core.py:2919-2927). MLX's ``W`` is model-major (``(n_models,
        n, n)``), so the per-model slice is ``W[model_idx]`` rather than
        torch's ``W[:, :, model_idx]``."""
        if self.W is None:
            raise RuntimeError(
                "AMICAMLXNG.get_unmixing_matrix() requires a fitted model; "
                "call fit() first."
            )
        self._check_model_idx(model_idx)
        return np.array(self.W[model_idx].T)

    def get_rho(self, model_idx: int = 0) -> np.ndarray:
        """Generalized-Gaussian shape parameter ``rho`` for model
        ``model_idx`` (issue #287 port of ``AMICATorchNG.get_rho``, torch_impl/
        core.py:3157-3182; issue #142).

        One value per (mixture component, source): ``rho == 2`` is Gaussian-
        shaped, ``rho == 1`` Laplacian, ``rho < 1`` heavier-tailed. Only the
        generalized-Gaussian family (``pdftype=0``) updates ``rho``; for every
        non-zero code (1-4) it stays frozen at ``rho0`` and does not describe
        the fitted density (see :meth:`get_pdftype`).

        Returns
        -------
        np.ndarray of float, shape (n_mix, n_sources)
        """
        if self.rho is None or self.comp_list is None:
            raise RuntimeError(
                "AMICAMLXNG.get_rho() requires a fitted model; call fit() first."
            )
        self._check_model_idx(model_idx)
        # Defense-in-depth, matching state_dict()'s isfinite sweep: a
        # degenerate multi-model fit can leave one model's rho non-finite
        # without the aggregate LL tripping nan_ll. Refuse rather than return
        # a silent NaN.
        if not bool(mx.all(mx.isfinite(self.rho)).item()):
            raise RuntimeError(
                "AMICAMLXNG.get_rho(): rho holds non-finite values (a "
                "degenerate fit); inspect stop_reason and refit."
            )
        idx = self.comp_list[:, model_idx]
        return np.array(self.rho[:, idx])

    # ------------------------------------------------------------------
    # Persistence (issue #287)
    # ------------------------------------------------------------------
    # Full fitted-parameter snapshot -- the same 12-name set as AMICATorchNG's
    # _PARAM_TENSORS (torch_impl/core.py:3372-3382): A/W/c/comp_list/mean/
    # sphere are what transform()/get_*matrix() read back; mu/alpha/beta/rho/
    # gm are the mixture-PDF EM state; pdtype is the per-source density-family
    # code (issue #265) -- a non-default pdftype model, or the adaptive
    # switcher's chosen 1/4 assignments, would otherwise silently revert to GG
    # on reload. comp_list and pdtype are integer arrays (dtype preserved on
    # load); the rest are float32.
    _PARAM_ARRAYS = (
        "A", "W", "c", "mu", "alpha", "beta", "rho", "gm",
        "comp_list", "mean", "sphere", "pdtype",
    )  # fmt: skip
    # Integer arrays in _PARAM_ARRAYS: their dtype is restored explicitly
    # rather than following the float32 default the rest take.
    _INT_PARAM_ARRAYS = ("comp_list", "pdtype")
    _INT_PARAM_DTYPES = {"comp_list": np.int64, "pdtype": np.int32}

    # This backend owns its own format_version, independent of AMICATorchNG's
    # (currently 3): the two payloads are never interchangeable (different
    # param layouts, no dtype/device fields here), so there is no reason for
    # the version numbers to track each other.
    _SAVE_FORMAT_VERSION = 1

    def state_dict(self) -> dict:
        """Serialize the fitted model to a plain, framework-agnostic dict.

        The returned dict has three parts: ``config`` (the constructor
        arguments needed to rebuild the object), ``params`` (the fitted
        arrays, as numpy), and ``extra`` (scalar/schedule state). Every value
        is a numpy array or a plain Python primitive, so the dict is JSON/
        ``.npz``-safe (see :meth:`save`). Rebuild with :meth:`from_state_dict`.

        Raises if the model is unfitted or degenerate (a fit that ended on a
        non-finite log-likelihood): a NaN model must not be persisted
        silently.
        """
        if self.A is None:
            raise RuntimeError(
                "AMICAMLXNG.state_dict() requires a fitted model; call fit() first."
            )
        if self.stop_reason in self._DEGENERATE_STOP_REASONS:
            raise RuntimeError(
                f"Refusing to serialize a degenerate model (stop_reason="
                f"{self.stop_reason!r}): fit() hit a non-finite log-likelihood "
                f"at iteration {self.iteration}. Fix the instability (lower "
                f"lrate, disable Newton, or check data conditioning) before "
                f"saving."
            )
        # Defense-in-depth: catch a non-finite parameter even if stop_reason
        # bookkeeping ever misses it (the codebase has known NaN-suppression
        # risks). isfinite on the integer comp_list/pdtype is trivially
        # all-True.
        nonfinite = [
            name
            for name in self._PARAM_ARRAYS
            if not bool(mx.all(mx.isfinite(getattr(self, name))).item())
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
            # block_size is the value the fit actually ran at -- which, under
            # do_opt_block, is the size the search chose rather than the one
            # the constructor was given (issue #232), so a reloaded model
            # reproduces the run it came from; the sweep bounds ride along so
            # a re-fit can search again if asked.
            "block_size": self.block_size,
            "do_opt_block": self.do_opt_block,
            "blk_min": self.blk_min,
            "blk_max": self.blk_max,
            "blk_step": self.blk_step,
            # lrate/newtrate/rholrate are annealed during fit; persist the
            # original constructor values (lrate0/newtrate0/rholrate0) and
            # restore the mutated ones from ``extra`` below.
            "lrate": self.lrate0,
            "minlrate": self.minlrate,
            "lratefact": self.lratefact,
            "maxdecs": self.maxdecs,
            "use_min_dll": self.use_min_dll,
            "min_dll": self.min_dll,
            "maxincs": self.maxincs,
            "use_grad_norm": self.use_grad_norm,
            "min_nd": self.min_nd,
            "newt_ramp": self.newt_ramp,
            "newt_start": self.newt_start,
            "newtrate": self.newtrate0,
            "do_newton": self.do_newton,
            "rho0": self.rho0,
            "minrho": self.minrho,
            "maxrho": self.maxrho,
            "rholrate": self.rholrate0,
            "rholratefact": self.rholratefact,
            # Density-family selection (issue #265): needed so a reloaded
            # model rebuilds with the right pdftype/dorho/do_choose_pdfs and
            # switch schedule instead of the GG default.
            "pdftype": self.pdftype,
            "kurt_start": self.kurt_start,
            "num_kurt": self.num_kurt,
            "kurt_int": self.kurt_int,
            "invsigmin": self.invsigmin,
            "invsigmax": self.invsigmax,
            "doscaling": self.doscaling,
            "scalestep": self.scalestep,
            # Component sharing (issue #263): persisted so a reloaded
            # multi-model run keeps its schedule; the merged comp_list itself
            # is in params.
            "share_comps": self.share_comps,
            "share_start": self.share_start,
            "share_iter": self.share_iter,
            "comp_thresh": self.comp_thresh,
            "do_mean": self.do_mean,
            "do_sphere": self.do_sphere,
            "do_approx_sphere": self.do_approx_sphere,
            "mineig": self.mineig,
            "mineig_rel": self.mineig_rel,
            "seed": self.seed,
            # Best-of-N restarts (issue #198). Persisted so a reloaded model
            # reconstructs its exact configuration; the restart the fit
            # actually kept is in ``extra`` below.
            "n_restarts": self.n_restarts,
            "restart_seeds": self.restart_seeds,
        }
        params = {name: np.array(getattr(self, name)) for name in self._PARAM_ARRAYS}
        extra = {
            "sldet": float(self.sldet),
            "iteration": int(self.iteration),
            "ll_history": [float(v) for v in self.ll_history],
            "final_ll": None if self.final_ll_ is None else float(self.final_ll_),
            "stop_reason": self.stop_reason,
            "n_kurt_done": int(self.n_kurt_done),
            "n_newton_fallbacks": int(self.n_newton_fallbacks),
            "lrate": float(self.lrate),
            "lrate_cap": float(self.lrate_cap),
            "newtrate": float(self.newtrate),
            "rholrate": float(self.rholrate),
            # Per-restart records (issue #198): which seeds ran, what each
            # returned, and why each stopped.
            "restart_seeds_": list(self.restart_seeds_),
            "restart_lls_": [float(v) for v in self.restart_lls_],
            "restart_stop_reasons_": list(self.restart_stop_reasons_),
        }
        return {
            "format_version": self._SAVE_FORMAT_VERSION,
            "config": config,
            "params": params,
            "extra": extra,
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "AMICAMLXNG":
        """Rebuild a fitted :class:`AMICAMLXNG` from :meth:`state_dict` output.

        Unlike ``AMICATorchNG.from_state_dict`` there is no ``device``
        argument: this backend always runs on ``mx.default_device()``.
        """
        version = state.get("format_version")
        if version != cls._SAVE_FORMAT_VERSION:
            raise ValueError(
                f"unsupported AMICAMLXNG state format_version: {version!r} "
                f"(expected {cls._SAVE_FORMAT_VERSION})"
            )
        for section in ("config", "params", "extra"):
            if section not in state:
                raise ValueError(
                    f"malformed AMICAMLXNG state: missing {section!r} section "
                    f"(format_version={version}); the payload may be truncated."
                )
        config = dict(state["config"])
        obj = cls(**config)
        obj._load_params(state)
        return obj

    def _load_params(self, state: dict) -> None:
        """Restore fitted arrays/scalars from :meth:`state_dict` output onto
        this instance."""
        params = state["params"]
        missing = [name for name in self._PARAM_ARRAYS if name not in params]
        if missing:
            raise ValueError(f"malformed AMICAMLXNG state: missing params {missing}")
        # Guard against config/params drift: A and comp_list must match the
        # dimensions the constructor just derived, or transform()/the E-step
        # would fail later with a confusing matmul error far from load().
        A = np.asarray(params["A"])
        if A.shape != (self.n_channels, self.n_comps):
            raise ValueError(
                f"restored A has shape {A.shape}, expected "
                f"{(self.n_channels, self.n_comps)} for n_channels="
                f"{self.n_channels}, n_models={self.n_models}"
            )
        comp_list = np.asarray(params["comp_list"])
        if comp_list.shape != (self.n_channels, self.n_models):
            raise ValueError(
                f"restored comp_list has shape {comp_list.shape}, expected "
                f"{(self.n_channels, self.n_models)}"
            )
        for name in self._PARAM_ARRAYS:
            value = np.asarray(params[name])
            dtype = self._INT_PARAM_DTYPES.get(name, np.float32)
            setattr(self, name, mx.array(value.astype(dtype)))

        # Rebuild the MLX-only per-iteration caches (module docstring):
        # unlike AMICATorchNG, which recomputes log|det W| and
        # lgamma(1+1/rho) inline on every call, this backend hoists them to
        # once-per-iteration cached arrays. A loaded model has no fit history
        # to hoist them from, so they are rebuilt here from the restored
        # params -- before this returns, transform()/_forward()/comp_used and
        # every get_* accessor must work exactly as they would mid-fit.
        self._comp_used_arr = mx.array(
            np.isin(np.arange(self.n_comps), np.unique(np.array(self.comp_list)))
        )
        self._refresh_lgamma_table()
        assert self.W is not None  # just set by the loop above
        logdets = [
            mx.linalg.slogdet(self.W[h], stream=_CPU)[1] for h in range(self.n_models)
        ]
        self._logdet_W = mx.stack(logdets)
        # sphere was just replaced, so any cached back-map describes the old
        # one. _sphere_np backs _pinv_sphere (get_sensor_mixing_matrix,
        # _identify_shared_comps) at float64 precision during a live fit, but
        # only the float32 ``sphere`` is a persisted param (the fixed
        # 12-name set above) -- so a reloaded model's _sphere_np is the
        # float32 sphere upcast to float64, not the higher-precision value
        # _preprocess originally computed. This cannot affect transform()
        # (which reads self.sphere directly, so it stays bit-identical
        # pre/post round trip): only get_sensor_mixing_matrix() on a
        # reloaded model carries this small extra rounding.
        self._sphere_np = np.array(self.sphere, dtype=np.float64)
        self._sphere_pinv = None

        extra = state["extra"]
        self.sldet = extra["sldet"]
        self.iteration = extra["iteration"]
        self.ll_history = list(extra["ll_history"])
        self.final_ll_ = extra["final_ll"]
        self.stop_reason = extra["stop_reason"]
        self.n_kurt_done = extra["n_kurt_done"]
        self.n_newton_fallbacks = extra["n_newton_fallbacks"]
        self.lrate = extra["lrate"]
        self.lrate_cap = extra["lrate_cap"]
        self.newtrate = extra["newtrate"]
        self.rholrate = extra["rholrate"]
        self.restart_seeds_ = list(extra.get("restart_seeds_", []))
        self.restart_lls_ = list(extra.get("restart_lls_", []))
        self.restart_stop_reasons_ = list(extra.get("restart_stop_reasons_", []))

    def save(self, filepath: str) -> None:
        """Persist the fitted model to ``filepath`` as a single ``.npz``
        (issue #287).

        Device- and framework-agnostic by construction: ``config``/``extra``
        are embedded as JSON-encoded 0-d string arrays and ``params`` as
        native numpy arrays, all written by ``np.savez_compressed`` -- no
        torch coupling, no pickle. Reload with :meth:`load`. Raises the same
        refusal guards as :meth:`state_dict` (unfitted, degenerate, or
        non-finite parameters).
        """
        state = self.state_dict()
        np.savez_compressed(
            filepath,
            format_version=state["format_version"],
            config=json.dumps(state["config"]),
            extra=json.dumps(state["extra"]),
            **state["params"],
        )

    @classmethod
    def load(cls, filepath: str) -> "AMICAMLXNG":
        """Rebuild a fitted :class:`AMICAMLXNG` from a file written by
        :meth:`save`.

        Wrong ``format_version``, a missing ``config``/``extra``/``params``
        section, or a truncated archive missing one of the 12 param arrays
        each raise a named ``ValueError`` naming what is missing, rather than
        loading a silently partial model.
        """
        with np.load(filepath, allow_pickle=False) as data:
            files = set(data.files)
            for section in ("format_version", "config", "extra"):
                if section not in files:
                    raise ValueError(
                        f"malformed AMICAMLXNG save file {filepath!r}: missing "
                        f"{section!r} (the file may be truncated or corrupted)."
                    )
            version = int(data["format_version"])
            if version != cls._SAVE_FORMAT_VERSION:
                raise ValueError(
                    f"unsupported AMICAMLXNG save format_version: {version!r} "
                    f"(expected {cls._SAVE_FORMAT_VERSION})"
                )
            config = json.loads(data["config"].item())
            extra = json.loads(data["extra"].item())
            missing_params = [name for name in cls._PARAM_ARRAYS if name not in files]
            if missing_params:
                raise ValueError(
                    f"malformed AMICAMLXNG save file {filepath!r}: missing "
                    f"params {missing_params} (the file may be truncated or "
                    f"corrupted)."
                )
            params = {name: np.array(data[name]) for name in cls._PARAM_ARRAYS}
        state = {
            "format_version": version,
            "config": config,
            "params": params,
            "extra": extra,
        }
        return cls.from_state_dict(state)
