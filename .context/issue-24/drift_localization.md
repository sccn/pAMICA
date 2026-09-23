# Issue #24 drift localization: which M-step term drifts from Fortran

**Date:** 2026-07-02. Follow-up to `findings.md` (which proved the -3.46 gap is a residual
per-iteration M-step bug, not init-basin). This doc localizes the drifting term(s).

---

## RESOLVED 2026-07-03 -- root cause is the natural-gradient A-update (transposed / wrong side)

The `-3.46` descent is a **transpose + multiply-side error in the natural-gradient A-update**,
proven to machine precision and fixed. See `root_cause_Aupdate.py` (self-contained repro).

- The prototype stores `A` as **Fortran's A^T** (its "true unmixing = inv(A).T" convention; its
  consequence, components are ROWS of each stored block, is ADR 0006, issue #333). So
  Fortran's step `A_fort <- A_fort - lr*A_fort@(I - G/dgm)` (G = g^T b) must become, transposed,
  `A <- A - lr*(I - G^T/dgm) @ A` (LEFT-multiply, TRANSPOSED direction). The prototype instead does
  `A <- A - lr*A @ (I - G/dgm)` (right-multiply, untransposed) -- wrong on **both** counts.
- Of the four candidate forms at k=0 (matched Fortran sphere), only the left/transposed form matches
  Fortran's A_new to **7.8e-16**; the current form is off by **1.6e-3** (`root_cause_Aupdate.py` [3]).
- With the fix, the real pinned-lrate fit **ASCENDS to -3.4265 / corr 0.648**, matching Fortran's
  pinned-NG endpoint (-3.4269 / 0.645), instead of descending to -3.4974 / 0.506.
- Invisible at the fixed point because `G = g^T b ~ (g^T b)^T = G^T` when the natural gradient -> 0,
  so `corrected_mstep_prototype.py::fixed_point_test` passed while the free-running fit descended.

**PORTED TO PRODUCTION & VALIDATED (2026-07-03).** All fixes are in the shipped backends
`torch_impl/amica_torch_ng.py` (`AMICATorchNG`) and `pamica.py` (`AMICA_NumPy`): the A-update
transpose, exact-EM mu/beta (`fp` not `dpdf`), the digamma rho update (Bugs 1+2), the symmetric-ZCA
sphere (cov/N), the Newton-path orientation (`A -= lrate*H^T @ A`), the W/A output transpose
(get_*_matrix / transform), and the NumPy Jacobian LL (`sldet` + `logsumexp`, previously positive).
Result: `AMICATorchNG` Newton fit ascends to LL -3.4017 / corr 0.997 (Fortran -3.4018 / 0.998),
0 Newton fallbacks; NumPy trajectory is bit-identical (i100 LL -3.4110 both). The
`test_end_to_end_correlation_vs_fortran` gate (>0.95) now PASSES (xfail removed); the NG<->NumPy
sufficient-stat parity tests pass to 1e-8.

**The two theories below are SUPERSEDED / an artifact** (kept for the record):
- "Bug 3 mu/beta denominator" is **DISPROVEN as a formula bug.** The mu/beta exact-EM update is
  bit-exact with Fortran (~1e-13) once the *same sphere* is used (`root_cause_Aupdate.py` [2]). The
  4.5e-3 "drift" was a **sphere-comparison artifact**: Python spheres with `torch.cov` (cov/(N-1)),
  Fortran uses cov/N, a pure scalar `sqrt(N/(N-1))=1.0000164` apart, amplified by the singular
  `sum(u*rho*|y|^(rho-2))` denominator (worst in outer mixtures -> the "scales with |mu|" fingerprint).
  That sphere difference is **benign** in a consistent run (fixing it moves the endpoint by 2e-4).
- Bugs 1+2 (rho) are still real and confirmed bit-exact; they are necessary but not sufficient.

## Method

Teacher-forced per-iteration diff (`localize_drift.py`). Fortran's true NG trajectory
(`do_newton=0`, `lrate=0.05`) is obtained by running the binary with `max_iter=1..K` from the
Python init (`do_history` is broken -- it fails to `mkdir out/history` and aborts, so we re-run
incrementally). For each k we seed `CorrectedNG` from Fortran's state at iter k, run ONE M-step,
and compare each parameter to Fortran's state at iter k+1. Because every comparison restarts
from a known-good Fortran state, errors do not compound; the per-param one-step residual
localizes the drift.

Two payoff harnesses confirm each fix end-to-end with lrate pinned at 0.05 (the only regime the
fit-loop ratchet does not mask): a rho-only fix and a full rho fix.

## What drifts, and what does not

| param | one-step residual (k=0, from identical init) | verdict |
| ----- | -------------------------------------------- | ------- |
| W / A (unmixing) | 1.6e-6 defect | FAITHFUL (natural-gradient A update is correct) |
| alpha | ~0 | faithful |
| c     | ~3.6e-16 (Fortran c is machine zero) | faithful; the omitted c update is NOT the drift |
| **rho** | 0.149 (37.5% of the step) | **BUG 1+2 (below)** |
| **mu**  | 4.5e-3 (~1.3% of the step), grows with k | **BUG 3 (dominant, open)** |
| **beta**| grows with k, exceeds the step by ~k=8 | **BUG 3 (open)** |

## CONFIRMED BUGS

### Bug 1 -- rho numerator is missing a factor of `rho` (proven)

Fortran (`amica17.f90:1560-1578`, non-MKL branch) builds the rho numerator as:
`tmpy=|y|` -> `logab=ln|y|` -> `tmpy=|y|^rho` -> **`logab=ln(|y|^rho)=rho*ln|y|`** ->
`drho_numer = sum(u * tmpy * logab) = rho * sum(u*|y|^rho*ln|y|)`.

The prototype computes `sum(u*|y|^rho*ln|y|)` -- **missing the leading `rho`**.

Proof (`probe_rho.py`): seed the init, run one Fortran step. Python's iter-1 rho is off by
37.5% of the step without the factor, and matches Fortran to **1.7e-5** with it.

### Bug 2 -- rho per-component mask zeros the numerator at the clamp boundary (confirmed)

The prototype masks `mask = (rho != 1.0) & (rho != 2.0)` and zeros the ENTIRE numerator term for
any component at rho=1.0 or 2.0. Fortran has no such per-component skip; it only guards
per-SAMPLE underflow (`where(|y|^rho < eps) logab = 0`, `:1570`).

Fingerprint: with Bug 1 fixed, the teacher-forced `rho_abs` is at float precision (~7e-6) for
k=0-3, then jumps to **exactly 5.00e-2 = rholrate** at k>=4 -- precisely when a component's rho
first hits the `minrho=1.0` clamp. Fixing the mask (compute the term for all components, use only
the per-sample underflow guard) removes that jump.

### But Bugs 1+2 are NOT the dominant descent cause

End-to-end with lrate pinned at 0.05 (`verify_rhofix.py`, `verify_fullrho.py`):

```
rho fix          | pinned-NG endpoint (Fortran = -3.4269, ascending)
none             | -3.4974   (descends)
factor only      | -3.4972
factor + no mask | -3.4974
```

Fixing rho barely moves the endpoint. The descent is driven by BUG 3.

## ~~OPEN: Bug 3 -- mu/beta exact-EM denominator drift (dominant)~~ DISPROVEN 2026-07-03

**This section is WRONG (kept for the record).** See the RESOLVED banner at the top. The mu/beta
update is bit-exact with Fortran once the *same* sphere is used; the residual below was the
`torch.cov` (N-1) vs Fortran (N) sphere mismatch amplified by the singular denominator, and the
"the A/W update is faithful (defect 6e-5)" claim was an artifact of the row-normalized `rowdefect`
metric (the true A error was 1.6e-3 -- the real bug). Original (incorrect) reasoning:

The natural-gradient A/W update is faithful (defect 6e-5) and the mixture-summed score is right,
but the per-mixture **mu** and **beta** updates carry a small systematic residual that compounds
into the descent. Characterization (`probe_mu.py`):

- Present with AND without `doscaling` (4.47e-3 vs 4.57e-3) -> it is the exact-EM update, not the
  column rescale.
- Concentrated in the OUTER mixtures (mu ~ +/-1: residual 4.5e-3; the center mixture mu~0:
  residual 2e-5). Scales with |mu|.
- Too large (4.5e-3) to be float64 summation-order noise (~1e-8), so it is a real
  formula/definitional difference, not numerics.
- The mu update is `mu += dmu_numer/dmu_denom`, `dmu_denom = sbeta*sum(ufp/y)` (`amica17.f90:1537,
  1978`). The numerator `sum(ufp)` is per-mixture and NOT independently verified (W parity only
  confirms the mixture-SUMMED `g`), so the residual could live in the per-mixture numerator or in
  the singular `sum(ufp/y)` denominator (dominated by small-y `~|y|^(rho-2)` terms).

Next step to close it: seed from a Fortran state where rho has spread, dump Python's per-mixture
`dmu_numer` and `dmu_denom`, and compare against a hand-computed Fortran reference for a single
component (the same tactic that nailed Bug 1). Check the `rho<=2` vs `>2` denominator branch and
any small-y handling.

## FLAGGED: reference-source oddity (not our bug, note for anyone recompiling)

`amica17.f90:1465` (non-MKL `fp`): `vrda_exp((rho - dble(0.0))*tmpvec, ...)` -> `|y|^rho`, whereas
the MKL branch `:1462` uses `(rho - dble(1.0))` -> `|y|^(rho-1)` (the mathematically correct GG
score `fp = rho*sign(y)*|y|^(rho-1)`). The `amica15mac` binary behaves per the correct exponent
(our `_score = rho*sign(y)*|y|^(rho-1)` keeps W parity at 6e-5), so the shipped binary is fine,
but a non-MKL recompile from this source would compute the wrong score. Flag before rebuilding.

## BUGS TO FIX (when porting the corrected M-step into production)

- [ ] **Bug A (A-update transpose/side, ROOT CAUSE)** -- `corrected_mstep_prototype.py:195-197`:
      the natural-gradient step is `A_cols - lrate*(A_cols @ dirs[h])` with `dirs[h] = I - dWtmp/dgm`.
      It must be `A_cols - lrate*((I - dWtmp.T/dgm) @ A_cols)` (transpose `dWtmp`, LEFT-multiply),
      because `A` is stored as Fortran's `A^T`. Proven machine-exact; flips the fit from descending
      (-3.4974) to ascending (-3.4265, corr 0.648 vs Fortran 0.645). Mirror in shipped
      `amica_torch_ng.py::_update_parameters` and `pamica.py`. **Also audit the Newton path**: it
      builds the Newton direction from the same untransposed `dA_h` and right-multiplies
      (`corrected_mstep_prototype.py:171-197`), so it needs the matching orientation fix + the
      `dA(i,k)`/`dA(k,i)` term order (Fortran `:1825`) before Newton is wired in.
- [ ] **Bug 1 (rho factor)** -- `corrected_mstep_prototype.py:102-110`: `drho_n` term must be
      `rho_h * |y|^rho * ln|y|` (add the leading `rho`). Mirror in the shipped
      `amica_torch_ng.py` rho path and `pamica.py`. (Confirmed bit-exact with the fix.)
- [ ] **Bug 2 (rho mask)** -- `corrected_mstep_prototype.py:104-108`: drop the per-component
      `(rho!=1)&(rho!=2)` mask; keep only a per-sample underflow guard (Fortran `:1570`).
- [ ] **Sphere (cosmetic parity)** -- `corrected_mstep_prototype.py::_preprocess` (and shipped
      `amica_torch_ng.py`): use `torch.cov(Xc, correction=0)` (cov/N, Fortran-matched) not the
      default cov/(N-1). Benign for convergence (endpoint moves ~2e-4) but removes a 5e-6 sphere
      mismatch; do it so validation harnesses compare like-for-like.
- [ ] **Flag (reference)** -- `amica17.f90:1465` `(rho - dble(0.0))` should be `(rho - dble(1.0))`
      to match the MKL branch; only matters for a non-MKL rebuild.

## Reproduce

```
# ROOT CAUSE (sphere + mu/beta bit-exact + A-update forms; add --fit for the 200-iter parity):
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python .context/issue-24/root_cause_Aupdate.py [--fit]

# teacher-forced per-param residual (add --fixrho to apply Bug 1) -- historical, sphere-contaminated:
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python .context/issue-24/localize_drift.py [run_dir] [--fixrho]
```
