# ADR 0004: Rank-deficient input handling and a relative eigenvalue floor

**Status:** accepted
**Date:** 2026-08-15
**Owner:** Seyed Yahya Shirazi

## Context

Rank-deficient input is ordinary in EEG/MEG, not exotic: Maxwell filtering leaves
306-channel Elekta data at numerical rank ~70, average referencing removes exactly one
dimension, and channel interpolation removes one per interpolated channel. The Fortran
reference handles this and pamica did not, which issue #221 surfaced from a real MEG
analysis.

Fortran detects the rank, sizes the model to it, and keeps a pseudo-inverse to map
components back to sensors:

```fortran
numeigs = min(pcakeep, count(eigs > mineig))   ! amica15.f90:395, mineig = 1e-15
nw = numeigs                                   ! amica15.f90:545
allocate(Spinv(nx,numeigs))                    ! amica15.f90:550-560
```

pamica had none of the three. The symmetric ZCA sphere therefore inverted
`sqrt(lambda ~ 0)` and the fit died at `nan_ll` on iteration 0 (issue #223).

`mineig` is an **absolute** floor on covariance eigenvalues, which makes it
unit-dependent. Two consequences measured on the bundled sample EEG:

- MEG in Tesla has eigenvalues ~1e-26, entirely below 1e-15. Fortran computes
  `numeigs = 0` and proceeds.
- Average-referenced EEG lands at `lambda_min/lambda_max = 8.5e-17`, so whether the
  zero dimension is detected depends on the recording's absolute scale. It is decided
  by luck.

## Decision

Port all three pieces of the Fortran mechanism, and **default to a relative floor**:
`mineig_rel = 1e-12`, applied as `mineig_rel * largest_eigenvalue`. Setting
`mineig_rel=None` restores Fortran's absolute-only behavior exactly.

The rank decision lives in `pamica/rank.py` and is called by all three array backends,
so they cannot drift (`.rules/backend_parity.md`).

`mineig_rel` **replaces** the absolute floor rather than combining with it. Taking the
larger of the two was tried first and is wrong: a relative floor for Tesla-scale data is
~1e-35 in absolute terms, so `max()` silently discards it and reproduces the bug.

## Consequences

**Parity is unaffected for well-conditioned data.** Real EEG has
`lambda_min/lambda_max ~ 5e-4`, eight orders above the relative floor, so nothing is
reduced and results are bit-identical (verified against the pre-change implementation
across single-model, Newton, and two-model fits).

**Results change for rank-deficient data, in the intended direction.** On real EEG
projected to rank 20:

| threshold | rank found | reconstruction error |
|---|---|---|
| `mineig=1e-15` (Fortran) | 24 | 1.98e-09 |
| `mineig_rel=1e-12` (default) | 20 | 7.52e-15 |

**This is a deliberate divergence from the reference**, the first where pamica's default
is not what Fortran does. It is recorded in `docs/guides/amica-differences.md` so it
cannot be mistaken for a parity defect.

**Average-referenced EEG now yields n-1 components rather than n.** Correct, but a
visible behavior change for existing users, since the n-th component was previously
numerical noise from a singular sphere.

**A new obligation:** `get_sensor_mixing_matrix()` is now the way to obtain scalp maps.
`get_mixing_matrix()` still returns the sphered-space `A`, which is not projectable to
sensors when the sphere is non-square.

## Alternatives considered

- **Absolute floor only, matching Fortran exactly.** Loses on MEG (rank 0) and makes
  average-reference detection scale-dependent. Rejected: bug-compatibility with the
  reference is not the goal, numerical agreement on valid data is.
- **`max(mineig, mineig_rel * lambda_max)`.** Implemented first, then measured: the
  absolute term dominates for small-scale data and reproduces the failure the relative
  floor exists to prevent.
- **Reduce inside the wrapper rather than the backends.** Would leave `AMICATorchNG`
  and the other backends broken when used directly, and duplicate the policy per entry
  point.
- **`get_mixing_matrix()` returns sensor space when reduced.** Rejected: silently
  switching what a method returns based on data conditioning is worse than a second,
  explicitly-named method.

## Receipts

- Issue #221 (MEG report), issue #223 (parity gap), PR #224
- `amica15.f90:395,545,483-490,550-560`; `amica15_header.f90:66`
- `pamica/rank.py`, `pamica/tests/test_rank_policy.py`
- `docs/guides/amica-differences.md`
