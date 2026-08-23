# ADR 0002: Adaptive-PDF selection via the amica15 density families

**Status:** accepted
**Date:** 2026-07-06
**Owner:** Seyed Yahya Shirazi

## Context

Issue #26 asked `AMICATorchNG` to gain the per-source density-family selection that the
GG-only PyTorch backend lacked. An earlier investigation concluded no runnable oracle
existed, because it read `amica17.f90` (the repo's reference source, which is GG-only) and
the default `input.param` runs `pdftype 0`. That was the wrong source: the validation
binary is `amica15mac`, whose source `amica15.f90` implements five source-density families
selected per-source by `pdtype(i,h)`. The families are therefore a genuine, bit-comparable
parity target, not a beyond-parity feature. The user requires that defaults stay identical
to the Fortran design (GG remains the default and bit-for-bit unchanged).

## Decision

Port the five `amica15.f90` families into `AMICATorchNG`, exposed through Fortran's exact
`pdftype` interface: 0 generalized Gaussian (default), 2 Gaussian, 3 logistic, 4
sub-Gaussian cosh+, and `pdftype=1` = the extended-Infomax adaptive switcher (Fortran's
`do_choose_pdfs` trigger), which flips each source between the super-Gaussian (code 1) and
sub-Gaussian (code 4) cosh densities by kurtosis sign on the `kurt_start`/`num_kurt`/
`kurt_int` schedule. `rho` is frozen for every non-GG family (`amica15.f90:3704`), and the
single-component families 1/4 require `n_mix=1`. The ground-truth `amica15.f90`/
`amica15_header.f90` are copied into `pamica/`.

## Consequences

- Fixed families (0/2/3/4/1) are bit-exact against the literal Fortran `z0`/`fp` (~1e-15)
  and converge to `amica15mac` within ~0.005 LL with the optimizer matched (Newton on).
- `pdftype=0` runs are byte-for-byte the pre-#26 implementation (validated `_pdtype_h`
  returns `None`, and the existing NG parity suite is unchanged).
- The dynamic `do_choose_pdfs` switch is dead code even in `amica15.f90` (`m2sum`/`m4sum`
  are never accumulated), so the auto-switcher has no bit-exact oracle; it is validated by
  real-data log-likelihood (finite, non-decreasing) instead. This is documented, not hidden.
- The cosh families' Newton curvature is not always positive-definite, so NG falls back to
  natural gradient on those iterations exactly as Fortran does; LL still reaches parity.

## Alternatives considered

- **Fortran-faithful `do_choose_pdfs` reconstructed from amica17:** rejected once amica15 was
  found to be the binary's source; amica17 only declares the arrays and the switch body is
  absent in both, so there is nothing faithful to reconstruct beyond the family densities.
- **Reuse the NumPy `amica_pdf.py` family set (Laplace/Student-t/logistic/GMM):** rejected;
  its numbering and formulas do not match `amica15.f90`, and it is used by no fit loop, so it
  is not an oracle. The issue's "Laplace/Student-t" phrasing traces to this unvalidated code.
- **Mixture-only families (0/2/3), defer cosh 1/4:** rejected; the user chose the full scope
  including the auto-switcher, which needs the single-component cosh densities.

## Receipts

- `pamica/amica15.f90` select-cases at :1295 (likelihood) / :1467 (score); `dorho=.false.`
  at :3704; `do_choose_pdfs` at :612; the `m2sum`/`m4sum` moment buffers are allocated/zeroed
  (:608-609) but never accumulated, confirming the dynamic switch is dead code in the binary.
- The extended-Infomax intent comes from the upstream AMICA MATLAB wrapper `runamica15.m`
  (sccn/amica), which documents the schedule parameters verbatim (not copied into this repo):
  ```
  %   kurt_start          for ext. infomax, iter to start kurtosis calc, def=3
  %   num_kurt            for ext. infomax, number of kurtosis calc, def=5
  %   kurt_int            for ext. infomax, iteration interval between calc, def=1
  ```
  and defaults `pdftype=0; kurt_start=3; num_kurt=5; kurt_int=1;`. The super/sub-Gaussian
  scores `y +/- tanh(y)` (amica15 codes 1/4) are the classic extended-Infomax nonlinearities.
- `pamica/torch_impl/amica_torch_ng.py`: `_log_pdf_and_deriv`, `_score`, `_choose_pdfs`.
- `pamica/tests/torch_tests/test_ng_pdf_families.py` (formula parity + real-data + opt-in
  binary integration behind `AMICA_RUN_FORTRAN=1`).
