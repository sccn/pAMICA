# Multi-model NG log-likelihood: best-iterate safeguard (issue #51)

> **Historical record.** These figures predate epic #324, which changed every
> backend's default trajectory. The re-measured `keep_best` ensemble (seeded,
> against the pinned v0.3.3 native binary) is in `.context/issue-351/findings.md`
> (script `.context/issue-351/keep_best_ensemble.py`) and in ADR 0003's
> "Re-measured after epic #324" section.

**Bottom line.** The multi-model NG log-likelihood was ~0.02 lower and ~13x more
variable than Fortran because `AMICATorchNG.fit` returned the *last* EM iterate
under a deliberately non-monotone learning-rate schedule. Returning the *best*
iterate (`keep_best`, default on) removes the variance pathology; the remaining
mean gap is convergence speed, not a worse optimum -- with more iterations NG
reaches Fortran's exact solution.

## Root cause: return-last, not a bad basin

An NG-only sweep (20 seeds, `n_models=2`, 100 iters, real sample EEG) reproduced
the #51 finding (mean -3.3738, sd 0.040) and localized the variance to a **late
overshoot**, not a wrong basin:

- The single variance-driving seed (#3) climbed to LL **-3.3573** (dead in the
  pack) by iter 97, then **crashed to -3.5452** in its final two iterations after
  Newton went non-positive-definite and fell back to the natural gradient; the
  lrate ramp re-inflated the step and the run ended mid-crash.
- 9 of 20 "good" seeds also ended a small amount below their own peak.

The lrate schedule anneals only *after* an LL decrease (`amica15.f90:1038-1058`,
mirrored in NG), so the last iterate is not guaranteed to be the best. `fit`
returned the last iterate.

## Fix: return the best iterate

`AMICATorchNG` tracks the highest-LL iterate and restores it when the run ends
more than `_KEEP_BEST_TOL` (1e-9) below that peak. `keep_best=True` by default;
`final_ll_` reports the returned iterate's LL, while `ll_history` stays the true
trajectory. Inactive under `do_reject` (the good-sample set, hence the LL
normalization, changes across iterations). A monotone single-model fit has
best == last, so no restore fires and **issue #24 parity stays bit-exact**
(verified: max parameter difference 0.0 with keep_best on vs off).

## Results (real sample EEG, Fortran binary + NG, NO MOCK)

`ensemble_ll.py`, N=20 each, `n_models=2`, matched schedule:

| max_iter=100 | mean LL | sd | mean gap | sd ratio | KS p | TOST(±0.01) |
|---|---:|---:|---:|---:|---:|---|
| Fortran | -3.3541 | 0.0031 | -- | -- | -- | -- |
| NG return-last | -3.3738 | 0.0399 | -0.0197 | 12.7x | 9.5e-6 | inconclusive |
| **NG keep_best** | **-3.3634** | **0.0064** | **-0.0093** | **2.0x** | 5.6e-5 | inconclusive |

keep_best cuts the variance from **12.7x -> 2.0x** Fortran's sd (the headline
"~13x more variable" defect) and halves the mean gap. See `ll_before_after.png`.

## The residual is convergence speed, not a worse optimum

The ~0.009 mean gap that remains at 100 iters is *iteration budget*, not a wrong
term. NG keep_best mean LL vs iteration budget (8 seeds):

| max_iter | NG mean | NG sd | Fortran mean (100 it) |
|---:|---:|---:|---:|
| 100 | -3.3639 | 0.0076 | -3.3541 |
| **200** | **-3.3541** | 0.0050 | -3.3541 |
| 300 | -3.3523 | 0.0040 | -3.3541 |

At 200 iterations NG reaches Fortran's **exact** mean (-3.3541); by 300 it slightly
exceeds it. NG's per-iteration progress is ~2x slower than Fortran's, but it
converges to the same optimum -- as expected from the M-step being bit-exact vs
Fortran (#27). This is optimizer efficiency, not correctness: the reachable
solution is identical.

## Acceptance

1. **Variance pathology removed** at matched budget (12.7x -> 2.0x). *(Held.)*
2. **Same optimum**: NG reaches Fortran's exact mean LL with adequate iterations
   (200-iter mean == -3.3541). *(Held.)*
3. **Single-model #24 parity** stays bit-exact under the safeguard. *(Held --
   `test_keep_best_single_model_is_bit_exact`.)*

## Reproduction

- `ensemble_ll.py [N] [MAX_ITER]` -> `ensemble_ll.npz` (Fortran + NG, return-last
  and keep_best final LLs). `plot_ll.py` renders `ll_before_after.png`.
- Real sample data + the macOS Fortran binary (x86_64, Rosetta) only.
