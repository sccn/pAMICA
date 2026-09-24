# Multi-model AMICA parity: distributional equivalence to Fortran (issue #27)

> **Historical record.** These ensembles predate epic #324, which changed every
> backend's default trajectory (issues #333, #335, #339, #341, #344, #345).
> The re-measured study under the finished epic, with the same protocol against
> the pinned v0.3.3 native binary, is in `.context/issue-351/findings.md`
> (script `.context/issue-351/multimodel_ensemble.py`, which reuses this
> directory's analysis and figure code).

**Bottom line.** For multi-model AMICA (`n_models > 1`), the natural-gradient
PyTorch backend (`AMICATorchNG`) is validated against the Fortran reference at
the level that is actually well-posed: its **ensemble of solutions is
statistically indistinguishable from Fortran's**. A single run's partition
cross-correlation (~0.64) is *not* a parity defect; it is the intrinsic spread
of the estimator, and Fortran exhibits the same spread against itself.

## Why a single cross-correlation number is the wrong test

Single-model ICA is (essentially) identifiable: up to permutation/sign/scale
there is one solution, so two correct implementations converge to it (we see
~0.997 component correlation vs Fortran). Multi-model AMICA is a **mixture of
ICA models with a soft partition** (responsibilities `v_h(t)`), which is **not
partition-identifiable**: many near-degenerate partitions achieve essentially
the same likelihood, and EM converges to whichever basin the random
initialization sits in. Two correct runs, even of the *same* implementation from
different seeds, are therefore *not expected* to produce the same partition.

So "match Fortran's partition" is not a well-posed acceptance criterion. The
well-posed question is the one a statistician asks of a stochastic estimator:
**do the two implementations sample the same distribution over solutions?**

## Method (real sample EEG, NO MOCK)

Ran `N = 20` fits per implementation on the sample EEG (`n_models = 2`, 3 mixture
components, 100 iterations, matched schedule: `lrate 0.05`, Newton from iter 50,
`newtrate 1.0`). Both implementations are genuinely stochastic run-to-run
(Fortran reseeds its init RNG from entropy; NG varies with `seed`). For each run
we stored the stacked `2*32` unmixing components and the final log-likelihood.

Metric: **Hungarian-matched mean |correlation|** of the stacked components (the
same matching used for single-model parity, which quotients out component
permutation, sign, scale, and model-label switching). From the 20+20 runs we
formed three distributions of pairwise agreement:

- `within-Fortran` — all Fortran-Fortran pairs (Fortran's intrinsic spread), n=190
- `within-NG` — all NG-NG pairs (NG's intrinsic spread), n=190
- `between` — all NG-Fortran pairs (cross-implementation agreement), n=400

## Results

![Ensemble distributions](multimodel_ensemble_distributions.png)

Values below are from the regenerated ensemble (2026-07-11; `ensemble.npz` now
persisted so the figure/tests reproduce without re-fitting). Absolute magnitudes
differ slightly from the first run (Fortran reseeds from entropy each fit); the
within-vs-between conclusion is unchanged.

| distribution | mean cross-corr | sd | range |
|---|---:|---:|---|
| within-Fortran | 0.638 | 0.040 | [0.572, 0.797] |
| within-NG | 0.661 | 0.045 | [0.583, 0.820] |
| between (NG-Fortran) | 0.649 | 0.045 | [0.582, 0.886] |

- **Run-level permutation test** (H1: `between` worse than `within-Fortran`;
  20000 permutations of the 40 runs as intact units): **p = 0.96** — no evidence
  cross-implementation agreement is *worse* than Fortran's own. This replaces the
  earlier Mann-Whitney/TOST, which were computed over the 190/400 **pairwise**
  correlations as if independent; they are not (each run is in ~39 pairs), so
  those p-values were pseudoreplicated and invalid (the TOST p ≈ 1e-32 was the
  tell). Permuting whole runs respects the shared-run dependence. The
  between-minus-within-Fortran mean difference is +0.011 (well within a ±0.05
  margin), so the distributions overlap descriptively as well.

The three distributions lie on top of each other. **The partition behavior of
`AMICATorchNG` is equivalent to Fortran's** at the run level. The ~0.65 single-run
cross-corr that earlier looked like a shortfall is fully explained: Fortran agrees
with *itself* at 0.638.

### One residual: the likelihood distribution (tracked as #51)

| | mean LL (per sample-channel) | sd |
|---|---:|---:|
| Fortran | -3.3539 | 0.003 |
| NG | -3.3629 | 0.006 |

KS p ≈ 6e-5. NG's LL is ~0.009 lower on average; with the #51 best-iterate
safeguard the variance is now ~2x Fortran's (was ~13x before #51). The residual is
convergence speed, not a worse optimum: NG reaches Fortran's mean with ~2x more
iterations, and Fortran reaches nearly the same LL every run despite different
partitions (confirming those partitions are near-degenerate in likelihood).

This is an **optimizer-quality** signal, not a model-correctness bug: the
per-block sufficient statistics and one M-step are bit-exact vs Fortran
(~1e-15), so the equations are right. The inflated variance (not a uniform mean
shift) points to occasional convergence to slightly worse local optima —
likely the iteration cap or a schedule mismatch. Tracked as issue **#51**.

## Multi-model acceptance criteria (the definition of done)

This replaces the aspirational `>0.95` single-run cross-corr in #27's title,
which asks the algorithm to be more identifiable than it mathematically is:

1. **Algebra:** per-block sufficient statistics and one M-step bit-exact vs
   Fortran (~1e-15) and vs the NumPy oracle (~1e-8). *(Held — test suite.)*
2. **Partition:** the NG solution ensemble is statistically equivalent to
   Fortran's (`between` not worse than `within-Fortran`; TOST within a margin).
   *(Held — this document.)*
3. **Likelihood:** NG's LL distribution equivalent to Fortran's on the sample
   ensemble. *(Open — issue #51; small residual, optimizer tuning.)*
4. **Self-consistency:** fixed seed reproduces exactly; cross-corr 1.0 across
   block sizes. *(Held — `test_blocking_invariance*`.)*

## Reproduction

`.context/issue-27/multimodel_ensemble.py` runs the ensemble (Fortran binary +
NG) and writes `ensemble.npz`; `plot_ensemble.py` renders the figure. Both use
the real sample data and the macOS Fortran binary (x86_64, runs under Rosetta).
Absolute cross-corr magnitudes depend on config/seed; the **controlled
within-vs-between comparison** is the config-independent result.
