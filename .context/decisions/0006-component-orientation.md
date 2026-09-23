# ADR 0006: Components are rows of each stored mixing block

**Status:** accepted
**Date:** 2026-09-22
**Owner:** Seyed Yahya Shirazi

Amends [ADR 0001](0001-torch-backend-natural-gradient-em.md), which introduced the storage convention without naming its consequence for per-component operations.

## Context

Since issue #24 every array backend (PyTorch, NumPy, MLX) stores the mixing matrix `A` with shape `(n, n_comps)` and indexes each model's block by `comp_list`,
but each block holds the transpose of the reference's per-model mixing matrix:
`get_mixing_matrix(h)` returns `A[:, comp_list[:, h]].T`, which is the reference's `A(:, comp_list(:, h))`.
Source `i` of model `h` is therefore ROW `i` of the stored block `A[:, comp_list[:, h]]`,
while its density parameters live at `mu[:, comp_list[i, h]]`, `beta[:, comp_list[i, h]]` (the reference's `sbeta`), `alpha` and `rho`.
A stored COLUMN is not a component: it collects one sphered channel's loadings across the model's components.

The reference's `doscaling` (amica15.f90:1843-1851) divides each component's mixing vector `A(:,k)` by its norm and rescales `mu(:,k) *= norm`, `sbeta(:,k) /= norm`.
That is an exact change of scale: the source is multiplied by the norm, its density is rescaled to match, and the log-likelihood does not change.
Every backend instead normalized stored columns with that same compensation (issue #333), which is not a change of scale of any component.
A compensated rescale of stored row 5 changes the log-likelihood by exactly 0.0 on the sample EEG; the same rescale of stored column 5 changes it by -3.9e-2.
Because `doscaling` is on by default, every default fit left the reference's trajectory from the first iteration:
seeded from the same initialization, the native binary and pamica differed by 7.2e-5 in `A` and 6.6e-5 in `mu` after one iteration,
while with `doscaling` off they agreed to 1e-15 and 1e-11.
Scale-blind endpoint metrics hid it (log-likelihood, Hungarian-matched correlation), but fitted component norms ranged over [0.94, 1.07] in one-model fits and [0.05, 1.97] in two-model fits, where the reference's are exactly 1.

The same orientation error affects the `share_comps` metric and merge, which compare and tie stored columns (issue #334).

## Decision

Keep the storage convention for now and make every per-component operation act on block rows.
In this change (epic #324 Phase 7) `doscaling` normalizes, for every model `h` and source `i`, the row `A[i, comp_list[:, h]]`,
multiplies `mu[:, comp_list[i, h]]` by the row's norm and divides `beta[:, comp_list[i, h]]` by it, leaving a zero-norm row untouched as the reference does.
The rule is identical in the three backends (`_rescale_components`) and runs where the reference runs it (after the `A` update, before the unmixing matrices are rebuilt, and before a share merge).
The reference parses `scalestep` but never reads it and rescales every iteration; pamica keeps `scalestep` as an extension,
now counted from 1 like the reference's other cadences (iterations `scalestep`, `2*scalestep`, ...), so the default of 1 is the reference
(row 14 of `docs/guides/amica-differences.md`).

Epic #324 Phase 8 (issue #334) then changes the storage itself to component rows, `A` of shape `(n_comps, n)` with `comp_list` indexing rows,
so a stored row is a component in every configuration, including merged ones.
Until then, a column merged by `share_comps` belongs to rows of several blocks, where the per-block rule is not a change of scale;
it is applied uniformly, model by model, and Phase 8 replaces it.

## Consequences

- Default trajectories change on every backend, toward the reference.
  Seeded from pamica's initialization, `A`, `mu` and `sbeta` now match the native binary to float64 round-off after one and three natural-gradient iterations
  (after one: `A` 5e-16, `mu` 8e-11, `sbeta` 1e-14; the column rule was 7e-5 off).
  A 100-iteration Newton run against the seeded binary keeps its log-likelihood (2.3e-4 to 2.1e-4 apart; the remaining gap is the Newton start, issue #335),
  and the fitted component norms are exactly 1 instead of [0.94, 1.07].
  Two-model fits improve most: at 100 iterations the matched correlation with the seeded binary rises from 0.850 to 0.99998.
- Scale-blind results do not change materially: log-likelihood, component directions, correlation and Amari distance against the bundled `amicaout` fixture move by about 1e-4.
  Element-wise `A`, `mu` and `sbeta` now match the reference.
- `doscaling=False` is byte-identical to the previous code on every backend, pinned by an in-process comparison with the pre-fix classes.
- Saved models need no conversion: the stored format is unchanged, and a model fitted before this change is still a valid model with non-unit component scales.
  Refit only when comparing parameters element by element with the reference.
- Several test recipes that relied on a non-monotone trajectory (the `keep_best` overshoot, the `min_dll` stop) no longer overshoot at their old settings:
  the column rule's perturbation was part of what made them overshoot. They were retuned (see the Phase 7 pull request).
  ADR 0003's `keep_best` variance figures were measured under the column rule and will be re-measured in the epic's final parity re-measurement.
- `scalestep > 1` now rescales on iterations `scalestep`, `2*scalestep`, ... instead of 1, `1+scalestep`, ...; the default is unaffected.
- Remaining known difference: pamica does not normalize its drawn initial `A` (the reference normalizes a drawn one, amica15.f90:818-819, but not a loaded one).
  The two draws come from different random generators anyway, and the first iteration's rescale normalizes every component.

## Alternatives considered

- **Change the storage to component rows in the same change:** rejected for this phase.
  The layout change touches the update, the accessors, persistence and the share code; the rescale fix alone is small, and landing it first gives Phase 8 a trajectory that is already reference-faithful to stay byte-identical to.
- **Keep the column rule and document it as a difference:** rejected. It is not a change of scale, so it is not an alternative normalization but a perturbation of the fit, and it made element-wise parity with the reference impossible.

## Receipts

- amica15.f90:1843-1851 (per-iteration `doscaling`), :818-819 and :1039-1040 (normalization of a drawn initial and restart `A`), :793-800 (a loaded `A` is used as is), :3686-3688 (`scalestep` parsed, never read).
- Issues #333 (this change), #334 (Phase 8, component-row layout), #335 (Newton start), #24 (the storage convention), epic #324.
- `pamica/tests/test_doscaling_rows.py` (invariance, cross-backend, byte-identity and the gated native oracle), `pamica/tests/native_oracle.py` (seeded reference runs).
