# ADR 0007: Store the mixing matrix with one component per row

**Status:** accepted
**Date:** 2026-09-23
**Owner:** Seyed Yahya Shirazi

Amends [ADR 0006](0006-component-orientation.md), which named the per-block convention and fixed `doscaling` within it,
and [ADR 0001](0001-torch-backend-natural-gradient-em.md), which introduced the storage.

## Context

Since issue #24 every array backend stored `A` as `(n, n_comps)` and took model `h`'s block as `A[:, comp_list[:, h]]`,
the transpose of the reference's per-model mixing matrix, so a component was a ROW of its block
while the component ids in `comp_list`, and the density columns `mu[:, k]`, `beta[:, k]`, `alpha[:, k]`, `rho[:, k]`, indexed stored COLUMNS.
A stored column held one sphered channel's loadings across a model's components, not a component.
ADR 0006 made `doscaling` act on block rows, but `share_comps` still used the column ids for both of its steps (issue #334):

- The metric compared de-sphered stored columns, not scalp maps.
  On a two-model fit of the sample EEG (150 iterations) the |cosine| between a stored column and the true map `get_sensor_mixing_matrix(h)[:, i]` had median 0.27 and 0.39;
  a planted exact duplicate component scored 0.74 and was not merged at `comp_thresh=0.99`.
- The merge tied one channel's loadings across two models instead of two components' mixing vectors,
  and the `gm`-weighted `dAk/zeta` average then averaged those.
  Forcing it on a planted identical pair changed the log-likelihood by -0.186, where the reference's fold changes it by exactly 0.

The reference's own scan never merges (its `Spinv2` is declared but never allocated, so every similarity is NaN),
but its `load_comp_list` seeds a merged `comp_list`, and its update from that state is a bit-level oracle.

## Decision

Every array backend (PyTorch, NumPy, MLX) stores `A` as `(n_comps, n)`:
row `k` is component `k`'s sphered-space mixing vector, the reference's column `A(:, k)`,
and model `h`'s block is `A[comp_list[:, h], :]`, the same `n x n` matrix as before, so `W_h = inv(block)` is unchanged.
Everything that touches `A` follows:
the natural-gradient and Newton update scatter each model's step into the rows `comp_list` names (the `dAk/zeta` average now averages one component across the models that share it);
`doscaling` rescales each component row once, including a shared one, and leaves a merged-away row alone;
the share metric compares `pinv(sphere) @ A.T`, whose column `comp_list[i, h]` is column `i` of `get_sensor_mixing_matrix(h)`;
the merge re-points `comp_list` and retires the folded row, as the reference does, copying and averaging nothing.
The per-backend helper `_component_sensor_maps` returns the vectors the metric compares.

Persistence converts or refuses, through one shared function (`pamica.component_layout.rows_from_legacy_columns`):
the PyTorch `state_dict` is `format_version` 4 and the MLX save format 2;
a version 3 (PyTorch) or version 1 (MLX) payload is converted without loss when its `comp_list` is unmerged
(every block carries over element for element),
and refused with a message to refit when a component id is shared across models,
because that merge was computed under the column semantics and has no component-row counterpart.
The `AMICA` wrapper's own save (version 2) wraps the backend payloads and goes through the same path.

The EEGLAB export writes `A` in the reference's layout for every model count:
`(nw, num_comps)` column-major, the bytes of the component-row `A` in C order.
`loadmodout15.m` ignores this file; `load_results` reads it back
and refuses an `A` that does not invert the `W` beside it (a multi-model directory written before this change).

## Consequences

- Every configuration without a merge is byte-identical to the previous code on every backend:
  one, two and three models, `doscaling` and Newton on and off, `pdftype` 0 and 1 (NumPy supports 0 only), and `do_reject`,
  pinned against the pre-change package loaded from git.
  The weight-gradient norm `ndtmpsum` moves by float round-off (2.2e-16 relative in float64, one float32 unit on MLX):
  it now sums squares per component row, as the reference sums per component.
- Fits with `share_comps=True` where a merge fires change, toward the reference.
  From a merged `load_comp_list` state the PyTorch and NumPy updates match the native binary to float64 round-off
  (after 3 iterations, worst of `doscaling` on and off: `A` 1.5e-12, `mu` 3.7e-9, `sbeta` 1.8e-11, log-likelihood 4.7e-14, at the no-merge noise floor),
  where the column semantics was off by 0.21 in `A` and 4.3e-4 in log-likelihood (worst of the same two).
  End to end (2 models, 300 iterations, `comp_thresh=0.95`), the scan now merges three true pairs (map |cos| 0.956-0.971)
  where the column metric merged pairs whose maps had |cos| 0.06, 0.35 and 0.55.
- Merging is earlier and more complete when the models are still similar:
  both models start near the identity, so a scan in the first iterations now merges most components
  (it compares their true maps; on the sample with seed 42, 32 of 32 at iteration 8 with `comp_thresh=0.95`, 24 with 0.99),
  and a model that loses its responsibility can collapse.
  The reference behaves the same way: its similarity on its own early state merges exactly the pairs pamica's does,
  and its update from the same merged states loses the second model's responsibility in step with pamica's, to NaN where pamica's goes non-finite.
  The reference's default `share_start` of 100 avoids this; several short test recipes were retuned.
- Saved models: unmerged saves load as before; saves with merged components must be refit.
  Multi-model EEGLAB directories written before this change must be written again for `load_results`.
- Initial-`A` normalization is not part of this change: the reference normalizes a drawn initial `A` and pamica does not,
  which changes default trajectories and is issue #341.

## Alternatives considered

- **Keep the column layout and translate component ids to block rows inside the share code:**
  rejected. A component shared by two models sits in a row of two different blocks in that layout,
  so the tied object cannot be stored once, and every future per-component operation would need the same translation.
- **Convert merged old saves by re-deriving the merge under the new metric:**
  rejected. The old merge tied parameters that are not components, so there is no fitted state to convert; a refit is the only honest answer.
- **Leave the EEGLAB `A` in the old C-order layout:** rejected. The file then had no meaning outside pamica for several models,
  and the reference's layout is also what `load_results` should read from a reference run.

## Receipts

- amica15.f90:1749-1761 (`dAk/zeta` over `comp_list`), :1843-1851 (`doscaling` over components),
  :1916-1963 (`identify_shared_comps`), :825-846 (`load_comp_list`, which reads the file named `c`).
- Issues #334 (this change), #24 (the storage convention), #333 and ADR 0006 (`doscaling`), #341 (initial `A`), epic #324.
- `pamica/tests/test_component_rows.py` (byte identity, semantics, export, the gated merged-state oracle),
  `pamica/tests/test_component_rows_persistence.py` (round trips, conversion, refusal).
