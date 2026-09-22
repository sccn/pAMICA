# ADR 0005: Restore the PCA residual in the MNE export

**Status:** accepted
**Date:** 2026-09-22
**Owner:** Seyed Yahya Shirazi

## Context

A rank-reduced fit, from an explicit `pcakeep`/`pcadb` or from automatic numerical-rank detection, models only the retained PCA subspace.
The rest of the data, the PCA residual, is never part of the decomposition.
Issue #322, filed by an external tester working on Maxwell-filtered MEG, reported that `AMICAICA.apply` silently drops that residual.
`to_mne_ica` exported only the retained rows of the PCA basis, so on the bundled 32-channel EEG with `pcakeep=20`,
`apply` with nothing excluded returned the input with a relative error of 0.080.
The 12 discarded dimensions can carry signal of interest,
and a user who excludes one artifact component does not expect every dimension outside the model to disappear with it.

The Fortran reference behaves the same way.
It keeps `numeigs = min(pcakeep, count(eigs > mineig))` dimensions (amica15.f90:413 and 466) and sizes the model to them (`nw = numeigs`, amica15.f90:563).
In the default `do_approx_sphere` path the written sphere `S` is `nx x nx`, but its rows past `numeigs` are zero (amica15.f90:501-508),
and `W`/`A` are `nw x nw`.
The reference output therefore has no representation of the discarded subspace,
so a back-projection through the `nchan x nw` inverse of that output (as EEGLAB's `icawinv` is built) can only reconstruct the retained subspace.
That is an inference from the shape of the output; EEGLAB itself is not part of this repository.
pamica's own EEGLAB export pads a reduced sphere with zero rows the same way (`numpy_impl/load.py`, `write_amicaout`).

MNE-Python already has a mechanism for this.
`ICA._pick_sources` keeps rows of `pca_components_` past `n_components_` as residual PCA components during `apply`,
controlled by `n_pca_components` (`None` keeps every row).
MNE's own `ICA.fit` stores the full PCA, so its default `apply` restores the residual.

## Decision

`AMICAICA.fit` computes the full orthonormal PCA basis once, after the backend fit succeeds and before publishing state,
and stores it as `pca_components_` (`n_channels x n_channels`) and `pca_explained_variance_` (`n_channels`).
The retained rows are exactly the basis the export always used.
For a reduced sphere, the residual rows are an orthonormal basis of the sphere's null space (from a full SVD),
rotated to the eigenvectors of the fit data's population covariance restricted to that subspace and ordered by descending variance.
The covariance is accumulated over column blocks, so no copy of the data is made and no time series is stored.
`to_mne_ica` exports the full basis and leaves `n_pca_components` at `None`,
so MNE's `apply` restores the residual by default, as MNE's own ICA does.
The reference (rank-reduced) reconstruction is MNE's existing knob, `apply(..., n_pca_components=ica.n_components_)`;
no new keyword is added.

## Consequences

- **Rank-reduced fits reconstruct the input by default.**
  On the bundled EEG with `pcakeep=20` (10 iterations, seed 42), `apply` with nothing excluded now has a relative error of 1.3e-15 (it was 0.080).
  Excluding a component changes the data by exactly that component's back-projection (7.7e-15 relative to a backend oracle)
  and leaves the residual subspace untouched (1.5e-15).
  The residual variances equal the discarded covariance eigenvalues (5.7e-13 relative).
- **This goes beyond the Fortran reference**, which has no representation of the residual.
  The difference is recorded in `docs/guides/amica-differences.md`.
- **Only reconstruction changes.**
  Sources, component maps, `unmixing_matrix_`, `pca_mean_`, `n_components_` and the log-likelihood are unchanged;
  `get_sources` still equals `AMICA.transform` (2.3e-15 relative).
- **Full-rank fits are bit-identical.**
  There is no residual and the covariance is never computed;
  pin tests assert the export and the `apply` output against the pre-change formula, bit for bit.
- **Cost, reduced fits only:** one extra `O(n_channels^2 x n_samples)` covariance pass over the fit data,
  a transient bounded by the block size (about 20 MB at 306 channels),
  and an `n_channels x n_channels` matrix held on the wrapper and written by `ica.save`.
- **The basis lives in the wrapper**, as plain NumPy on the fitted sphere.
  A backend routed through `AMICAICA` later (the MLX backend, epic #324 Phase 4) inherits it with no backend change,
  but a backend used directly exposes no residual basis.

## Restoring the reference behavior

```python
ica.apply(raw.copy(), exclude=[0], n_pca_components=ica.n_components_)

mne_ica = ica.to_mne_ica().copy()
mne_ica.n_pca_components = mne_ica.n_components_  # persisted by mne_ica.save()
```

## Alternatives considered

- **Opt-in only, keeping the rank-reduced reconstruction as the default.**
  It matches the reference, but keeps the silent loss the issue reports as the default for every `pcakeep` or rank-deficient user,
  and disagrees with MNE's own `ICA`, whose default `apply` restores the residual.
  The residual was never part of the decomposition, so removing it is not ICA's decision to make.
- **A `restore_pca_residual` keyword on `AMICAICA.apply`**, as proposed in #322.
  It duplicates MNE's existing `n_pca_components`, gives one behavior two spellings that can disagree,
  and exists only on the wrapper, not on the exported `mne.preprocessing.ICA` that users save and pass around.
- **Storing the basis in the three backends' persisted state** (`state_dict`, `.npz`, the EEGLAB export) instead of computing it in the wrapper.
  The backends keep the sphere but not the fit covariance, so each would need the data at fit time, a format change and a cross-backend test,
  for a quantity only the MNE export consumes; the EEGLAB format has no slot for it at all (the reference zero-pads the sphere).
  Computing it in the wrapper leaves every backend untouched and bit-identical.

## Receipts

- Issue #322 (report), epic #324
- `amica15.f90:413,466,501-508,563`
- MNE-Python 1.12 `ICA._pick_sources` and `ICA.apply(n_pca_components=...)`
- `pamica/mne_compat/core.py` (`_pca_basis`, `_population_covariance`)
- `pamica/tests/mne_tests/test_pca_residual.py`
- `docs/guides/amica-differences.md`
