# Plan: Make MLX a first-class, end-to-end pamica backend (epic)

Integration branch: `dev` (repo policy; `main` is release-only).
Epic branch: `feature/issue-324-epic-mlx-first-class`, worktree `../epic-mlx-first-class`.
Phase PRs squash-merge into the epic branch; the epic merges into `dev` with a regular merge commit.

## Context

The MLX backend reached full fitting and non-fitting parity with `AMICATorchNG` in epic #278,
but it is still not usable end to end.
Every MLX user is a raw-`AMICAMLXNG` user: the `AMICA` and `AMICAICA` wrappers construct torch unconditionally (#313),
MLX has no explicit `pcakeep`/`pcadb` (#323),
and no params-file path (#304).
We are the first users (per-session AMICA on 64-channel BioSemi EEG on Apple Silicon, `pcakeep = nbchan - 1` after average reference),
so the bar is a workflow we trust, not a feature checklist.

Evidence gathered while planning (2026-09-22, all on the bundled real EEG sample, 32 channels x 30504 frames):

1. `AMICAICA.apply` after a `pcakeep=20` fit, with nothing excluded, returns data with relative error 0.080 against the input:
   the discarded 12-dimensional PCA residual is silently dropped (#322).
   Exporting the full PCA basis instead gives 1.2e-15, excluding a component leaves the residual untouched (1.2e-15),
   `get_sources` is unchanged (2.3e-15), and `apply(n_pca_components=n_components_)` reproduces today's output exactly (0.0).
   MNE's own `ICA` keeps the full PCA basis and restores the residual by default (`ICA._pick_sources`, MNE 1.12.1).
2. `AMICATorchNG` validates neither `pcakeep` nor `pcadb`:
   `pcakeep=-3` silently fits 29 sources (negative slice), `pcakeep=2.7` silently truncates to 2,
   `pcakeep=0` and `pcadb<=0` produce a degenerate `nan_ll` fit.
3. `AMICATorchNG`'s upfront `mir_step` gate rejects `pcakeep=32` on 32-channel data (no reduction happens);
   that is the value in the bundled `input.param` and `sample_params.json`.
4. Fortran reference review (`amica15.f90`):
   `numeigs = min(pcakeep, count(eigs > mineig))` (413, 466);
   the written `S` is `nx x nx`, and in the default `do_approx_sphere` path its rows beyond `numeigs` are zero (501-508),
   so the reference output carries no representation of the discarded subspace;
   `pcadb` is parsed (3459-3461) but never used.
   pamica's EEGLAB export already pads a reduced sphere with zero rows (`numpy_impl/load.py`, `write_amicaout`).
5. With the JSON-schema spellings mapped to canonical names (`min_grad_norm->min_nd`, `max_decs->maxdecs`, `share_int->share_iter`),
   `sample_params.json` and `input.param` agree on all 47 shared keys except the data path `files`.
   Without the mapping, `AMICA.from_params_file(sample_params.json)` does not apply `max_decs`/`min_grad_norm`/`share_int`.

## What already exists to reuse (do not rebuild)

- `pamica/rank.py::numerical_rank`: the shared rank decision, called by all three array backends.
- `AMICATorchNG._pca_reduction_requested` / `_pca_reduced` and the MLX `_pca_reduced` port.
- `AMICAICA.to_mne_ica`: retained-basis extraction (eigh for a square sphere, SVD for a reduced one).
- `pamica/fortran_params.py::read_fortran_param_file` and `FORTRAN_TO_PAMICA_KEY`.
- `AMICA.from_params_file` content sniffing and `AMICA.fit`'s per-call file-default merge.
- `AMICAMLXNG` full public surface (transform, accessors, mir/pmi, LLt scoring, variance_order, EEGLAB export, state_dict/save/load).
- Test patterns: `pamica/tests/test_rank_policy.py` (cross-backend anti-drift), `pamica/tests/test_mlx_*_cross_backend.py`, `pamica/tests/mne_tests/`.

## Phases

| # | Slug | Issue | Implementer | Wave | Target |
|---|---|---|---|---|---|
| 1 | pca-policy | #323 | Opus | 1 | epic branch |
| 2 | pca-residual | #322 | Opus | 1 | `dev` directly (see judgment call 1) |
| 3 | params-surface | #304 | Sonnet | 1 | epic branch |
| 4 | backend-selection | #313 | Opus | 2 | epic branch |
| 5 | backend-guards | #306 | Sonnet | 2 | epic branch |
| 6 | e2e-validation | #315 | Opus | 3 | epic branch |

Every phase: `uv run ruff check --fix . && uv run ruff format .`, `uv run ty check`, full `uv run pytest` (MLX suites run on this host),
existing Fortran-parity tests green (default path bit-identical), atomic commits, PR reviewed with `/review-pr` (pr-review-toolkit, Sonnet), all findings addressed.
Implementers never run `pre-commit install` in a worktree (shared hooks hazard).

### Phase 1: pca-policy (#323)

Decided policies:

- `pamica/rank.py` gains `validate_pca_reduction(pcakeep, pcadb) -> None`:
  `pcakeep` is `None` or an integral (`numbers.Integral`, `bool` rejected) `>= 1`;
  `pcadb` is `None` or a real (`numbers.Real`, `bool` rejected), finite, `> 0`;
  otherwise `ValueError` naming the parameter and value.
  Both set is allowed: `pcakeep` takes precedence and `pcadb` is ignored, matching the reference (which ignores `pcadb` entirely);
  both bundled param files set both. Documented, logged at INFO once at construction.
- `pamica/rank.py` gains `pca_reduction_requested(pcakeep, pcadb, n_channels) -> bool`:
  `pcakeep` set: `pcakeep < n_channels`; else `pcadb` set: `True` (cannot be known before the eigenvalues); else `False`.
- `numerical_rank` calls `validate_pca_reduction` first (single fit-time choke point, catches post-construction attribute changes).
- All three constructors validate (fail fast). Torch `_pca_reduction_requested` delegates to the shared predicate with the fitted data's channel count.
- MLX constructor gains `pcakeep: Optional[int] = None, pcadb: Optional[float] = None` (same position and docstring semantics as torch),
  passes them to `numerical_rank`, gains `_pca_reduction_requested` and the same upfront `mir_step` gate and message as torch,
  and persists both in `state_dict()["config"]` (additive, like `keep_best`: a payload without them loads with `None`, no format bump).
- Rewrite every docstring, comment and error message that says MLX has no explicit `pcakeep`/`pcadb`
  (`mlx_impl/core.py` around 2405-2440, 3240-3310) and the differences-page section "`mir_step`'s upfront PCA-reduction gate: MLX has none".

Deliverables: `pamica/rank.py`, `torch_impl/core.py`, `numpy_impl/core.py`, `mlx_impl/core.py`,
`docs/api/mlx-backend.md`, `docs/guides/amica-differences.md` (backend-table row "Explicit `pcakeep`/`pcadb`"; `pcadb` recorded as a pamica extension), `docs/changelog.md`,
tests below.

Decision gate (tests, pre-registered):

1. `test_rank_policy.py`: validator accepts/rejects the listed cases (including `True`, `2.7`, `0`, `-3`, `np.int64(5)` accepted, `nan`/`inf`/`0`/`-5` pcadb); predicate truth table; `numerical_rank` raises on invalid input.
2. New `pamica/tests/test_pca_reduction_cross_backend.py` on the real sample:
   (a) `pcakeep=20`: torch, NumPy and MLX all fit 20 sources with a `(20, 32)` sphere; torch vs NumPy sphere `allclose(rtol=1e-10)`; torch vs MLX within float32 tolerance; `sldet` agrees;
   (b) same for a `pcadb` value that reduces the sample (implementer picks and records it);
   (c) `pcakeep` + `pcadb` equals `pcakeep` alone on every backend;
   (d) invalid values raise `ValueError` at construction on every backend;
   (e) `mir_step > 0`: torch and MLX both reject `pcakeep < n` up front and both accept `pcakeep >= n`;
   (f) torch vs MLX after a short fit with `pcakeep=20`: Hungarian-matched component correlation at or above the threshold the existing MLX cross-backend suites use (cite it);
   (g) MLX `save`/`load` round trip keeps `pcakeep`/`pcadb` and the reduced sphere; a payload with those config keys removed still loads.
3. Regression: `AMICA.from_params_file(input.param)` then `fit(mir_step=1)` no longer raises.
4. MLX via `pytest.importorskip`; torch and NumPy unconditional.

### Phase 2: pca-residual (#322)

Decided policies:

- The residual is restored by default, through MNE's native mechanism:
  `to_mne_ica` exports the full orthonormal PCA basis (`n_channels x n_channels`) and the full `pca_explained_variance_`;
  `n_components_` stays the retained rank; `n_pca_components` stays `None`, so MNE's `apply` keeps every PCA component (MNE's own `ICA` default).
  Reference (rank-reduced) behavior: `apply(..., n_pca_components=ica.n_components_)`. No new keyword.
- The basis is computed once in `AMICAICA.fit` and stored as documented fitted attributes (suggested names mirror MNE: `pca_components_`, `pca_explained_variance_`);
  nothing is recomputed from data at `apply` time and no time series is stored.
  Retained rows: exactly today's `to_mne_ica` computation from the fitted sphere.
  Residual rows: null space of the fitted sphere (full SVD), rotated to the eigenvectors of the pre-whitened fit data's population covariance restricted to that null space,
  descending variance, variances clipped at 0.
  Covariance accumulated in column blocks (two pass: mean, then centered blocks); no full-size data copies (306-channel MEG must stay cheap).
- Full-rank fits: export unchanged. The first commit is a pin test captured against the current code.
- `to_mne_ica` logs at INFO when a residual is exported: its dimension and the opt-out.
- Documentation must state plainly that this goes beyond the reference (user requirement):
  ADR `.context/decisions/NNNN-restore-pca-residual-in-mne-export.md`;
  a `docs/guides/amica-differences.md` section covering what the reference does (the model spans `numeigs` dimensions; `S` rows beyond `numeigs` are zero in the default path, amica15.f90:501-508; back-projection from the reference output, for example EEGLAB's through the `nchan x nw` `icawinv`, reconstructs only the retained subspace, dropping the residual without notice),
  what pamica does, why (MNE's own default; the residual was never part of the decomposition, so it is not ICA's to remove),
  how to restore the reference behavior, and that sources, maps and likelihood are unchanged (only reconstruction changes);
  explicit statements in the `AMICAICA` class Notes and the `fit`, `to_mne_ica` and `apply` docstrings;
  `docs/api/mne-compat.md`; a changelog entry flagged as a behavior change.

Decision gate (tests on real data, pre-registered):

1. Pin: full-rank export attributes and `apply` output identical to the pre-change code.
2. `pcakeep=20`: `apply()` with no exclusions reproduces the input (relative error `<= 1e-10`);
   `apply(n_pca_components=n_components_)` equals the retained-subspace reconstruction;
   `exclude=[k]` changes the data by exactly minus component `k`'s back-projection, with a residual-subspace component `<= 1e-10` relative;
   `get_sources` equals `amica_.transform` at the existing tolerance;
   residual variances equal the discarded covariance eigenvalues (`rtol=1e-8`); `pca_components_` orthonormal.
3. `n_models=2` with `pcakeep`: every exported model restores the residual (the `c` offset does not leak into it).
4. `Epochs` input; `ica.save` then `mne.preprocessing.read_ica` gives the same `apply` output; float `n_pca_components` works.
5. Mixed channel types (non-uniform `pre_whitener_`): residual restored in original units.
6. Automatic rank reduction (average-referenced EEG, rank 31, no `pcakeep`): `apply()` with no exclusions reproduces the input.

### Phase 3: params-surface (#304)

Decided policies:

- `pamica/fortran_params.py` gains `read_params_file(path) -> dict`, the single params-file entry point:
  JSON when the first non-whitespace character is `{` or `[` (top level must be an object), Fortran text otherwise.
  Both formats return canonical pamica keys: JSON keys pass through one alias table
  (`min_grad_norm->min_nd`, `max_decs->maxdecs`, `numrej->maxrej`, `num_mix_comps->num_mix`, `share_int->share_iter`);
  a JSON file carrying both an alias and its canonical key raises `ValueError`.
- `writestep`, `do_history`, `histstep` move from unsupported to translated (identity):
  the reader translates what any backend supports, and each consumer warns about what it cannot apply (the wrapper already does; #312 tracks torch/MLX).
- `AMICA.from_params_file` uses `read_params_file`.
  Behavior change: `sample_params.json`'s `max_decs`/`min_grad_norm`/`share_int` are now applied (changelog).
- `AMICA_NumPy(params_file=...)` and `load_default_params` use `read_params_file`, then map canonical keys to NumPy's spellings
  (`min_nd->min_grad_norm`, `maxdecs->max_decs`, `share_iter->share_int`); NumPy's keyword spellings are unchanged.
  Data-location keys come from the same parsed dict (no second `json.load`).
  Keys NumPy does not consume produce one `logger.warning` naming them.
- NumPy `pdftype != 0` raises `NotImplementedError` at construction (the legacy backend implements only the generalized Gaussian);
  `numpy_impl/params.json` default `pdftype` changes 1 -> 0; the module docstring states the reason (per `.rules/backend_parity.md`).
- MLX's params-file path is `AMICA.from_params_file(..., backend="mlx")`, delivered in Phase 4; this phase's differences-table row says so.

Decision gate (pre-registered):

1. Anti-drift: `read_params_file(sample_params.json)` and `read_params_file(input.param)` agree on every shared key except `files`.
2. Alias/canonical conflict, non-object JSON and unparseable text each raise `ValueError`.
3. NumPy from `input.param` equals NumPy from `sample_params.json` on every shared hyperparameter, and `fit()` with no data runs from `input.param` (test chdirs to `sample_data`, like the binary).
4. NumPy `pdftype` 1-4 raise `NotImplementedError`; 0 fits.
5. Wrapper from `sample_params.json` applies `maxdecs=3`, `min_nd`, `share_iter` to the backend, and the "not applied" warning no longer lists them.
6. NumPy's unconsumed-key warning names the keys; `writestep`/`do_history`/`histstep` from Fortran text reach NumPy, and the wrapper warns they are not applied.

### Phase 4: backend-selection (#313), detailed plan before it starts

Decided policies:

- `AMICA(..., backend="torch")`, accepting `"torch"` and `"mlx"`; anything else raises `ValueError`;
  `"mlx"` without MLX installed raises `ImportError` with the install hint at construction.
  Default stays `"torch"` (float64 Fortran parity); no `"auto"`.
- `device` and `dtype` are torch-only; passing either with `backend="mlx"` raises `ValueError` (MLX is float32-only).
- The constructor-kwarg and file-param partitions are computed per backend from the backend class signature (one helper),
  so `AMICA.from_params_file(path, backend="mlx")` works and unapplied keys are warned per backend.
- The #50 degenerate-fit contract, `converged_`/`is_fitted_`/`stop_reason_`, restart records and `mir_history_` behave identically for both backends.
- One wrapper save format for both backends; the payload records its backend and `load` dispatches;
  a payload without the field is a torch payload (the only kind written before this change);
  loading an MLX payload without MLX installed raises `ImportError` naming the backend.
- `AMICAICA(..., backend="torch")` forwards to `AMICA`; `device` torch-only;
  `to_mne_ica` and the Phase 2 residual computation read mean, sphere and `c` through one backend-agnostic float64 accessor (no `.cpu()` in `mne_compat`).
- End-to-end tests, both backends (MLX `importorskip`): wrapper fit, transform, save, load, transform;
  `from_params_file(input.param, backend="mlx")`; `pcakeep` through the wrapper (closes #323's third bullet);
  `AMICAICA(backend="mlx")` on real EEG: `get_sources` equals `transform` (float32 tolerance), residual restored by `apply`, `ica.save`/`read_ica`;
  wrapper EEGLAB export from MLX reloads through `loadmodout`;
  torch vs MLX through the wrapper: same `n_components_`, Hungarian-matched component correlation at the existing threshold.
- Docs: `docs/guides/backends.md`, `docs/api/mne-compat.md`, differences page (drop the torch-only-wrapper caveat), README quick start, the `amica.py` line in `AGENTS.md`.

### Phase 5: backend-guards (#306), detailed plan before it starts

The three #306 items (degenerate-fit guards on raw accessors, `transform` input validation, named `from_state_dict` config errors),
identical on torch and MLX, one shared test per item in `pamica/tests/`.

### Phase 6: e2e-validation (#315), detailed plan before it starts

- `validate_implementations.py --backend {torch,numpy,mlx}` (default `torch`, unchanged output), producing each backend's row against the Fortran binary.
- An end-to-end workflow test mirroring our own use:
  average-referenced real EEG, `pcakeep = nbchan - 1`, fit through `AMICA(backend="mlx")` and `AMICAICA(backend="mlx")`,
  EEGLAB export and reload, MNE `apply` with residual restoration, params-file driven run.
- Docs: an Apple Silicon (MLX) path in `docs/getting-started.md`; the validation guide's parity table gains the NumPy and MLX harness rows.

## Prerequisites (user actions)

None. MLX, MNE and the Fortran binary are available on this host.

## Agent budget

Six implementers (one per phase; at most three concurrent), plus one pr-review-toolkit run per phase PR and one for the epic PR (Sonnet specialists).

## Open judgment calls (veto any)

1. Phase 2 (#322) ships straight to `dev`, not through the epic branch: it is the external tester's issue and independent of MLX. The epic branch merges `dev` before Phase 4 (precedent: epic #278 fast-forwarded from `dev`).
2. The residual basis is computed by the MNE wrapper at fit time, not stored in the three backends' persisted state: the fitted sphere already defines the retained subspace, so this avoids three persistence-format changes ("PCA is cheap").
3. `pcakeep` and `pcadb` together are allowed, `pcakeep` wins (the reference ignores `pcadb`; both bundled param files set both).
4. New validation turns previously silent bad values (`pcakeep <= 0` or non-integer, `pcadb <= 0`) into `ValueError`.
5. The `mir_step` gate no longer rejects `pcakeep >= n_channels`.
6. JSON key canonicalization changes what `AMICA.from_params_file(sample_params.json)` applies.
7. NumPy `pdftype != 0` raises; the NumPy `params.json` default changes from 1 to 0.
8. `writestep`/`do_history`/`histstep` are translated from Fortran text (NumPy applies them; the wrapper warns).
9. The wrapper's backend default stays `"torch"`; no `"auto"`.
10. Scope: #306 and #315 join the epic for end-to-end trust; #307, #312, #314, #317, #319, #320 stay out.

## Verification

Each phase's gate tests above, run by the implementer and re-run by the lead before merge (the lead also spot-checks cited lines and keystone numbers).
Epic PR: full suite on the epic branch, `validate_implementations.py` for every backend, pr-review-toolkit review, then user approval of the final merge.
