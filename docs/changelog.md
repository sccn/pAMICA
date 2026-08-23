# Changelog

Release notes are also published on the
[GitHub releases page](https://github.com/sccn/pAMICA/releases).

## Unreleased

- **The MLX backend now supports all five source-density families** (issue
  #265, epic #260 Phase 4, porting the PyTorch backend's issue #26).
  `AMICAMLXNG` takes `pdftype`/`kurt_start`/`num_kurt`/`kurt_int` with
  `AMICATorchNG`'s names, defaults and semantics: the fixed families (2
  Gaussian, 3 logistic, 4 sub-Gaussian cosh+, 1 super-Gaussian cosh-) via a
  per-source `pdtype` dispatch in `_score`/`_log_pdf`, and the `pdftype=1`
  extended-Infomax adaptive switcher between codes 1/4 by kurtosis sign on the
  usual schedule, plus a new `get_pdftype()` accessor. `pdftype=0` (the
  default) is byte-for-byte the pre-#265 implementation: the `_pdtype_h`
  `None` fast path adds zero graph nodes, verified by an epic-tip-vs-new
  before/after fit comparison (bit-identical `A`/`ll_history`, single- and
  multi-model). The fixed families' `z0`/`fp` match the literal
  `amica15.f90` forms through MLX's float32 evaluation to 1e-6
  (`rtol=atol`), and a matched 100-iteration fit lands on the float64
  PyTorch likelihood to within ~1e-7 for every family (four orders inside
  the 0.05 gate). `rho` is frozen for every non-GG family
  (`self.dorho = pdftype == 0`), which also skips the `drho_n`
  accumulation, the per-iteration lgamma-table refresh, and the digamma
  pull -- dead work AMICATorchNG still pays every iteration for a frozen
  `rho` (a deliberate MLX-only WORK divergence, not a numeric one). The
  switcher accumulates its kurtosis moments in numpy float64 on the host
  (an MLX-motivated mechanism difference, not a decision difference) and
  has no bit-exact oracle -- the reference declares `do_choose_pdfs` but
  never accumulates the moments that would drive it -- so it is
  behavior-validated on real EEG, as ADR 0002 already scoped for the
  PyTorch backend. `share_comps` does not synchronize `pdtype` across a
  merged pair, documented on `shared_components()`. Corrected an inaccurate
  cell in `docs/guides/amica-differences.md`'s backend table along the way:
  the legacy NumPy backend's fit path (`_compute_log_pdf`) has no `pdtype`
  parameter and only ever implemented the generalized-Gaussian family, not
  "all five" as the table previously (incorrectly) claimed. Evidence:
  `.context/issue-265/pdf_family_findings.md`.
- **The MLX backend now supports Newton** (issue #264). `AMICAMLXNG` takes
  `do_newton`/`newt_start`/`newtrate`/`newt_ramp` with `AMICATorchNG`'s names,
  defaults and semantics: the same curvature accumulators, the same
  per-source-pair 2x2 solve behind the same unguarded `prod > 1`
  positive-definiteness test, the same learning-rate ramp to `newtrate` (and to
  `lrate_cap` on a fallback), the same `maxdecs` ratchets, and the same
  `n_newton_fallbacks` counter. It runs entirely in float32, which was
  pre-registered as a go/no-go rather than assumed: on the bundled sample the
  finalized curvature matches a float64 PyTorch twin to 4e-7 relative, one
  warmed Newton M-step moves `A` to within 2.4e-7 of the twin's, a matched
  100-iteration fit reaches -3.41149 against float64's -3.41149, and the
  positive-definiteness guard never comes within 1.9 of its boundary across six
  full-data fits (zero fallbacks, monotone likelihood). Evidence and the gate
  script: `.context/issue-264/`. `do_newton` is off by default and every
  accumulator it needs is gated on it, so natural-gradient fits — including
  multi-model and `share_comps` ones — are bit-identical to before.
- **Multi-model Newton no longer crashes on the NumPy backend** (issue #267).
  `numpy_impl` finalized the curvature by dividing its `(data_dim, num_models)`
  accumulators by `dgm[:, None]`, a `(num_models, 1)` model mass that broadcasts
  only for one model, so every multi-model Newton fit raised `ValueError:
  operands could not be broadcast together` on the first iteration Newton was
  active. The issue reported it from a `share_comps` collapse, but it needed no
  sharing at all. Now `dgm[None, :]`, matching the PyTorch backend's
  `dgm.unsqueeze(0)`. Single-model fits are unaffected.
- **The MLX backend now supports component sharing** (issue #263). `AMICAMLXNG`
  takes `share_comps`/`share_start`/`share_iter`/`comp_thresh` with
  `AMICATorchNG`'s names, defaults and validation, runs the same merge schedule
  and 6-iteration post-merge A-freeze, masks the mixture updates and the
  gradient norm by `comp_used`, and exposes `comp_used` and
  `shared_components()`. The merge decision is not reimplemented: it calls the
  NumPy `identify_shared_components` kernel on host float64 `pinv(sphere) @ A`,
  the metric the PyTorch and NumPy backends already share, so all three decide
  identically from the same fitted state. Sharing is off by default and inert
  for `n_models=1`; with it off, every masking and freezing step added here is a
  no-op, so a fit is bit-identical to the same fit with sharing enabled but
  never scheduled (see the `gm` entry below for the one float32-ULP shift
  multi-model fits see relative to the previous release).
- **The MLX multi-model A-update now weights with the previous iteration's
  `gm`** (issue #263; issue #219 raised the same ordering question for
  `numpy_impl`'s `ndtmpsum` and flagged the array backends as follow-up, since
  fixed in PyTorch and now here). Fortran builds `dAk` in the accumulation pass,
  before `update_params` reassigns `gm` (amica15.f90:1749-1761, :1788); MLX used
  the just-updated `gm`. The weights cancel analytically for a disjoint
  `comp_list`, so single-model fits stay byte-for-byte identical and default
  multi-model fits are unaffected except at float32-ULP scale (the two `gm`
  snapshots genuinely differ, so the cancelling division rounds differently;
  measured at most 2.98e-8 in `dAk` on the bundled sample). A fit that shares
  components moves its shared columns differently (by ~1e-2 in `A`) and now
  matches the PyTorch backend to float32 precision.
- **NumPy `share_comps` now measures similarity on de-sphered sensor-space
  maps, matching the PyTorch backend and the Fortran reference** (issue #258).
  `identify_shared_components` used to compare mixing columns directly in the
  sphered space; it now takes `pinv(sphere) @ A`, the same `Spinv` back-map
  `AMICATorchNG` and `amica15.f90` use (:1916, :568-578), so both backends
  reach the identical merge decision from the same fitted state. Borderline
  merge decisions near `comp_thresh` can change relative to a pre-#258 NumPy
  fit, even on a full-rank sphere.
- **The MLX backend now has convergence stops** (issue #248). `AMICAMLXNG`
  implemented neither, so an Apple-GPU fit always ran to `max_iter`; it now
  carries `use_min_dll`/`min_dll`/`maxincs`, `use_grad_norm`/`min_nd` and the
  likelihood-decrease branch's gradient-norm half, with the same names, defaults
  and `stop_reason` strings as `AMICATorchNG`, and stops at the same iteration
  as it on the same data.
- **`share_comps` on the NumPy backend now runs the same algorithm as the
  PyTorch one** (issues #240, #242). A column shared by two models took one
  A-step per contributing model, the second against an already-stepped `A`,
  instead of the reference's single `gm`-weighted average applied once
  (amica15.f90:1749-1761, :1807); the post-merge A-freeze was missing entirely;
  and merged-away columns were divided 0/0 and masked, which also silenced a
  genuine collapse in a live column. Merged-away columns are now indexed out of
  the mixture updates instead of masked, and the share settings that would
  freeze `A` permanently (`share_int <= 6`) are rejected at construction, as in
  `AMICATorchNG`. Sharing is off by default, and fits with it off are bit-
  identical.
- **A NumPy fit that ends non-finite no longer reports success** (issue #240).
  `fit()` checks the fitted parameters at exit, not only the likelihood, and
  sets `converged=False` with a `stop_reason` naming what went non-finite.
  Periodic `writestep`/`histstep` checkpoints are gated by the same check and
  skipped with a logged reason rather than persisting NaN that `loadmodout`
  would read back without complaint; the last valid checkpoint stays on disk.
- **Checkpoint cadence matches the reference** (issue #240). `writestep` and
  `histstep` are now anchored on the Fortran-style 1-indexed iteration
  (`mod(iter, writestep) == 0`, amica15.f90:1124/1130), so the first checkpoint
  lands at iteration `writestep`. The 0-indexed transcription fired at iteration
  0, so every fit wrote a checkpoint after its first iteration whatever
  `writestep` said. Final results are unaffected: `fit()` always writes the
  converged result.
- **`share_comps` works on rank-reduced and rank-deficient fits** (issue #253,
  reported from Maxwell-filtered MEG in #221). The PyTorch merge metric mapped
  mixing columns back to sensor space with `inv(sphere)`, which raised
  "Component sharing needs an invertible sphere" on exactly the data class that
  rank detection had just made fittable. It now uses `pinv(sphere)`, the
  reference's own `Spinv` back-map under reduction (amica15.f90:568-578), and
  `share_comps` with `pcakeep`/`pcadb` is no longer rejected at construction.
  Full-rank fits are unaffected: `pinv` equals `inv` to ~1e-15 there, and the
  bundled sample reproduces its previous `comp_list` and log-likelihood bit for
  bit.

## 0.3.2

Rank-deficient input support across every backend, a much faster default block
size, and a reproducible Fortran reference for parity runs.

- **Rank-deficient data now works** (issue #223, reported from Maxwell-filtered
  MEG in #221). The numerical rank of the data covariance is detected and the
  model is sized to it, porting the reference's `mineig`/`numeigs`/`Spinv`
  machinery, which pamica had not implemented: previously such a fit died with
  `nan_ll` on the first iteration. New `get_sensor_mixing_matrix()` returns
  sensor-space scalp maps when the sphere is no longer square. The rank policy
  is shared by the PyTorch, NumPy and MLX backends so they cannot disagree.
- **Rank detection defaults to a relative eigenvalue floor** (`mineig_rel=1e-12`)
  rather than the reference's absolute `mineig=1e-15`, which is unit-dependent:
  MEG in Tesla yields rank zero under it, and average-referenced EEG is detected
  only by luck. Pass `mineig_rel=None` for the reference's exact behavior. Well
  conditioned data is unaffected and stays bit-identical. See ADR 0004 and
  `docs/guides/amica-differences.md`, which now lists every deliberate difference
  from the reference in one table.
- **The MNE wrapper scales by channel type** before fitting, following MNE's own
  ICA convention (issue #225). Required for mixed magnetometer/gradiometer data,
  whose units differ by orders of magnitude; it decides which directions survive
  rank reduction. A single channel type is unaffected. `AMICAICA` also exports
  rank-reduced fits, and no longer rejects `pcakeep`/`pcadb`.
- **`block_size` default raised from 512 to 8192** (issue #216), ~6x faster per
  iteration on CPU float64 for the bundled sample. Every backend was
  dispatch-bound at the old value. Runs compared bit-for-bit against the binary
  must set the same value on both sides.
- **EEGLAB output of a rank-reduced fit is readable** (issue #164): the sphere is
  padded to the `nx*nx` record the reference writes, and read back column-major.
- **Parity runs are now controlled experiments** (issue #228). The harness
  forwards every setting the binary understands instead of six hardcoded keys,
  seeds the reference run and pins it to one thread, and defaults to the
  seedable native engine rather than the unseedable bundled fixture. Two
  reference runs are now bit-identical where before they differed by up to 0.59.
- `benchmarks/reproduce_table1.py` reproduces the paper's parity table from the
  bundled sample, and the validation guide states what each row costs to verify
  (issue #144).
- **Both Fortran convergence criteria were dead in `numpy_impl` and now work**
  (issue #212). `AMICA_NumPy` stored the raw log-likelihood sum instead of
  Fortran's per-sample-per-channel normalization, reporting `-3317862.78` where
  the reference reports `-3.3`; `min_dll` defaults to `1e-9`, so `use_min_dll`
  could never fire from genuine convergence. Separately, `nd` was built from the
  raw block sum rather than the `gm`-weighted mapped directions, reporting
  ~5.4e3 against Fortran's ~5.7e-2 and staying flat across iterations, so
  `use_grad_norm`/`min_nd` was equally unreachable. Both now match the reference
  formulas, and `AMICA_NumPy.ll_history` is on the same scale as the other
  backends.
- Added the three missing Fortran convergence stops to `AMICATorchNG`
  (issue #207): `use_min_dll`/`min_dll`/`maxincs` (small-likelihood-increase
  stop), `use_grad_norm`/`min_nd` (weight-gradient-norm stop), and the
  lrate-decrease branch's missing gradient-norm half
  (`stop_reason="grad_norm_floor"`). Fixes the reported case where, under
  `do_newton=True`, `lrate` settles at `newtrate` and oscillates instead of
  annealing, so the pre-existing `lrate_floor` check never fired and
  `max_iter` was the only working stop. All five new constructor arguments
  persist through `state_dict()`/`from_state_dict()`; older saved files
  (missing these keys) still load, falling back to the Fortran-faithful
  defaults.

## 0.3.1

Rho-rate schedule fixes across all backends and a reproducible-seed option in the
native binary build.

- Fixed the rho learning-rate (`rholrate`) schedule to match Fortran
  `amica15.f90`: it is a `maxdecs`-ratcheted ceiling (reset to `rholrate0` each
  iteration/fit, tightened only after `maxdecs` persistent log-likelihood
  decreases, gated on `iter > newt_start`), not a per-decrease monotone decay.
  The previous decay collapsed the rho rate toward ~1e-5 and froze the source
  shape. Fixed in the PyTorch and NumPy backends (#194, issue #193) and the MLX
  backend (#197, issue #195).
- Native binary build: reproducible `seed` option. A `seed <int>` line in
  `input.param` now seeds the random initialization deterministically (per-rank,
  no system clock), so a native run is reproducible run to run; without it the
  default stays clock-random. Also makes `random_seed` portable across compilers
  via `random_seed(SIZE=...)`. Adopted from sccn/amica PR #54; the released
  binaries (rebuilt by CI) carry the option (#196).
- Documented the #145 investigation (Newton-vs-Fortran weak-component divergence
  at long budgets): resolved as init-basin sensitivity on under-determined
  components, not a dynamics bug (identical init gives matching results); the
  optional init-robustness enhancement is tracked in #198.

## 0.3.0

MNE-Python compatibility layer (epic #139), additive: the scikit-learn-style
`AMICA` API and the byte-identical EEGLAB I/O are unchanged.

- `pamica.mne_compat.AMICAICA`, an MNE-facing wrapper that fits AMICA directly
  from an `mne.io.Raw`/`Epochs` (`picks=...`, epochs concatenated along time like
  MNE's own ICA) and interoperates with the standard MNE ICA consumer surface:
  `get_sources`, `apply`, `get_components`, `plot_components` and `plot_sources`.
  `to_mne_ica()`
  returns a fully-populated `mne.preprocessing.ICA` (including `reject_`/
  `n_samples_`, so `ICA.save` and `plot_properties` work), so the whole MNE ICA
  ecosystem (component plotting, `find_bads_eog`/`_ecg`, exclusion workflows)
  works on an AMICA decomposition. The export maps pamica's mean, symmetric-ZCA
  sphere and unmixing into MNE's `pca_mean_`/`pca_components_`/`unmixing_matrix_`,
  writing the sphere as `V diag(1/sqrt(e)) V^T` with `V` orthonormal so MNE's
  scalp maps are in channel space; `to_mne_ica().get_sources(raw)` reproduces
  `AMICA.transform(X)` to float64 precision, pinned on real sample EEG. `fit`
  rejects PCA reduction (`pcakeep`/`pcadb`, which leaves the sphere rank-deficient
  and the export invalid) and non-finite input, and a degenerate fit is refused
  by the consumer methods rather than emitting NaNs. MNE is an
  optional extra (`pip install pamica[mne]`); `import pamica` never requires it,
  and a dedicated CI job runs the wrapper tests with the extra installed (phase 1,
  single-model, #140).
- Multi-model exposure through the MNE wrapper: `AMICAICA(n_models=...)` fits a
  mixture of ICA models, and since MNE's `ICA` represents only one unmixing,
  each model is exported as its own single-model `mne.preprocessing.ICA` via
  `to_mne_ica(model_idx=...)` (and the `model_idx` argument on `get_sources`/
  `apply`/`get_components`/`plot_components`/`plot_sources`). The per-sample model
  dominance MNE cannot represent is exposed directly: `get_model_probability(inst)`
  returns `P(model | sample)` (`(n_models, n_samples)`, columns sum to 1) and
  `plot_model_probability(inst)` draws the per-model probability plus best-model
  log-likelihood over time. These build on a new public live accessor,
  `AMICA.model_loglik`/`model_probability` (and the `AMICATorchNG` equivalents),
  which score arbitrary data through the stored sphere/mean; the training-data
  path (without `do_reject`) is pinned bit-for-bit against the E-step's own `Lht`. The per-model export
  folds each model's data-space center `c` into `pca_mean_`, so the round trip
  holds for the multi-model case too. `pamica.viz.plot_model_probability` now also
  accepts a live `lht` array, not only a written `AmicaOutput` (phase 2, #141).
- pamica-specific fitted metadata is inspectable through the MNE wrapper rather
  than silently dropped by the `mne.preprocessing.ICA` export: `get_pdftype(model_idx=...)`
  returns each component's source-density family code (0-4, named by
  `pamica.mne_compat.PDFTYPE_NAMES`), `get_rho(model_idx=...)` the
  generalized-Gaussian shape parameters, and `shared_components()` the components
  merged across models by `share_comps`. The same accessors are added to
  `AMICA`/`AMICATorchNG` (phase 3, #142).
- Separation-quality metrics are available directly on an MNE object:
  `AMICAICA.mir(inst, model_idx=...)` (Mutual Information Reduction, in nats) and
  `AMICAICA.pmi(inst, model_idx=...)` (pairwise mutual information between the
  fitted sources), so MNE-side users get the same metrics as EEGLAB-side users.
  Both extract the fitted channels from the `Raw`/`Epochs` and delegate to
  `AMICA.mir`/`pmi` (#133); the results match the array API exactly (phase 4,
  #143).

## 0.2.2

GitHub repository rename to pAMICA and a `__version__` fix.

- Fixed `pamica.__version__` reporting the stale `0.1.2`: `version.py` hardcoded
  the version and the release sync never touched it, so the 0.2.1 wheel shipped
  correct distribution metadata but a wrong runtime attribute. `__version__` now
  derives from the installed package metadata, so `pyproject.toml` is the single
  source of truth and it can never drift again (#182).
- Canonicalized `pyAMICA` -> `pAMICA` URLs after the GitHub repository was
  renamed `sccn/pyAMICA` -> `sccn/pAMICA`. The documentation site moved to
  <https://eeglab.org/pAMICA/>, so the old `eeglab.org/pyAMICA` links (including
  the README docs badge) now 404; the repository URLs, codecov, the native
  binary resolver's default repository, the docs badge, and `git clone`/`cd`
  snippets are updated to match. GitHub redirects the old repo URLs, and the
  package/import name stays lowercase `pamica` (#184).

## 0.2.1

PyPI publishing, release-metadata sync, the pAMICA display title, and
native-engine documentation.

- Packaging and release: a PyPI publish workflow (`publish.yml`) uploads the
  `pamica` sdist and wheel via Trusted Publishing (OIDC) when a GitHub release
  is published, and `scripts/sync_version.py` keeps the release version in step
  across `pyproject.toml`, `CITATION.cff` and `.zenodo.json` (the publish job
  fails a release whose tag disagrees with them). The display title is now
  **pAMICA**; the package, import and `pip install pamica` stay lowercase
  `pamica` (pip name matching is case-insensitive, so `pip install pAmica`
  resolves to the same project) (#177).
- Native engine docs and validation wiring: a dedicated `AMICANative`
  documentation page (usage, binary cache/SHA-256 verification,
  `PAMICA_NATIVE_BINARY`, the `python -m pamica.native` installer, and the
  offline `native/build.sh` fallback), and `validate_implementations.py` gains
  `--native-engine`/`--fortran-binary` so the real Fortran reference runs as a
  backend on any platform, not only through the bundled macOS `amica15mac`
  fixture (#147 phase 5, #179).

## 0.2.0

Package rename to align with the reserved PyPI name.

- Renamed the Python package `pyAMICA` -> `pamica`: the import path is now
  `import pamica` and the distribution installs as `pip install pamica` (pip
  name matching is case-insensitive, so `pip install pAmica` resolves to the
  same project). The GitHub repository (`sccn/pyAMICA`), the documentation
  domain (`eeglab.org/pyAMICA`), and the release-asset repository are unchanged
  (#176).

## 0.1.3

Native Fortran run engine, separation-quality metrics, LLt output parity, and
the `loadmodout` byte-order fix.

- Native Fortran run engine (`AMICANative`), the fourth backend alongside NumPy,
  PyTorch and MLX. It runs the AMICA Fortran reference itself and returns an
  `AmicaOutput` with the usual accessors, so it is the parity oracle the Python
  backends are checked against. The reference is now built dependency-free (a
  single-rank MPI shim removes the Open MPI runtime, on top of sccn/amica PR
  \#53's no-MKL recipe; proven identical to real Open MPI at machine epsilon) and
  released as a self-contained binary for macOS arm64, Linux x64/arm64 and Windows
  x64 (Windows arm64 runs the x64 binary via emulation until a native toolchain
  exists, issue #173). The binary is resolved for the host and downloaded from the
  release on first use (SHA-256 verified); `python -m pamica.native` installs it
  explicitly, or set `PAMICA_NATIVE_BINARY` to a local build (epic #165).
- Fixed `loadmodout` reading `W`, `sbeta` and `rho` in the wrong byte order:
  it used C order where the writer, genuine Fortran output and EEGLAB's
  `loadmodout15.m` all use column-major (F order). The consequence was that
  `AmicaOutput.W` came back transposed, silently corrupting genuine Fortran
  output and everything derived from it (`A`, `svar`, `origord`), and
  `sbeta`/`rho` were scrambled whenever `num_mix > 1` (the default). A
  write-then-read round trip cancels the error, so no self-consistency test
  could catch it; the fix is pinned by recomputing the bundled Fortran
  fixture's own reported log-likelihood from the loaded parameters (an external
  oracle). The writer's multi-model `W` layout, which interleaved models and was
  not EEGLAB-readable, is corrected to genuine Fortran (model axis slowest);
  single-model output is byte-identical to before. `AmicaOutput` gains a
  supported `sources(X, model=0)` accessor (the loaded-fit counterpart of the
  live model's `transform`) so downstream source derivations no longer hand-roll
  the sphere/unmixing composition (#159). Migration note: a *multi-model*
  `amicaout` directory written by an earlier pamica (whose `W` used the old
  model-interleaved layout) must be regenerated with `write_amica_output`, not
  just re-loaded; there is no version marker to detect the old layout (genuine
  Fortran output carries none either), and the pre-fix multi-model `W` was never
  in the correct convention regardless. Single-model directories are unaffected
  (byte-identical before and after).
- Separation-quality metrics (`pamica.metrics`): `mir` (Mutual Information
  Reduction, in nats) measures how much mutual information a fitted unmixing
  removes from the data. A direct port of `getMIR.m` from bigdelys/pre_ICA_cleaning
  (Apache-2.0; see `THIRD_PARTY_NOTICES.md`), verified against the original at
  1.7e-15 relative on the bundled sample EEG (#134).
- `pairwise_mi` and `block_diagonal_order` (`pamica.metrics`): the pairwise
  mutual-information matrix between fitted sources, plus a greedy
  nearest-neighbour-chain ordering that clusters dependent components near the
  diagonal. A clean-room reimplementation: the reference (`minfojp.m` in
  postAmicaUtility) is GPL-2.0-or-later and pamica is BSD-3-Clause, so its
  source was never read. Agrees with that reference at r=0.9887 on identical
  signals (#135).
- `LLt` output parity with the Fortran reference: both backends now write the
  per-timepoint, per-model log-likelihood file that the reference binary
  produces on every run, and `loadmodout` reads it with the correct column-major
  layout (it previously used C order, scrambling `Lht`/`Lt`). Verified
  bit-exactly in both directions against EEGLAB's real `loadmodout15.m`. Under
  `do_reject`, rejected samples are written as exactly `0.0`, matching Fortran:
  those zeros are load-bearing, since its `load_rej` reconstructs the rejection
  mask from them (#155).
- `AMICATorchNG`/`AMICA` gain `mir()`/`pmi()` accessors that compose the
  fitted unmixing the documented way (`get_unmixing_matrix(model_idx) @
  sphere` for MIR, `transform(X, model_idx)` for PMI) and delegate to
  `pamica.metrics.mir`/`pairwise_mi`, so callers no longer hand-compose the
  transform themselves. `fit()` also accepts `mir_step` (default `0`, off) to
  record MIR waypoints during training in `mir_history_` as
  `(iteration, mir_nats, variance)`; like `ll_history_`, it is a true
  trajectory that a `keep_best` restore does not rewrite. PCA reduction
  (`pcakeep`/`pcadb`) is rejected up front with a named error, since it
  leaves the sphere rank-deficient and MIR's log-Jacobian undefined (#137).
- Visualization module (`pamica.viz`): `plot_pmi_heatmap` and
  `plot_model_probability`, backend-agnostic views over `AmicaOutput` that return
  a `Figure` (and accept an optional `ax`/`axes`) rather than mutating pyplot
  global state, plus `read_eeglab_set_metadata` for the sample rate pamica
  itself has no notion of. Both plots are verified against the MATLAB reference:
  the smoothed model probability matches `smooth_amica_prob` at r=0.9886, and
  `pairwise_mi` matches `minfojp` at r=0.9887 (#136).
- Fixed `numpy_impl.pdf.compute_pdf` using `gammaln` where the generalized
  Gaussian needs `gamma`, which made the returned density negative for every
  `rho` outside the special-cased 1 and 2 (it integrated to -8.82 at the default
  `rho0=1.5`). Affected `numpy_impl.viz.plot_pdf_fits`; the fit path was never
  affected, as it uses its own log-space implementation (#136).

## 0.1.2

Outlier-rejection parity in the NumPy backend, repo-wide type-checking, and the
full validation-evidence documentation.

- NumPy backend outlier rejection: the Fortran `do_reject` outlier-rejection path
  is ported to `AMICA_NumPy` via the same `good_idx` mechanism as the PyTorch
  backend, so the NumPy reference now drops per-sample outliers on the
  `rejstart`/`rejint`/`maxrej` schedule (#123).
- Rejection robustness: a non-finite log-likelihood is now distinguished from an
  over-aggressive `rejsig`, so an over-tight rejection threshold fails with a
  clear message instead of a silent non-finite result (#127).
- Type checking enforced: repo-wide `ty` diagnostics fixed (496 to 0) and `ty`
  added to CI alongside a pre-commit config (ruff + ty) (#124, #125).
- Documentation: the validation guide is expanded into a full evidence page,
  source-density bit-exactness, cross-platform device/precision invariance
  (cross-backend equivalence matrix and IC topomaps), the EEGLAB drop-in
  round-trip, and the other validated behaviors (#108).

## 0.1.1

Validation-methodology and correctness fixes since 0.1.0.

- Amari distance: a second, permutation- and scale-invariant unmixing-matrix
  comparison metric (Amari, Cichocki & Yang 1996) alongside Hungarian-matched
  correlation, used throughout the Fortran-parity validation (#120).
- Multi-model equivalence test: switched to a valid run-level permutation test
  that respects the dependence among the 40 runs' pairwise correlations, instead
  of a pseudoreplicated Mann-Whitney/TOST (#115).
- Parity and performance tables added to the paper, with the full results,
  native-Fortran CPU core-scaling rows, and per-run detail in the docs (#112).
- Type-safety fixes in `validate_implementations.py` (`run_fortran_amica`
  return type, `load_eeglab_data` dtype annotation) (#118).
- JOSS draft-PDF build workflow, `.zenodo.json` with ROR-based citation
  metadata, and an MLX backend API reference page (#110, #105, #107).
- Corrected a stale float32-speedup claim and added a funding acknowledgement
  (#114).

## 0.1.0

First public release.

- PyTorch natural-gradient EM backend (`AMICATorchNG`) at Fortran parity on real
  EEG (single-model log-likelihood ~ -3.40, Hungarian-matched component
  correlation ~ 0.997).
- Backends: CPU, NVIDIA GPU (CUDA), and Apple GPU (MLX); float64 for parity,
  float32 for speed.
- All five source-density families, mixture of ICA models, Newton updates,
  component sharing, and outlier rejection.
- EEGLAB drop-in output: `write_amica_output` writes the `loadmodout15` format,
  and `variance_order` gives the EEGLAB back-projected-variance component order.
- Spatially-distributed channel-subset selection and a data-size (k-factor)
  cross-backend equivalence sweep for the benchmarks.
- scikit-learn-style `AMICA` interface, save/load, and a documentation site.
