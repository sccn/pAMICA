# Progress Summary

Internal status snapshot for the PyTorch AMICA effort. The living source of truth for parity
status is the **Known Issues** section of `../AGENTS.md`; this file summarizes what has landed and
what remains, as of epic #324 (after v0.3.3). User-facing detail is in `docs/changelog.md`.

## Delivered

### Natural-gradient EM backend at Fortran parity (epic #9 / issue #24)
- `AMICATorchNG` (`pamica/torch_impl/core.py`) is the sole PyTorch backend; the `AMICA`
  scikit-learn-style wrapper wraps it directly.
- Root cause of the historical parity gap: the natural-gradient A-update was transposed and
  multiplied on the wrong side. Fix plus exact-EM mixture updates, the digamma rho update, the
  symmetric-ZCA sphere, the output transpose, and the Jacobian log-likelihood bring single-model
  results to the Fortran fixed point.
- Newton is ported from the NumPy reference and stays positive-definite (0 fallbacks on the
  sample data).
- **Metrics:** single-model LL ~ -3.40 (Fortran -3.4018); Hungarian-matched component correlation
  ~0.997, clearing the >0.95 gate. Re-measured under epic #324 (#351) against the bundled
  200-iteration fixture: LL -3.4017 vs -3.4019, correlation 0.998, Amari 4.8e-3, on all three backends.

### Adaptive PDF selection (issue #26)
- All five `amica15.f90` `pdftype` density families ported to `AMICATorchNG`: 0 generalized
  Gaussian (default, byte-for-byte unchanged), 2 Gaussian, 3 logistic, 4 sub-Gaussian cosh+, and
  the extended-Infomax kurtosis switcher (`pdftype=1`).
- Fixed families are bit-exact vs the literal Fortran `z0`/`fp` (~1e-15) and converge within
  ~0.005 LL of the binary. The dynamic switcher is dead code in the binary (no bit-exact oracle),
  so it is validated by real-data LL. See ADR 0002.

### Multi-model AMICA (issue #27)
- Validated by **distributional equivalence**: multi-model AMICA is not partition-identifiable, so
  the NG-vs-Fortran partition cross-correlation distribution is compared with Fortran's own
  run-to-run spread. Re-measured under epic #324 (#351): between minus within-Fortran +0.006
  (run-level permutation p=0.88; Amari +0.005, p=0.051), final LL -3.3541 vs -3.3543 (KS p=0.83).
  Per-block sufficient stats agree with Fortran to round-off.
- Per-model bias `c` update (`update_c`) ported to both backends, guarded to a no-op for
  `n_models=1` so single-model parity stays bit-exact. See `.context/issue-27/`.

### Best-iterate safeguard (issue #51)
- `AMICATorchNG.fit` returns the highest-LL iterate (`keep_best`, default on; `final_ll_` reports
  the returned iterate's LL, `ll_history` keeps the true trajectory), not the last iterate under
  the non-monotone lrate schedule. It cut multi-model LL sd from 12.7x to 2.0x Fortran's at a
  matched 100-iter budget; re-measured under epic #324 (#351) the sd ratio is 1.0x with or without
  it, and a restore fired in one of 20 seeded fits (300-iteration budget only). Single-model #24 parity stays bit-exact
  (monotone => no restore). See ADR 0003.

### Degenerate-fit contract (issues #50, #306, #339)
- The `AMICA` wrapper no longer treats a degenerate fit (`stop_reason` nan_ll / singular_ll /
  nan_direction / nan_params, or restart_error under best-of-N) as usable. `fit` sets `is_fitted_`
  only on a non-degenerate stop and exposes `converged_` / `stop_reason_`; `transform` / `get_*` /
  `write_amica_output` / `save` raise a clear degenerate error instead of returning NaN sources.
- Since #306 the raw `AMICATorchNG`, `AMICAMLXNG` and `AMICA_NumPy` accessors refuse a degenerate
  fit too; since #339 every backend stops on a non-finite likelihood, update direction or
  parameter before using it.

### Component sharing (issues #60, #334)
- `share_comps` runs on all three backends. Since #334 (ADR 0007) `A` stores one component per row,
  so the metric compares de-sphered component mixing vectors (rows of `A` through `pinv(sphere)`,
  the scalp maps) across models, and a merge above `comp_thresh` ties the two components: `comp_list`
  is re-pointed and they share one mixing vector and density (Fortran `identify_shared_comps`,
  amica15.f90:1916). The A-freeze is the reference's unconditional schedule (ADR 0008, #345): from
  `share_start` on, `A` is held on every iteration with `mod(iter, share_iter) <= 5`, on every fit,
  sharing on or off.
- The `gm`-weighted `dAk/zeta` step averages a shared component over its models; byte-identical to
  the per-model update when no merge fires.
- `keep_best` (#51) is disabled under sharing (a merge changes the parameter count, so pre-/post-merge
  LLs are not comparable and restoring an earlier snapshot would revert the merge); `fit` returns the
  last, merged iterate like Fortran.
- OFF by default, and a no-op for `n_models=1`. The reference binary's own scan never merges
  (`Spinv2` is never allocated, so every similarity is NaN), but the update from a merged state
  seeded through `load_comp_list` matches PyTorch and NumPy to float64 round-off
  (`pamica/tests/test_component_rows.py`, opt-in with `AMICA_RUN_FORTRAN=1`).

### MLX first-class and reference fidelity (epic #324)
- `AMICA(backend="mlx")` and `AMICAICA(backend="mlx")` run every wrapper feature on MLX (#313);
  MLX has explicit `pcakeep`/`pcadb` (#323); one params-file reader serves every backend (#304).
- Default fits on every backend follow the reference more closely, so trajectories differ from
  0.3.3: the reference's iteration order and unconditional A-freeze (#339, #345, ADR 0008),
  `doscaling` of component rows (#333, ADR 0006), component-row storage and correct sharing (#334,
  ADR 0007), 1-based schedule gates (#335), a normalized initial `A` (#341) and the reference's
  single-precision constants (#344), each decided in one shared module (`schedule.py`,
  `component_layout.py`, `initialization.py`, `reference_constants.py`, with `rank.py`).
- `AMICAICA.apply` restores the PCA residual of rank-reduced fits (#322, on `dev`).

### Structure and infrastructure
- Module rename into `numpy_impl/` and `torch_impl/` with topic-based names (issue #34); the public
  import surface (`from pamica import AMICA, AMICA_NumPy, AMICATorchNG, AMICANative`) is stable.
- UV is the canonical environment (`pyproject.toml` + `uv.lock`); the legacy conda env is retired.
- CI is live and green on `main`: ruff lint/format, pytest (excluding slow/Fortran-binary parity),
  and a build + clean-env import matrix on Python 3.12 and 3.13. Typos check green.
- Validation harness (`validate_implementations.py --backend {torch,numpy,mlx}`, a comma-separated
  list, or `all`; default `torch`, whose report is unchanged) runs each backend against one Fortran
  reference run with the same settings and matches components via the Hungarian algorithm, on real
  sample EEG (#315). All three meet the Fortran bar on the bundled sample (re-measured under epic
  #324 in #351: LL within 2.8e-4, correlation 0.9991, Amari 0.004 from independent starts, inside the
  reference's own seed-to-seed LL sd of 2.6e-4; from a shared start LL within 1.6e-6; rows and bars
  in `docs/guides/validation.md`, run records in `.context/issue-351/`), pinned by the
  `AMICA_RUN_FORTRAN`-gated test in `test_fortran_param_forwarding.py`.

## Remaining

- **Performance benchmark:** DONE. The "runtime within 2-3x of Fortran" success criterion has been
  measured and met/exceeded: CUDA float64 ~4.5x over a 16-thread CPU, MLX ~7x on Apple Silicon
  (#77/#84; see `AGENTS.md`'s Performance section).
- **Test hardening:** `AMICA.save`/`load` and `plot_components` coverage landed (issue #15,
  closed). Single-channel / single-sample edge tests and numerical-stability regression tests
  (mincond / minlog / maxdble / mineig bounds) remain open, not tracked under #15.
