# Progress Summary

Internal status snapshot for the PyTorch AMICA effort. The living source of truth for parity
status is the **Known Issues** section of `../AGENTS.md`; this file summarizes what has landed and
what remains as of the v0.1.0 preparation.

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
  ~0.997, clearing the >0.95 gate.

### Adaptive PDF selection (issue #26)
- All five `amica15.f90` `pdftype` density families ported to `AMICATorchNG`: 0 generalized
  Gaussian (default, byte-for-byte unchanged), 2 Gaussian, 3 logistic, 4 sub-Gaussian cosh+, and
  the extended-Infomax kurtosis switcher (`pdftype=1`).
- Fixed families are bit-exact vs the literal Fortran `z0`/`fp` (~1e-15) and converge within
  ~0.005 LL of the binary. The dynamic switcher is dead code in the binary (no bit-exact oracle),
  so it is validated by real-data LL. See ADR 0002.

### Multi-model AMICA (issue #27)
- Validated by **distributional equivalence**: multi-model AMICA is not partition-identifiable, so
  the NG-vs-Fortran partition cross-correlation distribution is statistically indistinguishable
  from Fortran's own run-to-run spread (Mann-Whitney p=0.97, TOST within +/-0.05). Per-block
  sufficient stats are bit-exact vs Fortran.
- Per-model bias `c` update (`update_c`) ported to both backends, guarded to a no-op for
  `n_models=1` so single-model parity stays bit-exact. See `.context/issue-27/`.

### Best-iterate safeguard (issue #51)
- `AMICATorchNG.fit` returns the highest-LL iterate (`keep_best`, default on; `final_ll_` reports
  the returned iterate's LL, `ll_history` keeps the true trajectory), not the last iterate under
  the non-monotone lrate schedule. Cuts multi-model LL sd from 12.7x to 2.0x Fortran's at a
  matched 100-iter budget. Single-model #24 parity stays bit-exact (monotone => no restore). See
  ADR 0003.

### Degenerate-fit contract (issue #50)
- The `AMICA` wrapper no longer treats a degenerate fit (`stop_reason` nan_ll / singular_ll) as
  usable. `fit` sets `is_fitted_` only on a converged fit and exposes `converged_` /
  `stop_reason_`; `transform` / `get_mixing_matrix` / `get_unmixing_matrix` / `save` raise a clear
  degenerate error instead of returning NaN sources.

### Component sharing (issue #60)
- `share_comps` ported to `AMICATorchNG`: on the `share_start`/`share_iter` schedule, components
  that are near-collinear across different models (cosine angle of their de-sphered mixing columns
  above `comp_thresh`) are merged into one shared mixing column and density, with an A-freeze for
  ~6 iterations after each merge (Fortran `identify_shared_comps`, amica15.f90:1898).
- The M-step already sums sufficient statistics through `comp_list` (index_add), so shared
  components update jointly; the A-update was refactored to accumulate columns as Fortran's
  `gm`-weighted average (`dAk/zeta`, so shared columns are averaged, not summed) -- byte-identical
  to the per-model update when unshared.
- `keep_best` (#51) is disabled under sharing (a merge changes the parameter count, so pre-/post-merge
  LLs are not comparable and restoring an earlier snapshot would revert the merge); `fit` returns the
  last, merged iterate like Fortran.
- OFF by default, and a no-op for `n_models=1`, so single-model (#24) and default multi-model (#27)
  results are byte-for-byte unchanged (verified by the full torch suite). No bit-exact oracle: the
  reference's `Spinv2` similarity metric is *declared but never allocated* (unrunnable, like the dead
  `do_choose_pdfs`, #26), so the intended algorithm is implemented and behavior-validated. See
  `tests/torch_tests/test_ng_sharing.py`.

### Structure and infrastructure
- Module rename into `numpy_impl/` and `torch_impl/` with topic-based names (issue #34); the public
  import surface (`from pamica import AMICA, AMICA_NumPy, AMICATorchNG`) is stable.
- UV is the canonical environment (`pyproject.toml` + `uv.lock`); the legacy conda env is retired.
- CI is live and green on `main`: ruff lint/format, pytest (excluding slow/Fortran-binary parity),
  and a build + clean-env import matrix on Python 3.12 and 3.13. Typos check green.
- Validation harness (`validate_implementations.py`) runs the PyTorch (NG) backend against the
  Fortran binary (`amica15mac`) and matches components via the Hungarian algorithm, on real sample
  EEG; it does not exercise the NumPy or MLX backends. NumPy-vs-Fortran parity lives in pytest
  (`test_sample_data_numpy_vs_fortran`), MLX validation in `mlx_tests/` plus the cross-backend
  suites (extending the harness itself is tracked as issue #315).

## Remaining before / around v0.1.0

- **Performance benchmark:** DONE. The "runtime within 2-3x of Fortran" success criterion has been
  measured and met/exceeded: CUDA float64 ~4.5x over a 16-thread CPU, MLX ~7x on Apple Silicon
  (#77/#84; see `AGENTS.md`'s Performance section).
- **Test hardening:** `AMICA.save`/`load` and `plot_components` coverage landed (issue #15,
  closed). Single-channel / single-sample edge tests and numerical-stability regression tests
  (mincond / minlog / maxdble / mineig bounds) remain open, not tracked under #15.
