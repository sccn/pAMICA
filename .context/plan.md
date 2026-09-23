# pamica Development Plan

## Project Overview
**Goal:** Python (PyTorch) implementation of AMICA that reproduces the Fortran binary's results within tolerance.
**Stack:** Python 3.12+, PyTorch (MPS/CUDA/CPU), NumPy/SciPy, Fortran reference.
**Definition of done:** Component correlation >0.95 with Fortran, LL within ~1% of Fortran, no NaN/Inf, runtime within ~2-3x of Fortran.

## Development Tasks
<!-- Status markers: [ ] pending, [~] in progress, [x] complete -->

### Priority 1: Parity blockers — DONE (#24)
- [x] Fix likelihood sign/scaling vs Fortran (the ~13x factor was a pre-parity basic-backend
      artifact; NG uses Jacobian LL, single-model LL ~ -3.40 vs Fortran -3.4018)
- [x] Stabilize Newton optimization (ported from NumPy, positive-definite, 0 fallbacks on sample data)
- [x] Add numerical-stability bounds from Fortran (`mincond`, `minlog`, `maxdble`, `mineig` present;
      regression tests still owed, see Priority 3)
- [x] Add NaN/Inf checks (degenerate-fit contract #50 refuses NaN output instead of suppressing it)
- [x] Ensure identical initialization vs Fortran (symmetric-ZCA sphere, seed, starting mixing matrix)
- [x] Raise component correlation with Fortran to >0.95 (now ~0.997, Hungarian-matched)

### Priority 2: Missing core features
- [x] Port outlier rejection (`do_reject` / `reject_data`) to the torch backend (done in `AMICATorchNG`)
- [x] Wire up / stabilize Newton for the torch backend (done in `AMICATorchNG`, posdef, issue #24)
- [x] Adaptive PDF selection for `AMICATorchNG` (issue #26). Corrected the "no oracle" finding:
      the reference binary is `amica15mac` = `amica15.f90` (now copied into `pamica/`), which
      implements five `pdtype` density families; the repo's `amica17.f90` is a later GG-only trim.
      Ported all five (0 GG default, 2 Gaussian, 3 logistic, 4 sub-G cosh+, 1 super-G cosh-) plus the
      extended-Infomax kurtosis auto-switcher (`pdftype=1`). Fixed families are bit-exact vs the
      literal Fortran `z0`/`fp` and converge within ~0.005 LL of the binary; the dynamic switcher is
      dead code in the binary (no bit-exact oracle) so it is LL-validated. `pdftype=0` default
      unchanged. Tests in `tests/torch_tests/test_ng_pdf_families.py`.
- [x] Multi-model AMICA per-model bias `c` update (issue #27): ported to both backends, guarded
      no-op for `n_models=1`; controlled A/B shows +0.011 cross-corr, gap is intrinsic partition
      ambiguity (see `.context/issue-27/multimodel_c_update.md`).
- [x] Degenerate-fit contract (issue #50): the `AMICA` wrapper no longer marks a degenerate fit
      (`stop_reason` nan_ll/singular_ll) as usable. `fit` sets `is_fitted_` only on a converged fit
      and exposes `converged_`/`stop_reason_`; `transform`/`get_mixing_matrix`/`get_unmixing_matrix`/
      `save` raise a clear degenerate error (mirroring `state_dict`) instead of returning NaN sources.
- [x] Best-iterate safeguard (issue #51): `AMICATorchNG.fit` returns the highest-LL iterate
      (`keep_best`, `final_ll_`), not the last, so a late Newton-fallback overshoot no longer leaves
      the model below a peak it reached. Root cause was return-last, not a bad basin (the sole
      variance-driving seed peaked at -3.357 then crashed to -3.545 in its final iterations).
      Single-model #24 parity stays bit-exact (monotone => no restore). See ADR 0003,
      `.context/issue-51/`.
- [x] Component sharing (issue #60): `share_comps` multi-model reassignment ported to
      `AMICATorchNG` (de-sphered cosine-similarity merge, `share_start`/`share_iter`/`comp_thresh`
      schedule + A-freeze). OFF by default so single-model (#24)/default multi-model (#27) parity is
      byte-for-byte. No bit-exact oracle (reference `Spinv2` metric is dead code, like #26);
      behavior-validated. See `tests/torch_tests/test_ng_sharing.py`.

### Priority 3: Testing & validation
- [x] Real-data test suite exercising the PyTorch backend end-to-end (issue #7) - Phase 1, issue #10:
      suite collects cleanly and passes (documented parity xfails, 0 errors)
- [x] Integration tests comparing against Fortran outputs (`tests/torch_tests/`) - Phase 1, issue #10
- [ ] Numerical-stability regression tests (mincond/minlog/maxdble/mineig)
- [ ] Edge cases (single channel, single sample)
- [ ] `AMICA.save`/`load` and `plot_components` coverage (issue #15)
- [x] Performance benchmark: NG runtime vs Fortran binary (#77/#84): CUDA float64 ~4.5x over a
      16-thread CPU; MLX ~7x on Apple Silicon; runtime criterion met (see below)

### Infrastructure / migration
- [x] Migrate environment from conda `torch-312` to UV; declare the PyTorch stack in `pyproject.toml`
      (Phase 1, issue #10: `torch`, `pytest`, `pytest-cov` added to `pyproject.toml`/`uv.lock`,
      `.python-version` pinned to 3.12 for torch/MPS compatibility)
- [x] Set up CI (see `.rules/ci_cd.md`): ruff lint/format, pytest (excluding slow/Fortran-binary
      parity), build + clean-env import matrix on Python 3.12/3.13; typos check. Green on `main`.
- [x] Fix legacy test function signatures (`load_data_file` extra parameter) - Phase 1, issue #10

## Success Criteria
- [x] Component correlation > 0.95 with Fortran (~0.997, single model)
- [x] LL convergence within 1% of Fortran (LL ~ -3.40 vs -3.4018)
- [x] No NaN/Inf during optimization (degenerate-fit contract #50)
- [x] Runtime within 2-3x of Fortran (met/exceeded: CUDA float64 ~4.5x faster, MLX ~7x, #77/#84)
- [x] Real (non-mock) test suite green

## Notes
- Correctness is defined by parity with the Fortran binary, not by convergence alone.
- Detailed feature status: `feature_parity.md`. NumPy-to-PyTorch migration is complete (see AGENTS.md).

## Release Readiness (JOSS track) — started 2026-07-10
Endgame ordering: finish the three open benchmark issues -> documentation ->
transfer to github.com/sccn -> JOSS paper. The repo will move to
`github.com/sccn/pyAMICA` (full transfer, preserving issues/PRs/history) and docs
will be hosted at `eeglab.org/pyAMICA`.

### Phase R1: Benchmark completion — DONE
- [x] #90 Data-size (k-factor) frames sweep at 70ch (PR #95). Cross-backend IC
      equivalence rises with frames/channel and plateaus ~0.98; knee between k=30-60
      (data-specific). Canonical: `.context/issue-90/ksweep_findings.md`.
- [x] #91 Spatially-distributed channel subsets (PR #99): greedy farthest-point
      (k-center) selection over real electrode 3D coords, so reduced montages are
      whole-head; `mne` formalized as a `viz` extra. `benchmarks/channel_selection.py`.
- [x] #92 EEGLAB drop-in output parity (PR #100): `write_amica_output` +
      `variance_order`; loadmodout15-readable output; MATLAB round-trip verified. Caught
      and fixed a column-major mixture-param format bug (see [[amica92-eeglab-dropin]] /
      `.context/scratch_history.md`).

### Phase R2: Documentation — content DONE, standup remains
- [x] MkDocs Material site + concepts/API/guides, `docs` extra, `docs.yml` Pages
      workflow, and community health files (CONTRIBUTING, CODE_OF_CONDUCT, CITATION.cff)
      built in #97; README de-WIP'd and modernized in #102. `site_url:
      https://eeglab.org/pyAMICA/` (numpy docstrings, git-revision-date fallback).
- [ ] **Standup (needs the transfer):** deploy Pages at `eeglab.org/pyAMICA` under the
      sccn org custom domain (`sccn/pyAMICA` project Pages at the `/pamica` subpath).

### Phase R3: Transfer to github.com/sccn — REMAINS (user)
- [ ] GitHub repo transfer (preserves issues/PRs/stars/history; auto-redirects old
      URLs). Post-transfer: update badge/repo URLs (README, CITATION.cff, paper.md),
      wire up the `eeglab.org/pyAMICA` docs deploy target.

### Phase R4: JOSS paper — DONE
- [x] `paper.md` + `paper.bib` (PR #102, #101): summary, statement of need, validation,
      state of the field, acknowledgments. Drafted via `manuscript-writing`, polished
      via `humanizer`, passed an independent `paper-review`. Authors: Shirazi
      (corresponding), Delorme, Makeig. See [[amica-release-readiness]].

### Remaining before actual JOSS submission (user)
- [ ] R3 transfer + R2 docs standup (above).
- [ ] Archived release (Zenodo/Software Heritage) with a matching version.
- [ ] PyPI distribution name (the name `pyamica`/`pamica` is taken on PyPI; the
      `import pamica` name is unaffected; deferred to release).
- [ ] Fill the `paper.md` corresponding-author ORCID / confirm co-author details at
      submission (ORCIDs set: Shirazi 0000-0001-5557-259X, Delorme 0000-0002-0799-3557,
      Makeig 0000-0002-9048-8438).
