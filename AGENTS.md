# pamica Instructions

## Project Context
**Purpose:** Python implementation of AMICA (Adaptive Mixture Independent Component Analysis) that reproduces the results of the reference Fortran binary. Targets EEG/EMG source separation with GPU/MPS/CPU support.
**Tech Stack:** Python 3.12+, PyTorch (primary backend, MPS/CUDA/CPU), NumPy/SciPy (legacy backend), matplotlib. Reference implementation is Fortran (`amica15.f90`, the source of the validation binary; `funmod2.f90`).
**Architecture:** The scikit-learn-style `AMICA` interface wraps `AMICATorchNG` (`torch_impl/core.py`) by default, or `AMICAMLXNG` (`mlx_impl/core.py`) with `backend="mlx"` (#313); `AMICATorchNG` is the natural-gradient EM port that reaches Fortran parity (Newton, exact-EM mixture updates, symmetric-ZCA sphere, Jacobian LL). This is the single PyTorch backend: the earlier Adam/autograd backends (`AMICATorch`, `AMICATorchV2`) and their mixture/optimizer/PDF helper modules were removed in issue #32 as superseded. The legacy NumPy implementation (`numpy_impl/core.py`, retained as `AMICA_NumPy`) carries the same parity fixes plus baralpha and outlier rejection (`do_reject`, ported from the PyTorch backend's `good_idx` mechanism in issue #123). Correctness is defined by parity with the Fortran binary, validated by `validate_implementations.py`.

## Architecture Map
```
pamica/
├── amica.py                 # Main scikit-learn-style AMICA interface (wraps AMICATorchNG, or AMICAMLXNG with backend="mlx")
├── __init__.py              # Exposes AMICA, AMICATorchNG, AMICA_NumPy (legacy), AMICANative, metrics, viz plots
├── torch_impl/              # PyTorch backend
│   ├── core.py              #   Natural-gradient EM port (AMICATorchNG); Fortran-parity, primary backend
│   └── utils.py             #   Preprocessing (sphering, PCA), device selection
├── mlx_impl/                # Optional MLX backend (Apple GPU; AMICAMLXNG, #76/#81)
│   └── core.py              #   float32 GPU E/M-step + CPU-stream linalg; all five pdf families, full torch-equivalent surface (epic #278)
├── numpy_impl/              # Legacy NumPy reference (topic-named modules, issue #34)
│   ├── core.py              #   AMICA_NumPy (incl. inlined Newton); pdf.py, data.py, load.py, viz.py, utils.py, cli.py
│   └── ...
├── mne_compat/core.py       # AMICAICA: MNE Raw/Epochs wrapper over AMICA (either backend), to_mne_ica, PCA residual (#322)
├── native/                  # AMICANative: runs the released Fortran binary as a backend (engine.py, resolver.py)
├── metrics/, viz.py         # MIR/PMI metrics and backend-agnostic plots
│   # Shared decisions every array backend calls (.rules/backend_parity.md):
├── rank.py                  #   numerical rank and pcakeep/pcadb policy (#223, #323)
├── schedule.py              #   1-based iteration gates: Newton, rejection, share/A-freeze, scalestep (#335, #345)
├── initialization.py        #   the reference's normalized initial A (#341)
├── component_layout.py      #   component-row A and legacy-save conversion (#334, ADR 0007)
├── reference_constants.py   #   the reference's single-precision density constants (#344)
├── blocktune.py             #   block-size auto-tuner policy (#232)
├── restarts.py              #   best-of-N restart policy (#198)
├── fortran_params.py        #   params-file reader (JSON + Fortran input.param, #132/#304) for every backend
├── amica15.f90, funmod2.f90 # Fortran reference source (read-only, for parity; amica17.f90 is a later GG-only trim)
├── sample_data/             # Sample EEG data + Fortran binary (amica15mac)
└── tests/                   # Tests, incl. tests/torch_tests/ (vs-Fortran parity)

validate_implementations.py  # Runs each backend vs Fortran (--backend), Hungarian matching, reports
```
Module names are topic-based (`core`/`pdf`/`data`/... under `numpy_impl/`,
`core`/`utils` under `torch_impl/`); the old `pamica.py`/`amica_*.py`/`amica_torch_ng.py`
prefixes were dropped in issue #34. The public import surface is stable:
`from pamica import AMICA, AMICA_NumPy, AMICATorchNG, AMICANative`. The optional MLX backend is
imported separately (`from pamica.mlx_impl import AMICAMLXNG`) so `import pamica` never
requires MLX; install it with `uv pip install mlx` or the `mlx` extra (Apple Silicon only).

## Environment Setup
Canonical environment is **UV** (per global standards). The PyTorch stack is declared in
`pyproject.toml` with `uv.lock` pinned; the legacy conda env (`torch-312`) is retired. Migration
history is in `.context/plan.md`.
```bash
uv sync                          # Install dependencies
uv run pytest                    # Run tests
uv run ruff check --fix . && uv run ruff format .
```
MPS note: run with `PYTORCH_ENABLE_MPS_FALLBACK=1` for ops MPS does not yet support. The NG backend
computes in float64 for Fortran parity, which MPS cannot represent, so parity runs use CPU or CUDA
(the `AMICA` wrapper falls back to CPU automatically when a device is not pinned).

**Performance (#63, `.context/issue-63/perf_findings.md`, `benchmarks/benchmark_gpu.py`):** the E-step
pow-dedup (dropping the unused `dpdf`) is ~-35% and bit-identical. **`block_size` default is 8192
(#216, raised from 512).** Every backend is dispatch-bound at small blocks, making this the single
largest throughput knob: ~6x on CPU float64 for the bundled sample (217 -> 36 ms/it). 8192 rather
than larger because peak block memory scales with it and 8192 stays safe at high channel counts.
Runs compared bit-for-bit against the Fortran binary must set `block_size` on both sides (the
bundled `input.param` uses 512); the trajectory shifts ~1e-6 with it, inside parity tolerance.
`do_opt_block` (#232, `pamica/blocktune.py`, all three backends, OFF by default) searches for the
per-host optimum instead of using the static default; its choice is timing-based and so
machine-dependent, which is why parity runs must leave it off and pin `block_size`. Unlike Fortran,
a candidate that cannot be allocated is skipped rather than aborting the run.
CUDA float64 is ~4.5x over a 16-thread CPU (RTX 4090, warmed) and agrees with the CPU LL
to 5 sig digits (auto-selected by the wrapper). float32 now converges reliably on
full-size data across seeds (#75 guarded the one float32-only divide-by-zero: a sample rounding an
activation to exactly 0 gave `0/0` in the mu denominator; not a summation-precision problem, so it
needs no float64 and holds on MPS) and is required on Apple GPUs (no float64). float32 is NOT a
general speedup: the #63 "5-19x" estimate was superseded by #84's matched cross-platform sweep,
which found CUDA is overhead-bound so f32==f64 (~36 ms) and the Apple-GPU win is the MLX backend,
not float32 itself (on CPU f32 is only modestly faster and scales better across cores). float32 is
~7-sig-digit, not float64-parity, so use float64 for Fortran-parity runs. CPU intra-op threads are workload-limited (~4 was the sweet spot in the measured
laptop sweep; 8+ regressed).

**Cross-platform benchmark (#77, `.context/issue-77/benchmark_findings.md`, `benchmarks/benchmark_dimsweep.py`,
real 70-ch EEG):** on Apple Silicon the **MLX backend is the GPU win: ~15-25 ms/it, flat across 16-70
channels, ~7x over torch-CPU and faster than an RTX 4090 (CUDA ~36 ms/it) at EEG scale**. The #77
PyTorch-MPS figures (162-255 ms/it, at or worse than CPU) were measured at the then-default
`block_size=512`; #216's block_size sweep (bundled 32-ch sample) found PyTorch-MPS far more
block-size-sensitive than CPU or MLX -- 431 -> 30.5 ms/it from 512 to the current 8192 default (still
behind CPU's 21.7 ms/it there), and down to 13.5 ms/it (beating CPU's 15.8) at a further-tuned
single-block size -- the whole 30504-frame sample as one block, memory-limited rather than a free win
(peak block memory scales with `block_size`, which is why 8192 stays the shipped default) -- so
"PyTorch-MPS never wins" is not a general claim. MLX stays fastest throughout the sweep, so it
remains the recommendation over `device="mps"` on Apple hardware. CUDA float64 stays the bit-safe
NVIDIA path. All backends agree on the LL to ~3 digits on real data.
Multi-model MLX (#81) also wins (~5x over torch-CPU; MPS still loses at the inherited `block_size=512`
-- not yet re-swept at 8192 like the single-model figures above). Component sharing (#263),
Newton (#264, float32, validated against a float64 torch twin -- see `.context/issue-264/`) and the
non-GG pdf families (#265, including the adaptive switcher; see `.context/issue-265/`) are all
ported; source extraction (`transform` and the mixing/unmixing/`rho` accessors) and persistence
(`state_dict`/`.npz` save-load) landed in epic #278 Phase 1 (#287); the best-iterate safeguard
(`keep_best`) landed in Phase 2 (#288); outlier rejection, the LLt-stash-backed scoring accessors,
the EEGLAB export, and MIR/PMI landed in Phase 3 (#289), and `variance_order` in the polish round.
With explicit `pcakeep`/`pcadb` (#323) and the wrappers' `backend="mlx"` (#313, epic #324), the only
remaining MLX gap vs the PyTorch backend is float32-only precision (Apple GPUs have no float64).

## Key Files
- **Main interface:** `pamica/amica.py` (thin wrapper over `AMICATorchNG`, or `AMICAMLXNG` with
  `backend="mlx"`, #313; `AMICAICA` in `pamica/mne_compat/core.py` takes the same `backend`)
- **PyTorch backend:** `pamica/torch_impl/core.py` (`AMICATorchNG`, natural-gradient EM,
  Fortran-parity; ADR `.context/decisions/0001-torch-backend-natural-gradient-em.md`). This is the
  only PyTorch backend; the basic `AMICATorch`/`AMICATorchV2` paths were removed in #32.
- **Validation harness:** `validate_implementations.py`
- **Fortran reference binary:** `pamica/sample_data/amica15mac`
- **Sample data:** `pamica/sample_data/`

## Current Status
- PyTorch backend with GPU/MPS/CPU support; the `AMICATorchNG` natural-gradient EM backend now
  matches the Fortran reference with Newton enabled and positive-definite (issue #24): against the
  bundled 200-iteration `amicaout` fixture, LL within 1.4e-4 and component correlation 0.998 on
  all three backends (re-measured under epic #324 in #351; ~0.997 when #24 closed).
- `AMICA(backend="mlx")` and `AMICAICA(backend="mlx")` (epic #324 Phase 4, #313) run the MLX backend end to
  end: fit, `from_params_file`, `pcakeep`, the #50 degenerate-fit contract, `.pt` save/load (wrapper
  `format_version` 2 records the backend; version 1 still loads as torch), EEGLAB export and MNE `apply`.
  `device`/`dtype` are torch-only; the default stays `"torch"`.
- `validate_implementations.py --backend {torch,numpy,mlx}` (a comma-separated list, or `all`; default
  `torch`, whose report is unchanged) runs each backend against one Fortran reference run with the same
  settings and matches components via the Hungarian algorithm (#315). All three meet the Fortran bar on
  the bundled sample (re-measured under epic #324 in #351: LL within 2.8e-4, correlation 0.9991, Amari
  0.004 from independent starts, where the reference's own seed-to-seed LL sd is 2.6e-4; from a shared
  start LL within 1.6e-6, correlation 0.99999993; rows and bars in `docs/guides/validation.md`), pinned
  by the `AMICA_RUN_FORTRAN`-gated test in `test_fortran_param_forwarding.py`.
- Epic #324 aligned every backend's default fit with the reference, so default trajectories differ
  from 0.3.3 (changelog warning): the reference's per-iteration order, with the exit before the update
  and the A-freeze on every fit (ADR 0008), `doscaling` of component rows (ADR 0006), component-row
  storage (ADR 0007), 1-based schedule gates, a normalized initial `A` and the reference's
  single-precision constants, each decided once in a shared module (map above).
- Newton and exact-EM updates are implemented in `AMICATorchNG` and the legacy NumPy `numpy_impl/core.py`
  (both Fortran-faithful). Adaptive PDF (#26) is DONE (all five `pdftype` families + ext-Infomax
  switcher); full multi-model matching (#27) is validated by distributional equivalence.
- See `.context/feature_parity.md` and `.context/progress_summary.md` for detailed roadmaps.

## Known Issues (parity blockers)
**Single-model parity: DONE (#24).** The natural-gradient A-update transpose fix (plus exact-EM
mixture updates, digamma rho update, symmetric-ZCA sphere, Jacobian LL) brought both `AMICATorchNG`
and the legacy NumPy `numpy_impl/core.py` to Fortran's solution (LL ~ -3.40, Hungarian-matched component
correlation ~0.997 at the time and 0.998 re-measured under epic #324, > 0.95 gate cleared; root cause
in `.context/issue-24/`). Also resolved: Newton
stability (posdef, 0 fallbacks), backend consolidation (#32/#31), NumPy CLI save/load format (#30),
NG save/load persistence (#36), and the degenerate-fit contract (#50: the `AMICA` wrapper marks a
degenerate fit unusable via `converged_`/`stop_reason_` and refuses `transform`/`get_*`/`save`,
instead of returning NaN sources; since #306 the raw backends refuse their own accessors too, and
since #339 every backend stops on a non-finite likelihood, direction or parameter before using it).

**Adaptive-PDF selection: DONE (#26).** `AMICATorchNG` now supports all five `amica15.f90`
source-density families via `pdftype`: 0 generalized Gaussian (default, unchanged), 2 Gaussian,
3 logistic, 4 sub-Gaussian cosh+, and the extended-Infomax adaptive switcher (`pdftype=1`, which
flips each source between super-Gaussian code 1 and sub-Gaussian code 4 by kurtosis sign on the
`kurt_start`/`num_kurt`/`kurt_int` schedule). Key correction to the earlier "no oracle" finding: the
validation binary is `amica15mac` = `amica15.f90` (now copied into `pamica/`), which *does*
implement the families; the repo's `amica17.f90` is a later GG-only trim, and the reference binary
was never amica17. The fixed families are bit-exact vs the literal Fortran `z0`/`fp` (~1e-15) and
converge to the binary within ~0.005 LL (Newton-matched). The dynamic `do_choose_pdfs` switch is
dead code even in amica15 (the moment buffers are never accumulated), so the auto-switcher has no
bit-exact oracle and is validated by real-data LL. `pdftype=0` stays the default and is
byte-for-byte unchanged. See `.context/decisions/` and `pamica/tests/torch_tests/test_ng_pdf_families.py`.

**Component sharing (#60, #334): DONE.** `share_comps` runs on all three backends. Every backend
stores `A` as `(n_comps, n_channels)`, one component per row (ADR 0007, #334), so a `comp_list` id
names the same component in `A` and in the densities. On the `share_start`/`share_iter` schedule,
components whose de-sphered mixing vectors (rows of `A` mapped through `pinv(sphere)`, i.e. the
scalp maps) are near-collinear across models, with cosine above `comp_thresh`, are merged: the merge
re-points `comp_list`, so the two sources share one component (mixing vector and density), and the
`gm`-weighted `dAk/zeta` step averages that component over the models that share it
(Fortran `identify_shared_comps`, amica15.f90:1916). Merged-away rows are frozen. The A-freeze is
the reference's schedule and applies to every fit, sharing on or off (ADR 0008, #345): from
`share_start` on, `A` is held on every iteration with `mod(iter, share_iter) <= 5` (100-105,
200-205, ... by default). Sharing is OFF by default; byte-identical when no merge fires. The reference
binary's own scan never merges (`Spinv2` is never allocated, so every similarity is NaN), but the
update from a merged state seeded through `load_comp_list` matches it to float64 round-off
(`pamica/tests/test_component_rows.py`, `AMICA_RUN_FORTRAN=1`).

**Open (non-blocking, tracked):**
- **Multi-model (#27): VALIDATED by distributional equivalence.** Multi-model AMICA is not
  partition-identifiable, so exact partition parity with Fortran is the wrong acceptance bar (the
  `>0.95` cross-corr in #27's title asks the algorithm to be more identifiable than it is). The right
  test is whether the two implementations sample the same distribution over solutions. Re-measured
  under epic #324 (#351, N=20 each, real sample EEG, `n_models=2`, pinned v0.3.3 binary): mean
  pairwise cross-corr within-Fortran 0.626, within-pamica 0.638, between 0.632; between minus
  within-Fortran +0.006 (inside the ±0.05 margin; run-level permutation p=0.88; by Amari distance
  +0.005, p=0.051, pamica's ensemble spreading slightly more than Fortran's); final LL -3.3541 vs
  -3.3543 (Kolmogorov-Smirnov (KS) p=0.83). The single-run ~0.63 cross-corr matches Fortran's agreement with itself.
  See `docs/guides/validation.md`, `.context/issue-351/` and, for the pre-epic record,
  `.context/issue-27/multimodel_distributional_equivalence.md`.
  Supporting: per-block sufficient stats are bit-exact vs Fortran; the per-model bias `c` update
  (Fortran `update_c`: `c[i,h] = sum_t v_h*x / sum_t v_h`) is ported to both backends, guarded to a
  no-op for `n_models=1` (single-model parity stays bit-exact), see
  `.context/issue-27/multimodel_c_update.md`.
- **Best-iterate safeguard (#51): DONE.** NG's multi-model LL was ~0.02 lower and ~13x more variable
  than Fortran because `fit` returned the *last* EM iterate under the non-monotone lrate schedule; the
  variance was driven by late Newton-fallback overshoots (one seed peaked at -3.357 then crashed to
  -3.545 in its final iterations). `AMICATorchNG.fit` now returns the highest-LL iterate (`keep_best`,
  default on; `final_ll_` reports the returned iterate's LL, `ll_history` stays the true trajectory).
  At a matched 100-iter budget it cut the LL sd from 12.7x to 2.0x Fortran's (the residual ~0.009 mean
  gap was then read as convergence speed). Re-measured under epic #324 (#351, seeded 20-run ensemble
  against the pinned binary): the late overshoots are gone (largest LL dip 1.2e-5 in 20 trajectories,
  no Newton fallbacks), the sd ratio is 1.0x at 100 iterations with or without `keep_best`, a restore
  fired in one of 20 seeded fits, at the 300-iteration budget only (gain 4.2e-6), and the mean gap is
  +8e-4 at 100 iterations and within 2e-4 at 200 and 300. The pre-epic code on the same seeds
  reproduces the old gap; 7 of its 20 fits stopped early on a `min_dll` check that counted LL dips as
  small gains (fixed in #339), and its full-length fits trail by 0.006. `keep_best` stays on. Single-model #24 parity stays byte-for-byte (monotone => no
  restore). Inactive under `do_reject`. See ADR 0003 and `.context/issue-351/`.

## Development Workflow
1. **Check context:** `.context/plan.md` for current tasks and priorities.
2. **Understand deeply:** `.context/ideas.md` (design), `.context/research.md` (Fortran-vs-Python parity).
3. **Branch:** `gh issue develop <issue-number>` (create an issue first, except minor fixes).
4. **Code:** Follow patterns in `.rules/`.
5. **Test:** Real data only (sample EEG + Fortran binary); see `.rules/testing.md`.
6. **Document failures:** Log dead ends in `.context/scratch_history.md`.
7. **Commit:** Atomic, <50 chars, no emojis, no AI attribution.
8. **PR + review:** Run `/review-pr` and address all findings (`.rules/code_review.md`).
9. **Merge:** CI green first (see below), then **squash merge** (`gh pr merge <n> --squash --delete-branch`).

## [CRITICAL] Core Principles
- **NO MOCKS:** Validate against real sample data and the Fortran binary, never fabricated data. Details: `.rules/testing.md`.
- **No technical debt carried forward:** Address ALL PR review findings; replace, don't deprecate. Details: `.rules/code_review.md`.
- **Numerical parity is the spec:** Correctness means matching Fortran output within tolerance, not merely "converging".
- **No one-off backends:** A behavior change lands in every backend that can support it, in the same PR, with a
  cross-backend test. Deliberate divergences from Fortran are recorded in `docs/guides/amica-differences.md`.
  Details: `.rules/backend_parity.md`.
- **Squash merge every PR** by default; use a regular merge commit **only** for PRs coming from an epic branch (to preserve the epic's per-phase history). Never merge until CI is green.

## [NEVER DO THIS]
- Never use mocks, stubs, or fake/synthetic data as the basis for correctness tests.
- Never use `pip`, `conda`, or `virtualenv`; use UV for Python.
- Never commit secrets, `.env` files, or credentials.
- Never leave empty catch blocks or silent failures (this codebase already has NaN-suppression risks).
- Never add backward-compatibility shims; replace directly.
- Never add a TODO without a linked issue.
- Never use emojis in commits, PRs, or code.

## [REFERENCE] Rules Directory
- `.rules/testing.md` - NO MOCK policy, coverage
- `.rules/backend_parity.md` - No one-off backends; shared decisions, cross-backend tests
- `.rules/python.md` - UV, ruff, ty
- `.rules/git.md` - Commit/branch conventions
- `.rules/code_review.md` - PR review toolkit and checklist
- `.rules/ci_cd.md` - GitHub Actions setup
- `.rules/documentation.md` - Docs conventions
- `.rules/self_improve.md` - Learning/pattern extraction
- `.rules/serena_mcp.md` - Serena MCP code intelligence (when available)

## Context Files
- `.context/plan.md` - Current tasks, phases, priorities
- `.context/research.md` - Fortran-vs-Python parity analysis (data structures, subroutines, divergence causes)
- `.context/ideas.md` - PyTorch design decisions and library options
- `.context/scratch_history.md` - Debugging notes, failed attempts, lessons
- `.context/feature_parity.md` - Feature comparison and implementation roadmap
- `.context/progress_summary.md` - Achievements and validation metrics snapshot
- `.context/decisions/` - Architecture Decision Records (copy `0000-template.md` to start one)

## Project Docs (top-level)
- `README.md` - Overview and quick start

---
Remember: parity with the Fortran reference is the definition of done. Check `.rules/` for detailed guidance.
