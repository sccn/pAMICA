# pamica Instructions

## Project Context
**Purpose:** Python implementation of AMICA (Adaptive Mixture Independent Component Analysis) that reproduces the results of the reference Fortran binary. Targets EEG/EMG source separation with GPU/MPS/CPU support.
**Tech Stack:** Python 3.12+, PyTorch (primary backend, MPS/CUDA/CPU), NumPy/SciPy (legacy backend), matplotlib. Reference implementation is Fortran (`amica17.f90`, `funmod2.f90`).
**Architecture:** The scikit-learn-style `AMICA` interface wraps `AMICATorchNG` (`torch_impl/core.py`), the natural-gradient EM port that reaches Fortran parity (Newton, exact-EM mixture updates, symmetric-ZCA sphere, Jacobian LL). This is the single PyTorch backend: the earlier Adam/autograd backends (`AMICATorch`, `AMICATorchV2`) and their mixture/optimizer/PDF helper modules were removed in issue #32 as superseded. The legacy NumPy implementation (`numpy_impl/core.py`, retained as `AMICA_NumPy`) carries the same parity fixes plus baralpha and outlier rejection (`do_reject`, ported from the PyTorch backend's `good_idx` mechanism in issue #123). Correctness is defined by parity with the Fortran binary, validated by `validate_implementations.py`.

## Architecture Map
```
pamica/
├── amica.py                 # Main scikit-learn-style AMICA interface (wraps AMICATorchNG)
├── __init__.py              # Exposes AMICA (PyTorch), AMICA_NumPy (legacy), numpy_impl, torch_impl
├── torch_impl/              # PyTorch backend
│   ├── core.py              #   Natural-gradient EM port (AMICATorchNG); Fortran-parity, primary backend
│   └── utils.py             #   Preprocessing (sphering, PCA), device selection
├── mlx_impl/                # Optional MLX backend (Apple GPU; AMICAMLXNG, #76/#81)
│   └── core.py              #   float32 GPU E/M-step + CPU-stream linalg (single- & multi-model GG, NG)
├── numpy_impl/              # Legacy NumPy reference (topic-named modules, issue #34)
│   ├── core.py              #   AMICA_NumPy (incl. inlined Newton); pdf.py, data.py, load.py, viz.py, utils.py, cli.py
│   └── ...
├── blocktune.py             # Shared block-size auto-tuner policy (#232, all backends)
├── restarts.py              # Shared best-of-N restart policy (#198, all backends)
├── fortran_params.py        # Fortran input.param reader for AMICA.from_params_file (#132)
├── amica17.f90, funmod2.f90 # Fortran reference source (read-only, for parity)
├── sample_data/             # Sample EEG data + Fortran binary (amica15mac)
└── tests/                   # Tests, incl. tests/torch_tests/ (vs-Fortran parity)

validate_implementations.py  # Runs both implementations, Hungarian component matching, reports
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
(`state_dict`/`.npz` save-load) landed in epic #278 Phase 1 (#287); the remaining MLX gaps --
`keep_best` (Phase 2, #288) and outlier rejection + LLt/MIR (Phase 3, #289) -- are tracked as epic
#278.

## Key Files
- **Main interface:** `pamica/amica.py` (thin wrapper over `AMICATorchNG`)
- **PyTorch backend:** `pamica/torch_impl/core.py` (`AMICATorchNG`, natural-gradient EM,
  Fortran-parity; ADR `.context/decisions/0001-torch-backend-natural-gradient-em.md`). This is the
  only PyTorch backend; the basic `AMICATorch`/`AMICATorchV2` paths were removed in #32.
- **Validation harness:** `validate_implementations.py`
- **Fortran reference binary:** `pamica/sample_data/amica15mac`
- **Sample data:** `pamica/sample_data/`

## Current Status
- PyTorch backend with GPU/MPS/CPU support; the `AMICATorchNG` natural-gradient EM backend now
  matches the Fortran reference (LL ~ -3.40, component correlation ~0.997) with Newton enabled
  and positive-definite (issue #24).
- Validation harness runs both implementations (NG + NumPy) and matches components via the Hungarian
  algorithm.
- Newton and exact-EM updates are implemented in `AMICATorchNG` and the legacy NumPy `numpy_impl/core.py`
  (both Fortran-faithful). Adaptive PDF (#26) is DONE (all five `pdftype` families + ext-Infomax
  switcher); full multi-model matching (#27) is validated by distributional equivalence.
- See `.context/feature_parity.md` and `.context/progress_summary.md` for detailed roadmaps.

## Known Issues (parity blockers)
**Single-model parity: DONE (#24).** The natural-gradient A-update transpose fix (plus exact-EM
mixture updates, digamma rho update, symmetric-ZCA sphere, Jacobian LL) brought both `AMICATorchNG`
and the legacy NumPy `numpy_impl/core.py` to Fortran's solution (LL ~ -3.40, Hungarian-matched component
correlation ~0.997, > 0.95 gate cleared; root cause in `.context/issue-24/`). Also resolved: Newton
stability (posdef, 0 fallbacks), backend consolidation (#32/#31), NumPy CLI save/load format (#30),
NG save/load persistence (#36), and the degenerate-fit contract (#50: the `AMICA` wrapper marks a
degenerate fit unusable via `converged_`/`stop_reason_` and refuses `transform`/`get_*`/`save`,
instead of returning NaN sources).

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

**Component sharing (#60): DONE.** `share_comps` multi-model reassignment is ported to
`AMICATorchNG`: on the `share_start`/`share_iter` schedule, components near-collinear across
different models (cosine angle of their de-sphered mixing columns above `comp_thresh`) are merged
into one shared mixing column + density, with an A-freeze for ~6 iterations after each merge
(Fortran `identify_shared_comps`, amica15.f90:1916). The M-step already sums sufficient stats
through `comp_list`; the A-update was refactored to accumulate shared columns the same way
(byte-identical when unshared), and merged-away columns are frozen (avoiding 0/0 NaN that Fortran
tolerates behind its `comp_used` mask). OFF by default and a no-op for `n_models=1`, so single-model
(#24) and default multi-model (#27) parity stay byte-for-byte (full torch suite green). The A-update
is the Fortran `gm`-weighted average (`dAk/zeta`), so shared columns are averaged not summed. No
bit-exact oracle: the reference `Spinv2` metric is *declared but never allocated* (unrunnable, like
the dead `do_choose_pdfs`, #26), so it is behavior-validated (`tests/torch_tests/test_ng_sharing.py`).

**Open (non-blocking, tracked):**
- **Multi-model (#27): VALIDATED by distributional equivalence.** Multi-model AMICA is not
  partition-identifiable, so exact partition parity with Fortran is the wrong acceptance bar (the
  `>0.95` cross-corr in #27's title asks the algorithm to be more identifiable than it is). The right
  test is whether the two implementations sample the same distribution over solutions. On an
  N=20-each ensemble (real sample EEG, `n_models=2`), the NG-vs-Fortran partition cross-corr
  distribution is **statistically equivalent to Fortran's own run-to-run distribution** (Mann-Whitney
  p=0.97, TOST equivalent within ±0.05; within-Fortran/within-NG/between all ~0.63-0.64). The
  single-run ~0.64 cross-corr is intrinsic estimator spread, not a defect -- Fortran agrees with
  *itself* at 0.63. See `.context/issue-27/multimodel_distributional_equivalence.md` (+ figure).
  Supporting: per-block sufficient stats are bit-exact vs Fortran; the per-model bias `c` update
  (Fortran `update_c`: `c[i,h] = sum_t v_h*x / sum_t v_h`) is ported to both backends, guarded to a
  no-op for `n_models=1` (single-model parity stays bit-exact), see
  `.context/issue-27/multimodel_c_update.md`.
- **Best-iterate safeguard (#51): DONE.** NG's multi-model LL was ~0.02 lower and ~13x more variable
  than Fortran because `fit` returned the *last* EM iterate under the non-monotone lrate schedule; the
  variance was driven by late Newton-fallback overshoots (one seed peaked at -3.357 then crashed to
  -3.545 in its final iterations). `AMICATorchNG.fit` now returns the highest-LL iterate (`keep_best`,
  default on; `final_ll_` reports the returned iterate's LL, `ll_history` stays the true trajectory).
  At matched 100-iter budget this cuts the LL sd from 12.7x to 2.0x Fortran's; the residual ~0.009
  mean gap is convergence speed, not a worse optimum (at 200 iters NG reaches Fortran's exact mean
  -3.3541, at 300 it exceeds it -- the M-step is bit-exact vs Fortran). Single-model #24 parity stays
  byte-for-byte (monotone => no restore). Inactive under `do_reject`. See ADR 0003 and
  `.context/issue-51/`.

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
