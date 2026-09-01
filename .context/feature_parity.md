# Feature Parity Report: Fortran vs PyTorch AMICA

**This file predates the MLX and NumPy backends and covers only the Fortran-vs-`AMICATorchNG`
comparison from issue #24's original parity push; it is not maintained against either backend added
since. The live, maintained backend-parity source of truth is
[`docs/guides/amica-differences.md`](../docs/guides/amica-differences.md)'s "At a glance" table,
which covers all three array backends. Treat this file as a historical snapshot.**

Compares the Fortran reference (`amica15.f90`, binary `amica15mac`) with the current PyTorch
backend `AMICATorchNG` (the sole PyTorch backend; the earlier Adam/autograd `AMICATorch` and
`AMICATorchV2` were removed in #32). The NumPy reference (`AMICA_NumPy`) is retained as an oracle
and CLI. Living status is in `../AGENTS.md` (Known Issues); ADRs are in `decisions/`.

**Headline:** single-model parity is done (LL ~ -3.40 vs Fortran -3.4018; Hungarian component
correlation ~0.997 with Newton positive-definite, 0 fallbacks). Adaptive PDF (#26), multi-model
(#27, distributional equivalence), best-iterate (#51), and the degenerate-fit contract (#50) are
done, as is component sharing (#60). Runtime vs Fortran has since been measured and the ~2-3x
criterion met/exceeded (CUDA float64 ~4.5x, MLX ~7x on Apple Silicon; see `AGENTS.md`'s
Performance section and `plan.md`'s checked runtime item) -- the "unmeasured" row below is stale.

## Core AMICA features

| Feature | Fortran | AMICATorchNG | Status | Notes |
|---------|---------|--------------|--------|-------|
| Multiple models | yes | yes | Complete | `n_models`; per-model bias `c` update (#27) |
| Mixture of Gaussians | yes | yes | Complete | `num_mix` components |
| Generalized Gaussian PDF | yes | yes | Complete | default `pdftype=0`, byte-for-byte unchanged |
| Natural gradient | yes | yes | Complete | A-update parity fix (#24) |
| Learning-rate adaptation | yes | yes | Complete | Fortran-matched lrate schedule |
| Convergence criteria | yes | yes | Complete | dLL and grad-norm checks |
| Best-iterate return | yes | yes | Complete | `keep_best` / `final_ll_` (#51) |

## Optimization methods

| Feature | Fortran | AMICATorchNG | Status | Notes |
|---------|---------|--------------|--------|-------|
| Natural gradient | yes | yes | Complete | exact-EM mixture updates |
| Newton method | yes | yes | Complete | positive-definite, 0 fallbacks on sample data |
| Newton ramping | yes | yes | Complete | Fortran-style ramp |

## Data preprocessing

| Feature | Fortran | AMICATorchNG | Status | Notes |
|---------|---------|--------------|--------|-------|
| Mean removal | yes | yes | Complete | `do_mean` |
| Sphering (whitening) | yes | yes | Complete | symmetric-ZCA (#24) |
| PCA dimension reduction | yes | yes | Complete | `pcakeep` |

## Advanced features

| Feature | Fortran | AMICATorchNG | Status | Notes |
|---------|---------|--------------|--------|-------|
| Outlier rejection | yes | yes | Complete | `do_reject` |
| Adaptive PDF selection | yes | yes | Complete | 5 `pdftype` families + ext-Infomax switcher (#26) |
| Component sharing | yes | yes | Complete | `share_comps` multi-model merge ported (#60); off by default, behavior-validated (no bit-exact oracle) |
| History tracking | yes | yes | Complete | `ll_history`, grad norms |

## Input/output

| Feature | Fortran | AMICATorchNG | Status | Notes |
|---------|---------|--------------|--------|-------|
| Binary data loading | yes | yes | Complete | `.fdt` files |
| Parameter files | yes | yes | Complete | JSON |
| Checkpoint save/load | yes | yes | Complete | NG state_dict persistence (#36); tested (#15, closed) |
| Result loading | yes | yes | Complete | via `loadmodout()` |

## Numerical stability

| Feature | Fortran | AMICATorchNG | Status | Notes |
|---------|---------|--------------|--------|-------|
| Log-space computation | yes | yes | Complete | for stability |
| Eigenvalue thresholds | yes | yes | Complete | `min_eig` |
| NaN/Inf handling | yes | yes | Complete | degenerate-fit contract refuses NaN output (#50) |
| Min/max bounds regression tests | yes | yes | **Untested** | mincond/minlog/maxdble/mineig present, no regression tests |

## Performance and hardware

| Feature | Fortran | AMICATorchNG | Notes |
|---------|---------|--------------|-------|
| OpenMP parallelization | yes | n/a | PyTorch uses tensor ops / device parallelism |
| GPU support | no | yes | CUDA / MPS / CPU automatic (float64 parity runs on CPU/CUDA) |
| Runtime vs Fortran | baseline | **measured, met/exceeded** | CUDA float64 ~4.5x over 16-thread CPU, MLX ~7x on Apple Silicon (#77/#84); see `AGENTS.md` Performance section |

## Validation status (issue #24)

| Metric | Fortran | AMICATorchNG | Match |
|--------|---------|--------------|-------|
| Final LL (single model) | -3.4018 | ~ -3.40 | yes (Jacobian-normalized) |
| Component correlation | - | ~0.997 | yes (Hungarian-matched, clears >0.95) |
| Newton | posdef | posdef | yes (0 fallbacks on sample data) |
| Multi-model per-block sufficient stats | - | bit-exact | yes |

Multi-model full-fit partition cross-correlation (~0.64 single-run) is intrinsic estimator spread,
not a defect: Fortran agrees with itself at ~0.63, and the NG partition ensemble is statistically
indistinguishable from Fortran's run-to-run distribution (#27, `.context/issue-27/`).

## Remaining roadmap

- [x] **Performance benchmark** vs the Fortran binary (CPU/CUDA/MPS); 2-3x runtime criterion
      met/exceeded (#77/#84).
- [x] **Component sharing** (`share_comps`, `share_start`/`share_iter`, `comp_thresh`) ported to
      `AMICATorchNG` (#60), off by default, behavior-validated on real data.
- [x] **Test hardening:** `save`/`load` + `plot_components` coverage landed (#15, closed).
      Single-channel/single-sample edge cases and numerical-stability regression tests remain
      open; not tracked under #15.
