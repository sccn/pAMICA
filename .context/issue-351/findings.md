# Phase 15 findings: parity re-measured under epic #324 (issue #351)

Recorded 2026-09-23.
Every user-facing parity figure of pamica, the Python port of Adaptive Mixture Independent Component Analysis (AMICA),
was re-measured with the library code of epic #324 at 09b8247 (with Phase 16's documentation, #353)
against the pinned v0.3.3 native reference binary (`pamica.native.resolver`).
No library code changed in this phase.
Phase 17 (#354, bb22df6) was merged afterwards; the reference settings are pinned so that its default changes leave these protocols as measured (see "Pins after Phase 17").

Hosts:
- Mac: Apple M4 Pro, 14 cores, 64 GB, macOS 27; MLX 0.32.0, torch 2.12.1, NumPy 2.5.0.
- hallu: Linux, 32 cores, NVIDIA RTX 4090, torch 2.12.1+cu130; shared, with the reference binary kept to 8-12 threads in total.

"Before" figures are those the documents quoted before this phase.
Where a before figure was re-measured in this phase, it used e38aa11 (Phase 6, the last commit before the epic's changes to the fit), run from a separate tree.
Paths are relative to `.context/issue-351/`.
Log-likelihood is abbreviated LL; correlations are Hungarian-matched absolute correlations of unmixing rows.

## Old and new figures

| Figure | Before | Now | Command | Commit | Host | Raw record |
|---|---|---|---|---|---|---|
| Harness, 100 iterations, seed 42, independent starts: LL gap PyTorch / NumPy / MLX | 2.9e-5 / 2.9e-5 / 3.2e-5 | 2.71e-4 / 2.72e-4 / 2.76e-4 | `validate_implementations.py --backend all` | 09b8247 (before: e38aa11) | Mac | `raw/harness_run1`, `raw/harness_run2`, `raw/harness_pre_epic_e38aa11` |
| Harness: mean / min correlation, Amari distance (PyTorch) | 0.9992 / 0.9935, 0.0037 | 0.9991 / 0.9918, 0.0038 (NumPy 0.9991 / 0.9917) | same | same | Mac | `raw/harness_run1` |
| Harness seed spread over 8 seeds: reference LL standard deviation (SD); cross-pair LL gap median (range) | not measured | 2.6e-4 (pamica 2.7e-4); 3.5e-4 (1.7e-5 to 8.2e-4) | `harness_gap.py ROOT OUT.json --spread ...` | 09b8247 | Mac | `raw/harness_gap_head.json` |
| Harness from a shared start: LL gap, correlation, Amari | 2.35e-4, 0.99999, 5.8e-4 (e38aa11) | 1.6e-6, 0.99999993, 3.9e-5 | `harness_gap.py` | 09b8247, e38aa11 | Mac | `raw/harness_gap_head.json`, `raw/harness_gap_pre_epic_e38aa11.json` |
| Harness runtimes, reference / PyTorch / NumPy / MLX (s) | 10.1 / 17.9 / 32.5 / 3.2 (same session) | 10.0 / 18.1 / 32.6 / 3.0 | harness | 09b8247, e38aa11 | Mac | `raw/harness_run1`, `raw/harness_pre_epic_e38aa11` |
| 200-iteration `amicaout` fixture: LL gap PyTorch / NumPy / MLX; correlation (min); Amari | 2.2-2.3e-4; 0.9973; 6.3e-3 (before Phase 11) | 1.42e-4 / 1.32e-4 / 1.44e-4; 0.9983 (0.982-0.983); 4.8e-3 | `fixture_parity.py ROOT BACKEND` | 09b8247 | Mac | `raw/fixture_parity_head.txt` |
| MLX float32 vs PyTorch float64 from a shared start, 100 / 200 iterations: LL gap, correlation, Amari | not characterized | 5.0e-6, 0.99999991, 5.0e-5 / 2.4e-6, 0.9999998, 6.3e-5 | `precision_agreement.py` | 09b8247 | Mac | `raw/precision_agreement.json` |
| Table 1, bundled single model: Amari over 5 run pairs | 0.006 | 0.0109 (per seed 0.0031, 0.0037, 0.0380, 0.0029, 0.0069); reference vs itself 0.0185 | `benchmarks/reproduce_table1.py --tier bundled` | 09b8247 | Mac CPU, 2842 s | `raw/table1_bundled` |
| Bundled single model, 10 seeded reference runs against the 5 pamica fits: Amari (max), correlation | not measured | 0.0044 (0.0083), 0.9988 over 50 pairs; reference vs itself 0.0054 (0.0101), 0.9985 over 45; pamica vs itself 0.0039, 0.9991 over 10 | `bundled_single_basins.py` | 09b8247 | Mac | `raw/bundled_basins` |
| Score functions; sufficient statistics | ~1e-15 | log-density 8.9e-16, score 1.8e-15; 4.5e-13 absolute (3e-16 relative) | `reproduce_table1.py --tier bundled` | 09b8247 | Mac | `raw/table1_bundled/report.json` |
| Multi-model correlation: cross; within-reference; within-pamica | 0.65; 0.64 | 0.632; 0.626; 0.638; difference +0.006, permutation p 0.88, run bootstrap 90% interval -0.002 to 0.014 | `reproduce_table1.py --tier bundled`, `equivalence_check.py` | 09b8247 | Mac | `raw/table1_bundled/report.json`, `raw/multimodel/equivalence_check.json` |
| Multi-model Amari: cross; within-reference; within-pamica | 0.163; 0.174 (p > 0.999) | 0.172; 0.166; 0.176; difference +0.005, p 0.051, interval 0.002 to 0.009 | same | 09b8247 | Mac | same |
| Multi-model ensemble LL: reference; pamica | -3.3539; -3.3629 (Kolmogorov-Smirnov (KS) p 6e-5) | -3.3543 (SD 0.002); -3.3541 (SD 0.003); KS p 0.83. The pre-epic pamica refit against the same reference fits gives -3.3627 (0.006), KS p 1e-5 | same, and `multimodel_pamica_fits.py` at e38aa11 | 09b8247, e38aa11 | Mac | `raw/table1_bundled`, `raw/multimodel` |
| keep_best: pamica/reference LL SD ratio at 100 iterations | 12.7x without, 2.0x with | 1.0x with and without; 1 restore in 20 fits (at 300 iterations, gain 4.2e-6); 0 Newton fallbacks | `keep_best_ensemble.py run`, `verify`, `report` | 09b8247 | Mac, reference seeded 0-19, 1 thread | `raw/keep_best` |
| Mean LL gap (pamica minus reference) at 100 / 200 / 300 iterations | -0.009 / 0 / +0.002 | +8.1e-4 / +2.1e-4 / +1.0e-4 (KS p 0.83 / 0.57 / 0.98) | same | 09b8247 | Mac | `raw/keep_best/ensemble.json` |
| Cross-backend LL after 25 iterations (dimsweep) | maximum pairwise ~0.003 | 1e-5 at 32 and 48 channels; 1.1e-3 at 70 channels (k≈6); NumPy (unseeded) 4.2e-3 | `benchmarks/benchmark_dimsweep.py` | 09b8247 | Mac, hallu | `raw/dimsweep` |
| Dimsweep timing at 70 channels, e38aa11 / 09b8247 (ms per iteration) | MLX 33.3, CPU float64 190.8, MPS 165.1 | MLX 33.1, CPU float64 189.4, MPS 166.7 | same | e38aa11, 09b8247 | Mac (quiet) | `raw/dimsweep/timing70_*.json` |
| Newton, independent seeds, 70 channels, 2000 iterations | seed 42: 10-11 of 70 components below 0.9; other seeds ~0.995 | see the Newton table below | `run_newton_hallu.sh`, `newton_seeds.py compare` | 09b8247 | hallu (CUDA float64; reference 10 threads) | `raw/newton` |
| Newton from a shared start: correlation (min), Amari, LL gap | 0.9974 (0.9475) | 0.9999998 (0.9999964), 3.8e-5, 7.6e-8 | `newton_seeds.py pamica/fortran --fixinit` | 09b8247 | hallu | `raw/newton` |
| `share_comps` end to end (300 iterations, threshold 0.95) | 3 merges, LL -3.3416 | 1 merge at iteration 100 (\|cos\| 0.970), LL -3.3410 (-3.3393 without sharing); 63 of 64 components | `share_comps_counts.py` | 09b8247 | Mac | `raw/share_comps` |
| Early scans at iteration 8 / 20, thresholds 0.95 and 0.99 | 32, 24 / 24, 5 | 30, 17 / 22, 5 | same | 09b8247 | Mac | `raw/share_comps` |
| Reference-formula scan on the reference's own state | 32, 32, 30 | 32, 32, 29 (identical `comp_list`s; 8-iteration states agree to 5.2e-6 in A and 1.6e-7 in LL) | same | 09b8247 | Mac | `raw/share_comps` |
| Table 1 external tier LL (on002718, 5 seeds) | gap ~0.0003 at -3.6993 | reference -3.699346 (SD 6.5e-5), pamica -3.699341 (4.1e-5), gap 5.6e-6 | `reproduce_table1.py --tier external --threads 16` | 09b8247 | hallu, CUDA float64 (lead's run, 26185 s) | `raw/table1_external` |
| Table 1 external correlation; Amari | 0.998 (reference vs itself ~0.999) | 0.99965 (SD 0.00048, min component 0.9602), reference vs itself 0.99827 (min 0.9167); Amari 0.00126 (SD 0.00062), reference vs itself 0.00249 | same | 09b8247 | hallu | `raw/table1_external/report.json` |

External tier per seed (mean correlation / min correlation / Amari):
201 0.9999 / 0.9996 / 0.0008; 202 0.9987 / 0.9602 / 0.0024; 203 0.9999 / 0.9985 / 0.0012; 204 1.0000 / 0.9998 / 0.0007; 205 0.9998 / 0.9955 / 0.0012.
The reference fits took 1972-2559 s and the pamica fits 1886-5013 s.

Newton pairs (mean / min correlation / components below 0.9):

| Pair | Mean | Min | Below 0.9 |
|---|---:|---:|---:|
| pamica 42 vs reference 1 | 0.963 | 0.667 | 8 |
| pamica 42 vs reference 2 | 0.986 | 0.858 | 3 |
| pamica 13 vs reference 1 | 0.996 | 0.946 | 0 |
| pamica 13 vs reference 2 | 0.981 | 0.773 | 3 |
| pamica 7 vs reference 1 | 0.995 | 0.943 | 0 |
| pamica 7 vs reference 2 | 0.982 | 0.786 | 3 |
| reference 1 vs reference 2 | 0.985 | 0.841 | 2 |
| pamica 42 vs pamica 13 | 0.962 | 0.658 | 7 |

Final LLs: reference seed 1 -3.697804 and seed 2 -3.697709; pamica seed 42 -3.697569, 13 -3.697816 and 7 -3.697812; shared start -3.697812.
The shared-start reference run matches reference seed 1 at 0.996 (min 0.955).
MLX float32 (supplementary, Mac): seed 7 stopped on the `lrate` floor after 952 iterations at LL -3.698091 (CUDA float64 -3.697913 at the same iteration), after 121 decreases;
it matches CUDA seed 7 at 0.985 (min 0.907) and reference seed 1 at 0.983 (min 0.905).
MLX seed 42 (928 iterations) matches CUDA seed 42 at 0.955, with 10 components below 0.9 (`raw/newton_mlx`).

## Harness: why the independent-start gap grew

The harness fits pamica and the reference from independent random starts at one seed,
so its LL gap is one draw from the start-to-start spread.
Over 8 seeds the reference's own final LL has SD 2.6e-4 and range 8.0e-4, pamica's SD 2.7e-4 and range 8.0e-4,
and the cross-implementation gaps have median 3.5e-4 (1.7e-5 to 8.2e-4).
Before the epic, seed 42 happened to fall near the low end of that spread (2.9e-5);
the epic changed the initial draw (the normalized initial mixing matrix, #341) and the trajectory, and seed 42 now falls near its middle (2.71e-4).
The lead reproduced the binary's LL at seeds 42, 6 and 3.
From a shared start, the reference loaded with pamica's initial state (`native_oracle.run_seeded_reference`), the gap is 1.6e-6, with correlation 0.99999993 and Amari 3.9e-5;
the same comparison at e38aa11 gives 2.35e-4, 0.99999 and 5.8e-4.
The shared state's orientation was checked by the first-iteration LL, which agrees to 9e-16 as loaded and differs by 1.3e-5 when transposed (`raw/orientation_check.txt`).
The documents now give the independent-start figure with the reference's spread beside it, and the shared-start figure.

## 70-channel dimsweep: round-off sensitivity

At 32 and 48 channels the backends that share a start agree to 1e-5 in LL after 25 iterations.
At 70 channels on the 30,000-frame excerpt, k = frames / channels² ≈ 6, the five seeded backends spread over 1.1e-3:
MLX -3.21643, CPU float64 -3.21746, MPS -3.21638, CUDA float64 -3.21723, CUDA float32 -3.21689.
The lead's perturbation check measured the sensitivity directly:
changing one sample of the 70-channel data by 1e-9 µV changes the CPU float64 LL at iteration 25 by 4.4e-4,
so differences of this size arise from round-off amplification on data this poorly determined.
The NumPy backend is unseeded in this benchmark and differs by 4.2e-3 (-3.22054).

## Protocol changes and why

1. Harness: unchanged.
   The seed-spread and shared-start diagnostics (`harness_gap.py`) were added because the single-seed independent-start gap moved tenfold and needed an interpretation.
2. `benchmarks/reproduce_table1.py`: every reference setting is spelled out in `REFERENCE_SETTINGS`.
   These are the bundled `input.param` values, which `AMICANative`'s defaults mirrored when Table 1 was measured, plus `do_approx_sphere 1`.
   The ensemble's pamica fit pins `lrate=0.05`.
   Phase 17 moved the `AMICANative` and wrapper defaults, and the pins keep Table 1's protocol as it was measured.
3. Bundled single-model tier: the five-pair protocol is unchanged.
   The 50-pair basin study was added because one of the five reference runs (seed 303, LL -3.4006 against about -3.3996 for the others) sets the five-pair Amari mean.
4. Multi-model ensemble: the protocol is unchanged (20 fits per implementation, 2 models, 3 mixture components, 100 iterations, bundled sample; the reference draws its own clock-based seeds).
   The ±0.05 margin is checked with a run-level bootstrap 90% interval of the difference.
   The earlier pairwise two one-sided tests (TOST) treated 190 or 400 pairwise values as independent, although each run appears in about 39 pairs; the #27 record had already retired it in favor of the run-level permutation test.
   The multi-model Amari distance takes the better of the two model pairings.
   Over the 780 run pairs of the new ensemble that choice lowers the mean by 0.0185 (median 0) and swaps the pairing in 333 pairs (`raw/multimodel/pairing_correction.txt`).
   The reference runs with `invsigmin` 0.0 and pamica with 1e-8; `validation.md` states this.
5. keep_best ensemble: the reference is seeded 0-19 and single-threaded, so each run is reproducible (the #51 runs were clock-seeded).
   The 100- and 200-iteration budgets are read from prefixes of each 300-iteration run.
   Separate 100-iteration runs and pamica histories agree exactly with those prefixes (`raw/keep_best/verify.json`, all differences 0.0).
6. Newton: pamica on CUDA float64 at seeds 42, 13 and 7, against the pinned v0.3.3 binary seeded 1 and 2.
   The data are the full 70-channel ds002718 recording, rounded to float32.
   The early stops are off on both sides, so both run 2000 iterations.
   A shared-start pair (A = I, mu = linspace(-1, 1, 3), sbeta = 1, rho = 1.5, alpha = 1/3) replaces #145's matched-initialization run.
   The #145 comparison used the gfortran `amica15_linux` build (SHA 43b9f0e3...), which draws a new start on every unseeded run:
   two runs differ after one iteration in LL (-3.51144 against -3.51323) and in A (max 9.6e-3; `raw/seedcheck_issue145_binary`).
   Both reference-vs-reference figures are therefore reported.
7. Dimsweep and `share_comps`: unchanged; the counts script re-runs the documented scans.

## Why the multi-model LL gap closed

The earlier -0.009 ensemble gap came mostly from the code before the epic.
Refitting the pamica half with e38aa11 against the same 20 reference fits gives -3.3627 (SD 0.006, KS p 1e-5).
7 of those 20 fits stopped early on the pre-#339 `min_dll` check (mean -3.3679).
The 13 full-length fits average -3.3600; the difference that remains is attributed to that code's update rule.
With the epic's code the ensemble gap is +2e-4 (KS p 0.83).
The seeded keep_best ensemble shows the same at matched budgets: +8.1e-4, +2.1e-4 and +1.0e-4 at 100, 200 and 300 iterations, with equal or smaller SD for pamica.
The documents no longer attribute the gap to convergence speed.
On Amari distance, the pre-epic ensemble was tighter (within-pamica 0.151, cross 0.160, p 0.998); the current one is slightly wider than the reference's own (0.176 and 0.172 against 0.166, p 0.051).

## share_comps details

End to end, 300 iterations at threshold 0.95: 1 merge at iteration 100 (|cos| 0.970), LL -3.3410; -3.3393 with sharing off; 63 of 64 components kept.
Early scans: at iteration 8, 30 merges (0.95) and 17 (0.99); at iteration 20, 22 and 5; at 0.9, 31 and 30.
The early mass-merge collapse: 28 merges, gm2 0.572 to 2.18e-3 to 9.0e-4, second scan 2.44e-4, a non-finite LL at iteration 23.
The gated freeze-free oracle gives 4.222e-3 on both sides.

## Pins after Phase 17

Before Phase 17, pinned and unpinned calls wrote byte-identical `input.param` files at all 8 native call sites, with a negative control (`pin_check.py`, `raw/pin_check.json`).
The pinned files were saved (`raw/pin_check_pinned_params_pre_phase17.json`).
After the merge of bb22df6:
- The 3 seeded-runner calls are byte-identical.
- The 5 `AMICANative` calls write the same settings in Phase 17's key order, plus `do_approx_sphere 1`, the binary's compiled value (`amica15_header.f90:13`); nothing else is added, removed or changed (`raw/pin_check_post_phase17.json`).
- `pin_run_check.py` ran the binary on the pre- and post-Phase-17 forms of the file for the single-model, multi-model and Newton configurations: bundled sample, seeded, single-threaded, 20 or 60 iterations.
  All 13 numeric output files are byte-identical, and `out.txt` matches once the per-iteration wall-clock times are removed (`raw/pin_run_check_post_phase17.json`).
- `REFERENCE_SETTINGS` now pins `do_approx_sphere` as well.
- Without the pins, 15 settings of the native calls would take Phase 17's defaults, among them `lrate` 0.1, `do_newton` 0, `newt_start` 20, `invsigmin` 1e-4 and `max_decs` 5.

## Gates

Before and after the Phase 17 merge:
- ruff check and format, `ty check .`, typos and `mkdocs build --strict` are clean.
- pytest (`CI=1`, 8 workers, no coverage): 1955 passed, 40 skipped, 3 xfailed before the merge; 1974 passed, 40 skipped, 3 xfailed after it (`raw/pytest_full.log`, `raw/pytest_full_post_phase17.log`).
- `AMICA_RUN_FORTRAN=1` on `test_fortran_param_forwarding.py` and the two early-merge oracles: 37 passed both times (`raw/gated_tests.log`, `raw/gated_tests_post_phase17.log`).
