# Validation & Parity

**Correctness in pamica is defined as parity with the reference Fortran binary,
not merely as convergence.** A run is correct when it reproduces the Fortran output within numerical tolerance.
This page collects the full verification evidence: bit-exact score functions, single-model parity,
multi-model distributional similarity, cross-platform device and precision invariance,
the EEGLAB drop-in round-trip, and the remaining validated behaviors.
Every result uses real EEG and the reference Fortran binary; none uses synthetic data.
The multi-model and score-function checks use only the bundled EEGLAB tutorial sample (32 channels,
30504 samples at 128 Hz, ~238 s); the single-model headline additionally uses an external recording
(OpenNeuro ds002718 sub-002, 70 channels) at $k=\text{frames}/\text{channels}^2\approx153$, well past
the ~60 threshold where cross-backend agreement plateaus (below), since the bundled sample sits at
that threshold's boundary ($k\approx30$). Extending to a multi-subject, multi-dataset validation is
planned future work, not yet done.
Throughout, IC abbreviates independent component and LL log-likelihood.

## Validation at a glance

| What was checked | How | Result |
|---|---|---|
| Source-density score and log-density (non-GG families) | vs the literal `amica15.f90` expressions | bit-exact ($<10^{-12}$) |
| Per-block sufficient statistics and one M-step | vs Fortran | bit-exact ($\sim\!10^{-15}$) |
| Single-model solution (`do_newton=0`, $k\approx153$) | log-likelihood, component correlation vs Fortran | LL within ~0.0005 of $-3.6993$; correlation 0.998 |
| Single-model solution (`do_newton=0`, bundled, $k\approx30$) | Amari distance vs Fortran | 0.011 over the protocol's 5 run pairs, one of whose reference runs ended in another basin; 0.004 over 50 pairs (Fortran vs Fortran 0.005) |
| Every backend against the reference (harness defaults, bundled) | `validate_implementations.py --backend all`: PyTorch, NumPy and MLX each vs Fortran | from independent starts: LL within 2.8e-4 (the reference's own seed-to-seed standard deviation is 2.6e-4), correlation 0.9991, Amari distance 0.004, for all three; from a shared start: LL within 1.6e-6, correlation 0.99999993 ([per-backend rows](#parity-rows-per-backend)) |
| Multi-model solution | distributional similarity over 20-run ensembles | between-implementation correlation within 0.006 of Fortran's own run-to-run agreement (one-sided permutation $p = 0.88$; Amari distance $p = 0.051$); final log-likelihood $-3.3541$ against $-3.3543$ (KS $p = 0.83$) |
| Device and precision invariance | same independent components across CPU/CUDA/MPS/MLX, float32/float64, Linux/macOS | identical (1.000) across all eight torch/MLX combinations |
| Cross-backend log-likelihood | same settings and start, 25 iterations, every backend | agree to 1e-5 at 32 and 48 channels; at 70 channels ($k\approx6$) within 1.1e-3, the size of a round-off perturbation's effect ([details](#cross-backend-log-likelihood-agreement-single-model)) |
| EEGLAB output | `write_amica_output` round-trip through `loadmodout15` | single-model bytes are an exact serialization; loads with correct layout |
| Degenerate fits | non-finite log-likelihood, update direction or parameters | refused, never returned as NaN sources |

## The validation harness

`validate_implementations.py` runs pamica's backends and the Fortran reference binary on the bundled sample EEG,
matches components across implementations with the Hungarian algorithm,
and reports log-likelihood and per-component correlation. It always uses real sample data and the Fortran binary, never synthetic data.
Conformity with Fortran is measured with two metrics used throughout this page: Hungarian-matched component correlation,
and the Amari distance (`amari_distance` in `validate_implementations.py`),
a standard unmixing-matrix comparison metric (Amari, Cichocki & Yang, 1996) that is permutation- and scale-invariant by construction and so needs no assignment step.

### Running it per backend

`--backend` selects which pamica backends are compared against the reference (issue #315):

```bash
uv run python validate_implementations.py                  # PyTorch only (the default)
uv run python validate_implementations.py --backend numpy  # the legacy NumPy backend
uv run python validate_implementations.py --backend mlx    # Apple Silicon; needs `uv sync --extra mlx`
uv run python validate_implementations.py --backend all    # torch, numpy and mlx
```

A comma-separated list such as `--backend torch,mlx` also works.
Every backend gets the same settings: `sample_params.json` read through the shared canonical reader (`block_size=512`, Newton on from iteration 50),
plus `--max-iter` (default 100) and `--seed` (default 42).
The reference runs once, with the same settings, the seed pinned and one thread, and each backend is compared against that one run.
NumPy receives the settings under its own key names, through the backend's own translation table;
PyTorch and MLX both run through `AMICA(backend=...)`, whose constructors share the canonical names.
A setting a backend cannot apply is named in a warning rather than dropped silently.

Each backend gets its own report: `validation_report.txt` for PyTorch (the name the default run has always used),
and `validation_report_numpy.txt` or `validation_report_mlx.txt` for the others.
Passing `--backend` explicitly also prints a one-row-per-backend summary with runtimes and saves it as `parity_summary.md`;
the default run prints exactly the report it always has.
`--backend mlx` on a host without MLX stops before running anything, with exit status 2 and the install hint.

The reference binary is resolved as before:
by default the native engine (`PAMICA_NATIVE_BINARY`, or the release binary for the host, cached after the first download),
falling back, with a warning, to the bundled macOS x86_64 `amica15mac`, which cannot be seeded;
`--fortran-binary PATH` runs a specific binary.

### Parity rows per backend

Measured on 2026-09-23 with the code of epic #324 (issue #351) on an Apple M4 Pro (14 cores, 64 GB, macOS 27; MLX 0.32.0, PyTorch 2.12.1, NumPy 2.5.0)
against the v0.3.3 release native engine (`amica15-macos-arm64`, SHA-256 `c8b2ac7f...`), with the harness defaults:

```bash
uv run python -c "from pamica.native import resolver; print(resolver.resolve(version='v0.3.3'))"
uv run python validate_implementations.py --backend all \
  --fortran-binary ~/.cache/pamica/bin/v0.3.3/amica15-macos-arm64
```

The reference is pinned to that explicit release binary (the first line downloads it and verifies its checksum) rather than to the default resolution,
because the resolver caches its `latest` download without refreshing it,
so on a machine that fetched an earlier release the default can run an older binary than the one these rows were measured with.

| Backend | Precision | Final LL | LL difference from Fortran | Mean matched correlation | Min matched correlation | Amari distance | Runtime (s) | Expected bar |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Fortran (reference) | float64 | -3.411274 | | | | | 10.0 | the reference |
| PyTorch (`AMICA`) | float64 | -3.411003 | 0.000271 | 0.9991 | 0.9918 | 0.0038 | 18.1 | correlation > 0.95, Amari < 0.05, LL difference < 0.005 |
| NumPy (`AMICA_NumPy`) | float64 | -3.411003 | 0.000272 | 0.9991 | 0.9917 | 0.0038 | 32.6 | the PyTorch bar, and final LL within 1e-5 of PyTorch's |
| MLX (`AMICA(backend="mlx")`) | float32 | -3.410998 | 0.000276 | 0.9991 | 0.9917 | 0.0038 | 3.0 | the PyTorch bar, and final LL within 1e-4 of PyTorch's |

Every backend meets its bar.
All four runs stop at the 100-iteration budget, so the rows compare matched trajectories rather than converged optima;
the converged single-model evidence is in the next section.
The two float64 backends agree with each other to about 1e-6 in log-likelihood.
MLX computes in float32 (about seven significant digits per operation), so its bar against PyTorch is looser:
it lands within about five significant digits of the float64 likelihood, which is float32 consistency, not float64 parity.
Runtime is one run's wall-clock time for the fit alone and varies by about a third between runs on the same host;
the reference's includes process start-up and is single-threaded.
Epic #324 left the runtimes unchanged: the code before its changes to the fit (e38aa11), run in the same session, took 10.1, 17.9, 32.5 and 3.2 seconds.

**The log-likelihood difference in these rows mostly reflects the two starting points.**
The harness starts each side from its own draw: numpy's `RandomState(42)` for pamica, gfortran's generator seeded with 42 for the reference.
After 100 iterations the final log-likelihood still depends on where a fit started.
Over eight seeds, the reference's own final log-likelihood has a standard deviation of 2.6e-4 and a range of 8.0e-4 (pamica's: 2.7e-4 and 8.0e-4),
and over the 64 pairs of a pamica start and a reference start the median difference is 3.5e-4 (range 1.7e-5 to 8.2e-4).
The 2.7e-4 in the table lies within that spread.
Before those changes the same run showed 2.9e-5, which reflected this particular pair of starts:
from a shared start, that code trailed the reference by 2.4e-4 after 100 iterations, which offset most of the difference between the two starts.

The update rule itself is compared from a shared start:
pamica's seed-42 initialization is written into the reference's `load_*` files, and both sides run the harness settings for 100 iterations, the reference on one thread
(`.context/issue-351/harness_gap.py`).

| Code | LL difference | Mean matched correlation | Amari distance |
|---|---:|---:|---:|
| before epic #324's changes to the fit (e38aa11) | 2.4e-4 | 0.99999 | 5.8e-4 |
| with them | 1.6e-6 | 0.99999993 | 3.9e-5 |

The earlier row starts from that code's own initialization, which predates the component-row layout (issue #334) and the normalized initial mixing matrix (issue #341);
from each code's seeded state, the reference's first-iteration log-likelihood matches pamica's to 9e-16, so the state was written in the orientation the reference reads.
Against the bundled `amicaout` output (200 reference iterations with the `input.param` settings, from its own unseeded start),
the three backends are within 1.3e-4 to 1.4e-4 in log-likelihood, with mean matched correlation 0.9983 (minimum 0.982 to 0.983) and Amari distance 4.8e-3.

The float32 MLX backend and the float64 PyTorch backend draw their start from the same generator, so they can be compared from the same start directly.
With the harness settings, after 100 iterations they differ by 5.0e-6 in log-likelihood, with mean matched correlation 0.99999991 (minimum 0.9999994) and Amari distance 5.0e-5;
after 200 iterations by 2.4e-6, 0.9999998 and 6.3e-5 (`.context/issue-351/precision_agreement.py`).

The same bars are pinned by `test_backend_meets_its_parity_bar_against_fortran` in `pamica/tests/test_fortran_param_forwarding.py`,
which runs when `AMICA_RUN_FORTRAN=1` is set (as the weekly macOS job does).

## Single-model parity

With Newton disabled (`do_newton=0`), the natural-gradient backend reaches Fortran's solution,
averaged over 5 seeds at a matched 2000-iteration budget. The headline correlation is measured on an
external, unambiguously well-determined recording (OpenNeuro ds002718, 70 channels, $k\approx153$) so
it is not sensitive to whether a particular small dataset happens to be well-conditioned; the bundled
32-channel sample ($k\approx30$, at the project's own data-adequacy boundary) gives a consistent Amari
distance:

- Log-likelihood ~ -3.6993 ($k\approx153$; Fortran ~ -3.6993, gap ~0.0003).
- Hungarian-matched component correlation ~0.998 ($k\approx153$; Fortran-vs-Fortran self-consistency
  over the same 5 seeds: ~0.999), clearing the >0.95 gate.
- Amari distance on the bundled sample: 0.011 over the protocol's five run pairs, 0.004 over 50 pairs (next subsection).

### The bundled sample

On the bundled 32-channel sample ($k\approx30$) the reproduction tier fits five pamica seeds (301-305) and five reference runs, one pair per seed,
Newton off, 2000 iterations (re-measured on 2026-09-23 with the code of epic #324; `.context/issue-351/raw/table1_bundled/`):

| seed | reference final LL | pamica final LL | mean matched correlation | min matched correlation | Amari distance |
|---:|---:|---:|---:|---:|---:|
| 301 | -3.3995 | -3.3997 | 0.9995 | 0.9960 | 0.0031 |
| 302 | -3.3996 | -3.3996 | 0.9993 | 0.9952 | 0.0037 |
| 303 | -3.4006 | -3.3997 | 0.9267 | 0.4528 | 0.0380 |
| 304 | -3.3995 | -3.3995 | 0.9997 | 0.9981 | 0.0029 |
| 305 | -3.3996 | -3.3994 | 0.9982 | 0.9893 | 0.0069 |

The five pairs average a correlation of 0.985 and an Amari distance of 0.011;
the ten pairs among the five reference runs average 0.971 and 0.019.
The reference run of seed 303 ended in a lower-likelihood basin (-3.4006, against -3.3995 to -3.3996 for the other four), and its pair sets the five-pair means.
The tier does not seed its reference runs (`AMICANative` draws a clock-based seed per run), so a rerun draws new starts, and an event like this one may or may not recur.
With the unmixing matrices kept (`.context/issue-351/bundled_single_basins.py`),
ten seeded reference runs (seeds 1-10, final LL -3.39984 to -3.39945) against the same five pamica fits give, over all 50 pairs,
a mean Amari distance of 0.0044 (largest 0.0083) and a mean correlation of 0.9988;
the reference against itself, over 45 pairs, 0.0054 (largest 0.0101) and 0.9985; pamica against itself, over 10 pairs, 0.0039 and 0.9991.
Before epic #324 the five-pair figures were an Amari distance of ~0.006 (Fortran against Fortran ~0.005) and a correlation of ~0.998.

The fixed source-density families are bit-exact against the literal Fortran score/derivative expressions (~1e-15),
and the backend converges to the binary's solution within ~0.005 log-likelihood on either dataset.

### Newton-enabled runs and the initialization basin

The comparison above disables Newton (`do_newton=0`) to isolate the algorithm from its starting point.
With Newton enabled (`do_newton=1`, the reference's default), agreement at the full 2000-iteration budget depends on where a fit starts.
Re-measured on 2026-09-23 with the code of epic #324 on the full 70-channel recording
(pamica on CUDA in float64, seeds 42, 13 and 7; the pinned v0.3.3 reference seeded 1 and 2; the `input.param` settings with the early stops off;
`.context/issue-351/newton_seeds.py`), Hungarian-matched correlation of the 70 unmixing rows:

| Pair | Mean | Min | Components below 0.9 |
|---|---:|---:|---:|
| pamica seed 42 vs reference seed 1 | 0.963 | 0.667 | 8 |
| pamica seed 42 vs reference seed 2 | 0.986 | 0.858 | 3 |
| pamica seed 13 vs reference seed 1 | 0.996 | 0.946 | 0 |
| pamica seed 13 vs reference seed 2 | 0.981 | 0.773 | 3 |
| pamica seed 7 vs reference seed 1 | 0.995 | 0.943 | 0 |
| pamica seed 7 vs reference seed 2 | 0.982 | 0.786 | 3 |
| reference seed 1 vs reference seed 2 | 0.985 | 0.841 | 2 |
| pamica seed 42 vs pamica seed 13 | 0.962 | 0.658 | 7 |
| pamica vs reference, both from the same start | 0.9999998 | 0.9999964 | 0 |
| reference from that start vs reference seed 1 | 0.996 | 0.955 | 0 |

The weakest, under-determined components settle into different basins from different starts, in the reference's runs as in pamica's:
pamica's seed 42 differs from the reference's seed 1 on 8 components, and the reference's seed 2 differs from its own seed 1 on 2 and from each pamica seed on 3.
The alternative basins have equal or higher likelihood
(final log-likelihood: reference seeds 1 and 2, -3.697804 and -3.697709; pamica seeds 42, 13 and 7, -3.697569, -3.697816 and -3.697812).
The same start is the deterministic one of issue #145 ($A = I$, `mu` at -1, 0 and 1, `sbeta` = 1, `rho` = 1.5), fed to the reference through its `load_*` files;
from it, pamica and the reference end with the same components after 2000 Newton iterations
(mean correlation 0.9999998, Amari distance $3.8\times10^{-5}$, final log-likelihoods $7.6\times10^{-8}$ apart).
In issue #145, measured before epic #324's changes to the fit, two clock-seeded reference runs agreed at 0.9997 (minimum 0.998),
pamica's seeds 42, 13 and 7 reached 0.942, 0.994 and 0.996 against them, and the same start gave 0.997.
The seeded pair above (0.985) is a second measurement of the reference's own seed-to-seed spread, on one more pair of starts.
See issue #145 and the optional init-robustness follow-up #198.

A single-seed supplementary run gives the float32 side at this data size:
the MLX backend (float32), with the same keywords and the same start as the CUDA seed-7 fit, stops on its learning-rate floor at iteration 951,
after about 120 likelihood decreases below float32 resolution, at log-likelihood -3.69809 (the float64 fit was at -3.69791 at that iteration).
Its components match the float64 seed-7 fit at a mean correlation of 0.985 (minimum 0.907) and the reference's seed 1 at 0.983 (minimum 0.905);
with seed 42, float32 and float64 differ on 10 of the 70 components (mean 0.955).
At this budget the weak components' float32 basin varies the way a change of start does (`.context/issue-351/raw/newton_mlx/`).

### Source-density families are bit-exact

AMICA models each source with one of the reference's five `pdftype` density families.
For every family other than the default generalized Gaussian, the vectorized log-density and score reproduce the literal `amica15.f90` expressions
to float64 precision (test bound $<10^{-12}$, observed $\sim\!10^{-15}$):
the source model is not an approximation of the Fortran one, it is the same function.
That includes the normalizing constants at the precision the binary uses them:
the reference writes them as single-precision literals widened to double, and every backend has used those values since issue #344
([the differences guide](amica-differences.md#single-precision-constants-issue-344)).
Seeded with pamica's initialization, the native binary's log-likelihood for families 2, 4 and 1 matches PyTorch's to $3\times10^{-15}$ after one and three iterations,
where the decimals' double values had been off by $3.7\times10^{-10}$, $2.0\times10^{-8}$ and $-2.1\times10^{-8}$
(`pamica/tests/test_reference_constants.py`, opt-in with `AMICA_RUN_FORTRAN=1`).
The generalized Gaussian has no closed-form literal to compare against, since its score depends on the adaptive shape $\rho$,
so the default family is validated by the single-model parity above instead.
The oracle column below records which check applies to each family:

| `pdftype` | Density family | Score $f_p(y)$ | Bit-exact oracle |
|---|---|---|---|
| 0 (default) | Generalized Gaussian (adaptive shape $\rho$) | GG score (shape-dependent) | single-model parity above |
| 1 | Extended-Infomax adaptive switch (super- ↔ sub-Gaussian by kurtosis sign) | $y + \tanh y$ / $y - \tanh y$ | real-data LL (see below) |
| 2 | Gaussian | $y$ | yes |
| 3 | Logistic | $\tanh(y/2)$ | yes |
| 4 | Sub-Gaussian (cosh$^{+}$) | $y - \tanh y$ | yes |

`pdftype=0` is the default and is byte-for-byte the pre-family implementation.
The `pdftype=1` extended-Infomax switcher flips each source between the super-Gaussian (code 1) and
sub-Gaussian (code 4) densities on a kurtosis schedule; its dynamic switch has no bit-exact oracle
(the reference's `do_choose_pdfs` accumulator is dead code in the binary), so it is validated by real-data
log-likelihood instead. Each fixed family converges within ~0.005 LL of the binary at a matched Newton budget.
See `pamica/tests/torch_tests/test_ng_pdf_families.py` and ADR 0002.

## Multi-model distributional similarity

Multi-model AMICA is not partition-identifiable: fits from different starts reach different partitions of nearly the same likelihood,
so a single-run partition comparison with Fortran cannot serve as the acceptance bar.
The comparison is between the distributions of solutions the two implementations sample.
An ensemble of `N = 20` fits per implementation on the bundled sample EEG (`n_models = 2`, 3 mixture components, 100 iterations, matched schedule)
gives these distributions of pairwise agreement
(`benchmarks/reproduce_table1.py --tier bundled`, re-measured on 2026-09-23 with the code of epic #324 against the pinned v0.3.3 native binary,
whose runs here draw their own clock-based seeds; the 40 fits are saved in `.context/issue-351/raw/table1_bundled/bundled_multimodel_ensemble.npz`):

| Distribution (pairwise Hungarian-matched \|corr\|) | Mean | SD | Range |
|---|---:|---:|---|
| within-Fortran (Fortran vs Fortran) | 0.626 | 0.036 | [0.568, 0.868] |
| within-pamica (pamica vs pamica) | 0.638 | 0.042 | [0.578, 0.805] |
| between (pamica vs Fortran) | 0.632 | 0.042 | [0.564, 0.845] |

![Multi-model solution-ensemble cross-correlation and log-likelihood distributions for pamica and Fortran.](../assets/figures/multimodel-ensemble.png){ width=640 }
/// caption
Pairwise Hungarian-matched component correlation (A) and final log-likelihood (B) for 20 pamica and 20 Fortran multi-model fits of the sample EEG.
The three agreement distributions overlap, and so do the two likelihood distributions.
///

The three means lie within 0.012 of each other.
The between-minus-within-Fortran difference is +0.006, inside the $\pm 0.05$ margin the original study set
(run-level bootstrap 90% interval -0.002 to 0.014, `.context/issue-351/equivalence_check.py`).
The 190/400 pairwise values are not independent (each of the 40 runs appears in ~39 pairs),
so a Mann-Whitney or TOST applied to the pairwise values is pseudoreplicated and its p-value is invalid.
The significance test permutes the 40 runs as intact units instead (20000 permutations, statistic = within-Fortran minus between-implementation mean correlation).
For the one-sided hypothesis that cross-implementation agreement is worse than Fortran's own run-to-run agreement, it gives $p = 0.88$.

The single-run cross-correlation of ~0.63 matches Fortran's agreement with itself (0.63), so it measures the estimator's run-to-run spread.
The per-block sufficient statistics and one M-step agree with the reference to round-off (~$3\times10^{-16}$ relative).
The final log-likelihoods agree too: pamica $-3.3541 \pm 0.003$, Fortran $-3.3543 \pm 0.002$ (Kolmogorov-Smirnov $p = 0.83$).
The ensembles of this study measured before epic #324's changes to the fit differed on this one metric
(pamica $-3.363 \pm 0.006$ against Fortran $-3.354 \pm 0.003$, $p \approx 6\times10^{-5}$), a gap attributed then to convergence speed.
Refitting the pamica half with that code (e38aa11) against the same 20 reference fits gives $-3.3627 \pm 0.006$ ($p = 1\times10^{-5}$)
(`.context/issue-351/multimodel_pamica_fits.py`).
Seven of those 20 fits stop early on `min_dll`, whose check counted likelihood dips as small gains until issue #339 (mean $-3.3679$),
and the 13 that run the full 100 iterations average $-3.3600$, so the old gap came partly from the early stops and partly from the update rule of that code.
A seeded ensemble with the same settings (the reference seeded 0-19 and single-threaded) agrees within $8\times10^{-4}$ at 100 iterations and within $2\times10^{-4}$ at 200 and 300
([ADR 0003](https://github.com/sccn/pAMICA/blob/main/.context/decisions/0003-best-iterate-safeguard.md)).

### Amari distance: a second, assignment-free metric

The correlation above needs a Hungarian assignment step to resolve component permutation before it can be computed.
The Amari distance does not: it is permutation- and scale-invariant by construction,
so it is an independent check on the same 20-run ensembles, computed from the saved unmixing matrices with no refitting
(`.context/issue-351/multimodel_ensemble.py`, which reuses `.context/issue-27/amari_distance.py`).
Each stacked 2-model matrix is split into its per-model 32x32 blocks;
since which Fortran model corresponds to which pamica model is not identified, both label pairings are tried and the lower-distance pairing is kept, per run pair.
This pairing correction lowers the mean distance by 0.019 on this ensemble (333 of the 780 run pairs take the swapped pairing),
the same order as the gaps between the groups below, so part of those gaps may reflect how often each group needs the swap.

| Distribution (Amari distance, lower is better) | Mean | SD |
|---|---:|---:|
| within-Fortran (Fortran vs Fortran) | 0.166 | 0.017 |
| within-pamica (pamica vs pamica) | 0.176 | 0.025 |
| between (pamica vs Fortran) | 0.172 | 0.022 |

By this metric pamica's ensemble spreads slightly more than the reference's (0.176 against 0.166),
and the between-implementation distance lies between the two: +0.005 from within-Fortran (bootstrap 90% interval 0.002 to 0.009).
The same one-sided run-level permutation test gives $p = 0.051$.
Before epic #324's changes to the fit, pamica's ensemble was the tighter one:
refit with that code against the same reference fits, within-pamica 0.151 and between 0.160 ($p = 0.998$),
and the original ensembles of this study measured 0.154 and 0.163 against the bundled `amica15mac` binary's 0.174 ($p > 0.999$).

??? note "Per-run detail (all 40 runs, both metrics)"

    Table 1 in the paper and the group summaries above report distribution means;
    the table below gives each of the 40 runs' own mean agreement to its own group's other 19 runs (`within`) and to all 20 opposite-implementation runs (`between`), for both metrics.
    Regenerate with `uv run python .context/issue-351/multimodel_ensemble.py --from-npz .context/issue-351/raw/table1_bundled/bundled_multimodel_ensemble.npz`,
    which writes `.context/issue-351/per_run_detail.csv` (with each run's final log-likelihood) and the figure above.

    | implementation | run | corr within | corr between | Amari within | Amari between |
    |---|---:|---:|---:|---:|---:|
    | Fortran | 0 | 0.6169 | 0.6061 | 0.1710 | 0.1853 |
    | Fortran | 1 | 0.6348 | 0.6508 | 0.1615 | 0.1593 |
    | Fortran | 2 | 0.6145 | 0.6064 | 0.1756 | 0.1876 |
    | Fortran | 3 | 0.6250 | 0.6356 | 0.1655 | 0.1684 |
    | Fortran | 4 | 0.6290 | 0.6424 | 0.1654 | 0.1663 |
    | Fortran | 5 | 0.6118 | 0.6047 | 0.1707 | 0.1853 |
    | Fortran | 6 | 0.6268 | 0.6334 | 0.1610 | 0.1648 |
    | Fortran | 7 | 0.6238 | 0.6247 | 0.1700 | 0.1765 |
    | Fortran | 8 | 0.6332 | 0.6471 | 0.1643 | 0.1696 |
    | Fortran | 9 | 0.6344 | 0.6458 | 0.1606 | 0.1648 |
    | Fortran | 10 | 0.6057 | 0.6019 | 0.1782 | 0.1876 |
    | Fortran | 11 | 0.6353 | 0.6534 | 0.1627 | 0.1631 |
    | Fortran | 12 | 0.6182 | 0.6147 | 0.1680 | 0.1796 |
    | Fortran | 13 | 0.6497 | 0.6503 | 0.1590 | 0.1642 |
    | Fortran | 14 | 0.6270 | 0.6343 | 0.1637 | 0.1679 |
    | Fortran | 15 | 0.6315 | 0.6322 | 0.1653 | 0.1715 |
    | Fortran | 16 | 0.6101 | 0.6119 | 0.1701 | 0.1772 |
    | Fortran | 17 | 0.6181 | 0.6287 | 0.1698 | 0.1718 |
    | Fortran | 18 | 0.6428 | 0.6764 | 0.1596 | 0.1537 |
    | Fortran | 19 | 0.6222 | 0.6371 | 0.1638 | 0.1664 |
    | pamica | 0 | 0.6507 | 0.6421 | 0.1703 | 0.1671 |
    | pamica | 1 | 0.6491 | 0.6338 | 0.1687 | 0.1687 |
    | pamica | 2 | 0.6264 | 0.6206 | 0.1829 | 0.1785 |
    | pamica | 3 | 0.6277 | 0.6324 | 0.1860 | 0.1730 |
    | pamica | 4 | 0.6390 | 0.6358 | 0.1804 | 0.1706 |
    | pamica | 5 | 0.5984 | 0.6025 | 0.1983 | 0.1880 |
    | pamica | 6 | 0.6608 | 0.6492 | 0.1605 | 0.1615 |
    | pamica | 7 | 0.6266 | 0.6175 | 0.1823 | 0.1786 |
    | pamica | 8 | 0.6177 | 0.6205 | 0.1859 | 0.1751 |
    | pamica | 9 | 0.6587 | 0.6442 | 0.1633 | 0.1644 |
    | pamica | 10 | 0.6385 | 0.6269 | 0.1747 | 0.1749 |
    | pamica | 11 | 0.6291 | 0.6259 | 0.1800 | 0.1752 |
    | pamica | 12 | 0.6608 | 0.6499 | 0.1642 | 0.1631 |
    | pamica | 13 | 0.6591 | 0.6456 | 0.1638 | 0.1665 |
    | pamica | 14 | 0.6237 | 0.6252 | 0.1823 | 0.1733 |
    | pamica | 15 | 0.6357 | 0.6361 | 0.1795 | 0.1704 |
    | pamica | 16 | 0.6256 | 0.6219 | 0.1789 | 0.1720 |
    | pamica | 17 | 0.6408 | 0.6326 | 0.1784 | 0.1742 |
    | pamica | 18 | 0.6173 | 0.6221 | 0.1868 | 0.1751 |
    | pamica | 19 | 0.6663 | 0.6528 | 0.1574 | 0.1609 |

## Cross-platform device and precision invariance

The strongest reassurance that pamica is a single, well-defined implementation is that it recovers the *same*
independent components no matter where or how it runs. Fitting the same real EEG (ds002718 sub-002, 147,000 frames, 70 channels, 2000 iterations)
on every backend and Hungarian-matching the unmixing components across them
(measured before epic #324's changes to the fit and not re-measured since; issue #351 re-measured the figures elsewhere on this page):

![Cross-backend IC-equivalence matrix at 70 channels.](../assets/figures/cross-backend-equivalence-matrix.png){ width=680 }
/// caption
Mean Hungarian-matched \|correlation\| of the recovered components between every pair of backends.
The eight torch/MLX combinations, CPU, CUDA, MPS, and MLX, at both float32 and float64, on macOS-arm64 and Linux-x86_64, all agree at **1.000**.
///

**Every torch/MLX backend is identical to every other at 1.000**: the same decomposition on any device, at any
precision, on either operating system. This is the definitive "float32 == float64" and "GPU == CPU" result.
The two native-Fortran builds (macOS-arm64 and Linux-x86_64) agree with each other at 0.972 and with the
torch/MLX cluster at ~0.90. That residual is not a backend defect: Fortran does not even reach 1.000 against
*itself* across platforms, because each native run is seeded from the clock and settles into a different (equally valid)
local optimum on the weakly-determined components only. As the decomposition becomes better-determined that gap closes
(next section), and the component maps are visibly the same down every row:

![Variance-ordered IC scalp maps recovered by each backend.](../assets/figures/cross-backend-ic-topomaps.png){ width=600 }
/// caption
IC scalp maps at 70 channels, variance-ordered (IC1 = highest back-projected variance, EEGLAB convention),
Hungarian-matched and sign-aligned across backends (rows). Each map is the de-sphered sensor-space projection.
The well-determined components are indistinguishable across all backends.
///

## Data adequacy and cross-backend equivalence

Whether backends recover the *same* components depends on how well-determined the decomposition is, captured by the data-adequacy factor:

$$k = \frac{\text{frames}}{\text{channels}^2}$$

The two sweeps below were measured before epic #324's changes to the fit and have not been re-measured since.

As `k` grows, cross-backend component equivalence rises toward 1.0;
at the rule-of-thumb minimum (`k` around 20-30) only the strongest components are backend-reproducible,
while the rest are under-determined and settle into different but equally valid local optima (AMICA is non-convex).
This is why the native-Fortran rows above sit at ~0.90 (70 channels, `k` = 30) rather than 1.0. Two independent sweeps confirm it.

### Sweeping channels at fixed frames

Holding frames at 147,000 and increasing the channel count lowers `k`.
Comparing MLX-float32 against the independent native-Fortran-float64 build (the hardest cross-implementation pair):

| channels | k = frames/ch² | mean matched \|corr\| | components > 0.95 |
|---:|---:|---:|---:|
| 16 | 574 | **0.997** | 16/16 |
| 32 | 144 | 0.974 | 27/32 |
| 48 | 64 | 0.954 | 34/48 |
| 70 | 30 | 0.898 | 20/70 |

At high `k` **every backend, including the independent Fortran build, recovers identical components** (0.997, all 16/16 at `k` = 574).
At `k` = 30, the rule-of-thumb minimum, only the strongest ~20/70 components are reproducible; the rest are under-determined.

### Sweeping frames at fixed channels

Holding channels at 70 and increasing frames raises `k`, on the same real EEG:

![Cross-backend IC equivalence versus the data-adequacy factor k at 70 channels.](../assets/figures/kfactor-equivalence.png){ width=640 }
/// caption
Mean Hungarian-matched cross-backend \|correlation\| versus $k = \text{frames} / \text{channels}^2$ (70 channels, 2000 iterations,
native-Fortran and PyTorch-CUDA float64/float32 backends).
Equivalence saturates at ~0.98 once $k \geq 60$.
///

| frames | k | mean \|corr\| | components >0.95 |
|---|---|---|---|
| 73,500 | 15 | 0.911 | 55.2% |
| 147,000 | 30 | 0.929 | 56.7% |
| 294,000 | 60 | 0.982 | 90.0% |
| 490,000 | 100 | 0.983 | 94.8% |
| 747,750 | 152 | 0.982 | 92.4% |

The `k` = 30 row here (0.929) sits above the 0.898 at `k` = 30 in the channel-sweep table because the two average different backend sets:
the channel sweep reports only the hardest MLX-versus-native-Fortran pair, while this frame sweep averages over the native-Fortran and PyTorch-CUDA float64/float32 cluster.

!!! note "The threshold is data-specific"
    For this recording the equivalence knee falls **between k=30 and k=60**; below
    it the backends settle into different (equally valid) local optima, above it
    they recover the same components. Where that knee sits depends on the data
    (signal-to-noise ratio, effective rank, source structure), so this is not a
    universal value of `k`. The plateau is ~0.98 rather than 1.0 because of
    intrinsic estimator spread and the float32 path, not a backend defect.

### Why the plateau sits at ~0.98, not 1.0

At the largest data size (k=152) the residual below 1.0 splits cleanly by precision.
The two double-precision implementations, an independent native Fortran binary and the PyTorch-CUDA backend, agree at 0.995:

| Pair (at k=152) | \|corr\| |
|---|---:|
| native-Fortran f64 vs PyTorch-CUDA f64 | 0.995 |
| native-Fortran f64 vs PyTorch-CUDA f32 | 0.971 |
| PyTorch-CUDA f64 vs PyTorch-CUDA f32 | 0.979 |

This is cross-*implementation* agreement, not just cross-device. The residual gap is dominated by the float32 path (rounding accumulated over 2000 iterations, plus an early stop when the natural-gradient learning rate hit its floor), which is a convergence/precision effect rather than a backend defect.

## EEGLAB drop-in round-trip

pamica writes the same on-disk format EEGLAB's AMICA plugin reads, so a fit is a drop-in replacement:
no re-sorting, sign-flipping, or reformatting. After a fit, `write_amica_output(dir)` writes the raw binary
files (`gm`, `W`, `S`, `mean`, `c`, `alpha`, `mu`, `sbeta`, `rho`, `comp_list`, `LL`) that EEGLAB's
`loadmodout15.m` loads.

The round-trip is verified two ways:

- **Byte-level:** for a single model the written files are an exact float64 serialization of the fitted parameters.
  `W` is byte-identical in C order;
  the sphere `S` and the non-square mixture parameters and `c`/`comp_list` are column-major (Fortran layout),
  matching the reference `amicaout` files.
  The default symmetric zero-phase component analysis (ZCA) sphere happens to be its own transpose to about 1e-17,
  which is why an earlier column-major/C-order mismatch in the square-sphere write path went unnoticed,
  until it was measured against an asymmetric (`do_approx_sphere=False`) sphere (issue #336).
- **Reader-level:** the directory loads through `loadmodout15.m` (and its NumPy port `loadmodout`) with the
  expected shapes and the correct column-major layout. The MATLAB round-trip during development is what caught,
  and fixed, a column-major format bug in the mixture-parameter arrays.

`variance_order()` reproduces EEGLAB's IC ordering (IC1 = highest back-projected variance) in Python without a
disk round-trip. For `n_models > 1` every file is also in the reference's layout (`W` since issue #159, `A` since issue #334),
and the directory round-trips through both readers; the values differ from any one native run
only because multi-model AMICA is not partition-identifiable (see the multi-model discussion above).
Full usage is in the [EEGLAB interoperability guide](eeglab.md); tests are in `pamica/tests/torch_tests/test_amica_ng_wrapper.py`.

## Performance across backends

Throughput on real EEG (OpenNeuro ds002718 sub-002; `n_mix=3`, `pdftype=0`, `block_size=512`, warmed, min-of-repeats).
CPU, MPS, and MLX were measured on Apple Silicon; CUDA on a separate NVIDIA RTX 4090 host,
so MLX-versus-CUDA reads as "best Apple-GPU path versus a strong NVIDIA GPU", not a same-box comparison.
These tables predate epic #324.
A same-session check on the Apple M4 Pro at 70 channels (`benchmark_dimsweep.py`, 30000 frames, 25 iterations, three repeats; issue #351)
found per-iteration cost unchanged by the epic: 33.3 and 33.1 ms for MLX, 190.8 and 189.4 ms for torch-CPU float64, 165.1 and 166.7 ms for torch-MPS float32, before and after.

### Single-model, ms/iteration

| channels | MLX f32 | CUDA f32 | CUDA f64 | torch-CPU f32 | torch-CPU f64 | torch-MPS f32 | NumPy f64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 15.4 | 35.5 | 35.0 | 52 | 71 | 189 | 142 |
| 32 | 21.3 | 35.5 | 36.2 | 143 | 161 | 162 | 287 |
| 48 | 19.5 | 36.0 | 35.9 | 151 | 168 | 168 | 426 |
| 70 | 25.2 | 35.6 | 38.6 | 173 | 193 | 255 | 622 |

MLX is the fastest option on Apple Silicon and stays roughly flat with channel count (~7x over torch-CPU).
PyTorch-MPS is *not* a win at this `block_size=512` (at or worse than CPU); it is also markedly more
block-size-sensitive than the CPU or MLX figures above (issue #216, bundled sample): it falls to 30.5
ms/iteration at the current 8192 default, still behind CPU's 21.7 ms/iteration there, and to 13.5
ms/iteration at a further-tuned single-block size that puts the whole sample in one block -- memory-limited
rather than a free win, since peak block memory scales with `block_size`, which is why 8192 stays the
shipped default -- below the CPU's 15.8 ms there. MLX remains fastest throughout,
so it stays the recommendation over `device="mps"` on Apple hardware.
CUDA float32 and float64 are near-identical here (launch-bound at this size). NumPy is the reference implementation, not a production path.

### Block-size sensitivity

`block_size` trades memory for dispatch overhead, and backends differ sharply in how much they benefit.
Measured on an Apple M4 Pro (14 cores), the bundled 32-channel sample (30504 frames), float32,
`n_mix=3`, `pdftype=0`, seed 42, warm, per-iteration cost in ms (issue #216):

| block_size | PyTorch-MPS | MLX | PyTorch-CPU |
|---:|---:|---:|---:|
| 512 (former default) | 431.3 | 29.5 | 144.0 |
| 2048 | 89.7 | 12.4 | 49.3 |
| 8192 (current default) | 30.5 | 11.3 | 21.7 |
| 30504 (single block) | 13.5 | 11.4 | 15.8 |

All three backends are dispatch-bound at small block sizes, but by very different margins: PyTorch-MPS
improves 32x from 512 to a single block, PyTorch-CPU 9x, MLX only 2.6x. Log-likelihood after 40
iterations is unchanged across every block size on both devices (-3.43856 to six significant digits),
so this is a pure throughput knob on this data, not a correctness one, and the comparison across block
sizes is valid. MLX is the fastest Apple backend at every block size measured here, including the
current 8192 default and the further-tuned single-block setting, so it stays the recommendation on
Apple hardware regardless of how `block_size` is tuned. The `30504` row is a memory-bound extreme
(the whole sample as one block), not a free win: peak block memory scales with `block_size`, which is
why 8192, not 30504, stays the shipped default.

Since the optimum moves with host, device and data, `do_opt_block` (issue #232, off by default) can
search for it instead: it times candidate sizes on your actual data at the start of `fit` and keeps
the fastest, on all three backends, under Fortran's `blk_min`/`blk_max`/`blk_step` names. The choice
is timing-based and therefore machine-dependent, so a parity run must leave it off and pin
`block_size`; see [the block-size search](amica-differences.md#the-block-size-search-picks-a-machine-dependent-value-issue-232)
for the full caveat and for how a candidate that cannot be allocated is handled (skipped, not fatal --
the one place the reference implementation exits where it should degrade).

### CPU core-scaling and native Fortran

The table above uses each platform's default thread count and has no Fortran row.
A separate core-count sweep (`--threads`, real ds002718 sub-002 EEG, 70 channels, `n_mix=3`,
`pdftype=0`, `do_newton` off, `block_size=512`) adds native Fortran (via `OMP_NUM_THREADS`) alongside torch-CPU (`set_num_threads`) and NumPy (`threadpoolctl`) on the same two machines as above, GPU backends run once since they are thread-independent:

| backend (Intel Core i9-13900K / RTX 4090 workstation, 24 cores / 32 threads) | 4c | 8c | 12c | 16c | 24c |
|---|---:|---:|---:|---:|---:|
| native-fortran f64 | 69.5 | 43.2 | 49.0 | 40.0 | **30.0** |
| torch-CPU f64 | 105.9 | 91.0 | 92.6 | 142.5 | 212.8 |
| torch-CPU f32 | 84.6 | 69.5 | 71.8 | 70.9 | 73.0 |
| NumPy f64 | 794.6 | 810.3 | 871.7 | 855.5 | 866.1 |

GPU (thread-independent, run once): CUDA f64 = 38.5, CUDA f32 = 36.2.

| backend (Apple Silicon, 14 cores: 10P + 4E) | 4c | 8c |
|---|---:|---:|
| native-fortran f64 | 100.0 | **70.0** |
| torch-CPU f64 | 131.4 | 169.9 |
| torch-CPU f32 | 111.7 | 144.4 |
| NumPy f64 | 627.0 | 627.4 |

GPU (thread-independent, run once): MLX f32 = 33.4, MPS f32 = 217.7.

Native Fortran with OpenMP is the only CPU backend that scales with cores:
on all 24 of the 13900K's cores it beats the RTX 4090 (30 vs 38.5 ms/iteration),
and on 8 of the Mac's 14 cores it is faster than either torch-CPU precision.
torch-CPU f64 peaks around 8 cores then regresses from oversubscription (91 to 213 ms/iteration going from 8 to 24 cores on the workstation);
torch-CPU f32 is faster and scale-stable but never catches the GPU.
NumPy is thread-flat (BLAS/Python-bound) and slowest everywhere.
MLX remains the efficiency winner overall: ~33 ms/iteration, flat, no tuning, beating a 450 W RTX 4090 and a 24-core Core i9-13900K workstation,
at a fraction of the power and cost;
native-Fortran@24c is marginally faster in raw ms/iteration only by pinning every core of a much larger, hotter machine.
Native-Fortran timing at ≤32 channels is at or below the binary's ~10 ms stamp resolution, so trust the 48/70-channel scaling curve;
the full 16/32/48/70-channel grid is in the result JSONs alongside `.context/issue-84/phase2_cpu_scaling.md`.

### Multi-model (n_models=2), ms/iteration

| channels | MLX f32 | torch-CPU f32 | torch-MPS f32 | NumPy f64 |
|---:|---:|---:|---:|---:|
| 32 | 38 | 187 | 291 | 869 |
| 70 | 45 | 224 | 270 | 928 |

The Apple-GPU win extends to multi-model at this `block_size=512`: MLX ~38-45 ms/iteration, ~5x over
torch-CPU, with MPS still losing. Unlike the single-model figures above, this configuration has not
been re-swept at the current 8192 default (issue #216 covered single-model only), so whether the gap
narrows here too is untested.

### Cross-backend log-likelihood agreement (single-model)

The log-likelihood each backend reaches with the same settings and the same start,
from the throughput sweep's own runs (`benchmark_dimsweep.py`: the first 30000 frames of ds002718 sub-002,
the first 32, 48 or 70 channels, 25 iterations, `block_size=512`, no Newton, seed 42;
re-measured on 2026-09-23 with the code of epic #324, issue #351, on the Apple M4 Pro and, for CUDA, the RTX 4090 host):

| channels | MLX f32 | CUDA f64 | CUDA f32 | torch-CPU f64 | torch-MPS f32 | NumPy f64 |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | -3.28488 | -3.28489 | -3.28488 | -3.28489 | -3.28488 | -3.28478 |
| 48 | -3.20839 | -3.20838 | -3.20839 | -3.20838 | -3.20839 | -3.20893 |
| 70 | -3.21643 | -3.21723 | -3.21689 | -3.21746 | -3.21638 | -3.22054 |

At 32 and 48 channels every backend that starts from the shared draw agrees to 1e-5, so float32 matches float64 to five significant digits.
At 70 channels the sweep's 30000 frames give $k\approx6$, and 25 iterations at `lrate=0.1` amplify round-off:
two float64 runs of the same code on CPU and CUDA differ by 2.3e-4, the backends by up to 1.1e-3,
and adding 1e-9 µV to one sample changes the torch-CPU value by 1.3e-15 after one iteration and 4.4e-4 after 25
(the trajectory has a likelihood decrease on its 25th iteration).
The 70-channel differences are of the same size as that sensitivity,
and the code before epic #324's changes to the fit (e38aa11) shows the same growth.
NumPy is called without a seed here, so it starts from its own draw, which is why its column sits apart (by up to 4.2e-3 at 70 channels).
The component-level float32 comparison is in [the per-backend rows](#parity-rows-per-backend):
from a shared start, MLX float32 and PyTorch float64 agree to a mean matched correlation of 0.99999991 after 100 iterations.

## Other validated behaviors

Beyond the core parity results, the following AMICA features are implemented and validated. Where the reference
binary contains a bit-exact oracle it is used; where the reference code path is unrunnable (declared but never
allocated, so it cannot be exercised even in Fortran) the feature is behavior-validated on real EEG instead, and
guarded to a no-op so the parity results above stay byte-for-byte unchanged.

| Behavior | Status | Validation |
|---|---|---|
| Best-iterate safeguard (`keep_best`, #51) | on by default | returns the highest-LL iterate. It cut the multi-model LL sd from 12.7x to 2.0x Fortran's before epic #324; re-measured with the epic's code, the sd ratio is 1.0x with or without it and a restore fired in one of 20 seeded fits, at the 300-iteration budget only (gain 4.2e-6). Single-model parity stays bit-exact (monotone, no restore). ADR 0003 |
| Per-model bias `c` update (#27) | on for `n_models>1` | Fortran `update_c`; per-block stats bit-exact; no-op for `n_models=1` |
| Component sharing (`share_comps`, #60, #334) | off by default | Fortran `identify_shared_comps` ported; the scan itself has no bit-exact oracle (`Spinv2` is never allocated, so the reference's scan computes NaN similarities and never merges), but the update from a merged state seeded through the reference's `load_comp_list` matches the native binary to float64 round-off on PyTorch and NumPy (`test_component_rows.py`, opt-in with `AMICA_RUN_FORTRAN=1`); byte-identical when unshared |
| Outlier rejection (`do_reject`, #123) | off by default | `good_idx` mechanism on all three backends (NumPy, PyTorch, MLX -- the last landed epic #278 Phase 3, #289); MLX/NumPy ports validated vs the PyTorch backend |
| Degenerate-fit contract (#50, #306, #339) | always | a fit that stops on a non-finite log-likelihood (`nan_ll`/`singular_ll`), update direction (`nan_direction`) or parameters (`nan_params`) is marked unusable (`converged_=False`, with `stop_reason_`), and every output path refuses it instead of returning NaN sources: the `AMICA` wrapper's `transform`/`get_*`/`write_amica_output`/`save`, and since issue #306 the raw `AMICATorchNG`, `AMICAMLXNG` and `AMICA_NumPy` output accessors too (`pamica/tests/test_backend_guards.py`, `pamica/tests/test_nonfinite_stops.py`). |
| End-to-end workflow (#315) | always, PyTorch and MLX | average-referenced sample EEG with `pcakeep = n_channels - 1` through `AMICAICA`: 31 components, `get_sources` equal to `transform`, an exclusion that removes exactly one back-projection and keeps the residual, the EEGLAB export reloaded by `loadmodout`, `save`/`load`, an `input.param`-driven fit, and the two backends' sources Hungarian-matched with a minimum correlation of at least 0.999 (measured 0.999999999) |

Tests live under `pamica/tests/`: `torch_tests/test_ng_backend.py`, `torch_tests/test_ng_sharing.py`, `torch_tests/test_amica_ng_wrapper.py`, `test_numpy_reject.py`, and `mne_tests/test_end_to_end_workflow.py`.

## Which convergence criterion actually stops a fit

AMICA ships four stops. On recordings the size of the bundled sample, only two of
them fire, and it is worth knowing which before concluding that one is broken.

Two of the four defaults differ between the backends, so read the column that
matches the entry point you use. `AMICA_NumPy` resolves its defaults from the
bundled `pamica/numpy_impl/params.json`; `AMICA`/`AMICATorchNG` and `AMICAMLXNG`
take theirs from the constructor signature (`max_iter` from `fit`); Fortran
compiles in the values in `amica15_header.f90` and the bundled
`pamica/sample_data/input.param` overrides several.

| Stop | `AMICA` / `AMICATorchNG` | `AMICA_NumPy` | `AMICAMLXNG` (MLX) | Fortran (compiled / `input.param`) |
|---|---|---|---|---|
| `max_iter` | **100** | 2000 | **100** | none / 2000 |
| `min_dll` (`use_min_dll`) | on, `1e-9` | on, `1e-9` | on, `1e-9` | on, `1e-9` |
| `min_nd` (`use_grad_norm`) | on, `1e-7` | on, `1e-7` (named `min_grad_norm`) | on, `1e-7` | on, `1e-7` |
| `minlrate` (`lrate_floor`) | `1e-12` | `1e-12` | `1e-12` | `1e-12` / `1e-8` |
| `do_newton` | **off** | **on** | **off** | off / on |

The MLX column dates from issue #248, which ported both stops to that backend;
before it, an MLX fit had no convergence criterion at all and always ran to
`max_iter`. `do_newton` joined it in issue #264 (float32 throughout; see the
backend-differences guide), taking `AMICATorchNG`'s off-by-default.

Which of them actually ends a fit:

- **`min_dll` normally wins**, at iteration 326-1076 depending on the BLAS build,
  when `max_iter` is large enough to let it. At `AMICATorchNG`'s
  default `max_iter=100` the fit always ends on `max_iter` before `min_dll` can
  fire, so the default PyTorch run is iteration-limited, not converged. Raise
  `max_iter` if you want the likelihood stop to be the one that decides.
- **`min_nd` never fires** on a recording this size, in any of the four
  implementations. This is the subject of the rest of this section.
- **`minlrate` needs sustained likelihood decreases** to anneal the learning rate
  all the way to the floor. The bundled sample stops on `min_dll` (or `max_iter`)
  long before that, so it is not a stop you will meet here.

**`min_nd` is not reachable on a recording this size, in any of the four
implementations.** Running the reference binary to completion under a matched
configuration, its own gradient norm oscillates rather than shrinking:

| iteration | Fortran `nd` |
|---:|---:|
| 1000 | 4.7e-5 |
| 1300 | 2.8e-5 |
| 1500 | 3.1e-5 |
| 1700 | 3.2e-5 |
| 2000 | 2.5e-5 |

It then plateaus at 1.0-1.65e-5 out to iteration 5073 without ever crossing the
`1e-7` threshold, which sits about two orders of magnitude below the reference's
own floor. The Python backends plateau roughly two orders higher again: near
a fixed point `dAk` tends to zero, so the norm is measuring a near-total
cancellation where floating-point and BLAS ordering differences dominate what is
left.

This is a property of the data rather than a defect. 30504 samples is small
against the free parameters (1024 in `A` alone, before the mixture parameters),
so the natural-gradient residual has a finite-sample noise floor above the
threshold. The reference ships the same default and behaves the same way, so
retuning it here would mean changing a Fortran-faithful default to manufacture a
desired outcome.

What data size *would* make `min_nd` meaningful has not been characterized. If
you need a gradient-based stop, measure the plateau on your own recording first
and set `min_nd` above it; otherwise leave `min_dll` to do the work.

## Reproducing these results

The paper's parity table reproduces through one entry point, `benchmarks/reproduce_table1.py`.
It prints each measured value next to the paper row it corresponds to, so the two can be checked off
directly. The compute device is auto-detected (float64 is required for parity, so Apple GPUs fall back
to CPU with a printed reason) and the Fortran reference binary is resolved per platform, so this is not
macOS-only.

```bash
uv run python benchmarks/reproduce_table1.py --tier bundled      # no download
uv run python benchmarks/reproduce_table1.py --tier external \
  --data benchmarks/data/ds002718_sub-002_eeg70_full.npy         # needs the download
```

Two tiers, because the rows do not all need the same data. The **bundled** tier covers the Amari
distance, the score-function and sufficient-statistics checks, and the multi-model rows, all from the
committed sample. The **external** tier covers the headline correlation and log-likelihood rows, which
need the well-determined recording ($k\approx153$); that dataset is public but is a manual download.

Verification is not free, and the cost is documented rather than glossed:
[`benchmarks/README_dimsweep.md`](https://github.com/sccn/pAMICA/blob/main/benchmarks/README_dimsweep.md)
gives measured wall-clock for each tier, the download size, the hardware assumed, and what changes
without a GPU. Read it before starting the external tier. There is deliberately no cheap reduced-budget
tier: below $k\approx60$ the decomposition is under-determined and backends diverge for legitimate
reasons, so a faster variant would reproduce a noisier number and invite the misreading this section
exists to prevent.

Two general checks remain useful and are much quicker, but note that neither reproduces a row of
the paper's table: `validate_implementations.py` defaults to a single seed at 100 iterations with `do_newton`
read from `sample_params.json` (it reproduces the [per-backend harness rows](#parity-rows-per-backend) above),
and `pytest` runs the parity and behavior suite.

```bash
uv run python validate_implementations.py --backend all  # single-model parity report, every backend
uv run pytest                                             # the full parity/behavior test suite
```

The multi-model ensemble and Amari detail regenerate from saved fits (no re-fitting) with
`uv run python .context/issue-351/multimodel_ensemble.py --from-npz .context/issue-351/raw/table1_bundled/bundled_multimodel_ensemble.npz`. The cross-platform benchmark and equivalence
figures are produced by `benchmarks/benchmark_decompose.py` (and the sweep scripts alongside it);
the underlying findings are in `.context/issue-84/` and `.context/issue-90/`.

### Parameter files

`sample_data/sample_params.json` is the JSON parameter file used above (loaded via
`AMICA.from_params_file`); its keys mostly reuse Fortran's `.param` names (`lrate`, `do_newton`,
`rho0`, `block_size`, `max_iter`, `num_models`, ...), but not all of them match one-to-one
(for example `num_mix` here vs `num_mix_comps` in Fortran's `input.param`).

`AMICA.from_params_file` also reads the literal Fortran `input.param` text format directly
(issue #132), so the exact file that drives the reference binary can drive pamica too, instead of
maintaining a hand-translated JSON copy. The format is auto-detected by sniffing the file content (JSON starts with
`{`/`[`; anything else is read as the Fortran text format -- the extension is
never trusted), so no new API is needed:

```python
from pamica import AMICA

model = AMICA.from_params_file("sample_data/input.param")   # Fortran text format
model = AMICA.from_params_file("sample_data/sample_params.json")  # JSON, as before
```

Every backend reads through one function now (issue #304): `pamica.fortran_params.read_params_file`
is the single params-file entry point, used by `AMICA.from_params_file` above and by
`AMICA_NumPy(params_file=...)` / `AMICA_NumPy.from_params_file` (the legacy backend's `params_file`
used to accept only JSON, raising a raw `json.JSONDecodeError` on a Fortran text file).
It content-sniffs the same way (JSON if the file starts with `{`/`[`, Fortran text otherwise --
a JSON top level that is not an object raises `ValueError`) and returns pamica's canonical keys
either way.
A JSON file's own keys pass through one alias table, `JSON_ALIAS_TO_CANONICAL`:
pamica's JSON schema (`sample_params.json`, `numpy_impl/params.json`) spells five settings
differently from the canonical/constructor name --

| JSON schema key   | canonical pamica key | Note                                          |
| ------------------ | --------------------- | ---------------------------------------------- |
| `min_grad_norm`    | `min_nd`              | same rename Fortran's own keyword needs        |
| `max_decs`         | `maxdecs`             | same rename Fortran's own keyword needs        |
| `numrej`           | `maxrej`              | same rename Fortran's own keyword needs        |
| `num_mix_comps`    | `num_mix`             | same rename Fortran's own keyword needs        |
| `share_int`        | `share_iter`          | JSON-schema-only; Fortran's own spelling already matches |

-- and a file carrying both a setting's alias and its canonical key (e.g. both `max_decs` and
`maxdecs`) raises `ValueError` naming both rather than picking one silently.
Applying this table is itself a behavior change for the PyTorch wrapper:
`sample_params.json`'s own `max_decs`/`min_grad_norm`/`share_int` settings previously matched
neither a named `fit()` parameter nor an `AMICATorchNG` keyword under their raw JSON spelling,
so they were only named in the "not applied" warning rather than applied;
fitting from that file now applies them.

The Fortran-text branch (`read_fortran_param_file`) parses Fortran's whitespace-separated
`key value` lines (`#` full-line comments, plus a deliberately permissive inline `" #..."`
trailing comment; ints/floats/strings, including Fortran's `d`/`D` double-precision exponent
marker; `0`/`1` boolean flags using Fortran's own `k == 1` semantics) into a dict targeting
pamica's actual Python call surface: `AMICA.fit`'s named parameters (`max_iter`, `lrate`,
`do_mean`, `do_sphere`, `do_newton`) and `AMICATorchNG` constructor keywords. It was built by
reading every `case('...')` arm of `amica15.f90`'s parameter parser (~amica15.f90:3100-3700)
against `AMICATorchNG`'s constructor and `validate_implementations.py`'s `_NG_PARAMS`/
`_HANDLED_KEYS`. `AMICA.from_params_file` stashes the translated dict on the returned instance, and
`fit()` applies it as **per-call defaults**: an argument passed explicitly to `fit()` always wins
over the file's value, whether that argument is one of the five named parameters above or an
`AMICATorchNG` keyword passed through `**kwargs` (e.g. `block_size`, `rho0`, `newt_start`).

89 Fortran keywords are recognized; 60 (59 distinct pamica-side names) are translated and 29 are
deliberately unsupported (checkpoint warm-start, per-family EM freeze toggles, FIR/DFT
pre-filtering, console reporting, ...) --
a keyword this reader drops always fires a `logger.warning` naming it, whether that is because
it is a real Fortran keyword pamica has no equivalent for, or because it is not a Fortran keyword
this reader recognizes at all (the bundled `sample_data/input.param` template itself carries three
such stale entries -- `field_blocksize`, `doPCA`, `load_W` -- that predate this parser and are not
`case('...')` arms in `amica15.f90` either, so the reference binary already ignores them too). A
malformed line (a keyword with no value, or a non-empty file where not one keyword is recognized
by the reference parser at all -- e.g. a JSON file mistakenly handed to `read_fortran_param_file`
directly) raises `ValueError` rather than being dropped or defaulted silently. Data-location
metadata the file carries (`files`, `outdir`, `data_dim`, `field_dim`, ...) matches no
`fit()`/`AMICATorchNG` parameter by design; `fit()` names these in a single warning as
"not applied" rather than forwarding or silently dropping them.

Only three keywords are renamed, because Fortran spells them differently from the pamica-side
(constructor) name:

| Fortran keyword (`input.param`) | pamica key   | Note                                    |
| -------------------------------- | ------------ | ---------------------------------------- |
| `min_grad_norm`                  | `min_nd`     | matches `AMICATorchNG.min_nd`            |
| `max_decs`                       | `maxdecs`    | matches `AMICATorchNG.maxdecs`           |
| `numrej`                         | `maxrej`     | matches `AMICATorchNG.maxrej`            |

`num_mix_comps`/`num_mix` both collapse to the pamica key `num_mix`, read directly by
`from_params_file` to size the instance (`AMICA(n_mix=...)`) before the rest of the dict ever
reaches `fit()`. `share_iter` is **not** renamed -- it already matches `AMICATorchNG.share_iter`
exactly, so this table has no entry for it. `sample_params.json`'s own JSON schema instead spells
the same setting `share_int`; the JSON alias table above (`JSON_ALIAS_TO_CANONICAL`) is what maps
that JSON-schema spelling -- and the JSON schema's `max_decs`/`min_grad_norm` spellings for the
two renamed settings above -- to the canonical names on the way in, not this Fortran-keyword table.

`do_opt_block`, `blk_min`, `blk_max` and `blk_step` moved from the unsupported table to identity
mappings with issue #232: pamica now implements the block-size search under Fortran's own four
names and Fortran's arithmetic stepping, so a file carrying them is applied rather than warned
about and dropped. pamica's *defaults* for the three bounds differ (Fortran's 128-1024 is far
below where any pamica backend peaks), but a file that sets them is honored as written.

`writestep`, `do_history` and `histstep` moved the same way with issue #304: the reader
translates what *any* pamica backend can honor, not just the one a given call targets, and
each consumer warns about what *it* cannot apply.
The legacy NumPy backend is the one backend that implements periodic on-disk checkpointing
under these exact names today (`numpy_impl/core.py`'s fit loop and `_write_history`);
torch/MLX have no matching mechanism yet (issue #312), so `AMICA.fit` names these three in
its own "not applied" warning instead of silently dropping them.

Every other translated keyword keeps its Fortran spelling; see `FORTRAN_TO_PAMICA_KEY` and
`FORTRAN_UNSUPPORTED_KEYS` in `pamica/fortran_params.py` for the full tables.

The legacy NumPy backend reads through the same `read_params_file` (`AMICA_NumPy(params_file=...)`
or the `from_params_file` classmethod -- renamed from `from_json_file`, since it now accepts
both formats), mapping the canonical keys above to its own attribute spellings (`min_nd` ->
`min_grad_norm`, `maxdecs` -> `max_decs`, `share_iter` -> `share_int`; `maxrej`/`num_mix` need
no translation, already matching).
A params-file setting this backend does not consume (e.g. `kurt_start`/`num_kurt`/`kurt_int`,
the adaptive-pdf schedule it has no family switch for) is named in a single `logger.warning`
rather than silently dropped.
`AMICA_NumPy(pdftype=...)` with anything other than `0` -- the only source-density family
this backend implements -- raises `NotImplementedError` at construction (see the
backend-differences table above).
The NumPy CLI (`python -m pamica.numpy_impl.cli`) reads through the same helper
(`_read_numpy_keyed_params`), so it accepts both formats too, instead of its own separate
`json.load` (issue #304).
