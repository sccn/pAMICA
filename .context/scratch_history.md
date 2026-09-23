# pamica Scratch History

Debugging notes, failed attempts, and lessons. Prevents repeating dead ends.

## Debugging notes

### Likelihood divergence
- Fortran: iter 1 LL ~ -3.51 -> converges ~ -3.44. Early Python: ~1.42e6 -> ~1.51e6.
- Hypothesis: sign flip or missing `log()`; check `-log(p(x))` vs `log(p(x))`.
- Enhanced model produced positive LLs -> traced to wrong GG normalization constant (see research.md).

### NaN issues
- Occurred ~iter 49-50 (Newton start). `RuntimeWarning: invalid value in divide` at
  `dmu = updates['dmu'] / updates['dalpha']` -> `dalpha` near zero. Add epsilon to denominators.
- Gradient norm jumped from ~70 to ~2e5 at Newton start before clipping was added.

### Fortran structure references
- `get_updates_and_likelihood` (~line 1174), `accum_updates_and_likelihood` (~line 1671),
  `update_params` (~line 1878). Fortran separates accumulation; Python merged it.
- `funmod2.f90` holds special-function helpers (gamln/psifun/etc.), not the PDFs; PDF computation is inline in `amica17.f90`. (`fastlog/fastexp/fastpow` are declared but unused stubs in `amica17_header.f90`.)

## Known differences to watch
- Data layout: Fortran column-major vs Python row-major default (handled; verify edge cases).
- PDF computed inline in Python vs separate module in Fortran.
- Block processing simplified in Python vs Fortran's block-size optimization.
- Fortran 1-based vs Python 0-based indexing; OpenMP -> multiprocessing/joblib; BLAS/LAPACK -> scipy.linalg.

## Quick fixes to try
- Epsilon on all divisions: `dmu / (dalpha + 1e-10)`.
- Verify LL sign: possibly `ll = -sum(log(pdf + 1e-10))`.
- Bound values: `clip(value, -1500, 1500)` (Fortran `minlog`).

## Test data
- Sample data: 32 channels, 30504 samples, 32-bit float. Fortran output in `amicaout/`,
  Python in `pyresults/` (load via `loadmodout()`).

## Performance
- Fortran ~0.02s/iter; Python ~0.5s/iter (~25x slower). Matrix ops likely the bottleneck; profile.

## Debugging commands
```bash
python -m pytest pamica/tests/test_sample_data.py::test_sample_data_light -v -s
python -m pamica.amica_cli pamica/sample_data/sample_params.json --verbose --outdir debug_out
# Fortran build (if needed):
gfortran -O3 -fopenmp amica17.f90 funmod2.f90 -o amica -llapack -lblas
```

## Issue #159 (loadmodout W convention): OPEN, blocks scalp-topography plots
ROOT-CAUSED: `loadmodout` reads `W` (load.py:174), `sbeta` (:278) and `rho`
(:292) in C order; the writer writes all of them `order="F"` and only `alpha`/`mu`
are read back correctly. So `out.W`/`out.sbeta`/`out.rho` are wrong for GENUINE
Fortran output, and `out.A`/`out.svar`/`out.origord` (derived from W) with them.
`plot_topo_pdf` was CUT from Phase 4 over this: there is no correct formula on the
shipped loader. Do NOT hand-roll sources from an AmicaOutput until #159 lands. Three natural tests are degenerate here and will mislead you: the
activation-mean test (fitted mixture is near-symmetric, so it matches trivially
given E[sphered]=0), best-match correlation (shift-invariant, cannot see c at
all), and histogram-vs-PDF L1 (too insensitive to a 20%-of-spread shift). Full
evidence in the issue and `.context/issue-136/matlab_viz_verification.md`.

## Issue #136 (MIR/PMI visualizations): MATLAB gate for plots
Same run-and-observe posture as #155, extended to figures. postAmicaUtility is GPL
(pop_topohistplot.m / pop_modPMI.m carry explicit GPL-2.0-or-later headers), pamica
is BSD-3-Clause, so its .m source was never read: the gate used only `help` text,
rendered figures, and black-box I/O. Every plotted quantity is pinned to MATLAB
(mir vs Apache-2.0 getMIR.m at 1.7e-15; P(model|data) vs LLt2v at 1.4e-14; v
bit-exact; smoothed probability r=0.9886; pairwise_mi vs minfojp r=0.9887). Full
record + traps + how to re-run: `.context/issue-136/matlab_viz_verification.md`.
Three traps worth knowing before touching viz or metrics: (1) MATLAB's mInfoMatrix
is stored ALREADY REORDERED -- compared raw it reads r=-0.13 and looks like our PMI
is broken; un-permuted it is r=0.9887. (2) A naive `convolve(..., mode="same")`
Hanning smooth zero-pads and silently corrupts both plot edges (Lht ~ -108 got
dragged to -60), producing confidently wrong probabilities; divide by the window
overlap. (3) Do NOT derive sources from an `AmicaOutput` at all yet: `loadmodout` reads
`W`, `sbeta` and `rho` in the WRONG byte order, so there is no correct formula on
the shipped loader (#159). This entry asserted a formula TWICE and was wrong both
times, in opposite directions, and a reviewer escalated one of them as a critical
bug in correct code. The lesson is the meta one: do not assert a convention from
reasoning or from self-consistency checks; only an EXTERNAL oracle settles it.
(4) Chasing that turned up a REAL pre-existing bug: `numpy_impl/pdf.py` used
`gammaln` where the generalized Gaussian needs `gamma`, making compute_pdf return
a NEGATIVE density for any rho outside {1,2} (integral -8.82 at the default
rho0=1.5). Fit path unaffected (core.py has its own correct log-space version);
the shipped `numpy_impl/viz.py: plot_pdf_fits` was drawing wrong curves. Survived
because tests only ever covered rho=1.0/2.0, the special-cased correct branches.

## Issue #155 (LLt output): MATLAB interop re-verified for the new file
`LLt` (per-timepoint, per-model log-likelihood) was added to the writer in PR
#156, so it went through the same #92 MATLAB gate below. Result: real
`loadmodout15.m` under MATLAB R2025b reads pamica's `LLt` bit-exactly (single-
and multi-model), and both readers agree bit-exactly on the genuine Fortran
fixture. Full record + how to re-run:
`.context/issue-155/matlab_interop_verification.md`. The automated tests only
pin "we read Fortran"; only the MATLAB run pins "EEGLAB reads us", since a
self-consistently-wrong writer/reader pair passes every round-trip test. Re-run
it by hand if the LLt layout, `write_amicaout`, or `loadmodout` ever change.

## Issue #92 (EEGLAB drop-in output): the column-major layout fix (KEY)
Fortran/EEGLAB store arrays **column-major**. The MATLAB round-trip exposed that
pamica wrote the non-square mixture params (`alpha`/`mu`/`sbeta`/`rho`, shape
`(num_mix, num_comps)`) and `c`/`comp_list` in **C-order**, so real MATLAB
`loadmodout15.m` (column-major reads) got scrambled mixture params -- e.g. the
per-component mixture proportions did NOT sum to 1. FIXED (issue #92): the writer
`write_amicaout` and BOTH numpy readers (`loadmodout`, `data.py:load_results`) now
use `order="F"` for those arrays. Diagnostic that nails the layout: read genuine
`sample_data/amicaout/alpha` -- `reshape(3,32,order='F').sum(0)` is all 1.0
(correct); C-order is garbage `[0.49..1.31]`.
- `W` (square) stays C-order in the writer AND is byte-identical to Fortran: the
  internal-vs-true-unmixing transpose (#24) cancels against Fortran's column-major
  storage (`self.W` C-order bytes == Fortran column-major `W_true` bytes). `S` is
  symmetric (order-agnostic); `mean`/`gm`/`LL` are 1-D.
  CORRECTION (2026-09-22, #336): "`S` is symmetric (order-agnostic)" is wrong.
  Fortran writes `S` column-major like every other EEGLAB-read 2-D file; the
  square branch wrote it C-order, which only looked "order-agnostic" because the
  default zero-phase component analysis (ZCA) sphere is symmetric to ~1e-17.
  With `do_approx_sphere=False` the sphere is genuinely asymmetric and the old
  C-order write came back exactly transposed (measured `max|S_loaded - S| = 0.51`
  on the bundled sample). Fixed in #336: `S` is now `order="F"` in both branches.
- [SUPERSEDED by #159 -- this bullet's premise EXPIRED, read the update below]
  Remaining, deliberately-out-of-scope quirk: `loadmodout`/`load_results` still
  read the **W** file C-order, so the port's `mod.W` is the transpose of MATLAB's
  and its derived `A`/`svar`/`origord` use `pinv(self.W_int @ S)`. This does NOT
  corrupt values (unlike the mixture bug, which did) and nothing consumes those
  fields for correctness; every parity test matches `W` via transpose-tolerant
  Hungarian |corr|. Do NOT flip the W read without re-checking #24/#37 parity.
  UPDATE (#159): the "nothing consumes those fields for correctness" premise is
  no longer true -- #136/#137 derive sources from them -- and #92 ALSO missed
  `sbeta`/`rho` in the reader (they still lack `order="F"`; only `alpha`/`mu`/`c`/
  `comp_list` were fixed). Proven by recomputing the bundled Fortran fixture's OWN
  reported LL from its OWN written params: F-order W + F-order mixture reproduces
  its -3.4018730 to 5 significant digits (-3.4018468), while the shipped C-order
  read is off by 0.10-0.15. `A`/`svar`/`origord` are derived from the mis-read W
  and are therefore wrong too.
  Consequence: `AMICATorchNG.variance_order()` (uses `W_fort=self.W.T`, matching
  real MATLAB) is validated against the MATLAB-faithful column-major reader, NOT
  the numpy port's `origord`.

## Lessons / check first next time
- [ ] Positive LL almost always means a wrong PDF normalization constant.
- [ ] NaN at a phase transition (e.g. Newton start) points at unclipped gradients or zero denominators.
- [ ] Confirm initialization parity (seed, sphering, starting matrices) before chasing algorithmic bugs.
- [ ] Legacy test signatures drift (`load_data_file` dropped a parameter) - update call sites.
