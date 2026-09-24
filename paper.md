---
title: 'pamica: GPU-accelerated Adaptive Mixture Independent Component Analysis in Python with Fortran parity'
tags:
  - Python
  - PyTorch
  - independent component analysis
  - blind source separation
  - EEG
  - neuroscience
authors:
  - name: Seyed Yahya Shirazi
    orcid: 0000-0001-5557-259X
    corresponding: true
    affiliation: 1
  - name: Arnaud Delorme
    orcid: 0000-0002-0799-3557
    affiliation: "1, 2"
  - name: Scott Makeig
    orcid: 0000-0002-9048-8438
    affiliation: 1
affiliations:
  - name: Swartz Center for Computational Neuroscience, Institute for Neural Computation, University of California San Diego, USA
    index: 1
  - name: Centre de Recherche Cerveau et Cognition (CerCo), CNRS, University of Toulouse, France
    index: 2
date: 9 August 2026
bibliography: paper.bib
---

# Summary

Independent Component Analysis (ICA) separates electroencephalographic and magnetoencephalographic (EEG/MEG) recordings into maximally independent sources that isolate brain, muscle, and artifact activities for downstream analysis [@makeig1995independent; @vigario1997independent; @iversen2019megeeg].
Adaptive Mixture ICA (AMICA) [@palmer2006super; @palmer2007modeling; @palmer2012amica] models each source with a flexible, self-adjusting probability density and lets several ICA models coexist,
one per segment of a recording.
Among the algorithms benchmarked by @delorme2012independent it returned the least dependent and most dipolar decompositions,
properties that the study links to physiological interpretability.
Its reference implementation, by Jason Palmer, is a Fortran program distributed as a compiled binary and usually run from MATLAB/EEGLAB;
it runs only on the central processing unit (CPU) and has no Python interface.

`pamica` reproduces the reference Fortran's single-model results within numerical tolerance, and its multi-model solutions spread like the reference's own,
while running on the CPU, NVIDIA graphics processing units (GPUs, via CUDA), and Apple GPUs (through Apple's MLX array framework [@mlx2023]).
It reimplements the algorithm on PyTorch [@paszke2019pytorch], NumPy [@harris2020array], and SciPy [@virtanen2020scipy],
and exposes a scikit-learn-style estimator under a BSD-3-Clause license.
It writes the format EEGLAB's AMICA loader reads, so established MATLAB tooling consumes its results unchanged.
The software is at <https://github.com/sccn/pAMICA> (archived at doi:10.5281/zenodo.21312148).

# Statement of need

AMICA decompositions are well suited to equivalent-dipole source localization and automated component classification [@piontonachini2019iclabel].
Yet the reference implementation is used mainly through MATLAB, a growing obstacle as neuroimaging analysis moves toward Python, for example MNE-Python [@gramfort2013meg]:
modern pipelines need an AMICA that runs natively in Python and on a GPU, and that is validated to reproduce the Fortran reference numerically.

General-purpose Python ICA implementations do not fill this gap.
`scikit-learn` and `MNE-Python` provide FastICA [@hyvarinen2000independent] and Infomax [@bell1995information; @lee1999independent],
while Picard [@ablin2018faster] offers faster-converging maximum-likelihood ICA;
none implement AMICA's mixture of models, adaptive generalized-Gaussian densities, or Newton updates, so none can reproduce its decompositions.
`pamica` serves analysts who want AMICA-quality decompositions in Python and anyone with a GPU who wants faster runs than the CPU-only binary;
it is also a transparent reference for methodologists to build on.
The parity measurements reported here use EEG.
The algorithm itself is modality-agnostic, and an external user has fit rank-reduced, Maxwell-filtered MEG with it end to end (see the research impact statement),
but parity with the reference has not yet been measured on MEG.

# State of the field

`pamica` is designed to work alongside the reference Fortran AMICA used with EEGLAB [@delorme2004eeglab]:
it keeps the same output format, adds a Python API with GPU support,
and can run the reference Fortran itself through a bundled dependency-free native build.
Three other Python AMICA reimplementations have appeared as of August 2026 [@huberty2025amicapython; @esmaeili2025amica; @herforth2026pyamica], oriented toward MNE-Python;
`pamica` adds a scikit-learn-style array API, output in EEGLAB's exact on-disk layout, an MLX backend, and an optional MNE-Python wrapper.
Its validation includes score functions checked against the literal Fortran expressions to floating-point resolution,
and a distributional comparison for the non-identifiable multi-model case.

# Software design

The central design decision was to define correctness as numerical parity with Palmer's Fortran,
and most of the other choices follow from it.
Wrapping the binary would have secured parity but inherited its CPU-only, MATLAB-facing design, while a reimplementation must demonstrate parity.
`pamica` ports the algorithm natively and also ships the reference binary as a runnable engine,
so users can check the parity claims on their own data and hardware.
For the same reason the port follows the reference's natural-gradient [@amari1998natural] expectation-maximization (EM) formulation.
An early Adam/autograd backend was removed because it converged to different optima, which made results labeled "AMICA" hard to compare with the reference.
The port covers exact-EM mixture updates, a positive-definite Newton step [@palmer2008newton], symmetric sphering,
the five source-density families, a mixture of ICA models, component sharing, and the mutual-information metrics used to score separation quality [@frank2023optimal].

Three array backends (PyTorch, MLX, NumPy) implement the same algorithm behind a common estimator API, with the reference Fortran a fourth.
NumPy is kept as a readable executable specification.
MLX is included because it is the more efficient path on Apple GPUs: in the block-size sweep, PyTorch's Metal backend was slower than MLX at every size,
and at `pamica`'s shipped defaults slower than the CPU it runs beside (see the [documentation](https://eeglab.org/pAMICA/guides/validation/)).
Double precision is the default because the parity comparisons need it; single precision is available for Apple GPUs, which have no float64.

# Validation

Parity is measured two ways: by Hungarian-matched component correlation,
and by the Amari distance [@amari1996new], a relabeling- and scale-invariant metric that needs no assignment step.
Both implementations ran EEGLAB's default of 2000 iterations with Newton disabled (`pamica`'s own default), to isolate the algorithm from initialization.
With Newton enabled, some of the weakest components settle into different basins of equal or higher likelihood from different starts, in the reference's runs as in `pamica`'s:
against one reference run, one `pamica` seed of three differed on eight of seventy components and the other two matched at 0.995 and 0.996,
and two reference runs from different seeds differed on two.
From a shared start the two implementations end with the same components (correlation 0.9999998).
The single-model comparison uses a well-determined external recording (OpenNeuro ds002718, mirrored in the NEMAR archive [@delorme2022nemar] as [on002718](https://doi.org/10.82901/nemar.on002718),
$k\approx153$, where $k$ = frames over squared channel count [@frank2025sufficient]) alongside the bundled 32-channel sample ($k\approx30$).
On the bundled sample one of the five reference runs ended in a lower-likelihood basin and sets the five-pair Amari mean;
over 50 run pairs the distance between the implementations (0.004) is close to the reference's own run-to-run distance (0.005).
A mixture of ICA models is not partition-identifiable,
so the implementations are compared by the distributions of solutions they sample, over ensembles of 20 runs each (\autoref{fig:ensemble}).
One-sided run-level permutation tests of whether cross-implementation agreement is worse than the reference's own run-to-run agreement
give $p=0.88$ by correlation and $p=0.051$ by Amari distance, by which `pamica`'s ensemble spreads slightly more than the reference's.

| Regime | Metric (dataset) | Result (mean) |
|---|---|---|
| Single | Log-likelihood gap (on002718) | $6\times10^{-6}$, both at $-3.6993$ |
| Single | Component correlation (on002718) | 0.9996 (Fortran vs Fortran 0.998) |
| Single | Amari distance (bundled) | 0.011 (5 run pairs); 0.004 (50 pairs) |
| Single | Score functions; sufficient statistics | $\le 2\times10^{-15}$ absolute; $3\times10^{-16}$ relative |
| Multi | Correlation, one run: cross; within-Fortran | 0.632; 0.626 (sd 0.04) |
| Multi | Amari, one run: cross; within-Fortran | 0.172; 0.166 (sd 0.02) |
| Multi | Ensemble agreement, cross $-$ within-Fortran | correlation $+0.006$ ($p=0.88$); Amari $+0.005$ ($p=0.051$) |
| Multi | Ensemble log-likelihood: Fortran; `pamica` | $-3.3543$; $-3.3541$ (Kolmogorov-Smirnov $p=0.83$) |

: Parity of `pamica` with the Fortran reference. Multi-model rows are over 20-run ensembles (190 within-Fortran and 190 within-`pamica`, plus 400 cross-implementation pairs); sd is the standard deviation, given where computed;
$p$ values are one-sided run-level permutation tests.

![Multi-model ensemble partition-correlation (A) and log-likelihood (B) distributions, 20 `pamica` and 20 Fortran fits of the sample EEG; vertical lines mark each mean.
A's three distributions overlap: the single-run correlation of ~0.63 matches the reference's agreement with itself.
B's two means lie 0.0002 apart on a ~0.02 axis.\label{fig:ensemble}](docs/assets/figures/multimodel-ensemble.png){ width=100% }

From a shared start with the harness settings, the single-precision MLX backend and the double-precision PyTorch backend differ by $5\times10^{-6}$ in log-likelihood after 100 iterations,
with mean matched component correlation 0.9999999 and Amari distance $5\times10^{-5}$.
In the throughput benchmark's runs on a 30,000-frame excerpt of the external recording, the backends that share a start agree after 25 iterations to $10^{-5}$ in log-likelihood on 32 and 48 channels;
with 70 channels ($k\approx6$) they differ by up to $10^{-3}$, about the effect of perturbing one sample by $10^{-9}$ µV.
Double precision remains the default for parity work, and double-precision CUDA is the reproducible NVIDIA path.

On real 70-channel EEG at `block_size=512`, per-iteration cost is 25 ms for MLX on an Apple GPU, 39 ms for double-precision CUDA on an RTX 4090,
and 30 ms for native Fortran on a 24-core i9-13900K, against 193 ms for PyTorch on an Apple-Silicon CPU and 255 ms for PyTorch-Metal.
That comparison is block-size-confounded: PyTorch-Metal is far more block-size-sensitive than the CPU it runs beside, falling from 431 to 30.5 ms/iteration on the bundled 32-channel sample
between `block_size=512` and `pamica`'s current 8192 default, still behind the CPU's 21.7 ms/iteration there, and to 13.5 ms/iteration, below the CPU's 15.8 ms,
at a further-tuned single-block setting that is memory-limited rather than a free win, since peak block memory scales with `block_size`, which is why 8192 stays the shipped default.
MLX stays the fastest Apple backend throughout the sweep.
Full tables, the data-size sweep, and reproduction commands are in the [documentation](https://eeglab.org/pAMICA/guides/validation/); the correctness harness never uses synthetic data.
The harness and sample data are in the source repository, and every row of Table 1 but the two on002718 entries re-runs from the bundled sample alone.

# Research impact statement

AMICA's source model and estimation were published in 2006-2008 [@palmer2006super; @palmer2007modeling; @palmer2008newton],
and it ranked first of the twenty-two algorithms benchmarked by @delorme2012independent.
MNE-Python is publicly weighing which of the Python ports to adopt (`mne-tools/mne-python` issue 13819),
where an MNE core developer and the author of a competing port lean toward `pamica`,
since this Center develops AMICA and maintains the reference Fortran and EEGLAB.

`sccn/pAMICA` has been public since 2021, with public issues, pull requests, nine releases (six on PyPI), and a Zenodo archive.
Three researchers outside the author group have filed issues, two of them mid-analysis on their own data:
a 236-channel, 8.3-million-sample EEG decomposition (issue 207) and Maxwell-filtered 306-channel MEG (issue 221);
the third is adopting AMICA for a multiverse analysis in MNE-Python (issue 206).
Integration into this Center's Python preprocessing and into NEMAR, the public archive we operate (800 datasets, ~40,000 participants, 55 TB), is in progress.

# AI usage disclosure

Generative AI was used in this project, disclosed here under the journal's policy.

**Tools.** Anthropic's Claude models (Sonnet and Opus families), through the Claude Code command-line assistant.
The instructions given to them are public in the repository (`AGENTS.md`, `CLAUDE.md`, `.rules/`).

**Scope.** The source code (translating the reference Fortran into Python, refactoring, scaffolding tests), the documentation, and the drafting and copy-editing of this manuscript.

**Human oversight.** The authors made the design decisions and take responsibility for the result.
Parity as the correctness criterion, the natural-gradient EM formulation, the backend architecture,
the distributional treatment of the multi-model case, and the acceptance thresholds were chosen by the authors.
Every AI-assisted change was reviewed by a human before merge and run through the parity harness,
which scores output against the reference binary on real recordings.
The reported numbers came from running the software and were checked against their run records, as was every bibliographic entry.

# Acknowledgments

We thank Jason Palmer and his advisor Ken Kreutz-Delgado, co-developers of AMICA, for the reference implementation,
and the EEGLAB community for the tools and sample data used to validate this work.
Two authors developed the methods `pamica` builds on: S.M. co-developed AMICA [@palmer2012amica] and A.D. is a lead developer of EEGLAB [@delorme2004eeglab].
This work was supported by The Swartz Foundation (Old Field, NY) to the Swartz Center for Computational Neuroscience and by National Institutes of Health grant R01-NS047293 (to A.D. and S.M.).

# References
