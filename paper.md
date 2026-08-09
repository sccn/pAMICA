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

Independent Component Analysis (ICA) separates electroencephalographic and magnetoencephalographic (EEG/MEG) recordings into maximally independent sources
that isolate brain, muscle, and artifact activities for downstream analysis [@makeig1995independent; @vigario1997independent; @iversen2019megeeg].
Adaptive Mixture ICA (AMICA) [@palmer2012amica] generalizes single-model ICA to a mixture of such models with adaptive source densities,
and produces the least dependent and most dipolar (hence most physiologically interpretable) EEG decompositions among the algorithms benchmarked by @delorme2012independent.
Its reference implementation, written by Jason Palmer, is a Fortran program distributed as a compiled binary callable from MATLAB/EEGLAB:
awkward to install, restricted to the central processing unit (CPU), and unusable from Python.

`pamica` is a Python implementation of AMICA that reproduces the reference Fortran results within numerical tolerance
while running on the CPU, NVIDIA graphics processing units (GPUs, via CUDA), and Apple GPUs (through Apple's MLX array framework [@mlx2023]).
It is a complete reimplementation on PyTorch [@paszke2019pytorch], NumPy [@harris2020array], and SciPy [@virtanen2020scipy] rather than a wrapper around the binary,
and exposes a scikit-learn-style estimator under a BSD-3-Clause license.
It writes the format EEGLAB's AMICA loader reads, so established MATLAB tooling consumes its results unchanged.
The software is at <https://github.com/sccn/pAMICA> (archived at doi:10.5281/zenodo.21312148).

# Statement of need

AMICA decompositions are well suited to equivalent-dipole source localization and automated component classification [@piontonachini2019iclabel].
Yet the reference implementation is MATLAB-only Fortran, an increasing obstacle as neuroimaging analysis moves toward Python, for example MNE-Python [@gramfort2013meg]:
modern pipelines need an AMICA that runs natively in Python and on a GPU, and that is validated to reproduce the Fortran reference numerically.

General-purpose Python ICA implementations do not fill this gap.
`scikit-learn` and `MNE-Python` provide FastICA [@hyvarinen2000independent] and Infomax [@bell1995information; @lee1999independent],
while Picard [@ablin2018faster] offers faster-converging maximum-likelihood ICA;
none implement AMICA's mixture of models, adaptive generalized-Gaussian source densities, or Newton updates,
so none reproduce AMICA decompositions.
`pamica` targets EEG and MEG analysts wanting AMICA-quality decompositions inside a Python pipeline,
GPU users wanting faster runs than the CPU-only binary,
and methodologists needing a transparent reference implementation to build on.

# State of the field

`pamica` complements rather than replaces the reference Fortran AMICA used with EEGLAB [@delorme2004eeglab]:
it keeps the same output format, adds a Python API with GPU support,
and can run the reference Fortran itself through a bundled dependency-free native build.
Two other Python AMICA reimplementations have appeared [@esmaeili2025amica; @herforth2026pyamica], both providing MNE-Python-compatible objects;
`pamica` adds a scikit-learn-style array API, byte-identical EEGLAB I/O, an MLX backend, and an optional MNE-Python wrapper.
What sets it apart is the depth of its Fortran-parity validation:
score functions exact to floating-point resolution against the literal Fortran expressions,
and a distributional-similarity framework for the non-identifiable multi-model case.

# Software design

The governing decision was to treat numerical parity with Palmer's Fortran, rather than convergence to some independent solution, as the definition of correctness.
The rest follows from it.
Wrapping the existing binary would have secured parity for free but inherited its CPU-only, MATLAB-facing design; reimplementing put parity at risk.
`pamica` does both, porting the algorithm natively and shipping the reference binary as a selectable backend,
so users can reproduce the parity claims on their own data and hardware.
For the same reason the port follows the reference's natural-gradient [@amari1998natural] expectation-maximization (EM) formulation
rather than a modern automatic-differentiation optimizer:
an Adam/autograd backend was written early and then deleted, because it converged to different optima and made the name "AMICA" ambiguous.
The port covers exact-EM mixture updates, a positive-definite Newton step [@palmer2008newton], symmetric zero-phase-component-analysis sphering,
the five source-density families, a mixture of ICA models, and component sharing across models.
It also computes mutual information reduction and pairwise mutual information, the separation-quality metrics used to benchmark ICA algorithms [@delorme2012independent; @frank2023optimal].

Three array backends (PyTorch, MLX, NumPy) sit behind one estimator.
That duplication is deliberate: the NumPy path stays readable as an executable specification,
while MLX exists because PyTorch's Metal backend is consistently slower than the CPU on Apple hardware (Table 2).
Double precision is the default because parity demands it;
single precision is offered for Apple GPUs, which have no float64, at a documented cost of a few significant digits.

# Validation

Conformity is measured two ways: by Hungarian-matched component correlation,
and by the Amari distance [@amari1996new], a relabeling- and scale-invariant unmixing-matrix metric needing no assignment step.
Both implementations ran AMICA's default 2000 iterations with Newton disabled, to isolate the algorithm from initialization.
The single-model comparison uses a well-determined external recording
(OpenNeuro ds002718, $k\approx153$, where $k$ = frames over squared channel count [@frank2025sufficient])
alongside the bundled 32-channel sample ($k\approx30$).
A mixture of ICA models is not partition-identifiable, so exact partition parity is the wrong bar for the multi-model case;
it is judged instead by whether the two implementations sample a similar distribution of solutions, over ensembles of 20 runs each (\autoref{fig:ensemble}).
A permutation test finds no evidence that cross-implementation agreement is worse than Fortran's own run-to-run agreement.

| Regime | Metric (dataset) | Result (mean) |
|---|---|---|
| Single-model | Log-likelihood gap (ds002718) | within ~0.0005 of $-3.6993$ |
| Single-model | Component correlation (ds002718) | 0.998 |
| Single-model | Amari distance (bundled sample) | 0.006 |
| Single-model | Score functions, sufficient statistics (bundled) | exact, $\sim\!10^{-15}$ |
| Multi-model | Component correlation, one run: cross; within-Fortran | 0.65; 0.64 (sd 0.05) |
| Multi-model | Amari distance, one run: cross; within-Fortran | 0.163; 0.174 (sd 0.02) |
| Multi-model | Ensemble agreement, cross $-$ within-Fortran | correlation $+0.011$ ($p=0.96$); Amari $-0.011$ ($p>0.999$) |

: Parity of `pamica` with the Fortran reference.
Values are means (sd, standard deviation) over matched components or, for multi-model, over 20-run ensembles (190 within-, 400 cross-implementation run pairs).

![Multi-model ensemble partition-correlation (A) and log-likelihood (B) distributions for 20 `pamica` and 20 Fortran fits of the sample EEG; dashed lines mark each mean.
The within-Fortran, within-`pamica`, and between-implementation correlation distributions overlap,
so the single-run correlation reflects intrinsic run-to-run spread rather than a gap to the reference.
Panel B's apparent separation is a 0.009 gap on a ~0.035 axis.\label{fig:ensemble}](docs/assets/figures/multimodel-ensemble.png){ width=100% }

All backends converge to the same single-model log-likelihood on real EEG (maximum pairwise difference ~0.003),
and single-precision runs agree with double precision to four or five significant digits.
On Apple Silicon, MLX is the fastest backend (Table 2); double-precision CUDA is the reproducible NVIDIA path.

| Backend (device) | Precision | ms / iteration |
|---|---|---:|
| MLX (Apple GPU) | float32 | 25 |
| CUDA (RTX 4090) | float64 | 39 |
| Fortran (i9-13900K, 24 cores) | float64 | 30 |
| Fortran (Apple Silicon, 8 cores) | float64 | 70 |
| PyTorch CPU (Apple Silicon) | float64 | 193 |
| PyTorch MPS (Apple GPU) | float32 | 255 |
| NumPy (Apple Silicon) | float64 | 622 |

: Single-model throughput on real 70-channel EEG (ds002718; `n_mix`=3, `pdftype`=0, `block_size`=512; warm, minimum of repeated runs).
The Fortran rows come from a core-count sweep at each backend's plateau; other CPU rows use platform-default threads.

The correctness harness never uses synthetic data, and the multi-model and score-function checks need no external download.
Full tables, the data-size sweep, and reproduction commands are in the [documentation](https://eeglab.org/pAMICA/guides/validation/).

# Research impact statement

`pamica` was first released in July 2026, so its case rests on readiness and early use rather than accumulated citations.

Its research role is to make an established method usable where it was not.
AMICA underlies a substantial EEG literature but has been reachable only through the MATLAB-callable Fortran binary.
`pamica` reproduces that binary's results from Python and writes its output format,
so a decomposition computed here is read by existing EEGLAB analyses unchanged.
It also redistributes the reference Fortran as a dependency-free build for macOS, Linux, and Windows,
removing a long-standing installation obstacle for users of the original.

Early signals: seven tagged releases, four on the Python Package Index (513 downloads in the month before submission), and an archived Zenodo record.
Two users outside the author group have opened issues from their own analyses, one from a 236-channel, 8.3-million-sample decomposition;
MNE-Python is publicly weighing which AMICA implementation to adopt.
Integration into this Center's Python preprocessing tools and the NEMAR archive is underway, not complete.
The validation harness, sample data, and reproduction commands ship with the package, so Table 1 can be re-run independently.

# AI usage disclosure

Generative AI was used in this project and is disclosed here under the journal's AI usage policy.

**Tools.** Anthropic's Claude models (Sonnet and Opus families), driven through the Claude Code command-line assistant.
The instructions given to them are public in the repository (`AGENTS.md`, `CLAUDE.md`, and `.rules/`).

**Scope.** Assistance covered the source code (translating the reference Fortran into Python, refactoring, scaffolding tests),
the project documentation, and the drafting and copy-editing of this manuscript.

**Human oversight.** The authors made the design decisions and take responsibility for the result.
Parity with the Fortran reference as the correctness criterion, the natural-gradient EM formulation, the backend architecture,
the treatment of the multi-model case as distributional rather than pointwise, and the acceptance thresholds were chosen by the authors, not by a model.
Every AI-assisted change was reviewed by a human before merge and run through the parity harness,
which scores output against the reference binary on real recordings rather than against generated fixtures.
The reported numbers came from running the software and were checked against their source records, as was every bibliographic entry.

# Acknowledgements

We thank Jason Palmer and his advisor Ken Kreutz-Delgado, co-developers of AMICA, for the reference implementation.
We also thank the EEGLAB community for the tools and sample data used to validate this work.
Two of the authors are original developers of the methods `pamica` builds on:
S.M. co-developed the AMICA algorithm [@palmer2012amica] and A.D. is a lead developer of EEGLAB [@delorme2004eeglab].
This work was supported by The Swartz Foundation (Old Field, NY) to the Swartz Center for Computational Neuroscience
and by National Institutes of Health grant R01-NS047293 (to A.D. and S.M.).

# References
