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
Adaptive Mixture ICA (AMICA) [@palmer2012amica] models each source with a flexible, self-adjusting probability density
and lets several ICA models coexist, one per segment of a recording.
Among the algorithms benchmarked by @delorme2012independent it recovers the least dependent and most dipolar decompositions,
and therefore the most physiologically interpretable ones.
Its reference implementation, by Jason Palmer, is a Fortran program distributed as a compiled binary callable from MATLAB/EEGLAB:
awkward to install, restricted to the central processing unit (CPU), and unusable from Python.

`pamica` reproduces the reference Fortran results within numerical tolerance
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
none implement AMICA's mixture of models, adaptive generalized-Gaussian densities, or Newton updates, so none can reproduce its decompositions.
`pamica` is for analysts who want AMICA-quality decompositions in Python, for anyone with a GPU who wants faster runs than the CPU-only binary,
and for methodologists who need a transparent reference to build on.
Validation to date is on EEG; the algorithm is modality-agnostic but MEG is untested.

# State of the field

`pamica` complements rather than replaces the reference Fortran AMICA used with EEGLAB [@delorme2004eeglab]:
it keeps the same output format, adds a Python API with GPU support,
and can run the reference Fortran itself through a bundled dependency-free native build.
Three other Python AMICA reimplementations have appeared as of August 2026 [@huberty2025amicapython; @esmaeili2025amica; @herforth2026pyamica], oriented toward MNE-Python;
`pamica` adds a scikit-learn-style array API, byte-identical EEGLAB I/O, an MLX backend, and an optional MNE-Python wrapper.
Its Fortran-parity validation goes further than theirs: score functions exact to floating-point resolution against the literal Fortran expressions,
and a distributional framework for the non-identifiable multi-model case.

# Software design

The governing decision was to treat numerical parity with Palmer's Fortran, rather than convergence to some independent solution, as the definition of correctness.
The rest follows from it.
Wrapping the binary would have secured parity for free but inherited its CPU-only, MATLAB-facing design; reimplementing put parity at risk.
`pamica` does both, porting the algorithm natively and also shipping the reference binary as a runnable engine,
so users can reproduce the parity claims on their own data and hardware.
For the same reason the port follows the reference's natural-gradient [@amari1998natural] expectation-maximization (EM) formulation
rather than an automatic-differentiation optimizer:
an Adam/autograd backend was written early and then deleted, because it converged to different optima and made the name "AMICA" ambiguous.
The port covers exact-EM mixture updates, a positive-definite Newton step [@palmer2008newton], symmetric sphering,
the five source-density families, a mixture of ICA models, component sharing, and the mutual-information metrics used to score separation quality [@frank2023optimal].

Three array backends (PyTorch, MLX, NumPy) implement the same algorithm behind a common estimator API, with the reference Fortran a fourth.
The duplication is deliberate: NumPy stays readable as an executable specification,
and MLX exists because PyTorch's Metal backend is slower than the CPU on Apple hardware.
Double precision is the default because parity demands it; single precision is available for Apple GPUs, which have no float64.

# Validation

Parity is measured two ways: by Hungarian-matched component correlation,
and by the Amari distance [@amari1996new], a relabeling- and scale-invariant metric that needs no assignment step.
Both implementations ran AMICA's default 2000 iterations with Newton disabled (`pamica`'s own default), to isolate the algorithm from initialization.
With Newton enabled and independent seeds, some of the weakest components settle into a different basin of equal or higher likelihood:
on one seed of three this affected ten of seventy components, while the other two matched the reference at ~0.99.
A matched initialization restores ~0.997, so this is a property of the initialization, not a parity defect.
The single-model comparison uses a well-determined external recording
(OpenNeuro ds002718, $k\approx153$, where $k$ = frames over squared channel count [@frank2025sufficient])
alongside the bundled 32-channel sample ($k\approx30$).
A mixture of ICA models is not partition-identifiable, so exact partition parity is the wrong bar there;
it is judged instead by whether the implementations sample a similar distribution of solutions, over ensembles of 20 runs each (\autoref{fig:ensemble}).
A permutation test finds no evidence that cross-implementation agreement is worse than Fortran's own run-to-run agreement.

| Regime | Metric (dataset) | Result (mean) |
|---|---|---|
| Single | Log-likelihood gap (ds002718) | within ~0.0005 of $-3.6993$ |
| Single | Component correlation (ds002718) | 0.998 |
| Single | Amari distance (bundled) | 0.006 |
| Single | Score functions, sufficient statistics | exact, $\sim\!10^{-15}$ |
| Multi | Correlation, one run: cross; within-Fortran | 0.65; 0.64 (sd 0.05) |
| Multi | Amari, one run: cross; within-Fortran | 0.163; 0.174 (sd 0.02) |
| Multi | Ensemble agreement, cross $-$ within-Fortran | correlation $+0.011$ ($p=0.96$); Amari $-0.011$ ($p>0.999$) |
| Multi | Ensemble log-likelihood: Fortran; `pamica` | $-3.3539$; $-3.3629$ (Kolmogorov-Smirnov $p=6\times10^{-5}$) |

: Parity of `pamica` with the Fortran reference. Multi-model rows are over 20-run ensembles (190 within-, 400 cross-implementation pairs); sd is the standard deviation, given where computed.
The final row is the one metric on which the ensembles differ significantly, and it is convergence speed rather than optimum quality: `pamica` reaches Fortran's mean by 200 iterations and passes it by 300.

![Multi-model ensemble partition-correlation (A) and log-likelihood (B) distributions, 20 `pamica` and 20 Fortran fits of the sample EEG; dashed lines mark each mean.
A's three distributions overlap, so the single-run correlation reflects intrinsic run-to-run spread, not a gap to the reference.
B's separation is Table 1's 0.009 gap on a ~0.035 axis.\label{fig:ensemble}](docs/assets/figures/multimodel-ensemble.png){ width=100% }

All backends converge to the same single-model log-likelihood on real EEG (maximum pairwise difference ~0.003),
and single precision matches double to four or five significant digits on that log-likelihood.
Component-level float32 agreement is not yet characterized at a matched iteration budget, so float64 remains the default for parity work,
and double-precision CUDA is the reproducible NVIDIA path.

On real 70-channel EEG, per-iteration cost is 25 ms for MLX on an Apple GPU, 39 ms for double-precision CUDA on an RTX 4090,
and 30 ms for native Fortran on a 24-core i9-13900K, against 193 ms for PyTorch on an Apple-Silicon CPU
and 255 ms for PyTorch-Metal, which never beats the CPU it runs beside.
Full tables, the data-size sweep, and reproduction commands are in the [documentation](https://eeglab.org/pAMICA/guides/validation/); the correctness harness never uses synthetic data.

# Research impact statement

`pamica` was first released in July 2026, so its case rests on readiness and early use rather than accumulated citations.

Because `pamica` writes the reference binary's output format, existing EEGLAB analyses read its decompositions unchanged,
so a lab can adopt Python without re-tooling everything downstream.
It also redistributes the reference Fortran as a dependency-free build for macOS, Linux, and Windows.

Seven releases have been published, four of them to the Python Package Index (513 downloads in the month before submission), with an archived Zenodo record.
One user outside the author group reported a bug from their own 236-channel, 8.3-million-sample decomposition (`sccn/pAMICA` issue 207);
another, the author of a competing reimplementation, raised completeness and packaging questions (issue 206).
MNE-Python is publicly weighing which AMICA implementation to adopt (`mne-tools/mne-python` issue 13819).
Integration into this Center's Python preprocessing and the NEMAR archive is underway, not complete.
The harness, sample data, and reproduction commands ship with the package, so a third party can re-run Table 1.

# AI usage disclosure

Generative AI was used in this project, disclosed here under the journal's policy.

**Tools.** Anthropic's Claude models (Sonnet and Opus families), through the Claude Code command-line assistant.
The instructions given to them are public in the repository (`AGENTS.md`, `CLAUDE.md`, `.rules/`).

**Scope.** The source code (translating the reference Fortran into Python, refactoring, scaffolding tests), the documentation, and the drafting and copy-editing of this manuscript.

**Human oversight.** The authors made the design decisions and take responsibility for the result.
Parity as the correctness criterion, the natural-gradient EM formulation, the backend architecture,
the distributional treatment of the multi-model case, and the acceptance thresholds were chosen by the authors, not by a model.
Every AI-assisted change was reviewed by a human before merge and run through the parity harness,
which scores output against the reference binary on real recordings rather than generated fixtures.
The reported numbers came from running the software and were checked against their run records, as was every bibliographic entry.

# Acknowledgements

We thank Jason Palmer and his advisor Ken Kreutz-Delgado, co-developers of AMICA, for the reference implementation,
and the EEGLAB community for the tools and sample data used to validate this work.
Two authors developed the methods `pamica` builds on: S.M. co-developed AMICA [@palmer2012amica] and A.D. is a lead developer of EEGLAB [@delorme2004eeglab].
This work was supported by The Swartz Foundation (Old Field, NY) to the Swartz Center for Computational Neuroscience
and by National Institutes of Health grant R01-NS047293 (to A.D. and S.M.).

# References
