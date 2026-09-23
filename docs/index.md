# pAMICA

Python (PyTorch) implementation of **Adaptive Mixture Independent Component
Analysis (AMICA)** that reproduces the results of the reference Fortran binary,
with GPU / Apple-GPU / CPU support. AMICA is a blind source separation algorithm
widely used for electroencephalography (EEG) and electromyography (EMG) source
decomposition.

pamica exposes a scikit-learn-style interface over a natural-gradient
expectation-maximization (EM) backend that matches the Fortran reference
(`amica15`) to within numerical tolerance: single-model log-likelihood and
Hungarian-matched component correlation both agree with Fortran on real EEG.

## Why pamica

- **Fortran parity is the specification.** Correctness is defined as matching the
  reference Fortran output within tolerance, not merely converging. See
  [Validation & Parity](guides/validation.md).
- **Multiple backends, one API.** The default PyTorch natural-gradient EM backend
  runs on CUDA, CPU, and (in float32) Apple MPS; an optional
  [MLX backend](guides/backends.md) targets Apple-Silicon GPUs, and a legacy
  NumPy reference is retained as an oracle. See [Backends & Devices](guides/backends.md).
- **Feature-complete against the reference.** Newton optimization, exact-EM
  mixture updates, all five source-density families, component sharing, and
  outlier rejection are ported and validated.

## Quick links

- [Getting Started](getting-started.md): install and run your first decomposition.
- [Backends & Devices](guides/backends.md): pick CUDA / CPU / MLX and float32 vs float64.
- [Validation & Parity](guides/validation.md): how correctness is defined and checked.
- [pamica vs. AMICA](guides/amica-differences.md): every deliberate difference from the reference.
- [API Reference](api/index.md): the `AMICA` interface and the backend classes.

!!! note "Precision and parity"
    The natural-gradient backend computes in float64 for Fortran parity. Apple
    MPS cannot represent float64, so parity runs use CPU or CUDA; float32 is
    required on MPS/MLX but is ~7-significant-digit, not float64-parity, and
    is not a general speedup (the Apple-GPU speed comes from the MLX backend).
    Use float64 for reference-parity runs.
