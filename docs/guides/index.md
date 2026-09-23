# User Guide

The user guide covers how to run pamica in practice and how its results relate
to the reference implementation.

- **[Backends & Devices](backends.md)**: the available compute backends
  (PyTorch natural-gradient EM, optional MLX, legacy NumPy), how to select one
  through `AMICA` and `AMICAICA`, device selection (CUDA / CPU / MPS), float32
  vs float64, and performance guidance on real EEG.
- **[EEGLAB interoperability](eeglab.md)**: writing a fit as an EEGLAB
  `amicaout` directory and reading it with `loadmodout15`.
- **[Validation & Parity](validation.md)**: how correctness is defined as
  parity with the Fortran reference, the validation harness, and how
  cross-backend equivalence depends on data adequacy.
- **[Differences vs AMICA](amica-differences.md)**: every place pamica
  deliberately behaves differently from the Fortran reference, why, and how to
  restore the reference behavior.
