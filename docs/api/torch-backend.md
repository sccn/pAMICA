# PyTorch backend (AMICATorchNG)

The natural-gradient EM backend that reaches Fortran parity (Newton, exact-EM
mixture updates, symmetric-ZCA sphere, Jacobian log-likelihood). The
[`AMICA`](amica.md) interface delegates to this class; use it directly for
lower-level control.

Its defaults are the ones the wrapper uses
(compared with the compiled binary's and EEGLAB's in [Default settings](../guides/amica-differences.md#default-settings-issue-354)).
With `device=None` on a Mac, the constructor runs a float64 model on the CPU, because Metal Performance Shaders (MPS) cannot represent float64,
and logs a warning,
so `AMICATorchNG(n_channels)` with default arguments runs on Apple Silicon (issue #354).
An explicit `device="mps"` needs `dtype=torch.float32` and raises `ValueError` without it.

::: pamica.AMICATorchNG
