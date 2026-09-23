# Getting Started

## Installation

pamica uses [UV](https://docs.astral.sh/uv/) for environment and dependency
management.

### From source

```bash
git clone https://github.com/sccn/pAMICA.git
cd pAMICA
uv sync            # install the project and its dependencies into .venv
```

A packaged release on the Python Package Index (PyPI) is planned; until then,
install from source as above.

### Optional Apple-Silicon GPU backend

The MLX backend is Apple-only and is therefore an optional extra; `import
pamica` never requires it.

```bash
uv sync --extra mlx   # or, in an existing environment: uv pip install mlx
```

Then select it with `AMICA(backend="mlx")` (or `AMICAICA(backend="mlx")` for MNE users);
it computes in float32.
See [Selecting a backend](guides/backends.md#selecting-a-backend).

## Quickstart

The main entry point is the scikit-learn-style [`AMICA`](api/amica.md) class,
which wraps the natural-gradient EM backend.

```python
import numpy as np
from pamica import AMICA

# X is (n_channels, n_samples); use real EEG/EMG rather than random data
# for a meaningful decomposition.
X = np.random.randn(32, 10000)

amica = AMICA(n_models=1, n_mix=3)
amica.fit(X, max_iter=100)

# Unmixed sources and the mixing/unmixing matrices
S = amica.transform(X)
A = amica.get_mixing_matrix(0)     # mixing matrix for model 0
W = amica.get_unmixing_matrix(0)   # unmixing matrix for model 0

print("final log-likelihood:", amica.final_ll_)
```

!!! warning "Use `final_ll_`, not `ll_history_[-1]`"
    With the best-iterate safeguard the returned parameters can be an earlier,
    higher-likelihood iterate, so `final_ll_` is the log-likelihood of the
    *returned* model. `ll_history_` is the true per-iteration trajectory and may
    dip below its peak on a late overshoot.

## Choosing a device and precision

`AMICA` auto-selects a device. Because the backend computes in float64 for
Fortran parity and Apple MPS cannot represent float64, an auto-selected MPS
device is redirected to CPU; pass `device="mps"` with `dtype=torch.float32` to
run on MPS explicitly. See [Backends & Devices](guides/backends.md) for the full
matrix and performance guidance.

## Next steps

- [Backends & Devices](guides/backends.md) — CUDA / CPU / MLX and float32 vs float64.
- [Validation & Parity](guides/validation.md) — comparing against the Fortran reference.
- [API Reference](api/index.md) — full parameter and method documentation.
