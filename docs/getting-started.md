# Getting Started

## Installation

pamica uses [UV](https://docs.astral.sh/uv/) for environment and dependency
management.

### From the Python Package Index (PyPI)

Released versions are on PyPI as `pamica`:

```bash
uv add pamica                # in a uv project
uv pip install pamica        # or into an existing environment
```

The optional extras install the same way, for example `uv add "pamica[mlx]"` or `uv add "pamica[mne]"`.
The changes listed under "Unreleased" in the [changelog](changelog.md) are not in a release yet;
install from source to use them.

### From source

```bash
git clone https://github.com/sccn/pAMICA.git
cd pAMICA
uv sync            # install the project and its dependencies into .venv
```

### Optional Apple Silicon backend (MLX)

The MLX backend runs on the Apple Silicon graphics processing unit (GPU).
It is Apple-only and is therefore an optional extra; `import pamica` never requires it.

```bash
uv sync --extra mlx   # or, in an existing environment: uv pip install mlx
```

Then select it with `AMICA(backend="mlx")` (or `AMICAICA(backend="mlx")` for MNE users);
it computes in float32.
[Apple Silicon (MLX)](#apple-silicon-mlx) below walks through the whole workflow.

## Quickstart

The main entry point is the scikit-learn-style [`AMICA`](api/amica.md) class,
which wraps the natural-gradient expectation-maximization (EM) backend.

```python
import numpy as np
from pamica import AMICA

# X is (n_channels, n_samples); use real electroencephalography (EEG) or
# electromyography (EMG) recordings rather than random data for a meaningful
# decomposition.
X = np.random.randn(32, 10000)

amica = AMICA(n_models=1, n_mix=3)
amica.fit(X, max_iter=100)

# Unmixed sources and the mixing/unmixing matrices
S = amica.transform(X)
A = amica.get_mixing_matrix(0)     # model 0's mixing matrix, sphered space
W = amica.get_unmixing_matrix(0)   # model 0's unmixing matrix, sphered space
maps = amica.get_sensor_mixing_matrix(0)  # scalp maps, input-channel space

print("final log-likelihood:", amica.final_ll_)
```

!!! warning "Use `final_ll_`, not `ll_history_[-1]`"
    With the best-iterate safeguard the returned parameters can be an earlier,
    higher-likelihood iterate, so `final_ll_` is the log-likelihood of the
    *returned* model. `ll_history_` is the true per-iteration trajectory and may
    dip below its peak on a late overshoot.
    As in the reference, a fit that stops on a convergence check returns the
    parameters `final_ll_` was computed from, while one that runs to `max_iter`
    takes one more update after its last likelihood.

## Choosing a device and precision

`AMICA` auto-selects a device.
Because the backend computes in float64 for Fortran parity and Apple's Metal Performance Shaders (MPS) device cannot represent float64,
an auto-selected MPS device is redirected to the central processing unit (CPU), with a logged warning;
the PyTorch backend does this itself, so the raw `AMICATorchNG` chooses the same way.
Pass `device="mps"` with `dtype=torch.float32` to run on MPS explicitly.
On Apple Silicon the MLX backend below is faster than MPS.
See [Backends & Devices](guides/backends.md) for the full matrix and performance guidance.

## Apple Silicon (MLX)

On a Mac with Apple Silicon, the MLX backend (MLX is Apple's array framework) runs AMICA on the GPU
and is the fastest pamica backend there.
It carries the full feature set of the default PyTorch backend,
and both wrappers select it with `backend="mlx"`:

```bash
uv sync --extra mlx --extra mne   # MLX, plus MNE-Python if you use AMICAICA
```

```python
from pamica import AMICA
from pamica.mne_compat import AMICAICA

model = AMICA(backend="mlx").fit(X, seed=42)   # X is (n_channels, n_samples)
sources = model.transform(X)

ica = AMICAICA(backend="mlx", random_state=42).fit(raw)   # raw is an MNE Raw
```

MLX computes in float32 (about seven significant digits), because Apple GPUs have no float64:
an MLX fit agrees with the float64 PyTorch backend to float32 tolerance, not bit for bit,
so use the default `backend="torch"` for a run you intend to compare against the Fortran reference.
The backends guide has the complete workflow,
from `pcakeep` on average-referenced EEG through save/load, a params-file run, the EEGLAB export and MNE's `apply`
([End to end on the Apple GPU](guides/backends.md#end-to-end-on-the-apple-gpu)),
and the measured precision ([Precision on the MLX backend](guides/backends.md#precision-on-the-mlx-backend)).

## Next steps

- [Backends & Devices](guides/backends.md): CUDA / CPU / MLX and float32 vs float64.
- [Validation & Parity](guides/validation.md): comparing against the Fortran reference.
- [pamica vs. AMICA](guides/amica-differences.md): every deliberate difference from the reference,
  including pamica's [default settings](guides/amica-differences.md#default-settings-issue-354) beside the compiled binary's and EEGLAB's.
- [API Reference](api/index.md): full parameter and method documentation.
