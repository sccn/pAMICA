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

The MLX backend runs on the Apple-Silicon graphics processing unit (GPU).
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

`AMICA` auto-selects a device.
Because the backend computes in float64 for Fortran parity and Apple's Metal Performance Shaders (MPS) device cannot represent float64,
an auto-selected MPS device is redirected to the central processing unit (CPU);
pass `device="mps"` with `dtype=torch.float32` to run on MPS explicitly.
On Apple Silicon the MLX backend below is faster than MPS.
See [Backends & Devices](guides/backends.md) for the full matrix and performance guidance.

## Apple Silicon (MLX)

On a Mac with Apple Silicon, the MLX backend runs AMICA on the GPU
and is the fastest pamica backend there
(see [Performance on real EEG](guides/backends.md#performance-on-real-eeg)).
MLX is Apple's array framework;
pamica's MLX backend, `AMICAMLXNG`, carries the full feature set of the default PyTorch backend,
and both wrappers, `AMICA` and the MNE-Python wrapper `AMICAICA`, select it with `backend="mlx"`.

### Install

Install the MLX extra, plus the MNE extra if you work with MNE-Python objects:

```bash
uv sync --extra mlx --extra mne
uv run python -c "import mlx.core as mx; print(mx.default_device())"  # Device(gpu, 0)
```

The second line checks that MLX sees the GPU.
`import pamica` never imports MLX, so the same code runs on hosts without it;
there, `AMICA(backend="mlx")` raises `ImportError` with the install hint.

### Fit, transform, save and load

```python
from pamica import AMICA

# X is your own (n_channels, n_samples) recording.
model = AMICA(backend="mlx")
model.fit(X, max_iter=200, seed=42)
sources = model.transform(X)             # float32 sources, (n_sources, n_samples)
maps = model.get_sensor_mixing_matrix()  # scalp maps, (n_channels, n_sources)
print(model.final_ll_, model.stop_reason_)

model.save("model.pt")                   # the file records backend="mlx"
model = AMICA.load("model.pt")           # restored on the MLX backend
```

Every `fit` keyword the PyTorch backend takes works here too
(`pcakeep`, `block_size`, `do_newton`, `n_restarts`, ...),
except the PyTorch-only `device` and `dtype`, which raise `ValueError`.
A Fortran `input.param` or pamica JSON params file drives the MLX backend the same way it drives the default one:

```python
model = AMICA.from_params_file("input.param", backend="mlx").fit(X)
```

For EEGLAB, `model.write_amica_output("amicaout")` writes the files EEGLAB's `loadmodout15` reads.
Call it before `save`/`load`: a reloaded model carries no per-sample log-likelihood,
so its export omits the `LLt` file (with a warning).

### With MNE-Python

After an average reference the EEG rank is one less than the channel count,
so keep `n_channels - 1` principal components:

```python
import mne
from pamica.mne_compat import AMICAICA

raw = mne.io.read_raw_eeglab("/path/to/your_recording.set", preload=True)
raw.set_eeg_reference("average")
n_eeg = len(mne.pick_types(raw.info, eeg=True))

ica = AMICAICA(backend="mlx", random_state=42)
ica.fit(raw, picks="eeg", pcakeep=n_eeg - 1, max_iter=200)
sources = ica.get_sources(raw)               # an MNE Raw of the components
clean = ica.apply(raw.copy(), exclude=[0])   # remove component 0
ica.amica_.write_amica_output("amicaout")    # the same fit, for EEGLAB
```

`apply` removes only the excluded components' back-projection.
The principal component analysis (PCA) residual that `pcakeep` left out of the decomposition is restored, as MNE's own `ICA` does
(see [the differences page](guides/amica-differences.md#amicaicaapply-restores-the-pca-residual-issue-322)).

### Precision

MLX computes in float32 (about seven significant digits), because Apple GPUs have no float64.
An MLX fit agrees with the float64 PyTorch backend to float32 tolerance,
and on the bundled sample it matches the Fortran reference as closely as the float64 backends do
(see [the harness rows](guides/validation.md#parity-rows-per-backend)).
It is not bit-for-bit Fortran parity, though:
for a run you intend to compare against the reference binary, use the default `backend="torch"` (float64).
[Selecting a backend](guides/backends.md#selecting-a-backend) has the measured agreement and the details.

## Next steps

- [Backends & Devices](guides/backends.md) — CUDA / CPU / MLX and float32 vs float64.
- [Validation & Parity](guides/validation.md) — comparing against the Fortran reference.
- [API Reference](api/index.md) — full parameter and method documentation.
