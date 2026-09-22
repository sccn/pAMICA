# Backends & Devices

pamica ships one primary PyTorch backend behind the [`AMICA`](../api/amica.md)
interface, plus an optional Apple-GPU backend and a legacy NumPy reference.

## Backends

| Backend | Class | Role |
|---|---|---|
| PyTorch natural-gradient EM | [`AMICATorchNG`](../api/torch-backend.md) | **Default.** Fortran-parity backend; CUDA / CPU, and float32 on MPS. |
| MLX (Apple GPU) | [`AMICAMLXNG`](../api/mlx-backend.md) (`pamica.mlx_impl`) | Optional Apple-Silicon GPU backend; float32 only. |
| NumPy reference | [`AMICA_NumPy`](../api/numpy-backend.md) | Legacy oracle + CLI; carries the same parity fixes. |

The `AMICA` wrapper and the MNE wrapper `AMICAICA` build `AMICATorchNG` by default and `AMICAMLXNG` with `backend="mlx"`
(see [Selecting a backend](#selecting-a-backend)).
As of epic #278 the MLX backend carries the full `AMICATorchNG`-equivalent surface
(all pdf families, Newton, rejection, persistence, EEGLAB export, MIR/PMI),
and since epic #324 every wrapper convenience runs on it too.
See the [backend table](amica-differences.md#backend-differences).

## Selecting a backend

Pass `backend="mlx"` to `AMICA` or `AMICAICA` to fit on the Apple GPU (issue #313):

```python
from pamica import AMICA

model = AMICA()                  # AMICATorchNG: float64, Fortran parity (the default)
model = AMICA(backend="mlx")     # AMICAMLXNG: Apple GPU, float32
```

Everything the wrappers offer works on both backends:
`fit` with any backend keyword (`pcakeep`, `seed`, `n_restarts`, ...), `from_params_file`,
`transform` and the matrix accessors, the degenerate-fit contract (`converged_`, `is_fitted_`, `stop_reason_`),
`save`/`load`, the EEGLAB export and the MNE path.

- `backend="mlx"` needs MLX (`uv pip install mlx` or the `mlx` extra; Apple Silicon only).
  Without it the constructor raises `ImportError`, and `import pamica` never imports MLX either way.
- `device` and `dtype` are PyTorch-only.
  MLX always runs on its default device (the Apple GPU) in float32,
  so `AMICA(backend="mlx", device=...)` and `fit(..., dtype=...)` on an MLX model raise `ValueError`.
- The default stays `"torch"`, the float64 Fortran-parity path; there is no automatic selection.
- `AMICA.from_params_file(path, backend="mlx")` applies the file's settings through `AMICAMLXNG`'s own constructor,
  and names any setting that backend cannot take in its "not applied" warning.
- `save` records the backend (`format_version` 2), so `AMICA.load` restores the model on the backend that fit it.
  Files written before backend selection (version 1) still load, as PyTorch models.
  Loading an MLX model needs MLX installed and takes no `device`.

The raw classes stay available for direct use (`from pamica.mlx_impl import AMICAMLXNG`),
but the wrappers are the supported route: they add the degenerate-fit contract, the params-file reader and the MNE export.

### Precision on the MLX backend

MLX computes in float32 (about 7 significant digits), so an MLX fit is float32-consistent rather than float64-parity.
On the bundled 32-channel sample, the same data, seed and 10 iterations through both wrappers give the same components in the same order
(Hungarian-matched source correlation above 0.9999999, with and without `pcakeep`)
and log-likelihoods within 5e-6 of each other.

`get_sphere()`, `get_mean()` and `get_model_center()` return float64 arrays on both backends.
An MLX fit's mean and per-model centers are float32 values;
its sphere is the float64 matrix it was built from, which its float32 GPU computation agrees with to float32 rounding.
After `AMICA.load`, an MLX model's sphere is the saved float32 sphere, upcast.

In the MNE path, `AMICAICA(backend="mlx")` exports through the same float64 accessors.
`get_sources` agrees with the MLX `transform` to float32 tolerance (measured 2e-7 relative),
and excluding a component removes its back-projection to the same tolerance.
Reconstruction does not inherit the float32 rounding:
`apply` with nothing excluded returns the input to float64 round-off (measured 1.4e-15), the principal component analysis (PCA) residual of a `pcakeep` fit included,
because MNE's mixing is the float64 pseudo-inverse of the exported unmixing and the PCA basis is orthonormal.

### End to end on the Apple GPU

A per-session workflow for average-referenced electroencephalography (EEG), whose rank is `n_channels - 1`:

```python
import mne
from pamica import AMICA
from pamica.mne_compat import AMICAICA

# X is (n_channels, n_samples)
model = AMICA(backend="mlx").fit(X, pcakeep=X.shape[0] - 1, seed=42)
sources = model.transform(X)                # float32, (n_channels - 1, n_samples)
maps = model.get_sensor_mixing_matrix()     # scalp maps, (n_channels, n_channels - 1)
model.write_amica_output("amicaout")        # EEGLAB: loadmodout15('amicaout')

model.save("session01.pt")                  # records backend="mlx"
model = AMICA.load("session01.pt")          # back on the MLX backend

# The same settings from a Fortran input.param or a JSON params file:
model = AMICA.from_params_file("input.param", backend="mlx").fit(X)

# MNE: fit from a Raw, then remove a component (the PCA residual is kept).
raw = mne.io.read_raw_eeglab("session01.set", preload=True)
n_eeg = len(mne.pick_types(raw.info, eeg=True))  # good EEG channels
ica = AMICAICA(backend="mlx", random_state=42)
ica.fit(raw, picks="eeg", pcakeep=n_eeg - 1)
clean = ica.apply(raw.copy(), exclude=[0])
```

`write_amica_output` runs before `save` here because a model restored by `load` carries no per-sample log-likelihood,
so its export omits the `LLt` file (with a warning).

## Device selection

With the PyTorch backend, `AMICA(device=...)` accepts `"cuda"`, `"cpu"`, `"mps"`, or `None` (auto):

- **`None` (auto)** — selects CUDA if available, else CPU. An auto-selected MPS
  device is redirected to CPU because the parity default is float64, which MPS
  cannot represent.
- **`"cuda"`** — the bit-safe path for float64 Fortran parity on NVIDIA GPUs.
- **`"mps"`** — requires `dtype=torch.float32`. Note that PyTorch-MPS is not a
  performance win for AMICA (see below); prefer the MLX backend on Apple hardware.

## Precision: float64 vs float32

- **float64** — the default; required for Fortran-parity runs. CUDA float64
  agrees with the CPU log-likelihood to ~5 significant digits.
- **float32** — required on the Apple GPUs (MPS/MLX have no float64) and
  ~7-significant-digit, not float64-parity. It is not a general speedup: CUDA is
  overhead-bound so float32 is about as fast as float64, while on CPU float32 is
  modestly faster and scales better across cores. The Apple-GPU speed win comes
  from the MLX backend (see below), not float32 itself. Use it for exploratory or
  large-scale runs where exact reference parity is not required.

## Performance on real EEG

Measured on real 70-channel EEG (see the project benchmarks and
`.context/issue-77/`) at `block_size=512`:

- On **Apple Silicon**, the **MLX backend is the GPU win** (~15-25 ms/iteration,
  roughly flat from 16 to 70 channels), several times faster than torch-CPU and
  faster than an RTX 4090 at EEG scale. **PyTorch-MPS does not win at this block
  size** (162-255 ms/iteration, at or worse than CPU). A block-size sweep (issue
  #216, bundled sample) found PyTorch-MPS far more block-size-sensitive than CPU
  or MLX: it falls from 431 to 30.5 ms/iteration between `block_size=512` and the
  current 8192 default, still behind CPU's 21.7 ms/iteration there, and down to
  13.5 ms/iteration at a further-tuned single-block size that puts the whole
  30504-frame sample in one block -- memory-limited rather than a free win, since
  peak block memory scales with `block_size`, which is why 8192 stays the shipped
  default -- below the CPU's 15.8 ms there. MLX stays fastest throughout, so it
  remains the recommendation over `device="mps"` on Apple hardware. See
  [Block-size sensitivity](validation.md#block-size-sensitivity) for the full sweep.
- On **NVIDIA**, CUDA float64 is the bit-safe path (~4.5x over a 16-thread CPU,
  warmed); float32 is faster still.
- On **CPU**, intra-op threads are workload-limited; around 4 threads was the
  sweet spot in the measured laptop sweep, with 8+ regressing.

All backends agree on the log-likelihood to ~3 significant digits on real data.

!!! note "Cross-backend equivalence and data adequacy"
    Whether two backends recover the *same* independent components depends on how
    well-determined the decomposition is (the data-adequacy factor
    `k = frames / channels^2`). See [Validation & Parity](validation.md); the
    full data-size sweep is being finalized.
