# Backends & Devices

pamica ships one primary PyTorch backend behind the [`AMICA`](../api/amica.md)
interface, plus an optional Apple-GPU backend and a legacy NumPy reference.

## Backends

| Backend | Class | Role |
|---|---|---|
| PyTorch natural-gradient EM | [`AMICATorchNG`](../api/torch-backend.md) | **Default.** Fortran-parity backend; CUDA / CPU, and float32 on MPS. |
| MLX (Apple GPU) | [`AMICAMLXNG`](../api/mlx-backend.md) (`pamica.mlx_impl`) | Optional Apple-Silicon GPU backend; float32 only. |
| NumPy reference | [`AMICA_NumPy`](../api/numpy-backend.md) | Legacy oracle + CLI; carries the same parity fixes. |

The `AMICA` wrapper uses `AMICATorchNG`. The MLX backend is imported separately
(`from pamica.mlx_impl import AMICAMLXNG`) so that `import pamica` never
requires MLX.

## Device selection

`AMICA(device=...)` accepts `"cuda"`, `"cpu"`, `"mps"`, or `None` (auto):

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
  current 8192 default, and to 13.5 ms/iteration at a further-tuned single-block
  size, below the CPU's 15.8 ms there. MLX stays fastest throughout, so it
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
