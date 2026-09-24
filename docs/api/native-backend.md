# Native backend (AMICANative)

`AMICANative` runs the **real AMICA Fortran reference binary** as a fourth
backend, alongside the [PyTorch](torch-backend.md), [MLX](mlx-backend.md) and
[NumPy](numpy-backend.md) backends. Because it executes the literal reference
implementation, it is the strongest possible parity oracle: rather than checking
pamica *against* Fortran, you can run Fortran *as* a pamica backend, with no
separate toolchain to build or drive.

The reference is built dependency-free (a single-rank MPI shim removes the Open
MPI runtime, on top of the no-MKL recipe) and released as a self-contained binary
per platform. The engine resolves the binary for your host, downloads it from the
GitHub release on first use (verifying its SHA-256), caches it, and runs it.

```python
import numpy as np
from pamica import AMICANative

# X is (n_channels, n_samples) of real EEG/EMG.
model = AMICANative(n_models=1, n_mix=3).fit(X)

out = model.output_             # an AmicaOutput (W, A, mixing/unmixing, LL, ...)
sources = model.transform(X)    # source activations, EEGLAB variance order
```

`fit` writes the data and a full `input.param` (so it does not depend on an
installed `sample_data`), runs the binary, and loads the result with
`loadmodout`, exposed as `model.output_` (a
`pamica.numpy_impl.load.AmicaOutput`). `n_models` and `n_mix` are friendly aliases;
any Fortran `input.param` field (`max_iter`, `lrate`, `pdftype`, `do_newton`,
...) can be passed as a keyword. A collapsed fit (non-finite weights) is raised
as a clear degenerate-fit error rather than an opaque SVD failure.

The `input.param` it writes carries pamica's shared defaults (issue #354),
the values the PyTorch, MLX and NumPy backends default to, under the binary's keywords:
`lrate` 0.1, Newton off, `max_iter` 100, `do_opt_block` off, `pdftype` 0, and so on
([Default settings](../guides/amica-differences.md#default-settings-issue-354) has the full table).
The binary's `block_size` counts one thread's share of a block, and it processes no block at all,
writing an all-NaN fit, when `max_threads * block_size` exceeds the samples (issue #292).
Unless you pass `block_size`, the engine writes pamica's 8192-sample block, capped at the data's length, divided by `max_threads` (10 unless given),
so a default run keeps pamica's block and never meets that case; a `block_size` you pass is written as is.
pamica settings the binary has no keyword for, such as `keep_best` and `mineig_rel`, are not written,
so the binary returns its last iterate and applies the absolute `mineig` floor.
Keys only the binary reads (`max_threads`, `byte_size`, `writestep`, the `load_*` and `update_*` switches) keep the bundled `input.param`'s values.
Before issue #354 the engine wrote the bundled `pamica/sample_data/input.param`'s settings instead
(`lrate` 0.05, Newton on from iteration 50, `max_iter` 2000, `block_size` 512).
To run as that file or an EEGLAB-written one configures the binary, pass the file's settings as keywords
([Reproducing an EEGLAB run](../guides/amica-differences.md#reproducing-an-eeglab-run) shows how).

## Binary resolution and caching

On the first run for a given host, the engine downloads the matching release
asset and caches it, so later runs are offline:

- **Cache location:** `~/.cache/pamica/bin/<version>/` (or
  `$XDG_CACHE_HOME/pamica/bin/`; override with `PAMICA_NATIVE_CACHE`).
- **Integrity:** the download is staged in a temporary directory, verified
  against its `.sha256` release asset, marked executable, and only then moved
  atomically into the cache. A file therefore exists at the cache path only once
  it has passed its checksum; a failed or tampered download never runs.
- **Platforms:** prebuilt binaries are attached to each release for macOS arm64,
  Linux x64, Linux arm64 and Windows x64. Windows arm64 has no native Fortran
  toolchain yet ([#173](https://github.com/sccn/pAMICA/issues/173)); it maps to
  the x64 binary, which runs under Windows 11 ARM's x64 emulation.

Install the binary explicitly (for example to pre-populate the cache in an
offline or CI environment):

```bash
python -m pamica.native                 # download + cache the latest release binary
python -m pamica.native --version v0.2.1 # a specific release
python -m pamica.native --print         # print where it resolves to; do not download
```

## Using a local build

Set `PAMICA_NATIVE_BINARY` to a locally built binary to bypass the resolver
entirely (this is what the tests use, and the fallback on a platform with no
prebuilt asset):

```bash
export PAMICA_NATIVE_BINARY=/path/to/amica15
```

Build one with `native/build.sh` (gfortran + LAPACK; the single-rank MPI shim
means no MPI runtime is required). If no prebuilt binary matches your host and
`PAMICA_NATIVE_BINARY` is unset, the resolver raises a clear, actionable error
pointing at the build script.

## Validation harness

`validate_implementations.py` can source the reference through this engine
instead of the bundled macOS-only `amica15mac` fixture, so the real Fortran
reference can be compared against pamica's backends on any platform
(`--backend torch`, the default, or `numpy`, `mlx`, a comma-separated list, or `all`):

```bash
# Resolve/download the native binary (or honor PAMICA_NATIVE_BINARY):
uv run python validate_implementations.py --native-engine

# Or point at a specific binary, and compare every backend against it:
uv run python validate_implementations.py --fortran-binary /path/to/amica15 --backend all
```

::: pamica.native.engine.AMICANative
