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
`loadmodout`, exposed as `model.output_` (an
[`AmicaOutput`](numpy-backend.md)). `n_models` and `n_mix` are friendly aliases;
any Fortran `input.param` field (`max_iter`, `lrate`, `pdftype`, `do_newton`,
...) can be passed as a keyword. A collapsed fit (non-finite weights) is raised
as a clear degenerate-fit error rather than an opaque SVD failure.

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
python validate_implementations.py --native-engine

# Or point at a specific binary, and compare every backend against it:
python validate_implementations.py --fortran-binary /path/to/amica15 --backend all
```

::: pamica.native.engine.AMICANative
