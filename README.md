# pAMICA: Adaptive Mixture ICA

[![CI](https://github.com/sccn/pAMICA/actions/workflows/ci.yml/badge.svg)](https://github.com/sccn/pAMICA/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/sccn/pAMICA/branch/main/graph/badge.svg)](https://codecov.io/gh/sccn/pAMICA)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21312148.svg)](https://doi.org/10.5281/zenodo.21312148)
[![Docs](https://img.shields.io/badge/docs-eeglab.org%2FpAMICA-blue)](https://eeglab.org/pAMICA/)

Python (PyTorch) implementation of Adaptive Mixture Independent Component Analysis
(AMICA) that reproduces the reference Fortran implementation within numerical
tolerance, with CPU, NVIDIA GPU (CUDA), and Apple GPU (MLX) support. It targets
EEG/EMG blind source separation and is a drop-in replacement for EEGLAB's AMICA:
single-model output is byte-identical to the Fortran reference and loads directly
in EEGLAB.

Single-model results match the Fortran reference (Hungarian-matched component correlation ~ 0.998
on well-determined data, Newton disabled); see the
[documentation](https://eeglab.org/pAMICA/) for validation details and the
backend-selection guide.

## Overview

AMICA (Adaptive Mixture ICA) is an advanced blind source separation algorithm that uses adaptive mixtures of independent component analyzers. This implementation provides:

- Multiple source models
- Different PDF types
- Newton optimization
- Component sharing
- Outlier rejection
- Data preprocessing (mean removal, sphering)

## Installation

The canonical environment is [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/sccn/pAMICA.git
cd pAMICA
uv sync                     # install dependencies into a managed venv
uv run pytest               # optional: run the tests
```

The optional Apple-GPU backend (MLX, Apple Silicon only) installs with the `mlx`
extra: `uv pip install mlx`.

## Usage

pamica exposes a scikit-learn-style estimator backed by the PyTorch
natural-gradient EM implementation:

```python
import numpy as np
from pamica import AMICA

# X is (n_channels, n_samples) of real EEG/EMG; float64 gives Fortran parity
model = AMICA(n_models=1, n_mix=3).fit(X)

sources = model.transform(X)       # (n_sources, n_samples)
A = model.get_mixing_matrix()      # sensor-space scalp maps
order = model.variance_order()     # EEGLAB IC order (IC1 = highest variance)
```

### Backends and precision

The wrapper auto-selects a device and computes in float64 for Fortran parity.

- CPU and CUDA (float64) are bit-reproducible; use them for parity runs.
- float32 (about 7 significant digits, not parity) is required on the Apple GPUs
  and modestly faster on CPU; it is not a general speedup, since CUDA is
  overhead-bound (float32 is about as fast as float64).
- On Apple Silicon the MLX backend is the fastest option; import it explicitly.

```python
AMICA(device="cuda").fit(X)               # NVIDIA GPU, float64
from pamica.mlx_impl import AMICAMLXNG    # Apple GPU (install the mlx extra)
```

### EEGLAB interoperability

pamica writes results in EEGLAB's AMICA output format, so a run drops into an
EEGLAB workflow with no manual re-ordering or sign-flipping:

```python
model.write_amica_output("amicaout")   # gm, W, S, mean, c, alpha, mu, sbeta, rho, ...
```

```matlab
mod = loadmodout15('amicaout');   % components in EEGLAB variance order
```

### Legacy NumPy CLI

The NumPy reference backend keeps a JSON-driven command-line interface:

```bash
python -m pamica.numpy_impl.cli params.json --outdir results
```

See the [documentation](https://eeglab.org/pAMICA/) for the full API, the
parameter reference, and the backend-selection and validation guides.

## Citation

If you use pamica, please cite it (see [CITATION.cff](CITATION.cff)) and the
original AMICA method:

> Palmer, J. A., Kreutz-Delgado, K., & Makeig, S. (2012). AMICA: An adaptive
> mixture of independent component analyzers with shared components. Technical
> report, Swartz Center for Computational Neuroscience, UC San Diego.

## License

This project is licensed under the BSD 3-Clause License - see the [LICENSE](LICENSE) file for details.
