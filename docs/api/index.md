# API Reference

The public import surface is stable:

```python
from pamica import AMICA, AMICA_NumPy, AMICANative, AMICATorchNG
```

- **[`AMICA`](amica.md)**: the main scikit-learn-style interface.
  Wraps a natural-gradient EM backend: PyTorch by default, or MLX with `backend="mlx"`.
  Start here.
- **[`AMICATorchNG`](torch-backend.md)**: the PyTorch natural-gradient EM
  backend (Fortran parity). The `AMICA` interface delegates to this class by default.
- **[`pamica.metrics`](metrics.md)**: separation-quality metrics (`mir`,
  `pairwise_mi`, `block_diagonal_order`) as free functions over plain arrays.
  Also reachable as `AMICA.mir`/`AMICA.pmi` on a fitted model.
- **[`pamica.viz`](viz.md)**: backend-agnostic plots over a written
  `amicaout` directory.
- **[`AMICA_NumPy`](numpy-backend.md)**: the legacy NumPy reference
  implementation, retained as an oracle and for its command-line interface.
- **[`AMICANative`](native-backend.md)**: the Fortran reference binary itself,
  run on your data as a fourth backend.

The optional Apple-Silicon GPU backend is imported separately and is not part of
the default import surface:

```python
from pamica.mlx_impl import AMICAMLXNG  # requires the `mlx` extra
```

- **[`AMICAMLXNG`](mlx-backend.md)**: the optional Apple-GPU (MLX) backend; the
  fastest option on Apple Silicon (float32).
  `AMICA(backend="mlx")` builds it without this import.

The optional MNE-Python wrapper is likewise imported explicitly:

```python
from pamica.mne_compat import AMICAICA  # requires the `mne` extra
```

- **[`AMICAICA`](mne-compat.md)**: fit AMICA from an MNE `Raw`/`Epochs` and
  interoperate with `mne.preprocessing.ICA` (`get_sources`, `apply`,
  `plot_components`, `to_mne_ica`), on either backend (`backend="mlx"`).
