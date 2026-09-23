# AMICA

The main scikit-learn-style interface.
It wraps a natural-gradient expectation-maximization (EM) backend:
[`AMICATorchNG`](torch-backend.md) by default (float64, Fortran parity),
or the Apple-GPU [`AMICAMLXNG`](mlx-backend.md) with `backend="mlx"` (float32, issue #313).
Every method works the same on both;
see [Selecting a backend](../guides/backends.md#selecting-a-backend) for the rules and the precision of each.

```python
from pamica import AMICA

model = AMICA(backend="mlx").fit(X, pcakeep=X.shape[0] - 1, seed=42)
model.save("model.pt")          # records the backend
model = AMICA.load("model.pt")  # restored on the MLX backend
```

`device` and a `dtype` fit keyword apply to the PyTorch backend only and raise `ValueError` with `backend="mlx"`;
`backend="mlx"` without MLX installed raises `ImportError`.

`get_sphere()`, `get_mean()` and `get_model_center(model_idx)` return the fitted preprocessing as float64 arrays on either backend,
so the transform can be composed by hand:
`transform(X)` is `W @ (sphere @ (X - mean[:, None]) - c[:, None])` with `W = get_unmixing_matrix()`.
`get_sensor_mixing_matrix()` gives the scalp maps in input-channel space, which is the only valid back-map after rank reduction.

`save` writes `format_version` 2, which records the backend;
`load` restores the model on that backend and still reads version 1 files, which predate backend selection and always hold a PyTorch model.

::: pamica.AMICA
