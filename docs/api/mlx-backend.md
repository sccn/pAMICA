# MLX backend (AMICAMLXNG)

The optional Apple-Silicon GPU backend. It runs the natural-gradient EM E/M-step
on the Apple GPU in float32 (Apple GPUs have no float64), with the small
per-iteration linear algebra on MLX's CPU stream. It supports single- and
multi-model natural-gradient AMICA across all five source-density families
(`pdftype` 0-4, including the extended-Infomax adaptive switcher) and is the
fastest option on Apple hardware; see [Backends & Devices](../guides/backends.md)
for the performance comparison.

Source extraction (`transform` and the `get_mixing_matrix`/`get_unmixing_matrix`/
`get_sensor_mixing_matrix`/`get_rho` accessors) and persistence
(`state_dict`/`from_state_dict` and `.npz` `save`/`load`) are implemented (epic
#278 Phase 1, issue #287); the `.npz` format is device- and framework-agnostic
(JSON-encoded config/extra plus native param arrays, no torch coupling). The
remaining gaps are `keep_best` (epic #278 Phase 2) and outlier rejection
(`do_reject`) + LLt/MIR (epic #278 Phase 3).

MLX is an optional dependency (Apple Silicon only), so it is imported separately
and is not part of the default `import pamica` surface:

```python
from pamica.mlx_impl import AMICAMLXNG  # requires the `mlx` extra
```

Install it with `uv pip install mlx` or the `mlx` extra (`pip install pamica[mlx]`).
Because it computes in float32, use the [PyTorch backend](torch-backend.md) on
CUDA/CPU for float64 Fortran-parity runs.

The module also exports `PDFTYPE_NAMES`, the mapping from the density-family
codes `get_pdftype()` returns (0-4) to their human-readable names (generalized
Gaussian, super-Gaussian cosh, Gaussian, logistic, sub-Gaussian cosh); it is
the same mapping `pamica.mne_compat` exposes.

::: pamica.mlx_impl.AMICAMLXNG
