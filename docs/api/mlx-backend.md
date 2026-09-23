# MLX backend (AMICAMLXNG)

The optional Apple-Silicon GPU backend. It runs the natural-gradient EM E/M-step
on the Apple GPU in float32 (Apple GPUs have no float64), with the small
per-iteration linear algebra on MLX's CPU stream. It supports single- and
multi-model natural-gradient AMICA across all five source-density families
(`pdftype` 0-4, including the extended-Infomax adaptive switcher) and is the
fastest option on Apple hardware; see [Backends & Devices](../guides/backends.md)
for the performance comparison.

Most users reach this backend through the wrappers rather than this class:
`AMICA(backend="mlx")` and the MNE wrapper `AMICAICA(backend="mlx")` (epic #324 Phase 4, issue #313)
add the params-file reader, the degenerate-fit contract, `.pt` `save`/`load` and the MNE export on top of it.
See [Selecting a backend](../guides/backends.md#selecting-a-backend).

```python
from pamica import AMICA

model = AMICA(backend="mlx").fit(X, seed=42)  # builds AMICAMLXNG
```

Source extraction (`transform` and the `get_mixing_matrix`/`get_unmixing_matrix`/
`get_sensor_mixing_matrix`/`get_rho` accessors) and persistence
(`state_dict`/`from_state_dict` and `.npz` `save`/`load`) are implemented (epic
#278 Phase 1, issue #287); the `.npz` format is device- and framework-agnostic
(JSON-encoded config/extra plus native param arrays, no torch coupling).
Since issue #334 the format is version 2, which stores the mixing matrix `A` with one component per row;
a version 1 save is converted on load, unless `share_comps` had merged components, which raises `ValueError`. The
best-iterate safeguard (`keep_best`) is implemented (epic #278 Phase 2, issue
#288). Outlier rejection (`do_reject`), the LLt-stash-backed scoring accessors
(`model_loglik`/`model_probability`), the EEGLAB `write_amica_output` export,
and the Mutual Information Reduction (MIR) and Pairwise Mutual Information (PMI) diagnostics (`mir`/`pmi`, `fit(mir_step=...)` waypoints) are
implemented (epic #278 Phase 3, issue #289). The `variance_order` accessor
(the EEGLAB back-projected-variance component order) landed in the epic's
post-Phase-3 polish round, ahead of merge to `dev`, completing epic #278.

Explicit principal component analysis (PCA) reduction followed in epic #324 Phase 1 (issue #323).
`pcakeep` and `pcadb` carry the PyTorch backend's names, defaults, validation and precedence,
all from the shared `pamica.rank` policy.
Both default to `None`, which leaves only automatic `mineig`/`mineig_rel` rank detection.
`pcakeep` must be an integer of at least 1 and `pcadb` a finite number greater than 0,
or the constructor raises `ValueError`.
When both are set, `pcakeep` takes precedence and `pcadb` is ignored,
as in the reference, which parses `pcadb` but never uses it;
with `do_sphere=False` both are ignored, with one warning.
A reduced fit has a non-square `(n_channels, n_channels_in)` sphere,
and both parameters persist through `state_dict`/`save`.
`fit(mir_step > 0)` rejects an explicit reduction request up front, exactly as the PyTorch backend does
(see [the differences guide](../guides/amica-differences.md#explicit-dimensionality-reduction-pcakeep-and-pcadb-issue-323)).
With it there are no remaining gaps against the PyTorch backend other than float32-only precision
(Apple GPUs have no float64).
`get_sphere()`, `get_mean()` and `get_model_center()` (issue #313) return the fitted preprocessing as float64 arrays,
with the PyTorch backend's names and shapes;
the mean and centers are this backend's float32 values, and the sphere is its float64 host copy.

```python
model = AMICAMLXNG(n_channels=X.shape[0], pcakeep=X.shape[0] - 1)  # e.g. average reference
model.fit(X)
model.get_sensor_mixing_matrix()  # (n_channels_in, pcakeep) scalp maps
```

MLX is an optional dependency (Apple Silicon only), so the class is imported separately
and is not part of the default `import pamica` surface
(`AMICA(backend="mlx")` imports it on first use):

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
