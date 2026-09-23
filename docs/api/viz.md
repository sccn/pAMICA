# Visualization (pamica.viz)

Top-level, backend-agnostic plots for a fitted model's output
(`pamica.numpy_impl.load.AmicaOutput`, as returned by
`pamica.numpy_impl.load.loadmodout`). Unlike the legacy
`pamica.numpy_impl.viz` module, these functions **return a `Figure`** and
accept an optional `ax`/`axes` to draw on, instead of returning `None` and
mutating pyplot's global state.

- **`plot_pmi_heatmap`**: a components-by-components pairwise mutual-information
  heatmap (see `pamica.metrics.pairwise_mi`), reordered to cluster related
  components near the diagonal.
- **`plot_model_probability`**: for a multi-model fit, two stacked panels: each
  model's posterior probability over time, and the log-likelihood of the most
  probable model at each timepoint.
A per-component scalp-topography plot is not included: the one upstream
reference it could be checked against (EEGLAB's `pop_topohistplot`) is broken
on current EEGLAB, so it was cut rather than shipped unverified. For scalp maps,
use the MNE wrapper's [`plot_components`](mne-compat.md).

```python
from pamica import plot_pmi_heatmap, plot_model_probability
```

::: pamica.viz.plot_pmi_heatmap

::: pamica.viz.plot_model_probability
