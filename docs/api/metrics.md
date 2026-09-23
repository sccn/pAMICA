# Separation metrics (pamica.metrics)

Quality metrics for a decomposition, as free functions over plain arrays. They
are backend-agnostic and independent of any fitted model object, so they work on
sources from any source.

- **`mir`**: Mutual Information Reduction, in nats: how much mutual information
  a linear unmixing removes from the data. Higher is a better separation. It
  needs the **full raw-data-to-sources transform** (the unmixing composed with
  the sphering matrix), which must be square and invertible, since the estimate
  includes a log-Jacobian term.
- **`pairwise_mi`**: the symmetric mutual-information matrix between sources.
  Its diagonal is each source's own entropy, not a mutual information.
- **`block_diagonal_order`**: a permutation that clusters mutually dependent
  components near the diagonal, for reading structure out of a `pairwise_mi`
  matrix.

Most callers should prefer the model accessors, which compose the transform for
you and refuse an unusable fit: [`AMICA.mir`](amica.md) and
[`AMICA.pmi`](amica.md). Use these free functions when you have sources or an
unmixing from somewhere else. To plot a `pairwise_mi` matrix, see
[`plot_pmi_heatmap`](viz.md), which applies `block_diagonal_order` itself and
masks the entropy diagonal.

```python
from pamica.metrics import mir, pairwise_mi, block_diagonal_order
```

## Provenance

`mir` is a direct port of `getMIR.m` from
[bigdelys/pre_ICA_cleaning](https://github.com/bigdelys/pre_ICA_cleaning)
(Apache-2.0); the license is vendored in `THIRD_PARTY_NOTICES.md`. It agrees
with the original to 1.7e-15 relative on the bundled sample data.

`pairwise_mi` and `block_diagonal_order` are a clean-room reimplementation. The
comparable MATLAB code (`minfojp.m` in postAmicaUtility) is GPL-2.0-or-later and
pamica is BSD-3-Clause, so that source was never read; the implementation works
from the published description in Delorme et al. (2012), "Independent EEG
sources are dipolar", PLoS ONE. It agrees with that reference at r=0.9887 on
identical signals.

::: pamica.metrics.mir

::: pamica.metrics.pairwise_mi

::: pamica.metrics.block_diagonal_order
