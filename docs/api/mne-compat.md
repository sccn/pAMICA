# MNE-Python compatibility (AMICAICA)

`AMICAICA` fits AMICA directly from an [MNE-Python](https://mne.tools)
`Raw`/`Epochs` and hands the result back through the standard MNE ICA surface.
It is **additive**: the scikit-learn-style [`AMICA`](amica.md) interface and the
[EEGLAB output](../guides/eeglab.md) are unchanged; this is a
second entry point for MNE users, not a replacement.

MNE is an optional dependency, so `import pamica` never requires it. Install the
extra and import the wrapper explicitly:

```bash
pip install pamica[mne]
```

```python
import mne
from pamica.mne_compat import AMICAICA

raw = mne.io.read_raw_eeglab("subject.set", preload=True)

ica = AMICAICA(n_mix=3, random_state=42).fit(raw, picks="eeg", max_iter=100)

sources = ica.get_sources(raw)     # an mne.io.RawArray of component activations
maps = ica.get_components()        # scalp maps, shape (n_channels, n_components)
ica.plot_components()              # native mne.viz topographies

# Reconstruct with some components removed:
clean = ica.apply(raw.copy(), exclude=[0, 3])
```

`fit` accepts a `Raw` or `Epochs` (epochs are concatenated along time, as MNE's
own ICA does), any MNE `picks` selector, and forwards remaining keywords
(`max_iter`, `lrate`, `do_newton`, ...) to [`AMICA.fit`](amica.md). It rejects
non-finite input and PCA reduction (`pcakeep`/`pcadb`, which leaves the sphere
rank-deficient so the full-rank export would be invalid), and a degenerate
(diverged) fit is refused by the consumer methods rather than emitting NaNs.

## Interoperating with `mne.preprocessing.ICA`

`to_mne_ica()` returns a fully-populated
[`mne.preprocessing.ICA`](https://mne.tools/stable/generated/mne.preprocessing.ICA.html),
so the entire MNE ICA ecosystem (plotting, `find_bads_eog`/`_ecg`, exclusion
workflows) works on an AMICA decomposition:

```python
mne_ica = ica.to_mne_ica()
eog_idx, scores = mne_ica.find_bads_eog(raw)
mne_ica.plot_scores(scores)
```

The wrapper's `get_sources`, `apply`, `get_components`, `plot_components` and
`plot_sources` delegate to this object, so they reproduce `AMICA.transform`
exactly: MNE
computes sources as `unmixing_matrix_ @ pca_components_ @ (X - pca_mean_)`, and
the export maps pamica's mean, symmetric-ZCA sphere and unmixing into those
matrices (writing the sphere as `V diag(1/√e) Vᵀ` with `V` orthonormal so MNE's
scalp maps come out in channel space). The equivalence
`to_mne_ica().get_sources(raw) == AMICA.transform(X)` is pinned by the test
suite on real sample EEG.

## Multi-model fits

AMICA can learn a mixture of ICA models (`n_models > 1`). MNE's `ICA` represents
only one unmixing matrix, so each model is exported as its own single-model
`mne.preprocessing.ICA`, and the per-sample *model dominance* (which model best
explains each timepoint) is exposed directly, since MNE has no concept for it:

```python
ica = AMICAICA(n_models=2, random_state=42).fit(raw, max_iter=100)

# Per-model: model_idx selects the model on every consumer method.
sources_m1 = ica.get_sources(raw, model_idx=1)
ica.plot_components(model_idx=1)
model1 = ica.to_mne_ica(model_idx=1)   # a standard ICA for model 1

# Model dominance over time (P(model | sample), columns sum to 1):
prob = ica.get_model_probability(raw)  # (n_models, n_samples)
ica.plot_model_probability(raw)        # per-model probability + best-model LL
```

`get_model_probability`/`plot_model_probability` build on the public
`AMICA.model_loglik`/`model_probability` accessors, which score arbitrary data
through the fitted sphere and mean. Each per-model export folds that model's
data-space center into `pca_mean_`, so `to_mne_ica(model_idx=h).get_sources(raw)`
reproduces `AMICA.transform(X, model_idx=h)` (with `X` the picked channel array)
for every model, not just the first.

## Inspecting pamica-specific metadata

An `mne.preprocessing.ICA` has no field for AMICA's adaptive source densities or
component sharing, so rather than drop them, the wrapper exposes them directly:

```python
from pamica.mne_compat import AMICAICA, PDFTYPE_NAMES

ica = AMICAICA(n_models=2, random_state=42).fit(raw, max_iter=100)

families = ica.get_pdftype(model_idx=0)        # (n_components,) codes 0-4
names = [PDFTYPE_NAMES[c] for c in families]    # e.g. "generalized_gaussian"
rho = ica.get_rho(model_idx=0)                  # (n_mix, n_components) GG shape
shared = ica.shared_components()                # [(model, comp), ...] groups
```

`get_pdftype` returns each component's density family (0 generalized Gaussian,
1 super-Gaussian cosh, 2 Gaussian, 3 logistic, 4 sub-Gaussian cosh; they differ
per component only under the adaptive switcher `pdftype=1`). `get_rho` is the
generalized-Gaussian shape (meaningful for `pdftype=0`). `shared_components`
lists components merged across models by `share_comps` (empty otherwise). The
same three accessors exist on the scikit-learn-style [`AMICA`](amica.md).

## Separation-quality metrics

The [MIR/PMI metrics](metrics.md) are available directly on an MNE object, so
MNE-side users get the same separation-quality numbers as EEGLAB-side users:

```python
mir_nats, variance = ica.mir(raw, model_idx=0)   # mutual information reduced
mi_matrix = ica.pmi(raw, model_idx=0)            # pairwise MI between sources
```

Both extract the fitted channels from the `Raw`/`Epochs` and delegate to
`AMICA.mir`/`pmi`, reproducing the array API exactly.

::: pamica.mne_compat.AMICAICA
