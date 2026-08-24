# EEGLAB interoperability

pamica is a drop-in replacement for EEGLAB's AMICA: a fit written to disk loads
directly with the same reader EEGLAB uses (`loadmodout15.m`), with the components
in the same order and orientation, so no manual re-sorting, sign-flipping, or
reformatting is needed.

## Writing EEGLAB-readable output

After a fit, call `write_amica_output` with a destination directory:

```python
from pamica import AMICA

model = AMICA(n_models=1, n_mix=3)
model.fit(X)                      # X is (n_channels, n_samples)
model.write_amica_output("amicaout")
```

This writes the raw binary files EEGLAB's AMICA loader reads:

| File | Contents |
| --- | --- |
| `gm` | model probabilities |
| `W` | unmixing weights (post-sphering) |
| `S` | sphering matrix |
| `mean` | data mean |
| `c` | per-model centers |
| `alpha`, `mu`, `sbeta`, `rho` | source mixture-density parameters |
| `comp_list` | component ids (for component sharing) |
| `LL` | log-likelihood per iteration |
| `LLt` | per-timepoint, per-model log-likelihood, plus the per-timepoint total |

For a single model the bytes are identical to the reference Fortran binary's
`amicaout` files, so the directory is interchangeable with a native AMICA run.

`LLt` is what `loadmodout15.m` turns into `Lht`/`Lt` and the model-probability
odds `v`; it is written after a fresh `fit()`, and omitted (with a warning) for
a model restored from `load()`, which carries no E-step stash. Under
`do_reject`, rejected samples are written as exactly `0.0` to match the
reference: AMICA's own `load_rej` reconstructs the rejection mask from those
zeros, so they are load-bearing rather than padding.

Like the reference, pamica writes the E-step that produced the last entry of
`LL`, so `LLt` describes the parameters as they stood one M-step *before* the
`W` written beside it (issue #157; see
[amica-differences](amica-differences.md#the-written-llt-is-one-m-step-older-than-the-w-beside-it-issue-157)).
Use `model.model_loglik(X)` if you want the log-likelihood of the written
parameters themselves.

## Loading in EEGLAB / MATLAB

In MATLAB with the AMICA plugin on the path:

```matlab
mod = loadmodout15('amicaout');
% mod.W   : unmixing weights (n x n x num_models)
% mod.A   : component scalp maps, columns ordered IC1..ICn by variance
% mod.S   : sphering matrix
% mod.svar: back-projected variance per component
```

`loadmodout15` applies the EEGLAB conventions on load: it orders components by
back-projected variance (IC1 has the highest), derives the sensor-space mixing
`A = pinv(W * S)`, and normalizes each map to unit norm. Because pamica writes
the same format, the components you get in EEGLAB match a native AMICA run.

## Variance ordering in Python

To get the EEGLAB display order without a disk round-trip, use `variance_order`,
which ranks sources by the same back-projected variance (IC1 = highest):

```python
order = model.variance_order()           # source indices, highest variance first
A = model.get_mixing_matrix()[:, order]  # scalp maps in EEGLAB order
W = model.get_unmixing_matrix()[order]   # unmixing rows in EEGLAB order
```

Pass `return_svar=True` to also get the per-component variances.

## Multi-model note

Single-model output files are byte-identical in layout to a native AMICA run. For
`n_models > 1` the per-model axis layout is self-consistent (it round-trips
through `loadmodout15` and pamica's own reader) but is not byte-identical to a
native multi-model AMICA run; see the multi-model equivalence discussion in
[Validation & Parity](validation.md).
