# pamica vs. AMICA: every deliberate difference

pamica reproduces Jason Palmer's Fortran AMICA numerically, and parity with that
reference is how correctness is defined here. This page lists every place pamica
**deliberately** behaves differently, why, and how to restore the reference behavior.

Anything not on this page is intended to match the reference. If you find a difference
that is not listed, that is a bug worth
[reporting](https://github.com/sccn/pAMICA/issues) — not a documented choice.

## At a glance

| # | Area | Fortran AMICA | pamica default | Why | Restore reference |
|---|---|---|---|---|---|
| 1 | Rank threshold | absolute floor `mineig=1e-15` | relative floor `mineig_rel=1e-12` | the absolute floor is unit-dependent: MEG in Tesla yields rank 0, and average-referenced EEG is detected by luck | `mineig_rel=None` |
| 2 | Zero numerical rank | `numeigs = 0`, continues | `ValueError` naming cause and fix | fitting a zero-dimensional model is not a recoverable state | — (no reason to want it) |
| 3 | Returned iterate | last EM iterate | highest-likelihood iterate (`keep_best`) | the lrate schedule is non-monotone; late Newton overshoots cut LL variance 12.7x → 2.0x | `keep_best=False` |
| 4 | Newton | on (`do_newton=1`) | off | isolates the algorithm from initialization for parity work | `do_newton=True` |
| 5 | Degenerate fits | returns NaN sources, and writes them out on its `writestep` cadence | PyTorch refuses `transform`/`get_*`/`save`; NumPy reports `converged=False` with a `stop_reason`, refuses the final write, and skips each periodic checkpoint with a logged reason (leaving the last valid one on disk) | NaN sources silently poison downstream analysis, and `loadmodout` reads a NaN checkpoint back without complaint | — (see ADR 0003, issues #50 and #240) |
| 6 | Precision | float64 | float64 (float32 on Apple GPUs) | Apple GPUs have no float64; float32 agrees to ~7 significant digits, not bit-parity | `dtype=torch.float64` |
| 7 | Sensor-space maps | `Spinv` applied internally | `get_sensor_mixing_matrix()` | `get_mixing_matrix()` returns sphered-space `A`; switching its meaning by data conditioning would be worse | — |
| 8 | Columns merged away by `share_comps` | updated to NaN, then hidden by the `comp_used` mask | frozen at their last finite value (never divided) | a fit must not end holding NaN parameters, mask or no mask; the columns are dead either way | — (see issues #60, #240) |

Rows 1, 2 and 7 arrived with [ADR 0004](https://github.com/sccn/pAMICA/blob/main/.context/decisions/0004-rank-deficient-input-handling.md);
row 3 with ADR 0003; row 5 with issue #50; row 8 with issues #60 and #240.

Two `share_comps` details are pamica's own because the reference cannot decide
them: the A-freeze window after a merge is anchored on `share_start` (the literal
`mod(iter, share_iter)` misaligns unless `share_start` is a multiple of
`share_iter`, and freezes A permanently for `share_iter <= 6`, which both array
backends reject up front), and the merge similarity metric has no bit-exact
oracle at all — the reference's `Spinv2` is declared but never allocated, so its
own reassignment is unrunnable.

## 1. Relative rank threshold (the one changed default)

The reference decides how many dimensions are real with an absolute floor on covariance
eigenvalues:

```fortran
numeigs = min(pcakeep, count(eigs > mineig))   ! amica15.f90:395, mineig = 1e-15
```

Because the floor is absolute, it depends on the physical units of the input:

- **MEG in Tesla** has covariance eigenvalues ~1e-26. Every one is below `1e-15`, so the
  reference computes `numeigs = 0`.
- **Average-referenced EEG** — an everyday preprocessing step — sits at
  `lambda_min/lambda_max = 8.5e-17` on the bundled sample. Whether the rank-deficient
  dimension is caught depends on the recording's absolute scale.

pamica defaults to a scale-free floor instead, `mineig_rel * largest_eigenvalue`. On real
EEG projected to rank 20:

| threshold | rank found | reconstruction error |
|---|---|---|
| `mineig=1e-15` (reference) | 24 | 1.98e-09 |
| `mineig_rel=1e-12` (pamica) | 20 | 7.52e-15 |

**This does not affect ordinary data.** Real EEG has `lambda_min/lambda_max ~ 5e-4`,
eight orders of magnitude above the relative floor, so nothing is reduced and results are
bit-identical to the reference. The difference appears only where the reference was
already unreliable.

`mineig_rel` *replaces* the absolute floor rather than combining with it — for Tesla-scale
data a relative floor is ~1e-35 in absolute terms, so taking the larger of the two would
silently discard it.

```python
from pamica import AMICA

AMICA().fit(X)                      # relative floor (default)
AMICA().fit(X, mineig_rel=None)     # exactly the reference's absolute floor
AMICA().fit(X, mineig_rel=1e-9)     # stricter rank detection
```

## Working with rank-deficient data

Rank deficiency is routine — Maxwell filtering, average referencing, channel
interpolation all cause it. pamica sizes the model to the detected rank, so a
306-channel MEG recording at rank 70 yields 70 sources, not 306.

```python
m = AMICA().fit(X)                       # X is (306, T), numerical rank ~70
m.model_.n_channels                      # 70  - sources actually estimated
m.model_.n_channels_in                   # 306 - input channels

S = m.transform(X)                       # (70, T) sources
A = m.model_.get_sensor_mixing_matrix()  # (306, 70) scalp maps
```

`get_mixing_matrix()` returns `A` in the sphered space. Use
`get_sensor_mixing_matrix()` for anything sensor-shaped (topographies, dipole fitting,
export) — when rank reduction is active the sphere is non-square and only the
pseudo-inverse maps back.

### Mixed channel types (MEG)

Magnetometers and gradiometers have different physical units, and the difference is not
cosmetic: scaling barely affects a full-rank fit, but it decides *which* directions
survive rank truncation. On mixed data, badly scaled input retained 72.6% of signal
variance against 99.2% for correctly scaled input.

The MNE wrapper handles this for you, following MNE's own ICA convention — one `std` per
channel type, applied as `X / pre_whitener_`:

```python
from pamica.mne_compat import AMICAICA

fitted = AMICAICA().fit(raw)     # channel types scaled automatically
fitted.pre_whitener_             # (n_channels, 1) scale actually applied
```

For a single channel type this changes nothing: AMICA's sphering absorbs a global
rescale exactly (verified — the two spheres' ratio is one constant to ~1e-15).

Using the array API (`pamica.AMICA`) directly, scale by channel type yourself before
fitting; there is no `info` from which to infer types.

## Backend differences

Separate from reference divergences: the optional MLX backend is a subset.

| Feature | PyTorch | NumPy | MLX | Native Fortran |
|---|---|---|---|---|
| Newton | yes | yes | `NotImplementedError` | yes |
| PDF families | all five | all five | GG only | all five |
| Component sharing | yes | yes | no | yes |
| Outlier rejection | yes | yes | no | yes |
| Precision | f64/f32 | f64 | f32 only | f64 |
| Rank detection | yes | yes | yes | yes (absolute floor) |
| `min_dll` stop | yes | yes | **no** | yes |
| `min_nd` stop / gradient norm | yes | yes (as `min_grad_norm`) | **no** | yes |
| `keep_best` best-iterate restore | yes | no | no | n/a |
| MIR diagnostic | yes | no | no | n/a |
| Persistence | `state_dict` | EEGLAB `amicaout` | none | EEGLAB `amicaout` |

Most MLX limitations fail loudly: `do_newton=True` and non-GG `pdftype` raise
`NotImplementedError`, and every unsupported parameter is simply absent from the
constructor, so passing it raises `TypeError`.

**The convergence stops are the exception, and the one difference to plan
around.** `pamica/mlx_impl/core.py` implements neither stop — it has no
`min_dll`/`use_min_dll`/`maxincs` and no `min_nd`/`use_grad_norm`/`ndtmpsum`, so
its only outcomes are `max_iter`, `lrate_floor`, and the degenerate reasons
(`nan_ll`, `singular_ll`, `nan_params`).

The distinction is not that an MLX fit runs to `max_iter` — a PyTorch fit does
too at its own `max_iter=100` default, because `min_dll` needs several hundred
iterations to trigger on a recording this size. The distinction is what raising
`max_iter` buys you. On the other backends, a larger budget lets the likelihood
stop end the fit when it stops improving. **On MLX a larger budget is simply
spent**, however long the fit has stopped making progress.

Nothing warns you about this, because there is no parameter to reject. Two
consequences:

- Moving a working configuration from the PyTorch backend to MLX for the
  Apple-GPU speedup does more work than the same `max_iter` implies. Size
  `max_iter` for MLX from where the other backends actually stop improving on
  your data.
- MLX cannot be evaluated on gradient-norm behaviour at all, because it never
  computes the quantity. Any statement elsewhere in these guides about whether
  the `min_nd` threshold is reachable covers Fortran, PyTorch and NumPy only.
  (`numpy_impl` spells that same threshold `min_grad_norm`.)

Tracked in issue #248.

## Component sharing on rank-reduced fits

`share_comps` merges components that are near-collinear *across models*,
comparing each pair of mixing columns after mapping them back to input-channel
(sensor) space.
The PyTorch backend built that back-map with `inv(sphere)`, so it refused every
rank-reduced or rank-deficient fit: with rank reduction active the sphere is
`(n_kept, n_channels)` and has no inverse, and a square sphere fitted on
rank-deficient data is singular.
The back-map is now `pinv(sphere)`, which is what the reference itself carries
under reduction (`Spinv(nx, numeigs)`, amica15.f90:550-560), so sharing works at
any rank (issue #253, reported from Maxwell-filtered MEG in #221).

```python
m = AMICA(n_models=2).fit(X, share_comps=True)   # X may be rank-deficient
m.shared_components()                            # groups of (model, source) pairs
```

For a full-rank square sphere `pinv` equals `inv` to about 1e-15, far below the
`comp_thresh` decision boundary (0.99 by default), so merge decisions on
well-conditioned data are unchanged; the bundled sample reproduces its previous
`comp_list` and log-likelihood bit for bit.

The NumPy backend reaches the merge decision from a different metric: its
`identify_shared_components` compares the columns of `A` directly in the sphered
space, with no back-map at all, so it never needed an inverse and already ran at
any rank.
The two backends can therefore identify different pairs on the same data; the
NumPy sharing path is being reworked in issues #240 and #242.
