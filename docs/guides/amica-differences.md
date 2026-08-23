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
| 9 | Block-size search | on (`do_opt_block=1`), sweeps 128–1024, **aborts** if a candidate cannot allocate | off; sweeps 4096–32768; a candidate that cannot allocate is skipped and the fit continues | the choice is timing-based and therefore machine-dependent, which a parity run cannot have; Fortran's range sits far below where any pamica backend peaks; and running out of memory is a reason to use a smaller block, not to stop | `do_opt_block=True` (but pin `block_size` for a bit-for-bit comparison) |

Rows 1, 2 and 7 arrived with [ADR 0004](https://github.com/sccn/pAMICA/blob/main/.context/decisions/0004-rank-deficient-input-handling.md);
row 3 with ADR 0003; row 5 with issue #50; row 8 with issues #60 and #240;
row 9 with issue #232.

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
numeigs = min(pcakeep, count(eigs > mineig))   ! amica15.f90:413, mineig = 1e-15
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
| Newton | yes | yes | yes (float32) | yes |
| PDF families | all five | GG only | all five (float32) | all five |
| Component sharing | yes | yes | yes | yes |
| Outlier rejection | yes | yes | no | yes |
| Precision | f64/f32 | f64 | f32 only | f64 |
| Rank detection | yes | yes | yes | yes (absolute floor) |
| `min_dll` stop | yes | yes | yes | yes |
| `min_nd` stop / gradient norm | yes | yes (as `min_grad_norm`) | yes | yes |
| `keep_best` best-iterate restore | yes | no | no | n/a |
| MIR diagnostic | yes | no | no | n/a |
| Persistence | `state_dict` | EEGLAB `amicaout` | none | EEGLAB `amicaout` |

The NumPy row's "GG only" corrects an earlier version of this table, which
listed "all five": `AMICA_NumPy._compute_log_pdf` (its fit-path density
function) has no `pdtype` parameter at all, so the legacy backend never
implemented the non-GG families the PyTorch and MLX backends carry (issue
#265).

Most MLX limitations fail loudly: `transform` and every unsupported parameter
(outlier rejection, save/load) are simply absent from the constructor or raise
`NotImplementedError`, rather than silently downgrading.

The convergence stops used to be the exception — MLX implemented neither, so a
fit there always spent the whole iteration budget. Issue #248 closed that gap:
`pamica/mlx_impl/core.py` now carries `use_min_dll`/`min_dll`/`maxincs` and
`use_grad_norm`/`min_nd` with the PyTorch backend's names, defaults and
`stop_reason` strings (`min_dll`, `grad_norm`, `grad_norm_floor`), and computes
the weight-gradient norm `ndtmpsum` every iteration. A configuration moved
between the two backends does the same work and stops for the same reason on the
same iteration (`pamica/tests/test_mlx_convergence_stops.py` asserts exactly
that on the bundled sample). Statements elsewhere in these guides about the
`min_nd` threshold being unreachable on small recordings now cover MLX too;
`numpy_impl` spells that same threshold `min_grad_norm`.

Component sharing was the other gap, closed by issue #263: `AMICAMLXNG` now takes
`share_comps`/`share_start`/`share_iter`/`comp_thresh` with the PyTorch
backend's names, defaults and validation, runs the same merge schedule and
post-merge A-freeze, and exposes `comp_used`/`shared_components()`.
It does not re-derive the merge metric — it calls the same
`identify_shared_components` kernel the NumPy backend uses, on host float64
`pinv(sphere) @ A`, so all three backends decide identically from one fitted
state (`pamica/tests/test_mlx_sharing_cross_backend.py` pins that against
`AMICATorchNG`).
Row 8 of the "At a glance" table at the top of this page (merged-away columns
frozen at their last finite value, not left NaN behind the mask) holds in MLX as
well.

Newton was next, closed by issue #264: `AMICAMLXNG` takes
`do_newton`/`newt_start`/`newtrate`/`newt_ramp` with the PyTorch backend's names,
defaults and semantics, accumulates the same curvature statistics, applies the
same per-source-pair 2x2 solve behind the same raw `prod > 1` guard, and counts
rejections in `n_newton_fallbacks`. It runs entirely in float32 — Apple GPUs have
no FP64 — which was pre-registered as a go/no-go rather than assumed: on the
bundled sample the curvature matches a float64 PyTorch twin to 4e-7 relative, a
matched 100-iteration fit lands on the float64 likelihood to five significant
digits, and the positive-definiteness guard never comes within 1.9 of its
boundary. Evidence and the gate script are in `.context/issue-264/`. `newtrate`
is a float32 ceiling like `lrate_cap`, so a Newton fit on MLX should be treated
as ~7-significant-digit, not float64-parity — use the PyTorch backend for
Fortran-parity runs, as the Precision row above already implies.

The non-GG PDF families were the last of the four, closed by issue #265:
`AMICAMLXNG` takes `pdftype`/`kurt_start`/`num_kurt`/`kurt_int` with the
PyTorch backend's names, defaults and semantics — all five `amica15.f90`
families (0 GG, 2 Gaussian, 3 logistic, 4 sub-Gaussian cosh+, and the
`pdftype=1` extended-Infomax adaptive switcher between codes 1/4 by kurtosis
sign) — and exposes `get_pdftype()`. `pdftype=0` stays byte-for-byte the
pre-#265 implementation (the `_pdtype_h` `None` fast path adds zero graph
nodes; verified by a before/after fit comparison on the bundled sample). The
fixed families' `z0`/`fp` match the literal Fortran forms through MLX's
float32 evaluation to 1e-6 (absolute, since code 4's `y - tanh(y)` cancels
catastrophically near `y=0` in float32 — measured 100% relative error at
`y=1e-4` — so the true parity claim is against the formula, not against a
Taylor-stabilized substitute), and a matched-budget fit lands on the float64
PyTorch likelihood to within 0.05 for every family. `self.dorho` (a flag, set
to `pdftype == 0`) gates the `drho_n` accumulation and the per-iteration
lgamma-table refresh here, skipping work AMICATorchNG always pays for a
frozen non-GG `rho` (a deliberate MLX-only WORK divergence, not a numeric
one) — its digamma pull is already gated behind the same flag, so that part
is unchanged. Like `share_comps`'s merge metric, the switcher has no
bit-exact oracle
— the reference declares `do_choose_pdfs` (`pdftype=1`) but never accumulates
the moments that would drive it — so it is behavior-validated on real EEG, and
`share_comps` does NOT synchronize `pdtype` across a merged pair (see
`shared_components()`'s docstring). Evidence is in `.context/issue-265/`.

## Component sharing on rank-reduced fits

`share_comps` merges components that are near-collinear *across models*,
comparing each pair of mixing columns after mapping them back to input-channel
(sensor) space.
The PyTorch backend built that back-map with `inv(sphere)`, so it refused every
rank-reduced or rank-deficient fit: with rank reduction active the sphere is
`(n_kept, n_channels)` and has no inverse, and a square sphere fitted on
rank-deficient data is singular.
The back-map is now `pinv(sphere)`, which is what the reference itself carries
under reduction (`Spinv(nx, numeigs)`, amica15.f90:568-578), so sharing works at
any rank (issue #253, reported from Maxwell-filtered MEG in #221).

```python
m = AMICA(n_models=2).fit(X, share_comps=True)   # X may be rank-deficient
m.shared_components()                            # groups of (model, source) pairs
```

For a full-rank square sphere `pinv` equals `inv` to about 1e-15, far below the
`comp_thresh` decision boundary (0.99 by default), so merge decisions on
well-conditioned data are unchanged; the bundled sample reproduces its previous
`comp_list` and log-likelihood bit for bit.

The NumPy backend now reaches the merge decision from the same metric (issue
#258): `identify_shared_components` takes the de-sphered sensor-space maps
directly -- `pinv(sphere) @ A`, the same back-map described above -- instead of
comparing columns of the sphered `A`.
The two backends therefore make the identical merge decision from the same
fitted state (`pamica/tests/test_numpy_share_comps.py::test_numpy_merge_decision_matches_torch_backend`).
The MLX backend calls that same kernel on the same host float64 inputs (issue
#263), so all three agree.
On one real fitted 2-model state the top candidate cross-model pair measured
0.992 cosine similarity in sensor space against 0.970 in the old sphered
space -- close enough that, with the default `comp_thresh=0.99`, the two
metrics disagree on whether that pair merges. A NumPy fit that shares
components can therefore reach a different `comp_list` than it did before
#258, even on a full-rank, well-conditioned sphere; only the `pinv`-vs-`inv`
comparison two paragraphs above is unaffected by that swap.

## `final_ll_` trails a final-iteration merge (issue #269)

If a `share_comps` merge fires on the LAST fit iteration, the returned
`A`/`W`/`comp_list` are already post-merge, but the reported log-likelihood
(`final_ll_` in `AMICATorchNG`/`AMICAMLXNG`, `self.ll[-1]` in the NumPy
backend) still reports the value computed under the PRE-merge `comp_list`.
The merge's effect on the likelihood only shows up in the next iteration's
E-step, which never runs.

This is not a bug: it matches the reference ordering. `identify_shared_comps`
runs after the iteration's likelihood has already been accumulated
(amica15.f90:1856-1858 vs the earlier LL accumulation), so Fortran has the same
gap. All three backends share it by construction, and it is pinned as
behavior rather than fixed (`test_merge_on_the_final_iteration_completes` in
each of `tests/torch_tests/test_ng_sharing.py`,
`tests/test_numpy_share_comps.py` and `tests/mlx_tests/test_mlx_sharing.py`).

One interaction worth knowing: PyTorch's `keep_best` safeguard (row 3 above)
is disabled whenever `share_comps` is on, precisely because a merge changes
the parameter count mid-fit -- restoring an earlier snapshot would silently
undo the merge. So under sharing, every backend returns the last iterate, and
`final_ll_`/`self.ll[-1]` trailing a final-iteration merge is not a
`keep_best` artifact; it happens the same way with `keep_best=False`.

## The block-size search picks a machine-dependent value (issue #232)

`do_opt_block` times a few candidate `block_size` values on your data and
device at the start of `fit` and keeps the fastest, under Fortran's own four
parameter names (`do_opt_block`, `blk_min`, `blk_max`, `blk_step`) and
Fortran's arithmetic stepping. It is available on all three backends.

Because the winner is decided by measured time, **two machines can pick
different block sizes for the same data**, and their trajectories then differ
at the same ~1e-6 level any `block_size` change produces (see
[Block-size sensitivity](validation.md#block-size-sensitivity)). That is why
the search is **off by default** in every backend, unlike Fortran, whose header
default turns it on. A run being compared bit-for-bit against the reference
binary must leave `do_opt_block` off and pin `block_size` on both sides.

The pamica sweep bounds are re-derived rather than copied. Fortran sweeps
128–1024, which is entirely below where any pamica backend peaks; the pamica
defaults (4096–32768 in steps of 4096) bracket the measured CPU optimum and
include the shipped `block_size=8192`. A file that sets `blk_min`/`blk_max`/
`blk_step` explicitly is honored as written, so a literal Fortran
`input.param` means the same thing on both sides.

The behavioral difference that motivated the port is what happens when a
candidate does not fit in memory. The search walks *upward* into larger blocks,
which is exactly where memory runs out, and Fortran's `determine_block_size`
calls `allocate_blocks` with no `stat=`, so the run aborts. In pamica the
failing candidate is skipped, the upward walk stops (every larger candidate
would fail too), and the fit continues at the largest size that actually ran --
or at the `block_size` you configured, if nothing could be timed. Candidates
are additionally capped by `n_samples` and by a conservative estimate of one
block's peak, so the search usually finds its ceiling without having to walk
into a failure at all.

One consequence worth stating plainly: on the NumPy backend this flag used to
default to **on** (following Fortran's header) while its sweep ran over
128–1024, so every NumPy fit quietly re-tuned itself to a small block and
ignored the `block_size` it was given. It is now off by default there too, and
that backend's default `block_size` is the shipped 8192.
