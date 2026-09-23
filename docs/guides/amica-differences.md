# pamica vs. AMICA: every deliberate difference

pamica reproduces Jason Palmer's Fortran AMICA numerically, and parity with that
reference is how correctness is defined here. This page lists every place pamica
**deliberately** behaves differently, why, and how to restore the reference behavior.

Anything not on this page is intended to match the reference. If you find a difference
that is not listed, that is a bug worth
[reporting](https://github.com/sccn/pAMICA/issues).

## At a glance

| # | Area | Fortran AMICA | pamica default | Why | Restore reference |
|---|---|---|---|---|---|
| 1 | Rank threshold | absolute floor `mineig=1e-15` | relative floor `mineig_rel=1e-12` | the absolute floor is unit-dependent: MEG in Tesla yields rank 0, and average-referenced EEG is detected by luck | `mineig_rel=None` |
| 2 | Zero numerical rank | `numeigs = 0`, continues | `ValueError` naming cause and fix | fitting a zero-dimensional model is not a recoverable state | none (no reason to want it) |
| 3 | Returned iterate | last EM iterate | highest-likelihood iterate (`keep_best`) | the lrate schedule is non-monotone, so a fit can end below its peak (before epic #324, late Newton overshoots made the multi-model LL sd 12.7x Fortran's, 2.0x with `keep_best`; with the epic's code a restore fired in one of 20 seeded fits, at the 300-iteration budget only; ADR 0003) | `keep_best=False` |
| 4 | Newton | off in the compiled defaults (amica15_header.f90:19); the bundled `input.param` turns it on (`do_newton 1`) | off | the compiled default; Newton-off fits isolate the algorithm from initialization for parity work | `do_newton=True`, as the bundled `input.param` sets |
| 5 | Degenerate fits | returns NaN sources, and writes them out on its `writestep` cadence | the `AMICA` wrapper, on either backend, refuses `transform`/`get_*`/`save`; the raw `AMICATorchNG`, `AMICAMLXNG` and NumPy `AMICA` backends refuse their own output accessors on a degenerate fit too (issue #306); NumPy additionally reports `converged=False` with a `stop_reason`, refuses the final write, and skips each periodic checkpoint with a logged reason (leaving the last valid one on disk). Every backend stops before a non-finite value is applied or returned: a non-finite likelihood (`nan_ll`, or `singular_ll` for an infinite one; never recorded in the history), update direction (`nan_direction`) or parameter after an update (`nan_params`), where the reference applies a NaN step and exits on the next likelihood | NaN sources silently poison downstream analysis, and `loadmodout` reads a NaN checkpoint back without complaint | none (see issues #50, #240, #306 and #339) |
| 6 | Precision | float64 | float64 (float32 on Apple GPUs) | Apple GPUs have no float64; float32 agrees to ~7 significant digits, not bit-parity | `dtype=torch.float64` |
| 7 | Sensor-space maps | `Spinv` applied internally | `get_sensor_mixing_matrix()` | `get_mixing_matrix()` returns sphered-space `A`; switching its meaning by data conditioning would be worse | none |
| 8 | Components merged away by `share_comps` | mixing vector and density updated to NaN, then hidden by the `comp_used` mask | frozen at their last finite value (never divided, never rescaled) | a fit must not end holding NaN parameters, mask or no mask; the components are dead either way | none (see issues #60, #240, #334) |
| 9 | Block-size search | on (`do_opt_block=1`), sweeps 128–1024, **aborts** if a candidate cannot allocate | off; sweeps 4096–32768; a candidate that cannot allocate is skipped and the fit continues | the choice is timing-based and therefore machine-dependent, which a parity run cannot have; Fortran's range sits far below where any pamica backend peaks; and running out of memory is a reason to use a smaller block, not to stop | `do_opt_block=True` (but pin `block_size` for a bit-for-bit comparison) |
| 10 | Restarts across seeds | none (its `maxrestarts` only *recovers* from an early NaN) | available as `n_restarts`, **off by default** (`n_restarts=1`) | the weakest under-determined components are init-basin sensitive, so best-of-N buys robustness; but a default that ran N fits would change every result and cost N times as long | `n_restarts=1` (the default) |
| 11 | Reconstruction after rank reduction (`AMICAICA.apply`) | output has no representation of the discarded principal component analysis (PCA) subspace (sphere rows past `numeigs` are zero), so any back-projection drops it | the MNE export carries the full PCA basis, so `apply` restores the residual | MNE's own `ICA` does; the residual was never part of the independent component analysis (ICA) decomposition, so it is not ICA's to remove | `apply(..., n_pca_components=ica.n_components_)` |
| 12 | `pcadb` | parsed (amica15.f90:3459-3461), never used | unset by default; when set alone, keeps the dimensions within `pcadb` dB of the largest eigenvalue; ignored when `pcakeep` is also set | a dB cut is a scale-free way to drop low-variance directions; letting `pcakeep` win preserves what a reference `input.param` that sets both (both bundled files do) means to the binary | leave `pcadb` unset (the default), or set `pcakeep` |
| 13 | Preprocessing with `do_sphere=False` | divides each channel by its standard deviation and adds a log-determinant term to the likelihood (amica15.f90:516-526) | identity sphere with a zero log-determinant, on all three array backends: the data are fitted unscaled | not a deliberate choice: an existing divergence, found during epic #324 and not yet ported | none yet (issue #328) |
| 14 | `scalestep` | parsed (amica15.f90:3686), never used; rescales every iteration (:1843) | rescales every `scalestep` iterations counted from 1; default 1 matches the reference | pamica has always honored the keyword, and keeping it costs nothing; since issue #333 it counts from 1 like the reference's live cadences (`writestep`, `histstep`) instead of firing on the first iteration | `scalestep=1`, the default |
| 15 | Restart after a non-finite likelihood (NumPy backend only) | within the first `restartiter` iterations, redraws `A` up to `maxrestarts + 1` times (`numrestarts > maxrestarts` ends the run, amica15.f90:1022-1049); `startover` is set at :1046 and never cleared, so after the first restart the binary never calls `update_params` again (:1115-1122) and the redrawn parameters are never fitted | redraws `A` up to `maxrestarts` times within the same window (counted from 1; `restartiter=0` disables it) and resumes fitting after each restart, with the learning rates and likelihood history reset; the restart uses up that iteration of `max_iter` | the reference's recovery path cannot resume, so its count has nothing to recover into; a restart that fits is the evident intent. PyTorch and MLX have no restart-on-NaN path at all: they stop on the non-finite likelihood (`stop_reason="nan_ll"`), and their `n_restarts` (row 10) is a different mechanism | none for the stalled loop; `maxrestarts` one higher reproduces the reference's count, and `restartiter=0` stops on the first non-finite likelihood, as the reference does with `restartiter=0` |
| 16 | Defaults of `input.param` keys | the compiled-in defaults are single-precision literals widened to double (amica15_header.f90:66-74), so a key missing from `input.param` takes the float32 rounding of its default: `comp_thresh` is 0.9900000095, and likewise `mineig`, `minlrate`, `rholrate`, `rholratefact`, `invsigmin`, `min_dll`, `min_grad_norm` and `lrate` (0.1000000015, printed by the binary; without the key its ceiling `lrate0` is never set, and the binary's `lrate` drops to 0 after the first iteration); a value given in `input.param` is read as double | the decimal values (`lrate=0.1`), given or defaulted | they are user inputs, and the reference reads any value it is given as double; pamica's native engine and the seeded oracles write every one of these keys, so they never meet the compiled-in values. The constants the reference hard-codes the same way are its values in every backend since issue #344 ([below](#single-precision-constants-issue-344)) | pass the float32 rounding, for example `lrate=float(numpy.float32(0.1))` |

Rows 1, 2 and 7 arrived with [ADR 0004](https://github.com/sccn/pAMICA/blob/main/.context/decisions/0004-rank-deficient-input-handling.md);
row 3 with ADR 0003; row 5 with issue #50, extended to the raw backends by issue #306 and to non-finite steps and updates by issue #339; row 8 with issues #60, #240 and #334;
row 9 with issue #232; row 10 with issue #198; row 11 with issue #322 (ADR 0005);
row 12 with issue #323; row 13 is recorded, not yet resolved, by issue #328;
row 14 with issue #333 (ADR 0006);
row 15 with issue #335, which also made the NumPy restart window count from 1;
row 16 with issue #344, which adopted the reference's hard-coded single-precision constants.

The A-freeze is the reference's arithmetic, applied as it is (issue #345):
once `iter >= share_start`, every iteration with `mod(iter, share_iter) <= 5` holds the `A` update,
together with the `lrate` ramp and the `rholrate` reset that share its branch (amica15.f90:1803).
The reference applies it to every fit, one model or several, whether or not `share_comps` is on,
so with the defaults (`share_start = share_iter = 100`) every backend holds `A` on iterations 100-105, 200-205, and so on.
Until issue #345 pamica held `A` only under `share_comps` with two or more models, on each scan iteration and the five after it, counted from `share_start`.
The literal remainder starts the window on the scan iteration only when `share_start` is a multiple of `share_iter`, as with the defaults;
otherwise the window and the scan fall on different iterations, in the reference and in pamica alike.
A `share_iter` below 7 would leave every remainder at 5 or less, holding `A` permanently from `share_start` on,
so every backend rejects it, sharing on or off (the NumPy backend takes it as `share_int` or `share_iter`).
For the same reason every backend requires `share_start >= 1` whether or not sharing is on:
`share_start=0` would start the freeze on the first iteration.

One `share_comps` detail is pamica's own because the reference cannot decide it:
the merge similarity metric has no bit-exact oracle, because the reference's `Spinv2` is declared but never allocated.
Its scan still runs, but every similarity it computes is NaN, so a reference run with `share_comps` on never merges anything:
seeded from pamica's initialization, the pinned binary leaves `comp_list` unchanged
and reports 64 unique components after a scan at iteration 8, even at `comp_thresh=0`.
The merged state the scan would produce does have an oracle:
the reference's `load_comp_list` seeds a merged `comp_list`, and the PyTorch
and NumPy updates from such a state match the native binary to float64 round-off
([below](#component-sharing-compares-and-ties-components-issue-334)).

## 1. Relative rank threshold

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

## `AMICAICA.apply` restores the PCA residual (issue #322)

A rank-reduced fit models only the retained PCA subspace,
whether the reduction comes from an explicit `pcakeep`/`pcadb` or from automatic rank detection on Maxwell-filtered, average-referenced or interpolated data.
The rest of the data, the PCA residual, is never part of the decomposition.
This is row 11 of the table above, and it goes beyond what the reference does.

**What the reference does.**
Fortran keeps `numeigs = min(pcakeep, count(eigs > mineig))` dimensions (amica15.f90:413 and 466) and sizes the model to them (`nw = numeigs`, amica15.f90:563).
In the default `do_approx_sphere` path its written sphere `S` is `nx x nx`, but the rows past `numeigs` are zero (amica15.f90:501-508),
and `W`/`A` are `nw x nw`.
Its output therefore has no representation of the residual,
so a back-projection through the `nchan x nw` inverse of that output (as EEGLAB's `icawinv` is built) can only reconstruct the retained subspace,
dropping the residual without notice.
That is an inference from the shape of the output; EEGLAB itself is not part of this repository.
pamica's own EEGLAB export (`write_amica_output`) pads a reduced sphere with zero rows the same way, and is unchanged.

**What pamica does.**
`AMICAICA.fit` stores the full orthonormal PCA basis as `pca_components_` (`n_channels x n_channels`) and `pca_explained_variance_`,
computed once from the fitted sphere and the fit data.
The first `n_components_` rows are the retained subspace, exactly as before.
The remaining rows span the residual, ordered by descending variance, and their variances are the covariance eigenvalues the reduction discarded.
`to_mne_ica` exports the full basis, and MNE's `ICA.apply` keeps rows past `n_components_` as residual PCA components
(its default `n_pca_components=None` keeps them all).
So `apply` with nothing excluded returns the input, and excluding a component removes that component's back-projection and nothing else.
On the bundled 32-channel EEG with `pcakeep=20`, `apply` with nothing excluded used to lose 8.0% of the signal (relative error 0.080);
it now reproduces the input to 1.3e-15.

**Why.**
It is MNE's own ICA default: `mne.preprocessing.ICA.fit` stores the full PCA, so its `apply` restores the residual,
and an AMICA decomposition handed to MNE should behave like any other.
More fundamentally, the residual was never part of the decomposition, so it is not ICA's to remove.
Excluding an artifact component should not also delete every dimension the model never saw, which can carry signal of interest.

**What does not change.**
Sources, component maps, the unmixing matrix, `n_components_` and the log-likelihood are the same as before; only reconstruction changes.
`get_sources` still equals `AMICA.transform`, and a full-rank fit, which has no residual, exports bit-identically.
The backends and their persisted state are untouched.

**How to restore the reference behavior.**
Pass MNE's own `n_pca_components` to keep only the retained subspace:

```python
from pamica.mne_compat import AMICAICA

ica = AMICAICA().fit(raw, pcakeep=20)
clean = ica.apply(raw.copy(), exclude=[0])  # residual restored (default)
ref = ica.apply(raw.copy(), exclude=[0], n_pca_components=ica.n_components_)  # reference

mne_ica = ica.to_mne_ica().copy()
mne_ica.n_pca_components = mne_ica.n_components_  # the same, persisted by mne_ica.save()
```

A float `n_pca_components` selects rows by cumulative explained variance, following MNE's own rule, so it can keep part of the residual;
MNE rejects a fraction that would select fewer rows than `n_components_`.
The decision and the alternatives considered are recorded in ADR 0005.

## Backend differences

Separate from reference divergences, this table compares the backends.
The raw MLX backend (`AMICAMLXNG`) has the PyTorch backend's full feature surface except precision:
it is float32-only, because Apple GPUs have no float64.
The `AMICA` and `AMICAICA` wrappers construct it with `backend="mlx"` (epic #324 Phase 4, issue #313),
so the params-file reader, the degenerate-fit contract, `.pt` `save`/`load`, the EEGLAB export and the MNE path all run on it
(see [Selecting a backend](backends.md#selecting-a-backend)).

| Feature | PyTorch | NumPy | MLX | Native Fortran |
|---|---|---|---|---|
| Newton | yes | yes | yes (float32) | yes |
| PDF families | all five | GG only | all five (float32) | all five |
| Component sharing | yes | yes | yes | yes |
| Outlier rejection | yes | yes | yes | yes |
| Precision | f64/f32 | f64 | f32 only | f64 |
| Rank detection | yes | yes | yes | yes (absolute floor) |
| Explicit `pcakeep`/`pcadb` | yes | yes | yes | `pcakeep` only; `pcadb` parsed but unused |
| `min_dll` stop | yes | yes | yes | yes |
| `min_nd` stop / gradient norm | yes | yes (as `min_grad_norm`) | yes | yes |
| `keep_best` best-iterate restore | yes | no | yes | n/a |
| `n_restarts` best-of-N restarts | yes | yes | yes | n/a |
| Mutual Information Reduction (MIR) diagnostic | yes | no | yes | n/a |
| `variance_order` (EEGLAB component order) | yes | no (issue #317) | yes | applied by `loadmodout15` on load |
| Per-sample scoring (`model_loglik`/`model_probability`) | yes | no | yes | n/a |
| Through the `AMICA` and `AMICAICA` wrappers | yes (`backend="torch"`, the default) | no, used directly | yes (`backend="mlx"`) | no, used directly as `AMICANative` |
| Persistence | `state_dict`, `AMICA.save` `.pt` + EEGLAB `amicaout` export | EEGLAB `amicaout` | `state_dict`/`.npz` `save`-`load`, `AMICA.save` `.pt` + EEGLAB `amicaout` export | EEGLAB `amicaout` |
| Parameter file (JSON or Fortran `input.param`) | yes (`AMICA.from_params_file`, #132) | yes (`AMICA_NumPy(params_file=...)` / `from_params_file`, #304) | yes (`AMICA.from_params_file(..., backend="mlx")`) | the binary reads `input.param`; `AMICANative` takes its keys as keywords |
| A second `fit` on the same instance | starts from a fresh initialization | continues from the previous fit ([see below](#refitting-a-numpy-instance-continues-from-the-previous-fit-issue-312)) | starts from a fresh initialization | every run starts fresh, or from `load_*` files |

The NumPy row's "GG only" (generalized Gaussian, GG) corrects an earlier
version of this table, which listed "all five": `AMICA_NumPy._compute_log_pdf`
(its fit-path density function) has no `pdtype` parameter at all, so the
legacy backend never implemented the non-GG families the PyTorch and MLX
backends carry (issue #265).
Issue #304 made this an enforced contract rather than a silent gap:
`AMICA_NumPy(pdftype=...)` with anything other than `0` now raises
`NotImplementedError` at construction (the `.rules/backend_parity.md`
narrow exception).
The bundled `numpy_impl/params.json`'s default `pdftype` changed `1` -> `0`
to match; it was never read by the fit path, so this was a dormant
divergence, not a behavior change to any actual fit.

The NumPy row's params-file reader (`params_file=...` / `from_params_file`,
renamed from `from_json_file`) now accepts the same two formats as the
PyTorch wrapper -- pamica's JSON schema and the literal Fortran
`input.param` text, content-sniffed -- through the same shared
`pamica.fortran_params.read_params_file` (issue #304).
Previously `params_file` was JSON-only and a Fortran text file raised a raw
`json.JSONDecodeError`; the NumPy CLI accepts both formats the same way now
too, through the same underlying helper, instead of its own separate
`json.load`.

`transform` and the `get_mixing_matrix`/`get_unmixing_matrix`/
`get_sensor_mixing_matrix`/`get_rho` accessors, plus `state_dict`/
`from_state_dict` and a device- and framework-agnostic `.npz` `save`/`load`,
were added in epic #278 Phase 1 (issue #287): `AMICAMLXNG.transform` mirrors
`AMICATorchNG.transform` semantically, deriving the unmixing composition from
MLX's own `_forward` (its `W` is `(n_models, n, n)`, not torch's
`(n, n, n_models)`), and the `.npz` layout embeds `config`/`extra` as
JSON-encoded scalars alongside the 12 native param arrays -- no torch
coupling, no pickle. The best-iterate safeguard (`keep_best`, row 3 above)
followed in epic #278 Phase 2 (issue #288), a direct port of PyTorch's #51
restore decision onto MLX's float32 trajectory. Outlier rejection
(`do_reject`), the LLt-stash-backed `model_loglik`/`model_probability`
accessors, the EEGLAB `write_amica_output` export, and the MIR and Pairwise Mutual Information (PMI)
diagnostics followed in epic #278 Phase 3 (issue #289) -- see their own
sections below. Any genuinely unsupported parameter still fails loudly: it
is simply absent from the constructor, rather than silently downgrading.

One MLX failure mode used to be worse than loud — it was uncatchable. MLX
0.32's CPU-stream `mx.linalg.inv` does not raise a Python exception on a
singular per-model mixing block `A[comp_list[:, h], :]`: LAPACK's LU
failure aborts the whole process (`libc++abi: ... [Inverse::eval_cpu] LU
factorization failed`), which no `try`/`except` around `fit` can catch.
Issue #274 closed that gap: `_update_unmixing_matrices` now condition-checks
each per-model matrix host-side (`np.linalg.cond`, cheap — the method already
crosses to the CPU stream for `inv`/`slogdet` once per iteration) immediately
before calling `inv`, and raises a catchable `RuntimeError` naming the model
index, iteration and condition number in its place. The threshold
(`_INV_COND_THRESHOLD`, 1e12) is set empirically rather than from float32's
~1/eps precision-loss point (~8-17e6, depending on convention): isolated
per-trial subprocess measurement showed the actual LU-abort onset is not a
clean function of condition number — near-duplicate-column matrices aborted
anywhere from cond~9e8 to beyond cond~5e10, while column-scaled-toward-zero
matrices never aborted even past cond~1e16 — and separately, an existing
adversarial test (`test_fallback_ramps_toward_lrate_cap_and_counts`, which
repeatedly steps `A` from the same deliberately under-determined block)
legitimately reaches cond~4.4e9 without ever hitting the abort. 1e12 clears
that observed legitimate maximum by ~225x while staying far below where a
genuinely singular `A` — the issue's literal example, a duplicated component
column — actually lands (~1e15-1e17). The guard is read-only (verified
bit-identical `A`/`W` on a short fit with and without it) and adds negligible
per-iteration overhead (~30 microseconds per model, measured on the bundled
sample).

Non-finite entries need separate handling, because they cannot be fed to
`np.linalg.cond` directly (its SVD raises `LinAlgError` on NaN/inf, a
different failure than the one this guard targets). A matrix that is
non-finite in EVERY entry — the observed shape of a zero-responsibility
("dead") model's corruption, where dividing by `dgm==0` propagates NaN/inf
through the whole per-model direction matrix — carries no structural signal
to check, so it is left to flow through to `inv`/`slogdet` unguarded exactly
as before the guard existed; `inv` returns NaN rather than aborting, caught
downstream by `fit`'s existing `nan_params` guard on that same iteration.
Any OTHER non-finite pattern first has its non-finite entries replaced with
`0.0` (a neutral fill, not an extreme sentinel — an extreme fill makes any
stray non-finite entry read as infinitely ill-conditioned by pure scale
mismatch, unable to distinguish a merely-corrupted-but-fine matrix from a
genuinely singular one) before the condition check runs. This closes a gap a
review pass found in the guard's first version: a matrix that is BOTH
non-finite in one unrelated entry AND structurally singular elsewhere (an
exact duplicate column plus a stray NaN) used to skip the check entirely and
reach the same uncatchable abort the guard exists to prevent.

**Containment is not complete**, and this is by design rather than an
oversight: no scalar condition-number threshold, on the sanitized matrix or
otherwise, can guarantee catching every conceivable abort-capable matrix. The
empirically observed LU-abort onset (cond~9e8 to beyond cond~5e10,
matrix-structure-dependent, not a clean function of cond alone) sits below
the 1e12 threshold, so a believed-rare residual window remains between "the
guard's check passes" and "this specific matrix would actually have aborted".
Running `inv` itself in a disposable per-call subprocess would close that
window completely, but was rejected as disproportionate: a subprocess spawn
on the per-iteration hot path is a far larger and less predictable cost than
one host-side `np.linalg.cond` call, for a defect this guard already makes
rare in practice.

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
A-freeze, and exposes `comp_used`/`shared_components()`.
It does not re-derive the merge metric — it calls the same
`identify_shared_components` kernel the NumPy backend uses, on the host float64
sensor maps `pinv(sphere) @ A.T` (one column per component, issue #334), so all
three backends decide identically from one fitted state
(`pamica/tests/test_mlx_sharing_cross_backend.py` pins that against
`AMICATorchNG`).
Row 8 of the "At a glance" table at the top of this page (merged-away components
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
comparing each pair of component mixing vectors after mapping them back to
input-channel (sensor) space.
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
well-conditioned data are unchanged; when that change landed, the bundled
sample reproduced its previous `comp_list` and log-likelihood bit for bit.

All three backends reach the merge decision through one kernel, `identify_shared_components`
(the NumPy backend since issue #258, MLX since issue #263),
on the same host float64 sensor-space maps, `pinv(sphere) @ A.T` since issue #334,
so they make the identical merge decision from the same fitted state
(`pamica/tests/test_numpy_share_comps.py::test_numpy_merge_decision_matches_torch_backend`
and `pamica/tests/test_mlx_sharing_cross_backend.py`).

## Component sharing compares and ties components (issue #334)

The reference's mixing matrix `A(nw, num_comps)` holds component `k` in column `k`,
and `comp_list(i, h)` names the component source `i` of model `h` uses,
for its mixing vector and its density alike.
Every pamica backend now stores the same matrix transposed, `A` of shape `(n_comps, nw)`,
so a component id names the same component in `A` as in `mu`, `sbeta` (`beta`), `alpha` and `rho`
([ADR 0007](https://github.com/sccn/pAMICA/blob/main/.context/decisions/0007-component-row-layout.md)).
Before issue #334 the backends stored each model's block transposed inside an `(nw, n_comps)` array
and indexed its COLUMNS by component id, but a stored column is one sphered channel's loadings across a model's components, not a component.
`share_comps` therefore compared and tied the wrong vectors:

- The metric now compares the components' scalp maps: the vector it uses for source `i` of model `h`
  is exactly column `i` of `get_sensor_mixing_matrix(h)`.
  The old metric's vectors had median |cosine| 0.27 to 0.39 with those maps on a two-model fit of the sample,
  so it merged pairs whose maps disagreed and missed pairs whose maps agreed.
- A merge now re-points `comp_list`, as the reference's does:
  the two sources share one mixing vector and one density, nothing is copied or averaged at the merge,
  and a merge of two identical components leaves the log-likelihood exactly unchanged (the old fold changed it by -0.186 on such a pair).
  The `gm`-weighted `dAk/zeta` step then averages the models' steps for that one component.

Seeded from a merged state through the reference's `load_comp_list`,
the PyTorch and NumPy updates match the native binary to float64 round-off after one and three iterations,
with `doscaling` on and off (`pamica/tests/test_component_rows.py`, opt-in with `AMICA_RUN_FORTRAN=1`).
On the bundled sample (2 models, seed 42, Newton on, 300 iterations, `share_start=100`, `share_iter=100`, `comp_thresh=0.95`)
the scan at iteration 100 merges one pair whose maps agree (|cos| 0.970) and the scans at 200 and 300 merge nothing,
so the fit ends with 63 of 64 components at log-likelihood -3.3410 (-3.3394 with sharing off; measured with the finished epic #324, issue #351).
Right after this change, before the reference's iteration order (issue #339) and the later changes of epic #324, the same fit merged three such pairs (|cos| 0.956 to 0.971),
and the old metric three whose maps did not (|cos| 0.06, 0.35 and 0.55).

Two consequences to know:

- A scan early in a fit merges most components, and the reference's formula does the same.
  Both models start near the identity, so their components stay near-collinear for the first iterations
  (on the sample with 2 models, seed 42 and PyTorch's defaults, a scan at iteration 8 merges 30 of 32 at `comp_thresh=0.95` and 17 at 0.99;
  one at iteration 20 merges 22 and 5).
  The reference's own scan cannot show this (it never merges, above),
  but its similarity with `Spinv2 = Spinv^T Spinv`, applied to the binary's own state after 8 iterations from pamica's initialization,
  merges exactly the pairs the PyTorch and NumPy scans merge: 32, 32 and 29 at `comp_thresh` 0.9, 0.95 and 0.99.
- A model left with few components of its own then loses its responsibility, and can end in a non-finite fit.
  In a short recipe (4096 samples, seed 23, `share_start=11`, `share_iter=11`, `comp_thresh=0.9`), 28 merges at iteration 11
  dropped the second model's `gm` from 0.57 to 9.0e-4 within two iterations
  (in pamica's own fit, whose `A` is held on those iterations, amica15.f90:1803), and the fit went non-finite in iteration 23.
  Seeded with the same merged states and run without the `A`-freeze, the reference's update behaves the same way:
  from the state after the first scan it drops the second model's `gm` to 4.2e-3 within two iterations,
  matching pamica's update rule run without the freeze from that state (not the 9.0e-4 above, which includes the freeze),
  and from the state after the second scan (iteration 22) it reaches zero in one iteration, as pamica's does, and the binary reports NaN and reinitializes.
  This is the algorithm on models that have not yet separated, not a defect of the port.
  The reference's default `share_start=100` avoids it; keep `share_start` well past the first iterations.
- Saved models: a PyTorch `state_dict` (now `format_version` 4) or MLX save (now format 2) from an earlier version
  is converted without loss unless `share_comps` had merged components,
  in which case loading raises `ValueError` and the model must be refit.
  `AMICA.save` files go through the same conversion.
  The EEGLAB export's `A` file is the reference's layout for any number of models;
  `load_results` refuses a multi-model directory written by an earlier version.

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

It can only happen on a fit that runs to `max_iter`.
A fit that stops on a convergence check (`min_dll`, the gradient norm, the `lrate` floor)
exits before that iteration's update and scan, as the reference does (amica15.f90:1111, issue #339),
so its returned parameters are the ones `final_ll_` was computed from.

One interaction worth knowing: the `keep_best` safeguard (row 3 above,
implemented on PyTorch and, since epic #278 Phase 2, MLX) is disabled
whenever `share_comps` is on, precisely because a merge changes the parameter
count mid-fit -- restoring an earlier snapshot would silently undo the merge.
The same guard disables it under `do_reject`, whose good-sample set changes
mid-fit for the analogous reason (ADR 0003; the single `track_best` condition
covers both, on both PyTorch and, since epic #278 Phase 3, MLX). So under
sharing, every backend returns
the last iterate, and
`final_ll_`/`self.ll[-1]` trailing a final-iteration merge is not a
`keep_best` artifact; it happens the same way with `keep_best=False`.

## The written `LLt` is one M-step older than the `W` beside it (issue #157)

`LLt` is the per-timepoint, per-model log-likelihood written alongside the
model, the array EEGLAB's `loadmodout15.m` returns as `mod.Lht`/`mod.Lt`.
pamica writes the values its last E-step computed, which is what the reference
does: `modloglik`/`loglik` are allocated once (amica15.f90:2617-2620), filled
by each E-step (amica15.f90:1406-1411) and dumped verbatim by `write_output`
(amica15.f90:2338-2343).

Fortran's iteration runs `get_updates_and_likelihood` (amica15.f90:996), then
`update_params` (amica15.f90:1122), then `write_output` (amica15.f90:1126 for a
`writestep` checkpoint, 1146 at the end). So the `LLt` on disk belongs to the
parameters as they stood *before* the M-step whose `W`/`A` sit next to it. It
is not the likelihood of the written decomposition; it is the likelihood of its
immediate predecessor, and it is the E-step that produced the last entry of the
written `LL` trajectory.
The exception is a fit that stops on a convergence check:
the reference exits before that iteration's `update_params` (amica15.f90:1111),
and since issue #339 every pamica backend does too,
so the final write's `LLt` is the likelihood of the `W`/`A` beside it.
The relation that holds on both sides, on the
committed reference output as much as on pamica's, is

```
LLt[num_models, :].sum() / (n_good_samples * nw) == LL[-1]
```

It holds bit for bit with one reference-faithful exception: a `do_reject` fit
whose rejection fires on its last iteration, just before the final write. `LL(iter)` is
normalized over the good set as it stood *before* that rejection
(amica15.f90:1770), and `reject_data` then shrinks `numgoodsum`
(amica15.f90:2252) and zeroes the rejected samples' `modloglik`/`loglik`
(amica15.f90:2231-2234), so the two sides stop counting the same samples and a
small residual remains — 0.011 on the bundled sample for a pass that drops 68
of 4096 samples, identical on the two array (PyTorch/NumPy) backends, and of
the same order in the binary itself. MLX shows the same magnitude of residual
under an analogous config (epic #278 Phase 3, #289: ~0.011 measured on a
one-model 4096-sample rejection pass), not an independently-fixed constant
across all three backends' differing block schedules and configs, but the
same reference-faithful mechanism producing residuals of the same order. It
scales with how much that one pass drops. Any later iteration re-normalizes
over the shrunk set and the equality returns exactly. pamica reproduces this
rather than papering over it, and pins both halves as behavior
(`test_llt_invariant_breaks_when_rejection_fires_on_the_last_iteration`
and `test_llt_invariant_returns_one_iteration_after_a_rejection` in
`pamica/tests/test_llt_stash.py`, plus the MLX-specific
`test_llt_invariant_holds_for_a_default_fit`-adjacent tests in
`pamica/tests/mlx_tests/test_mlx_llt_stash.py` and the reject-specific
zero-sentinel pin in `pamica/tests/mlx_tests/test_mlx_export.py`).

pamica adopted this deliberately (2026-08-23, issue #157). Between issues #155
and #157 it instead recomputed `LLt` from the post-update parameters, which
made the file self-consistent with the `W` beside it but *not* comparable with
the binary's — and cost a full extra pass over the data at every write. Being
byte-comparable with the reference is this project's definition of correct, so
the reference's ordering won. The recompute never existed on MLX in the first
place: its LLt stash (epic #278 Phase 3, #289) was ported straight from the
issue #157 design, so there was no earlier recompute to remove there. On the
array backends the recompute's removal also cut ~77 ms per NumPy `writestep`
checkpoint and one full E-step per PyTorch fit on the bundled 32-channel
sample.

The one place pamica has no reference to follow is its `keep_best` safeguard
(row 3 above), which Fortran does not have. There the stashed `LLt` is rolled
back with the parameters, and because the snapshot is taken *before* that
iteration's M-step, the restored parameters and the restored `LLt` come from
the same point in the loop: under a restore there is no staleness at all.
Either way the rule is one sentence — **the exported `LLt` is the E-step that
produced the exported `final_ll_`**.

If you want the log-likelihood of the parameters actually written, compute it:
`model.model_loglik(X)` on the PyTorch or MLX backend returns exactly that
(the MLX accessor was ported in epic #278 Phase 3, #289 -- see
`pamica/tests/mlx_tests/test_mlx_scoring.py`).

## MLX's rejection statistic reads the LLt stash, not a second forward pass (issue #298)

`do_reject` needs one thing `AMICATorchNG`'s LLt stash does not otherwise
provide: the per-sample log-likelihood of the CURRENT good set, at the exact
point in the loop the reject schedule fires. `AMICATorchNG` gets this with a
dedicated `_sample_ll` call -- a second forward pass over the good set,
separate from the E-step that already computed (and stashed) the same
quantity moments earlier. Issue #298 tracks removing that redundant pass on
PyTorch.

The MLX backend (epic #278 Phase 3, #289) never had that redundancy to begin
with: it reads the rejection statistic straight out of `_llt_ll` (indexed by
`good_idx`), the same stash `write_amica_output`'s `LLt` file is built from --
the NumPy backend's design (`numpy_impl/core.py`'s `_last_ll_samples`,
likewise read from its own stash), applied to MLX ahead of PyTorch. This is a
**design decision, not a numeric one**: the two statistics are mathematically
identical (both are the E-step's own `logsumexp` per sample), so MLX and
PyTorch reject the same samples from the same data/config/seed
(`pamica/tests/test_mlx_reject_cross_backend.py::test_reject_same_sample_set_across_backends`
pins this). The difference is purely which backend pays for a second pass over
the good set and which does not -- a WORK divergence, matching the
`do_choose_pdfs`-adjacent `drho_n`/lgamma-table WORK divergences already
documented above, not a correctness one.

## Explicit dimensionality reduction: `pcakeep` and `pcadb` (issue #323)

All three array backends take `pcakeep` and `pcadb`, the explicit PCA reduction,
with the same names, defaults (`None`), validation and precedence, decided once in `pamica/rank.py`.
`pcakeep` keeps that many principal dimensions and is the reference's own keyword:
`numeigs = min(pcakeep, count(eigs > mineig))` (amica15.f90:413, 466),
so a value at or above the channel count keeps every dimension the data have.
`pcadb` is a pamica extension (row 12 above):
the reference parses it (amica15.f90:3459-3461) but never reads it again, so the binary never reduces by it,
while pamica keeps the dimensions whose covariance eigenvalue lies within `pcadb` dB of the largest.

When both are set, `pcakeep` takes precedence and `pcadb` is ignored, with one INFO log line saying so.
That is what the reference does, and it matters in practice:
both bundled parameter files set `pcakeep 32` and `pcadb 30`,
and on the 32-channel sample `pcadb=30` alone would keep 26 dimensions where the binary keeps all 32.
Both are ignored, with one WARNING, when `do_sphere=False`:
reduction happens only while sphering, as in the reference, whose no-sphere branch sets `numeigs = nx` (amica15.f90:527).

A value that cannot mean anything raises `ValueError` at construction on every backend:
`pcakeep` must be an integer of at least 1 (not a `bool`, not a float), and `pcadb` a finite number greater than 0.
The check runs again at fit time, so a value assigned to the attribute after construction fails there.
Before #323 the PyTorch and NumPy backends accepted these silently:
`pcakeep=-3` sliced from the end and fitted 29 of 32 dimensions, `pcakeep=2.7` fitted 2,
and `pcakeep=0` or `pcadb <= 0` ran to a degenerate `nan_ll` fit.
The reference has no such check;
with `pcakeep <= 0` it sets `numeigs = min(pcakeep, ...) <= 0` and carries on, the situation row 2 describes.

## `do_sphere=False` does not scale the channels (issue #328)

This is row 13 of the table above: a known divergence, not a design decision, recorded here until it is resolved.

**What the reference does.**
With `do_sphere 0` the reference does not skip preprocessing.
Its no-sphere branch (amica15.f90:516-526) takes the data covariance and sets
`S = diag(1/sqrt(var_i))` for every channel with positive variance,
so each channel is divided by its standard deviation before fitting,
and it accumulates `sldet = sum_i 0.5*log(S(i,i))` into the likelihood.
`numeigs = nx`, so no dimension is dropped (pamica matches that part; see the previous section).

**What pamica does.**
All three array backends use an identity sphere and `sldet = 0` when `do_sphere=False`,
so they fit the data unscaled.

**Consequences.**
A `do_sphere=False` fit is not trajectory-comparable to the binary's:
the natural-gradient EM is not scale-invariant through its fixed near-identity initialization (unit-norm components, whatever the data's scale) and its step sizes, so the two fit differently scaled data.
The reported log-likelihood also differs by the log-determinant term.
That term looks inconsistent in the reference itself:
`log|det S|` for `S = diag(1/sqrt(var_i))` is `-0.5 * sum_i log(var_i)`,
but the reference adds `0.5*log(S(i,i)) = -0.25 * log(var_i)` per channel, half of that.
Being a constant added to every model's per-sample log-likelihood,
it cancels in the model posteriors and does not change the fitted parameters, only the reported likelihood.
The default `do_sphere=True` path is unaffected.

**How to restore the reference behavior.**
There is no option for it yet.
Issue #328 decides whether to port the per-channel scaling to all three backends
and whether to reproduce the reference's `0.5*log(S(i,i))` term or use the full log-determinant with an escape hatch.

## `mir_step`'s upfront PCA-reduction gate

MIR is undefined on a rank-reduced sphere,
so `fit(mir_step > 0)` checks for PCA reduction in two places, identically on `AMICATorchNG` and `AMICAMLXNG`.
An explicit reduction request is rejected up front with a `ValueError`, before any preprocessing runs.
That means `pcakeep` below the channel count, or any `pcadb`,
since whether it cuts anything depends on eigenvalues that do not exist yet.
`pcakeep` at or above the channel count is not a request and passes,
nor is any request with `do_sphere=False`, which never reduces.
Before #323 the PyTorch gate rejected every explicit `pcakeep`, including the bundled `input.param`'s `pcakeep 32` on 32 channels,
and MLX had no explicit reduction to gate.

Automatic reduction (`mineig`/`mineig_rel` rank detection) cannot be known before the sphere exists,
so it is caught per waypoint instead, by `mir()`'s geometry-based `_pca_reduced()` guard (issue #300):
the first scheduled waypoint records a warned `NaN` entry in `mir_history_`,
later waypoints are skipped, and the fit itself completes.

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

The search is not free: it costs two accumulate passes per candidate, about 16
EM iterations' worth under the defaults (0.54 s on torch-CPU, 2.8 s on NumPy for
the bundled 32-channel sample). Two passes rather than one because the first
pass at a given block size pays one-off costs that belong to no candidate — most
sharply on Metal, where a new block shape triggers shader compilation and
inflated single measurements about fourfold on an M4 Pro. The cost pays for
itself across a normal multi-hundred-iteration fit and does not across a very
short one, which is the other half of why this is opt-in. On the bundled
32-channel sample the block-size curve is flat enough that the win is modest
(torch-CPU picks 16384 for ~1.13x over the 8192 default; NumPy picks 16384 for
~1.03x); the big wins live where the optimum sits far from 8192, most of all
PyTorch-MPS, whose single-block optimum runs ~2.3x faster than the 8192 default
in the validation guide's block-size table.

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

One consequence for earlier NumPy results: on that backend this flag used to
default to **on** (following Fortran's header) while its sweep ran over
128–1024, so every NumPy fit quietly re-tuned itself to a small block and
ignored the `block_size` it was given. It is now off by default there too, and
that backend's default `block_size` is the shipped 8192.

That old search was worse than merely mistuned: because it timed `X.T @ X`,
whose cost grows linearly with block size, it was structurally guaranteed to
pick `blk_min`. Every NumPy fit therefore ran at 128 regardless of what the
sweep bounds said. The place this mattered most is
`test_sample_data_numpy_vs_fortran`, the issue #24 NumPy-vs-Fortran gating
test: it requests `block_size=512` to match the reference `input.param`, but
its historical *effective* block size was 128, so the parity it demonstrated
was never at the size it asked for. That test has been re-verified on the
literal 512 it now actually gets, under the new default, and still passes:
Hungarian-matched component correlation 0.981 (gate > 0.9) and final
log-likelihood −3.4039 after 150 iterations. Parity is unaffected — consistent
with the block-size invariance measured in issue #216 — but the value it runs
at is now the value it specifies.

## Best-of-N restarts have no reference counterpart (issue #198)

`n_restarts=k` runs the fit `k` times from different seeds and keeps the one
with the highest returned log-likelihood. It is available on all three array
backends with the same names and semantics, and it is **off by default**
(`n_restarts=1`), which is the parity-preserving setting: with one restart the
machinery draws nothing, copies nothing and resets nothing, so the trajectory
is bit-for-bit what it was before the feature existed. That bit-identity is
asserted, per backend, in `test_ng_restarts.py`, `test_numpy_restarts.py` and
`test_mlx_restarts.py`.

Fortran has nothing to compare this against. Its `maxrestarts`/`restartiter`
machinery (`amica15.f90:1022-1052`, ported to the NumPy backend as
`numrestarts`) is a *recovery* path — it redraws the mixing matrix after an
early non-finite likelihood and continues the same run — not a search over
seeds, and it never compares two completed fits. So this is a pamica extension,
validated by construction rather than against the binary: `n_restarts=k` must
return exactly the argmax of the same `k` fits run independently, state for
state, which is what the acceptance test in each backend's suite pins.

Why it exists: issue #145 showed the backend and the binary agree to 0.9974
mean component correlation from an *identical* initialization, so the
disagreement seen from *random* inits is basin choice, not a dynamics
difference. The basins that differ are the weakest, under-determined components
(7–10 of them on a 70-channel, k=152 recording), and picking the
highest-likelihood of several seeds is the standard way to stop one unlucky
draw from deciding the result. It extends issue #51's `keep_best` — best
iterate *within* a run — across runs.

Practical notes:

- `n_restarts > 1` requires a base `seed` (or an explicit `restart_seeds` list
  of exactly that length) and refuses to run without one. Best-of-N is only
  meaningful if the winning fit can be reproduced, and seeding from the clock
  would make that impossible. Derived seeds are `seed, seed+1, …`.
- Restarts run **serially**, so a fit costs `n_restarts` times as long.
  Parallel restarts are a possible follow-up, not current behavior.
- Every restart is recorded on the fitted model in `restart_seeds_`,
  `restart_lls_` and `restart_stop_reasons_` (index-aligned, and populated even
  for a single-restart fit), and the winner is named in one INFO log line.
- A restart that ends degenerate is **excluded from the selection but still
  recorded**, with a NaN likelihood. If every restart is degenerate, the model
  is left holding the last one, so the degenerate-fit contract (issue #50)
  applies exactly as it does to a single degenerate fit.
- A restart that **raises** is isolated the same way. A truly singular mixing
  matrix makes the unmixing inversion raise rather than produce a non-finite
  likelihood, and inside a search that must not discard the seeds that already
  worked: the failure is recorded as `stop_reason="restart_error"` (degenerate,
  so excluded from selection, and refused by the same contract if every restart
  hits it), logged with the exception type and message, and the next seed runs.
  Only the multi-restart path catches — `n_restarts=1` still raises exactly as
  it did before this feature existed, because bit-identity includes error
  behavior.
- On the NumPy backend only the winner is written to `outdir` at the end of the
  fit, but the periodic `writestep` checkpoints of every restart pass through
  the same files while the fit runs, so a losing restart's intermediate output
  can appear on disk mid-fit. The final write replaces it — what is on disk when
  `fit` returns is the winner's state, which is a tested claim, not just a
  documented intention.

## Refitting a NumPy instance continues from the previous fit (issue #312)

This is a difference between the backends, recorded in the backend table above,
not a divergence from the reference.

**What happens.**
`AMICA`, `AMICATorchNG` and `AMICAMLXNG` draw a fresh initialization on every `fit`,
so fitting the same instance twice with the same seed reproduces the first fit exactly.
`AMICA_NumPy` does not.
Its `_initialize_parameters` initializes a parameter only while it is `None`,
so a second `fit` on the same instance starts from the first fit's `A`, `mu`, `alpha`, `beta`, `rho`, `gm` and `c`
and from its annealed learning-rate ceilings,
and it appends to the first fit's `ll` and `nd` histories instead of starting new ones.
On the bundled sample, the second fit's starting `A` is exactly the first fit's final `A`,
and three plus three iterations leave six entries in `ll`.
Passing `restart_seeds` (or `n_restarts > 1`) resets this state before each fit, so those paths start fresh.

**Why.**
The initialize-only-if-unset pattern lets the legacy backend honor starting values assigned before `fit`.
It is not a deliberate cross-backend design:
a warm start that works on purpose, on every backend, is the subject of issue #312,
which also covers the reference's own `load_*` warm-start keywords.

**How to get a fresh fit.**
Construct a new `AMICA_NumPy` instance for each fit, as the NumPy command-line interface and the validation harness do,
or pass `restart_seeds=[seed]`.

## Single-precision constants (issue #344)

This section records a parity fix, not a difference.
The reference writes several constants as default-kind Fortran literals, which are single precision, widened to double with `dble`.
The compiler rounds the decimal to float32, so the binary uses that rounding, not the decimal:
`log(dble(2.506628274))` is the log of 2.5066282749176025.
pamica used the decimals' double values until issue #344.
Every backend now uses the reference's values, defined once in `pamica/reference_constants.py`:

| Where | Reference | Value in the binary | Against the value used before |
|---|---|---|---|
| Generalized Gaussian at `rho == 2` | `log(dble(1.772453851))`, amica15.f90:1313 | log of 1.7724539041519165 | 3.0e-8 above `0.5 * log(pi)` |
| Gaussian, `pdftype` 2 | `log(dble(2.506628274))`, amica15.f90:1333 | log of 2.5066282749176025 | 3.7e-10 above |
| Sub-Gaussian cosh, `pdftype` 4 | `log(dble(4.132731354))`, amica15.f90:1359 | log of 4.1327314376831055 | 2.0e-8 above |
| Super-Gaussian cosh, `pdftype` 1 | `log(dble(1.858073988))`, amica15.f90:1371 | log of 1.8580739498138428 | 2.1e-8 below |
| Guard of the rho update, `epsdble` | `1.0e-16`, amica15_header.f90:73, used at amica15.f90:1558 | 1.0000000168623835e-16 | 1.7e-24 above |

The other normalizers, `log(dble(2.0))` and the logistic family's `log(dble(4.0))`, are exact in single precision.
A log-normalizer that is larger by some amount lowers the log-density of its mixtures by that amount.
The default `maxrho = 2` clamps mixtures at exactly 2, so default fits take the first row,
and their log-likelihood and updates move with the responsibility of those mixtures.
Seeded with a warm two-model state that has mixtures at `rho == 2`,
the PyTorch and NumPy updates now match the native binary to float64 round-off
(log-likelihood 2.2e-15 after one iteration and 1.6e-13 after three),
where the previous code was off by 2.8e-9 and 1.2e-8 in log-likelihood and by up to 8.4e-3 in `mu` after three iterations,
in a low-mass mixture whose update amplifies any difference (`pamica/tests/test_component_rows.py`).
Seeded with pamica's initialization for `pdftype` 2, 4 and 1,
the PyTorch log-likelihood matches the binary's to 2.7e-15,
where it was off by exactly each literal's rounding (`pamica/tests/test_reference_constants.py`).
The MLX backend casts the same values to float32;
at `rho == 2` its host-side `lgamma` table holds the reference's normalizer.

Three kinds of single-precision literal keep their decimal values in pamica:

- the defaults of `input.param` keys (row 16 of the table at the top);
- the scales of the random initialization, `0.05`, `0.1` and `0.01` (amica15.f90:756, 771, 814 and 1035):
  pamica draws its initialization from its own random generator, so the reference's cannot be reproduced either way,
  and the parity tests seed the binary with pamica's initialization;
- the starting values of the rejection pass's running maximum and minimum log-likelihood (amica15.f90:2210-2211), which pamica does not need.

`pamica/tests/test_reference_constants.py` pins the sweep of both reference sources that found these.

## Unmapped Fortran keywords

This page's own contract ("anything not on this page is intended to match the
reference") only covers *behavior*. A Fortran keyword that pamica's parameter
translator does not map at all is a separate, narrower question, and it has
its own enumeration: `pamica/fortran_params.py`'s `FORTRAN_UNSUPPORTED_KEYS`
lists every keyword the reference parser accepts that this project does not
translate into a pamica constructor/`fit()` argument, with a one-line reason
for each. That dict, not this page, is the exhaustive list.

Most of those keywords are unmapped because the feature genuinely has no
pamica equivalent (checkpoint warm-start, per-family EM freeze toggles,
console/file-reporting cadence, ...). Three, though, are dead even in the
reference binary itself — the same category as the `do_choose_pdfs` and
`Spinv2` dead code documented above:

- **`filter_length`, `dft_length`** — parsed, printed, and broadcast to every
  MPI rank (`amica15.f90` `case('filter_length')`/`case('dft_length')`,
  ~3263-3280), but no filter or DFT subroutine exists anywhere in
  `amica15.f90`/`funmod2.f90` to consume either value. A pre-filtering
  pipeline was apparently planned and never built.
- **`decwindow`** — parsed into the header the same way, but its
  `MPI_BCAST(decwindow, ...)` call is commented out (`amica15.f90:192`), so a
  multi-process reference run never even shares the parsed value past rank 0;
  every worker rank keeps the compiled-in default regardless of what the
  parameter file said.
- (Already documented above:) `do_choose_pdfs`'s `pdftype=1` auto-switcher is
  declared but its moment buffers (`m2sum`/`m4sum`) are never accumulated, and
  `share_comps`'s `Spinv2` similarity metric is declared but never allocated
  (its scan runs, computes NaN similarities and never merges).

One live (non-dead) divergence is worth recording explicitly rather than
leaving it implicit in the unsupported-keys table: Fortran's `do_rho` can
freeze the generalized-Gaussian shape parameter while keeping `pdftype=0` —
an independent toggle, "use the GG family but do not adapt its shape this
run." Every pamica backend instead hard-derives the freeze from `pdftype`
itself: `AMICATorchNG`/`AMICAMLXNG` set `self.dorho = (pdftype == 0)`
directly (so `pdftype=0` always updates rho, never frozen), and the legacy
NumPy backend has no `pdftype` family switch at all — it always runs the
GG update, the same effective always-on freeze-off behavior. So the
combination Fortran supports (GG family, frozen shape) is unreachable on any
current pamica backend. Restore-reference: none currently exposed; `do_rho`
would need to become an independent constructor keyword to port this on
demand.

Two clusters of keywords are neither dead code nor "no pamica equivalent" —
they are live Fortran features with a real gap, filed rather than silently
carried:

- **Lifecycle** (checkpoint warm-start `load_*`, `writestep`/`do_history`/
  `histstep` periodic checkpointing on torch/MLX, per-family EM freeze
  toggles, `fix_init`): issue #312.
- **Diagnostics** (`write_nd`'s per-component gradient trajectory, the
  per-iteration run log): issue #314.
