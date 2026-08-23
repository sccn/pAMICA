# MLX Newton: float32 go/no-go evidence (issue #264, epic #260 Phase 3)

Verdict: **GO (float32)**.
All four decision criteria pass with two to three orders of margin,
and the fifth (a measurement, not a bar) came back at exactly zero.
No float64-curvature middle path is needed.

Every number below was produced on Apple Silicon (M-series GPU, MLX 0.32.0) from
the repo root of the worktree, on the bundled real sample EEG
(`pamica/sample_data/eeglab_data.fdt`, 32 channels x 30504 samples).

CI does run the MLX tests.
The "Test (macOS 26, Apple Silicon)" job installs MLX and executes `pamica/tests/mlx_tests/` plus
the MLX cross-backend files, so this port is guarded there like any other backend;
only the ubuntu jobs skip them.
What stays local is this gate itself, a multi-minute measurement sweep rather than a test,
so the numbers below are the evidence of record for the float32 decision.
Anything written as a test has to hold on that runner too,
which turned out to be a real constraint: see
"Machine-dependent decrease timing" below.

## How to reproduce

```
uv sync --extra mlx
uv run python .context/issue-264/gate.py
```

`gate.py` implements the five criteria exactly as they were pre-registered in the phase plan,
prints the verdict, and is committed alongside this file so the numbers can be re-derived.
Its `_TracedMLX` subclass only *observes* the conditioning margin;
it recomputes `prod - 1` from the production `_newton_direction`'s own inputs and does not alter the fit.

## G1 -- formula: curvature and M-step vs a float64 PyTorch twin

One MLX (float32) state copied into an `AMICATorchNG` (float64) twin,
so only the arithmetic differs;
`_finalize_newton_stats` and one full `_update_parameters` are then compared.
Bar: max relative error < 1e-4, `A` gap < 1e-4.

| warm-up iters | sigma2 | lambda | kappa | A gap after one Newton M-step |
|---|---|---|---|---|
| 1 | 3.81e-07 | 3.50e-07 | 2.01e-07 | 9.84e-08 |
| 55 | 3.77e-07 | 3.58e-07 | 2.11e-07 | 2.41e-07 |

**PASS**, with ~3 orders of margin, and the agreement does not degrade as the curvature grows
(it is an order of magnitude larger at iteration 55 than at iteration 1).
Both states are positive definite on both backends, so this compares the Newton branch, not the fallback.

Two further float64 comparisons, run as tests rather than as gate criteria:

* two models, disjoint and with a forced shared column (`share_comps` interaction):
  curvature within 4.9e-07 relative, `A` within 1.3e-07, `ndtmpsum` within 5e-08 relative;
* the early 256-sample block, where the curvature is genuinely non-positive-definite
  (min off-diagonal `prod - 1` = -0.96 across 344 of 992 pairs):
  both backends reject, both count exactly one fallback,
  and the MLX fallback step is *bit-identical* to a `do_newton=False` model stepped from the same state.

## G2 -- stability: full-data fits, two schedules, three seeds

150 iterations each on the full 30504-sample recording.
Bar: all likelihoods finite, `stop_reason` not in `("nan_ll", "singular_ll", "nan_params")`.

| config | seed | final LL | stop | iters | fallbacks | min(prod-1) | overshoot |
|---|---|---|---|---|---|---|---|
| default (`newt_start=20`, `newtrate=0.5`, `lrate=0.1`, `block=8192`) | 1 | -3.40612 | max_iter | 150 | 0 | 2.075 | 0.0 |
| default | 3 | -3.40595 | max_iter | 150 | 0 | 2.128 | 0.0 |
| default | 42 | -3.40606 | max_iter | 150 | 0 | 2.074 | 0.0 |
| sample_params (`newt_start=50`, `newtrate=1.0`, `lrate=0.05`, `block=512`) | 1 | -3.40423 | max_iter | 150 | 0 | 1.978 | 0.0 |
| sample_params | 3 | -3.40430 | max_iter | 150 | 0 | 2.039 | 0.0 |
| sample_params | 42 | -3.40389 | max_iter | 150 | 0 | 1.988 | 0.0 |

**PASS**: 6/6 fits finite, non-degenerate, zero Newton fallbacks.
The `sample_params` row is the issue #24 parity configuration,
whose float64 PyTorch counterpart also reports zero fallbacks on this data.

## G3 -- quality: matched 100-iteration budget vs float64 PyTorch Newton

Bar: `mlx.final_ll_ >= torch_f64.final_ll_ - 0.05`,
and zero fallbacks on the `sample_params` config.

| config | MLX float32 | torch float64 (`keep_best`) | torch float64 (last iterate) | delta | MLX fallbacks |
|---|---|---|---|---|---|
| default | -3.41149 | -3.41149 | -3.41149 | +0.00001 | 0 |
| sample_params | -3.41097 | -3.41096 | -3.41096 | -0.00000 | 0 |

**PASS** by ~4 orders of margin:
float32 Newton does not merely stay inside the 0.05 band,
it lands on the float64 likelihood to five significant digits.
The `keep_best` and last-iterate float64 numbers are identical here
because the trajectory is monotone (see G5),
so the comparison is not sensitive to MLX lacking `keep_best`.
For reference, the Fortran binary reaches -3.41125 on this configuration
(`.context/issue-145/setup_and_config.md`).

## G4 -- conditioning: how close the guard came to firing

`min(prod - 1)` over the accepted off-diagonal source pairs,
minimized across every Newton iteration of each fit.
Bar: >= 1e-3.
The float64 reference measurement on this data is 2.09
(`.context/issue-145/setup_and_config.md:67-77`).

Measured: **1.978 to 2.128** across the six fits above.

**PASS**, three orders above the bar, and statistically indistinguishable from the float64 reference.
This is the criterion that would have exposed a float32 curvature problem first:
`prod` is a product of two responsibility-weighted sums of squares
compared against exactly 1.0, with no epsilon margin.
On this data it never comes within 1.9 of the boundary,
which is also why float32 and float64 never disagree about the decision.

## G5 -- overshoot exposure (measurement, not a bar)

`max(ll_history) - final_ll_` per fit, since MLX has no `keep_best` best-iterate safeguard (issue #51).

Measured: **0.000e+00 on all six fits** -- the likelihood is monotone under Newton on this data,
so the final iterate *is* the best iterate.

No `keep_best` follow-up issue is warranted for MLX on this evidence.
The decided policy was to file one if any fit exceeded 0.05;
none came within measurement resolution of it.

## Non-criterion observation: an aggressive `share_comps` cutoff

At `comp_thresh=0.9` on a 2-model fit
-- a cutoff loose enough to merge 26-31 of the 32 cross-model pairs,
i.e. to collapse the two models onto one another --
Newton can drive a component to zero curvature (`sigma2*kappa -> 0`)
and the fit aborts on the `nan_params` guard.
Measured across four seeds, 40 iterations, `share_start=10`, `share_iter=8`:

| seed | MLX float32 | AMICATorchNG float32 |
|---|---|---|
| 1 | `nan_params` (26 shared) | `nan_ll` (26 shared) |
| 7 | `max_iter` (27 shared) | `max_iter` (28 shared) |
| 42 | `nan_params` (27 shared) | `max_iter` (29 shared) |
| 3 | `max_iter` (30 shared) | `max_iter` (30 shared) |

This is a property of the near-total-merge regime rather than an MLX defect:
the PyTorch backend degenerates on the same configuration at seed 1,
and which seeds survive is trajectory chaos
(the two backends do not even agree on how many pairs merge).
At the shipped `comp_thresh=0.99` default, where two genuinely near-collinear pairs merge,
`share_comps` + Newton runs to budget on both backends.
Both failure modes are loud -- the fit stops and reports a degenerate `stop_reason` -- never a silent wrong answer.
Pinned as a documented regime, not as a test.

## Machine-dependent decrease timing (why the schedule tests are data-driven)

The two `newtrate`/`numdecs` ratchet tests need likelihood decreases to fall on particular sides of
`newt_start`, and *when* a fit decreases is BLAS- and hardware-dependent.
This repo already documents the same effect for the `min_dll` stop,
whose firing iteration spans 326-1076 depending on the BLAS build (`docs/guides/validation.md`);
the per-platform breakdown behind that range
-- 326 on macOS-arm64, 412 on Linux-x86_64 with a CUDA-enabled torch build, and 1076 on the GitHub runner --
is recorded at `pamica/tests/torch_tests/test_ng_convergence.py:682-683`.
The first version of these tests hardcoded `newt_start=18` and `newt_start=2`,
which straddled the cadence on the development machine
but went vacuous on the CI Apple-Silicon runner,
failing its own non-vacuity guard.

They now derive their configuration from the executing machine.
A probe fit whose `newt_start` sits past the budget reports where the decreases actually land,
and `newt_start` is chosen from that.
This is sound because the natural-gradient prefix is independent of `newt_start`:
for `it < newt_start` every branch that reads it is false on both sides
(Newton activation, the two `it > newt_start` ceiling ratchets, the `it == newt_start` counter reset),
so the two runs are bit-identical there,
and the likelihood recorded at `it == newt_start` is shared too
because `fit` computes it from the previous iteration's parameters.

Two further changes were needed to make the guards hold rather than merely be checked.
The `newtrate` test was split into complementary halves
-- one run with a ratchet before the gate (suppression), one with `newt_start=0` (admission) --
so neither depends on a single run producing cycles on both sides.
And the fixture's `newtrate` was raised from 1.0 to 2.0:
at 1.0 the post-switch-on trajectory was monotone on larger sample counts,
leaving the ratchet cadence nothing to fire on.
The final configuration was verified over 4 seeds x 2 block sizes x 3 sample counts
-- a deliberately harsher stand-in for cross-machine variation --
with all 16 variants satisfying every assertion and non-vacuity guard.

## Related: the NumPy multi-model Newton broadcast (issue #267)

`numpy_impl/core.py` finalized the curvature with `updates["dgm"][:, None]`,
a `(num_models, 1)` model mass against `(data_dim, num_models)` accumulators.
That broadcasts only when `num_models == 1`,
so *every* multi-model NumPy Newton fit raised
`ValueError: operands could not be broadcast together with shapes (32,2) (2,1)`
on the first iteration Newton was active
-- considerably broader than the `share_comps` collapse the issue reported it from.
Reproduced before the fix on a 3-iteration 2-model fit of the sample EEG, fixed to `[None, :]`
(the torch backend's `dgm.unsqueeze(0)`),
and regression-tested for 1, 2 and 3 models in
`pamica/tests/test_numpy_newton_multimodel.py`.

The MLX port uses the model-major layout, so its equivalent line is `acc["dgm"][:, None]`
-- the same trap on the opposite axis.
`test_multimodel_newton_mstep_is_finite` and the cross-backend multi-model test cover it there.

## Bit-identity of the untouched paths

`do_newton=False` fits are byte-for-byte what they were before this phase.
Verified by running three configurations before and after the core change
(single-model, 2-model, 2-model with `share_comps`; 40 iterations each on 8192 real samples)
and comparing the full likelihood trajectory, the raw bytes of `A`, and `ndtmpsum`:
identical in all three.
