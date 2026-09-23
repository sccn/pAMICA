# Issue #24 findings: init-basin RULED OUT; residual per-iteration M-step bit-parity bug

**Date:** 2026-07-02. **Status:** decisive. Part of epic #9; follow-up to #21.

## Question

The corrected NG M-step (issue #21) is a faithful *fixed point* (Fortran's converged
solution is stable under it), yet a full fit from the Python numpy init converges to a
worse optimum (LL ~ -3.46, corr ~0.51) than Fortran (-3.402, corr 0.9997). Issue #24 asks:
is the -3.46 gap (i) purely a different random-init basin, or (ii) a residual per-iteration
M-step/Newton bit-parity bug? The test: start **both** implementations from the **same**
init and compare.

## Method

`verify_init_basin.py` seeds the Fortran binary from the Python NG init (seed=42) via
Fortran's `load_*` files (float64 column-major `mean, A, mu, sbeta, rho, alpha, gm, c` under
`indir`), then runs the corrected `CorrectedNG` (from `../issue-21/`) from the identical init
and compares. Both use the `input.param` hyperparameters (lrate 0.05, newt_start 50,
newt_ramp 10, newtrate 1.0, max_decs 3, ...) on CPU/float64.

Two implementation facts made this work:

- **Transpose:** the Python true unmixing is `inv(self.A).T` (= Fortran `W`), so the Fortran
  mixing to load is `A_fortran = self.A.T`. (2026-09-22: a component is therefore a ROW of
  `self.A`'s block, not a column; see ADR 0006 and issue #333 for what that means for
  `doscaling` and `share_comps`.)
- **`load_sphere` is unusable** in the reference binary: its load path never allocates the
  `Stmp2` temp it later dereferences (amica17.f90:553) and segfaults. Worked around by NOT
  loading the sphere and letting Fortran compute its own -- it is the same symmetric ZCA
  (`do_approx_sphere=.true.` default) as `CorrectedNG._preprocess`.

**Seeding is validated:** `max|S_fortran - S_python| = 5.1e-6` and the iteration-0 LL is
**identical** on both sides (`-3.51313`). Both truly start from the same point.

## Result (200 iterations from the identical init)

|            config | Fortran_end | Python_end | corr(F, gold) | corr(P, gold) |
| ----------------: | ----------: | ---------: | ------------: | ------------: |
|           ng_only |     -3.4269 |    -3.4597 |         0.645 |         0.511 |
|  ng_only_pinnedlr |     -3.4269 |    -3.4974 |         0.645 |         0.508 |
|       full_newton |     -3.4018 |    -3.4600 |         0.998 |         0.510 |

`gold` = Fortran's own reference run (`amicaout`, LL -3.40187). `corr` = mean abs
correlation of Hungarian-matched rows of the basis-invariant total spatial filter `W@S`.

Per-iteration LL (full_newton):

```
Fortran : i0=-3.51313 i1=-3.46899 i10=-3.44944 i50=-3.44101 i60=-3.43068 i100=-3.41097 i199=-3.40175
Python  : i0=-3.51313 i1=-3.46952 i10=-3.45828 i50=-3.45972 i60=-3.45981 i100=-3.46004 i199=-3.46000
```

Per-iteration LL (ng_only, lrate pinned at 0.05, ratchet off):

```
Fortran : i0=-3.51313 i1=-3.46899 i5=-3.45548 i10=-3.44944 i20=-3.44542 ... i199=-3.42692  (ascends)
Python  : i0=-3.51313 i1=-3.46952 i5=-3.45890 i10=-3.45828 i20=-3.46280 ... i199=-3.49745  (ascends then DESCENDS)
```

## Conclusion

1. **Init-basin is ruled out (outcome ii).** From the *identical* init, Fortran reaches the
   gold -3.402 (corr 0.998) while Python plateaus at -3.460 (corr 0.51). Same start, different
   endpoint => a **residual per-iteration M-step/Newton bit-parity bug**, not initialization.

2. **The bug is in the first-order natural-gradient M-step, not (only) Newton.** With Newton
   OFF, Python still diverges from Fortran. With lrate *pinned at 0.05* (Fortran's NG value,
   held for its whole NG run), Python tracks Fortran for the first ~5 iterations (i1 matches to
   5e-4), then **reverses and descends** (-3.458 -> -3.497). So the M-step step is not an
   ascent direction of the true likelihood once the parameters move -- a subtle residual, not
   the gross sign error #21 already fixed (that diverged from iteration 1).

3. **The fit-loop lrate ratchet masks the descent.** With the Fortran-matched ratchet ON, the
   same run "plateaus" at -3.4597 instead of descending to -3.497: the ratchet detects the LL
   decreases and collapses lrate toward the floor, freezing parameters at the best-so-far
   point. This is why earlier sessions saw a stable "-3.46/-3.47 plateau" and mistook it for a
   converged basin.

4. **Newton is a secondary blocker.** Fortran's Newton (lrate ramps 0.05->1.0 at iter 50-60)
   breaks through -3.44 -> -3.402. Python's Newton *fires* (0 fallbacks) but does not help,
   because it starts from the already-off-track -3.46 point.

## Next step (localization, follow-up)

Per-iteration M-step diff against Fortran ground truth: run Fortran with `do_history=1`,
`histstep=1` to dump per-iteration state, seed `CorrectedNG` from Fortran's iteration-k state
+ sphere, run one M-step, and diff each parameter delta (dA, dmu, dbeta, drho, alpha, and the
column-scaling / c update) to find which term drifts. Concrete suspects, in order:

- **c update is omitted** in `corrected_mstep_prototype._update_parameters` (Fortran
  `update_c=1`, `c = sum(v*x)/sum(v)`); c is ~0 at convergence but non-zero mid-run.
- **column scaling** (`doscaling`, scalestep=1) coupling with the mu/beta rescale.
- residual **mu/beta/rho** exact-EM term or ordering vs amica17.f90:1890-2014.

This supersedes the handoff's framing that Newton is "the one open gate-blocker": the
natural-gradient M-step itself still has a residual bit-parity gap that must be closed first.

## Reproduce

```
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python .context/issue-24/verify_init_basin.py [run_dir]
```
