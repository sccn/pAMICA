# ADR 0003: Return the best-log-likelihood iterate from AMICATorchNG.fit

**Status:** accepted
**Date:** 2026-07-06
**Owner:** neuromechanist

## Context

Issue #51: on the real sample EEG, multi-model `AMICATorchNG` (`n_models=2`,
100 iterations) reaches a final log-likelihood distribution that is ~0.02 lower
and ~13x more variable than the Fortran reference, even though the per-block
sufficient statistics and one M-step are bit-exact vs Fortran (~1e-15). The
partition distribution is already statistically equivalent to Fortran's (ADR-era
work on #27), so this is an optimizer-quality residual, not a correctness bug.

Diagnosing the variance (NG-only sweep, 20 seeds): the spread is driven almost
entirely by a *late overshoot*, not a bad basin. One seed climbed to LL -3.3573
(dead in the pack) then crashed to -3.5452 in its final two iterations after
Newton went non-positive-definite and fell back to the natural gradient; the
lrate ramp re-inflated the step and the run ended mid-crash. Nine of twenty
"good" runs also ended a small amount below their own peak. The learning-rate
schedule is deliberately non-monotone (both NG and Fortran anneal the rate only
*after* an LL decrease, `amica15.f90:1038-1058`), so the *last* EM iterate is not
guaranteed to be the best one. `fit` returned the last iterate.

## Decision

`AMICATorchNG.fit` tracks the highest-log-likelihood iterate and restores it when
the run ends more than `_KEEP_BEST_TOL` (1e-9, per sample-channel LL) below that
peak. A new `keep_best: bool = True` constructor flag controls it; `final_ll_`
exposes the log-likelihood of the *returned* parameters (while `ll_history` stays
the true per-iteration trajectory, overshoot and all). The safeguard is inactive
under `do_reject`, where the good-sample set (and thus the LL normalization)
changes across iterations and per-iteration LLs are not comparable.

## Consequences

- The pathological low-LL tail is removed and the LL variance collapses toward
  Fortran's: on the 20-seed sample ensemble NG goes from mean -3.3738 (sd 0.040)
  to mean -3.363 (sd ~0.007). See `.context/issue-51/`.
- **Single-model issue #24 parity stays bit-exact.** A monotone fit has its best
  iterate == its last iterate, the gap is 0 < tol, no restore fires, and the
  returned parameters are byte-for-byte identical to `keep_best=False` (verified:
  max parameter difference 0.0 across A/W/mu/beta/alpha/rho/c/gm at 100 iters).
- New obligation: `ll_history[-1]` is no longer the fitted model's LL when a
  restore fired. Consumers must read `final_ll_` (or `max(ll_history)`). The
  validation harness and ensemble scripts were updated accordingly.
- A residual mean gap (~0.009) remains at 100 iterations, but it is *convergence
  speed*, not a worse optimum: at 200 iterations NG reaches Fortran's exact mean
  LL (-3.3541) and by 300 slightly exceeds it. NG's per-iteration progress is ~2x
  slower than Fortran's; the reachable solution is identical (the M-step is
  bit-exact vs Fortran, #27). Not a correctness issue.
- `state_dict` format bumped 2 -> 3 (adds `keep_best`, `final_ll`).

## Alternatives considered

- **Do nothing / raise `max_iter`:** more iterations let a crashed run's annealing
  recover, but waste compute on every run and do not help a fit that already
  peaked and then overshot. Rejected as the primary fix (kept as a knob).
- **Trust-region / reject-the-update in-loop:** clamp any LL-decreasing step. This
  changes the optimization trajectory and risks perturbing the bit-exact
  single-model path; return-best is a pure post-hoc selection that leaves the
  trajectory (and `ll_history`) untouched. Rejected.
- **Rewrite `ll_history[-1]` to the best value:** dishonest (hides the overshoot)
  and breaks the fixed-length trajectory contract. Rejected in favor of a
  separate `final_ll_`.
- **Restore best even on a degenerate (nan_ll) stop:** would salvage finite
  parameters from a diverged run, but entangles with issue #50's degenerate-fit
  contract; the safeguard is skipped for degenerate stops and left to #50.

## Addendum (epic #278 Phase 2, issue #288)

The safeguard now also governs `AMICAMLXNG`: same `keep_best` default, same
`_KEEP_BEST_TOL`, and the same `share_comps`/`do_reject` exclusions (the
latter joined in Phase 3, issue #289) -- see `pamica/mlx_impl/core.py`'s
`_snapshot_params`/`_restore_params` and
`pamica/tests/mlx_tests/test_mlx_keepbest.py`.

## Receipts

- `pamica/torch_impl/amica_torch_ng.py` (`keep_best`, `_snapshot_params`/
  `_restore_params`, `final_ll_`, `_KEEP_BEST_TOL`).
- `pamica/tests/torch_tests/test_ng_backend.py::test_keep_best_*`.
- `.context/issue-51/ensemble_ll.py` (Fortran-vs-NG LL ensemble, real data).
- Fortran schedule: `pamica/amica15.f90:1038-1058` (anneal-on-decrease).
