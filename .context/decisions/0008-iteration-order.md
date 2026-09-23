# ADR 0008: Follow the reference's iteration order

**Status:** accepted
**Date:** 2026-09-23
**Owner:** Seyed Yahya Shirazi

## Context

Each iteration of the reference's main loop (amica15.f90:949-1142) computes the likelihood and the update direction,
then responds to a likelihood decrease (halves `lrate`, scales `rholrate`, ratchets the ceilings at `maxdecs`) and runs the stopping checks,
exits before any parameter update when a check fires (:1111),
and otherwise applies the update with the rates it just set (:1122).
All three pamica backends applied the update first and checked afterward,
so a decrease's halved rate took effect one iteration late (issue #339),
and a fit that stopped on a convergence check returned parameters one update past the likelihood it reported.
Separately, the reference holds the `A` update, the `lrate` ramp and the `rholrate` reset
on every iteration with `iter >= share_start` and `mod(iter, share_iter) <= 5` (:1803),
whether or not `share_comps` is on;
pamica applied a freeze only under `share_comps`, anchored on `share_start` (issue #345),
so every default fit of 100 or more iterations diverged from the reference at iterations 100-105.

## Decision

Every backend runs the reference's order:
E-step (likelihood, sufficient statistics, the step for `A` and its norm `ndtmpsum`),
then the non-finite check, the decrease response, `min_dll`, the gradient-norm stop and the Newton-switch reset,
then an exit before any update when a check fired,
otherwise the update with the rates just set, then checkpoints, then outlier rejection.
`_update_parameters` is split into `_update_direction` (the direction and its norms) and the update that applies it, sharing one set of intermediates.
The `A` update holds on exactly the reference's iterations, through `schedule.share_freeze`, for every fit;
`share_iter` (NumPy `share_int` or `share_iter`) below 7 and `share_start` below 1 are rejected in every backend whether or not sharing is on:
the first would hold `A` permanently from `share_start` on, the second would start the freeze on the first iteration.
The rho rate is split into a working rate (`rholrate`, scaled on each decrease and reset to its ceiling inside the `A` branch)
and its ceiling (`rholrate_cap`, ratcheted at `maxdecs`), the reference's `rholrate` and `rholrate0`.
Every backend meets a non-finite value the same way (issue #339 review):
a non-finite likelihood is never recorded in the history;
a non-finite step or gradient norm, which would pass both `<= min_nd` checks, stops the fit before the update (`nan_direction`);
non-finite parameters right after an update stop it at once (`nan_params`).
Both new reasons are degenerate, like `nan_ll`.

## Consequences

- Fits with no likelihood decrease, no convergence stop and fewer than `share_start` iterations are byte-identical to before on every backend.
  Fits with a decrease, a convergence stop, or 100+ iterations at the default share settings change trajectory, toward the reference:
  on a 30-iteration seeded run with eight decreases, the worst per-iteration log-likelihood gap to the binary fell from 1.4e-2 to 2.7e-5 (PyTorch) and 8.2e-5 (NumPy),
  against a binary-versus-binary noise floor (1 thread against 2 or 4) of 4.1e-5 to 1.3e-4 over three runs;
  on a 16-iteration freeze run the gap fell from 6.1e-3 to 6.6e-7, at a floor of 4.0e-7 to 4.3e-7.
- The invariant is that `final_ll_` is exactly the log-likelihood of the returned parameters whenever the fit ends on a convergence stop or a `keep_best` restore,
  not that it equals `ll_history[-1]`:
  on a convergence stop the returned parameters are the ones whose likelihood is `ll_history[-1]`,
  and after a restore they are an earlier iterate's, whose likelihood is an earlier `ll_history` entry.
  At `max_iter` the last iteration still takes its update, as in the reference, so there `final_ll_ == ll_history[-1]` is one update behind the returned parameters.
  A convergence stop no longer runs a share scan, a kurtosis switch, a `mir_history_` waypoint or a rejection pass on its stopping iteration.
- `share_iter` values from 1 to 6, and `share_start` values below 1, that were accepted with sharing off now raise.
  No bundled configuration or parameter file used one.
- A non-finite value that used to be applied or returned now ends the fit as degenerate:
  a NaN step (PyTorch, NumPy and MLX), and non-finite parameters after the last update (PyTorch, which returned them as `max_iter`).
  NumPy's history no longer ends with the NaN likelihood that stopped it,
  its restart-on-NaN takes over a post-update corruption only when `A`/`W` alone went non-finite (a restart redraws only `A`),
  and its restart clears `numincs`, as the reference's NaN comparison does.
- The PyTorch and MLX saves carry `rholrate_cap` as an additive field; a save without it loads with the ceiling equal to its saved `rholrate`.
  Loading refuses a missing or non-finite learning rate with a `ValueError` naming the field.
- The MNE export's `n_iter_` is now `iteration + 1`, the number of E-steps that ran.

## Alternatives considered

- **Apply the halved rate one iteration late, as before, and document it:** the late rate compounds with the chaotic sensitivity of the fit, so the per-iteration gap to the binary grew from round-off to 1e-2 within a few iterations after the first decrease.
- **Keep the freeze anchored on `share_start` and gated on `share_comps`:** pamica's own arithmetic, chosen because the literal `mod(iter, share_iter)` misaligns when `share_start` is not a multiple of `share_iter`.
  The literal arithmetic is unambiguous, costs nothing, and the reference applies it to every default fit that reaches iteration 100, so the anchored window was a divergence on the default path.
- **One rho rate, as before:** the reference scales `rholrate` on a decrease and resets it to `rholrate0` inside the `A` branch; a single variable cannot do both,
  so the pamica ceiling ratchet and working-rate scaling could not both match.

## Receipts

- Issues #339 and #345; epic #324, Phase 11.
- amica15.f90:1019-1111 (checks and exit), :1122 (`update_params`), :1803-1816 (the `A` branch), :1136-1140 (rejection).
- `pamica/tests/test_iteration_order.py` and `pamica/tests/test_nonfinite_stops.py` (always-on, every backend) and `pamica/tests/test_iteration_order_native_oracle.py` (opt-in with `AMICA_RUN_FORTRAN=1`).
