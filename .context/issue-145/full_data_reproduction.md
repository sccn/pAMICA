# Issue #145 - reproduction on adequate (k~152) full-length data

## Setup
- Data: full ds002718 sub-002, **70 channels x 747,750 frames**, so the
  data-adequacy factor **k = frames / ch^2 ~= 152** (well above the k>=60
  rule-of-thumb; the bundled 32ch fixture is only k=29.8, which is data-inadequate
  and where Fortran itself is not self-consistent - hence not a valid oracle for
  #145).
- Reference: two independent Fortran `amica15` fits (`F_d03`, `F_pdb0`),
  W read F-order (70x70).
- Torch: `AMICATorchNG`, CUDA float64, `do_newton=True`, 2000 iters, seeds
  42 / 7 / 13. All three converged cleanly (0 Newton fallbacks; final LL
  -3.6975 / -3.6978 / -3.6978).
- Comparison: Hungarian-matched `|corr|` of de-normalized W rows.
  Scripts + W's + verdict preserved on hallu at `/mnt/local/taskB/`
  (`VERDICT_full70ch.txt`, `compare_local.py`).

## Result (Hungarian-matched |corr|, 70 components)

| pairing        | n | mean   | min (worst) |
|----------------|---|--------|-------------|
| within-Fortran | 1 | 0.9997 | 0.9983      |
| within-torch   | 3 | 0.9631 | 0.6540 (0.5121) |
| between (t-F)  | 6 | 0.9771 | 0.7697 (0.4822) |

Per torch seed vs Fortran:
- **seed 7:  mean 0.996, min 0.94, 0/70 below 0.9** - matches Fortran.
- **seed 13: mean 0.994, min 0.87, 1/70 below 0.9** - matches Fortran (1 weak comp).
- **seed 42: mean 0.942, min 0.48, 10-11/70 below 0.9** - COLLAPSE (the #145 signal).

## Verdict: the defect reproduces, but it is seed-dependent, not systematic

- **Fortran is a valid oracle here**: two independent Fortran runs agree at
  0.9997/0.9983 (vs only ~0.93/0.68 on the k=30 32ch fixture). So at k~152 the
  data is adequate and self-consistency is meaningful. This validates the issue
  premise and confirms #90's k-factor hypothesis (bulk equivalence rises to ~1.0
  as k grows; it was ~0.90 at k=30).
- **Torch usually matches Fortran** (2 of 3 seeds: ~0.99+ mean, essentially all
  70 components), but **~1 seed in 3 (seed 42) lands in a different basin that
  reshuffles ~10 of the 70 weak/under-determined components** down to |corr| ~0.48.
- **The collapsing seed is at a marginally HIGHER LL** (-3.6975 vs -3.6978), i.e.
  it is a genuinely different, slightly-better-LL local optimum - not a failure to
  converge and not a worse fit. This is the fingerprint of the Fable agent's root
  cause: the Newton step is faithful (~1e-9 vs Fortran); the drift is mu/beta/rho
  exact-EM ill-conditioning on weak components, amplified by Newton's sharpening,
  which occasionally tips a fit into a neighboring basin Fortran's plain-EM path
  does not enter.

## Next step
Implement the faithful mu/beta/rho ill-conditioning guard (keep the Newton step,
stabilize the exact-EM weak-component updates so the Newton-preconditioned path
stays in Fortran's basin), then re-run this exact ensemble on
`/mnt/local/pamica-145` and re-measure - expectation: seed 42 stops collapsing
(all seeds -> ~0.99 vs Fortran, matching within-Fortran self-consistency).

Byproduct already filed: the rholrate reset porting bug (#193, PR #194; same bug
in MLX #195) was found during this investigation but is independent of the
weak-component collapse.
