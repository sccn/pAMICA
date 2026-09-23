# ADR 0001: Rewrite the PyTorch backend as a natural-gradient EM port, not Adam+autograd

**Status:** accepted
**Date:** 2026-07-02 (accepted 2026-07-03)
**Owner:** Seyed Yahya Shirazi

## Context

The default PyTorch backend (`torch_impl/amica_torch.py`, and the experimental
`amica_torch_v2.py`) reframes AMICA as "minimize negative log-likelihood with Adam over
reparameterized tensors" (`softmax`/`exp`/`sigmoid` on `alpha`, `beta`, `rho`), driven by
autograd. AMICA is fundamentally an EM / natural-gradient fixed-point iteration with
closed-form sufficient-statistic updates; the Fortran reference (`amica17.f90`) and the legacy
NumPy `pamica.py` implement it that way. The reframing follows a different optimization
trajectory and converges to a different fixed point, which is the primary cause of the poor
component correlation with Fortran (~0.46-0.9 vs target >0.95). It also blocks scaling: the
autograd path materializes all-sample intermediates and retains the graph, so peak memory grows
with recording length and will OOM on realistic 128-256 channel data. Parity with the Fortran
binary is the definition of done for this project.

## Decision

Rewrite the PyTorch backend as a direct, vectorized port of the NumPy/Fortran natural-gradient
(and Newton) update rules. Parameterize `W` directly (no `pinv(A.T).T` per forward), broadcast
the E-step over `(model, mix, source, block)` instead of Python loops, accumulate sufficient
statistics block-wise, and drop Adam/autograd for the parameter updates. The NumPy `pamica.py`
`_get_block_updates` / `_update_parameters` are the reference spec; validation runs in float64.

## Consequences

- **Easier:** parity with Fortran (same trajectory, same fixed point, step-by-step comparable);
  memory scales as O(block) so large recordings and multi-GPU streaming become feasible; the
  Newton path can be ported verbatim from NumPy instead of fighting autograd.
- **Harder:** we lose "free" autograd gradients and must maintain the closed-form updates by
  hand; GPU/MPS kernels must be written as batched tensor ops; the scikit-learn-style public
  `AMICA` interface must be re-pointed at the new module.
- **Obligation:** keep `validate_implementations.py` (real sample data + Fortran binary) green as
  the source of truth; no synthetic-data correctness tests.

**Outcome (2026-07-03, issue #24 / PR #25):** implemented as `AMICATorchNG`
(`torch_impl/amica_torch_ng.py`), wired into `AMICA(backend="ng")`. Single-model reaches Fortran
parity (LL ~ -3.40, Hungarian component correlation ~0.997 with Newton positive-definite, 0
fallbacks); the NumPy backend carries the same fixes. The root-cause fix was the natural-gradient
A-update (it was transposed / multiplied on the wrong side); see
`.context/issue-24/root_cause_Aupdate.py`. Multi-model M-step is bit-exact vs Fortran; full-fit
partition matching is tracked in #27, adaptive-PDF in #26.

## Alternatives considered

- **Keep Adam+autograd, tune harder:** rejected. Even with the three math bugs fixed, Adam on the
  reparameterized surface does not reproduce the natural-gradient fixed point; parity would remain
  coincidental and the memory/scaling problem persists.
- **Promote `amica_torch_v2.py` as-is:** rejected. It carries the same swapped-factorization
  fallback and a Newton path that double-steps (natural-gradient mutation of A plus an Adam step
  on the same iteration), NaN-ing at Newton start.
- **Keep NumPy as the only correct backend:** rejected. It cannot use GPU/MPS and does not scale;
  the project's purpose is a GPU-capable port at parity.

## Receipts

- Fortran LL normalization: `amica17.f90:1866` (`LL = LLtmp2 / (num_samples*nw)`).
- Fortran per-source mixture reduction: `amica17.f90:1313-1360`.
- NumPy reference updates: `pamica.py:505-730` (block updates, natural-gradient + Newton).
- Bug catalog and perf/memory analysis: `.context/research.md` (2026-07-02 design review).

## Addendum (2026-07-04, issue #32)

The superseded Adam/autograd backends this ADR argued against (`AMICATorch`,
`AMICATorchV2`) and their helper modules (`mixture_models`, `optimizers`,
`newton_optimizer`, `adaptive_pdf`, `fortran_output`) were removed. `AMICATorchNG`
is now the sole PyTorch backend, and the public `AMICA` interface wraps it directly
(the `backend=` selector and the basic-backend-only `debug`/`output_dir` fit args
were dropped). The Context/Alternatives sections above describe the pre-removal state
and are retained as the historical record.

## Addendum (2026-09-22, issue #333)

The #24 fix left `A` stored with each model's block equal to the TRANSPOSE of the reference's per-model mixing matrix
(`get_mixing_matrix(h)` returns `A[:, comp_list[:, h]].T`), so a component is a ROW of its block, not a stored column.
[ADR 0006](0006-component-orientation.md) states that convention precisely,
records the `doscaling` defect it caused (stored columns were normalized; fixed in epic #324 Phase 7)
and the planned move to a component-row layout for `share_comps` (Phase 8, issue #334).

## Addendum (2026-09-23, issue #334)

[ADR 0007](0007-component-row-layout.md) replaces the storage this ADR introduced:
`A` is now `(n_comps, n)` with one component per row and `comp_list` indexing rows,
so each model's block, and every result without a `share_comps` merge, is unchanged,
while the share metric and merge act on components as the reference's do.
Saved models convert without loss unless a merge had fired.
