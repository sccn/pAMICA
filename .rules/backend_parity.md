# Backend Parity - No One-Off Implementations

## Core Philosophy: One Algorithm, Four Executions
pamica has four backends (PyTorch, NumPy, MLX, native Fortran) implementing **one**
algorithm. A behavior that exists in one and not the others is not a feature, it is
drift. Users pick a backend for hardware reasons and reasonably expect the same answer.

## [STRICT] Behavior changes land in every backend
A change to what the algorithm *does* must be ported to every backend that supports the
surrounding feature, in the same PR. This is not deferred to a follow-up issue.

- **DO:** Implement in all applicable backends before opening the PR
- **DO:** Add a cross-backend test asserting they agree (see below)
- **DON'T:** Ship "PyTorch first, others later" - the follow-up does not happen, and the
  gap is invisible to users until it bites one

### The narrow exception
A backend may lack a behavior only when it *cannot* support it, and then:
1. The reason is stated in the backend's module docstring
2. The call raises `NotImplementedError` with that reason, never a silent difference
3. The divergence is listed in `docs/guides/amica-differences.md`

Existing legitimate examples: MLX's non-fitting surface gained `transform`,
the mixing/unmixing/`rho` accessors and `state_dict`/`.npz` save-load in epic
#278 Phase 1 (#287); `do_reject`, `keep_best` and LLt/MIR are still absent --
tracked as epic #278 Phases 2/3 (#288/#289). MLX is float32-only because
Apple GPUs have no float64.

"I only had time for one" is not an exception.

## [STRICT] Shared decisions live in shared modules
When backends must agree on a *decision* (not a computation), factor the decision into a
single module they all call. Parallel reimplementations drift silently; a shared function
cannot.

- Reference: `pamica/rank.py` - numerical-rank detection, called by all three array
  backends. Each backend still builds its own sphering matrix in its own array library,
  because the PyTorch path is bit-exact against Fortran and must not be routed through a
  different eigensolver. Only the *policy* is shared.
- Rule of thumb: if the same threshold, ordering, or cap appears in two backends, it
  belongs in one place.

## Cross-backend tests are mandatory
Every shared-behavior change needs a test that fails when the backends disagree.

```python
def test_all_backends_agree_on_the_rank(rank_deficient):
    """Anti-drift guard: every backend must size its model identically."""
    # PyTorch and NumPy always run, so a divergence between them cannot land.
    # MLX uses importorskip - Apple Silicon only.
```

Pattern: `pamica/tests/test_rank_policy.py`. Put these in `pamica/tests/` (not a
backend-specific subdirectory) so it is obvious they are not one backend's problem.

Optional backends (MLX) use `pytest.importorskip`; the always-available backends must be
compared unconditionally, or CI proves nothing.

## Divergence from Fortran is allowed, silence is not
Fortran is the correctness reference, not a ceiling. pamica may improve on it - but every
deliberate difference is recorded in `docs/guides/amica-differences.md` with what it is,
why, and how to restore the reference behavior. An unrecorded difference is
indistinguishable from a parity bug, which is the thing this project most needs to be
able to rule out.

When you change a default away from the reference:
1. Write an ADR (`.context/decisions/`)
2. Add the row to `docs/guides/amica-differences.md`
3. Provide the escape hatch that restores reference behavior, and test it
4. Confirm well-conditioned data is bit-identical, and say so in the PR

## Checklist
- [ ] Implemented in every backend that can support it
- [ ] Shared decision factored into a shared module, not copied
- [ ] Cross-backend agreement test added
- [ ] `NotImplementedError` (not silence) where a backend genuinely cannot
- [ ] Divergences recorded in `docs/guides/amica-differences.md`
- [ ] ADR written if a default now differs from Fortran
- [ ] Full-rank / ordinary-path parity confirmed unchanged
