# Issue #265: non-GG PDF families on the MLX backend (epic #260 Phase 4)

## What was done

Ported the four fixed `amica15.f90` source-density families (2 Gaussian, 3
logistic, 4 sub-Gaussian cosh+, 1 super-Gaussian cosh-) and the `pdftype=1`
extended-Infomax adaptive switcher from `AMICATorchNG` (issue #26) into
`AMICAMLXNG`:

- Module-level `_score`/`_log_pdf` (renamed from `_score_gg`/`_log_pdf_gg`)
  take an optional per-source `pdtype` array; `pdtype is None` is byte-for-byte
  the pre-#265 GG-only body, and `_pdtype_h(h)` returns `None` exactly when
  `self.pdftype == 0` -- the fast path that keeps `pdftype=0` fits unchanged.
- A stable `_logcosh(x) = |x| - log(2) + log1p(exp(-2|x|))` (never
  `mx.log(mx.cosh(x))`, which overflows float32 from `|x| >= 90`).
- Constructor validation ported verbatim from `AMICATorchNG` (`pdftype in
  (0,1,2,3,4)`, codes 1/4 require `n_mix=1`, the `kurt_start`/`num_kurt`/
  `kurt_int` schedule validated only under `pdftype=1`).
- `self.dorho = pdftype == 0` gates the `drho_n` accumulator and the
  per-iteration lgamma-table refresh here. `AMICATorchNG` already gates its
  digamma pull behind the same `self.dorho` flag (core.py:1483-1489), so
  that part is not a divergence; its genuine dead work for a frozen non-GG
  `rho` is the `drho_n` accumulation, which it pays unconditionally every
  iteration in `_get_block_updates` (a deliberate MLX-only WORK divergence,
  not a numeric one).
- `_choose_pdfs`/`_pdtype_from_kurtosis`: the switcher's second-moment
  accumulation runs in numpy float64 on the host (pulled from small per-block
  GPU partials), not on the MLX float32 graph -- an MLX-motivated mechanism
  difference from AMICATorchNG with identical decision semantics.
- `get_pdftype(model_idx=0)` accessor.
- `shared_components()`'s docstring now carries AMICATorchNG's caveat: a
  merge does not synchronize `pdtype` across the merged pair.

## pdftype=0 bit-identity check

Compared the epic-tip (`a8401fa`, pre-#265) `AMICAMLXNG` against the new one,
both at `pdftype=0` (the only value the old constructor accepted), on the
bundled sample EEG:

| config | A bit-identical | `ll_history` equal | `final_ll_` |
|---|---|---|---|
| single model, `n_mix=3`, seed=42, 30 iters | **True** (max abs diff 0.0) | **True** | -3.440725326538086 (both) |
| 2 models, `do_newton=True`, `newt_start=3`, seed=7, 15 iters | **True** | **True** | (match) |

Reproduced by loading the old file as a sibling module inside the `pamica`
package (so its relative imports resolve) and running both constructors from
the identical seed; see the cross-backend test's
`test_gg_path_bit_identical_none_vs_code0` for the standing regression pin at
the density-function level.

## y==0 frequency per non-GG family

Policy 4 mirrors AMICATorchNG's `safe_y` substitution (0 for `y==0` lanes in
`ufp/y`) exactly for every family, even though the true `fp/y` limit at `y=0`
is a finite nonzero constant for codes 2/3/1 and 0 (via an `O(y^3)` Taylor
term) for code 4. Measured how often that lane actually fires: ran 15
iterations of the E-step/M-step loop for each fixed family on the full
bundled recording (32 channels x 30504 samples, float32, `block_size=8192`)
and counted exact-zero `y` entries out of every `(sample, channel, mixture)`
triple touched:

| pdftype | family | zero-`y` count | total elements | frequency |
|---|---|---:|---:|---:|
| 2 | Gaussian | 0 | 43,925,760 | 0 |
| 3 | logistic | 0 | 43,925,760 | 0 |
| 4 | sub-Gaussian cosh+ | 0 | 14,641,920 | 0 |
| 1 | super-Gaussian cosh- | 0 | 14,641,920 | 0 |

Zero occurrences across every family on this real recording -- the guard
never fires in practice here (the GG path's own analogous guard fires "<=1
sample per iteration" per the AMICATorchNG issue #75 note, i.e. already rare;
these non-GG families did not trigger it at all in this sweep). This is a
measurement, not a behavior change: the guard stays in place unconditionally.

## Fixed-family float32 (MLX) vs float64 (PyTorch) closed-form agreement

`_score`/`_log_pdf` evaluated in float32 against the literal `amica15.f90`
`z0`/`fp` forms (`_fortran_z0`/`_fortran_fp`, ported from
`torch_tests/test_ng_pdf_families.py`) on the same 65-point grid used there
(`y` in `[-8, 8]`, `|y| > 1e-3`, `rho=1.5`):

| code | family | max abs err (z0) | max abs err (fp) |
|---|---|---:|---:|
| 2 | Gaussian | 8.5e-7 | 0 |
| 3 | logistic | 5.7e-7 | 8.4e-8 |
| 4 | sub-Gaussian cosh+ | 2.0e-6 | 2.5e-7 |
| 1 | super-Gaussian cosh- | 2.0e-6 | 3.7e-7 |

All within `rtol=atol=1e-6` (`np.testing.assert_allclose` semantics: the
relative term dominates at large `|y|`, the absolute term at small `|y|`).
Code 4's `fp = y - tanh(y)` was separately probed on a dense near-zero grid to
confirm the catastrophic float32 cancellation the plan flagged: **100%
relative error at `y=1e-4`** (`fp32` rounds to exactly 0 there; the true value
is `~3.3e-13`), climbing to 4.8% at `y=1e-3` and negligible again by `y=0.25`
(the coarsest grid point the 65-point sweep actually samples past the `1e-3`
filter). The absolute-tolerance criterion absorbs this correctly because the
reference value itself is tiny in that regime; no Taylor-series branch was
added (policy 4 -- parity with the literal formula is the spec).

## Fixed-family matched-budget full-fit LL vs float64 PyTorch (100 iters, seed=3)

| pdftype | n_mix | MLX (float32) `final_ll_` | PyTorch (float64) `final_ll_` | gap |
|---|---|---:|---:|---:|
| 2 | 3 | -3.427496 | -3.427496 | -1.05e-7 |
| 3 | 3 | -3.428832 | -3.428832 | +2.02e-7 |
| 4 | 1 | -3.499060 | -3.499060 | +7.76e-8 |
| 1 | 1 | -3.470514 | -3.470514 | -1.15e-7 |

All four gaps are ~1e-7, four orders of margin inside the
`test_full_fit_ll_matches_torch_float64` gate (`mlx.final_ll_ >= torch -
0.05`).

## Switcher (pdftype=1) trajectory summary

`kurt_start=3, num_kurt=5, kurt_int=1`, 20-iteration fit, seed=0, bundled
sample: `n_kurt_done` reaches `5` (the full schedule runs), and every source
stayed in the super-Gaussian family (`get_pdftype()` unique codes `{1}`) --
the kurtosis of every one of the 32 sources stayed positive throughout this
run, so the sub-Gaussian branch (code 4) was never selected on this real
recording at this seed. This matches the AMICATorchNG note that real EEG
"rarely triggers" the sub-Gaussian switch; the branch itself is pinned
separately by the controlled-input unit test
`test_pdtype_from_kurtosis_decision` (both directions, plus the dead-model
keep-prior guard), since it has no data-driven trigger here. `num_kurt=0`
correctly disables switching (`n_kurt_done == 0`, every code stays at the
code-1 init). LL stayed finite and net non-decreasing in both cases. The
switcher has no bit-exact oracle in the reference binary (`do_choose_pdfs` is
set by `pdftype=1` at amica15.f90:612-613, but the `m2sum`/`m4sum` moment
buffers allocated at :608-609 are never accumulated), so this is
behavior-validated only, exactly as ADR 0002 already scoped for the PyTorch
backend.

## Test gates

- `pamica/tests/mlx_tests/` + `pamica/tests/test_mlx_pdf_families_cross_backend.py`
  + `pamica/tests/test_mlx_sharing_cross_backend.py` +
  `pamica/tests/test_mlx_newton_cross_backend.py` +
  `pamica/tests/torch_tests/test_ng_pdf_families.py`: 133 passed, 5 skipped
  (opt-in `AMICA_RUN_FORTRAN=1` binary-integration tests).
- `uv run pytest -m "not slow" -n auto`: full suite green.
- `uv run ruff check . && uv run ruff format --check . && uv run ty check .`:
  zero diagnostics.
- The slow full-fit-vs-torch tests
  (`test_full_fit_ll_matches_torch_float64`, 4 parametrizations) were run
  once locally: all 4 passed (see the table above for the actual gaps).
