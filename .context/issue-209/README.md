# Issue #209: matched-stopping precision experiment

This directory is the **prep half** of #209 (approved plan: prepare the harness now, run it
on the RTX 4090 box separately, then rewrite `docs/guides/validation.md` from the results).
It contains only the experiment script; **no result files are committed here yet**.

- `precision_experiment.py` -- the experiment harness. Runs the f32-vs-f64 precision contrast
  and the Fortran-vs-torch implementation contrast under matched stopping (issue #209's
  design-decision comment), plus an unconstrained diagnostic that reports whether/when/why
  each precision would have stopped early on its own. See the script's module docstring for
  the full design (the two contrasts, the four-kwarg early-stop suppression, the amica15
  block-size gotcha this script works around).

## The one command to run on the CUDA box

```bash
uv sync
uv run python .context/issue-209/precision_experiment.py --device cuda
```

That uses every default: `budgets=2000,6000`, `seeds=0,1,2`, `k-values=10,20,30` (on the
bundled 32-channel sample, so k tops out at ~29.8, clamped to the 30,504 available frames),
`block_size=8192` (the repo default, #216), `n_mix=3`, `do_newton` off, and it auto-detects
`pamica/sample_data/amica15mac` for the implementation contrast -- which will report itself
unavailable there, since that binary is a macOS Mach-O executable and will not exec on Linux.
That is expected; the precision contrast (the actual ask of #209) does not depend on it. If a
Linux build is wanted for the implementation contrast too, build one first with
`benchmarks/fortran/build_amica.sh` and pass `--fortran-bin path/to/that/binary`.

Results land next to the script:
- `.context/issue-209/precision_experiment_<platform>-<machine>-cuda.json`
- `.context/issue-209/precision_experiment_<platform>-<machine>-cuda.md`

Commit both. If you re-run (e.g. after tweaking `--k-values`), pass `--tag` to avoid
overwriting a previous run's files, or just overwrite -- the tag defaults to
`<platform>-<machine>-<device>`, so repeat runs on the same box replace each other by default.

### Runtime estimate

Rough, not measured on that hardware: AGENTS.md's own benchmark note puts CPU float64 at
~36 ms/iteration on this exact bundled sample at `block_size=8192`, and CUDA float64 at ~4.5x
that CPU number on an RTX 4090 (so roughly 8-10 ms/iteration here; f32 is not reliably faster
per #84). The default sweep is 54 torch fits (18 unconstrained, capped at 6000 iters but most
will stop earlier via the default `min_dll`/`grad_norm` criteria; 36 matched, at fixed 2000 and
6000 iters) plus up to 18 Fortran fits if a working binary is available (CPU-bound, independent
of `--device`). Order-of-magnitude: **well under an hour**, likely 15-30 minutes for the torch
arm alone. Before committing to the full sweep, get an actual per-iteration number on your box:

```bash
uv run python .context/issue-209/precision_experiment.py --device cuda \
    --budgets 100 --seeds 0 --k-values 30
```

`wall_s` in the printed rows (and the JSON) is the per-fit wall time; divide by iterations run
to extrapolate the full sweep.

To shrink the real run instead of just calibrating: `--skip-fortran` drops the CPU-bound arm,
or narrow `--seeds`/`--k-values`/`--budgets`.

### Local CPU smoke check (already done, for reference)

```bash
uv run python .context/issue-209/precision_experiment.py \
    --device cpu --budgets 20 --seeds 0 --k-values 5
```

Runs end-to-end in ~2 seconds on the bundled sample (see the PR description for the captured
output). Not meant to produce numbers worth citing -- 20 iterations and one seed is only a
smoke check that the harness runs and every stop path is actually suppressed on the matched
arm.

## Optional larger-data hook

`--data path/to/data.npy` takes a `(channels, frames)` float array (`np.save`d) for a real
recording larger than the bundled 32-channel/30,504-frame sample -- e.g. the ds002718 sub-002
recording (70 channels, 747,750 frames) used in `.context/issue-90/ksweep_findings.md`, which is
what originally reached k=152. That is not bundled with the repo; supply it explicitly plus
`--channels` if you want to reach k values the bundled sample cannot (max k~=30 at 32 channels).
Real EEG only -- no synthetic-data option, per repo policy.

## Analysis plan (for the follow-up validation.md rewrite)

This PR does not touch `docs/guides/validation.md`; that lands once the CUDA results exist.
What to do with the numbers once they're in `precision_experiment_*.json`:

1. **Read `unconstrained` first.** Per the issue's second comment, whether f32 stops early (and
   at what iteration, via which `stop_reason`) is itself a candidate float32 finding, not noise.
   If f32 reliably hits `lrate_floor`/`grad_norm`/`min_dll` earlier than f64 across seeds and k,
   that is real and belongs in the guide's float32 guidance as its own statement, separate from
   the matched-budget LL/correlation numbers below.

2. **`precision_correlation` + the `precision` rows of `matched`** (implementation=torch,
   f32 vs f64, same seed/init/block_size, both forced to the full budget) are the controlled
   replacement for the confounded numbers at
   `docs/guides/validation.md:278-284` -- the "Why the plateau sits at ~0.98, not 1.0" table,
   which currently reads:

   | Pair (at k=152) | \|corr\| | what it actually varied |
   |---|---:|---|
   | native-Fortran f64 vs PyTorch-CUDA f64 | 0.995 | implementation |
   | native-Fortran f64 vs PyTorch-CUDA f32 | 0.971 | implementation + precision + 265 fewer f32 iterations |
   | PyTorch-CUDA f64 vs PyTorch-CUDA f32 | 0.979 | precision + 265 fewer f32 iterations |

   (root cause: `.context/issue-90/ksweep_findings.md:56` -- the f32 arm hit the lrate floor at
   iter 1735 of a 2000-iter budget and was compared against a full-length f64 run). Report the
   new `mean_abs_corr`/`min_abs_corr` **and** `ll_gap_f32_minus_f64` at matched 2000 and matched
   6000 iterations, per k, so the write-up can say whether the ~0.98 plateau is a real precision
   floor (holds at 6000 too) or was unfinished convergence (closes toward 1.0 at 6000 -- ADR 0003
   / #51 already established the NG backend needs ~2x Fortran's iterations at matched LL).

3. **`implementation_correlation` + the `implementation=fortran` rows of `matched`** are the
   separated implementation contrast (same precision, f64, Fortran vs torch) the issue asks for
   as a second, independent axis -- do not fold this back into the precision table. Caveat: the
   Fortran arm runs at `block_size` clamped to `frames // threads` (amica15 splits frames across
   `max_threads` OpenMP segments before chunking by `block_size`, and a block that spans a
   segment boundary silently produces an all-NaN fit -- see the script's `_fit_fortran`
   comments), while the torch arm keeps the full requested `block_size`. On a many-core CUDA box
   that clamp can land below the validated 512-8192 block-size range (`fortran_block_size` and
   `threads` are recorded in every `implementation_correlation` row, and the run's printed
   summary flags it explicitly), so an implementation gap at a given k could be a block-size
   artifact rather than a genuine implementation difference -- check `fortran_block_size` before
   attributing a low implementation-contrast correlation to Fortran-vs-torch behavior.

4. **The k=30 "definitive" claim** at `docs/guides/validation.md:200-204` ("all agree at 1.000")
   was already a controlled comparison (both arms ran the full 2000 iterations) and is not
   itself confounded; use the new k=10/20/30 (or wider, via `--data`) sweep to say whether it
   still holds once every arm's stopping is *forced* matched rather than happening to match, and
   whether the picture changes anywhere between k=30 and the old k=152 point.

5. **The JOSS paper claim** (`paper.md:119`: "single precision matches double to four or five
   significant digits" on the log-likelihood) is an LL claim, not a correlation claim, so it is
   a different statement from the rows above -- but it has not been checked against a
   matched-budget run either. Cross-check it directly against `final_ll`/`ll_gap_f32_minus_f64`
   in the matched rows before the paper is finalized (per the issue's scope note).

Nothing above is a conclusion -- it is the mapping from "what number" to "what claim it
replaces," to be filled in once the CUDA run's `precision_experiment_*.json` exists.

### Other caveats worth keeping in mind

- CUDA kernel non-determinism is not controlled here (no `torch.use_deterministic_algorithms`), so an anomalous single-seed dip should be checked against a re-run before being attributed to precision rather than run-to-run GPU noise.
- Preprocessing (mean removal, sphering) always runs in float64 on CPU regardless of `--device`/precision (`AMICATorchNG._preprocess`'s own docstring), so the precision contrast is strictly about the EM iterations, not the initial sphere.
