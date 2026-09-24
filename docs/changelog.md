# Changelog

Release notes are also published on the
[GitHub releases page](https://github.com/sccn/pAMICA/releases).

## Unreleased

MLX becomes a first-class backend, reachable through every wrapper feature (epic #324, completing the raw backend of epic #278),
and every backend's fitting follows the Fortran reference more closely.

!!! warning "Default fits differ from 0.3.3"
    A default fit on any backend (PyTorch, NumPy or MLX) follows a different trajectory than in 0.3.3,
    so its fitted parameters differ, and results computed with an earlier version do not reproduce bit for bit.
    Five changes, each aligning a step with the reference and each described under
    [Fitting follows the reference](#fitting-follows-the-reference-every-backend), reach every default fit:
    `doscaling` normalizes each component's mixing vector, where it used to normalize stored columns (issue #333);
    the drawn initial mixing matrix has unit-norm components (issue #341);
    each iteration runs in the reference's order, so a likelihood decrease takes effect in the same iteration and a convergence stop returns the parameters its likelihood was computed from (issue #339);
    the reference's A-freeze holds the mixing update on iterations 100-105, 200-205, and so on, of every fit (issue #345);
    and the density normalizers are the reference's single-precision constants (issue #344).
    Every pamica entry point now also shares one set of defaults (issue #354, [Defaults and device selection](#defaults-and-device-selection)),
    which moves default fits in three of them:
    an `AMICA()` or `AMICAICA()` fit that sets no `lrate` now runs at 0.1, the backends' default, where it ran at 0.05,
    which raises the final log-likelihood of a default 100-iteration fit on the bundled sample by 0.008;
    a default `AMICA_NumPy` fit now runs without Newton and stops at 100 iterations, like the other backends,
    where it switched Newton on at iteration 20 and ran up to 2000 iterations;
    and a default `AMICANative` run gives the binary these shared settings, where it gave it the bundled `input.param`'s
    (`lrate` 0.05, Newton on, `max_iter` 2000, `block_size` 512).
    The raw PyTorch and MLX backends already used these values.
    Two more reach default fits in narrow cases.
    With schedule gates counted from 1 (issue #335), a `maxdecs` ratchet that completes on iteration `newt_start + 1` tightens the rho-rate ceiling whether or not Newton is on,
    and on NumPy a non-finite likelihood on iteration `restartiter + 1` ends the fit.
    With components stored as rows (issue #334), the gradient norm `ndtmpsum` is summed per component,
    which moves it by float round-off and can change a gradient-norm stop that falls within that round-off of `min_nd`.
    Fits with Newton, outlier rejection or merging `share_comps` change further through the same two issues.
    Seeded from the same initialization, `A`, `mu` and `sbeta` now match the native binary to float64 round-off over the first iterations;
    before, they departed from its trajectory on the first iteration.
    Scale-blind results, such as the log-likelihood, the component maps and the matched correlation with the reference, moved by small amounts in the measurements reported below.
    To compare parameters element by element with the reference or with an earlier fit, refit.
    Saved models load, converted where the storage changed;
    `share_comps` models in which components had merged are refused and must be refit
    ([Persistence](#persistence-and-exports)).

### Fitting follows the reference (every backend)

- **The drawn initial mixing matrix has unit-norm components, as in the reference** (issue #341, epic #324 Phase 12).
  The reference draws each model's block of `A` as `0.01 * (0.5 - u)`, sets its diagonal to one
  and divides every component by its Euclidean norm (amica15.f90:805-823);
  its restart after a non-finite likelihood redraws the same way (:1026-1044).
  pamica drew `I + 0.01 * (0.5 - u)` and never normalized it,
  so its initial components had norms up to about 5e-3 away from one.
  Every backend (PyTorch, NumPy and MLX) now draws its initial `A` through one shared function,
  `pamica.initialization.initial_mixing`, which follows the reference's recipe,
  and the NumPy restart after a non-finite likelihood redraws through it too.
  The generator is called exactly as before, so `mu` and `beta` start from the same values.
  A supplied or loaded `A` is used as is, as the reference uses a loaded one (:793-802):
  an `A` set on a NumPy model before `fit`, a NumPy refit, a PyTorch `state_dict`, a saved `AMICA` model and an MLX save.
  - **Behavior change: every fit from a drawn `A` starts from a different point.**
    Normalizing is not a compensated rescale, so it changes the first E-step and the trajectory after it.
    With `doscaling` on (the default), the first rescale used to normalize the components after the first iteration anyway,
    so default fits end close to where they did:
    on the bundled sample (PyTorch, seed 42, 100 iterations), the first log-likelihood moves by 2.7e-5 with one model and 1.7e-6 with two,
    and the final one by 1.8e-6 and 4.5e-4.
    With `doscaling=False` the initial scale was never corrected, so the change reaches the whole fit
    (the final log-likelihood moves by 6.6e-7 and 9.2e-4).
    When this change landed, the validation harness (100 iterations, all three backends, against the native binary from its own initialization) was unchanged at its precision:
    mean matched correlation 0.9991 and Amari distance 0.0038 before and after,
    and a log-likelihood gap of 2.7e-4 to 2.8e-4 (2.6e-4 to 2.7e-4 before).
  - **Behavior change (NumPy backend): a supplied initial `A` is checked at fit start.**
    An `A` set before `fit`, or left by a previous fit on the same instance, now raises `ValueError` naming the problem
    when its shape is not `(num_comps, n_channels)` (one row per component), when it holds a non-finite entry,
    or when a model's block is numerically singular.
    Before, a wrong shape or a singular block raised `LinAlgError` from deep in the fit,
    and a NaN entry ended the fit with no stop reason.
    The PyTorch and MLX backends take an `A` only through their saves, which are already validated.
  - The draw itself still cannot match the reference's, whose generator is gfortran's `random_number`.
    Two new gated oracles check the two halves instead:
    the binary's own drawn initialization (written by a run with `max_iter=0`) has unit-norm components with the recipe's shape,
    and the binary, loaded with pamica's draw before normalization and with its `A` update off,
    normalizes it to pamica's initial `A` within 4.4e-16.
  - Tests: `pamica/tests/test_initial_mixing.py` checks the recipe, the start of every fit on every backend
    (with a control that fails on the code before this change), the supplied and loaded paths and the refused ones, NumPy's `fix_init`, the NumPy restart redraw, and the two gated oracles.
    The byte-identity tests that pin earlier changes against older commits now start the older code from the new initial `A`
    (`pamica.tests.pre_change.with_normalized_initial_mixing`), while the live side keeps its pre-#344 constants,
    and still pass bit for bit.
    Data-driven tests whose trajectories moved were re-searched or re-recorded:
    the MLX `pdftype=1` restore recipe, the MLX MIR restore test, the MLX fit-path canary, the seed of the `doscaling` native oracle, and the early-merge collapse oracle.
- **Every backend follows the reference's iteration order** (issues #339 and #345, epic #324 Phase 11).
  Each iteration of every backend (PyTorch, NumPy and MLX) now runs in the reference's order (amica15.f90:949-1142,
  [ADR 0008](https://github.com/sccn/pAMICA/blob/main/.context/decisions/0008-iteration-order.md);
  [How AMICA works](concepts/how-amica-works.md#the-algorithm) walks through it):
  the E-step computes the log-likelihood and the update direction,
  then the likelihood-decrease response and the stopping checks run,
  a fit that stops leaves before any update,
  and otherwise the parameters are updated with the rates those checks just set.
  Fits with a likelihood decrease, fits that stop on a convergence check, and fits that reach `share_start` (iteration 100 by default) now end elsewhere, closer to the reference;
  every other fit is byte-identical to the code before this change.
  - **Behavior change: a likelihood decrease takes effect in the same iteration's update.**
    Before, the halved `lrate` and the scaled rho rate reached the update one iteration late.
    On a seeded 30-iteration run with eight decreases (`lrate=0.5`, no Newton, `doscaling` on),
    the largest per-iteration log-likelihood gap to the native binary fell from 1.4e-2 to 2.7e-5 (PyTorch) and 8.2e-5 (NumPy),
    and the largest gap in `A` from 0.22 to 8.1e-4 and 6.8e-4;
    the binary against itself (1 thread against 2 or 4, whose reduction order is not reproducible from run to run)
    differs by 4.1e-5 to 1.3e-4 in log-likelihood and 5.3e-4 to 1.2e-3 in `A` over three runs.
    On a run that ratchets at `maxdecs`, the log-likelihood gap fell from 1.3e-2 to 6.6e-6 (PyTorch) and 8.6e-6 (NumPy),
    and the gap in `A` from 0.14 to 3.0e-4 and 2.7e-4 (binary floors 5.7e-6 to 9.6e-6 and 1.4e-4 to 4.2e-4).
    Every decrease now falls on the reference's iterations.
  - **Behavior change: a fit that stops on a convergence check returns the parameters its `final_ll_` was computed from.**
    On a `min_dll`, gradient-norm or `lrate`-floor stop, the stopping iteration used to take its update anyway,
    so the returned parameters were one update past `final_ll_`, and the exported `LLt` one update behind them.
    That iteration now also runs no kurtosis switch, share scan, `mir_history_` waypoint or rejection pass.
    A fit that runs to `max_iter` still updates on its last iteration, as the reference does.
  - **Behavior change: the reference's A-freeze applies to every fit.**
    Once `iter >= share_start`, the reference holds the `A` update, its `lrate` ramp and its rho-rate reset
    on every iteration with `mod(iter, share_iter) <= 5` (amica15.f90:1803), whether or not `share_comps` is on.
    pamica held `A` only under `share_comps`, in a window counted from `share_start`.
    So every default fit of 100 or more iterations now holds `A` on iterations 100-105 (and 200-205, and so on), as the reference does.
    On a seeded 16-iteration run with `share_start=3` and `share_iter=10`,
    the gap to the binary fell from 6.1e-3 to 6.6e-7 in log-likelihood (binary floor 4.0e-7 to 4.3e-7) and from 0.19 to 5.0e-6 in `A`.
  - Parity on the bundled sample when this change landed, before and after, on all three backends:
    against the bundled 200-iteration reference output, the log-likelihood gap fell from 2.2e-4 to 2.3e-4 down to 1.2e-4 to 1.3e-4,
    the mean matched component correlation rose from 0.9972-0.9973 to 0.9982-0.9983 (minimum 0.969-0.970 to 0.981-0.982),
    and the Amari distance fell from 6.3e-3 to 4.8e-3.
    The validation harness's 100-iteration run kept its final log-likelihood (gap 2.6e-4 to 2.7e-4),
    and its mean matched correlation rose from 0.9988 to 0.9991 (Amari distance 0.0044 to 0.0038),
    because the reference holds `A` on its 100th iteration.
  - **Behavior change: every backend stops the same way on a non-finite value.**
    A non-finite log-likelihood is never recorded: the NumPy backend used to leave it as the last `self.ll` entry,
    which PyTorch and MLX never did.
    A non-finite update direction or gradient norm, which passes both `<= min_nd` checks, now stops the fit before the update
    with the new degenerate `stop_reason` `"nan_direction"`; it used to be applied.
    Non-finite parameters right after an update now stop PyTorch and NumPy as they stopped MLX (`"nan_params"`, same check and message),
    so a corruption on the last iteration no longer ends as `max_iter`.
    The `AMICA` wrapper treats both new reasons as degenerate (`converged_=False`, output refused).
    The NumPy backend names them in its own prose vocabulary and reports `converged=False`;
    inside its restart-on-NaN window it lets a restart take over only when `A`/`W` alone went non-finite, which a restart redraws.
    Its restart also clears the small-gain count `numincs`, as the reference's NaN comparison does.
  - **New validation:** `share_iter` (NumPy `share_int` or `share_iter`) must be an integer >= 7,
    and `share_start` an integer >= 1, on every backend whether or not `share_comps` is on:
    a `share_iter` below 7 would hold `A` permanently from `share_start` on, and `share_start=0` would start the freeze on the first iteration.
    The same checks apply through a Fortran params file.
    No bundled configuration used a smaller value.
    Loading a PyTorch `state_dict` or an MLX save refuses a missing or non-finite learning rate with a `ValueError` naming the field.
  - The rho learning rate is now two values, as in the reference:
    the working rate `rholrate`, scaled on each decrease and reset to its ceiling by each `A` update,
    and the ceiling `rholrate_cap`, ratcheted at `maxdecs`.
    PyTorch and MLX saves store `rholrate_cap`; a save without it loads with the ceiling equal to its `rholrate`.
  - Smaller consequences of the order:
    the MNE export's `n_iter_` is now `iteration + 1`, the number of E-steps that ran (it was `max(iteration, 1)`);
    an MLX fit that stops on non-finite parameters now records that iteration's log-likelihood;
    NumPy's restart after a non-finite likelihood now happens before the update, and its outlier rejection after the checkpoint writes, as in the reference.
  - Tests: `pamica/tests/test_iteration_order.py` checks the order, the decrease timing, the stop semantics of every convergence stop and the freeze through real fits on every backend,
    `pamica/tests/test_nonfinite_stops.py` the non-finite stops by injection,
    and `pamica/tests/test_iteration_order_native_oracle.py` (opt-in, `AMICA_RUN_FORTRAN=1`) is the native-binary comparison above.
- **`doscaling` rescales components, as the reference does** (issue #333, epic #324 Phase 7).
  **Behavior change:** default fits on every backend (PyTorch, NumPy and MLX) now follow the reference's trajectory,
  so their fitted parameters differ from those of earlier versions.
  Each model's mixing block was stored transposed relative to the reference (the issue #24 convention, ADR 0006),
  so a component was a row of the stored block,
  but `doscaling` (on by default) normalized stored columns, which is not a change of scale of any component and perturbed every iteration.
  It now divides each component's mixing vector by its norm and rescales that component's `mu` and `beta` to match,
  an exact change of scale that leaves the log-likelihood unchanged.
  Since issue #334 (below) a component is a row of the stored `A` itself.
  - Seeded from pamica's initialization, `A`, `mu` and `sbeta` now match the native reference binary to float64 round-off
    (after 1 iteration: `A` 5.0e-16, `mu` 7.8e-11, `sbeta` 1.1e-14, previously 7.2e-5, 6.6e-5 and 8.6e-5;
    after 3: 2.7e-13, 8.6e-10 and 2.7e-11, previously 1.4e-3, 6.7e-3 and 2.2e-3).
  - Fitted components now have unit norm, as in the reference;
    previously, after 100 seeded iterations, norms ranged over [0.94, 1.07] with one model and [0.05, 1.97] with two.
  - Scale-blind results barely moved when this change landed: against the bundled `amicaout` fixture after 200 iterations,
    the log-likelihood went from -3.401777 to -3.401673 (fixture: -3.401873),
    the matched correlation from 0.99752 to 0.99740 and the Amari distance from 5.95e-3 to 6.14e-3.
    Two-model fits improve most: after 100 seeded iterations, the matched correlation with the reference rose from 0.850 to 0.99998.
  - `doscaling=False` was byte-identical to the code before this change on every backend.
    Saved models load unchanged; refit only to compare parameters element by element with the reference.
  - `scalestep`, which the reference parses but never reads (it rescales every iteration), stays a pamica extension
    but now counts from 1: the rescale runs on iterations `scalestep`, `2*scalestep`, and so on, instead of 1, `1+scalestep`, and so on.
    The default of 1 is unaffected (row 14 of the [differences guide](guides/amica-differences.md#at-a-glance)).
    With `doscaling` on, every backend's constructor now raises `ValueError` for a `scalestep` that is not an integer of at least 1;
    `scalestep=0` used to fail mid-fit with a bare `ZeroDivisionError`.
  - New test helper `pamica/tests/native_oracle.py` seeds the native binary from a pamica state through its `load_*` files,
    for element-wise oracle tests (opt-in with `AMICA_RUN_FORTRAN=1`).
- **Iteration schedules count from 1, as the reference's do** (issue #335, epic #324 Phase 9).
  **Behavior change:** `newt_start`, `rejstart` and (on NumPy) `restartiter` now name iterations counted from 1, like the reference's `iter` (amica15.f90:949):
  the first Newton M-step is the `newt_start`-th iteration's, the first rejection follows the `rejstart`-th,
  and a non-finite likelihood restarts the fit only within the first `restartiter` iterations.
  The PyTorch, NumPy and MLX backends compared their 0-based loop index with these 1-based settings,
  so each of the following fired one iteration late or, for the restart window, covered one iteration too many:
  - the Newton switch (`iter .ge. newt_start`), the decrease-counter reset on the switch-on iteration (`iter == newt_start`),
    and the `maxdecs` ratchet of the rho-rate ceiling and of `newtrate` (`iter > newt_start`);
  - the outlier-rejection schedule (`iter == rejstart`, then every `rejint` iterations);
  - the NumPy backend's restart-on-NaN window (`iter .le. restartiter`).

  Measured against the pinned v0.3.3 native binary, seeded with pamica's own initialization, `doscaling` off and single-threaded:
  with `newt_start=3`, the first 6 iterations now match to round-off,
  the log-likelihood within 5.4e-11 (PyTorch) and 2.1e-12 (NumPy) and `A` within 4.2e-10 and 1.1e-10,
  where they deviated by 1.25e-3 and 3.1e-2 before.
  Over 100 iterations with the reference's own `newt_start=50`, the largest log-likelihood deviation drops from 1.33e-3 to 6.7e-6 (PyTorch) and 4.1e-6 (NumPy).
  That is the floor this recording sets before Newton even starts:
  one mixture component's shape sits at `rho=1`, where the location update divides by `|y|` and amplifies round-off from one iteration to the next.
  With `doscaling` on as well (the default, component rows since issue #333), the same 100-iteration comparison stays within 3.9e-6 (PyTorch) and 7.0e-6 (NumPy);
  with only the #333 fix it was 2.1e-4 apart, the gap ADR 0006 attributed to this Newton start.
  - Default fits (`do_newton=False`, `do_reject=False`) change in two narrow cases only.
    A `maxdecs` ratchet that completes on exactly iteration `newt_start + 1` (21 by default) now tightens the rho-rate ceiling, as the reference's does.
    On NumPy, a non-finite likelihood on iteration `restartiter + 1` (11 by default) now ends the fit instead of restarting it.
  - To reproduce a trajectory from before this change, add 1 to `newt_start` and `rejstart` (and to `restartiter` on NumPy).
  - **New validation**, with the same message on every backend: `newt_start` must be an integer >= 0 whether or not `do_newton` is on
    (it also gates the rho-rate ratchet), and `rejstart` an integer >= 1 when `do_reject` is on
    (with 1-based counting, `rejstart <= 0` silently skipped the reference's unconditional first pass; NumPy used to accept 0, PyTorch and MLX any value).
    On NumPy, `restartiter` and `maxrestarts` must be integers >= 0, and `histstep` an integer >= 1 when `do_history` is on
    (`histstep=0` was a bare `ZeroDivisionError` mid-fit).
    `restartiter=0` disables restart-on-NaN, as in the reference.
    NumPy's restart-on-NaN recovery itself differs from the reference's, which never resumes fitting after a restart;
    that is now recorded as row 15 of the differences guide.
  - Every schedule gate now lives in one shared module, `pamica/schedule.py`, which all three backends call.
    The share-merge, A-freeze, kurtosis-switch and `writestep`/`histstep` schedules already counted from 1 and are unchanged,
    and `scalestep` (1-based since issue #333) uses the same helper and validator.
  - `iteration`, `ll_history` and `mir_history_` keep their 0-based indexing.
  - Tests: `pamica/tests/test_schedule_gates.py` observes each gate through real fits on all three backends,
    and `pamica/tests/test_schedule_native_oracle.py` (opt-in, `AMICA_RUN_FORTRAN=1`) is the native-binary comparison above.
- **`share_comps` compares and merges components; `A` stores one component per row** (issue #334, epic #324 Phase 8).
  Every backend (PyTorch, NumPy and MLX) now stores the mixing matrix with one component per row,
  `A` of shape `(n_comps, n_channels)`, the reference's `A` transposed ([ADR 0007](https://github.com/sccn/pAMICA/blob/main/.context/decisions/0007-component-row-layout.md)).
  A component id in `comp_list` now names the same component in `A` as in the density parameters,
  so the share metric compares the components' scalp maps (`get_sensor_mixing_matrix`)
  and a merge ties the two components' mixing vectors and densities, as the reference's `identify_shared_comps` does.
  Before, both steps used stored columns of each model's block, which are not components.
  - **Behavior change: fits with `share_comps=True` in which a merge fires now differ.**
    Refit them.
    Seeded with a merged `comp_list` through the reference's `load_comp_list`,
    the PyTorch and NumPy updates match the native binary to float64 round-off
    (after 3 iterations, worst of `doscaling` on and off: `A` 1.5e-12, `mu` 3.7e-9, log-likelihood 4.7e-14),
    where the previous code was off by 0.21 in `A` and 4.3e-4 in log-likelihood.
    On the bundled sample (2 models, 300 iterations, `share_start=100`, `comp_thresh=0.95`)
    the scan now merges three pairs whose maps agree (|cos| 0.956 to 0.971), ending at log-likelihood -3.3416,
    where it merged pairs whose maps did not (|cos| 0.06, 0.35 and 0.55) and ended at -3.3484 (-3.3387 with sharing off).
    Because the metric now sees how similar the two models still are early in a fit, a scan in the first iterations merges most components,
    and a model left with few components of its own can then collapse.
    The reference behaves the same way: its similarity on its own early state merges the same pairs,
    and its update from the same merged states collapses in step (its own scan never merges, because its similarity is NaN).
    The reference's default `share_start=100` avoids that.
  - Every fit without a merge is byte-identical to the code before this change on every backend, sharing off or scheduled but not firing,
    with one exception at float round-off: the weight-gradient norm (`ndtmpsum`), which now sums per component like the reference.
  - Persistence and the EEGLAB `A` file change with the layout; see [Persistence and exports](#persistence-and-exports).
- **Every backend uses the reference's single-precision constants** (issue #344, epic #324 Phase 13).
  The reference writes several density normalizers as default-kind Fortran literals widened with `dble`, for example `log(dble(2.506628274))`,
  so the binary uses the float32 rounding of each decimal, not the decimal.
  pamica used the decimals' double values.
  The PyTorch, NumPy and MLX backends now take the reference's values from one module, `pamica/reference_constants.py`
  ([the differences guide](guides/amica-differences.md#single-precision-constants-issue-344) has the table).
  - **Behavior change: fits in which a mixture reaches `rho == 2` move slightly, toward the reference.**
    The default `maxrho = 2` clamps mixtures there, where the reference's normalizer, `log(dble(1.772453851))`, is 3.0e-8 above the `0.5 * log(pi)` pamica used,
    so default fits of the generalized Gaussian take this branch once a mixture reaches the clamp
    (on 4096 samples of the bundled recording, a two-model fit gets there on its fifth iteration).
    Seeded with a warm two-model state that has mixtures at `rho == 2`, the PyTorch and NumPy updates now match the native binary to float64 round-off:
    after three iterations the log-likelihood differs by 1.6e-13, `A` by 3.8e-13 and `mu` by 5.7e-9,
    where the previous code was off by 1.2e-8, 3.4e-8 and 8.4e-3.
  - **Behavior change: the Gaussian (`pdftype` 2) and the sub- and super-Gaussian cosh families (`pdftype` 4 and 1) report a different log-likelihood**,
    lower by 3.7e-10 and 2.0e-8 and higher by 2.1e-8, which now matches the binary's to 2.7e-15.
    Their parameter updates move only by round-off, since a family's normalizer shifts every mixture alike.
  - The underflow guard of the rho update, `epsdble`, is likewise the reference's `1.0e-16` in single precision (1.0000000168623835e-16).
  - The NumPy plotting helper `pamica.numpy_impl.pdf.compute_pdf`, which `viz.plot_pdf_fits` draws, takes its normalizers from the same module,
    so it draws the density the fit uses; its unused companion `compute_log_pdf` is removed.
  - Row 16 of the differences guide records the one kind of single-precision literal pamica keeps at its decimal value:
    the compiled-in defaults of `input.param` keys, which the binary uses only when the key is missing
    (then its `comp_thresh` is 0.9900000095); a value given in `input.param` is read as double, and pamica's native engine gives every one.
  - Tests: `pamica/tests/test_reference_constants.py` pins every constant against the float32 rounding of its literal on the cited reference line,
    computed by exact rational arithmetic; pins the sweep of both reference sources; checks that no backend keeps its own copy;
    and (opt-in, `AMICA_RUN_FORTRAN=1`) seeds the native binary for `pdftype` 2, 4 and 1.
    The merged-state oracle in `pamica/tests/test_component_rows.py` no longer holds `maxrho` at 1.99.
    The tests that compare a live backend bit for bit with code from before this change
    (`test_component_rows.py`, `test_doscaling_rows.py` and `mlx_tests/test_mlx_fit_noop.py`)
    give the live backend its old constants first, so they still isolate the change they were written for.

### MLX as a first-class backend, through every wrapper

- **Backend selection in `AMICA` and `AMICAICA`** (issue #313, epic #324 Phase 4).
  `AMICA` and `AMICAICA` gain a `backend` parameter:
  `"torch"` (the default, `AMICATorchNG`, float64 Fortran parity) or `"mlx"` (`AMICAMLXNG`, Apple GPU, float32 only).
  Every wrapper feature runs on MLX:
  `fit` with any backend keyword (including `pcakeep`), `from_params_file(..., backend="mlx")`,
  the #50 degenerate-fit contract, `save`/`load`, the EEGLAB export and the MNE path.
  An unknown backend raises `ValueError`, `backend="mlx"` without MLX installed raises `ImportError` at construction,
  and `device` or a `dtype` fit keyword with `backend="mlx"` raises `ValueError`, since MLX runs only on its default device in float32.
  `import pamica` still never imports MLX.
  The [backends guide](guides/backends.md#selecting-a-backend) has the rules and an Apple Silicon workflow.
  - The keywords `fit` takes from `**kwargs` and from a params file are derived from the selected backend class's own signature,
    and the "not applied" warning names that class.
    A keyword the selected backend does not take now raises `TypeError` from `AMICA.fit` itself, naming the backend,
    instead of from the backend constructor;
    every offending keyword of one call is named in that one error (issue #346 review).
  - The degenerate-fit contract uses each backend class's own `_DEGENERATE_STOP_REASONS`,
    so an MLX fit that stops on `nan_params` is refused like a PyTorch `nan_ll` fit.
  - **New accessors:** `get_sphere()`, `get_mean()` and `get_model_center(model_idx)` on `AMICATorchNG`, `AMICAMLXNG` and `AMICA`,
    with the same names and shapes on both backends and float64 arrays from both,
    guarded against degenerate fits like the other accessors (issue #306, below).
    `AMICA` also gains `get_sensor_mixing_matrix()`, which both backends already had.
  - `AMICAICA` reads the fitted mean, sphere and centers through those accessors, with no backend-specific array calls.
    An MLX export is float32-consistent (sources agree with the MLX `transform` within float32 tolerance),
    while `apply` with nothing excluded still returns the input to float64 round-off.
    A degenerate `AMICAICA` fit now leaves `pca_components_`/`pca_explained_variance_` as `None`; it was never exportable.
- **Explicit `pcakeep`/`pcadb` on MLX, with one validation policy for every backend** (issue #323, epic #324 Phase 1).
  `AMICAMLXNG` gains `pcakeep` and `pcadb` with `AMICATorchNG`'s names, defaults (`None`), position, validation and precedence.
  They go through the shared `pamica.rank` policy, so all three array backends keep the same rank and build the same sphere
  (cross-backend test on the bundled sample: torch vs NumPy sphere within 1e-10 relative, MLX within float32 rounding).
  `fit(mir_step > 0)` gains the same upfront reduction gate and message as the PyTorch backend.
  - **Behavior change: invalid `pcakeep`/`pcadb` raise `ValueError` at construction on every backend.**
    `pcakeep` must be an integer of at least 1 (a `bool` or a float is rejected) and `pcadb` a finite number greater than 0;
    a value assigned to the attribute after construction fails at fit time.
    The PyTorch and NumPy backends used to accept these silently:
    `pcakeep=-3` sliced from the end and fitted 29 of 32 sources on the bundled sample, `pcakeep=2.7` truncated to 2,
    and `pcakeep=0` or `pcadb <= 0` ran to a degenerate `nan_ll` fit.
    Because construction validates, a saved PyTorch model whose config carries such a value
    (only possible before this change) fails to load with the same `ValueError`.
    Setting both stays valid: `pcakeep` takes precedence and `pcadb` is ignored (one INFO log line),
    as in the reference, which parses `pcadb` but never uses it.
  - **Behavior change: `pcakeep`/`pcadb` with `do_sphere=False` are ignored with one warning.**
    No backend reduces without sphering, as in the reference, which keeps every dimension there (amica15.f90:527).
    The request used to be dropped silently, and the PyTorch `mir_step` gate still refused it.
    Every backend now logs one WARNING at construction, and the gate no longer counts it as a reduction request.
  - **Behavior change: `mir_step` no longer rejects `pcakeep >= n_channels`.**
    The PyTorch backend's upfront gate refused any explicit `pcakeep`,
    including the bundled `input.param`'s `pcakeep 32` on the 32-channel sample, which reduces nothing;
    `AMICA.from_params_file("input.param").fit(X, mir_step=1)` raised.
    The gate now rejects only a real request (`pcakeep` below the channel count, or any `pcadb`, while sphering),
    identically on the PyTorch and MLX backends.
- **One parameter-file reader for every backend** (issue #304, epic #324 Phase 3).
  `pamica/fortran_params.py` gains `read_params_file`, the single params-file entry point every backend uses.
  It content-sniffs JSON vs. the literal Fortran `input.param` text and returns pamica's canonical keys either way.
  A JSON file's own alias spellings (`min_grad_norm`, `max_decs`, `numrej`, `num_mix_comps`, `share_int`)
  are translated to the canonical/constructor names through one table, `JSON_ALIAS_TO_CANONICAL`;
  a file carrying both an alias and its canonical key raises `ValueError` naming both.
  `writestep`/`do_history`/`histstep` are translated (identity) keys:
  the legacy NumPy backend implements periodic on-disk checkpointing under these names,
  and the reader translates what any backend supports;
  the PyTorch and MLX backends have no such mechanism yet (issue #312), so `AMICA.fit` names them as not applied.
  The [parameter-files section](guides/validation.md#parameter-files) has the tables.
  - **Behavior change:** a fit from `AMICA.from_params_file` now applies `sample_params.json`'s own
    `max_decs`/`min_grad_norm`/`share_int` settings (as `maxdecs`/`min_nd`/`share_iter`).
    Under their raw JSON spelling they matched neither a named `fit()` parameter nor an `AMICATorchNG` keyword,
    so they were only named in the "not applied" warning.
  - **Behavior change (legacy NumPy backend):** `AMICA_NumPy(params_file=...)` accepts the literal Fortran `input.param` text format, not just JSON;
    a non-JSON file used to raise a raw `json.JSONDecodeError`.
    A params-file setting this backend does not consume is named in one `logger.warning` instead of silently vanishing.
    The NumPy CLI (`python -m pamica.numpy_impl.cli`) accepts both formats too, through the same reader, instead of its own separate `json.load`.
  - **Breaking change (legacy NumPy backend):** `AMICA_NumPy.from_json_file` is renamed to `from_params_file`
    (matching the wrapper's classmethod name, and accepting both formats), with no alias left behind.
  - **Breaking change (legacy NumPy backend):** `AMICA_NumPy(pdftype=...)` with anything other than `0` raises `NotImplementedError` at construction:
    this backend implements only the generalized-Gaussian source density, and used to ignore the setting silently
    (`pdftype` was read but never consulted by the fit path).
    The constructor's default and the bundled `numpy_impl/params.json`'s both changed from `1` to `0` to match;
    since the value was never read, no previously passing fit's numerics change.
  - `validate_implementations.py`'s own JSON-schema-to-Fortran-keyword alias table
    (`_FORTRAN_ALIASES`) is composed from the two shared tables (`JSON_ALIAS_TO_CANONICAL`, then `PAMICA_KEY_TO_FORTRAN_KEY`)
    instead of one hand-maintained entry, and its `max_decs` special case is gone:
    `load_sample_data` reads through `read_params_file`, so canonical keys reach `run_pytorch_amica`'s `ng_kwargs` automatically.
- **Raw backend accessors guard against degenerate fits and bad input shape** (issue #306, epic #324 Phase 5).
  **Behavior change:**
  `AMICATorchNG`, `AMICAMLXNG` and the legacy NumPy backend's fitted-output accessors
  (`transform`, `get_mixing_matrix`, `get_unmixing_matrix`, `get_sensor_mixing_matrix`, `get_rho`,
  `get_pdftype`, `shared_components`, `variance_order`, `model_loglik`, `model_probability`,
  `mir` and `pmi` on torch/MLX, plus `get_sphere`, `get_mean` and `get_model_center`;
  `transform`, `get_weights` and `get_sensor_mixing_matrix` on NumPy)
  raise `RuntimeError` when called on a fit the backend itself classified as degenerate,
  or when a fitted parameter holds a non-finite value,
  instead of silently returning NaN-tainted output.
  The accessors that take data (`transform`, `model_loglik`, `model_probability`, `mir` and `pmi`)
  also validate that the input is a 2D array with the model's fitted input channel count,
  raising the same named `ValueError` that `fit()` raises for the identical mistake,
  instead of a raw matmul/broadcast error.
  `model_probability` also tells apart a NaN log-likelihood (numerical corruption)
  from every model underflowing to `-inf` (an extreme outlier), where it used to report both the same way.
  `AMICATorchNG.from_state_dict` and `AMICAMLXNG.from_state_dict`
  raise `ValueError` naming the payload as the culprit when the saved config does not match the constructor,
  such as a missing or unexpected key,
  chaining the original `TypeError` instead of letting it propagate bare.
  This closes the gap the `AMICA` wrapper's own degenerate-fit guard (issue #50) never covered:
  a caller using a raw backend directly gets the same protection.
  See row 5 of the [differences guide](guides/amica-differences.md#at-a-glance).
- **The validation harness covers every backend** (issue #315, epic #324 Phase 6).
  `validate_implementations.py --backend {torch,numpy,mlx}` (or a comma-separated list, or `all`)
  compares each backend against one Fortran reference run with the same settings;
  the default remains `torch` and prints the same report as before.
  NumPy receives the settings through its own key-translation table, PyTorch and MLX through `AMICA(backend=...)`;
  an explicit `--backend` also prints and saves a one-row-per-backend summary with runtimes (`parity_summary.md`),
  and `--backend mlx` without MLX exits with status 2 and the install hint.
  When the harness landed, before the fitting changes above, all three backends met the reference bar on the bundled sample
  (log-likelihood within 3.2e-5, matched correlation 0.9992, Amari distance 0.004);
  the [validation guide](guides/validation.md#parity-rows-per-backend) has the current rows and each backend's expected bar,
  pinned by an `AMICA_RUN_FORTRAN`-gated test.
  An end-to-end workflow test (`pamica/tests/mne_tests/test_end_to_end_workflow.py`) runs a per-session workflow on both wrapper backends:
  average-referenced EEG with `pcakeep = n_channels - 1`, `AMICAICA`, the EEGLAB export and reload, `save`/`load` and an `input.param`-driven fit.
  - The getting-started page gains a short Apple Silicon (MLX) route that links to the backends guide's full workflow,
    and the differences guide records two existing divergences:
    `do_sphere=False` fits unscaled data where the reference divides each channel by its standard deviation (issue #328),
    and a second `fit` on the same `AMICA_NumPy` instance continues from the first (related to issue #312).
  - **Behavior change (legacy NumPy backend):** `AMICA_NumPy` writes files only when given an `outdir`.
    Its default was `./output`, so every fit wrote `out.txt` at construction, `writestep` checkpoints and its final results into the caller's working directory.
    The default is now `outdir=None`, which writes nothing, as the PyTorch and MLX backends never do unless asked.
    An explicit `outdir` (keyword, params file, or the command-line interface's `--outdir`, which still defaults to `output`) writes exactly what it did before.
  - **Fix:** `AMICA_NumPy.get_sensor_mixing_matrix` returned `pinv(sphere)` times the stored mixing matrix without the transpose the PyTorch and MLX backends apply,
    so its columns were the rows of the true mixing matrix rather than the components' sensor maps
    (about 10% away from the PyTorch backend's maps after five iterations on the bundled sample, and not an inverse of `get_weights() @ sphere`).
    It now matches the PyTorch backend to round-off (4.6e-12 relative), pinned by a cross-backend test.
    The NumPy backend's fit, `transform`, `get_weights` and EEGLAB export were not affected.
  - **Fix:** the legacy plotting helpers in `pamica.numpy_impl.viz` had the same orientation slip.
    `plot_components` drew rows of the mixing matrix as mixing vectors, and it and `plot_pdf_fits` formed activations from the raw data with no mean removal, no sphere and no transpose;
    `plot_model_comparison` skipped the sphere.
    They now plot the model's own sensor maps and sources (what `get_sensor_mixing_matrix` and `transform` return), checked against those accessors on a real fit.
    `load_results` also reads a rank-reduced fit's zero-padded sphere, which it used to reject.
- **`AMICA_NumPy` rejects unknown and unsupported keyword arguments** (issue #346, epic #324 Phase 14).
  `AMICA_NumPy(**kwargs)` used to forward every keyword into a params dict read with `params.get(...)`,
  so a typo (`max_iters=50`) or an option the legacy backend does not implement (`keep_best=True`)
  constructed silently and had no effect,
  unlike the PyTorch/MLX constructors, which take explicit keyword parameters and raise `TypeError` on an unknown name.
  - **Behavior change (legacy NumPy backend only): an unrecognized or unsupported keyword argument raises `TypeError` instead of being silently ignored.**
    A typo raises `TypeError` naming the offending keyword(s),
    with a `difflib`-based "did you mean" suggestion when a close match exists.
    An option implemented on the PyTorch backend (`AMICATorchNG`) but not this one
    (for example `keep_best`, `device`, `dtype`, the kurtosis-switch schedule)
    raises `TypeError` naming the option and pointing to `AMICA(backend='torch')`.
    `n_models`/`n_mix` (`AMICATorchNG`'s spelling of this backend's `num_models`/`num_mix`) get a message naming the correct spelling,
    and `n_channels` its own message, since this backend, like the `AMICA` wrapper, infers it from the data passed to `fit()`.
    Every offending keyword in one call is named in a single error, however many different kinds are mixed together.
  - The three settings this backend spells differently from `AMICATorchNG`
    (`min_nd`/`maxdecs`/`share_iter`, the canonical spelling a params file already resolves either way)
    are also accepted as keyword arguments directly,
    translated to this backend's own attribute name the same way the params-file route translates them.
    Passing both spellings of the same setting at once raises `TypeError` naming both, rather than picking one silently.
  - `files`/`data_dim`/`field_dim` (data-location metadata) work as keyword arguments directly, with the same meaning as in `params_file`;
    a params file and a keyword argument that set one of them to different values raise `TypeError`.
  - The accepted-keyword set is derived from the same source the params-file routing uses
    (`_CONSUMED_KEYS`/`_CANONICAL_TO_NUMPY_KEY`),
    and the unsupported-option list from `AMICATorchNG`'s own constructor signature,
    so neither can drift from what the constructor actually reads.
- **The MLX backend gains `AMICATorchNG`'s full surface** (epic #278).
  With it, the only difference between the raw MLX and PyTorch backends is precision (float32 only on Apple GPUs).
  - **`transform` and save/load** (issue #287, epic #278 Phase 1):
    source extraction (`transform`, plus the `get_mixing_matrix`/`get_unmixing_matrix`/`get_sensor_mixing_matrix`/`get_rho` accessors,
    mirroring `AMICATorchNG`'s issue #24/#27/#142/#223 conventions)
    and persistence (`state_dict`/`from_state_dict`, plus a device- and framework-agnostic `.npz` `save`/`load`:
    `config`/`extra` as JSON-encoded scalars, params as native arrays, no torch coupling, no pickle).
    `transform` derives the unmixing composition from MLX's own `_forward` rather than transcribing torch's tensor layout:
    MLX's `W` is `(n_models, n, n)`, not torch's `(n, n, n_models)`.
    Fitting was untouched (a default fit was bit-identical to before this phase).
  - **`keep_best` best-iterate safeguard** (issue #288, epic #278 Phase 2):
    `fit` tracks the highest-log-likelihood iterate and, if the run ends more than `_KEEP_BEST_TOL` (1e-9, the same constant as `AMICATorchNG`) below that peak,
    restores it instead of returning the last iterate.
    Same name, default (`keep_best=True`) and semantics as the PyTorch backend,
    including its inactivity under `share_comps` and `do_reject`.
    `ll_history` is never rewritten; only `final_ll_` and the twelve fitted-parameter arrays roll back.
    Persisted additively in `state_dict`'s config (a payload without the key loads with the default).
  - **Outlier rejection, the LLt stash, the EEGLAB export and MIR/PMI** (issue #289, epic #278 Phase 3):
    the LLt stash (issue #157), filled per block by the E-step (never a second forward pass) and rolled back on a `keep_best` restore;
    `do_reject` (issue #123's `good_idx` mechanism), with the rejection statistic read from the stash rather than a second forward pass,
    the NumPy backend's design, ahead of `AMICATorchNG`'s open follow-up to drop its own extra `_sample_ll` pass (issue #298);
    `model_loglik`/`model_probability` (issue #141);
    `write_amica_output` (issue #92), a thin adapter over the shared `numpy_impl.load.write_amicaout`;
    and `mir`/`pmi` plus `fit(mir_step=...)` waypoints (issue #137), including the #300 fitted-geometry PCA guard.
    Rejection state (`numrej`/`good_idx`) persists additively in `state_dict`'s `extra`;
    `mir_history_` stays out of both the `keep_best` snapshot and `state_dict` (a diagnostic trajectory, not a fitted parameter).
    A default fit (`do_reject` off, `mir_step=0`) was unaffected, verified bit-identical to the code before this phase,
    and MLX and PyTorch reject the same sample set on the same real data and configuration.
    `write_amica_output` also gained `state_dict`'s two-layer degenerate/non-finite refusal guard, on both the MLX and PyTorch backends:
    a caller using either backend class directly could previously write a NaN model to disk silently.
  - **`variance_order`** (issue #92, the epic's polish round): the EEGLAB back-projected-variance component order,
    validated on real data against a float64 `AMICATorchNG` twin holding identical fitted parameters
    (the component order matches exactly on a configuration with non-degenerate variance gaps).

### Defaults and device selection

- **`AMICA` and `AMICAICA` take their `fit` defaults from the backend** (issue #354, epic #324 Phase 17).
  **Behavior change: an `AMICA()` or `AMICAICA()` fit that sets no `lrate` now runs at 0.1, where it ran at 0.05.**
  The wrapper's 0.05 was the value of EEGLAB's `runamica15.m`,
  while `AMICATorchNG`, `AMICAMLXNG` and `AMICA_NumPy` default to 0.1, the compiled amica15 default (amica15_header.f90:68),
  so `AMICA().fit(X)` and `AMICATorchNG(n_channels).fit(X)` ran at different learning rates.
  `AMICA.fit` now reads the defaults of `max_iter`, `lrate`, `do_mean`, `do_sphere` and `do_newton` from the selected backend's signatures,
  so the wrapper and the backend cannot drift apart again;
  the other four already agreed, so only `lrate` moves.
  `AMICAICA` forwards its keywords to `AMICA.fit` and has no default of its own, so it follows.
  An explicit `lrate`, or one from a parameter file (both bundled files set 0.05), is unaffected.
  On the bundled sample (PyTorch, seed 42, the default 100 iterations), a default wrapper fit now ends at log-likelihood -3.42768, where it ended at -3.43566,
  and its sources match the earlier fit's with a mean Hungarian-matched correlation of 0.983 (minimum 0.931).
  Pass `lrate=0.05` to reproduce an earlier default wrapper fit.
  The differences guide gains a [table of the defaults](guides/amica-differences.md#default-settings-issue-354) of pamica, the compiled binary and `runamica15.m`,
  and says how to reproduce an EEGLAB run from the `input.param` it wrote.
  - Tests: `pamica/tests/test_wrapper_backends.py` checks on both backends that the wrapper resolves the backend's own defaults,
    that a default wrapper fit is the default backend fit (the same learning rate and log-likelihood trajectory),
    and that an explicit `lrate` still takes precedence;
    a cross-backend test (now in `pamica/tests/test_default_settings.py`) holds every constructor default that `AMICATorchNG` and `AMICAMLXNG` share equal,
    since the defaults table gives one pamica column for them.
    `pamica/tests/mne_tests/test_mne_backends.py` checks that an `AMICAICA` fit runs at the backend's default.
    No existing test depended on the old default;
    the figures quoted in two torch-against-MLX test docstrings were measured again at the new one.
- **`AMICA_NumPy` defaults to the other backends' settings** (issue #354).
  **Behavior change: a default NumPy fit now runs without Newton and stops at 100 iterations, like the PyTorch and MLX backends.**
  The bundled `pamica/numpy_impl/params.json`, which the NumPy constructor reads for its defaults, set `do_newton` on and `max_iter` to 2000,
  the only two settings it shares with `AMICATorchNG` on which the two disagreed;
  it now sets `do_newton` off and `max_iter` to 100,
  and the constructor's fallback for a parameter file without `max_iter` is 100 too (it was 2000), which also reaches the NumPy CLI.
  A fit that sets these, directly or through its own parameter file, is unaffected;
  pass `do_newton=True` and `max_iter=2000` to reproduce an earlier default NumPy fit.
  Fits shorter than `newt_start` (20 by default) iterations are byte-identical, since Newton had not started in them.
  - Tests: `pamica/tests/test_default_settings.py` holds every setting `AMICA_NumPy` shares with `AMICATorchNG` to `AMICATorchNG`'s default,
    read through a default NumPy construction and through the backend's own `params.json` loader,
    and holds `AMICAMLXNG`'s constructor and `fit` defaults to `AMICATorchNG`'s (moved from `test_wrapper_backends.py`).
    `pamica/tests/test_pamica.py::test_amica_initialization`, which pinned the old `max_iter`, now expects the shared default;
    no other test relied on the old values, since every NumPy fit that runs past `newt_start` in the suite sets `do_newton` itself.
- **`AMICANative` gives the binary pamica's shared defaults** (issue #354).
  **Behavior change: a native run that does not set them now runs at `lrate` 0.1, without Newton, for at most 100 iterations.**
  The engine wrote the bundled `pamica/sample_data/input.param`'s settings as its defaults:
  `lrate` 0.05, `minlrate` 1e-8, `maxdecs` 3, Newton on from iteration 50 at `newtrate` 1.0, `max_iter` 2000, `block_size` 512,
  `rholratefact` 0.5, `invsigmin` 0, `invsigmax` 100, `mineig` 1e-12, `numrej` 3 and `pcadb` 30.
  It now takes the default of every setting the binary has a keyword for from `AMICATorchNG`'s signature, through the shared keyword table of `pamica.fortran_params`,
  so the four entry points cannot drift apart.
  pamica settings without a binary keyword, such as `keep_best` and `mineig_rel`, are not written, and neither are settings whose default is `None` (`pcadb`, `seed`).
  The binary's `block_size` counts one thread's share of a block and leaves an all-NaN fit when `max_threads * block_size` exceeds the samples (issue #292),
  so unless `block_size` is given the engine writes pamica's 8192-sample block, capped at the data's length, divided by `max_threads`
  (819 with the default 10 threads on the bundled sample, where 8192 as is would process nothing).
  Keys only the binary reads (`max_threads`, `byte_size`, `writestep`, the `load_*` and `update_*` switches) keep their earlier values.
  To run as the bundled or an EEGLAB-written `input.param` configures the binary, pass the file's settings as keywords
  ([Reproducing an EEGLAB run](guides/amica-differences.md#reproducing-an-eeglab-run)).
  - Tests: `pamica/tests/test_default_settings.py` builds the engine's `input.param` without a binary, reads it back through `pamica.fortran_params.read_params_file`,
    and holds every setting it carries to `AMICATorchNG`'s default and every writable setting to being written.
    The native-binary oracles (`pamica/tests/native_oracle.py` and `test_schedule_native_oracle.py`) relied on the old defaults;
    they now start from the bundled `input.param` itself, the configuration they were measured against, and write byte-identical parameter files.
    `test_native_engine.py::test_native_engine_degenerate_fit_raises_clearly` took its NaN weights from the binary's zero-block case at the old `block_size` 512 on 2048 samples,
    which the new default avoids, so it pins `block_size=512`; the other native-engine tests pass at the new defaults.
- **The raw `AMICATorchNG` runs a default construction on the CPU on Apple Silicon** (issue #354).
  `AMICATorchNG(n_channels)` with default arguments raised `ValueError` on every Mac with Metal Performance Shaders (MPS):
  automatic device selection picks MPS, which cannot represent the float64 default.
  When `device=None` picks MPS for a float64 model, the constructor now uses the CPU and logs a warning, as the `AMICA` wrapper did.
  The wrappers and `AMICA.load` now pass `device` through and rely on the backend, and the wrapper's own copy of the fallback is removed.
  An explicit `device="mps"` at float64 still raises `ValueError`, and `device=None` with `dtype=torch.float32` still picks MPS.
  The warning now comes from the `pamica.torch_impl.core` logger, and the wrapper no longer also prints it when `verbose=True`.
  The MLX backend has no device choice of this kind, so the change is PyTorch-only.
  - Tests: `pamica/tests/torch_tests/test_amica_ng_wrapper.py` constructs and fits the raw backend with default arguments,
    checks that `dtype=torch.float32` keeps MPS and that an explicit `device="mps"` at float64 raises,
    and loads a saved model with `device=None`.
    The tests that need MPS skip without it; the macOS CI runner has it.

### MNE wrapper

- **`AMICAICA.apply` restores the PCA residual of rank-reduced fits** (issue #322).
  **Behavior change for `pcakeep`/`pcadb` and rank-deficient fits:**
  `AMICAICA.fit` now computes the full orthonormal PCA basis once
  (`pca_components_`, `n_channels x n_channels`, with `pca_explained_variance_`),
  and `to_mne_ica` exports all of it,
  so MNE's `ICA.apply` keeps the rows past `n_components_` as residual PCA components, as MNE's own ICA does.
  `apply` with nothing excluded now returns the input, and excluding a component removes only that component.
  Before, the subspace the reduction discarded was silently dropped
  (relative error 0.080 on the bundled EEG with `pcakeep=20`, now 1.3e-15).
  This goes beyond the Fortran reference, whose output has no representation of that residual;
  pass `n_pca_components=ica.n_components_` to `apply` for the reference's rank-reduced reconstruction.
  Sources, component maps and the log-likelihood are unchanged, and full-rank fits export bit-identically.
  `to_mne_ica` logs the residual's dimension and the opt-out at INFO.
  Recorded in ADR 0005 and the [differences guide](guides/amica-differences.md#amicaicaapply-restores-the-pca-residual-issue-322).

### Persistence and exports

- **Saved models convert to the component-row layout** (issue #334).
  The PyTorch `state_dict` is now `format_version` 4 and the MLX save format 2.
  An older save loads unchanged in content, converted without loss, unless `share_comps` had merged components:
  such a save raises `ValueError` asking for a refit, because its merges were computed under the old semantics.
- **`AMICA.save` format version 2 records the backend** (issue #313).
  `AMICA.save` records the backend in `wrapper["backend"]`, and `AMICA.load` restores the model on that backend.
  An MLX model's numpy arrays are stored as CPU tensors of the same dtype, so the file still loads with `torch.load(weights_only=True)`.
  Version 1 files, written before this change, still load (always as PyTorch models),
  and both versions go through the component-row conversion and refusal above.
  Loading an MLX file needs MLX installed and `device=None`.
  - **Fix:** a numpy scalar in the backend's config or fit record (for example `fit(X, seed=np.int64(42))`) is now saved as the equivalent Python number.
    `AMICA.save` used to write such a file, which `AMICA.load` then refused (the `weights_only` unpickler rejects numpy scalars).
    Any other value that loading could not read back now raises `TypeError` at save time.
- **Additive fields in the backend saves.**
  PyTorch and MLX saves store `rholrate_cap` (issue #339), a payload without it loading with the ceiling equal to its saved `rholrate`.
  MLX saves gain `pcakeep`/`pcadb` in `state_dict()["config"]` (issue #323), a payload without them loading with `None`, and no `format_version` bump.
  Both backends store `pcakeep`/`pcadb` as plain `int`/`float`, so a numpy-scalar request survives `AMICA.save`/`load`.
  Loading a PyTorch `state_dict` or an MLX save refuses a missing or non-finite learning rate with a `ValueError` naming the field.
- **EEGLAB export: the `A` file is in the reference's layout for any number of models** (issue #334).
  It is the reference's `A(nw, num_comps)` in column-major order
  (EEGLAB's `loadmodout15.m` ignores it; single-model files are byte-identical to before).
  `pamica.numpy_impl.data.load_results` reads it in that layout
  and refuses a multi-model directory written by an earlier version, whose `A` does not invert the `W` beside it;
  write such a directory again from the fitted or reloaded model.
- **Fix: the EEGLAB export wrote an asymmetric sphere transposed** (issue #336, epic #324 Phase 10).
  `write_amicaout` (`pamica/numpy_impl/load.py`), the shared writer called from `write_amica_output` on
  `AMICATorchNG`, `AMICAMLXNG` and the `AMICA` wrapper, and from the NumPy backend's own `_write_results`,
  wrote the square sphere matrix `S` in C order,
  while the Fortran reference and both readers (EEGLAB's `loadmodout15.m` and pamica's `loadmodout`) read it column-major.
  The default symmetric zero-phase component analysis (ZCA) sphere is its own transpose to about 1e-17,
  so the bug moved only that many bytes there;
  with `do_approx_sphere=False` the sphere is genuinely asymmetric, and the exported sphere came back exactly transposed
  (measured on the bundled sample, torch, 3 iterations:
  before the fix `max|S_loaded - S| = 0.51`, `max|S_loaded - S.T| = 0.0`;
  after, `max|S_loaded - S| = 0.0`, `max|S_loaded - S.T| = 0.51`).
  `S` is now written column-major in both the square and rank-reduced branches,
  and `load_results` reads a square sphere the same way.
  **Action needed for existing output:** a full-rank directory written with `do_approx_sphere=False`
  by an earlier pamica holds a C-order `S`.
  EEGLAB always read that transposed (this fix does not change EEGLAB's own reading, only pamica's),
  and pamica's corrected `loadmodout`/`load_results` now also read it transposed, with no error raised.
  Regenerate any such directory by re-running the fit and `write_amica_output` again;
  there is no on-disk version marker to detect the old layout, and this repository carries no compatibility shim to read it automatically.
  Directories from the default (symmetric) sphere and from a rank-reduced (`pcakeep`) fit are unaffected.

### Other fixes

- **Refits and best-of-N restarts after a rank reduction** (PyTorch and MLX backends).
  A rank reduction, explicit or automatic (`mineig`/`mineig_rel`, for example on Maxwell-filtered MEG),
  shrinks the model's `n_channels` to the kept rank, and the next fit validated its data against that shrunk count.
  A second `fit()` on the same instance, and the second restart of any `n_restarts > 1` fit,
  raised `ValueError: X has 32 channels, model expects 20`, which ended the whole multi-restart fit.
  Every fit now starts from the constructor's input channel count,
  including a fit on a model reloaded from `state_dict`/`save`.
  The NumPy backend was not affected.
- **Tests no longer make a full clone shallow** (issue #343).
  Tests that load historical code ran `git fetch origin <sha> --depth 1` to reach the pinned commit;
  in a full clone that records a shallow boundary, after which `git gc` can prune history.
  Every such test now goes through `pamica/tests/pre_change.py`, which only reads the repository:
  when the pinned commit is missing it fails under `CI` and otherwise skips,
  naming the command that fetches it (`git fetch origin <sha>`, or `git fetch --unshallow origin` in a shallow clone).
  `pamica/tests/test_pre_change_loader.py` runs the loader behind a logging `git` wrapper and asserts that nothing is fetched.

### Removed

- **The `package-data` entry for a `pamica/data` directory** (issue #354).
  `pyproject.toml` listed `"pamica" = ["data/*"]`, and no such directory exists, so the entry matched nothing.
  The built wheel holds the same files as before, `pamica/numpy_impl/params.json` included.
- **Dead legacy code in `pamica.numpy_impl.pdf`** (issue #352).
  `choose_pdf_type`, which nothing called, is removed,
  and `compute_pdf` loses its `pdftype` argument and the branches for three legacy densities no backend fits:
  `compute_pdf(y, rho)` draws the generalized Gaussian, the only density the NumPy backend fits,
  and every existing call already used it.

### Documentation

- **Parity figures re-measured with the finished epic** (issue #351, epic #324 Phase 15).
  The validation guide, the differences guide, ADR 0003 and the paper quote measurements of this release's code against the pinned v0.3.3 native binary;
  the run records are in `.context/issue-351/`.
  - The harness's independent-start log-likelihood difference is now 2.7e-4 (it was 2.9e-5),
    within the reference's own seed-to-seed spread at 100 iterations (standard deviation 2.6e-4 over eight seeds);
    from a shared start the difference is 1.6e-6 (2.4e-4 with the code before the epic).
  - The multi-model ensemble's log-likelihood matches the reference's (-3.3541 against -3.3543, Kolmogorov-Smirnov p = 0.83), where it trailed by 0.009;
    `keep_best` restored one of 20 seeded fits, at the 300-iteration budget only.
  - The `share_comps` example fit merges one pair (it merged three), and the early scans merge 30 and 17 components at iteration 8 (they merged 32 and 24).
- **The documentation describes the finished epic** (issue #352, epic #324 Phase 16).
  The concept pages walk one iteration in the reference's order
  (E-step, the likelihood-decrease response and the stopping checks, the exit before any update, then the update,
  with the Newton start, the A-freeze windows and the `doscaling` rescale),
  and the algorithm-flow figure shows the checks before the update.
  The API pages no longer render docstring lines that began with an issue number as headings,
  and the entry pages give the PyPI install.
  During epic #278's polish round the differences guide gained its
  [Unmapped Fortran keywords](guides/amica-differences.md#unmapped-fortran-keywords) section,
  which names three keywords that are dead in the reference itself (`filter_length`/`dft_length`/`decwindow`)
  and records the `do_rho`-vs-`pdftype` divergence.
- **The paper credits the Research Skills plugins** as the development harness,
  in its AI usage disclosure.

### Continuous integration

- **The Python jobs skip changes that touch no code.**
  A change limited to Markdown, the docs site, the paper (including the rebuilt `paper.pdf`), citation metadata,
  or the non-Python records under `.context/` runs only spell-checking and, for the paper, the PDF build.
  Python files under `.context/` still run the full CI.
- **The rebuilt `paper.pdf` is committed back on `dev` and `main` only.**
  Feature branches build it as an artifact, so a PR's head is never a bot commit whose checks wait for approval.

## 0.3.3

MLX fitting parity (convergence stops, component sharing, Newton, all five
source-density families), best-of-N restarts on every backend, the
Fortran-faithful stashed `LLt`, an OOM-safe block-size auto-tuner,
annotation-based rejection in the MNE wrapper, and `share_comps` on
rank-reduced fits; validated end to end on real Maxwell-filtered MEG by an
external tester (#221).

- **Best-of-N random restarts on all three backends** (issue #198, from the
  #145 investigation). `n_restarts` and `restart_seeds` are constructor
  parameters with identical semantics on `AMICATorchNG`, `AMICAMLXNG` and
  `AMICA_NumPy` (forwarded by the `AMICA` wrapper): the fit runs from several
  seeds and keeps the highest final log-likelihood, extending #51's within-run
  `keep_best` across runs -- #145 showed random-init disagreement with the
  reference is basin choice on weak components, not a dynamics difference.
  Seeds default to `seed, seed+1, ..., seed+N-1`; `n_restarts > 1` without a
  base seed or explicit seeds is refused (best-of-N must be reproducible, and
  pamica never seeds itself from the clock). Degenerate restarts are excluded
  from selection but recorded (`restart_seeds_`/`restart_lls_`/
  `restart_stop_reasons_`, NaN likelihood where degenerate); ties keep the
  earlier restart, so selection is deterministic. `n_restarts=1` (the default)
  is bit-identical to before. Fortran has no equivalent; recorded in
  `docs/guides/amica-differences.md`.
- **`mir()`'s PCA-reduction guard now checks the fitted sphere's geometry,
  not just which parameter caused the reduction** (issue #283).
  `AMICATorchNG._pca_reduced()` previously only checked whether `pcakeep`/
  `pcadb` were passed explicitly, so a fit whose rank reduction came from
  automatic `mineig`/`mineig_rel` numerical-rank detection slipped past the
  guard, and `mir()` then crashed with an opaque `numpy.linalg.LinAlgError`
  instead of the documented `ValueError` ("mir() is incompatible with PCA
  reduction"). The guard is now derived from the fitted sphere's shape
  (`sphere.shape[0] != sphere.shape[1]`), which catches both causes. The
  separate upfront `mir_step > 0` gate inside `fit()`, which runs before
  that fit's sphere exists, keeps an explicit-parameter check
  (`_pca_reduction_requested`) as a fail-fast for the one cause it can know
  about ahead of a fit; the auto-detected case there was already caught
  gracefully (warn + NaN, not a crash) at the first mid-fit MIR waypoint, so
  behavior there is unchanged. No NumPy-backend counterpart exists to fix:
  `mir()`/`pmi()` are `AMICATorchNG`-only (issues #137/#143), so there is no
  equivalent guard on that backend.
- **`LLt` is written from the E-step's stashed per-sample log-likelihood**
  (issue #157), on both the PyTorch and NumPy backends, instead of being
  recomputed by a fresh full-dataset forward pass at write time. This is the
  reference's own design -- `modloglik`/`loglik` are allocated once
  (amica15.f90:2617-2620), filled by every E-step and dumped verbatim by
  `write_output` -- and it removes the last full pass the write path paid for:
  a NumPy `writestep` checkpoint drops from 78 ms to 0.8 ms on the bundled
  32-channel sample, and a PyTorch fit no longer spends an extra E-step (12.8
  ms, about half an EM iteration) computing `LLt` even when nothing is written.
  **Behavior change:** pamica now inherits Fortran's one-M-step staleness. The
  written `LLt` is the E-step of the parameters as they stood *before* the
  M-step whose `W`/`A` sit beside it, so it satisfies the reference's own
  invariant `Lt.sum()/(n_good*nw) == LL[-1]` -- which the committed reference
  output satisfies exactly, and which the previous self-consistent recompute
  did not. That comparability was the point: see
  `docs/guides/amica-differences.md` for the decision (2026-08-23) and its
  Fortran citations. Under the PyTorch `keep_best` safeguard, which the
  reference has no counterpart for, the stash is rolled back with the
  parameters, so the exported `LLt` is the E-step that produced the exported
  `final_ll_`. `model_loglik(X)` still gives the log-likelihood of the written
  parameters if that is what you need. The `do_reject` zero sentinel is
  unchanged (a rejected sample's entries are zeroed exactly as
  amica15.f90:2231-2234 does), and one reference-faithful exception to the
  invariant is now pinned as behavior: a `do_reject` fit that rejects on the
  same iteration as the write leaves a small residual, because Fortran
  normalizes `LL(iter)` (amica15.f90:1770) before `reject_data` shrinks the
  good count (amica15.f90:1138, :2252) -- the binary shows it too.
- **Block-size auto-tuner with an OOM-safe fallback** (issue #232, split out of
  #216/#230), on all three backends behind Fortran's own four parameter names
  (`do_opt_block`, `blk_min`, `blk_max`, `blk_step`). With `do_opt_block=True`,
  `fit` times one accumulate pass per candidate block size on the real data and
  device before the first EM iteration and keeps the fastest; the choice and
  every timing are logged at INFO. **Off by default**, and the static
  `block_size=8192` default is unchanged: the winner is decided by measured
  time, so two machines can pick different sizes and their trajectories then
  differ at the ~1e-6 level any block-size change produces, which a
  Fortran-parity run cannot have (pin `block_size` and leave the search off).
  The tuner changes nothing about a fit beyond the block size itself -- the
  timed passes only read model state and consume no RNG, so a post-tune fit is
  bit-identical to one started directly at the chosen size, tested on every
  backend. It is not free either -- two passes per candidate, about 16 EM
  iterations' worth under the defaults -- which pays for itself over a normal
  multi-hundred-iteration fit and not over a very short one.
  The point of porting it is the failure mode the reference gets wrong: Fortran's
  `determine_block_size` walks *upward* into larger blocks and calls
  `allocate_blocks` with no `stat=`, so a candidate that cannot be allocated
  aborts the whole run. Here such a candidate is skipped, the upward walk stops,
  and the fit continues at the largest size that ran (or at the configured
  `block_size` if nothing could be timed). Candidates are additionally clamped
  to `n_samples` -- Fortran silently NaNs when `block_size` exceeds the frames
  available per thread (issue #292) -- and to a conservative estimate of one
  block's peak memory, so the search usually finds its ceiling without walking
  into a failure at all. The sweep bounds are re-derived rather than copied:
  Fortran's 128-1024 sits far below where any pamica backend peaks, so the
  pamica defaults (4096-32768 by 4096) bracket the measured CPU optimum and
  include the 8192 default, while `blk_step` keeps Fortran's arithmetic
  meaning so an `input.param` reads the same on both sides.
  Two consequences on the NumPy backend: `do_opt_block` there used to default
  to **on** (following Fortran's header) with a 128-1024 sweep, so every NumPy
  fit quietly re-tuned itself to a small block and ignored the `block_size` it
  was given -- it is now off by default, and that backend's default
  `block_size` is the shipped 8192 rather than 128. Its naive
  `determine_block_size` helper (which timed a bare `X.T @ X`, the shape of no
  work AMICA actually does, and could not fall back at all) is replaced
  outright by the shared `pamica/blocktune.py`. That helper was worse than
  mistuned: `X.T @ X` costs more as the block grows, so it was structurally
  guaranteed to pick `blk_min`, and every NumPy fit ran at 128 whatever the
  bounds said. This specifically affected `test_sample_data_numpy_vs_fortran`,
  the issue #24 NumPy-vs-Fortran gating test, which requests `block_size=512`
  to match the reference `input.param` but whose historical *effective* block
  size was 128. It has been re-verified at the literal 512 it now actually
  gets, under the new default, and still passes: Hungarian-matched component
  correlation 0.981 (gate > 0.9) and final log-likelihood -3.4039 after 150
  iterations.
  `do_opt_block`/`blk_min`/`blk_max`/`blk_step` also move from the Fortran
  param reader's unsupported table to identity mappings. See
  `docs/guides/amica-differences.md`.
- **`AMICA.from_params_file` now reads the literal Fortran `input.param` text
  format directly** (issue #132, JOSS reviewer feedback), content-sniffed
  (not extension-trusted) alongside the existing JSON schema so the same
  file can drive both the reference binary and pamica for a parity run. The
  translation (`pamica/fortran_params.py`, `read_fortran_param_file`) parses
  all 89 keywords `amica15.f90`'s parser accepts into pamica's actual
  constructor/`fit()` names, renaming the three that Fortran spells
  differently (`min_grad_norm` -> `min_nd`, `max_decs` -> `maxdecs`,
  `numrej` -> `maxrej`), and warns loudly (never drops silently) about the
  32 known keywords with no pamica equivalent (checkpoint warm-start,
  per-family EM freeze toggles, FIR/DFT pre-filtering, reporting cadence,
  ...; the `do_opt_block` search keys moved to identity mappings with #232,
  see below), about any keyword it does not
  recognize at all, and (hard `ValueError`) when a non-empty file yields
  zero recognized settings. `fit()` applies the translated dict as per-call
  defaults -- an explicitly passed argument always wins over the file's
  value -- and warns once about any file setting that matches neither a
  `fit()` nor `AMICATorchNG` parameter (data-location metadata like
  `files`/`outdir`/`data_dim`). See `docs/guides/validation.md#parameter-files`
  for the mapping table.
- **Widened CI coverage** (issue #246, building on #247's macOS Apple Silicon
  job): `ci.yml` now also triggers on pushes to `dev` (the integration
  branch), not just `main` and pull requests, so post-merge drift is caught
  immediately rather than only at the next PR. The macOS job installs the
  `mne` extra alongside `mlx`, so `pamica/tests/mne_tests` (the AMICAICA/MNE
  wrapper) runs against Accelerate as well as Linux/OpenBLAS. A new
  schedule-only `weekly-macos-slow.yml` workflow runs the full suite on
  macOS every Sunday, including the `@pytest.mark.slow` tests and the
  Fortran-parity tests (`AMICA_RUN_FORTRAN`, `PAMICA_NATIVE_BINARY`,
  `AMICA_FORTRAN_BIN`), none of which had ever run in CI before, without
  slowing down or blocking any PR.
- **Guarded the MLX backend's unmixing-matrix inversion against an
  uncatchable process abort** (issue #274). MLX 0.32's CPU-stream
  `mx.linalg.inv` does not raise a Python exception on a singular per-model
  `A[:, comp_list[:, h]]` — LAPACK's LU failure aborts the whole process
  (`libc++abi: ... [Inverse::eval_cpu] LU factorization failed`), which no
  `try`/`except` around `fit` can catch (found while verifying #271's `W`
  finiteness guard). `_update_unmixing_matrices` now condition-checks each
  per-model matrix host-side immediately before calling `inv` and raises a
  catchable `RuntimeError` naming the model, iteration and condition number
  instead. The threshold (1e12) is calibrated empirically, not from
  float32's ~1/eps precision-loss point: isolated-subprocess measurement
  showed the true LU-abort onset is not a clean function of condition number
  (observed anywhere from ~9e8 to beyond ~5e10, matrix-structure-dependent),
  and an existing adversarial test legitimately reaches cond~4.4e9 without
  aborting, so 1e7 (the precision-loss estimate) would have been a false
  positive on real, currently-passing behavior. 1e12 clears that observed
  legitimate maximum by ~225x while staying far below where a genuinely
  singular `A` (e.g. a duplicated component column) actually lands
  (~1e15-1e17). Read-only: verified bit-identical `A`/`W` on a short fit
  with and without the guard, and negligible added cost (~30 microseconds
  per model per iteration, measured on the bundled sample). The upstream MLX
  behavior is also being reported separately.
  Non-finite entries get their own handling: a matrix that is non-finite in
  EVERY entry (the observed shape of a dead, zero-responsibility model's
  corruption) carries no structural signal to check and is left to flow
  through to `inv`/`nan_params` exactly as before; any other non-finite
  pattern has its non-finite entries 0-filled (a neutral, non-scale-distorting
  value) before the condition check runs. This closes a gap a review pass
  found in the first version of the guard: a matrix that was BOTH non-finite
  in one entry AND structurally singular elsewhere (an exact duplicate column
  plus one stray NaN) used to skip the check entirely on any non-finite entry
  and still reach the uncatchable abort. Containment is intentionally not
  total — no scalar condition-number threshold can guarantee catching every
  abort-capable matrix, since the observed LU-abort onset (cond~9e8 to beyond
  cond~5e10) sits below the 1e12 threshold — and both docs and code comments
  now say so explicitly, alongside noting (and rejecting, as disproportionate
  to a now-rare defect) a fully complete alternative: running `inv` itself in
  a disposable per-call subprocess. Tests:
  `pamica/tests/mlx_tests/test_mlx_inv_guard.py` (singular and near-singular
  `A` on a real fitted model raise `RuntimeError` rather than aborting, the
  duplicate-column-plus-stray-NaN combination raises rather than aborting, a
  purely non-finite dead-model `A` flows through to `nan_params` without
  raising or aborting, the guard is bit-identical to an unguarded
  `inv`/`slogdet` call, and a standard fit is unaffected).
- **Pinned `mir_history_` against the `keep_best` rollback and against
  save/load** (issue #161, follow-up from #137/#160). Both claims already
  held before this PR and were already tested: `test_mir_history_survives_keep_best_restore`
  and `test_mir_history_empty_after_save_load` (`tests/torch_tests/test_ng_convergence.py`,
  added in #213, whose commit message already noted "Folds in issue #161") already force a genuine `keep_best` restore on real EEG (the
  established `n_models=2, do_newton=True, newt_start=1, lrate=0.5, seed=0`
  recipe) and assert `mir_history_` is neither truncated nor rewritten, and
  already round-trip a fit through `AMICA.save`/`load` and assert an empty
  `mir_history_` (`AMICATorchNG.from_state_dict` reconstructs via `__init__`,
  which sets `mir_history_ = []`, and `_load_params` never touches it). #245 later
  fixed a waypoint assertion in the first test and documented the one-update offset between `mir_history_[i]`
  (computed AFTER iteration `i`'s update) and `ll_history[i]` (the pre-update
  likelihood) on `AMICA.mir_history_` and in `AMICATorchNG.__init__`. This PR
  verified both tests and both docstrings against the code and added the one
  place that was missing the offset note: `AMICATorchNG.fit`'s `mir_step`
  docstring, the first place a reader would look. Documentation and test
  verification only, no logic change.
- **Documented that `final_ll_`/`self.ll[-1]` trails a `share_comps` merge
  landing on the final fit iteration** (issue #269). When a merge fires on the
  last iteration, the returned `A`/`W`/`comp_list` are already post-merge but
  the reported log-likelihood still reflects the pre-merge state, since
  `identify_shared_comps` runs after that iteration's LL is recorded
  (matching the reference ordering, amica15.f90:1856-1858). Documentation
  only, across all three backends (`torch_impl/core.py`, `numpy_impl/core.py`,
  `mlx_impl/core.py`) plus `docs/guides/amica-differences.md`; the behavior is
  unchanged and now pinned by a matching pin test in each of
  `tests/torch_tests/test_ng_sharing.py` and `tests/test_numpy_share_comps.py`
  (mirroring the MLX test added in #268).
- **The MLX backend now supports all five source-density families** (issue
  #265, epic #260 Phase 4, porting the PyTorch backend's issue #26).
  `AMICAMLXNG` takes `pdftype`/`kurt_start`/`num_kurt`/`kurt_int` with
  `AMICATorchNG`'s names, defaults and semantics: the fixed families (2
  Gaussian, 3 logistic, 4 sub-Gaussian cosh+, 1 super-Gaussian cosh-) via a
  per-source `pdtype` dispatch in `_score`/`_log_pdf`, and the `pdftype=1`
  extended-Infomax adaptive switcher between codes 1/4 by kurtosis sign on the
  usual schedule, plus a new `get_pdftype()` accessor. `pdftype=0` (the
  default) is byte-for-byte the pre-#265 implementation: the `_pdtype_h`
  `None` fast path adds zero graph nodes, verified by an epic-tip-vs-new
  before/after fit comparison (bit-identical `A`/`ll_history`, single- and
  multi-model). The fixed families' `z0`/`fp` match the literal
  `amica15.f90` forms through MLX's float32 evaluation to 1e-6
  (`rtol=atol`), and a matched 100-iteration fit lands on the float64
  PyTorch likelihood to within ~1e-7 for every family (four orders inside
  the 0.05 gate). `rho` is frozen for every non-GG family
  (`self.dorho = pdftype == 0`), which also skips the per-iteration
  lgamma-table refresh here and the `drho_n` accumulation, which
  `AMICATorchNG` still pays unconditionally in its `_get_block_updates` for
  a frozen `rho` (its digamma pull is already gated behind the same
  `self.dorho` flag, so that part is not a divergence -- a deliberate
  MLX-only WORK divergence, not a numeric one). The
  switcher accumulates its kurtosis moments in numpy float64 on the host
  (an MLX-motivated mechanism difference, not a decision difference) and
  has no bit-exact oracle -- the reference declares `do_choose_pdfs` but
  never accumulates the moments that would drive it -- so it is
  behavior-validated on real EEG, as ADR 0002 already scoped for the
  PyTorch backend. `share_comps` does not synchronize `pdtype` across a
  merged pair, documented on `shared_components()`. Corrected an inaccurate
  cell in `docs/guides/amica-differences.md`'s backend table along the way:
  the legacy NumPy backend's fit path (`_compute_log_pdf`) has no `pdtype`
  parameter and only ever implemented the generalized-Gaussian family, not
  "all five" as the table previously (incorrectly) claimed. Evidence:
  `.context/issue-265/pdf_family_findings.md`.
- **The MLX backend now supports Newton** (issue #264). `AMICAMLXNG` takes
  `do_newton`/`newt_start`/`newtrate`/`newt_ramp` with `AMICATorchNG`'s names,
  defaults and semantics: the same curvature accumulators, the same
  per-source-pair 2x2 solve behind the same unguarded `prod > 1`
  positive-definiteness test, the same learning-rate ramp to `newtrate` (and to
  `lrate_cap` on a fallback), the same `maxdecs` ratchets, and the same
  `n_newton_fallbacks` counter. It runs entirely in float32, which was
  pre-registered as a go/no-go rather than assumed: on the bundled sample the
  finalized curvature matches a float64 PyTorch twin to 4e-7 relative, one
  warmed Newton M-step moves `A` to within 2.4e-7 of the twin's, a matched
  100-iteration fit reaches -3.41149 against float64's -3.41149, and the
  positive-definiteness guard never comes within 1.9 of its boundary across six
  full-data fits (zero fallbacks, monotone likelihood). Evidence and the gate
  script: `.context/issue-264/`. `do_newton` is off by default and every
  accumulator it needs is gated on it, so natural-gradient fits — including
  multi-model and `share_comps` ones — are bit-identical to before.
- **Multi-model Newton no longer crashes on the NumPy backend** (issue #267).
  `numpy_impl` finalized the curvature by dividing its `(data_dim, num_models)`
  accumulators by `dgm[:, None]`, a `(num_models, 1)` model mass that broadcasts
  only for one model, so every multi-model Newton fit raised `ValueError:
  operands could not be broadcast together` on the first iteration Newton was
  active. The issue reported it from a `share_comps` collapse, but it needed no
  sharing at all. Now `dgm[None, :]`, matching the PyTorch backend's
  `dgm.unsqueeze(0)`. Single-model fits are unaffected.
- **The MLX backend now supports component sharing** (issue #263). `AMICAMLXNG`
  takes `share_comps`/`share_start`/`share_iter`/`comp_thresh` with
  `AMICATorchNG`'s names, defaults and validation, runs the same merge schedule
  and 6-iteration post-merge A-freeze, masks the mixture updates and the
  gradient norm by `comp_used`, and exposes `comp_used` and
  `shared_components()`. The merge decision is not reimplemented: it calls the
  NumPy `identify_shared_components` kernel on host float64 `pinv(sphere) @ A`,
  the metric the PyTorch and NumPy backends already share, so all three decide
  identically from the same fitted state. Sharing is off by default and inert
  for `n_models=1`; with it off, every masking and freezing step added here is a
  no-op, so a fit is bit-identical to the same fit with sharing enabled but
  never scheduled (see the `gm` entry below for the one float32-ULP shift
  multi-model fits see relative to the previous release).
- **The MLX multi-model A-update now weights with the previous iteration's
  `gm`** (issue #263; issue #219 raised the same ordering question for
  `numpy_impl`'s `ndtmpsum` and flagged the array backends as follow-up, since
  fixed in PyTorch and now here). Fortran builds `dAk` in the accumulation pass,
  before `update_params` reassigns `gm` (amica15.f90:1749-1761, :1788); MLX used
  the just-updated `gm`. The weights cancel analytically for a disjoint
  `comp_list`, so single-model fits stay byte-for-byte identical and default
  multi-model fits are unaffected except at float32-ULP scale (the two `gm`
  snapshots genuinely differ, so the canceling division rounds differently;
  measured at most 2.98e-8 in `dAk` on the bundled sample). A fit that shares
  components moves its shared columns differently (by ~1e-2 in `A`) and now
  matches the PyTorch backend to float32 precision.
- **NumPy `share_comps` now measures similarity on de-sphered sensor-space
  maps, matching the PyTorch backend and the Fortran reference** (issue #258).
  `identify_shared_components` used to compare mixing columns directly in the
  sphered space; it now takes `pinv(sphere) @ A`, the same `Spinv` back-map
  `AMICATorchNG` and `amica15.f90` use (:1916, :568-578), so both backends
  reach the identical merge decision from the same fitted state. Borderline
  merge decisions near `comp_thresh` can change relative to a pre-#258 NumPy
  fit, even on a full-rank sphere.
- **The MLX backend now has convergence stops** (issue #248). `AMICAMLXNG`
  implemented neither, so an Apple-GPU fit always ran to `max_iter`; it now
  carries `use_min_dll`/`min_dll`/`maxincs`, `use_grad_norm`/`min_nd` and the
  likelihood-decrease branch's gradient-norm half, with the same names, defaults
  and `stop_reason` strings as `AMICATorchNG`, and stops at the same iteration
  as it on the same data.
- **`share_comps` on the NumPy backend now runs the same algorithm as the
  PyTorch one** (issues #240, #242). A column shared by two models took one
  A-step per contributing model, the second against an already-stepped `A`,
  instead of the reference's single `gm`-weighted average applied once
  (amica15.f90:1749-1761, :1807); the post-merge A-freeze was missing entirely;
  and merged-away columns were divided 0/0 and masked, which also silenced a
  genuine collapse in a live column. Merged-away columns are now indexed out of
  the mixture updates instead of masked, and the share settings that would
  freeze `A` permanently (`share_int <= 6`) are rejected at construction, as in
  `AMICATorchNG`. Sharing is off by default, and fits with it off are bit-
  identical.
- **`comp_used` no longer goes stale across `share_comps` schedule points on
  the NumPy backend** (issue #240). `identify_shared_components` rebuilt the
  mask all-True on every call, and since the merge loop skips already-merged
  pairs, a later schedule point resurrected merged-away columns as live: the
  unmasked mixture updates then divided 0/0 for every merged-away column and
  the fit returned NaN mixture parameters while reporting success.
  `comp_used` is now derived from the final `comp_list` (exactly the set of
  referenced columns, matching how `AMICATorchNG.comp_used` is a derived
  property that cannot go stale), and merged-away columns are skipped and
  frozen at their last finite value instead of carrying NaN.
- **The `nd` gradient-norm metric is weighted by the pre-update model
  weights** (issue #219), on the NumPy and PyTorch backends, matching the
  reference's ordering: Fortran accumulates `dAk`/`nd` before `update_params`
  reassigns `gm` (amica15.f90:1749-1761, :1788). Single-model fits are
  unchanged (the weight cancels); the MLX counterpart landed with the #263
  sharing port (see that entry). Affects the `use_grad_norm`/`min_nd` stop
  under multi-model fits.
- **A NumPy fit that ends non-finite no longer reports success** (issue #240).
  `fit()` checks the fitted parameters at exit, not only the likelihood, and
  sets `converged=False` with a `stop_reason` naming what went non-finite.
  Periodic `writestep`/`histstep` checkpoints are gated by the same check and
  skipped with a logged reason rather than persisting NaN that `loadmodout`
  would read back without complaint; the last valid checkpoint stays on disk.
- **Checkpoint cadence matches the reference** (issue #240). `writestep` and
  `histstep` are now anchored on the Fortran-style 1-indexed iteration
  (`mod(iter, writestep) == 0`, amica15.f90:1124/1130), so the first checkpoint
  lands at iteration `writestep`. The 0-indexed transcription fired at iteration
  0, so every fit wrote a checkpoint after its first iteration whatever
  `writestep` said. Final results are unaffected: `fit()` always writes the
  converged result.
- **The MNE wrapper honors `bad_*` annotations during fitting** (issue #251,
  contributed by the project's external MEG tester). `AMICAICA.fit(...,
  reject_by_annotation=True)` (the default, matching
  `mne.preprocessing.ICA.fit`'s convention) excludes samples covered by
  annotations whose description starts with `bad` from the AMICA fit. The
  original-timeline mask is kept in `good_sample_mask_`, and scoring stays
  timeline-faithful: `get_model_probability` returns full-length,
  original-time-axis output with NaN over the rejected spans.
- **`share_comps` works on rank-reduced and rank-deficient fits** (issue #253,
  reported from Maxwell-filtered MEG in #221). The PyTorch merge metric mapped
  mixing columns back to sensor space with `inv(sphere)`, which raised
  "Component sharing needs an invertible sphere" on exactly the data class that
  rank detection had just made fittable. It now uses `pinv(sphere)`, the
  reference's own `Spinv` back-map under reduction (amica15.f90:568-578), and
  `share_comps` with `pcakeep`/`pcadb` is no longer rejected at construction.
  Full-rank fits are unaffected: `pinv` equals `inv` to ~1e-15 there, and the
  bundled sample reproduces its previous `comp_list` and log-likelihood bit for
  bit.

## 0.3.2

Rank-deficient input support across every backend, a much faster default block
size, and a reproducible Fortran reference for parity runs.

- **Rank-deficient data now works** (issue #223, reported from Maxwell-filtered
  MEG in #221). The numerical rank of the data covariance is detected and the
  model is sized to it, porting the reference's `mineig`/`numeigs`/`Spinv`
  machinery, which pamica had not implemented: previously such a fit died with
  `nan_ll` on the first iteration. New `get_sensor_mixing_matrix()` returns
  sensor-space scalp maps when the sphere is no longer square. The rank policy
  is shared by the PyTorch, NumPy and MLX backends so they cannot disagree.
- **Rank detection defaults to a relative eigenvalue floor** (`mineig_rel=1e-12`)
  rather than the reference's absolute `mineig=1e-15`, which is unit-dependent:
  MEG in Tesla yields rank zero under it, and average-referenced EEG is detected
  only by luck. Pass `mineig_rel=None` for the reference's exact behavior. Well
  conditioned data is unaffected and stays bit-identical. See ADR 0004 and
  `docs/guides/amica-differences.md`, which now lists every deliberate difference
  from the reference in one table.
- **The MNE wrapper scales by channel type** before fitting, following MNE's own
  ICA convention (issue #225). Required for mixed magnetometer/gradiometer data,
  whose units differ by orders of magnitude; it decides which directions survive
  rank reduction. A single channel type is unaffected. `AMICAICA` also exports
  rank-reduced fits, and no longer rejects `pcakeep`/`pcadb`.
- **`block_size` default raised from 512 to 8192** (issue #216), ~6x faster per
  iteration on CPU float64 for the bundled sample. Every backend was
  dispatch-bound at the old value. Runs compared bit-for-bit against the binary
  must set the same value on both sides.
- **EEGLAB output of a rank-reduced fit is readable** (issue #164): the sphere is
  padded to the `nx*nx` record the reference writes, and read back column-major.
- **Parity runs are now controlled experiments** (issue #228). The harness
  forwards every setting the binary understands instead of six hardcoded keys,
  seeds the reference run and pins it to one thread, and defaults to the
  seedable native engine rather than the unseedable bundled fixture. Two
  reference runs are now bit-identical where before they differed by up to 0.59.
- `benchmarks/reproduce_table1.py` reproduces the paper's parity table from the
  bundled sample, and the validation guide states what each row costs to verify
  (issue #144).
- **Both Fortran convergence criteria were dead in `numpy_impl` and now work**
  (issue #212). `AMICA_NumPy` stored the raw log-likelihood sum instead of
  Fortran's per-sample-per-channel normalization, reporting `-3317862.78` where
  the reference reports `-3.3`; `min_dll` defaults to `1e-9`, so `use_min_dll`
  could never fire from genuine convergence. Separately, `nd` was built from the
  raw block sum rather than the `gm`-weighted mapped directions, reporting
  ~5.4e3 against Fortran's ~5.7e-2 and staying flat across iterations, so
  `use_grad_norm`/`min_nd` was equally unreachable. Both now match the reference
  formulas, and `AMICA_NumPy.ll_history` is on the same scale as the other
  backends.
- Added the three missing Fortran convergence stops to `AMICATorchNG`
  (issue #207): `use_min_dll`/`min_dll`/`maxincs` (small-likelihood-increase
  stop), `use_grad_norm`/`min_nd` (weight-gradient-norm stop), and the
  lrate-decrease branch's missing gradient-norm half
  (`stop_reason="grad_norm_floor"`). Fixes the reported case where, under
  `do_newton=True`, `lrate` settles at `newtrate` and oscillates instead of
  annealing, so the pre-existing `lrate_floor` check never fired and
  `max_iter` was the only working stop. All five new constructor arguments
  persist through `state_dict()`/`from_state_dict()`; older saved files
  (missing these keys) still load, falling back to the Fortran-faithful
  defaults.

## 0.3.1

Rho-rate schedule fixes across all backends and a reproducible-seed option in the
native binary build.

- Fixed the rho learning-rate (`rholrate`) schedule to match Fortran
  `amica15.f90`: it is a `maxdecs`-ratcheted ceiling (reset to `rholrate0` each
  iteration/fit, tightened only after `maxdecs` persistent log-likelihood
  decreases, gated on `iter > newt_start`), not a per-decrease monotone decay.
  The previous decay collapsed the rho rate toward ~1e-5 and froze the source
  shape. Fixed in the PyTorch and NumPy backends (#194, issue #193) and the MLX
  backend (#197, issue #195).
- Native binary build: reproducible `seed` option. A `seed <int>` line in
  `input.param` now seeds the random initialization deterministically (per-rank,
  no system clock), so a native run is reproducible run to run; without it the
  default stays clock-random. Also makes `random_seed` portable across compilers
  via `random_seed(SIZE=...)`. Adopted from sccn/amica PR #54; the released
  binaries (rebuilt by CI) carry the option (#196).
- Documented the #145 investigation (Newton-vs-Fortran weak-component divergence
  at long budgets): resolved as init-basin sensitivity on under-determined
  components, not a dynamics bug (identical init gives matching results); the
  optional init-robustness enhancement is tracked in #198.

## 0.3.0

MNE-Python compatibility layer (epic #139), additive: the scikit-learn-style
`AMICA` API and the byte-identical EEGLAB I/O are unchanged.

- `pamica.mne_compat.AMICAICA`, an MNE-facing wrapper that fits AMICA directly
  from an `mne.io.Raw`/`Epochs` (`picks=...`, epochs concatenated along time like
  MNE's own ICA) and interoperates with the standard MNE ICA consumer surface:
  `get_sources`, `apply`, `get_components`, `plot_components` and `plot_sources`.
  `to_mne_ica()`
  returns a fully-populated `mne.preprocessing.ICA` (including `reject_`/
  `n_samples_`, so `ICA.save` and `plot_properties` work), so the whole MNE ICA
  ecosystem (component plotting, `find_bads_eog`/`_ecg`, exclusion workflows)
  works on an AMICA decomposition. The export maps pamica's mean, symmetric-ZCA
  sphere and unmixing into MNE's `pca_mean_`/`pca_components_`/`unmixing_matrix_`,
  writing the sphere as `V diag(1/sqrt(e)) V^T` with `V` orthonormal so MNE's
  scalp maps are in channel space; `to_mne_ica().get_sources(raw)` reproduces
  `AMICA.transform(X)` to float64 precision, pinned on real sample EEG. `fit`
  rejects PCA reduction (`pcakeep`/`pcadb`, which leaves the sphere rank-deficient
  and the export invalid) and non-finite input, and a degenerate fit is refused
  by the consumer methods rather than emitting NaNs. MNE is an
  optional extra (`pip install pamica[mne]`); `import pamica` never requires it,
  and a dedicated CI job runs the wrapper tests with the extra installed (phase 1,
  single-model, #140).
- Multi-model exposure through the MNE wrapper: `AMICAICA(n_models=...)` fits a
  mixture of ICA models, and since MNE's `ICA` represents only one unmixing,
  each model is exported as its own single-model `mne.preprocessing.ICA` via
  `to_mne_ica(model_idx=...)` (and the `model_idx` argument on `get_sources`/
  `apply`/`get_components`/`plot_components`/`plot_sources`). The per-sample model
  dominance MNE cannot represent is exposed directly: `get_model_probability(inst)`
  returns `P(model | sample)` (`(n_models, n_samples)`, columns sum to 1) and
  `plot_model_probability(inst)` draws the per-model probability plus best-model
  log-likelihood over time. These build on a new public live accessor,
  `AMICA.model_loglik`/`model_probability` (and the `AMICATorchNG` equivalents),
  which score arbitrary data through the stored sphere/mean; the training-data
  path (without `do_reject`) is pinned bit-for-bit against the E-step's own `Lht`. The per-model export
  folds each model's data-space center `c` into `pca_mean_`, so the round trip
  holds for the multi-model case too. `pamica.viz.plot_model_probability` now also
  accepts a live `lht` array, not only a written `AmicaOutput` (phase 2, #141).
- pamica-specific fitted metadata is inspectable through the MNE wrapper rather
  than silently dropped by the `mne.preprocessing.ICA` export: `get_pdftype(model_idx=...)`
  returns each component's source-density family code (0-4, named by
  `pamica.mne_compat.PDFTYPE_NAMES`), `get_rho(model_idx=...)` the
  generalized-Gaussian shape parameters, and `shared_components()` the components
  merged across models by `share_comps`. The same accessors are added to
  `AMICA`/`AMICATorchNG` (phase 3, #142).
- Separation-quality metrics are available directly on an MNE object:
  `AMICAICA.mir(inst, model_idx=...)` (Mutual Information Reduction, in nats) and
  `AMICAICA.pmi(inst, model_idx=...)` (pairwise mutual information between the
  fitted sources), so MNE-side users get the same metrics as EEGLAB-side users.
  Both extract the fitted channels from the `Raw`/`Epochs` and delegate to
  `AMICA.mir`/`pmi` (#133); the results match the array API exactly (phase 4,
  #143).

## 0.2.2

GitHub repository rename to pAMICA and a `__version__` fix.

- Fixed `pamica.__version__` reporting the stale `0.1.2`: `version.py` hardcoded
  the version and the release sync never touched it, so the 0.2.1 wheel shipped
  correct distribution metadata but a wrong runtime attribute. `__version__` now
  derives from the installed package metadata, so `pyproject.toml` is the single
  source of truth and it can never drift again (#182).
- Canonicalized `pyAMICA` -> `pAMICA` URLs after the GitHub repository was
  renamed `sccn/pyAMICA` -> `sccn/pAMICA`. The documentation site moved to
  <https://eeglab.org/pAMICA/>, so the old `eeglab.org/pyAMICA` links (including
  the README docs badge) now 404; the repository URLs, codecov, the native
  binary resolver's default repository, the docs badge, and `git clone`/`cd`
  snippets are updated to match. GitHub redirects the old repo URLs, and the
  package/import name stays lowercase `pamica` (#184).

## 0.2.1

PyPI publishing, release-metadata sync, the pAMICA display title, and
native-engine documentation.

- Packaging and release: a PyPI publish workflow (`publish.yml`) uploads the
  `pamica` sdist and wheel via Trusted Publishing (OIDC) when a GitHub release
  is published, and `scripts/sync_version.py` keeps the release version in step
  across `pyproject.toml`, `CITATION.cff` and `.zenodo.json` (the publish job
  fails a release whose tag disagrees with them). The display title is now
  **pAMICA**; the package, import and `pip install pamica` stay lowercase
  `pamica` (pip name matching is case-insensitive, so `pip install pAmica`
  resolves to the same project) (#177).
- Native engine docs and validation wiring: a dedicated `AMICANative`
  documentation page (usage, binary cache/SHA-256 verification,
  `PAMICA_NATIVE_BINARY`, the `python -m pamica.native` installer, and the
  offline `native/build.sh` fallback), and `validate_implementations.py` gains
  `--native-engine`/`--fortran-binary` so the real Fortran reference runs as a
  backend on any platform, not only through the bundled macOS `amica15mac`
  fixture (#147 phase 5, #179).

## 0.2.0

Package rename to align with the reserved PyPI name.

- Renamed the Python package `pyAMICA` -> `pamica`: the import path is now
  `import pamica` and the distribution installs as `pip install pamica` (pip
  name matching is case-insensitive, so `pip install pAmica` resolves to the
  same project). The GitHub repository (`sccn/pyAMICA`), the documentation
  domain (`eeglab.org/pyAMICA`), and the release-asset repository are unchanged
  (#176).

## 0.1.3

Native Fortran run engine, separation-quality metrics, LLt output parity, and
the `loadmodout` byte-order fix.

- Native Fortran run engine (`AMICANative`), the fourth backend alongside NumPy,
  PyTorch and MLX. It runs the AMICA Fortran reference itself and returns an
  `AmicaOutput` with the usual accessors, so it is the parity oracle the Python
  backends are checked against. The reference is now built dependency-free (a
  single-rank MPI shim removes the Open MPI runtime, on top of sccn/amica PR
  \#53's no-MKL recipe; proven identical to real Open MPI at machine epsilon) and
  released as a self-contained binary for macOS arm64, Linux x64/arm64 and Windows
  x64 (Windows arm64 runs the x64 binary via emulation until a native toolchain
  exists, issue #173). The binary is resolved for the host and downloaded from the
  release on first use (SHA-256 verified); `python -m pamica.native` installs it
  explicitly, or set `PAMICA_NATIVE_BINARY` to a local build (epic #165).
- Fixed `loadmodout` reading `W`, `sbeta` and `rho` in the wrong byte order:
  it used C order where the writer, genuine Fortran output and EEGLAB's
  `loadmodout15.m` all use column-major (F order). The consequence was that
  `AmicaOutput.W` came back transposed, silently corrupting genuine Fortran
  output and everything derived from it (`A`, `svar`, `origord`), and
  `sbeta`/`rho` were scrambled whenever `num_mix > 1` (the default). A
  write-then-read round trip cancels the error, so no self-consistency test
  could catch it; the fix is pinned by recomputing the bundled Fortran
  fixture's own reported log-likelihood from the loaded parameters (an external
  oracle). The writer's multi-model `W` layout, which interleaved models and was
  not EEGLAB-readable, is corrected to genuine Fortran (model axis slowest);
  single-model output is byte-identical to before. `AmicaOutput` gains a
  supported `sources(X, model=0)` accessor (the loaded-fit counterpart of the
  live model's `transform`) so downstream source derivations no longer hand-roll
  the sphere/unmixing composition (#159). Migration note: a *multi-model*
  `amicaout` directory written by an earlier pamica (whose `W` used the old
  model-interleaved layout) must be regenerated with `write_amica_output`, not
  just re-loaded; there is no version marker to detect the old layout (genuine
  Fortran output carries none either), and the pre-fix multi-model `W` was never
  in the correct convention regardless. Single-model directories are unaffected
  (byte-identical before and after).
- Separation-quality metrics (`pamica.metrics`): `mir` (Mutual Information
  Reduction, in nats) measures how much mutual information a fitted unmixing
  removes from the data. A direct port of `getMIR.m` from bigdelys/pre_ICA_cleaning
  (Apache-2.0; see `THIRD_PARTY_NOTICES.md`), verified against the original at
  1.7e-15 relative on the bundled sample EEG (#134).
- `pairwise_mi` and `block_diagonal_order` (`pamica.metrics`): the pairwise
  mutual-information matrix between fitted sources, plus a greedy
  nearest-neighbor-chain ordering that clusters dependent components near the
  diagonal. A clean-room reimplementation: the reference (`minfojp.m` in
  postAmicaUtility) is GPL-2.0-or-later and pamica is BSD-3-Clause, so its
  source was never read. Agrees with that reference at r=0.9887 on identical
  signals (#135).
- `LLt` output parity with the Fortran reference: both backends now write the
  per-timepoint, per-model log-likelihood file that the reference binary
  produces on every run, and `loadmodout` reads it with the correct column-major
  layout (it previously used C order, scrambling `Lht`/`Lt`). Verified
  bit-exactly in both directions against EEGLAB's real `loadmodout15.m`. Under
  `do_reject`, rejected samples are written as exactly `0.0`, matching Fortran:
  those zeros are load-bearing, since its `load_rej` reconstructs the rejection
  mask from them (#155).
- `AMICATorchNG`/`AMICA` gain `mir()`/`pmi()` accessors that compose the
  fitted unmixing the documented way (`get_unmixing_matrix(model_idx) @
  sphere` for MIR, `transform(X, model_idx)` for PMI) and delegate to
  `pamica.metrics.mir`/`pairwise_mi`, so callers no longer hand-compose the
  transform themselves. `fit()` also accepts `mir_step` (default `0`, off) to
  record MIR waypoints during training in `mir_history_` as
  `(iteration, mir_nats, variance)`; like `ll_history_`, it is a true
  trajectory that a `keep_best` restore does not rewrite. PCA reduction
  (`pcakeep`/`pcadb`) is rejected up front with a named error, since it
  leaves the sphere rank-deficient and MIR's log-Jacobian undefined (#137).
- Visualization module (`pamica.viz`): `plot_pmi_heatmap` and
  `plot_model_probability`, backend-agnostic views over `AmicaOutput` that return
  a `Figure` (and accept an optional `ax`/`axes`) rather than mutating pyplot
  global state, plus `read_eeglab_set_metadata` for the sample rate pamica
  itself has no notion of. Both plots are verified against the MATLAB reference:
  the smoothed model probability matches `smooth_amica_prob` at r=0.9886, and
  `pairwise_mi` matches `minfojp` at r=0.9887 (#136).
- Fixed `numpy_impl.pdf.compute_pdf` using `gammaln` where the generalized
  Gaussian needs `gamma`, which made the returned density negative for every
  `rho` outside the special-cased 1 and 2 (it integrated to -8.82 at the default
  `rho0=1.5`). Affected `numpy_impl.viz.plot_pdf_fits`; the fit path was never
  affected, as it uses its own log-space implementation (#136).

## 0.1.2

Outlier-rejection parity in the NumPy backend, repo-wide type-checking, and the
full validation-evidence documentation.

- NumPy backend outlier rejection: the Fortran `do_reject` outlier-rejection path
  is ported to `AMICA_NumPy` via the same `good_idx` mechanism as the PyTorch
  backend, so the NumPy reference now drops per-sample outliers on the
  `rejstart`/`rejint`/`maxrej` schedule (#123).
- Rejection robustness: a non-finite log-likelihood is now distinguished from an
  over-aggressive `rejsig`, so an over-tight rejection threshold fails with a
  clear message instead of a silent non-finite result (#127).
- Type checking enforced: repo-wide `ty` diagnostics fixed (496 to 0) and `ty`
  added to CI alongside a pre-commit config (ruff + ty) (#124, #125).
- Documentation: the validation guide is expanded into a full evidence page,
  source-density bit-exactness, cross-platform device/precision invariance
  (cross-backend equivalence matrix and IC topomaps), the EEGLAB drop-in
  round-trip, and the other validated behaviors (#108).

## 0.1.1

Validation-methodology and correctness fixes since 0.1.0.

- Amari distance: a second, permutation- and scale-invariant unmixing-matrix
  comparison metric (Amari, Cichocki & Yang 1996) alongside Hungarian-matched
  correlation, used throughout the Fortran-parity validation (#120).
- Multi-model equivalence test: switched to a valid run-level permutation test
  that respects the dependence among the 40 runs' pairwise correlations, instead
  of a pseudoreplicated Mann-Whitney/TOST (#115).
- Parity and performance tables added to the paper, with the full results,
  native-Fortran CPU core-scaling rows, and per-run detail in the docs (#112).
- Type-safety fixes in `validate_implementations.py` (`run_fortran_amica`
  return type, `load_eeglab_data` dtype annotation) (#118).
- JOSS draft-PDF build workflow, `.zenodo.json` with ROR-based citation
  metadata, and an MLX backend API reference page (#110, #105, #107).
- Corrected a stale float32-speedup claim and added a funding acknowledgment
  (#114).

## 0.1.0

First public release.

- PyTorch natural-gradient EM backend (`AMICATorchNG`) at Fortran parity on real
  EEG (single-model log-likelihood ~ -3.40, Hungarian-matched component
  correlation ~ 0.997).
- Backends: CPU, NVIDIA GPU (CUDA), and Apple GPU (MLX); float64 for parity,
  float32 for speed.
- All five source-density families, mixture of ICA models, Newton updates,
  component sharing, and outlier rejection.
- EEGLAB drop-in output: `write_amica_output` writes the `loadmodout15` format,
  and `variance_order` gives the EEGLAB back-projected-variance component order.
- Spatially-distributed channel-subset selection and a data-size (k-factor)
  cross-backend equivalence sweep for the benchmarks.
- scikit-learn-style `AMICA` interface, save/load, and a documentation site.
