"""Convergence-stop tests for AMICATorchNG (issue #207).

Real bundled sample EEG only -- no synthetic/mock data, per project policy.
Covers the three gaps identified against the Fortran reference (amica15.f90,
the actual source of the validated ``amica15mac`` binary):

1. ``use_min_dll``/``min_dll``/``maxincs`` (amica15.f90:1078-1090): stop after
   more than ``maxincs`` *consecutive* iterations whose likelihood gain is
   below ``min_dll``, resetting the counter on any larger gain.
2. ``use_grad_norm``/``min_nd`` (amica15.f90:1091-1097): stop once the
   weight-gradient RMS norm (``ndtmpsum``) falls to or below ``min_nd``,
   independent of whether the likelihood just decreased.
3. The lrate-decrease branch's missing ``.or. (ndtmpsum .le. min_nd)`` half
   (amica15.f90:1058) -- ``stop_reason="grad_norm_floor"``, distinct from the
   unconditional ``"grad_norm"`` stop above.

Several tests cross-check the real-data trajectory against an independent
reimplementation of the Fortran counting rule (``_predict_min_dll_stop``/
``_predict_grad_norm_stop``) applied post-hoc to a *disabled-stops* reference
run, rather than hardcoding "magic" iteration numbers: this proves the
implementation matches the rule, not just that it stops somewhere plausible.

Also covers PR #213 review findings on top of the original issue #207 suite:
``stop_reason`` shadowing under the shipped True/True defaults (grad_norm vs.
grad_norm_floor/min_dll); do_reject combined with each standalone stop; a
genuine keep_best overshoot restore under a new stop reason (not a monotone
trajectory); the five new config keys round-tripping through
state_dict()/from_state_dict(), including a simulated pre-#207 payload;
``AMICA.save()``/``load()`` actually exercised, not just claimed; a stop
reachable at the literal shipped default thresholds; and (folding in issue
#161) ``mir_history_`` surviving a keep_best restore untouched and coming
back empty after save/load.
"""

import math
from pathlib import Path
from typing import Any, Optional
from unittest import mock

import numpy as np
import pytest
import torch

from pamica.amica import AMICA
from pamica.torch_impl.core import AMICATorchNG

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "sample_data"
DATA_FILE = SAMPLE_DIR / "eeglab_data.fdt"
NW = 32
FIELD = 30504


@pytest.fixture(scope="module")
def real_data() -> np.ndarray:
    if not DATA_FILE.exists():
        pytest.skip("sample data missing")
    from pamica.torch_impl.utils import load_eeglab_data

    return load_eeglab_data(str(DATA_FILE), data_dim=NW, field_dim=FIELD).astype(
        np.float64
    )


def _fresh_ng(**kwargs: Any) -> AMICATorchNG:
    kwargs.setdefault("n_channels", NW)
    kwargs.setdefault("n_models", 1)
    kwargs.setdefault("n_mix", 3)
    kwargs.setdefault("device", "cpu")
    kwargs.setdefault("dtype", torch.float64)
    kwargs.setdefault("block_size", 512)
    return AMICATorchNG(**kwargs)


def _predict_min_dll_stop(
    ll_history: list, min_dll: float, maxincs: int
) -> Optional[int]:
    """Independent reimplementation of the Fortran maxincs consecutive-small-
    gain rule (amica15.f90:1078-1090), applied post-hoc to a COMPLETE LL
    trajectory (e.g. from a stops-disabled reference run). Returns the
    0-indexed ``ll_history`` position at which the stop would fire, or None.
    """
    numincs = 0
    for i in range(1, len(ll_history)):
        if ll_history[i] - ll_history[i - 1] < min_dll:
            numincs += 1
            if numincs > maxincs:
                return i
        else:
            numincs = 0
    return None


def _predict_grad_norm_stop(nd_history: list, min_nd: float) -> Optional[int]:
    """Independent reimplementation of the Fortran use_grad_norm rule
    (amica15.f90:1091-1097). ``nd_history[i]`` is ndtmpsum computed during
    fit-loop iteration ``i`` (0-indexed); the check only applies once a
    previous LL exists, i.e. i >= 1. Returns the first qualifying index, or
    None."""
    for i in range(1, len(nd_history)):
        if nd_history[i] <= min_nd:
            return i
    return None


def _fit_with_nd_history(ng: AMICATorchNG, data: np.ndarray, **fit_kwargs) -> list:
    """Fit ``ng``, recording ``ndtmpsum`` after every ``_update_parameters``
    call. This is an observational spy on the real method (calls straight
    through via ``side_effect``, does not replace or bypass any computation)
    -- the same pattern ``test_ng_backend.py`` already uses to inspect
    intermediate state (e.g. ``test_newton_finalize_uses_preupdate_mu``)."""
    nd_history: list = []
    original = ng._update_parameters

    def spy(acc, n):
        result = original(acc, n)
        nd_history.append(ng._ndtmpsum)
        return result

    with mock.patch.object(ng, "_update_parameters", side_effect=spy):
        ng.fit(data, verbose=False, **fit_kwargs)
    return nd_history


# --- gap 1: use_min_dll / maxincs (consecutive-count, reset-on-larger-gain) -


def test_min_dll_stop_matches_independent_reference(real_data):
    """The min_dll stop must fire at EXACTLY the iteration an independent
    reimplementation of the Fortran counting rule predicts from a
    stops-disabled reference trajectory on the same data/seed/config -- and
    the two runs' LL trajectories must be bit-identical up to that point
    (enabling the stop must not perturb the M-step math, only when the loop
    ends). The seed/newt_start/min_dll/maxincs here were chosen (by probing
    real data) to exercise a genuine reset: the likelihood gain dips below
    min_dll, recovers above it for several iterations as Newton switches on,
    then decays below it again for long enough to trip maxincs -- so this
    single scenario also covers the "reset on larger gain" path, not just a
    monotone approach to the threshold.
    """
    x = real_data[:, :8192]
    common = dict(seed=42, do_newton=True, newt_start=5)
    min_dll, maxincs = 0.0025, 3

    reference = _fresh_ng(use_min_dll=False, use_grad_norm=False, **common)
    reference.fit(x, max_iter=40, verbose=False)
    predicted = _predict_min_dll_stop(reference.ll_history, min_dll, maxincs)
    assert predicted is not None, "test setup: reference run never dips/recovers"

    stopped = _fresh_ng(
        use_min_dll=True,
        min_dll=min_dll,
        maxincs=maxincs,
        use_grad_norm=False,
        **common,
    )
    stopped.fit(x, max_iter=40, verbose=False)

    assert stopped.stop_reason == "min_dll"
    assert len(stopped.ll_history) == predicted + 1
    assert stopped.ll_history == reference.ll_history[: predicted + 1]


def test_min_dll_never_fires_before_two_ll_values(real_data):
    """An absurdly generous min_dll (any gain at all counts as "small") with
    maxincs=0 would fire on the FIRST qualifying iteration if the have_prev
    guard were missing -- Fortran's whole check block is nested inside
    ``if (iter > 1)`` (amica15.f90:1051), so it cannot fire while only one LL
    value exists."""
    ng = _fresh_ng(
        seed=42, use_min_dll=True, min_dll=1e6, maxincs=0, use_grad_norm=False
    )
    ng.fit(real_data[:, :4096], max_iter=5, verbose=False)
    # With min_dll=1e6 every gain from iteration 2 onward is "small", and
    # maxincs=0 means a single such iteration already exceeds it: the stop
    # must fire at the second LL value (ll_history length 2), never at the
    # first (which would indicate the have_prev guard is missing).
    assert ng.stop_reason == "min_dll"
    assert len(ng.ll_history) == 2


# --- gap 2: use_grad_norm / min_nd (unconditional per-iteration check) -----


def test_grad_norm_stop_matches_independent_reference(real_data):
    """The standalone grad_norm stop fires at exactly the iteration an
    independent scan of a stops-disabled ndtmpsum trajectory predicts, and
    leaves the LL trajectory bit-identical up to that point."""
    x = real_data[:, :8192]
    common = dict(seed=42, do_newton=False)
    min_nd = 0.02

    reference = _fresh_ng(use_min_dll=False, use_grad_norm=False, **common)
    nd_history = _fit_with_nd_history(reference, x, max_iter=20)
    predicted = _predict_grad_norm_stop(nd_history, min_nd)
    assert predicted is not None, "test setup: ndtmpsum never crosses min_nd"

    stopped = _fresh_ng(use_min_dll=False, use_grad_norm=True, min_nd=min_nd, **common)
    stopped.fit(x, max_iter=20, verbose=False)

    assert stopped.stop_reason == "grad_norm"
    assert len(stopped.ll_history) == predicted + 1
    assert stopped.ll_history == reference.ll_history[: predicted + 1]


def test_grad_norm_never_fires_on_first_iteration(real_data):
    """Same have_prev guard as min_dll (amica15.f90:1051): an absurdly
    generous min_nd must not fire while only one LL value exists."""
    ng = _fresh_ng(seed=42, use_min_dll=False, use_grad_norm=True, min_nd=1e6)
    ng.fit(real_data[:, :4096], max_iter=5, verbose=False)
    assert ng.stop_reason == "grad_norm"
    assert len(ng.ll_history) == 2


# --- gap 3: the decrease-branch's ``.or. ndtmpsum <= min_nd`` half ---------


def test_grad_norm_floor_fires_on_likelihood_decrease(real_data):
    """The reported bug (issue #207): under do_newton, lrate can sit at
    newtrate/oscillate instead of annealing toward minlrate, so the OLD
    ``lrate <= minlrate``-only check inside the decrease branch could never
    fire. This config (probed on real data) produces a genuine likelihood
    decrease at a known iteration while lrate is nowhere near minlrate; with
    use_grad_norm off (isolating this from the separate unconditional check)
    and a generous min_nd, the decrease branch must still stop via the new
    ``grad_norm_floor`` half -- proving gap 3 is fixed independent of gap 2.
    """
    x = real_data[:, :4096]
    ng = _fresh_ng(
        seed=1,
        do_newton=True,
        newt_start=2,
        newtrate=3.0,
        lrate=0.3,
        use_min_dll=False,
        use_grad_norm=False,
        min_nd=1.0,
    )
    ng.fit(x, max_iter=30, verbose=False)

    assert ng.stop_reason == "grad_norm_floor"
    assert ng.lrate > ng.minlrate  # confirms this is NOT the lrate_floor path
    # The stop must coincide with an actual LL decrease (the branch it lives
    # in), not an arbitrary iteration.
    assert ng.ll_history[-1] < ng.ll_history[-2]


def test_lrate_floor_still_reachable_without_grad_norm(real_data):
    """Regression guard: the pre-existing lrate_floor half of the decrease
    branch (amica15.f90:1058's first disjunct) must still fire on its own
    when grad_norm is nowhere near its floor -- gap 3 only ADDS a disjunct,
    it must not shadow the original condition. Same seed/config as
    ``test_grad_norm_floor_fires_on_likelihood_decrease`` (a known real-data
    decrease at iteration 23, lrate=2.4 there), but with ``minlrate`` raised
    above that lrate instead of ``min_nd`` raised above ndtmpsum -- the two
    tests together prove the ``or`` in Fortran's ``(lrate <= minlrate) .or.
    (ndtmpsum <= min_nd)`` (amica15.f90:1058) works from either side."""
    ng = _fresh_ng(
        seed=1,
        do_newton=True,
        newt_start=2,
        newtrate=3.0,
        lrate=0.3,
        minlrate=3.0,  # above the lrate (2.4) at the known decrease point
        use_min_dll=False,
        use_grad_norm=False,
        min_nd=0.0,  # ndtmpsum is never <= 0, so only lrate<=minlrate can fire
    )
    ng.fit(real_data[:, :4096], max_iter=30, verbose=False)
    assert ng.stop_reason == "lrate_floor"
    assert ng.lrate <= ng.minlrate
    # The exact crossing iteration (23 on macOS-arm64) is a snapshot, not an
    # invariant: the same claim elsewhere in this file varied 326 -> 1076 across
    # BLAS implementations. Assert the behaviour, which is that the stop fired
    # before the budget was exhausted, not the iteration it happened on.
    assert len(ng.ll_history) < 30


# --- stop_reason precedence (PR #213 review finding 2) ---------------------
#
# None of the three fit()-loop stop blocks (decrease branch; min_dll;
# grad_norm) short-circuits on an earlier one having already fired this same
# iteration -- Fortran has the same structure (independent leave=.true.
# assignments, no declared precedence), so this is not a fidelity bug. But it
# does mean that whichever block runs LAST and finds its own condition true
# wins. Fixed source order is: decrease branch, then min_dll, then grad_norm.
# The standalone grad_norm check (last) tests a strict superset of the
# decrease-branch's grad_norm_floor condition (ndtmpsum <= min_nd, without
# requiring a coincident decrease), so under the shipped use_grad_norm=True
# default, "grad_norm" always wins and "grad_norm_floor" is unreachable as a
# FINAL stop_reason. See the corrected use_grad_norm docstring in
# torch_impl/core.py.


def test_grad_norm_shadows_grad_norm_floor_under_shipped_defaults(real_data):
    """Same seed/config as ``test_grad_norm_floor_fires_on_likelihood_decrease``
    (which isolates the decrease-branch's grad_norm_floor half by setting
    ``use_grad_norm=False``), but with that one override removed so both
    stops sit at their shipped ``True`` defaults. The standalone grad_norm
    check (amica15.f90:1091-1097) does not require a likelihood decrease --
    only ``ndtmpsum <= min_nd`` -- so it fires as soon as that threshold is
    crossed, which happens well before the later iteration where a genuine
    decrease would let grad_norm_floor's narrower condition also become true.
    The two tests together prove the shadowing directly (identical setup,
    opposite stop_reason, depending only on ``use_grad_norm``) rather than by
    argument alone.
    """
    ng = _fresh_ng(
        seed=1,
        do_newton=True,
        newt_start=2,
        newtrate=3.0,
        lrate=0.3,
        min_nd=1.0,
        # use_min_dll / use_grad_norm left at their True shipped defaults.
    )
    ng.fit(real_data[:, :4096], max_iter=30, verbose=False)
    assert ng.stop_reason == "grad_norm"


# --- share_comps interaction: ndtmpsum must not go stale during the freeze -


def test_a_frozen_window_still_computes_fresh_grad_norm(real_data):
    """The comp_used mask in ndtmpsum only matters once share_comps has
    merged/frozen columns, so this is the one scenario that actually
    exercises it. Separately, and more fundamentally: Fortran computes
    dAk/ndtmpsum unconditionally every iteration in
    accum_updates_and_likelihood (amica15.f90:1749-1761), strictly BEFORE the
    later, freeze-gated update_A block that actually steps A
    (amica15.f90:1803). ``_update_parameters`` was refactored so the
    direction/dAk/ndtmpsum computation runs unconditionally too, with only
    the step itself gated on ``not self._a_frozen()``. This test proves both
    halves of that refactor: the mixing matrix's per-column DIRECTION does not
    move during a frozen iteration (only ``doscaling``'s unconditional
    unit-norm rescale touches its magnitude -- that rescale is a separate,
    non-frozen Fortran block, amica15.f90:1843-1854, so it is expected to
    still apply), AND ndtmpsum is NOT a stale repeat of the pre-freeze value
    across those same iterations (which would happen if the computation were
    still skipped).
    """
    ng = _fresh_ng(
        n_models=2,
        seed=3,
        do_newton=True,
        block_size=1024,
        share_comps=True,
        share_start=8,
        share_iter=10,
        comp_thresh=0.9,
        use_min_dll=False,
        use_grad_norm=False,
    )
    trace = []
    original = ng._update_parameters

    def spy(acc, n):
        assert ng.A is not None
        frozen = ng._a_frozen()
        a_before = ng.A.clone()
        result = original(acc, n)
        # Direction check robust to doscaling's per-column rescale: normalize
        # both snapshots to unit columns before comparing, so only an actual
        # gradient step (not a magnitude rescale) can fail this.
        before_dir = a_before / a_before.norm(dim=0, keepdim=True)
        after_dir = ng.A / ng.A.norm(dim=0, keepdim=True)
        trace.append(
            {
                "frozen": frozen,
                "direction_changed": not torch.allclose(
                    before_dir, after_dir, atol=1e-12
                ),
                "ndtmpsum": ng._ndtmpsum,
            }
        )
        return result

    with mock.patch.object(ng, "_update_parameters", side_effect=spy):
        ng.fit(real_data[:, :4096], max_iter=16, verbose=False)

    frozen_iters = [t for t in trace if t["frozen"]]
    assert frozen_iters, "test setup: no frozen iteration occurred in this run"
    for t in frozen_iters:
        assert not t["direction_changed"]
        assert t["ndtmpsum"] is not None and math.isfinite(t["ndtmpsum"])
    # Fresh per-iteration values, not the same stale number repeated.
    assert len({round(v, 15) for v in (t["ndtmpsum"] for t in frozen_iters)}) > 1
    # comp_used is exercised: the merge dropped at least one component.
    assert int(ng.comp_used.sum()) < ng.n_comps


# --- usability after a converged stop (transform/save/state_dict) ---------


def test_new_stop_reasons_are_not_degenerate():
    for reason in ("min_dll", "grad_norm", "grad_norm_floor"):
        assert reason not in AMICATorchNG._DEGENERATE_STOP_REASONS


def test_min_dll_stop_leaves_backend_usable(real_data, tmp_path):
    """transform()/state_dict()/from_state_dict() all work after a min_dll
    stop -- a converged stop is not a degenerate one (issue #50 contract)."""
    x = real_data[:, :8192]
    ng = _fresh_ng(
        seed=42,
        do_newton=True,
        newt_start=5,
        use_min_dll=True,
        min_dll=0.0025,
        maxincs=3,
        use_grad_norm=False,
    )
    ng.fit(x, max_iter=40, verbose=False)
    assert ng.stop_reason == "min_dll"

    sources = ng.transform(x, model_idx=0)
    assert np.isfinite(sources).all()

    state = ng.state_dict()
    loaded = AMICATorchNG.from_state_dict(state, device="cpu")
    assert loaded.stop_reason == "min_dll"
    assert loaded.A is not None and ng.A is not None
    assert torch.equal(loaded.A.cpu(), ng.A.cpu())


def test_grad_norm_floor_stop_leaves_wrapper_usable(real_data, tmp_path):
    """End-to-end through the AMICA wrapper: a grad_norm_floor stop must
    leave converged_/is_fitted_ True and transform()/save() usable, exactly
    like the pre-existing lrate_floor path -- the CRITICAL requirement that
    these are CONVERGED stops, not degenerate ones.

    ``save()`` is exercised for real here (review finding, PR #213: earlier
    revisions of this module claimed "transform()/save() usable" in this very
    docstring but never actually called ``AMICA.save()`` anywhere in the
    file)."""
    x = real_data[:, :4096]
    model = AMICA(n_models=1, n_mix=3, device="cpu", verbose=False)
    model.fit(
        x,
        max_iter=30,
        seed=1,
        do_newton=True,
        newt_start=2,
        newtrate=3.0,
        lrate=0.3,
        use_min_dll=False,
        use_grad_norm=False,
        min_nd=1.0,
        dtype=torch.float64,
    )
    assert model.stop_reason_ == "grad_norm_floor"
    assert model.converged_ is True
    assert model.is_fitted_ is True
    S = model.transform(x)
    assert np.isfinite(S).all()

    save_path = tmp_path / "grad_norm_floor_model.pt"
    model.save(str(save_path))
    loaded = AMICA.load(str(save_path))
    assert loaded.stop_reason_ == "grad_norm_floor"
    assert loaded.converged_ is True
    assert loaded.is_fitted_ is True
    S_loaded = loaded.transform(x)
    np.testing.assert_array_equal(S_loaded, S)


# --- keep_best / do_reject interaction (early stopping must not break them) -


def test_keep_best_restores_genuine_overshoot_under_min_dll_stop(real_data):
    """keep_best (issue #51) must actually exercise its restore branch, not
    just trivially satisfy ``final_ll_ == max(ll_history) == ll_history[-1]``
    on a monotone trajectory -- review finding (PR #213): the previous
    version of this test used a monotone config, under which those equalities
    hold identically whether the restore logic works or is a no-op, so it
    proved nothing about the restore itself.

    This config is the known non-monotone recipe from
    ``test_write_amica_output_ll_matches_kept_iterate`` (issue #92,
    ``test_amica_ng_wrapper.py``: real 2-model data, aggressive
    ``do_newton``/``lrate``), combined with a loosened ``min_dll`` so the run
    stops via the NEW ``min_dll`` stop_reason a few iterations after its
    peak, not via ``max_iter`` and not via a monotone approach to that peak.
    """
    x = real_data[:, :4096]
    ng = _fresh_ng(
        n_models=2,
        seed=0,
        do_newton=True,
        newt_start=1,
        lrate=0.5,
        block_size=1024,
        use_min_dll=True,
        min_dll=1e-4,
        maxincs=2,
        use_grad_norm=False,
        keep_best=True,
    )
    ng.fit(x, max_iter=60, verbose=False)
    assert ng.stop_reason == "min_dll"
    assert ng.final_ll_ == max(ng.ll_history)
    assert ng.final_ll_ in ng.ll_history
    # The genuine-overshoot proof: the restore branch only overwrites
    # final_ll_ when the run ends materially below its peak (issue #51), so
    # final_ll_ != ll_history[-1] is only reachable if that branch actually
    # ran -- it cannot happen on a monotone trajectory or a no-op restore.
    assert ng.final_ll_ != ng.ll_history[-1]
    assert ng.final_ll_ > ng.ll_history[-1]
    assert ng.ll_history.index(ng.final_ll_) < len(ng.ll_history) - 1


def test_do_reject_interaction_grad_norm_floor_stop_leaves_good_idx_usable(real_data):
    """do_reject changes numgoodsum (and so the LL normalization) every
    rejection pass; a converged stop under do_reject must still leave
    good_idx/A/transform usable, and keep_best's do_reject-inactive warning
    must not prevent the fit from completing normally.

    NOTE (review finding, PR #213): this test sets ``use_min_dll=False``, so
    it exercises the decrease-branch's ``grad_norm_floor`` half (gap 3), not
    ``min_dll`` -- renamed from
    ``test_do_reject_interaction_min_dll_stop_leaves_good_idx_usable`` to
    match. See the two tests below for do_reject combined with the standalone
    ``min_dll`` and ``grad_norm`` stops, which were the genuinely missing
    coverage.
    """
    x = real_data[:, :8192]
    ng = _fresh_ng(
        seed=1,
        do_reject=True,
        rejstart=2,
        rejint=3,
        maxrej=2,
        do_newton=True,
        newt_start=2,
        newtrate=3.0,
        lrate=0.3,
        use_min_dll=False,
        use_grad_norm=False,
        min_nd=1.0,
    )
    ng.fit(x, max_iter=30, verbose=False)

    assert ng.stop_reason == "grad_norm_floor"
    assert ng.good_idx is not None and int(ng.good_idx.numel()) < x.shape[1]
    sources = ng.transform(x, model_idx=0)
    assert np.isfinite(sources).all()
    state = ng.state_dict()  # must not raise (issue #50 usable-model contract)
    assert state["extra"]["stop_reason"] == "grad_norm_floor"


def test_do_reject_interaction_min_dll_stop_leaves_good_idx_usable(real_data):
    """do_reject combined with the standalone min_dll stop (gap 1): the
    genuinely missing coverage flagged alongside the rename above. Same
    do_reject schedule as the grad_norm_floor variant, with the min_dll
    config from ``test_min_dll_stop_matches_independent_reference``."""
    x = real_data[:, :8192]
    ng = _fresh_ng(
        seed=42,
        do_reject=True,
        rejstart=2,
        rejint=3,
        maxrej=2,
        do_newton=True,
        newt_start=5,
        use_min_dll=True,
        min_dll=0.0025,
        maxincs=3,
        use_grad_norm=False,
    )
    ng.fit(x, max_iter=40, verbose=False)

    assert ng.stop_reason == "min_dll"
    assert ng.good_idx is not None and int(ng.good_idx.numel()) < x.shape[1]
    sources = ng.transform(x, model_idx=0)
    assert np.isfinite(sources).all()
    state = ng.state_dict()  # must not raise (issue #50 usable-model contract)
    assert state["extra"]["stop_reason"] == "min_dll"


def test_do_reject_interaction_grad_norm_stop_leaves_good_idx_usable(real_data):
    """do_reject combined with the standalone grad_norm stop (gap 2): the
    other genuinely missing coverage flagged alongside the rename above. Same
    do_reject schedule as the two variants above, with the grad_norm config
    from ``test_grad_norm_stop_matches_independent_reference``."""
    x = real_data[:, :8192]
    ng = _fresh_ng(
        seed=42,
        do_reject=True,
        rejstart=2,
        rejint=3,
        maxrej=2,
        do_newton=False,
        use_min_dll=False,
        use_grad_norm=True,
        min_nd=0.02,
    )
    ng.fit(x, max_iter=20, verbose=False)

    assert ng.stop_reason == "grad_norm"
    assert ng.good_idx is not None and int(ng.good_idx.numel()) < x.shape[1]
    sources = ng.transform(x, model_idx=0)
    assert np.isfinite(sources).all()
    state = ng.state_dict()  # must not raise (issue #50 usable-model contract)
    assert state["extra"]["stop_reason"] == "grad_norm"


# --- disabled-by-default-off regression: both stops off is unaffected ------


def test_both_stops_disabled_never_produce_new_stop_reasons(real_data):
    """With both stops explicitly off, stop_reason must never be one of the
    three new values, even on a long, non-monotone (do_newton) real run --
    the disabled path stays exactly what it was before issue #207 added the
    checks (only max_iter/lrate_floor/nan_ll/singular_ll remain reachable)."""
    ng = _fresh_ng(
        seed=1,
        do_newton=True,
        newt_start=2,
        newtrate=3.0,
        lrate=0.3,
        use_min_dll=False,
        use_grad_norm=False,
    )
    ng.fit(real_data[:, :4096], max_iter=30, verbose=False)
    assert ng.stop_reason not in ("min_dll", "grad_norm", "grad_norm_floor")


def test_disabled_stops_default_path_matches_max_iter_on_short_run(real_data):
    """On a short, well-behaved (no do_newton) run with both stops at their
    Fortran-faithful defaults, neither is expected to fire in practice
    (real per-iteration LL gains and gradient norms stay far above 1e-9/1e-7
    this early -- see issue #207 investigation notes): the default-enabled
    path is behaviorally a no-op here, landing on max_iter exactly like the
    pre-#207 backend would have. This is the "no regression at realistic
    short budgets" complement to the explicit-disable tests above."""
    ng = _fresh_ng(seed=42, do_newton=False)  # defaults: both stops True
    ng.fit(real_data[:, :8192], max_iter=15, verbose=False)
    assert ng.stop_reason == "max_iter"
    assert len(ng.ll_history) == 15


# --- production-realistic thresholds (PR #213 review finding 7) ------------


def test_min_dll_stop_reachable_at_shipped_default_threshold(real_data):
    """Every min_dll/grad_norm test above loosens min_dll/min_nd by 5-6 orders
    of magnitude to force a fast stop in a short budget. That is fine for
    exercising the counting/gating logic, but it would not catch a scale bug
    in the comparison itself -- e.g. comparing a raw, un-normalized
    log-likelihood against min_dll instead of the normalized per-sample-
    channel value the docstring promises (``min_dll``'s own docstring flags
    this as a live concern: issue #212 found exactly this bug in the
    ``numpy_impl`` backend, whose un-normalized ``self.ll`` means its
    ``min_dll`` can never fire). A wrongly-scaled comparison in
    ``AMICATorchNG`` would still pass every other test in this file, since
    they all use thresholds loose enough to fire regardless of scale.

    This test uses the LITERAL shipped defaults -- ``min_dll=1e-9``,
    ``maxincs=5``, ``use_grad_norm=True``, ``min_nd=1e-7`` -- none
    overridden -- and asserts a stop is reachable at all, on real data, in a
    bounded budget. It is NOT ``@pytest.mark.slow``: it is the single slowest
    test in this module (~8s, the 326-iteration Newton fit itself, not test
    overhead) but still finishes in single-digit seconds, nowhere near the
    "thousands of iterations" this finding anticipated needing (issue #207
    review finding 1 -- ``slow`` means "invokes the macOS-only Fortran
    binary", not "takes a while", so a few extra seconds does not qualify).

    Reaching the default threshold organically needed real (not synthetic)
    tuning of WHICH real-data config gets there fast: the PR's own
    investigation (see the PR #213 description and
    ``test_disabled_stops_default_path_matches_max_iter_on_short_run`` above)
    found that ``newt_start=20`` at longer budgets (hundreds to thousands of
    iterations) does NOT reach either default threshold on this 32-channel
    sample -- per-iteration LL gains decay roughly like O(1/iter), not fast
    enough. Starting Newton much earlier (``newt_start=5``) on a smaller
    sample subset lets Newton's local quadratic convergence take over almost
    immediately, so consecutive gains cross 1e-9 within a few hundred
    iterations instead. This is a legitimate real-data config choice (same
    kind of seed/newt_start probing the other tests in this module already
    document doing), not a threshold change.

    The budget is deliberately generous. The iteration at which the stop fires
    varies by more than 3x with the BLAS in use: measured at 326 on macOS-arm64,
    412 on Linux-x86_64 with a CUDA-enabled torch build, and 1076 on the GitHub
    Linux runner. An earlier version of this test used ``max_iter=500`` and
    failed CI twice, first on the stop reason and then on a leftover
    ``len(ll_history) < 500`` bound (PR #213). The claim under test is that the
    default threshold is reachable at all, not that it is reached by any
    particular iteration, so both the budget and the bound track the budget
    rather than a constant fitted to one machine.
    """
    ng = _fresh_ng(seed=1, do_newton=True, newt_start=5, block_size=1024)
    ng.fit(real_data[:, :4096], max_iter=2000, verbose=False)
    assert ng.stop_reason == "min_dll"
    # A real early stop rather than exhausting the budget. Bound against the
    # budget itself, not a fixed number: the firing iteration is BLAS-dependent
    # (see docstring) and any constant below the budget is a platform trap.
    assert len(ng.ll_history) < 2000


# --- persistence: issue #207 config keys (PR #213 review finding 6) --------


def test_convergence_config_round_trips_through_state_dict(real_data):
    """The five issue #207 config keys (use_min_dll/min_dll/maxincs/
    use_grad_norm/min_nd) must persist through state_dict()/from_state_dict()
    like every other constructor argument (issue #36 persistence contract),
    not just the fitted parameter tensors. Non-default values throughout so a
    bug that silently fell back to the constructor defaults would be caught."""
    ng = _fresh_ng(
        seed=1,
        use_min_dll=False,
        min_dll=1e-4,
        maxincs=2,
        use_grad_norm=False,
        min_nd=1e-3,
    )
    ng.fit(real_data[:, :2048], max_iter=3, verbose=False)
    state = ng.state_dict()
    assert state["config"]["use_min_dll"] is False
    assert state["config"]["min_dll"] == 1e-4
    assert state["config"]["maxincs"] == 2
    assert state["config"]["use_grad_norm"] is False
    assert state["config"]["min_nd"] == 1e-3

    loaded = AMICATorchNG.from_state_dict(state, device="cpu")
    assert loaded.use_min_dll is False
    assert loaded.min_dll == 1e-4
    assert loaded.maxincs == 2
    assert loaded.use_grad_norm is False
    assert loaded.min_nd == 1e-3


def test_missing_convergence_keys_fall_back_to_fortran_defaults(real_data):
    """A state_dict payload saved before issue #207 has ``format_version==3``
    (deliberately not bumped -- see the comment at the ``format_version``
    check in ``AMICATorchNG.from_state_dict``) but no
    use_min_dll/min_dll/maxincs/use_grad_norm/min_nd keys in its ``config``.
    ``from_state_dict`` must still load such a payload, falling back to the
    constructor's Fortran-faithful defaults for those five keys -- not
    raising, and not silently defaulting to some other value."""
    ng = _fresh_ng(seed=1, use_min_dll=True, use_grad_norm=True)
    ng.fit(real_data[:, :2048], max_iter=3, verbose=False)
    state = ng.state_dict()
    for key in ("use_min_dll", "min_dll", "maxincs", "use_grad_norm", "min_nd"):
        del state["config"][key]  # simulate a pre-#207 payload

    loaded = AMICATorchNG.from_state_dict(state, device="cpu")
    assert loaded.use_min_dll is True
    assert loaded.min_dll == 1e-9
    assert loaded.maxincs == 5
    assert loaded.use_grad_norm is True
    assert loaded.min_nd == 1e-7


# --- mir_history_ vs keep_best / save-load (issue #161) --------------------


def test_mir_history_survives_keep_best_restore(real_data):
    """Issue #161 claim 1: ``mir_history_`` is a TRUE trajectory that a
    keep_best restore (issue #51) does not rewrite, so its last entry can be
    (and here, is) computed from the pre-restore, discarded parameters --
    NOT the restored parameters ``fit`` actually returns. The docstrings'
    corollary is that the fit-end MIR is ``model.mir(X)`` (computed from the
    returned parameters), never ``mir_history_[-1]``.

    Uses the same genuine-overshoot recipe as
    ``test_keep_best_restores_genuine_overshoot_under_min_dll_stop`` above,
    with ``mir_step=1`` added.

    ``mir_step=1`` is load-bearing, not incidental. The ``min_dll``/``maxincs``
    stop halts one or two iterations past the peak, so with any coarser step
    the last waypoint lands *before* the best iterate and the window a
    truncating restore would damage is never sampled -- a restore that dropped
    every waypoint after the best iterate would leave this fixture unchanged
    and the test would pass under the bug. Recording every iteration puts a
    waypoint strictly inside that window."""
    x = real_data[:, :4096]
    ng = _fresh_ng(
        n_models=2,
        seed=0,
        do_newton=True,
        newt_start=1,
        lrate=0.5,
        block_size=1024,
        use_min_dll=True,
        min_dll=1e-4,
        maxincs=2,
        use_grad_norm=False,
        keep_best=True,
    )
    ng.fit(x, max_iter=60, verbose=False, mir_step=1)
    assert ng.stop_reason == "min_dll"
    assert ng.final_ll_ != ng.ll_history[-1]  # the restore branch fired

    assert ng.mir_history_, "test setup: mir_step recorded nothing"
    last_it, last_mir, _ = ng.mir_history_[-1]
    # The trajectory runs to the final (pre-restore) iteration --
    # _snapshot_params/_restore_params never touch mir_history_, so it is
    # neither truncated nor rewritten by the restore that just fired above.
    # At mir_step=1 every iteration is a waypoint, so this holds for any
    # stopping iteration; the earlier equality-with-a-multiple-of-5 form was
    # satisfied by a truncating restore as well as a correct one.
    final_it = len(ng.ll_history) - 1
    assert ng.final_ll_ is not None
    best_it = ng.ll_history.index(ng.final_ll_)
    assert best_it < final_it, (
        "test setup: the restore must discard at least one iteration, or "
        "there is no truncation window to guard"
    )
    assert last_it == final_it, "the post-peak waypoints were dropped"
    # The count is what a partial truncation would move even if the last entry
    # happened to survive.
    assert len(ng.mir_history_) == final_it + 1, (
        "mir_history_ is not the full per-iteration trajectory"
    )

    # model.mir(X) reflects the RESTORED (actually-returned) parameters, and
    # differs from that stale pre-restore waypoint -- confirming the
    # documented distinction is real, not just an absence-of-crash check.
    mir_now, _ = ng.mir(x)
    assert not math.isclose(mir_now, last_mir, rel_tol=1e-6)


def test_mir_history_empty_after_save_load(real_data, tmp_path):
    """Issue #161 claim 2: ``mir_history_`` is not persisted in
    state_dict() (a diagnostic trajectory, not a fitted parameter), so a
    save/load round trip must yield an EMPTY ``mir_history_`` on the reloaded
    model -- not stale trajectory data leaked through some other path."""
    x = real_data[:, :2048]
    model = AMICA(n_models=1, n_mix=3, device="cpu", verbose=False)
    model.fit(x, max_iter=10, seed=1, dtype=torch.float64, mir_step=3)
    assert model.mir_history_, "test setup: mir_step recorded nothing"

    save_path = tmp_path / "mir_history_model.pt"
    model.save(str(save_path))
    loaded = AMICA.load(str(save_path))
    assert loaded.mir_history_ == []
