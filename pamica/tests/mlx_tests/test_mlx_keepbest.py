"""``keep_best`` best-iterate safeguard on the MLX backend -- issue #288, epic
#278 Phase 2 (porting ``AMICATorchNG``'s issue #51).

Apple-Silicon only, real sample EEG (no synthetic/mock), same module guards as
``test_mlx_backend.py``. Every bit-identity comparison in this module runs
INSIDE one test (same process, same machine): MLX float32 is bit-reproducible
run-to-run on one machine but NOT across GPU models (see
``test_mlx_transform.py``'s ``_NOOP_PIN_*`` module comment), so a recorded
literal constant from one development machine would be the wrong thing to pin
here -- comparing two fresh in-process fits sidesteps that risk entirely.

Recipes used below (verified empirically on this backend, not assumed from
the PyTorch analogue -- MLX's float32 trajectory does not have to overshoot
on the same config PyTorch's float64 one does):

* ``_FORCED_RESTORE_KWARGS`` (real EEG, first 4096 samples, ``max_iter=60``):
  ``n_models=2, n_mix=3, seed=0, block_size=1024, do_newton=True,
  newt_start=1, lrate=0.5, use_min_dll=True, min_dll=1e-4, maxincs=2,
  use_grad_norm=False`` -- the same aggressive-Newton config
  ``test_ng_convergence.py::test_keep_best_restores_genuine_overshoot_under_min_dll_stop``
  uses on PyTorch, and it reproduces a genuine overshoot here too: the fit
  stops via ``min_dll`` at iteration 56 (57 recorded LLs), peaks at
  ``ll_history[54]``, and ends ``best_ll - ll_history[-1] ~= 1.676e-4`` below
  that peak (measured on an Apple M4 Pro; the exact float32 value is
  machine-dependent, but the qualitative overshoot -- peak strictly above the
  final two entries -- reproduces across seeds, see
  ``test_keep_best_restores_a_genuine_overshoot`` below for the live
  measurement this module actually asserts against).
* ``_FORCED_RESTORE_PDFTYPE1_KWARGS``: the ``pdftype=1``/``n_mix=1`` analogue
  (seed 2, ``kurt_start=8``) chosen so the adaptive switcher's first kurtosis
  re-evaluation (scheduled at iteration 7) fires strictly AFTER the peak
  iterate (measured at iteration 5) -- the scenario
  ``test_pdftype1_forced_restore_rolls_back_n_kurt_done_with_pdtype`` needs to
  prove the restore rolls ``n_kurt_done`` back in step with ``pdtype``, not
  just the floating-point arrays.

The truncated-refit bit-identity check (both here and in the pdftype=1 test)
uses ``max_iter=argmax`` where ``argmax = int(np.argmax(ll_history))``: a
fresh fit run for exactly that many EM iterations ends holding the parameters
that PRODUCED ``ll_history[argmax]`` (the peak), because ``_fit_once``
computes each iteration's LL from the CURRENT parameters before that
iteration's own M-step runs and the loop for ``max_iter=argmax`` never enters
iteration ``argmax`` at all -- so ``max_iter=argmax+1`` (which DOES run
iteration ``argmax``'s M-step) is one M-step too far and is NOT bit-identical
to the restored snapshot. This was verified empirically before writing the
assertion below, not assumed.

Two more scenarios, added after PR #310 review:

* ``test_keep_best_does_not_rescue_a_diverged_fit_that_peaked_earlier`` covers
  the end-of-fit restore guard's ``stop_reason not in
  _DEGENERATE_STOP_REASONS`` exclusion, which the forced-restore tests above
  never exercise (they always end on a healthy ``min_dll`` stop). Uses the
  sanctioned error-injection subclass pattern (``.rules/testing.md``'s
  "Sanctioned Exception", the same construction as ``test_mlx_restarts.py``'s
  ``_NaNForSeeds``): a subclass that runs the real fit untouched for
  ``nan_after`` iterations -- long enough to capture a genuine, better
  ``best_snapshot`` -- then corrupts the real E-step's ``acc["ll"]`` to force
  a ``nan_ll`` stop. Deliberately NOT the grad_norm-stop or lrate_floor-stop
  variants of a degenerate-adjacent-but-healthy run: the restore guard
  branches on ``stop_reason in _DEGENERATE_STOP_REASONS`` as a set membership
  test, not on which specific reason fired, so ``nan_ll`` alone already
  exercises the branch; hunting a second forcing recipe for a different
  non-degenerate stop would add cost with no new code path covered.
* ``test_keep_best_restart_record_reports_the_restored_iterate`` covers the
  composition of ``keep_best`` with best-of-N restarts (issue #198): the
  ``_FORCED_RESTORE_KWARGS`` recipe run as restart index 0 (seed 0) inside an
  ``n_restarts=2`` search must record ``restart_lls_[0]`` as the RESTORED
  best iterate's LL, not the raw last iterate -- i.e. the same value a
  standalone seed-0 fit's ``final_ll_`` reports, confirmed bit-identical in
  the same process. The second restart (seed 1) is cheap: it converges in
  under 60 iterations on the same recipe with no forcing needed.
"""

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from pamica.mlx_impl import AMICAMLXNG  # noqa: E402  (after the MLX importorskip)
from pamica.mlx_impl.core import _KEEP_BEST_TOL  # noqa: E402

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "sample_data"
DATA_FILE = SAMPLE_DIR / "eeglab_data.fdt"
NW = 32
FIELD = 30504
NMIX = 3
BLOCK = 1024

pytestmark = [
    pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing"),
    pytest.mark.skipif(
        mx.default_device().type != mx.DeviceType.gpu, reason="no Apple GPU"
    ),
]


@pytest.fixture(scope="module")
def real_data() -> np.ndarray:
    from pamica.torch_impl.utils import load_eeglab_data

    return load_eeglab_data(str(DATA_FILE), data_dim=NW, field_dim=FIELD).astype(
        np.float64
    )


def _model(**kwargs: Any) -> AMICAMLXNG:
    params: dict[str, Any] = dict(n_channels=NW, n_mix=NMIX)
    params.update(kwargs)
    return AMICAMLXNG(**params)


# The aggressive-Newton recipe that genuinely overshoots on this backend (see
# the module docstring). n_mix=3, so pdftype stays at its GG default.
_FORCED_RESTORE_KWARGS: dict[str, Any] = dict(
    n_models=2,
    n_mix=NMIX,
    seed=0,
    block_size=BLOCK,
    do_newton=True,
    newt_start=1,
    lrate=0.5,
    use_min_dll=True,
    min_dll=1e-4,
    maxincs=2,
    use_grad_norm=False,
)
_FORCED_RESTORE_MAX_ITER = 60

# The pdftype=1 analogue, tuned so the first adaptive-switch pass lands after
# the peak iterate (module docstring).
_FORCED_RESTORE_PDFTYPE1_KWARGS: dict[str, Any] = dict(
    n_models=2,
    n_mix=1,
    pdftype=1,
    seed=2,
    block_size=BLOCK,
    do_newton=True,
    newt_start=1,
    lrate=0.5,
    use_min_dll=True,
    min_dll=1e-4,
    maxincs=2,
    use_grad_norm=False,
    kurt_start=8,
    num_kurt=5,
    kurt_int=1,
)
_FORCED_RESTORE_PDFTYPE1_MAX_ITER = 60


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_keep_best_defaults_to_true():
    m = _model()
    assert m.keep_best is True


def test_keep_best_off_is_accepted():
    m = _model(keep_best=False)
    assert m.keep_best is False


# ---------------------------------------------------------------------------
# B.1: monotone no-op (the phase's own bit-identity pin)
# ---------------------------------------------------------------------------


def test_monotone_default_fit_is_bit_identical_with_keep_best_on_or_off(real_data):
    """A default (Newton off, gentle lrate) fit never overshoots its own peak,
    so the restore branch's condition (``best_ll - ll_history[-1] >
    _KEEP_BEST_TOL``) is false on every iteration and keep_best is a no-op --
    the SAME invariant issue #24 single-model parity relies on for PyTorch,
    now pinned for MLX. Both fits run fresh in this same process, so this is
    safe against the cross-GPU float32 noise the module docstring describes."""
    x = real_data[:, :4096]
    on = _model(seed=42, block_size=BLOCK, keep_best=True)
    on.fit(x, max_iter=10, verbose=False)
    off = _model(seed=42, block_size=BLOCK, keep_best=False)
    off.fit(x, max_iter=10, verbose=False)

    assert on.stop_reason == off.stop_reason
    assert on.ll_history == off.ll_history
    assert on.final_ll_ == off.final_ll_
    # No restore fired: final_ll_ is exactly the last trajectory value.
    assert on.final_ll_ == on.ll_history[-1]
    for name in AMICAMLXNG._PARAM_ARRAYS:
        a = np.array(getattr(on, name))
        b = np.array(getattr(off, name))
        assert np.array_equal(a, b), f"{name} differs between keep_best on/off"
    assert on.n_kurt_done == off.n_kurt_done


# ---------------------------------------------------------------------------
# _snapshot_params / _restore_params primitives
# ---------------------------------------------------------------------------


def test_snapshot_is_a_copy_not_an_alias(real_data):
    m = _model(seed=7, block_size=BLOCK)
    m.fit(real_data[:, :2048], max_iter=3, verbose=False)

    snap = m._snapshot_params()
    assert m.A is not None
    a_np = np.array(m.A)
    orig = float(a_np[0, 0])
    a_np[0, 0] = orig + 5.0
    m.A = mx.array(a_np)  # the same kind of rebind _update_parameters does

    snap_a = np.array(snap["A"])
    assert float(snap_a[0, 0]) == orig  # the snapshot did not follow the rebind

    m._restore_params(snap)
    assert float(np.array(m.A)[0, 0]) == orig  # restore reverts it
    for name in AMICAMLXNG._PARAM_ARRAYS:
        assert np.array_equal(np.array(getattr(m, name)), np.array(snap[name])), name
    assert m.n_kurt_done == snap["n_kurt_done"]


# ---------------------------------------------------------------------------
# B.2: forced restore
# ---------------------------------------------------------------------------


def test_keep_best_restores_a_genuine_overshoot(real_data):
    """The forcing recipe (module docstring) actually exercises the restore
    branch on MLX float32, not just trivially satisfies the invariants on a
    monotone trajectory (the same genuine-overshoot bar PyTorch's
    ``test_keep_best_restores_genuine_overshoot_under_min_dll_stop`` sets)."""
    x = real_data[:, :4096]
    m = _model(keep_best=True, **_FORCED_RESTORE_KWARGS)
    m.fit(x, max_iter=_FORCED_RESTORE_MAX_ITER, verbose=False)

    assert m.stop_reason == "min_dll"
    peak = max(m.ll_history)
    gap = peak - m.ll_history[-1]
    print(f"forced-restore recipe: best_ll - last_ll = {gap!r}")  # noqa: T201
    assert gap > _KEEP_BEST_TOL, (
        "forcing recipe did not overshoot on this machine/MLX build; the "
        "restore branch was not exercised (see the module docstring for the "
        "recipe this is meant to force)"
    )

    assert m.final_ll_ == peak
    assert m.final_ll_ in m.ll_history
    assert m.final_ll_ != m.ll_history[-1]
    assert m.final_ll_ > m.ll_history[-1]
    argmax = m.ll_history.index(m.final_ll_)
    assert argmax < len(m.ll_history) - 1


def test_forced_restore_model_loglik_matches_state_dict_round_trip(real_data):
    """PR #318 review: ``_snapshot_params``/``_restore_params`` roll back
    ``W``/``rho``/etc, but the MLX-only per-iteration caches ``_logdet_W``
    (feeds ``log|det W|`` directly into every ``logV``) and ``_lgamma_table``
    (feeds the GG log-density) are NOT among ``_PARAM_ARRAYS`` -- so without
    capturing them too, a restore leaves them at the LAST (discarded)
    iterate's values, silently corrupting ``model_loglik``/
    ``model_probability``/``mir`` on the "restored" model even though
    ``W``/``rho`` themselves rolled back correctly.

    The oracle: a ``state_dict()``/``from_state_dict()`` round trip
    independently REBUILDS both caches from the restored ``W``/``rho``
    (``_load_params``, unaffected by this bug), so it is a ground truth the
    live in-memory restored model must agree with. This is the regression
    test itself, not a synthetic unit test of the caches in isolation --
    it fails before the fix and passes after (both measured; see below)."""
    x = real_data[:, :4096]
    m = _model(keep_best=True, **_FORCED_RESTORE_KWARGS)
    m.fit(x, max_iter=_FORCED_RESTORE_MAX_ITER, verbose=False)
    assert m.stop_reason == "min_dll"
    assert m.final_ll_ != m.ll_history[-1], "test setup: the restore did not fire"

    direct = m.model_loglik(x)
    round_tripped = AMICAMLXNG.from_state_dict(m.state_dict()).model_loglik(x)
    max_diff = float(np.abs(direct - round_tripped).max())
    print(
        f"forced-restore model_loglik vs state_dict round trip: max_diff={max_diff!r}"
    )  # noqa: T201, E501
    np.testing.assert_allclose(
        direct,
        round_tripped,
        rtol=0,
        atol=1e-6,
        err_msg=(
            "model_loglik(training X) diverged from the state_dict round-trip "
            "oracle after a keep_best restore -- _logdet_W/_lgamma_table are "
            "likely stale (PR #318 review)"
        ),
    )


def test_keep_best_restore_does_not_rewrite_ll_history(real_data):
    """``ll_history`` stays the true per-iteration trajectory regardless of
    whether a restore fires -- only ``final_ll_``/the parameter arrays roll
    back."""
    x = real_data[:, :4096]
    on = _model(keep_best=True, **_FORCED_RESTORE_KWARGS)
    on.fit(x, max_iter=_FORCED_RESTORE_MAX_ITER, verbose=False)
    off = _model(keep_best=False, **_FORCED_RESTORE_KWARGS)
    off.fit(x, max_iter=_FORCED_RESTORE_MAX_ITER, verbose=False)

    assert max(on.ll_history) - on.ll_history[-1] > _KEEP_BEST_TOL  # restore fired
    assert on.ll_history == off.ll_history  # keep_best never touches the trajectory
    assert off.final_ll_ == off.ll_history[-1]  # return-last is exactly the last
    assert on.final_ll_ is not None and off.final_ll_ is not None
    assert on.final_ll_ > off.final_ll_  # keep_best strictly beat return-last


def test_keep_best_restored_params_match_a_truncated_refit_bit_identically(real_data):
    """The parameters ``fit()`` returns after a restore are bit-identical to a
    FRESH same-seed fit stopped at ``max_iter=argmax`` (module docstring
    explains the off-by-one). Run entirely inside this one test process --
    same machine, same MLX build -- so the comparison is safe against MLX
    float32's cross-GPU non-reproducibility (module docstring)."""
    x = real_data[:, :4096]
    restored = _model(keep_best=True, **_FORCED_RESTORE_KWARGS)
    restored.fit(x, max_iter=_FORCED_RESTORE_MAX_ITER, verbose=False)
    assert max(restored.ll_history) - restored.ll_history[-1] > _KEEP_BEST_TOL
    assert restored.final_ll_ is not None

    argmax = restored.ll_history.index(restored.final_ll_)
    truncated = _model(keep_best=False, **_FORCED_RESTORE_KWARGS)
    truncated.fit(x, max_iter=argmax, verbose=False)

    assert truncated.stop_reason == "max_iter"
    assert len(truncated.ll_history) == argmax
    for name in AMICAMLXNG._PARAM_ARRAYS:
        a = np.array(getattr(restored, name))
        b = np.array(getattr(truncated, name))
        assert np.array_equal(a, b), (
            f"{name}: restored best iterate is not bit-identical to a fresh "
            f"fit truncated at max_iter=argmax ({argmax})"
        )
    assert restored.n_kurt_done == truncated.n_kurt_done


# ---------------------------------------------------------------------------
# B.3: n_kurt_done / pdtype consistency under pdftype=1
# ---------------------------------------------------------------------------


def test_pdftype1_forced_restore_rolls_back_n_kurt_done_with_pdtype(real_data):
    """``_snapshot_params`` captures ``n_kurt_done`` alongside ``pdtype`` (both
    in ``_PARAM_ARRAYS``/the extra scalar), so a restore rolls the switch
    counter back in lockstep with the density-family codes it gates --
    otherwise a restored model could report a switch count inconsistent with
    which sources actually got switched (silent-failure risk this ports from
    ``AMICATorchNG``).

    The recipe (module docstring) schedules the adaptive switcher's first
    kurtosis pass at iteration 7, strictly after the measured peak at
    iteration 5, so ``keep_best=False`` reaches ``n_kurt_done=5`` (all switch
    passes ran) with a mixed pdtype, while the restored ``keep_best=True``
    model must be pinned at the pre-switch state: ``n_kurt_done=0`` and
    ``pdtype`` unchanged from its all-super-Gaussian (code 1) init.
    """
    x = real_data[:, :4096]
    on = _model(keep_best=True, **_FORCED_RESTORE_PDFTYPE1_KWARGS)
    on.fit(x, max_iter=_FORCED_RESTORE_PDFTYPE1_MAX_ITER, verbose=False)
    off = _model(keep_best=False, **_FORCED_RESTORE_PDFTYPE1_KWARGS)
    off.fit(x, max_iter=_FORCED_RESTORE_PDFTYPE1_MAX_ITER, verbose=False)

    gap = max(on.ll_history) - on.ll_history[-1]
    assert gap > _KEEP_BEST_TOL, (
        "pdftype=1 forcing recipe did not overshoot on this machine/MLX "
        "build; the restore branch was not exercised"
    )
    assert on.ll_history == off.ll_history  # same trajectory either way

    # keep_best=False ran every scheduled switch pass to completion.
    assert off.n_kurt_done == 5
    assert set(np.unique(np.array(off.pdtype)).tolist()) <= {1, 4}
    assert 4 in np.array(off.pdtype)  # at least one source actually flipped

    # keep_best=True is pinned at the pre-switch snapshot.
    assert on.n_kurt_done == 0
    assert np.array_equal(np.array(on.pdtype), np.full_like(np.array(on.pdtype), 1))

    # Cross-check against a truncated fresh fit at the peak, the same
    # bit-identity recipe as the floating-point test above.
    assert on.final_ll_ is not None
    argmax = on.ll_history.index(on.final_ll_)
    truncated = _model(keep_best=False, **_FORCED_RESTORE_PDFTYPE1_KWARGS)
    truncated.fit(x, max_iter=argmax, verbose=False)
    assert truncated.n_kurt_done == on.n_kurt_done == 0
    assert np.array_equal(np.array(truncated.pdtype), np.array(on.pdtype))
    for name in AMICAMLXNG._PARAM_ARRAYS:
        assert np.array_equal(
            np.array(getattr(on, name)), np.array(getattr(truncated, name))
        ), name


# ---------------------------------------------------------------------------
# B.4: share_comps disables the safeguard
# ---------------------------------------------------------------------------


def test_keep_best_inactive_under_share_comps_logs_and_never_restores(
    real_data, caplog
):
    """``share_comps`` disables ``track_best`` (a merge changes the parameter
    count, so an earlier snapshot cannot be reverted to without silently
    undoing it, #60/#269): the safeguard must log once, at WARNING, that it is
    inactive, and ``fit()`` must return the last iterate even on the SAME
    aggressive-Newton recipe that genuinely overshoots without sharing."""
    x = real_data[:, :4096]
    kwargs = dict(_FORCED_RESTORE_KWARGS, share_comps=True)

    with caplog.at_level(logging.WARNING, logger="pamica.mlx_impl.core"):
        on = _model(keep_best=True, **kwargs)
        on.fit(x, max_iter=_FORCED_RESTORE_MAX_ITER, verbose=False)
    message = "\n".join(r.getMessage() for r in caplog.records)
    assert "keep_best is inactive under share_comps" in message

    off = _model(keep_best=False, **kwargs)
    off.fit(x, max_iter=_FORCED_RESTORE_MAX_ITER, verbose=False)

    # The trajectory still overshoots its own peak (sharing does not change
    # that) -- proving the warning/no-restore behavior is not vacuously true
    # on an already-monotone run.
    assert max(on.ll_history) - on.ll_history[-1] > _KEEP_BEST_TOL
    # ... yet no restore fired: final_ll_ is exactly the last trajectory value.
    assert on.final_ll_ == on.ll_history[-1]
    assert on.ll_history == off.ll_history
    for name in AMICAMLXNG._PARAM_ARRAYS:
        a = np.array(getattr(on, name))
        b = np.array(getattr(off, name))
        assert np.array_equal(a, b), f"{name} differs between keep_best on/off"


# ---------------------------------------------------------------------------
# B.5: persistence
# ---------------------------------------------------------------------------


def test_keep_best_round_trips_through_state_dict():
    m = _model(seed=3, block_size=BLOCK, keep_best=False)
    from pamica.torch_impl.utils import load_eeglab_data

    x = load_eeglab_data(str(DATA_FILE), data_dim=NW, field_dim=FIELD).astype(
        np.float64
    )[:, :2048]
    m.fit(x, max_iter=3, verbose=False)
    state = m.state_dict()
    assert state["config"]["keep_best"] is False

    m2 = AMICAMLXNG.from_state_dict(state)
    assert m2.keep_best is False


def test_missing_keep_best_key_loads_with_the_default(real_data):
    """Additive-compat pin: a phase-1-era payload's config dict has no
    ``keep_best`` key at all (it predates this phase). ``from_state_dict``
    calls ``cls(**config)``, so a missing key falls through to the
    constructor default (``True``) with no format_version bump needed --
    exactly torch's #207 precedent."""
    m = _model(seed=3, block_size=BLOCK, keep_best=False)
    m.fit(real_data[:, :2048], max_iter=3, verbose=False)
    state = m.state_dict()
    assert "keep_best" in state["config"]
    del state["config"]["keep_best"]

    m2 = AMICAMLXNG.from_state_dict(state)
    assert m2.keep_best is True  # falls back to the constructor default


# ---------------------------------------------------------------------------
# Degenerate-stop exclusion (PR #310 review): the restore guard's
# ``stop_reason not in _DEGENERATE_STOP_REASONS`` branch
# ---------------------------------------------------------------------------


class _NaNAfterIteration(AMICAMLXNG):
    """Forces a ``nan_ll`` stop after ``nan_after`` REAL iterations have run.

    Sanctioned error-injection subclass (``.rules/testing.md``'s "Sanctioned
    Exception"; same construction as ``test_mlx_restarts.py``'s
    ``_NaNForSeeds``, gated on iteration count instead of seed): every
    iteration up to ``nan_after`` runs the untouched production
    ``_accumulate_blocks``/``_update_parameters`` on real data, so
    ``best_snapshot`` captures a genuine, better earlier peak before the
    injected divergence -- exactly the scenario the guard exists for.
    """

    nan_after: int = 10**9  # effectively never, unless overridden

    def _accumulate_blocks(self, X, stash_llt=False):
        acc = super()._accumulate_blocks(X, stash_llt=stash_llt)
        if self.iteration >= self.nan_after:
            acc["ll"] = acc["ll"] * float("nan")
        return acc


_DEGENERATE_AFTER_KWARGS: dict[str, Any] = dict(
    n_channels=NW, n_models=1, n_mix=NMIX, seed=42, block_size=BLOCK, lrate=0.1
)
_DEGENERATE_NAN_AFTER = 5


def test_keep_best_does_not_rescue_a_diverged_fit_that_peaked_earlier(real_data):
    """The end-of-fit restore guard must actually refuse to rescue a
    diverged fit, not merely never happen to need to: a fit that ran
    ``_DEGENERATE_NAN_AFTER`` real iterations (so a genuine, better
    ``best_snapshot`` exists) and then hits an injected ``nan_ll`` must end
    degenerate with NO restore -- state_dict()'s refusal to persist a
    degenerate model (#50) would otherwise be silently bypassed by a
    restored, "successful-looking" return."""
    x = real_data[:, :4096]
    on = _NaNAfterIteration(keep_best=True, **_DEGENERATE_AFTER_KWARGS)
    on.nan_after = _DEGENERATE_NAN_AFTER
    on.fit(x, max_iter=20, verbose=False)

    assert on.stop_reason in AMICAMLXNG._DEGENERATE_STOP_REASONS
    assert on.stop_reason == "nan_ll"
    assert on.final_ll_ is not None and math.isnan(on.final_ll_)
    # A genuine, finite earlier trajectory existed for the guard to have
    # (wrongly) rescued, if it were broken.
    assert len(on.ll_history) == _DEGENERATE_NAN_AFTER
    assert all(math.isfinite(v) for v in on.ll_history)

    # No restore fired: bit-identical to keep_best=False on the identical
    # forced-degenerate trajectory. If the guard were broken, keep_best=True
    # would roll back to the earlier peak and diverge from this.
    off = _NaNAfterIteration(keep_best=False, **_DEGENERATE_AFTER_KWARGS)
    off.nan_after = _DEGENERATE_NAN_AFTER
    off.fit(x, max_iter=20, verbose=False)
    assert off.stop_reason == "nan_ll"
    for name in AMICAMLXNG._PARAM_ARRAYS:
        a = np.array(getattr(on, name))
        b = np.array(getattr(off, name))
        assert np.array_equal(a, b), (
            f"{name} differs: keep_best rescued a diverged fit despite the "
            f"degenerate stop"
        )

    # Positive proof the returned params are the diverged LAST state, not
    # the (better) snapshot: reconstruct what the snapshot would have held
    # (params before the peak iteration's own M-step, the same truncated-
    # refit trick as the forced-restore tests above) and confirm the
    # degenerate run's params differ from it.
    argmax = int(np.argmax(on.ll_history))
    would_be_snapshot = AMICAMLXNG(**_DEGENERATE_AFTER_KWARGS)
    would_be_snapshot.fit(x, max_iter=argmax, verbose=False)
    differs = any(
        not np.array_equal(
            np.array(getattr(on, name)), np.array(getattr(would_be_snapshot, name))
        )
        for name in AMICAMLXNG._PARAM_ARRAYS
    )
    assert differs, (
        "returned params equal the would-be best snapshot: the restore "
        "fired despite the degenerate stop"
    )


# ---------------------------------------------------------------------------
# keep_best x best-of-N restarts composition (PR #310 review, issue #198)
# ---------------------------------------------------------------------------


def test_keep_best_restart_record_reports_the_restored_iterate(real_data):
    """A restart's recorded ``restart_lls_`` entry must be the RESTORED best
    iterate's LL, not the raw last iterate -- ``_fit_once`` (called once per
    restart) applies its own keep_best restore before returning, and
    ``fit()`` reads ``final_ll_`` off that already-restored state, so this
    should hold by construction; this test pins it against regression.

    The forced-restore recipe (module docstring) runs as restart index 0
    (seed 0); a cheap, unforced seed 1 fills out ``n_restarts=2`` so the
    restart machinery (not just a bare single fit) is actually exercised."""
    x = real_data[:, :4096]

    single = _model(keep_best=True, **_FORCED_RESTORE_KWARGS)
    single.fit(x, max_iter=_FORCED_RESTORE_MAX_ITER, verbose=False)
    assert single.stop_reason == "min_dll"
    assert max(single.ll_history) - single.ll_history[-1] > _KEEP_BEST_TOL

    kwargs: dict[str, Any] = dict(_FORCED_RESTORE_KWARGS)
    del kwargs["seed"]  # superseded by restart_seeds below
    multi = _model(keep_best=True, n_restarts=2, restart_seeds=[0, 1], **kwargs)
    multi.fit(x, max_iter=_FORCED_RESTORE_MAX_ITER, verbose=False)

    assert multi.restart_seeds_ == [0, 1]
    assert multi.restart_stop_reasons_[0] == "min_dll"
    # The restart record for seed 0 reports the RESTORED best-iterate LL,
    # bit-identical to what a standalone seed-0 fit returns as final_ll_ --
    # not ll_history[-1], which the module docstring's recipe establishes is
    # strictly lower on this seed.
    assert multi.restart_lls_[0] == single.final_ll_
    assert multi.restart_lls_[0] != single.ll_history[-1]
