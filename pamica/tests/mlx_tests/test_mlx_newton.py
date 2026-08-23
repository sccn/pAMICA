"""Newton on the MLX backend: the MLX-only mechanics (issue #264).

What lives here is everything that is not a comparison against another backend:
the M-step ordering Newton depends on, the learning-rate schedule arms, the
multi-model shape handling, and the interaction with the ``share_comps``
A-freeze. The float32-vs-float64 numerical question -- whether the curvature and
the 2x2 solve survive single precision -- is the cross-backend file's job
(``pamica/tests/test_mlx_newton_cross_backend.py``), per
``.rules/backend_parity.md``.

Apple-Silicon only; the module self-skips when MLX or an Apple GPU is
unavailable. Real bundled sample EEG throughout, with no mocked curvature: the
non-positive-definite states below are reached by evaluating a genuinely
under-determined 256-sample block of the real recording, exactly as
``torch_tests/test_ng_backend.py`` reaches the same branch. The one ``mock`` use
is a pass-through SPY that observes an argument and calls the real method (the
same construction as ``test_newton_finalize_uses_preupdate_mu`` on the PyTorch
side); it replaces no logic.
"""

from pathlib import Path
from unittest import mock

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from pamica.mlx_impl import AMICAMLXNG  # noqa: E402  (after the MLX importorskip)

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "sample_data"
DATA_FILE = SAMPLE_DIR / "eeglab_data.fdt"
NW = 32
FIELD = 30504
NMIX = 3
SEED = 42
# One block of this many real samples under-determines the 32-channel curvature,
# so the positive-definiteness guard rejects it -- measured min off-diagonal
# ``prod - 1`` = -0.96 over 344 of the 992 pairs. That is how the fallback path
# is reached below without fabricating curvature.
SMALL_BLOCK = 256

pytestmark = [
    pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing"),
    pytest.mark.skipif(
        mx.default_device().type != mx.DeviceType.gpu, reason="no Apple GPU"
    ),
]


def _load_real_data(n_samples: int | None = None) -> np.ndarray:
    from pamica.torch_impl.utils import load_eeglab_data

    data = load_eeglab_data(str(DATA_FILE), data_dim=NW, field_dim=FIELD).astype(
        np.float64
    )
    return data if n_samples is None else data[:, :n_samples]


def _model_at(warmup: int, block_size: int = 1024, n_models: int = 1, **kwargs):
    """A Newton-enabled MLX model driven ``warmup`` production M-steps past init
    on the real sample, plus its sphered data and the model's ``iteration`` left
    at ``warmup``."""
    model = AMICAMLXNG(
        n_channels=NW,
        n_models=n_models,
        n_mix=NMIX,
        seed=SEED,
        block_size=block_size,
        do_newton=True,
        newt_start=0,
        **kwargs,
    )
    x_t = model._preprocess(_load_real_data(4096))
    model._initialize_parameters()
    for it in range(warmup):
        model.iteration = it
        model._update_parameters(model._accumulate_blocks(x_t), x_t.shape[1])
    model.iteration = warmup
    mx.eval(model.A, model.mu, model.alpha, model.beta, model.rho, model.gm, model.c)
    return model, x_t


def _posdef_at(model, acc, h: int = 0) -> bool:
    """The positive-definiteness verdict for model ``h`` on ``acc``, read from
    the production ``_newton_direction`` (so a test that claims to exercise a
    branch can prove it does)."""
    sigma2, lambda_, kappa = model._finalize_newton_stats(acc)
    dA_h = -acc["dWtmp"][h] / acc["dgm"][h] + mx.eye(model.n_channels)
    _, posdef = model._newton_direction(dA_h, sigma2[h], lambda_[h], kappa[h])
    return posdef


def _ratchet_count(rate: float, rate0: float, factor: float) -> int:
    """How many ``*= factor`` ratchets took ``rate0`` to ``rate``."""
    if rate == rate0:
        return 0
    k = round(np.log(rate / rate0) / np.log(factor))
    assert rate == pytest.approx(rate0 * factor**k), (
        f"{rate} is not {rate0} ratcheted by {factor} a whole number of times"
    )
    return k


# --- learning-rate schedule fixtures ----------------------------------------
#
# The two ratchet tests below need a run whose likelihood decreases straddle
# ``newt_start``: some maxdecs cycles completing before the Newton switch-on and
# some after. WHEN a fit decreases is BLAS- and hardware-dependent -- the same
# effect this repo already documents for the `min_dll` stop, which fires at
# iteration 326 on macOS-arm64, 412 on Linux-CUDA and 1076 on a GitHub runner
# (docs/guides/validation.md) -- so a hardcoded `newt_start` that straddles on
# one machine need not straddle on another, and the tests below originally went
# vacuous on the CI Apple-Silicon runner for exactly that reason.
#
# They are therefore data-driven: a probe fit whose ``newt_start`` sits past the
# budget reports where the decreases actually land ON THE EXECUTING MACHINE, and
# the real ``newt_start`` is chosen from that. This works because the
# natural-gradient PREFIX is independent of ``newt_start``: for ``it <
# newt_start`` every branch that reads it (Newton activation, the two ``it >
# newt_start`` ceiling ratchets, the ``it == newt_start`` counter reset) is false
# on both sides, so the two runs are bit-identical there. The likelihood recorded
# AT ``it == newt_start`` is shared too -- ``fit`` computes it from the previous
# iteration's parameters, before the first Newton M-step runs.

_SCHEDULE_DATA = 8192
_SCHEDULE_ITERS = 120
_SCHEDULE_MAXDECS = 2
# Both halves of the config were chosen by sweeping four seeds x two block sizes
# x three sample counts -- a deliberately harsher stand-in for the trajectory
# variation between machines -- and requiring every variant to satisfy every
# guard below. lrate=0.6 with maxdecs=2 overshoots often enough to complete
# decrease cycles in the natural-gradient phase while still reaching the budget
# (lrate=1.0 was tried first and rejected: the probe itself diverged to a
# nan_params stop on half the variants). newtrate=2.0 then guarantees the same
# in the NEWTON phase: at newtrate=1.0 the post-switch-on trajectory was
# monotone on the larger sample counts, leaving the ratchet cadence with nothing
# to fire on and both tests below vacuous.
_SCHEDULE_LRATE = 0.6
_SCHEDULE_NEWTRATE = 2.0


def _schedule_model(newt_start: int, do_newton: bool = True) -> AMICAMLXNG:
    """A config that overshoots often enough on the real sample to exercise the
    maxdecs cadence, at a caller-chosen Newton switch-on."""
    return AMICAMLXNG(
        n_channels=NW, n_mix=NMIX, seed=3, block_size=1024,
        lrate=_SCHEDULE_LRATE, lratefact=0.5, maxdecs=_SCHEDULE_MAXDECS,
        do_newton=do_newton, newt_start=newt_start, newtrate=_SCHEDULE_NEWTRATE,
    )  # fmt: skip


def _fit_schedule(newt_start: int, do_newton: bool = True, max_iter: int | None = None):
    m = _schedule_model(newt_start, do_newton=do_newton)
    m.fit(
        _load_real_data(_SCHEDULE_DATA),
        max_iter=_SCHEDULE_ITERS if max_iter is None else max_iter,
        verbose=False,
    )
    # Both floor branches skip the numdecs increment and stop the fit, so the
    # replays below would not describe a run that took one; a degenerate stop
    # would truncate the trajectory the straddle is read from.
    assert m.stop_reason not in ("lrate_floor", "grad_norm_floor"), (
        f"the fit stopped on {m.stop_reason}, a branch that bypasses the "
        "decrease counter; the replay cannot describe it"
    )
    assert m.stop_reason not in AMICAMLXNG._DEGENERATE_STOP_REASONS, (
        f"the schedule fixture diverged ({m.stop_reason}); it is meant to "
        "overshoot, not to blow up"
    )
    return m


def _ratchet_iterations(ll, maxdecs: int, newt_start: int | None = None) -> list[int]:
    """Iterations at which the decrease counter completes a ``maxdecs`` cycle and
    the ceilings ratchet. ``newt_start`` applies Fortran's counter reset on the
    switch-on iteration; ``None`` replays without it (the counterfactual)."""
    numdecs, hits = 0, []
    for i in range(1, len(ll)):
        if ll[i] < ll[i - 1]:
            numdecs += 1
            if numdecs >= maxdecs:
                hits.append(i)
                numdecs = 0
        if newt_start is not None and i == newt_start:
            numdecs = 0
    return hits


@pytest.fixture(scope="module")
def natural_gradient_prefix() -> list[float]:
    """The likelihood trajectory with Newton never switching on -- the prefix
    every ``newt_start`` shares (see the note above)."""
    return _fit_schedule(_SCHEDULE_ITERS + 1).ll_history


# --- (e) stability ----------------------------------------------------------


def test_newton_fit_is_stable_on_full_data():
    """A full-recording float32 Newton fit runs to its budget without diverging
    and improves the likelihood.

    float32 Newton is the whole risk this phase carried: the curvature is a sum
    of squares over 30504 samples and the guard compares a product of those sums
    against exactly 1. This is the MLX counterpart of
    ``torch_tests/test_ng_float32_stability.py``, whose failure mode was a
    ``nan_ll`` stop rather than a bad number.
    """
    m = AMICAMLXNG(n_channels=NW, n_mix=NMIX, seed=SEED, do_newton=True)
    m.fit(_load_real_data(), max_iter=100, verbose=False)

    hist = np.asarray(m.ll_history, dtype=float)
    assert np.all(np.isfinite(hist))
    assert m.final_ll_ is not None and np.isfinite(m.final_ll_)
    assert np.all(np.isfinite(np.array(m.A)))
    assert m.stop_reason not in AMICAMLXNG._DEGENERATE_STOP_REASONS
    assert hist[-1] > hist[0]  # ascent
    # Newton was reached (the schedule ran past newt_start) and stayed usable.
    assert len(hist) > m.newt_start
    assert m.n_newton_fallbacks == 0


# --- (f) schedule arms ------------------------------------------------------


def test_lrate_ramps_toward_newtrate_while_posdef():
    """With Newton active and positive definite the ramp climbs to ``newtrate``,
    above the natural-gradient ceiling ``lrate_cap`` (Fortran
    amica15.f90:1803-1816)."""
    model, x_t = _model_at(warmup=5, lrate=0.1, newtrate=0.5)
    assert _posdef_at(model, model._accumulate_blocks(x_t)), "state is not posdef"

    for it in range(5, 40):
        model.iteration = it
        model._update_parameters(model._accumulate_blocks(x_t), x_t.shape[1])
    mx.eval(model.A)

    assert model.n_newton_fallbacks == 0
    assert model.lrate > model.lrate_cap, "ramp never passed the natural-gradient cap"
    assert model.lrate == pytest.approx(model.newtrate)


def test_fallback_ramps_toward_lrate_cap_and_counts():
    """A rejected Newton direction ramps toward ``lrate_cap`` (not ``newtrate``)
    and increments ``n_newton_fallbacks`` once."""
    model, x_t = _model_at(warmup=0, block_size=SMALL_BLOCK, lrate=0.1, newtrate=0.5)
    model.iteration = 5  # >= newt_start, so Newton is active
    acc = model._get_block_updates(x_t[:, :SMALL_BLOCK])
    assert not _posdef_at(model, acc), "the block is posdef; the test is vacuous"

    for _ in range(30):
        model._update_parameters(acc, SMALL_BLOCK)
    mx.eval(model.A)

    assert model.n_newton_fallbacks == 30, "one count per rejected iteration"
    assert model.lrate == pytest.approx(model.lrate_cap)
    assert model.lrate < model.newtrate


def test_newtrate_ratchet_is_suppressed_before_newt_start(natural_gradient_prefix):
    """A ``maxdecs`` cycle completing BEFORE the Newton switch-on ratchets
    ``lrate_cap`` but must leave ``newtrate`` alone (Fortran
    amica15.f90:1056-1077 gates it on ``it > newt_start``).

    ``newt_start`` is read off the probe trajectory -- one past the first ratchet
    of the natural-gradient phase -- so a suppressed ratchet exists BY
    CONSTRUCTION on whatever machine is running, rather than by a constant that
    happened to straddle on the author's. The expected counts are then replayed
    from the run's own ``ll_history``, so what is pinned is the semantics, not a
    trajectory. The complementary direction -- that a ratchet past the gate DOES
    move ``newtrate`` -- is the next test; splitting them means neither depends
    on one run producing cycles on both sides.
    """
    prefix_hits = _ratchet_iterations(natural_gradient_prefix, _SCHEDULE_MAXDECS)
    assert prefix_hits, (
        "the sample recording produced no maxdecs ratchet at all in "
        f"{len(natural_gradient_prefix)} natural-gradient iterations, so no "
        "newt_start can put one before the gate: the DATA, not the config, is "
        "the problem here"
    )
    newt_start = prefix_hits[0] + 1
    assert newt_start < _SCHEDULE_ITERS - 20, (
        f"the first ratchet lands at iteration {prefix_hits[0]} of "
        f"{_SCHEDULE_ITERS}, leaving no Newton-phase headroom on this data"
    )

    m = _fit_schedule(newt_start)
    hits = _ratchet_iterations(m.ll_history, m.maxdecs, newt_start=newt_start)
    cap_ratchets = len(hits)
    newt_ratchets = sum(1 for i in hits if i > newt_start)

    # Guaranteed by the choice of newt_start: the prefix ratchet at
    # prefix_hits[0] < newt_start is shared with the probe bit-for-bit, so it
    # ratchets lrate_cap while the gate holds newtrate.
    assert newt_ratchets < cap_ratchets, (
        f"expected a ratchet before newt_start={newt_start} from the shared "
        f"prefix; got {hits}"
    )
    assert _ratchet_count(m.lrate_cap, m.lrate0, m.lratefact) == cap_ratchets
    assert _ratchet_count(m.newtrate, m.newtrate0, m.lratefact) == newt_ratchets


def test_newtrate_ratchets_at_maxdecs_once_newton_runs():
    """With Newton active from the first iteration every ``maxdecs`` cycle is
    past the gate, so ``newtrate`` ratchets in lockstep with ``lrate_cap``.

    This is the "admits" half of the gate, and it needs only that the run
    ratchets at all -- a far weaker requirement than a single run straddling the
    switch-on, which is what made the previous single-test formulation
    machine-dependent.
    """
    m = _fit_schedule(newt_start=0)
    hits = _ratchet_iterations(m.ll_history, m.maxdecs, newt_start=0)
    assert hits, (
        f"no maxdecs cycle completed in {len(m.ll_history)} iterations with "
        "Newton running throughout; the DATA did not decrease often enough to "
        "exercise the ratchet"
    )
    assert all(i > m.newt_start for i in hits)  # newt_start=0; i starts at 1
    assert _ratchet_count(m.lrate_cap, m.lrate0, m.lratefact) == len(hits)
    assert _ratchet_count(m.newtrate, m.newtrate0, m.lratefact) == len(hits)
    assert m.newtrate < m.newtrate0  # non-vacuous: the ceiling actually moved


def test_newtrate_never_ratchets_without_newton(natural_gradient_prefix):
    """``newtrate`` is inert for a natural-gradient fit, however many likelihood
    decreases it takes: the ratchet is gated on ``do_newton`` as well as on
    ``newt_start`` (only ``lrate_cap``, which is ungated, moves)."""
    prefix_hits = _ratchet_iterations(natural_gradient_prefix, _SCHEDULE_MAXDECS)
    assert prefix_hits, "the data produced no ratchet; nothing to gate"
    m = _fit_schedule(prefix_hits[0] + 1, do_newton=False)

    hits = _ratchet_iterations(m.ll_history, m.maxdecs, newt_start=None)
    assert any(i > m.newt_start for i in hits), (
        "no ratchet landed past newt_start, so the do_newton gate is untested"
    )
    assert _ratchet_count(m.lrate_cap, m.lrate0, m.lratefact) == len(hits)
    assert m.newtrate == m.newtrate0


def test_numdecs_resets_when_newton_switches_on(natural_gradient_prefix):
    """The decrease counter is cleared on the iteration Newton switches on
    (Fortran amica15.f90:1099-1102), so a partially filled count from the
    natural-gradient phase cannot ratchet the ceilings under the new schedule.

    Made observable in three data-driven steps, none of them a hardcoded
    trajectory. First ``newt_start`` is placed at an iteration where the probe
    shows the counter PARTIALLY filled -- that state is in the shared prefix, so
    it holds on any machine. Then a full-budget fit at that ``newt_start`` is
    scanned for the smallest budget at which the reset and no-reset replays
    disagree; one exists as soon as the Newton phase decreases at all, because
    the no-reset counter is strictly ahead and therefore completes its cycle
    strictly earlier. Finally the fit is repeated at exactly that budget, where
    the two hypotheses predict different ``lrate_cap`` values, and the observed
    one has to match the reset prediction. Truncating is sound because the loop
    is causal: iteration k depends only on the state after k-1, so a shorter
    budget reproduces the same prefix.
    """
    partial = None
    numdecs = 0
    for i in range(1, len(natural_gradient_prefix)):
        if natural_gradient_prefix[i] < natural_gradient_prefix[i - 1]:
            numdecs += 1
            if numdecs >= _SCHEDULE_MAXDECS:
                numdecs = 0
        if 0 < numdecs < _SCHEDULE_MAXDECS:
            partial = i
            break
    assert partial is not None, (
        "the sample recording never left the decrease counter partially filled "
        "in the natural-gradient phase, so the reset has nothing to clear: the "
        "DATA, not the config, is the problem here"
    )

    full = _fit_schedule(partial)
    ll = full.ll_history
    budget = next(
        (
            t
            for t in range(2, len(ll) + 1)
            if len(_ratchet_iterations(ll[:t], full.maxdecs, newt_start=partial))
            != len(_ratchet_iterations(ll[:t], full.maxdecs))
        ),
        None,
    )
    assert budget is not None, (
        f"over {len(ll)} iterations the two counter hypotheses never predicted "
        f"different ratchet counts at newt_start={partial}; the DATA did not "
        "decrease often enough in the Newton phase to expose the reset"
    )

    m = _fit_schedule(partial, max_iter=budget)
    assert len(m.ll_history) == budget, "the truncated fit stopped early"
    with_reset = _ratchet_iterations(m.ll_history, m.maxdecs, newt_start=partial)
    without_reset = _ratchet_iterations(m.ll_history, m.maxdecs)
    assert len(with_reset) != len(without_reset), (
        f"the reset is unobservable at budget {budget}: {with_reset} vs {without_reset}"
    )
    assert _ratchet_count(m.lrate_cap, m.lrate0, m.lratefact) == len(with_reset)


def test_newton_schedule_state_resets_at_fit_start():
    """A refit starts from the pristine ``newtrate`` ceiling with a zeroed
    fallback count, so a previously annealed run cannot leak into the next one
    (parity with the ``rholrate``/``lrate_cap`` resets, issue #195)."""
    m = AMICAMLXNG(n_channels=NW, n_mix=NMIX, seed=SEED, do_newton=True)
    m.newtrate = m.newtrate0 * 0.25
    m.n_newton_fallbacks = 7
    m._initialize_parameters()  # the call fit() makes
    assert m.newtrate == m.newtrate0
    assert m.n_newton_fallbacks == 0


# --- (g) the M-step ordering Newton depends on ------------------------------


def test_newton_finalize_uses_preupdate_mu():
    """``_update_parameters`` must finalize the Newton curvature (lambda folds in
    ``mu^2``) with the PRE-update ``mu``, before the exact-EM mu update moves it.

    Fortran folds that term in during E-step accumulation (amica15.f90:1666-1680),
    so finalizing after the M-step has moved mu yields a subtly wrong Hessian
    with no error and no NaN -- the failure the PyTorch backend shipped and fixed
    in issue #24, pinned there by the test of the same name. The spy observes the
    argument and calls the real method; no logic is replaced.
    """
    model, x_t = _model_at(warmup=0, block_size=SMALL_BLOCK)
    model.iteration = 5  # >= newt_start, so the finalization runs
    acc = model._get_block_updates(x_t[:, :SMALL_BLOCK])

    assert model.mu is not None
    # MLX arrays are immutable and self.mu is only ever rebound, so holding the
    # reference IS the snapshot (no clone needed, unlike torch).
    mu_pre = model.mu
    captured = {}
    original = model._finalize_newton_stats

    def spy(a):
        captured["mu"] = model.mu  # mu as seen by the finalization
        return original(a)

    with mock.patch.object(model, "_finalize_newton_stats", side_effect=spy):
        model._update_parameters(acc, SMALL_BLOCK)

    assert "mu" in captured, "Newton finalization was not invoked"
    assert np.array_equal(np.array(captured["mu"]), np.array(mu_pre)), (
        "Newton finalization saw the post-update mu (the issue #24 lambda bug)"
    )
    # Guard the test itself: mu genuinely moved, so pre != post is a real check.
    assert not np.array_equal(np.array(model.mu), np.array(mu_pre))


# --- (h) multi-model --------------------------------------------------------


@pytest.mark.parametrize("n_models", [1, 2])
def test_multimodel_newton_mstep_is_finite(n_models):
    """Newton through the real M-step for one and two models.

    The curvature accumulators are the only ``(n_models, ...)`` arrays whose
    reduction divides by the model mass, and getting that broadcast axis wrong
    is a crash for ``n_models > 1`` and a silent no-op for ``n_models == 1`` --
    which is exactly how it survived in the NumPy backend until issue #267. Both
    are driven here through ``_update_parameters``, not through the reduction
    alone.
    """
    model, x_t = _model_at(warmup=3, n_models=n_models)
    acc = model._accumulate_blocks(x_t)

    sigma2, lambda_, kappa = model._finalize_newton_stats(acc)
    for name, arr in (("sigma2", sigma2), ("lambda", lambda_), ("kappa", kappa)):
        assert arr.shape == (n_models, NW), f"{name} shape {arr.shape}"
        host = np.array(arr, dtype=np.float64)
        assert np.all(np.isfinite(host)), f"{name} is not finite"
        assert np.all(host > 0), f"{name} is not strictly positive"

    model._update_parameters(acc, x_t.shape[1])
    mx.eval(model.A, model.mu, model.alpha, model.beta, model.rho, model.gm, model.c)
    for name in ("A", "mu", "alpha", "beta", "rho", "gm", "c"):
        assert np.all(np.isfinite(np.array(getattr(model, name)))), name


def test_multimodel_newton_fit_completes():
    """And end to end: a 2-model Newton fit runs to its budget with a finite,
    improving likelihood."""
    m = AMICAMLXNG(
        n_channels=NW, n_models=2, n_mix=NMIX, seed=SEED, do_newton=True, newt_start=5
    )
    m.fit(_load_real_data(8192), max_iter=40, verbose=False)

    hist = np.asarray(m.ll_history, dtype=float)
    assert np.all(np.isfinite(hist))
    assert m.stop_reason not in AMICAMLXNG._DEGENERATE_STOP_REASONS
    assert hist[-1] > hist[0]
    assert len(hist) > m.newt_start, "Newton never switched on"


# --- (i) sharing interplay --------------------------------------------------


def test_frozen_iterations_do_not_count_newton_fallbacks():
    """A Newton direction discarded because ``share_comps`` is holding ``A``
    must not inflate ``n_newton_fallbacks``.

    Fortran nests the fallback report inside the same share-freeze-guarded
    ``update_A`` block as the step and the ramp (amica15.f90:1803-1816), so a
    frozen iteration reports nothing. The two halves below run the same
    non-positive-definite state through ``_update_parameters`` twice, differing
    only in whether the freeze window is open.
    """
    model, x_t = _model_at(
        warmup=0,
        block_size=SMALL_BLOCK,
        n_models=2,
        share_comps=True,
        share_start=1,
        share_iter=20,
    )
    acc = model._get_block_updates(x_t[:, :SMALL_BLOCK])
    assert not _posdef_at(model, acc), "the block is posdef; the test is vacuous"

    model.iteration = 0  # inside the freeze window (itf=1 == share_start)
    assert model._a_frozen()
    model._update_parameters(acc, SMALL_BLOCK)
    mx.eval(model.A)
    assert model.n_newton_fallbacks == 0, "a frozen iteration inflated the counter"

    model.iteration = 6  # past the 6-iteration settle window
    assert not model._a_frozen()
    model._update_parameters(acc, SMALL_BLOCK)
    mx.eval(model.A)
    assert model.n_newton_fallbacks == 1, "an unfrozen rejection was not counted"


def test_sharing_and_newton_fit_completes():
    """``share_comps`` and Newton together: merges land inside the Newton phase
    and the fit stays finite.

    A merged-away column receives no sufficient statistic, but the Newton
    curvature is indexed by (model, SOURCE) rather than by mixing column, so it
    never sees the 0/0 that the mixture updates mask -- both models' sources keep
    full responsibility mass whichever column they point at. This drives that
    claim through a real fit at the shipped ``comp_thresh`` default, where a
    couple of genuinely near-collinear pairs merge.

    (At a far looser cutoff, where nearly every column merges and the two models
    collapse onto one another, Newton can drive a component to zero curvature and
    the fit aborts on the ``nan_params`` guard. That is not MLX-specific --
    ``AMICATorchNG`` in float32 aborts on the same configuration at other seeds
    -- and it fails loudly rather than returning a wrong answer, so it is left as
    a documented property of that regime rather than pinned here; see
    ``.context/issue-264/newton_findings.md``.)
    """
    m = AMICAMLXNG(
        n_channels=NW, n_models=2, n_mix=NMIX, seed=SEED, block_size=1024,
        do_newton=True, newt_start=5,
        share_comps=True, share_start=10, share_iter=8,
    )  # fmt: skip
    m.fit(_load_real_data(8192), max_iter=40, verbose=False)

    hist = np.asarray(m.ll_history, dtype=float)
    assert np.all(np.isfinite(hist))
    assert m.stop_reason not in AMICAMLXNG._DEGENERATE_STOP_REASONS
    assert np.all(np.isfinite(np.array(m.A)))
    assert len(hist) > m.share_start, "sharing never got a chance to fire"
    assert m.shared_components(), "no merge fired; the interplay is untested"
