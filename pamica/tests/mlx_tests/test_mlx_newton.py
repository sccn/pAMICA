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


def test_newtrate_ratchets_only_at_maxdecs_after_newt_start():
    """``newtrate`` is a ceiling that ratchets on the ``maxdecs`` cadence and
    only once ``it > newt_start`` (Fortran amica15.f90:1056-1077), so it is a
    strict subset of the ``lrate_cap`` ratchets on the same run.

    The expected count is replayed from the run's own ``ll_history`` rather than
    hardcoded, so the pin is on the SEMANTICS, not on one trajectory.
    """
    m = AMICAMLXNG(
        n_channels=NW, n_mix=NMIX, seed=3, block_size=1024,
        lrate=1.0, lratefact=0.5, maxdecs=3,
        do_newton=True, newt_start=18, newtrate=1.0,
    )  # fmt: skip
    m.fit(_load_real_data(8192), max_iter=60, verbose=False)

    # Replay the decrease counter over the observed trajectory.
    ll = m.ll_history
    numdecs, cap_ratchets, newt_ratchets = 0, 0, 0
    for i in range(1, len(ll)):
        if ll[i] < ll[i - 1]:
            numdecs += 1
            if numdecs >= m.maxdecs:
                cap_ratchets += 1
                if i > m.newt_start:
                    newt_ratchets += 1
                numdecs = 0
        if i == m.newt_start:
            numdecs = 0

    assert m.lrate > m.minlrate, "run hit the lrate floor; use a gentler config"
    assert 0 < newt_ratchets < cap_ratchets, (
        "config no longer straddles newt_start; it cannot discriminate the gate"
    )
    assert _ratchet_count(m.lrate_cap, m.lrate0, m.lratefact) == cap_ratchets
    assert _ratchet_count(m.newtrate, m.newtrate0, m.lratefact) == newt_ratchets


def test_newtrate_never_ratchets_without_newton():
    """``newtrate`` is inert for a natural-gradient fit, however many likelihood
    decreases it takes: the ratchet is gated on ``do_newton`` as well as on
    ``newt_start`` (only ``lrate_cap``, which is ungated, moves)."""
    m = AMICAMLXNG(
        n_channels=NW, n_mix=NMIX, seed=3, block_size=1024,
        lrate=1.0, lratefact=0.5, maxdecs=3,
        do_newton=False, newt_start=18, newtrate=1.0,
    )  # fmt: skip
    m.fit(_load_real_data(8192), max_iter=60, verbose=False)

    ll = m.ll_history
    n_dec = sum(1 for i in range(1, len(ll)) if ll[i] < ll[i - 1])
    assert n_dec >= m.maxdecs, "config did not exercise the decrease path"
    assert m.lrate_cap < m.lrate0, "lrate_cap should still ratchet"
    assert m.newtrate == m.newtrate0


def test_numdecs_resets_when_newton_switches_on():
    """The decrease counter is cleared on the iteration Newton switches on
    (Fortran amica15.f90:1099-1102), so a partially filled count from the
    natural-gradient phase cannot ratchet the ceilings under the new schedule.

    This run decreases on exactly iterations 2 and 3 with ``maxdecs=2`` and
    ``newt_start=2``: without the reset the second decrease would complete the
    count and ratchet ``lrate_cap`` to 0.4; with it the counter restarts at
    iteration 2 and no ratchet fires. The counterfactual is computed from the
    run's own history, so this pins the semantics rather than the constant.
    """
    m = AMICAMLXNG(
        n_channels=NW, n_mix=NMIX, seed=3, block_size=1024,
        lrate=0.8, lratefact=0.5, maxdecs=2,
        do_newton=True, newt_start=2, newtrate=1.0,
    )  # fmt: skip
    m.fit(_load_real_data(8192), max_iter=8, verbose=False)

    ll = m.ll_history
    decreases = [i for i in range(1, len(ll)) if ll[i] < ll[i - 1]]
    assert m.stop_reason == "max_iter"

    def replay(reset: bool) -> int:
        numdecs, ratchets = 0, 0
        for i in range(1, len(ll)):
            if ll[i] < ll[i - 1]:
                numdecs += 1
                if numdecs >= m.maxdecs:
                    ratchets += 1
                    numdecs = 0
            if reset and i == m.newt_start:
                numdecs = 0
        return ratchets

    assert replay(reset=True) != replay(reset=False), (
        f"decreases at {decreases} do not straddle newt_start={m.newt_start}; "
        "the reset is unobservable on this run"
    )
    assert _ratchet_count(m.lrate_cap, m.lrate0, m.lratefact) == replay(reset=True)


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
