"""Iteration-schedule gates count from 1, as the reference does (issue #335).

The reference's main loop starts at ``iter = 1`` (amica15.f90:949) and states
every schedule against that counter, while pamica's fit loops count from 0. Until
issue #335 the Newton switch (``iter .ge. newt_start``), the counter reset on the
switch-on iteration, the ``iter > newt_start`` ceiling ratchets, the rejection
schedule and NumPy's restart-on-NaN window all compared the 0-based index with
the 1-based setting, so each fired one iteration late in every backend. Every
gate now goes through :mod:`pamica.schedule`.

Each gate is observed here through a backend's real fit on the bundled sample
EEG, with no mocks (``.rules/testing.md``): the likelihood trajectory, the
learning-rate ceilings a ratchet moves, and the rejection count. Cross-backend by
design (``.rules/backend_parity.md``): PyTorch and NumPy always run; MLX skips
per test (``pytest.importorskip`` plus an Apple-GPU guard), never the module.
The expected iterations are written out in the reference's own 1-based
arithmetic (``i + 1`` for 0-based index ``i``), not read back from
:mod:`pamica.schedule`, so a regression in that module cannot hide here. The
native-binary oracle for the same gates is ``test_schedule_native_oracle.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from pamica import schedule
from pamica.numpy_impl.core import AMICA as AMICA_NumPy
from pamica.torch_impl.core import AMICATorchNG
from pamica.torch_impl.utils import load_eeglab_data

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_data"
DATA_FILE = SAMPLE_DIR / "eeglab_data.fdt"
NW = 32
FIELD = 30504
NMIX = 3
SEED = 3
BLOCK = 1024
N_FRAMES = 8192
BACKENDS = ["torch", "numpy", "mlx"]

pytestmark = pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing")

# An aggressive natural-gradient step that overshoots early on this recording, so
# the likelihood-decrease bookkeeping has something to count within a short
# budget. lrate=0.6/maxdecs=2/seed=3/8192 frames is test_mlx_newton.py's
# schedule fixture: it was chosen there by sweeping seeds, block sizes and
# sample counts, and here it completes the first maxdecs cycle at iteration 9
# on all three backends (iteration 11 before issue #339 moved the decrease
# response ahead of the update). Every test below still reads the iterations it
# needs off the executing machine's own trajectory rather than trusting that
# number.
_OVERSHOOT: dict[str, Any] = dict(lrate=0.6, lratefact=0.5, maxdecs=2)
# The Newton-phase ceiling test_mlx_newton.py pairs with it: at newtrate=1.0
# the post-switch-on trajectory can be monotone, leaving the reset unobservable.
_OVERSHOOT_NEWTRATE = 2.0
_PROBE_ITERS = 40
_NEWTON_ITERS = 120
_ZERO_ONE_ITERS = 40


@pytest.fixture(scope="module")
def X() -> np.ndarray:
    return load_eeglab_data(str(DATA_FILE), data_dim=NW, field_dim=FIELD).astype(
        np.float64
    )[:, :N_FRAMES]


def _mlx_class():
    """``AMICAMLXNG``, or skip this test (never the whole module)."""
    mlx_core = pytest.importorskip(
        "pamica.mlx_impl.core", reason="MLX not installed (Apple Silicon only)"
    )
    mx = mlx_core.mx
    if mx.default_device().type != mx.DeviceType.gpu:
        pytest.skip("no Apple GPU")
    return mlx_core.AMICAMLXNG


@dataclass
class _Run:
    """What a fit leaves behind that the schedule decides, in one vocabulary."""

    ll: list[float]
    lrate: float
    lrate_ceiling: float  # lrate_cap (torch/MLX), lrate0 (NumPy)
    lrate_ceiling0: float
    rholrate: float  # the rho-rate ceiling: rholrate_cap on every backend
    rholrate0: float
    newtrate: float
    newtrate0: float
    numrej: int
    n_good: int
    n_newton_fallbacks: int | None  # NumPy does not count them


def _fit(
    backend: str,
    x: np.ndarray,
    max_iter: int,
    tmp_path: Path,
    seed: int = SEED,
    **cfg: Any,
) -> _Run:
    """One real fit on ``backend``; ``cfg`` uses the torch/MLX parameter names."""
    if backend == "numpy":
        cfg = dict(cfg)
        if "maxdecs" in cfg:
            cfg["max_decs"] = cfg.pop("maxdecs")
        m = AMICA_NumPy(
            num_models=1,
            num_mix=NMIX,
            seed=seed,
            block_size=BLOCK,
            max_iter=max_iter,
            use_tqdm=False,
            do_opt_block=False,
            writestep=10**7,
            outdir=str(tmp_path / f"np_{max_iter}_{len(list(tmp_path.iterdir()))}"),
            **cfg,
        )
        m.fit(x)
        assert m.num_good_samples is not None  # set by every fit
        return _Run(
            ll=[float(v) for v in m.ll],
            lrate=m.lrate,
            lrate_ceiling=m.lrate0,
            lrate_ceiling0=m._pristine_state["lrate0"],
            rholrate=m.rholrate_cap,
            rholrate0=m.rholrate0,
            newtrate=m.newtrate,
            newtrate0=m._pristine_state["newtrate"],
            numrej=m.numrej,
            n_good=int(m.num_good_samples),
            n_newton_fallbacks=None,
        )
    if backend == "torch":
        m = AMICATorchNG(
            n_channels=NW,
            n_mix=NMIX,
            seed=seed,
            block_size=BLOCK,
            device="cpu",
            dtype=torch.float64,
            keep_best=False,
            **cfg,
        )
    else:
        m = _mlx_class()(
            n_channels=NW,
            n_mix=NMIX,
            seed=seed,
            block_size=BLOCK,
            keep_best=False,
            **cfg,
        )
    m.fit(x, max_iter=max_iter, verbose=False)
    n_good = x.shape[1] if m.good_idx is None else int(np.asarray(m.good_idx).size)
    return _Run(
        ll=[float(v) for v in m.ll_history],
        lrate=m.lrate,
        lrate_ceiling=m.lrate_cap,
        lrate_ceiling0=m.lrate0,
        rholrate=m.rholrate_cap,
        rholrate0=m.rholrate0,
        newtrate=m.newtrate,
        newtrate0=m.newtrate0,
        numrej=m.numrej,
        n_good=n_good,
        n_newton_fallbacks=m.n_newton_fallbacks,
    )


def _ratchet_indices(
    ll: list[float], maxdecs: int, newt_start: int | None
) -> list[int]:
    """0-based ``ll`` indices at which the decrease counter completes a
    ``maxdecs`` cycle (amica15.f90:1056-1077), clearing it on the switch-on
    iteration ``iter == newt_start`` of the reference's 1-based counter
    (amica15.f90:1099) when ``newt_start`` is given."""
    numdecs, hits = 0, []
    for i in range(1, len(ll)):
        if ll[i] < ll[i - 1]:
            numdecs += 1
            if numdecs >= maxdecs:
                hits.append(i)
                numdecs = 0
        if newt_start is not None and i + 1 == newt_start:
            numdecs = 0
    return hits


def _ratchets(rate: float, rate0: float, factor: float) -> int:
    """How many ``*= factor`` ratchets took ``rate0`` to ``rate``."""
    k = round(math.log(rate / rate0) / math.log(factor)) if rate != rate0 else 0
    assert rate == pytest.approx(rate0 * factor**k), (
        f"{rate} is not {rate0} ratcheted by {factor} a whole number of times"
    )
    return k


_PROBES: dict[str, _Run] = {}


def _probe(backend: str, x: np.ndarray, tmp_path: Path) -> _Run:
    """The overshooting natural-gradient trajectory, with Newton enabled but
    never switched on. Every ``newt_start`` shares it up to the switch-on
    iteration, so the iterations a test needs are read off it. Cached per
    backend: it is deterministic for a given machine."""
    if backend not in _PROBES:
        _PROBES[backend] = _fit(
            backend,
            x,
            _PROBE_ITERS,
            tmp_path,
            do_newton=True,
            newt_start=10**6,
            newtrate=_OVERSHOOT_NEWTRATE,
            **_OVERSHOOT,
        )
    return _PROBES[backend]


# --- the arithmetic itself ---------------------------------------------------


def test_every_gate_fires_on_the_reference_iteration():
    """Each predicate, fed 0-based loop indices, first fires on the index whose
    1-based reference iteration the reference names.

    Pure arithmetic on the predicates' own arguments -- no data stands in for
    anything -- so it states the mapping in one place; the real-fit tests below
    are what show the backends actually use it.
    """
    idx = range(12)

    def first(pred) -> int:
        return next(i for i in idx if pred(i))

    def fires(pred) -> list[int]:
        return [i for i in idx if pred(i)]

    # iter .ge. newt_start (amica15.f90:1804): the 5th iteration is index 4.
    assert first(lambda i: schedule.newton_active(True, i, 5)) == 4
    assert not any(schedule.newton_active(False, i, 5) for i in idx)
    # newt_start 0 and 1 both mean "from the first iteration".
    assert first(lambda i: schedule.newton_active(True, i, 1)) == 0
    assert first(lambda i: schedule.newton_active(True, i, 0)) == 0
    # iter == newt_start (amica15.f90:1099): exactly once, on index 4; at
    # newt_start=1 on the first iteration, and never at newt_start=0 (the
    # reference's 1-based counter never equals 0).
    assert fires(lambda i: schedule.newton_switches_on(True, i, 5)) == [4]
    assert fires(lambda i: schedule.newton_switches_on(True, i, 1)) == [0]
    assert fires(lambda i: schedule.newton_switches_on(True, i, 0)) == []
    assert fires(lambda i: schedule.newton_switches_on(False, i, 1)) == []
    # iter > newt_start (amica15.f90:1067): closed on iteration newt_start
    # itself, open from the next one.
    assert not schedule.past_newton_start(4, 5)  # iteration 5
    assert schedule.past_newton_start(5, 5)  # iteration 6
    assert first(lambda i: schedule.past_newton_start(i, 5)) == 5
    assert not schedule.past_newton_start(0, 1)  # iteration 1
    assert schedule.past_newton_start(1, 1)  # iteration 2
    assert first(lambda i: schedule.past_newton_start(i, 0)) == 0
    # Rejection (amica15.f90:1136) with rejstart=4, rejint=3: the 4th
    # iteration (index 3), then every 3rd while numrej < maxrej (held at 0
    # here, so the modulo arm keeps firing).
    assert [i for i in idx if schedule.rejection_due(True, i, 4, 3, 0, 2)] == [3, 6, 9]
    assert not any(schedule.rejection_due(True, i, 4, 3, 0, 0) for i in idx)
    assert not any(schedule.rejection_due(False, i, 4, 3, 0, 2) for i in idx)

    # rejstart=1, rejint=1: every iteration from the first until maxrej passes
    # are spent, with numrej advancing as the fit loops advance it.
    def passes(rejstart: int, rejint: int, maxrej: int) -> list[int]:
        numrej, out = 0, []
        for i in idx:
            if schedule.rejection_due(True, i, rejstart, rejint, numrej, maxrej):
                out.append(i)
                numrej += 1
        return out

    assert passes(1, 1, 3) == [0, 1, 2]
    assert passes(1, 1, 1) == [0]
    # The ``iter == rejstart`` arm is unconditional: it fires even once the
    # modulo arm's budget is spent (here by an earlier modulo pass).
    assert passes(4, 1, 1) == [0, 3]
    # mod(iter, writestep) == 0 (amica15.f90:1124): the 5th and 10th, not the 1st.
    assert [i for i in idx if schedule.every(i, 5)] == [4, 9]
    assert fires(lambda i: schedule.every(i, 1)) == list(idx)
    # Share merges from share_start=3 every 4 (amica15.f90:1856): 3, 7, 11.
    assert [i for i in idx if schedule.periodic_due(i, 3, 4)] == [2, 6, 10]
    assert fires(lambda i: schedule.periodic_due(i, 1, 1)) == list(idx)
    # The A-freeze (amica15.f90:1803): from share_start=3 on, every iteration
    # whose remainder mod share_iter=8 is 0 to 5 (3, 4, 5, then 8 to 13),
    # whether or not share_comps is on (issue #345).
    assert [i for i in idx if schedule.share_freeze(i, 3, 8)] == [
        2,
        3,
        4,
        7,
        8,
        9,
        10,
        11,
    ]
    # iter .le. restartiter (amica15.f90:1022): the first 3 iterations; 0 turns
    # the recovery off.
    assert [i for i in idx if schedule.within_restart_window(i, 3)] == [0, 1, 2]
    assert fires(lambda i: schedule.within_restart_window(i, 1)) == [0]
    assert fires(lambda i: schedule.within_restart_window(i, 0)) == []


@pytest.mark.parametrize("interval", [0, -2])
def test_a_non_positive_interval_is_a_value_error_not_a_zero_division(interval):
    """The constructors validate every interval they pass in; the shared helpers
    still refuse a non-positive one by name instead of surfacing a bare
    ``ZeroDivisionError`` (or, for a negative one, a silently wrong cadence)."""
    with pytest.raises(ValueError, match=r"every: step must be >= 1"):
        schedule.every(0, interval)
    with pytest.raises(ValueError, match=r"periodic_due: interval must be >= 1"):
        schedule.periodic_due(0, 1, interval)


# --- Newton switch -----------------------------------------------------------


@pytest.mark.parametrize("newt_start", [1, 5])
@pytest.mark.parametrize("backend", BACKENDS)
def test_newton_takes_its_first_step_on_iteration_newt_start(
    backend, newt_start, X, tmp_path
):
    """The first Newton M-step is iteration ``newt_start``'s (0-based index
    ``newt_start - 1``), as ``iter .ge. newt_start`` puts it in the reference.

    Seen two ways on a real fit against its ``do_newton=False`` twin: the
    likelihood trajectories are bit-identical through index ``newt_start - 1``
    (computed before that iteration's M-step) and first differ at index
    ``newt_start``; and the learning rate has climbed the Newton ramp
    (``min(newtrate, lrate + min(1/newt_ramp, lrate))``, amica15.f90:1805) once
    per Newton iteration run, iterations ``newt_start`` through
    ``newt_start + 2``, instead of sitting at its natural-gradient ceiling.
    Before issue #335 the first difference was one index later and the ramp had
    run twice.
    """
    max_iter = newt_start + 2
    cfg: dict[str, Any] = dict(
        lrate=0.05, newtrate=1.0, newt_ramp=10, newt_start=newt_start
    )
    ng = _fit(backend, X, max_iter, tmp_path, do_newton=False, **cfg)
    nt = _fit(backend, X, max_iter, tmp_path, do_newton=True, **cfg)

    assert len(ng.ll) == len(nt.ll) == max_iter
    first_diff = next(i for i in range(max_iter) if ng.ll[i] != nt.ll[i])
    assert first_diff == newt_start, (
        f"the Newton fit first departs at ll index {first_diff}; the reference's "
        f"first Newton M-step (iteration {newt_start}) moves ll index {newt_start}"
    )

    # The ramp arithmetic below assumes no likelihood decrease halved lrate and
    # no Newton fallback retargeted it; both are facts about this data, checked
    # rather than assumed.
    assert all(b >= a for a, b in zip(nt.ll, nt.ll[1:])), "trajectory decreased"
    if nt.n_newton_fallbacks is not None:
        assert nt.n_newton_fallbacks == 0
    lrate = 0.05
    for _ in range(max_iter - newt_start + 1):  # Newton iterations run
        lrate = min(1.0, lrate + min(1.0 / 10, lrate))
    assert nt.lrate == pytest.approx(lrate, rel=1e-12)
    assert ng.lrate == pytest.approx(0.05, rel=1e-12)


@pytest.mark.parametrize("backend", BACKENDS)
def test_newt_start_zero_fits_exactly_as_one(backend, X, tmp_path):
    """``newt_start=0`` and ``newt_start=1`` give bit-identical fits, as the
    docstrings state.

    Newton is active from the first iteration either way. The two settings
    differ only in gates that cannot act on iteration 1: the switch-on reset
    (``iter == newt_start``) fires there for ``1`` and never for ``0``, but it
    clears a decrease counter that no comparison has touched yet; and the
    ``iter > newt_start`` ratchet gate is closed on iteration 1 for ``1``, where
    no ``maxdecs`` cycle can complete because iteration 1 has no predecessor to
    compare against. Checked on the overshooting configuration, so ratchets
    do fire later in the run and both of those gates are live.
    """
    cfg: dict[str, Any] = dict(
        do_newton=True, newtrate=_OVERSHOOT_NEWTRATE, **_OVERSHOOT
    )
    zero = _fit(backend, X, _ZERO_ONE_ITERS, tmp_path, newt_start=0, **cfg)
    one = _fit(backend, X, _ZERO_ONE_ITERS, tmp_path, newt_start=1, **cfg)

    assert zero.ll == one.ll
    for name in ("lrate", "lrate_ceiling", "rholrate", "newtrate"):
        assert getattr(zero, name) == getattr(one, name), name
    # Not vacuous: the run completed maxdecs cycles past iteration 1, so the
    # ratchet gate both settings read was exercised.
    assert _ratchets(one.lrate_ceiling, one.lrate_ceiling0, 0.5) > 0
    assert _ratchets(one.newtrate, one.newtrate0, 0.5) > 0


# --- the iter > newt_start ceiling ratchet -----------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_rho_rate_ratchet_opens_only_after_iteration_newt_start(backend, X, tmp_path):
    """A ``maxdecs`` ratchet on iteration ``k`` (1-based) tightens the rho-rate
    ceiling only when ``k > newt_start`` (amica15.f90:1067, whether or not Newton
    is on), while the ``lrate`` ceiling ratchets regardless.

    ``k`` is read off the executing machine's own trajectory: the first ratchet
    of the probe. Each fit then stops right after it. With ``newt_start = k``
    the gate is closed and the rho-rate ceiling must not move; with
    ``newt_start = k - 1`` it is open and must move exactly once. Before issue
    #335 the 0-based comparison kept the gate closed in both.
    """
    probe = _probe(backend, X, tmp_path)
    hits = _ratchet_indices(probe.ll, _OVERSHOOT["maxdecs"], None)
    assert hits, "the probe never completed a maxdecs cycle; the DATA is the problem"
    k = hits[0] + 1  # the reference iteration the first ratchet lands on

    for newt_start, rho_ratchets in ((k, 0), (k - 1, 1)):
        run = _fit(
            backend,
            X,
            k,
            tmp_path,
            do_newton=False,
            newt_start=newt_start,
            rholratefact=0.5,
            **_OVERSHOOT,
        )
        # Nothing before the ratchet reads newt_start, so this is the probe's
        # own trajectory: the ratchet under test is the one located above.
        assert run.ll == probe.ll[:k]
        f = _OVERSHOOT["lratefact"]
        assert _ratchets(run.lrate_ceiling, run.lrate_ceiling0, f) == 1
        assert _ratchets(run.rholrate, run.rholrate0, 0.5) == rho_ratchets, (
            f"newt_start={newt_start}: a ratchet on iteration {k} "
            f"{'must' if rho_ratchets else 'must not'} tighten the rho-rate ceiling"
        )
        assert run.newtrate == run.newtrate0  # do_newton=False never touches it


# A natural-gradient run whose second maxdecs cycle completes on iteration 21,
# one past the shipped default newt_start=20. Found by sweeping seeds 0-5,
# lrate 0.5/0.6/0.8, maxdecs 2/3, newt_ramp 10/1 and 4096/8192 frames for a
# ratchet at ll index 20, first on PyTorch and then checked on NumPy and MLX;
# re-searched when issue #333's component-row doscaling changed the default
# trajectories, and again when issue #339 moved the decrease response ahead of
# the update (seed 1, the previous choice, now ratchets at 8 and 16). Seven
# configurations qualify on PyTorch; this one, seed 3, also does on NumPy and
# MLX (seed 4, the widest-margin PyTorch candidate, ratchets at 16 in float32).
# It decreases at ll indices 4, 5, 8 and 14, 15, 20 on all three backends, so it
# ratchets at 8 and 20; the smallest decrease is 2.4e-3 and the smallest
# likelihood step of the run 4.4e-4, far above round-off, float32 included.
_DEFAULT_GATE_FRAMES = 4096
_DEFAULT_GATE_SEED = 3
_DEFAULT_GATE: dict[str, Any] = dict(lrate=0.6, lratefact=0.5, maxdecs=3, newt_ramp=1)


@pytest.mark.parametrize("backend", BACKENDS)
def test_rho_rate_ratchet_gate_at_the_shipped_default_newt_start(backend, X, tmp_path):
    """The same gate at ``newt_start=20``, the default of every backend: a
    ratchet on iteration 21 tightens the rho-rate ceiling, one on iteration 9
    does not, and with ``newt_start=21`` neither does. This is the one
    default-path behavior issue #335 changes (``do_newton=False``)."""
    x = X[:, :_DEFAULT_GATE_FRAMES]
    runs = {
        newt_start: _fit(
            backend,
            x,
            21,
            tmp_path,
            seed=_DEFAULT_GATE_SEED,
            do_newton=False,
            newt_start=newt_start,
            rholratefact=0.5,
            **_DEFAULT_GATE,
        )
        for newt_start in (20, 21)
    }
    hits = _ratchet_indices(runs[21].ll, _DEFAULT_GATE["maxdecs"], None)
    assert hits and hits[-1] == 20, (
        f"ratchets at ll indices {hits}; the configuration no longer completes a "
        "cycle on iteration 21 on this DATA"
    )
    assert runs[20].ll == runs[21].ll  # nothing before the ratchet reads it
    for newt_start, rho_ratchets in ((20, 1), (21, 0)):
        run = runs[newt_start]
        assert _ratchets(run.lrate_ceiling, run.lrate_ceiling0, 0.5) == len(hits)
        assert _ratchets(run.rholrate, run.rholrate0, 0.5) == rho_ratchets, (
            f"newt_start={newt_start}"
        )


# --- the switch-on counter reset ---------------------------------------------


# The switch-on reset is observable only when the likelihood decreases right
# after the switch-on iteration while the decrease counter is partly filled,
# which asks the first Newton M-step to overshoot. A search on PyTorch, about 45
# CPU-minutes (1 and 2 models; 4096 and 8192 frames; seeds 0-3; lrate 0.3 to
# 0.8; maxdecs 2 and 3; newt_ramp 10 and 1; newtrate 2 and 4; up to 8 candidate
# newt_start values per configuration, read off 60-iteration probes), found such
# runs only at lrate=0.8 with newt_start=3, where the natural gradient
# overshoots from the first iteration and the switch-on Hessian is not yet
# positive definite, so the step falls back to the natural gradient and
# overshoots again, plus one two-model run at lrate=0.5 and newt_start=11. This
# one-model configuration is the most robust of them: ``newt_ramp=1`` restores
# the rate after each halving, the decreases are 3.8e-3, 2.0e-2 and 3.0e-2, far
# above round-off, and the pattern holds on seeds 0-5 and all three backends.
_RESET_FRAMES = 4096
_RESET: dict[str, Any] = dict(
    do_newton=True,
    newt_start=3,
    newtrate=2.0,
    newt_ramp=1,
    lrate=0.8,
    lratefact=0.5,
    maxdecs=3,
)


@pytest.mark.parametrize("backend", BACKENDS)
def test_newton_switch_on_clears_the_decrease_counter(backend, X, tmp_path):
    """The decrease counter is cleared on the iteration Newton switches on
    (``iter == newt_start``, amica15.f90:1099), not one iteration later.

    With ``newt_start=3`` and ``maxdecs=3`` this 4-iteration fit decreases the
    likelihood on iterations 2, 3 and 4 (``ll`` indices 1-3). The reference
    clears the count of two after iteration 3, so the decrease on iteration 4
    starts a new count and nothing ratchets. Clearing it one iteration later,
    as the 0-based comparison did before issue #335, lets that decrease
    complete the cycle and ratchet the ``lrate`` ceiling. The fit's own
    trajectory is replayed under both rules to confirm that they disagree
    here, and the backend's ceiling must match the reference's.
    """
    newt_start, maxdecs = _RESET["newt_start"], _RESET["maxdecs"]
    run = _fit(backend, X[:, :_RESET_FRAMES], newt_start + 1, tmp_path, **_RESET)

    decreases = [i for i in range(1, len(run.ll)) if run.ll[i] < run.ll[i - 1]]
    assert decreases == [1, 2, 3], (
        f"decreases at ll indices {decreases}; the configuration no longer "
        "produces the pattern that exposes the reset on this DATA"
    )
    # _ratchet_indices resets where ``i + 1 == newt_start``; passing
    # newt_start + 1 replays the pre-fix rule, one iteration later.
    reference = len(_ratchet_indices(run.ll, maxdecs, newt_start))
    one_late = len(_ratchet_indices(run.ll, maxdecs, newt_start + 1))
    assert (reference, one_late) == (0, 1)
    assert _ratchets(run.lrate_ceiling, run.lrate_ceiling0, 0.5) == reference


# --- rejection ---------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_rejection_first_fires_on_iteration_rejstart(backend, X, tmp_path):
    """With ``rejstart=4`` the first rejection pass follows the 4th iteration's
    update (``iter == rejstart``, amica15.f90:1136): a 3-iteration fit has not
    rejected, a 4-iteration fit has, once, and dropped samples. ``rejint=3``
    keeps the modulo arm quiet before ``rejstart`` (``max(1, iter - rejstart)``
    is 1 there). Before issue #335 the 4-iteration fit had not rejected yet.
    """
    cfg: dict[str, Any] = dict(
        do_reject=True, rejstart=4, rejint=3, maxrej=1, rejsig=2.0
    )
    before = _fit(backend, X, 3, tmp_path, **cfg)
    assert before.numrej == 0 and before.n_good == X.shape[1]

    at = _fit(backend, X, 4, tmp_path, **cfg)
    assert at.numrej == 1
    assert at.n_good < X.shape[1]


# --- NumPy's restart-on-NaN window --------------------------------------------


class _NaNOnIteration(AMICA_NumPy):
    """Poisons the real E-step's likelihood on one 0-based iteration.

    The sanctioned error-injection pattern (``.rules/testing.md``; the same
    construction as ``test_sample_data.py``'s ``_InjectEarlyNaN``): every other
    code path, the fit and the restart itself included, is the real one on real
    data. Needed because a non-finite likelihood is not something the sample
    recording produces on demand at a chosen iteration.
    """

    nan_iter = -1

    def _get_updates_and_likelihood(self):
        upd = super()._get_updates_and_likelihood()
        if self.iter == self.nan_iter:
            upd["ll"] = float("nan")
        return upd


@pytest.mark.parametrize(
    ("restartiter", "nan_iter", "restarts"), [(3, 2, 1), (3, 3, 0), (0, 0, 0)]
)
def test_numpy_restart_window_is_the_first_restartiter_iterations(
    restartiter, nan_iter, restarts, X, tmp_path
):
    """A non-finite likelihood restarts from a fresh draw only within the first
    ``restartiter`` iterations (``iter .le. restartiter``, amica15.f90:1022).

    NumPy only: it is the one backend with a restart-on-NaN recovery (PyTorch
    and MLX stop on a non-finite likelihood, and their best-of-N
    ``n_restarts`` is a different mechanism). With ``restartiter=3``, a NaN on
    the 3rd iteration (index 2) restarts and the fit recovers; a NaN on the 4th
    (index 3) is past the window and ends the fit. ``restartiter=0`` disables
    the recovery, as in the reference: a NaN on the very first iteration ends
    the fit although ``maxrestarts`` would allow a restart. Before issue #335
    both the 4th iteration and, with ``restartiter=0``, the first still
    restarted.
    """
    m = _NaNOnIteration(
        num_models=1,
        num_mix=NMIX,
        seed=SEED,
        block_size=BLOCK,
        max_iter=8,
        use_tqdm=False,
        do_opt_block=False,
        writestep=10**7,
        restartiter=restartiter,
        maxrestarts=3,
        outdir=str(tmp_path / "out"),
    )
    m.nan_iter = nan_iter
    m.fit(X)
    assert m.numrestarts == restarts
    assert m.converged is bool(restarts), m.stop_reason
    if not restarts:
        assert m.stop_reason is not None and "Non-finite" in m.stop_reason
        # Stopped on the poisoned iteration, whose non-finite likelihood is not
        # recorded (as in the PyTorch/MLX ll_history, issue #339 review).
        assert len(m.ll) == nan_iter


def test_numpy_restart_clears_the_small_gain_count_as_the_reference_does(X, tmp_path):
    """The reference's checks still run on its restart iteration, and its
    ``min_dll`` comparison with the NaN likelihood is false, so it zeroes
    ``numincs`` (amica15.f90:1078-1090). With every gain counted as small
    (``min_dll=10``) and ``maxincs=3``, two small gains before a restart on the
    4th iteration (index 3) must not count after it: the fit stops on the 5th
    small gain after the restart, at index 8, not at index 6, where the gains
    carried across the restart (the behavior before the issue #339 review)
    would stop it."""
    m = _NaNOnIteration(
        num_models=1,
        num_mix=NMIX,
        seed=SEED,
        block_size=BLOCK,
        max_iter=20,
        use_tqdm=False,
        do_opt_block=False,
        writestep=10**7,
        restartiter=10,
        maxrestarts=3,
        use_min_dll=True,
        min_dll=10.0,
        maxincs=3,
        use_grad_norm=False,
        outdir=str(tmp_path / "out"),
    )
    m.nan_iter = 3
    m.fit(X)
    assert m.numrestarts == 1
    assert m.stop_reason == "Converged: small likelihood increase"
    # Indices 4 to 8 after the restart: one to start the history, then four
    # small gains, the fourth of which exceeds maxincs=3.
    assert m.iter == 8
    assert len(m.ll) == 5


# The epic base before issue #339, the last NumPy loop that updated before its
# restart check.
_PRE_339 = "5b6ae4f69eacf6aa18002ce336ed904f73438516"


def _updates_around_a_restart(base: Any, X: np.ndarray, tmp_path: Path) -> tuple:
    """Fit ``base`` with a NaN likelihood on iteration 2 inside the restart
    window, recording the iteration of every ``_update_parameters`` call with a
    pass-through recorder (it records, then calls the real method with the same
    arguments). Returns the recorded iterations and the restart count."""

    class _NaNOnTwo(base):
        def _get_updates_and_likelihood(self):
            upd = super()._get_updates_and_likelihood()
            if self.iter == 2:
                upd["ll"] = float("nan")
            return upd

    m = _NaNOnTwo(
        num_models=1,
        num_mix=NMIX,
        seed=SEED,
        block_size=BLOCK,
        max_iter=5,
        use_tqdm=False,
        do_opt_block=False,
        writestep=10**7,
        restartiter=10,
        maxrestarts=3,
        outdir=str(tmp_path / "out"),
    )
    updated: list = []
    real_update = m._update_parameters

    def record(*args, **kwargs):
        updated.append(m.iter)
        return real_update(*args, **kwargs)

    m._update_parameters = record
    m.fit(X)
    return updated, m.numrestarts


def test_numpy_restart_iteration_applies_no_update(X, tmp_path, tmp_path_factory):
    """The reference checks for a restart before ``update_params`` and skips
    the update on a restarting iteration (``startover``, amica15.f90:1115-1122).
    Since issue #339 the NumPy loop does the same: iteration 2, whose
    likelihood is NaN, redraws A and applies no update, while every other
    iteration updates. The same fit of the code before issue #339, the control,
    updated on iteration 2 too, from the parameters whose likelihood was NaN."""
    updated, restarts = _updates_around_a_restart(AMICA_NumPy, X, tmp_path / "now")
    assert restarts == 1
    assert updated == [0, 1, 3, 4]

    from pamica.tests.pre_change import load_pre_change_package

    pre = load_pre_change_package(
        _PRE_339, "pamica_pre339_gates", tmp_path_factory.mktemp("pre339")
    )
    old, old_restarts = _updates_around_a_restart(
        pre.numpy_impl.core.AMICA, X, tmp_path / "pre"
    )
    assert old_restarts == 1
    assert old == [0, 1, 2, 3, 4], "control: the old loop no longer updates first"


# --- validation of the settings the gates read ------------------------------


def _construct(backend: str, **kwargs: Any):
    """A backend built with ``kwargs`` (torch/MLX spelling), no data touched."""
    if backend == "numpy":
        return AMICA_NumPy(use_tqdm=False, **kwargs)
    if backend == "torch":
        return AMICATorchNG(n_channels=NW, device="cpu", **kwargs)
    return _mlx_class()(n_channels=NW, **kwargs)


@pytest.mark.parametrize(
    ("setting", "value", "minimum", "extra"),
    [
        # Checked whether or not Newton is on: it also gates the rho-rate
        # ratchet on the natural-gradient path.
        ("newt_start", -1, 0, {}),
        ("newt_start", -1, 0, {"do_newton": True}),
        ("newt_start", 2.5, 0, {}),
        ("newt_start", True, 0, {}),
        # Counted from 1: rejstart <= 0 would silently skip the reference's
        # unconditional first pass.
        ("rejstart", 0, 1, {"do_reject": True}),
        ("rejstart", -2, 1, {"do_reject": True}),
        ("rejstart", 2.0, 1, {"do_reject": True}),
        # Checked whether or not share_comps is on: the reference's A-freeze
        # reads it on every fit, and 0 would hold A from the first iteration
        # (issue #339 review).
        ("share_start", 0, 1, {}),
        ("share_start", -5, 1, {"share_comps": True}),
        ("share_start", 2.5, 1, {}),
        ("share_start", True, 1, {}),
    ],
)
@pytest.mark.parametrize("backend", BACKENDS)
def test_schedule_settings_are_rejected_with_one_message(
    backend, setting, value, minimum, extra
):
    """Every backend refuses the same ``newt_start``/``rejstart``/
    ``share_start`` values with the same message
    (``pamica.schedule.validate_iteration_setting``)."""
    with pytest.raises(ValueError) as exc:
        _construct(backend, **{setting: value}, **extra)
    assert str(exc.value) == f"{setting} must be an integer >= {minimum}, got {value!r}"


@pytest.mark.parametrize("backend", BACKENDS)
def test_boundary_schedule_settings_are_accepted(backend):
    """The smallest meaningful values construct, numpy integers included, and
    ``rejstart`` is inert (unchecked) while ``do_reject`` is off.
    ``share_start`` is not inert without sharing: the A-freeze reads it."""
    for kwargs in (
        {"newt_start": 0},
        {"newt_start": np.int64(3)},
        {"do_reject": True, "rejstart": 1},
        {"rejstart": 0},
        {"share_start": 1},
        {"share_start": np.int64(1), "share_comps": True},
    ):
        _construct(backend, **kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"do_history": True, "histstep": 0},
            "histstep must be an integer >= 1, got 0",
        ),
        ({"restartiter": -1}, "restartiter must be an integer >= 0, got -1"),
        ({"maxrestarts": -1}, "maxrestarts must be an integer >= 0, got -1"),
        ({"restartiter": 2.5}, "restartiter must be an integer >= 0, got 2.5"),
    ],
)
def test_numpy_only_schedule_settings_are_validated(kwargs, message):
    """The NumPy backend's own schedule settings (no PyTorch or MLX
    counterpart): ``histstep=0`` with ``do_history`` on used to be a bare
    ``ZeroDivisionError`` on the first iteration, and a negative
    ``restartiter``/``maxrestarts`` has no meaning."""
    with pytest.raises(ValueError) as exc:
        AMICA_NumPy(use_tqdm=False, **kwargs)
    assert str(exc.value) == message
    # The inert and boundary values construct.
    AMICA_NumPy(use_tqdm=False, do_history=False, histstep=0)
    AMICA_NumPy(use_tqdm=False, restartiter=0, maxrestarts=0)
