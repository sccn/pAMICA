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
# sample counts, and here it completes the first maxdecs cycle at iteration 11
# on all three backends. Every test below still reads the iterations it needs
# off the executing machine's own trajectory rather than trusting that number.
_OVERSHOOT: dict[str, Any] = dict(lrate=0.6, lratefact=0.5, maxdecs=2)
# The Newton-phase ceiling test_mlx_newton.py pairs with it: at newtrate=1.0
# the post-switch-on trajectory can be monotone, leaving the reset unobservable.
_OVERSHOOT_NEWTRATE = 2.0
_PROBE_ITERS = 40
_NEWTON_ITERS = 120


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
    rholrate: float
    rholrate0: float
    newtrate: float
    newtrate0: float
    numrej: int
    n_good: int
    n_newton_fallbacks: int | None  # NumPy does not count them


def _fit(backend: str, x: np.ndarray, max_iter: int, tmp_path: Path, **cfg) -> _Run:
    """One real fit on ``backend``; ``cfg`` uses the torch/MLX parameter names."""
    if backend == "numpy":
        cfg = dict(cfg)
        if "maxdecs" in cfg:
            cfg["max_decs"] = cfg.pop("maxdecs")
        m = AMICA_NumPy(
            num_models=1,
            num_mix=NMIX,
            seed=SEED,
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
            rholrate=m.rholrate,
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
            seed=SEED,
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
            seed=SEED,
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
        rholrate=m.rholrate,
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

    # iter .ge. newt_start (amica15.f90:1804): the 5th iteration is index 4.
    assert first(lambda i: schedule.newton_active(True, i, 5)) == 4
    assert not any(schedule.newton_active(False, i, 5) for i in idx)
    # newt_start 0 and 1 both mean "from the first iteration".
    assert first(lambda i: schedule.newton_active(True, i, 1)) == 0
    assert first(lambda i: schedule.newton_active(True, i, 0)) == 0
    # iter == newt_start (amica15.f90:1099): exactly once, on index 4.
    assert [i for i in idx if schedule.newton_switches_on(True, i, 5)] == [4]
    # iter > newt_start (amica15.f90:1067): from the 6th iteration, index 5.
    assert first(lambda i: schedule.past_newton_start(i, 5)) == 5
    # Rejection (amica15.f90:1136) with rejstart=4, rejint=3: the 4th
    # iteration (index 3), then every 3rd while numrej < maxrej (held at 0
    # here, so the modulo arm keeps firing).
    assert [i for i in idx if schedule.rejection_due(True, i, 4, 3, 0, 2)] == [3, 6, 9]
    assert not any(schedule.rejection_due(True, i, 4, 3, 0, 0) for i in idx)
    # mod(iter, writestep) == 0 (amica15.f90:1124): the 5th and 10th, not the 1st.
    assert [i for i in idx if schedule.every(i, 5)] == [4, 9]
    # Share merges from share_start=3 every 4 (amica15.f90:1856): 3, 7, 11.
    assert [i for i in idx if schedule.periodic_due(i, 3, 4)] == [2, 6, 10]
    # The A-freeze: the merge iteration and the 5 after it, anchored on
    # share_start (a documented pamica decision), share_start=3, share_iter=8.
    assert [i for i in idx if schedule.share_freeze(i, 3, 8)] == [
        2,
        3,
        4,
        5,
        6,
        7,
        10,
        11,
    ]
    # iter .le. restartiter (amica15.f90:1022): the first 3 iterations.
    assert [i for i in idx if schedule.within_restart_window(i, 3)] == [0, 1, 2]


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


@pytest.mark.parametrize("backend", BACKENDS)
def test_newton_takes_its_first_step_on_iteration_newt_start(backend, X, tmp_path):
    """With ``newt_start=5`` the first Newton M-step is the 5th iteration's (0-based
    index 4), as ``iter .ge. newt_start`` puts it in the reference.

    Seen two ways on a real fit against its ``do_newton=False`` twin: the
    likelihood trajectories are bit-identical through index 4 (computed before
    that iteration's M-step) and first differ at index 5; and the learning rate
    has climbed the Newton ramp (``min(newtrate, lrate + min(1/newt_ramp,
    lrate))``, amica15.f90:1805) once per Newton iteration run, iterations 5
    through 7, instead of sitting at its natural-gradient ceiling. Before issue
    #335 the first difference was at index 6 and the ramp had run twice.
    """
    newt_start, max_iter = 5, 7
    cfg = dict(lrate=0.05, newtrate=1.0, newt_ramp=10, newt_start=newt_start)
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
    for _ in range(max_iter - newt_start + 1):  # Newton iterations 5, 6, 7
        lrate = min(1.0, lrate + min(1.0 / 10, lrate))
    assert nt.lrate == pytest.approx(lrate, rel=1e-12)
    assert ng.lrate == pytest.approx(0.05, rel=1e-12)


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


# --- the switch-on counter reset ---------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_newton_switch_on_clears_the_decrease_counter(backend, X, tmp_path):
    """The decrease counter is cleared on the iteration Newton switches on
    (``iter == newt_start``, amica15.f90:1099), so a count left partly filled by
    the natural-gradient phase cannot ratchet the ceilings under Newton.

    The cross-backend form of ``mlx_tests/test_mlx_newton.py``'s
    ``test_numdecs_resets_when_newton_switches_on``, with the same data-driven
    construction: ``newt_start`` is the 1-based iteration at which the probe's
    counter is partly filled; the smallest budget at which the reference's
    reset and a no-reset replay predict different ratchet counts is found on
    the Newton fit's own trajectory (the loop is causal, so a shorter budget
    reproduces the prefix); and the fit truncated there must have ratcheted its
    ``lrate`` ceiling exactly as the reference's reset predicts.
    """
    probe = _probe(backend, X, tmp_path)
    maxdecs = _OVERSHOOT["maxdecs"]
    numdecs, partial = 0, None
    for i in range(1, len(probe.ll)):
        if probe.ll[i] < probe.ll[i - 1]:
            numdecs += 1
            if numdecs >= maxdecs:
                numdecs = 0
        if 0 < numdecs < maxdecs:
            partial = i
            break
    assert partial is not None, "the probe never left the counter partly filled"
    newt_start = partial + 1  # the reset lands on ll index ``partial``

    cfg = dict(
        do_newton=True,
        newt_start=newt_start,
        newtrate=_OVERSHOOT_NEWTRATE,
        **_OVERSHOOT,
    )
    ll = _fit(backend, X, _NEWTON_ITERS, tmp_path, **cfg).ll
    budget = next(
        (
            t
            for t in range(2, len(ll) + 1)
            if len(_ratchet_indices(ll[:t], maxdecs, newt_start))
            != len(_ratchet_indices(ll[:t], maxdecs, None))
        ),
        None,
    )
    assert budget is not None, (
        f"over {len(ll)} iterations the reset never changed a ratchet count at "
        f"newt_start={newt_start}; the DATA did not decrease enough under Newton"
    )

    run = _fit(backend, X, budget, tmp_path, **cfg)
    assert run.ll == ll[:budget], "the truncated fit left the full fit's trajectory"
    expected = len(_ratchet_indices(run.ll, maxdecs, newt_start))
    assert _ratchets(run.lrate_ceiling, run.lrate_ceiling0, 0.5) == expected


# --- rejection ---------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_rejection_first_fires_on_iteration_rejstart(backend, X, tmp_path):
    """With ``rejstart=4`` the first rejection pass follows the 4th iteration's
    update (``iter == rejstart``, amica15.f90:1136): a 3-iteration fit has not
    rejected, a 4-iteration fit has, once, and dropped samples. ``rejint=3``
    keeps the modulo arm quiet before ``rejstart`` (``max(1, iter - rejstart)``
    is 1 there). Before issue #335 the 4-iteration fit had not rejected yet.
    """
    cfg = dict(do_reject=True, rejstart=4, rejint=3, maxrej=1, rejsig=2.0)
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
        assert len(m.ll) == nan_iter + 1  # stopped on the poisoned iteration


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
    ],
)
@pytest.mark.parametrize("backend", BACKENDS)
def test_schedule_settings_are_rejected_with_one_message(
    backend, setting, value, minimum, extra
):
    """Every backend refuses the same ``newt_start``/``rejstart`` values with
    the same message (``pamica.schedule.validate_iteration_setting``)."""
    with pytest.raises(ValueError) as exc:
        _construct(backend, **{setting: value}, **extra)
    assert str(exc.value) == f"{setting} must be an integer >= {minimum}, got {value!r}"


@pytest.mark.parametrize("backend", BACKENDS)
def test_boundary_schedule_settings_are_accepted(backend):
    """The smallest meaningful values construct, numpy integers included, and
    ``rejstart`` is inert (unchecked) while ``do_reject`` is off, the pattern
    ``share_start`` follows."""
    for kwargs in (
        {"newt_start": 0},
        {"newt_start": np.int64(3)},
        {"do_reject": True, "rejstart": 1},
        {"rejstart": 0},
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
