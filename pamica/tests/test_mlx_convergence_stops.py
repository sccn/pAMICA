"""Convergence stops on the MLX backend, cross-checked against PyTorch (#248).

Before this, ``AMICAMLXNG`` implemented neither Fortran stop: an MLX fit always
spent the whole iteration budget while every other backend stopped on likelihood
stagnation. These tests pin the ported behaviour to the PyTorch backend, which is
the porting reference:

1. ``use_min_dll``/``min_dll``/``maxincs`` (amica15.f90:1060-1072)
2. ``use_grad_norm``/``min_nd`` over the per-iteration weight-gradient norm
   ``ndtmpsum`` (amica15.f90:1073-1079, accumulated at :1731-1743)
3. the likelihood-decrease branch's ``.or. (ndtmpsum .le. min_nd)`` half
   (amica15.f90:1040), ``stop_reason="grad_norm_floor"``

Cross-backend by design (``.rules/backend_parity.md``): the two backends must
stop at the SAME iteration with the SAME ``stop_reason`` on the same real data.
They cannot be compared bit-for-bit (MLX is float32 on the Apple GPU, and here
PyTorch runs float32 on the CPU, so reduction orders differ), so the thresholds
below are chosen coarse: at the iteration each stop's condition flips, the two
backends' values differ by ~1e-6 while the distance to the threshold is ~1e-4,
a ~100x margin. That is what makes the exact-iteration assertions stable rather
than a snapshot of one machine.

Real bundled sample EEG only, no synthetic data or mocks (``.rules/testing.md``).
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from pamica.torch_impl.core import AMICATorchNG

mx = pytest.importorskip("mlx.core", reason="MLX not installed (Apple Silicon only)")
mlx_core = pytest.importorskip(
    "pamica.mlx_impl.core", reason="MLX not installed (Apple Silicon only)"
)
AMICAMLXNG = mlx_core.AMICAMLXNG

SAMPLE_DIR = Path(__file__).resolve().parents[1] / "sample_data"
DATA_FILE = SAMPLE_DIR / "eeglab_data.fdt"
NW = 32
FIELD = 30504
NMIX = 3
SEED = 42
BLOCK = 8192  # matched on both backends: blocking shifts the trajectory ~1e-6

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


def _mlx(**kwargs):
    kwargs.setdefault("n_channels", NW)
    kwargs.setdefault("n_mix", NMIX)
    kwargs.setdefault("seed", SEED)
    kwargs.setdefault("block_size", BLOCK)
    return AMICAMLXNG(**kwargs)


def _torch(**kwargs) -> AMICATorchNG:
    """The same configuration on the porting reference: float32 (MLX has no
    float64) and no Newton (MLX has none), so the only remaining difference is
    the array library and the device."""
    kwargs.setdefault("n_channels", NW)
    kwargs.setdefault("n_models", 1)
    kwargs.setdefault("n_mix", NMIX)
    kwargs.setdefault("seed", SEED)
    kwargs.setdefault("block_size", BLOCK)
    kwargs.setdefault("device", "cpu")
    kwargs.setdefault("dtype", torch.float32)
    kwargs.setdefault("do_newton", False)
    return AMICATorchNG(**kwargs)


# --- the parameters exist and are validated --------------------------------


def test_negative_maxincs_is_rejected():
    """Same guard as AMICATorchNG: a negative maxincs would make the stop fire
    on the first small gain, silently, so it is refused at construction."""
    with pytest.raises(ValueError, match="maxincs must be >= 0"):
        _mlx(maxincs=-1)


# --- stop 1: use_min_dll / min_dll / maxincs -------------------------------


def test_min_dll_stops_at_the_same_iteration_as_torch(real_data):
    """The consecutive-small-gain stop fires on the same iteration, with the
    same stop_reason, as the PyTorch backend at a matched configuration.

    ``min_dll=7e-4`` is deliberately far coarser than the shipped ``1e-9``
    (which needs several hundred iterations on this recording, issue #218): the
    likelihood gain crosses 7e-4 between iterations 9 (~7.8e-4) and 10 (~6.1e-4),
    so the counter starts on a gain that is ~8e-5 clear of the threshold, while
    the two backends disagree by only ~1e-6 there. ``use_grad_norm`` is off to
    isolate this stop from the gradient-norm one.
    """
    mlx_m = _mlx(use_min_dll=True, min_dll=7e-4, maxincs=5, use_grad_norm=False)
    mlx_m.fit(real_data, max_iter=200, verbose=False)

    ng = _torch(use_min_dll=True, min_dll=7e-4, maxincs=5, use_grad_norm=False)
    ng.fit(real_data, max_iter=200, verbose=False)

    assert mlx_m.stop_reason == "min_dll"
    assert ng.stop_reason == "min_dll"
    assert len(mlx_m.ll_history) == len(ng.ll_history)
    # Stopped well inside the budget: a max_iter-bounded run would prove nothing.
    assert len(mlx_m.ll_history) < 200
    # The counting rule itself (Fortran numincs): the stop lands after MORE than
    # maxincs consecutive small gains, and the run of small gains is unbroken.
    gains = np.diff(np.asarray(mlx_m.ll_history))
    assert np.all(gains[-(mlx_m.maxincs + 1) :] < mlx_m.min_dll)
    assert gains[-(mlx_m.maxincs + 2)] >= mlx_m.min_dll


def test_min_dll_never_fires_before_two_likelihoods(real_data):
    """Fortran wraps all three stops in ``if (iter > 1)`` (amica15.f90:1033).
    An absurdly generous min_dll with maxincs=0 (stop on the first small gain)
    must therefore still produce two likelihood values, never one."""
    m = _mlx(use_min_dll=True, min_dll=1e6, maxincs=0, use_grad_norm=False)
    m.fit(real_data[:, :4096], max_iter=5, verbose=False)

    assert m.stop_reason == "min_dll"
    assert len(m.ll_history) == 2


# --- stop 2: use_grad_norm / min_nd (ndtmpsum) -----------------------------


def test_grad_norm_stops_at_the_same_iteration_as_torch(real_data):
    """The gradient-norm stop fires on the same iteration as PyTorch, and the
    gradient norm itself agrees between the backends.

    ``min_nd=1.02e-2`` sits between the norms at iterations 12 (~1.033e-2) and
    13 (~1.001e-2), so the crossing clears the threshold by ~1.3e-4 while the
    backends differ by ~3e-6. (The shipped ``1e-7`` is unreachable on a
    recording this size in ANY backend, issue #218, so it cannot be used to
    exercise the path.) ``use_min_dll`` is off to isolate this stop.
    """
    mlx_m = _mlx(use_min_dll=False, use_grad_norm=True, min_nd=1.02e-2)
    mlx_m.fit(real_data, max_iter=200, verbose=False)

    ng = _torch(use_min_dll=False, use_grad_norm=True, min_nd=1.02e-2)
    ng.fit(real_data, max_iter=200, verbose=False)

    assert mlx_m.stop_reason == "grad_norm"
    assert ng.stop_reason == "grad_norm"
    assert len(mlx_m.ll_history) == len(ng.ll_history)
    assert len(mlx_m.ll_history) < 200
    # Both stopped because the SAME quantity crossed: the norms agree to float32
    # precision, and each is at or below the threshold that ended its run.
    assert mlx_m._ndtmpsum is not None and ng._ndtmpsum is not None
    assert mlx_m._ndtmpsum <= mlx_m.min_nd
    assert mlx_m._ndtmpsum == pytest.approx(ng._ndtmpsum, rel=1e-3)


def test_grad_norm_never_fires_before_two_likelihoods(real_data):
    """The have_prev guard again (amica15.f90:1033), from the grad-norm side:
    a min_nd above any possible norm still leaves two likelihood values."""
    m = _mlx(use_min_dll=False, use_grad_norm=True, min_nd=1e6)
    m.fit(real_data[:, :4096], max_iter=5, verbose=False)

    assert m.stop_reason == "grad_norm"
    assert len(m.ll_history) == 2


def test_grad_norm_is_the_norm_of_the_step_actually_taken(real_data):
    """``ndtmpsum`` is Fortran's ``sqrt(sum(dAk*dAk)/(nw*num_comps))`` over the
    step direction built BEFORE the lrate scaling and before the A update
    applies it (amica15.f90:1731-1743, strictly ahead of update_params at
    :1789) -- not a norm of the updated A, and not the raw ``dWtmp`` block sum
    (the bug issue #212 found in the NumPy backend, where the reported value was
    five orders of magnitude off and carried no convergence information).

    Recomputed here from the step the fit actually took, independently of how
    the backend computed it: for a single model with rescaling disabled the
    applied step IS ``lrate * dAk`` (``A -= lrate * dAk``, with zeta == gm == 1),
    so ``dAk == (A_before - A_after) / lrate``. rtol is 1e-5 because that
    reconstruction subtracts two nearly equal float32 matrices.
    """
    data = real_data[:, :4096]
    m = _mlx(doscaling=False, block_size=2048)
    m.fit(data, max_iter=3, verbose=False)

    acc = m._accumulate_blocks(m._preprocess(data))
    a_before = np.array(m.A, dtype=np.float64)
    m._update_parameters(acc, data.shape[1])
    # _update_parameters ramps self.lrate before taking the step, so this is the
    # lrate that scaled it.
    dAk = (a_before - np.array(m.A, dtype=np.float64)) / m.lrate
    expected = np.sqrt(np.sum(dAk**2) / (m.n_channels * m.n_comps))

    assert m._ndtmpsum is not None
    assert m._ndtmpsum == pytest.approx(expected, rel=1e-5)


# --- stop 3: the decrease branch's ``.or. ndtmpsum <= min_nd`` half ---------


def test_grad_norm_floor_fires_on_a_likelihood_decrease(real_data):
    """The second disjunct of amica15.f90:1040. With ``use_grad_norm`` off (so
    the unconditional check of stop 2 cannot fire) and a min_nd above any norm
    this fit reaches, the only way out is the decrease branch -- and it must
    take it while lrate is nowhere near its floor, which is exactly the case the
    old lrate-only condition could not stop (issue #207).

    ``lrate=0.5`` is aggressive enough that the natural gradient overshoots on
    this recording and the likelihood decreases within a few iterations.
    """
    m = _mlx(lrate=0.5, use_min_dll=False, use_grad_norm=False, min_nd=1.0)
    m.fit(real_data[:, :4096], max_iter=40, verbose=False)

    assert m.stop_reason == "grad_norm_floor"
    assert m.lrate > m.minlrate  # NOT the pre-existing lrate_floor path
    assert m.ll_history[-1] < m.ll_history[-2]  # the branch it lives in
    assert len(m.ll_history) < 40


def test_lrate_floor_still_fires_when_grad_norm_cannot(real_data):
    """Regression guard on the other disjunct: the pre-existing ``lrate <=
    minlrate`` half must still stop on its own. Same overshooting config, but
    with ``min_nd=0.0`` (a norm is never <= 0) and ``minlrate`` raised above the
    working lrate, so only the lrate half can fire. The two tests together
    prove Fortran's ``or`` works from either side."""
    m = _mlx(
        lrate=0.5, minlrate=0.6, use_min_dll=False, use_grad_norm=False, min_nd=0.0
    )
    m.fit(real_data[:, :4096], max_iter=40, verbose=False)

    assert m.stop_reason == "lrate_floor"
    assert m.lrate <= m.minlrate
    assert len(m.ll_history) < 40


def test_grad_norm_shadows_grad_norm_floor_under_shipped_defaults(real_data):
    """Precedence, mirroring AMICATorchNG exactly (and Fortran's structure of
    independent ``leave = .true.`` assignments): the standalone grad_norm check
    runs last and tests a strict superset of the decrease-branch's condition, so
    with ``use_grad_norm`` left at its shipped True the reported reason is
    ``"grad_norm"``, never ``"grad_norm_floor"``. Same config as the test above
    with that single override removed."""
    m = _mlx(lrate=0.5, use_min_dll=False, min_nd=1.0)
    m.fit(real_data[:, :4096], max_iter=40, verbose=False)

    assert m.stop_reason == "grad_norm"


# --- regression: the new stops are inert at the shipped defaults -----------


def test_shipped_defaults_are_inert_on_a_100_iteration_fit(real_data):
    """The stops are on by default, so they must not silently change what an
    existing MLX run does. At the documented 100-iteration budget the shipped
    thresholds (min_dll=1e-9, min_nd=1e-7) are never met on this recording, so a
    default fit must produce the SAME trajectory, iteration for iteration, as
    one with both stops disabled -- bit-identical, which also proves computing
    ndtmpsum every iteration perturbs nothing."""
    default = _mlx()
    default.fit(real_data, max_iter=100, verbose=False)

    disabled = _mlx(use_min_dll=False, use_grad_norm=False)
    disabled.fit(real_data, max_iter=100, verbose=False)

    assert default.stop_reason == "max_iter"
    assert disabled.stop_reason == "max_iter"
    assert len(default.ll_history) == 100
    assert default.ll_history == disabled.ll_history
    assert default.final_ll_ == disabled.final_ll_
