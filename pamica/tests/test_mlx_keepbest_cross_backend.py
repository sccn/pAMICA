"""``keep_best`` best-iterate safeguard: PyTorch vs MLX CONTRACT parity
(issue #288, epic #278 Phase 2).

Cross-backend by design, so this lives in ``pamica/tests/`` rather than
``pamica/tests/mlx_tests/`` (``.rules/backend_parity.md``).

Unlike the arithmetic-agreement cross-backend suites (``test_mlx_newton_
cross_backend.py`` etc.), this one is NOT about two backends agreeing on
numbers from identical starting parameters: the restore DECISION depends on
each backend's own float trajectory (float64 natural-gradient on PyTorch,
float32 on MLX), and there is no reason those trajectories should overshoot
on the same iteration, or at all, for the same config -- so this module
deliberately does not assert decision equality on a shared config. What both
backends must agree on is the CONTRACT the safeguard promises: whenever
``keep_best`` is on and active (not disabled by ``share_comps``) and the fit
did not end degenerate, the returned ``final_ll_`` equals ``max(ll_history)``
-- true whether or not that particular run happened to overshoot its own
peak, and true independent of precision. See ``pamica/tests/mlx_tests/
test_mlx_keepbest.py`` for MLX's own overshoot-forcing recipes and
``pamica/tests/torch_tests/test_ng_backend.py``/``test_ng_convergence.py``
for PyTorch's.

Real bundled sample EEG only, no synthetic data or mocks (``.rules/testing.md``).
MLX is an optional Apple-Silicon backend, so the module self-skips via
``importorskip`` plus an Apple-GPU guard; PyTorch always runs.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from pamica.torch_impl.core import AMICATorchNG

mx = pytest.importorskip("mlx.core", reason="MLX not installed (Apple Silicon only)")
mlx_core = pytest.importorskip(
    "pamica.mlx_impl.core", reason="MLX not installed (Apple Silicon only)"
)
AMICAMLXNG = mlx_core.AMICAMLXNG

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_data"
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

    data = load_eeglab_data(str(DATA_FILE), data_dim=NW, field_dim=FIELD).astype(
        np.float64
    )
    return data[:, :4096]


# The same aggressive-Newton config on both sides: known to genuinely
# overshoot on PyTorch (test_ng_convergence.py's forcing recipe) and,
# independently verified, on MLX too (test_mlx_keepbest.py's module
# docstring) -- but this module does not rely on that coincidence holding
# forever, only on the CONTRACT below, which holds regardless.
_AGGRESSIVE_KWARGS: dict[str, Any] = dict(
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


def test_torch_keep_best_satisfies_the_max_contract(real_data):
    ng = AMICATorchNG(
        n_channels=NW,
        device="cpu",
        dtype=torch.float64,
        keep_best=True,
        **_AGGRESSIVE_KWARGS,
    )
    ng.fit(real_data, max_iter=60, verbose=False)
    assert ng.stop_reason not in ng._DEGENERATE_STOP_REASONS
    assert ng.final_ll_ == max(ng.ll_history)


def test_mlx_keep_best_satisfies_the_max_contract(real_data):
    m = AMICAMLXNG(n_channels=NW, keep_best=True, **_AGGRESSIVE_KWARGS)
    m.fit(real_data, max_iter=60, verbose=False)
    assert m.stop_reason not in m._DEGENERATE_STOP_REASONS
    assert m.final_ll_ == max(m.ll_history)


def test_both_backends_disable_the_safeguard_under_share_comps(real_data):
    """The ``share_comps`` exclusion is the one piece of the ``track_best``
    condition both backends currently share (MLX has no ``do_reject`` yet,
    epic #278 Phase 3) -- so it is the one decision this module CAN assert
    identically: on both backends, ``final_ll_`` is exactly ``ll_history[-1]``
    when sharing is on, never a restored earlier peak."""
    kwargs: dict[str, Any] = dict(_AGGRESSIVE_KWARGS, share_comps=True)

    ng = AMICATorchNG(
        n_channels=NW, device="cpu", dtype=torch.float64, keep_best=True, **kwargs
    )
    ng.fit(real_data, max_iter=60, verbose=False)
    assert ng.stop_reason not in ng._DEGENERATE_STOP_REASONS
    assert ng.final_ll_ == ng.ll_history[-1]

    m = AMICAMLXNG(n_channels=NW, keep_best=True, **kwargs)
    m.fit(real_data, max_iter=60, verbose=False)
    assert m.stop_reason not in m._DEGENERATE_STOP_REASONS
    assert m.final_ll_ == m.ll_history[-1]
