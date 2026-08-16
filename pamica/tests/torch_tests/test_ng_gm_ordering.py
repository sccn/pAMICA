"""dAk is weighted by the pre-update model weights (issue #219), PyTorch side.

Fortran builds ``dAk`` in ``accum_updates_and_likelihood`` (amica15.f90:1753)
and only reassigns ``gm`` in ``update_params`` (:1788), so it weights by the
previous iteration's model weights.

This matters more here than in ``numpy_impl``. There ``dAk`` feeds only the
``nd`` diagnostic; here the same ``dAk`` is the A-update
(``self.A = self.A - self.lrate * dAk``), so the ordering reaches fitted
parameters. ``.rules/backend_parity.md`` requires the shared behavior to be
pinned on both backends; the NumPy half is in
``pamica/tests/test_numpy_grad_norm.py``.

The discriminator: hold one accumulator dict fixed across two M-steps so the
*post*-update gm is identical by construction, and vary only the gm present when
``dAk`` is built. Weighting by the post-update value makes the two runs
identical; weighting by the pre-update value does not.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from pamica.torch_impl.core import AMICATorchNG
from pamica.torch_impl.utils import load_eeglab_data

DATA_FILE = Path(__file__).resolve().parents[2] / "sample_data" / "eeglab_data.fdt"
NW = 32
FIELD = 30504

pytestmark = pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing")


@pytest.fixture(scope="module")
def real_data() -> np.ndarray:
    X = load_eeglab_data(str(DATA_FILE), data_dim=NW, field_dim=FIELD).astype(
        np.float64
    )
    return np.ascontiguousarray(X[:, :4096])


def _shared_two_model_fit(real_data: np.ndarray) -> AMICATorchNG:
    """A fitted 2-model backend whose comp_list genuinely shares a column.

    Forced directly: ``share_comps`` only reaches a shared column once a merge
    fires, which a short fit on this sample does not guarantee.
    """
    m = AMICATorchNG(n_channels=NW, n_models=2, seed=42, device="cpu")
    m.fit(real_data, max_iter=3, verbose=False)
    assert m.comp_list is not None
    m.comp_list[0, 1] = m.comp_list[0, 0]
    return m


def _step_with_pre_update_gm(model, acc, n_samples, gm_pre):
    """One M-step with ``gm`` preset to ``gm_pre``; returns (ndtmpsum, A)."""
    saved = {
        name: getattr(model, name).clone()
        for name in ("A", "W", "mu", "beta", "rho", "alpha", "gm", "c")
        if getattr(model, name, None) is not None
    }
    model.gm = torch.tensor(gm_pre, dtype=model.dtype, device=model.device)
    model._update_parameters(acc, n_samples)
    nd = model._ndtmpsum
    assert nd is not None
    result = (float(nd), model.A.clone())
    for name, value in saved.items():
        setattr(model, name, value)
    return result


def _accumulate(model, real_data):
    X = model._preprocess(real_data)
    return model._get_block_updates(X), X.shape[1]


def test_ndtmpsum_uses_pre_update_model_weights(real_data):
    """Opposite pre-update weights must give different nd, with acc held fixed."""
    model = _shared_two_model_fit(real_data)
    acc, n = _accumulate(model, real_data)

    nd_a, _ = _step_with_pre_update_gm(model, acc, n, [0.9, 0.1])
    nd_b, _ = _step_with_pre_update_gm(model, acc, n, [0.1, 0.9])

    assert np.isfinite(nd_a) and np.isfinite(nd_b)
    assert nd_a != nd_b, (
        "nd is identical under opposite pre-update model weights, so dAk is "
        "being built from the post-update gm"
    )


def test_a_update_uses_pre_update_model_weights(real_data):
    """The consequence unique to this backend: the fitted A moves with it.

    ``dAk`` is the A-update here, so the ordering is not diagnostic-only.
    """
    model = _shared_two_model_fit(real_data)
    acc, n = _accumulate(model, real_data)

    _, a_first = _step_with_pre_update_gm(model, acc, n, [0.9, 0.1])
    _, a_second = _step_with_pre_update_gm(model, acc, n, [0.1, 0.9])

    assert torch.isfinite(a_first).all() and torch.isfinite(a_second).all()
    assert not torch.equal(a_first, a_second), (
        "A is unchanged under opposite pre-update model weights, so the "
        "A-update is being built from the post-update gm"
    )


def test_same_pre_update_gm_is_deterministic(real_data):
    """The complement, so the tests above cannot pass on unrelated sensitivity."""
    model = _shared_two_model_fit(real_data)
    acc, n = _accumulate(model, real_data)

    nd_first, a_first = _step_with_pre_update_gm(model, acc, n, [0.7, 0.3])
    nd_second, a_second = _step_with_pre_update_gm(model, acc, n, [0.7, 0.3])

    assert nd_first == nd_second
    assert torch.equal(a_first, a_second)


def test_single_model_is_unaffected(real_data):
    """gm == 1 for one model, so the ordering cannot be observable there.

    This is why single-model parity (issue #24) stays bit-exact across the
    change.
    """
    model = AMICATorchNG(n_channels=NW, n_models=1, seed=42, device="cpu")
    model.fit(real_data, max_iter=3, verbose=False)
    acc, n = _accumulate(model, real_data)

    nd_a, a_a = _step_with_pre_update_gm(model, acc, n, [1.0])
    nd_b, a_b = _step_with_pre_update_gm(model, acc, n, [1.0])
    assert nd_a == nd_b
    assert torch.equal(a_a, a_b)
