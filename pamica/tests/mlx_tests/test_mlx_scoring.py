"""``model_loglik``/``model_probability`` on the MLX backend -- issue #141,
epic #278 Phase 3/#289 (port of ``AMICATorchNG.model_loglik``/
``model_probability``, torch_impl/core.py:3041-3132).

Real sample EEG only (no synthetic/mock). Unlike the torch backend (float64
throughout), MLX's ``model_loglik`` recomputes the sphere-multiply in
float32 from raw data, while the training E-step's stash comes from
``_preprocess``'s float64-then-cast sphere -- two different float32
rounding paths through the same math, not a second copy of the same
computation. So the comparisons below are tolerance matches (measured
~1e-3 to ~5e-5 in practice), not the torch backend's bit-exact pin
(``test_ng_model_posterior.py``'s ``np.testing.assert_array_equal``).
"""

from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from pamica.mlx_impl import AMICAMLXNG  # noqa: E402  (after the MLX importorskip)

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "sample_data"
DATA_FILE = SAMPLE_DIR / "eeglab_data.fdt"
NW = 32
FIELD = 30504
NMIX = 3
BLOCK = 1024
SEED = 42

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
    )[:, :4096]


def _fit(X, n_models, max_iter=10, **kwargs) -> AMICAMLXNG:
    m = AMICAMLXNG(
        n_channels=NW,
        n_models=n_models,
        n_mix=NMIX,
        seed=SEED,
        block_size=BLOCK,
        **kwargs,
    )
    m.fit(X, max_iter=max_iter, verbose=False)
    return m


def test_model_loglik_reproduces_the_estep_forward_pass(real_data):
    """model_loglik(X) on the model's CURRENT (returned) parameters is the
    E-step's own per-model per-sample log-likelihood for those parameters:
    it must agree with a direct call to ``_forward`` on the same sphered
    data (the same computation the training loop runs internally, not the
    one-M-step-behind stash -- see
    test_model_loglik_is_the_estep_before_the_last_m_step below for that
    relationship)."""
    m = _fit(real_data, n_models=2, keep_best=False)
    lht = m.model_loglik(real_data)
    assert lht.shape == (2, real_data.shape[1])

    X_t = m.sphere @ (mx.array(real_data.astype(np.float32)) - m.mean)
    logV, *_ = m._forward(X_t)
    expected = np.array(logV).T
    np.testing.assert_allclose(lht, expected, rtol=0, atol=1e-4)


def test_model_loglik_is_the_estep_before_the_last_m_step(real_data):
    """Mirrors AMICATorchNG's issue #157 ordering pin
    (test_ng_model_posterior.py::test_model_loglik_matches_internal_lht):
    fit(max_iter=N-1).model_loglik(X) agrees with fit(max_iter=N)._llt_lht,
    and the M-step in between genuinely moves the per-sample LL."""
    m10 = _fit(real_data, n_models=2, max_iter=10, keep_best=False)
    m9 = _fit(real_data, n_models=2, max_iter=9, keep_best=False)
    assert len(m10.ll_history) == 10 and len(m9.ll_history) == 9
    assert m10._llt_lht is not None

    lht_pre = m9.model_loglik(real_data)
    np.testing.assert_allclose(lht_pre, m10._llt_lht, rtol=0, atol=2e-3)

    lht_post = m10.model_loglik(real_data)
    assert np.abs(lht_post - m10._llt_lht).max() > 1e-2


def test_model_probability_is_normalized(real_data):
    m = _fit(real_data, n_models=2)
    prob = m.model_probability(real_data)
    assert prob.shape == (2, real_data.shape[1])
    np.testing.assert_allclose(prob.sum(axis=0), 1.0, atol=1e-5)
    assert np.all(prob >= 0.0) and np.all(prob <= 1.0)


def test_single_model_probability_is_all_ones(real_data):
    m = _fit(real_data, n_models=1, max_iter=5)
    np.testing.assert_allclose(m.model_probability(real_data), 1.0, atol=1e-5)


def test_model_loglik_uses_stored_sphere_not_reprocess(real_data):
    """Scoring new data must not overwrite the fitted sphere/mean."""
    m = _fit(real_data, n_models=2)
    sphere_before = np.array(m.sphere).copy()
    mean_before = np.array(m.mean).copy()
    _ = m.model_loglik(real_data[:, :1000])  # a different-length slice
    np.testing.assert_array_equal(np.array(m.sphere), sphere_before)
    np.testing.assert_array_equal(np.array(m.mean), mean_before)


def test_model_loglik_requires_fit():
    m = AMICAMLXNG(n_channels=NW, n_models=2, n_mix=NMIX, seed=SEED)
    with pytest.raises(RuntimeError, match="fitted"):
        m.model_loglik(np.zeros((NW, 10)))


def test_model_loglik_rejects_non_finite_input(real_data):
    m = _fit(real_data, n_models=2, max_iter=5)
    bad = real_data.copy()
    bad[3, 100] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        m.model_loglik(bad)
    with pytest.raises(ValueError, match="non-finite"):
        m.model_probability(bad)
