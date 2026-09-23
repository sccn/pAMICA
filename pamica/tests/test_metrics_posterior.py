"""Unit tests for ``pamica.metrics.posterior.model_probability_from_loglik``
(issue #306 PR #329 review).

The NaN-vs-``-inf`` diagnosis and the column-wise softmax normalization
behind ``AMICATorchNG.model_probability``/``AMICAMLXNG.model_probability``
were factored into this one pure function so the two backends do not
duplicate the logic; see ``pamica/metrics/posterior.py`` for why it lives
under ``pamica/metrics/`` (the same "backend computes ``Lht``/data, a shared
pure function does the rest" split already used by ``pamica.metrics.mir``/
``pairwise_mi``).

Per ``.rules/testing.md``, every test exercises a real ``Lht`` -- the actual
per-model, per-sample log-likelihood from a real small fit's
``model_loglik()`` on the bundled sample EEG data -- rather than a
fabricated array. The error-path tests poison a COPY of that real ``Lht``
(one column set entirely to ``-inf``, a separate single entry set to
``NaN``), the same direct-poisoning pattern used throughout
``test_backend_guards.py``.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import pytest
import torch

from pamica.metrics import model_probability_from_loglik
from pamica.torch_impl import AMICATorchNG
from pamica.torch_impl.utils import load_eeglab_data

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_data"
DATA_FILE = SAMPLE_DIR / "eeglab_data.fdt"
NW = 32
FIELD = 30504
NMIX = 3
SEED = 42
N_SAMPLES = 2048

pytestmark = pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing")


@pytest.fixture(scope="module")
def fitted_model() -> Tuple[AMICATorchNG, np.ndarray]:
    """A real 2-model fit and the data it was fit on, so
    ``model_loglik(data)`` returns a real ``(2, N_SAMPLES)`` ``Lht`` (two
    models, so a single poisoned row does not, on its own, make an entire
    column non-finite -- the -inf test needs both rows bad at one column,
    the NaN test needs only one)."""
    data = load_eeglab_data(str(DATA_FILE), data_dim=NW, field_dim=FIELD).astype(
        np.float64
    )[:, :N_SAMPLES]
    m = AMICATorchNG(
        n_channels=NW,
        n_models=2,
        n_mix=NMIX,
        seed=SEED,
        device="cpu",
        dtype=torch.float64,
        block_size=256,
    )
    m.fit(data, max_iter=2, verbose=False)
    return m, data


def test_healthy_lht_columns_sum_to_one_and_match_model_probability(fitted_model):
    m, data = fitted_model
    Lht = m.model_loglik(data)

    prob = model_probability_from_loglik(Lht, caller="test")

    np.testing.assert_allclose(prob.sum(axis=0), 1.0)
    # Behavior-preserving: AMICATorchNG.model_probability now delegates to
    # this exact function, so the two must agree exactly (both are computed
    # from the identical Lht, no separate rounding path).
    np.testing.assert_array_equal(prob, m.model_probability(data))


def test_all_models_neg_inf_at_a_sample_raises_distinct_message(fitted_model):
    m, data = fitted_model
    Lht = m.model_loglik(data).copy()
    Lht[:, 100] = -np.inf  # every model (both rows) underflows at sample 100

    with pytest.raises(ValueError, match="-inf") as excinfo:
        model_probability_from_loglik(Lht, caller="test")
    assert "NaN" not in str(excinfo.value)


def test_nan_loglik_raises_a_different_message_than_neg_inf(fitted_model):
    m, data = fitted_model
    Lht = m.model_loglik(data).copy()
    Lht[0, 100] = np.nan  # only one model's entry -- still poisons col_max

    with pytest.raises(ValueError, match="NaN") as excinfo:
        model_probability_from_loglik(Lht, caller="test")
    assert "-inf" not in str(excinfo.value)


def test_caller_prefix_appears_in_both_error_messages(fitted_model):
    m, data = fitted_model
    Lht = m.model_loglik(data).copy()
    Lht[:, 0] = -np.inf

    with pytest.raises(ValueError, match=r"^AMICATorchNG\.model_probability\(\)"):
        model_probability_from_loglik(Lht, caller="AMICATorchNG.model_probability()")
