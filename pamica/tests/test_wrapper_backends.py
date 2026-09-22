"""The ``AMICA`` wrapper end to end on every backend it can build (issue #313).

Real bundled sample EEG only (32 channels x 30504 frames), no synthetic data or
mocks (``.rules/testing.md``). Every fit is a few iterations: these are wiring,
persistence and agreement tests, not convergence tests.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from pamica import AMICA
from pamica.torch_impl.core import AMICATorchNG
from pamica.torch_impl.utils import load_eeglab_data

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_data"
DATA_FILE = SAMPLE_DIR / "eeglab_data.fdt"
NW = 32
FIELD = 30504
SEED = 42
N_FRAMES = 8192  # enough for a stable few-iteration fit, cheap on every backend
MAX_ITER = 3

pytestmark = pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing")


@pytest.fixture(scope="module")
def real_data() -> np.ndarray:
    return load_eeglab_data(str(DATA_FILE), data_dim=NW, field_dim=FIELD).astype(
        np.float64
    )


@pytest.fixture(scope="module")
def X(real_data) -> np.ndarray:
    return real_data[:, :N_FRAMES]


# --- pin: a format_version 1 save (written before issue #313) still loads -----
def _v1_payload(model: AMICA) -> dict:
    """The payload ``AMICA.save`` wrote before issue #313, key for key.

    Version 1 carried no backend name: the wrapper could only build
    ``AMICATorchNG``, so every version 1 file holds a torch state dict.
    """
    assert model.model_ is not None
    return {
        "format_version": 1,
        "wrapper": {
            "n_models": model.n_models,
            "n_mix": model.n_mix,
            "verbose": model.verbose,
        },
        "backend": model.model_.state_dict(),
    }


def test_v1_payload_still_loads_and_transforms_identically(X, tmp_path):
    model = AMICA(device="cpu", verbose=False)
    model.fit(X, max_iter=MAX_ITER, block_size=1024, seed=SEED)
    path = tmp_path / "v1.pt"
    torch.save(_v1_payload(model), path)

    loaded = AMICA.load(str(path), device="cpu")

    assert type(loaded.model_) is AMICATorchNG
    assert loaded.is_fitted_ and loaded.converged_
    assert loaded.stop_reason_ == model.stop_reason_
    assert loaded.final_ll_ == model.final_ll_
    assert loaded.ll_history_ == model.ll_history_
    np.testing.assert_array_equal(loaded.transform(X), model.transform(X))
