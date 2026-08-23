"""Multi-model Newton on the NumPy backend (issue #267).

``_update_parameters`` finalized the Newton curvature by dividing the
``(data_dim, num_models)`` accumulators by ``dgm[:, None]``, i.e. a
``(num_models, 1)`` model mass. That broadcasts only when ``num_models == 1``,
so *every* multi-model Newton fit died with

    ValueError: operands could not be broadcast together with
                shapes (data_dim, num_models) (num_models, 1)

The issue reported it from a ``share_comps`` collapse, but the crash needs no
sharing at all -- it fires on the first iteration Newton is active with two
models. The fix broadcasts along the last axis (``dgm[None, :]``), matching the
PyTorch backend's ``dgm.unsqueeze(0)``.

Real bundled sample EEG only (``.rules/testing.md``).
"""

from pathlib import Path

import numpy as np
import pytest

from pamica import AMICA_NumPy as AMICA
from pamica.numpy_impl.data import load_data_file

_FDT = Path(__file__).resolve().parent.parent / "sample_data" / "eeglab_data.fdt"

pytestmark = pytest.mark.skipif(not _FDT.exists(), reason="sample data missing")

NW = 32
FIELD = 30504


def _real_data(n_samples: int = 4096) -> np.ndarray:
    data = load_data_file(str(_FDT), NW, FIELD, dtype=np.float32)
    return data[:, :n_samples].astype(np.float64)


@pytest.mark.parametrize("num_models", [1, 2, 3])
def test_multimodel_newton_fit_finalizes_curvature(num_models):
    """A short Newton fit completes for one, two and three models and leaves
    finite curvature shaped ``(data_dim, num_models)``.

    ``num_models=1`` is the case the old broadcast happened to survive, so it
    pins that the fix did not disturb it; 2 and 3 are the ones that raised.
    This is the numpy twin of ``torch_tests/test_ng_backend.py::
    test_newton_three_model_finite_and_shaped`` and
    ``mlx_tests/test_mlx_newton.py::test_three_model_newton_mstep_and_fit_are_finite``
    (issue #272): a non-degenerate stop and finite non-curvature parameters
    are asserted too, matching the depth of those two.
    """
    model = AMICA(
        num_models=num_models,
        num_mix=3,
        max_iter=4,
        seed=7,
        do_newton=True,
        newt_start=1,
        use_tqdm=False,
        do_opt_block=False,
        block_size=1024,
    )
    model.fit(_real_data())

    assert model.converged is True, f"degenerate fit: {model.stop_reason}"
    assert len(model.ll) == 4, "the fit did not complete all iterations"
    assert model.sigma2 is not None, "Newton was never active; the test is vacuous"
    assert model.lambda_ is not None and model.kappa is not None
    for name, arr in (
        ("sigma2", model.sigma2),
        ("lambda_", model.lambda_),
        ("kappa", model.kappa),
    ):
        assert arr.shape == (NW, num_models), f"{name} shape {arr.shape}"
        assert np.all(np.isfinite(arr)), f"{name} is not finite"
    # Curvature is a sum of squares over responsibilities: strictly positive.
    assert np.all(model.sigma2 > 0)
    assert np.all(model.kappa > 0)
