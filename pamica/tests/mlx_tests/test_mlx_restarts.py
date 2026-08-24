"""Best-of-N restarts on the MLX backend (issue #198).

Apple-Silicon only; the module self-skips without MLX or an Apple GPU. Real
sample EEG, float32 (Apple GPUs have no float64), and it runs in macOS CI.

The same two claims the other backends pin: ``n_restarts=1`` is bit-identical
to a fit that never heard of restarts, and ``n_restarts=k`` returns exactly the
argmax of the same *k* fits run independently, state for state. The
MLX-specific risk is the snapshot itself: MLX arrays support item assignment
and its graph is lazy, so a snapshot that aliased a live array (or captured an
unevaluated one) would silently roll forward with the next restart.
"""

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from pamica.mlx_impl import AMICAMLXNG  # noqa: E402

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "sample_data"
DATA_FILE = SAMPLE_DIR / "eeglab_data.fdt"
NW = 32
FIELD = 30504
NMIX = 3
MAX_ITER = 5
# Deliberately NOT ascending: the winner is then not the last restart, so the
# snapshot/restore path is what the acceptance test exercises.
SEEDS = [44, 43, 42]

pytestmark = [
    pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing"),
    pytest.mark.skipif(
        mx.default_device().type != mx.DeviceType.gpu, reason="no Apple GPU"
    ),
]


@pytest.fixture(scope="module")
def real_data() -> np.ndarray:
    from pamica.torch_impl.utils import load_eeglab_data

    data = load_eeglab_data(str(DATA_FILE), data_dim=NW, field_dim=FIELD)
    return data.astype(np.float64)[:, :4096]


def _model(**kwargs: Any) -> AMICAMLXNG:
    params: dict[str, Any] = dict(n_channels=NW, n_mix=NMIX)
    params.update(kwargs)
    return AMICAMLXNG(**params)


def _same(left: Any, right: Any) -> bool:
    """Exact equality for whatever the backend stores in a restart snapshot."""
    if left is None or right is None:
        return left is None and right is None
    if isinstance(left, mx.array):
        return np.array_equal(np.array(left), np.array(right))
    if isinstance(left, np.ndarray):
        return np.array_equal(left, right)
    if isinstance(left, float) and math.isnan(left):
        return isinstance(right, float) and math.isnan(right)
    return bool(left == right)


def _assert_same_state(fitted: AMICAMLXNG, reference: AMICAMLXNG) -> None:
    for name in AMICAMLXNG._RESTART_STATE_ATTRS:
        assert _same(getattr(fitted, name), getattr(reference, name)), name


# ---------------------------------------------------------------------------
# n_restarts=1 changes nothing
# ---------------------------------------------------------------------------


def test_single_restart_is_bit_identical_to_a_direct_fit(real_data):
    baseline = _model(seed=42)
    baseline.fit(real_data, max_iter=MAX_ITER, verbose=False)
    explicit = _model(seed=42, n_restarts=1)
    explicit.fit(real_data, max_iter=MAX_ITER, verbose=False)

    _assert_same_state(explicit, baseline)
    assert explicit.ll_history == baseline.ll_history
    assert explicit.final_ll_ == baseline.final_ll_


def test_single_restart_records_the_one_fit_it_ran(real_data):
    model = _model(seed=42, n_restarts=1)
    model.fit(real_data, max_iter=MAX_ITER, verbose=False)
    assert model.restart_seeds_ == [42]
    assert model.restart_lls_ == [model.final_ll_]
    assert model.restart_stop_reasons_ == [model.stop_reason]


# ---------------------------------------------------------------------------
# Acceptance: best-of-N is the argmax of the same fits run independently
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def independent_fits(real_data) -> list:
    fits = []
    for seed in SEEDS:
        model = _model(seed=seed)
        model.fit(real_data, max_iter=MAX_ITER, verbose=False)
        fits.append(model)
    return fits


def test_best_of_n_returns_the_argmax_of_the_independent_fits(
    real_data, independent_fits
):
    best_of_n = _model(seed=0, n_restarts=len(SEEDS), restart_seeds=SEEDS)
    best_of_n.fit(real_data, max_iter=MAX_ITER, verbose=False)

    expected = max(range(len(SEEDS)), key=lambda i: independent_fits[i].final_ll_)
    assert expected != len(SEEDS) - 1
    _assert_same_state(best_of_n, independent_fits[expected])
    assert best_of_n.ll_history == independent_fits[expected].ll_history
    assert best_of_n.seed == SEEDS[expected]


def test_records_are_aligned_with_the_independent_fits(real_data, independent_fits):
    best_of_n = _model(seed=0, n_restarts=len(SEEDS), restart_seeds=SEEDS)
    best_of_n.fit(real_data, max_iter=MAX_ITER, verbose=False)
    assert best_of_n.restart_seeds_ == SEEDS
    assert best_of_n.restart_lls_ == [fit.final_ll_ for fit in independent_fits]
    assert best_of_n.restart_stop_reasons_ == [
        fit.stop_reason for fit in independent_fits
    ]
    assert best_of_n.final_ll_ == max(best_of_n.restart_lls_)


def test_derived_seeds_run_the_documented_sequence(real_data):
    model = _model(seed=42, n_restarts=3)
    model.fit(real_data, max_iter=MAX_ITER, verbose=False)
    assert model.restart_seeds_ == [42, 43, 44]


def test_a_refit_repeats_the_same_search(real_data):
    model = _model(seed=42, n_restarts=3)
    model.fit(real_data, max_iter=MAX_ITER, verbose=False)
    first = list(model.restart_seeds_), list(model.restart_lls_)
    model.fit(real_data, max_iter=MAX_ITER, verbose=False)
    assert (list(model.restart_seeds_), list(model.restart_lls_)) == first


# ---------------------------------------------------------------------------
# Degenerate restarts
# ---------------------------------------------------------------------------


class _NaNForSeeds(AMICAMLXNG):
    """Forces a ``nan_ll`` stop for the listed seeds only.

    Error-path injection through the real fit, the pattern
    ``test_sample_data.py::test_restart_gives_up_after_maxrestarts`` already
    uses on the NumPy backend: one value is corrupted at one hook and every
    other seed runs the untouched algorithm on the real recording. It is the
    only way to get a *mixed* set of restarts -- NaN in the data degenerates
    every seed alike, and divergence from a plain hyperparameter is a
    machine-dependent outcome no CI can rely on.
    """

    nan_seeds: tuple = ()

    def _accumulate_blocks(self, X):
        acc = super()._accumulate_blocks(X)
        if self.seed in self.nan_seeds:
            acc["ll"] = acc["ll"] * float("nan")
        return acc


def _nan_model(nan_seeds, **kwargs: Any) -> AMICAMLXNG:
    model = _NaNForSeeds(n_channels=NW, n_mix=NMIX, **kwargs)
    model.nan_seeds = tuple(nan_seeds)
    return model


def test_a_degenerate_restart_is_recorded_but_never_selected(
    real_data, independent_fits
):
    model = _nan_model((SEEDS[0],), seed=0, n_restarts=len(SEEDS), restart_seeds=SEEDS)
    model.fit(real_data, max_iter=MAX_ITER, verbose=False)

    assert math.isnan(model.restart_lls_[0])
    assert model.restart_stop_reasons_[0] == "nan_ll"
    healthy = {1: independent_fits[1].final_ll_, 2: independent_fits[2].final_ll_}
    expected = max(healthy, key=lambda index: healthy[index])
    assert model.final_ll_ == healthy[expected]
    assert model.seed == SEEDS[expected]
    assert _same(model.A, independent_fits[expected].A)


def test_all_degenerate_restarts_keep_the_last_one(real_data):
    """With nothing to select the model is left holding the last restart, so a
    degenerate best-of-N fit reports exactly what a degenerate single fit
    does."""
    model = _nan_model(tuple(SEEDS), seed=0, n_restarts=len(SEEDS), restart_seeds=SEEDS)
    model.fit(real_data, max_iter=MAX_ITER, verbose=False)

    assert model.stop_reason == "nan_ll"
    final_ll = model.final_ll_
    assert final_ll is not None and math.isnan(final_ll)
    assert model.restart_seeds_ == SEEDS
    assert all(math.isnan(ll) for ll in model.restart_lls_)
    assert model.seed == SEEDS[-1]
