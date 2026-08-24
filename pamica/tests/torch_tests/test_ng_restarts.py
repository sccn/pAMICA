"""Best-of-N restarts on the PyTorch backend (issue #198).

Real sample EEG only. Two claims carry the feature:

* ``n_restarts=1`` (the default) is **bit-identical** to a fit that never heard
  of restarts -- the machinery must not perturb the parity path; and
* ``n_restarts=k`` returns exactly the argmax of the same *k* fits run
  independently, state for state, not merely a model with a similar likelihood.

Both are exact-equality assertions on real data, so they are machine-robust: no
timing, no tolerance, no dependence on which seed happens to win.
"""

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from pamica import AMICA, restarts
from pamica.torch_impl import AMICATorchNG

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "sample_data"
DATA_FILE = SAMPLE_DIR / "eeglab_data.fdt"
NW = 32
FIELD = 30504
NMIX = 3
MAX_ITER = 5
# Deliberately NOT ascending: with these seeds the winner is not the last
# restart, so the snapshot/restore path is what the acceptance test exercises.
SEEDS = [44, 43, 42]

pytestmark = pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing")


@pytest.fixture(scope="module")
def real_data() -> np.ndarray:
    from pamica.torch_impl.utils import load_eeglab_data

    data = load_eeglab_data(str(DATA_FILE), data_dim=NW, field_dim=FIELD)
    return data.astype(np.float64)[:, :4096]


def _model(**kwargs: Any) -> AMICATorchNG:
    params: dict[str, Any] = dict(n_channels=NW, n_mix=NMIX, device="cpu")
    params.update(kwargs)
    return AMICATorchNG(**params)


def _same(left: Any, right: Any) -> bool:
    """Exact equality for whatever the backend stores in a restart snapshot."""
    if left is None or right is None:
        return left is None and right is None
    if isinstance(left, torch.Tensor):
        return bool(torch.equal(left, right))
    if isinstance(left, np.ndarray):
        return np.array_equal(left, right)
    if isinstance(left, float) and math.isnan(left):
        return isinstance(right, float) and math.isnan(right)
    return bool(left == right)


def _assert_same_state(fitted: AMICATorchNG, reference: AMICATorchNG) -> None:
    for name in AMICATorchNG._RESTART_STATE_ATTRS:
        assert _same(getattr(fitted, name), getattr(reference, name)), name


# ---------------------------------------------------------------------------
# n_restarts=1 changes nothing
# ---------------------------------------------------------------------------


def test_single_restart_is_bit_identical_to_a_direct_fit(real_data):
    """The parity path. ``n_restarts=1`` must not draw, copy or reset anything:
    same parameters, same trajectory, same LLt, bit for bit."""
    baseline = _model(seed=42)
    baseline.fit(real_data, max_iter=MAX_ITER, verbose=False)
    explicit = _model(seed=42, n_restarts=1)
    explicit.fit(real_data, max_iter=MAX_ITER, verbose=False)

    _assert_same_state(explicit, baseline)
    assert explicit.ll_history == baseline.ll_history
    assert explicit.final_ll_ == baseline.final_ll_
    assert _same(explicit._llt_lt, baseline._llt_lt)
    assert _same(explicit._llt_lht, baseline._llt_lht)


def test_single_restart_records_the_one_fit_it_ran(real_data):
    model = _model(seed=42, n_restarts=1)
    model.fit(real_data, max_iter=MAX_ITER, verbose=False)
    assert model.restart_seeds_ == [42]
    assert model.restart_lls_ == [model.final_ll_]
    assert model.restart_stop_reasons_ == [model.stop_reason]


def test_an_explicit_single_seed_overrides_the_constructor_seed(real_data):
    """``restart_seeds=[s]`` with ``n_restarts=1`` is a deliberate override, so
    it must reproduce a plain fit from ``s`` -- not from the constructor seed."""
    overridden = _model(seed=42, n_restarts=1, restart_seeds=[7])
    overridden.fit(real_data, max_iter=MAX_ITER, verbose=False)
    direct = _model(seed=7)
    direct.fit(real_data, max_iter=MAX_ITER, verbose=False)
    _assert_same_state(overridden, direct)


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
    real_data, independent_fits, caplog
):
    """The acceptance test. Every attribute the fit path writes must match the
    winning seed's own single fit -- parameters, trajectory, LLt, counters and
    the tuned block size, not just the likelihood."""
    with caplog.at_level(logging.INFO, logger="pamica.torch_impl.core"):
        best_of_n = _model(seed=0, n_restarts=len(SEEDS), restart_seeds=SEEDS)
        best_of_n.fit(real_data, max_iter=MAX_ITER, verbose=False)

    expected = max(range(len(SEEDS)), key=lambda i: independent_fits[i].final_ll_)
    # The seeds are chosen so the winner is not simply the last restart, which
    # is the case where the live state would be right by accident.
    assert expected != len(SEEDS) - 1
    _assert_same_state(best_of_n, independent_fits[expected])
    assert best_of_n.ll_history == independent_fits[expected].ll_history
    assert best_of_n.final_ll_ == independent_fits[expected].final_ll_
    assert best_of_n.seed == SEEDS[expected]
    assert any(
        f"restart {expected + 1} (seed={SEEDS[expected]}) wins" in record.getMessage()
        for record in caplog.records
    ), "the winner must be named in one INFO line"


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
    """Without explicit seeds the restarts are seed, seed+1, ..."""
    model = _model(seed=42, n_restarts=3)
    model.fit(real_data, max_iter=MAX_ITER, verbose=False)
    assert model.restart_seeds_ == [42, 43, 44]


def test_a_refit_repeats_the_same_search(real_data):
    """``fit`` leaves ``self.seed`` on the winner, so a second fit must derive
    its seeds from the CONSTRUCTOR seed or the search would drift on every
    call."""
    model = _model(seed=42, n_restarts=3)
    model.fit(real_data, max_iter=MAX_ITER, verbose=False)
    first = list(model.restart_seeds_), list(model.restart_lls_)
    model.fit(real_data, max_iter=MAX_ITER, verbose=False)
    assert (list(model.restart_seeds_), list(model.restart_lls_)) == first


# ---------------------------------------------------------------------------
# Degenerate restarts
# ---------------------------------------------------------------------------


class _NaNForSeeds(AMICATorchNG):
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

    def _accumulate_blocks(self, X, stash_llt=False):
        acc = super()._accumulate_blocks(X, stash_llt=stash_llt)
        if self.seed in self.nan_seeds:
            acc["ll"] = acc["ll"] * float("nan")
        return acc


def test_a_degenerate_restart_is_recorded_but_never_selected(
    real_data, independent_fits
):
    """The first restart diverges; the winner must be the argmax of the two
    that did not, and the degenerate one must still appear in the records."""
    model = _NaNForSeeds(
        n_channels=NW, n_mix=NMIX, device="cpu", seed=0,
        n_restarts=len(SEEDS), restart_seeds=SEEDS,
    )  # fmt: skip
    model.nan_seeds = (SEEDS[0],)
    model.fit(real_data, max_iter=MAX_ITER, verbose=False)

    assert math.isnan(model.restart_lls_[0])
    assert model.restart_stop_reasons_[0] == "nan_ll"
    healthy = {1: independent_fits[1].final_ll_, 2: independent_fits[2].final_ll_}
    expected = max(healthy, key=lambda index: healthy[index])
    assert model.stop_reason == independent_fits[expected].stop_reason
    assert model.final_ll_ == healthy[expected]
    assert model.seed == SEEDS[expected]
    assert _same(model.A, independent_fits[expected].A)


def test_all_degenerate_restarts_keep_the_degenerate_contract(real_data, caplog):
    """A single NaN in the real EEG forces every restart to diverge (an
    error-path robustness test, not a correctness oracle -- the same route
    ``test_ng_rank_deficient.py`` uses). With nothing to select, the model is
    left holding the last restart so issue #50's contract applies unchanged, and
    every degenerate restart is still recorded."""
    bad = real_data[:, :4096].copy()
    bad[0, 0] = np.nan
    model = _model(seed=0, n_restarts=2, block_size=1024)
    with caplog.at_level(logging.WARNING, logger="pamica.torch_impl.core"):
        model.fit(bad, max_iter=3, verbose=False)

    assert model.stop_reason == "nan_ll"
    assert model.restart_seeds_ == [0, 1]
    assert all(math.isnan(ll) for ll in model.restart_lls_)
    assert model.restart_stop_reasons_ == ["nan_ll", "nan_ll"]
    assert any(
        "All 2 restarts ended degenerate" in r.getMessage() for r in caplog.records
    )
    with pytest.raises(RuntimeError, match="degenerate"):
        model.state_dict()


# ---------------------------------------------------------------------------
# Persistence and wrapper wiring
# ---------------------------------------------------------------------------


def test_state_dict_round_trips_the_configuration_and_the_records(real_data):
    model = _model(seed=0, n_restarts=len(SEEDS), restart_seeds=SEEDS)
    model.fit(real_data, max_iter=MAX_ITER, verbose=False)
    restored = AMICATorchNG.from_state_dict(model.state_dict(), device="cpu")

    assert restored.n_restarts == len(SEEDS)
    assert restored.restart_seeds == SEEDS
    assert restored._restart_seeds == SEEDS
    assert restored.restart_seeds_ == model.restart_seeds_
    assert restored.restart_lls_ == model.restart_lls_
    assert restored.restart_stop_reasons_ == model.restart_stop_reasons_
    assert _same(restored.A, model.A)


def test_a_payload_without_restart_records_still_loads(real_data):
    """Additive-only, like the issue #207 config keys: a model saved before
    #198 has no restart entries and must load with empty records."""
    model = _model(seed=42)
    model.fit(real_data, max_iter=MAX_ITER, verbose=False)
    payload = model.state_dict()
    for key in ("n_restarts", "restart_seeds"):
        payload["config"].pop(key)
    for key in restarts.RECORD_ATTRS:
        payload["extra"].pop(key)

    restored = AMICATorchNG.from_state_dict(payload, device="cpu")
    assert restored.n_restarts == 1
    assert restored.restart_seeds_ == []
    assert restored.restart_lls_ == []


def test_wrapper_forwards_n_restarts_and_mirrors_the_records(
    real_data, independent_fits
):
    """``AMICA.fit`` passes restart settings through to the backend and mirrors
    the records, so the wrapper is a full-fledged way to run best-of-N."""
    wrapper = AMICA(n_models=1, n_mix=NMIX, device="cpu", verbose=False)
    wrapper.fit(
        real_data,
        max_iter=MAX_ITER,
        lrate=0.1,
        n_restarts=len(SEEDS),
        restart_seeds=SEEDS,
        seed=0,
    )

    expected = max(range(len(SEEDS)), key=lambda i: independent_fits[i].final_ll_)
    backend = wrapper.model_
    assert backend is not None
    assert backend.n_restarts == len(SEEDS)
    assert wrapper.restart_seeds_ == SEEDS
    assert wrapper.restart_lls_ == [fit.final_ll_ for fit in independent_fits]
    assert wrapper.restart_stop_reasons_ == [
        fit.stop_reason for fit in independent_fits
    ]
    assert wrapper.final_ll_ == independent_fits[expected].final_ll_
    assert _same(backend.A, independent_fits[expected].A)
    assert wrapper.is_fitted_ is True


def test_wrapper_records_the_single_restart_of_a_default_fit(real_data):
    wrapper = AMICA(n_models=1, n_mix=NMIX, device="cpu", verbose=False)
    wrapper.fit(real_data, max_iter=MAX_ITER, lrate=0.1, seed=42)
    assert wrapper.restart_seeds_ == [42]
    assert wrapper.restart_lls_ == [wrapper.final_ll_]
