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
from pamica.torch_impl import core

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "sample_data"
DATA_FILE = SAMPLE_DIR / "eeglab_data.fdt"
NW = 32
FIELD = 30504
NMIX = 3
MAX_ITER = 5
# Ordered best-to-worst, so the winner is the FIRST restart: the snapshot/restore
# path is then what the acceptance test exercises, rather than the live state
# being right by accident. The seeds are chosen for separation, not convenience:
# measured on this fixture at MAX_ITER, the final LLs are
# -3.264177210 / -3.264757966 / -3.264984217, i.e. the winner leads the
# runner-up by 5.8e-4 and the last restart by 8.1e-4. That is ~8 orders of
# magnitude above float64 BLAS summation noise (~1e-12 relative on an LL of
# ~3.26), so re-ordering the reductions -- OpenBLAS vs Accelerate, one thread vs
# many -- cannot change which restart wins (the issue #241 flake class). The
# NumPy and MLX suites use the same seeds: NumPy agrees to 1e-9 and MLX
# (float32) to 6e-7, both far inside the same margin.
SEEDS = [54, 50, 42]

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
    assert expected != len(SEEDS) - 1, (
        "precondition: the winning seed must not be the last restart, or the "
        "restore path goes untested. The margin is 5.8e-4 (see SEEDS), far above "
        "float64 summation noise, so a failure here is a platform-BLAS ordering "
        "finding worth reporting -- not a correctness bug in the restart loop."
    )
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


class _RaiseForSeeds(AMICATorchNG):
    """Raises ``torch.linalg.LinAlgError`` for the listed seeds only.

    The failure mode a non-finite likelihood does NOT cover: a truly singular
    ``A`` makes ``torch.linalg.inv`` raise rather than return infinities, so the
    fit never reaches the in-loop guard. Injected at the same hook the real
    failure comes from (``_update_unmixing_matrices`` -> ``torch.linalg.inv``)
    and with the same exception type torch itself raises, so what is under test
    is the restart loop's isolation, not the injection.
    """

    raise_seeds: tuple = ()

    def _update_unmixing_matrices(self):
        if self.seed in self.raise_seeds:
            raise torch.linalg.LinAlgError(
                "linalg.inv: The diagonal element 3 is zero, the inversion "
                "could not be completed because the input matrix is singular."
            )
        return super()._update_unmixing_matrices()


def test_a_crashing_restart_does_not_kill_the_search(
    real_data, independent_fits, caplog
):
    """Isolation (issue #198 review). A restart that RAISES must not discard the
    restarts that already succeeded -- that would be the exact opposite of what
    best-of-N is for. It is recorded as a degenerate ``restart_error`` and the
    search continues to the next seed."""
    model = _RaiseForSeeds(
        n_channels=NW, n_mix=NMIX, device="cpu", seed=0,
        n_restarts=len(SEEDS), restart_seeds=SEEDS,
    )  # fmt: skip
    # The BEST seed crashes, so the winner has to come from the survivors: a
    # loop that silently kept the crashed restart's state would fail here.
    model.raise_seeds = (SEEDS[0],)
    with caplog.at_level(logging.WARNING, logger="pamica.torch_impl.core"):
        model.fit(real_data, max_iter=MAX_ITER, verbose=False)

    assert model.restart_stop_reasons_[0] == restarts.ERROR_STOP_REASON
    assert math.isnan(model.restart_lls_[0])
    assert restarts.ERROR_STOP_REASON in AMICATorchNG._DEGENERATE_STOP_REASONS
    healthy = {1: independent_fits[1].final_ll_, 2: independent_fits[2].final_ll_}
    expected = max(healthy, key=lambda index: healthy[index])
    assert model.final_ll_ == healthy[expected]
    assert model.seed == SEEDS[expected]
    _assert_same_state(model, independent_fits[expected])
    assert any("LinAlgError" in r.getMessage() for r in caplog.records), (
        "the caught exception must stay diagnosable in the log"
    )


def test_a_single_restart_still_raises(real_data):
    """The other half of the isolation contract: bit-identity with a pre-#198
    fit includes ERROR behavior, so the single-restart path must NOT catch."""
    model = _RaiseForSeeds(n_channels=NW, n_mix=NMIX, device="cpu", seed=SEEDS[0])
    model.raise_seeds = (SEEDS[0],)
    with pytest.raises(torch.linalg.LinAlgError):
        model.fit(real_data, max_iter=MAX_ITER, verbose=False)


def test_all_restarts_crashing_keeps_the_degenerate_contract(real_data, caplog):
    """Every restart raises: the model is unusable and says so, rather than the
    exception escaping (which would lose the records) or the fit reporting
    success on whatever partial state the crash left."""
    model = _RaiseForSeeds(
        n_channels=NW, n_mix=NMIX, device="cpu", seed=0,
        n_restarts=len(SEEDS), restart_seeds=SEEDS,
    )  # fmt: skip
    model.raise_seeds = tuple(SEEDS)
    with caplog.at_level(logging.WARNING, logger="pamica.torch_impl.core"):
        model.fit(real_data, max_iter=MAX_ITER, verbose=False)

    assert model.stop_reason == restarts.ERROR_STOP_REASON
    assert model.restart_seeds_ == SEEDS
    assert all(math.isnan(ll) for ll in model.restart_lls_)
    assert model.restart_stop_reasons_ == [restarts.ERROR_STOP_REASON] * len(SEEDS)
    assert any(
        "All 3 restarts ended degenerate" in r.getMessage() for r in caplog.records
    )
    with pytest.raises(RuntimeError, match="degenerate"):
        model.state_dict()
    # No per-sample likelihoods survive from a fit that never completed.
    assert model._llt_lht is None and model._llt_lt is None


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
# Interaction with the block-size search (issue #232)
# ---------------------------------------------------------------------------


def test_the_winners_tuned_block_size_is_restored(real_data, monkeypatch):
    """``do_opt_block`` re-times the block size inside every restart, so the
    tuned value is per-restart state and the winner's must survive the restore.

    The search's RETURN VALUE is stubbed rather than its timings: which size
    wins a real sweep is documented as machine-dependent (``blocktune``), so
    pinning it deterministically is the only way to assert this at all. The
    stub hands restart 1 a different size from restart 2, and restart 1 is the
    winner, so a loop that forgot ``block_size`` would end on 8192.
    """
    sizes = iter([4096, 8192, 16384])

    def fixed_search(**kwargs):
        return next(sizes)

    monkeypatch.setattr(core.blocktune, "search", fixed_search)
    model = _model(
        seed=0, n_restarts=len(SEEDS), restart_seeds=SEEDS, do_opt_block=True
    )
    model.fit(real_data, max_iter=MAX_ITER, verbose=False)

    expected = max(range(len(SEEDS)), key=lambda i: model.restart_lls_[i])
    assert expected == 0, "SEEDS puts the winner first; see the SEEDS comment"
    assert model.block_size == 4096


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


def test_wrapper_refuses_a_best_of_n_fit_whose_restarts_all_degenerate(
    real_data, tmp_path, caplog
):
    """Best-of-N must not weaken issue #50: when no restart produced a usable
    model, the wrapper marks the fit unusable and every output method refuses
    it, exactly as it does for a single degenerate fit -- with the per-restart
    records still readable for diagnosis."""
    bad = real_data[:, :4096].copy()
    bad[0, 0] = np.nan  # degenerates every seed alike (see the backend test)
    wrapper = AMICA(n_models=1, n_mix=NMIX, device="cpu", verbose=False)
    with caplog.at_level(logging.WARNING, logger="pamica.amica"):
        wrapper.fit(bad, max_iter=3, lrate=0.1, block_size=1024, seed=0, n_restarts=2)

    assert wrapper.converged_ is False
    assert wrapper.is_fitted_ is False
    assert wrapper.stop_reason_ == "nan_ll"
    assert wrapper.restart_seeds_ == [0, 1]
    assert all(math.isnan(ll) for ll in wrapper.restart_lls_)
    assert any("degenerate" in r.getMessage() for r in caplog.records)
    for action in (
        lambda: wrapper.transform(real_data[:, :512]),
        lambda: wrapper.get_mixing_matrix(),
        lambda: wrapper.save(str(tmp_path / "degenerate.pt")),
    ):
        with pytest.raises(RuntimeError, match="degenerate"):
            action()
