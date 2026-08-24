"""Best-of-N restarts on the NumPy backend (issue #198).

Real sample EEG only, and the same two claims the PyTorch suite pins:
``n_restarts=1`` is bit-identical to a fit that never heard of restarts, and
``n_restarts=k`` returns exactly the argmax of the same *k* fits run
independently -- state for state.

This backend is where restart isolation has teeth. Its ``_initialize_parameters``
only draws a parameter that is still ``None``, and ``_check_convergence``
ratchets ``lrate0``/``newtrate`` in place with nothing restoring them, so
without ``_reset_for_restart`` restart *k+1* would continue from restart *k*'s
solution and annealed schedule. The equality against independent fits is what
proves it does not.
"""

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pamica import AMICA_NumPy
from pamica.numpy_impl.data import load_data_file

_FDT = Path(__file__).resolve().parent.parent / "sample_data" / "eeglab_data.fdt"
NW = 32
FIELD = 30504
MAX_ITER = 4
# Deliberately NOT ascending: the winner is then not the last restart, so the
# snapshot/restore path is what the acceptance test exercises.
SEEDS = [44, 43, 42]

pytestmark = pytest.mark.skipif(not _FDT.exists(), reason="sample data missing")


@pytest.fixture(scope="module")
def real_data() -> np.ndarray:
    data = load_data_file(str(_FDT), NW, FIELD, dtype=np.float32)
    return data[:, :4096].astype(np.float64)


@pytest.fixture(scope="module")
def outdir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("numpy_restarts")


def _model(outdir: Path, **kwargs: Any) -> AMICA_NumPy:
    params: dict[str, Any] = dict(
        num_models=1,
        num_mix=3,
        max_iter=MAX_ITER,
        use_tqdm=False,
        outdir=str(outdir),
    )
    params.update(kwargs)
    return AMICA_NumPy(**params)


def _same(left: Any, right: Any) -> bool:
    """Exact equality for whatever the backend stores in a restart snapshot."""
    if isinstance(left, np.random.RandomState):
        left_state, right_state = left.get_state(), right.get_state()
        return left_state[0] == right_state[0] and np.array_equal(
            left_state[1], right_state[1]
        )
    if left is None or right is None:
        return left is None and right is None
    if isinstance(left, np.ndarray):
        return np.array_equal(left, right)
    if isinstance(left, float) and math.isnan(left):
        return isinstance(right, float) and math.isnan(right)
    return bool(left == right)


def _assert_same_state(fitted: AMICA_NumPy, reference: AMICA_NumPy) -> None:
    for name in AMICA_NumPy._RESTART_STATE_ATTRS:
        assert _same(getattr(fitted, name), getattr(reference, name)), name


# ---------------------------------------------------------------------------
# n_restarts=1 changes nothing
# ---------------------------------------------------------------------------


def test_single_restart_is_bit_identical_to_a_direct_fit(real_data, outdir):
    baseline = _model(outdir, seed=42)
    baseline.fit(real_data)
    explicit = _model(outdir, seed=42, n_restarts=1)
    explicit.fit(real_data)

    _assert_same_state(explicit, baseline)
    assert explicit.ll == baseline.ll
    assert explicit.nd == baseline.nd


def test_single_restart_records_the_one_fit_it_ran(real_data, outdir):
    model = _model(outdir, seed=42, n_restarts=1)
    model.fit(real_data)
    assert model.restart_seeds_ == [42]
    assert model.restart_lls_ == [model.ll[-1]]
    assert model.restart_stop_reasons_ == [model.stop_reason]


def test_an_explicit_single_seed_overrides_the_constructor_seed(real_data, outdir):
    overridden = _model(outdir, seed=42, n_restarts=1, restart_seeds=[7])
    overridden.fit(real_data)
    direct = _model(outdir, seed=7)
    direct.fit(real_data)
    _assert_same_state(overridden, direct)


# ---------------------------------------------------------------------------
# Acceptance: best-of-N is the argmax of the same fits run independently
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def independent_fits(real_data, outdir) -> list:
    fits = []
    for seed in SEEDS:
        model = _model(outdir, seed=seed)
        model.fit(real_data)
        fits.append(model)
    return fits


def test_best_of_n_returns_the_argmax_of_the_independent_fits(
    real_data, outdir, independent_fits
):
    best_of_n = _model(outdir, seed=0, n_restarts=len(SEEDS), restart_seeds=SEEDS)
    best_of_n.fit(real_data)

    expected = max(range(len(SEEDS)), key=lambda i: independent_fits[i].ll[-1])
    assert expected != len(SEEDS) - 1
    _assert_same_state(best_of_n, independent_fits[expected])
    assert best_of_n.ll == independent_fits[expected].ll
    assert best_of_n.seed == SEEDS[expected]


def test_records_are_aligned_with_the_independent_fits(
    real_data, outdir, independent_fits
):
    best_of_n = _model(outdir, seed=0, n_restarts=len(SEEDS), restart_seeds=SEEDS)
    best_of_n.fit(real_data)
    assert best_of_n.restart_seeds_ == SEEDS
    assert best_of_n.restart_lls_ == [fit.ll[-1] for fit in independent_fits]
    assert best_of_n.restart_stop_reasons_ == [
        fit.stop_reason for fit in independent_fits
    ]
    assert best_of_n.ll[-1] == max(best_of_n.restart_lls_)


def test_derived_seeds_run_the_documented_sequence(real_data, outdir):
    model = _model(outdir, seed=42, n_restarts=3)
    model.fit(real_data)
    assert model.restart_seeds_ == [42, 43, 44]


def test_a_refit_repeats_the_same_search(real_data, outdir):
    model = _model(outdir, seed=42, n_restarts=3)
    model.fit(real_data)
    first = list(model.restart_seeds_), list(model.restart_lls_)
    model.fit(real_data)
    assert (list(model.restart_seeds_), list(model.restart_lls_)) == first


def test_a_restart_does_not_inherit_the_previous_restart_schedule(real_data, outdir):
    """The concrete leak ``_reset_for_restart`` closes: a fit ratchets
    ``lrate``/``lrate0`` in place, so a second restart that inherited them would
    run a different schedule from the independent fit it is compared against."""
    model = _model(outdir, seed=0, n_restarts=len(SEEDS), restart_seeds=SEEDS)
    model.fit(real_data)
    winner = max(range(len(SEEDS)), key=lambda i: model.restart_lls_[i])
    reference = _model(outdir, seed=SEEDS[winner])
    reference.fit(real_data)
    assert model.lrate == reference.lrate
    assert model.lrate0 == reference.lrate0
    assert model.newtrate == reference.newtrate
    assert model.rholrate == reference.rholrate


# ---------------------------------------------------------------------------
# Degenerate restarts
# ---------------------------------------------------------------------------


class _NaNForSeeds(AMICA_NumPy):
    """Forces a non-finite likelihood for the listed seeds only.

    Error-path injection through the real fit, the pattern
    ``test_sample_data.py::test_restart_gives_up_after_maxrestarts`` already
    uses: one value is corrupted at one hook, every other seed runs the
    untouched algorithm on the real recording. It is the only way to get a
    *mixed* set of restarts on this backend -- NaN in the data raises in
    preprocessing (before any fit), and divergence from a plain
    hyperparameter is a machine-dependent outcome no CI can rely on.
    """

    nan_seeds: tuple = ()

    def _get_updates_and_likelihood(self):
        updates = super()._get_updates_and_likelihood()
        if self.seed in self.nan_seeds:
            updates["ll"] = float("nan")
        return updates


def _nan_model(outdir: Path, nan_seeds, **kwargs: Any) -> AMICA_NumPy:
    model = _NaNForSeeds(
        num_models=1,
        num_mix=3,
        max_iter=MAX_ITER,
        use_tqdm=False,
        outdir=str(outdir),
        # Fortran's restart-on-NaN recovery would redraw A and mask the
        # injection; this test is about the best-of-N loop, not that path.
        restartiter=0,
        maxrestarts=0,
        writestep=10_000_000,
        **kwargs,
    )
    model.nan_seeds = tuple(nan_seeds)
    return model


def test_a_degenerate_restart_is_recorded_but_never_selected(
    real_data, tmp_path, independent_fits
):
    """The first restart diverges; the winner must be the argmax of the two
    that did not, and the degenerate one must still appear in the records."""
    model = _nan_model(
        tmp_path,
        nan_seeds=(SEEDS[0],),
        seed=0,
        n_restarts=len(SEEDS),
        restart_seeds=SEEDS,
    )
    model.fit(real_data)

    assert math.isnan(model.restart_lls_[0])
    assert model.restart_seeds_ == SEEDS
    healthy = {1: independent_fits[1].ll[-1], 2: independent_fits[2].ll[-1]}
    expected = max(healthy, key=lambda index: healthy[index])
    assert model.converged is True
    assert model.ll[-1] == healthy[expected]
    assert model.seed == SEEDS[expected]
    assert _same(model.A, independent_fits[expected].A)


def test_all_degenerate_restarts_keep_the_degenerate_contract(real_data, tmp_path):
    """With nothing to select the model keeps the last restart, ``converged``
    stays False and nothing is written -- the pre-#198 contract, unchanged."""
    model = _nan_model(
        tmp_path,
        nan_seeds=tuple(SEEDS),
        seed=0,
        n_restarts=len(SEEDS),
        restart_seeds=SEEDS,
    )
    model.fit(real_data)

    assert model.converged is False
    assert model.restart_seeds_ == SEEDS
    assert all(math.isnan(ll) for ll in model.restart_lls_)
    assert len(model.restart_stop_reasons_) == len(SEEDS)
    assert not (tmp_path / "W").exists(), "a degenerate fit must not be written"
    assert "did not converge" in (tmp_path / "out.txt").read_text().lower()
