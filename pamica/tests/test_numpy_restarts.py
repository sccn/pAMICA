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

from pamica import AMICA_NumPy, restarts
from pamica.numpy_impl.data import load_data_file, load_results

_FDT = Path(__file__).resolve().parent.parent / "sample_data" / "eeglab_data.fdt"
NW = 32
FIELD = 30504
MAX_ITER = 5
# Ordered best-to-worst, so the winner is the FIRST restart and the restore path
# is what the acceptance test exercises. Chosen for separation: measured on this
# fixture at MAX_ITER the final LLs are -3.264177210 / -3.264757966 /
# -3.264984217, i.e. the winner leads the runner-up by 5.8e-4 and the last
# restart by 8.1e-4 -- ~8 orders of magnitude above float64 BLAS summation noise
# (~1e-12 relative on an LL of ~3.26), so an OpenBLAS/Accelerate swap cannot
# reorder the winners (the issue #241 flake class). The same seeds are used by
# the PyTorch suite, which agrees to 1e-9 here, and by the MLX suite (float32,
# 6e-7).
SEEDS = [54, 50, 42]

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
        # The FULL state tuple: (bit-generator name, key array, pos, has_gauss,
        # cached_gaussian). The last three are what make two Mersenne Twisters
        # with the same key still draw differently, so an exact-equality claim
        # that skipped them would pass on RNGs that are not in fact identical.
        left_state, right_state = left.get_state(), right.get_state()
        return (
            left_state[0] == right_state[0]
            and np.array_equal(left_state[1], right_state[1])
            and left_state[2:] == right_state[2:]
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
    assert expected != len(SEEDS) - 1, (
        "precondition: the winning seed must not be the last restart, or the "
        "restore path goes untested. The margin is 5.8e-4 (see SEEDS), far above "
        "float64 summation noise, so a failure here is a platform-BLAS ordering "
        "finding worth reporting -- not a correctness bug in the restart loop."
    )
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


class _RaiseForSeeds(AMICA_NumPy):
    """Raises ``numpy.linalg.LinAlgError`` for the listed seeds only.

    The failure mode a non-finite likelihood does NOT cover: a truly singular
    ``A`` makes ``numpy.linalg.inv`` raise rather than return infinities, so
    the fit never reaches the loop's own guard. Injected at the hook the real
    failure comes from (``_update_unmixing_matrices`` ->
    ``get_unmixing_matrices`` -> ``np.linalg.inv``) with the exception type
    NumPy itself raises, so what is under test is the restart loop's isolation.
    """

    raise_seeds: tuple = ()

    def _update_unmixing_matrices(self):
        if self.seed in self.raise_seeds:
            raise np.linalg.LinAlgError("Singular matrix")
        return super()._update_unmixing_matrices()


def _raising_model(outdir: Path, raise_seeds, **kwargs: Any) -> AMICA_NumPy:
    model = _RaiseForSeeds(
        num_models=1,
        num_mix=3,
        max_iter=MAX_ITER,
        use_tqdm=False,
        outdir=str(outdir),
        writestep=10_000_000,
        **kwargs,
    )
    model.raise_seeds = tuple(raise_seeds)
    return model


def test_a_crashing_restart_does_not_kill_the_search(
    real_data, tmp_path, independent_fits
):
    """Isolation (issue #198 review). A restart that RAISES must not discard the
    restarts that already succeeded; it is recorded as a degenerate
    ``restart_error`` and the search continues with the next seed."""
    # The BEST seed crashes, so the winner has to come from the survivors.
    model = _raising_model(
        tmp_path, (SEEDS[0],), seed=0, n_restarts=len(SEEDS), restart_seeds=SEEDS
    )
    model.fit(real_data)

    assert model.restart_stop_reasons_[0] == restarts.ERROR_STOP_REASON
    assert math.isnan(model.restart_lls_[0])
    healthy = {1: independent_fits[1].ll[-1], 2: independent_fits[2].ll[-1]}
    expected = max(healthy, key=lambda index: healthy[index])
    assert model.converged is True
    assert model.ll[-1] == healthy[expected]
    assert model.seed == SEEDS[expected]
    assert _same(model.A, independent_fits[expected].A)


def test_a_single_restart_still_raises(real_data, tmp_path):
    """The other half of the isolation contract: bit-identity with a pre-#198
    fit includes ERROR behavior, so the single-restart path must NOT catch."""
    model = _raising_model(tmp_path, (SEEDS[0],), seed=SEEDS[0])
    with pytest.raises(np.linalg.LinAlgError):
        model.fit(real_data)


def test_all_restarts_crashing_keeps_the_degenerate_contract(real_data, tmp_path):
    """Every restart raises: the fit reports failure and writes nothing, rather
    than the exception escaping (losing the records) or a partial state being
    reported as converged."""
    model = _raising_model(
        tmp_path, tuple(SEEDS), seed=0, n_restarts=len(SEEDS), restart_seeds=SEEDS
    )
    model.fit(real_data)

    assert model.converged is False
    assert model.stop_reason == restarts.ERROR_STOP_REASON
    assert model.restart_stop_reasons_ == [restarts.ERROR_STOP_REASON] * len(SEEDS)
    assert all(math.isnan(ll) for ll in model.restart_lls_)
    assert not (tmp_path / "W").exists(), "a crashed fit must not be written"


# ---------------------------------------------------------------------------
# Interaction with the on-disk write cadence and the block-size search
# ---------------------------------------------------------------------------


def test_the_written_output_is_the_winners_not_the_last_restarts(real_data, tmp_path):
    """Every restart's ``writestep`` checkpoints pass through the same files, so
    the on-disk result is only correct if the final write happens after the
    winner is restored. ``writestep=1`` makes every iteration of every restart
    write, which is the harshest version of that race; the documented caveat
    (a loser's intermediate output can appear mid-fit) becomes a tested claim
    about what is on disk when ``fit`` returns."""
    model = _model(
        tmp_path, seed=0, n_restarts=len(SEEDS), restart_seeds=SEEDS, writestep=1
    )
    model.fit(real_data)

    saved = load_results(tmp_path)
    assert _same(saved["A"], model.A)
    assert _same(saved["W"], model.W)
    assert _same(saved["mu"], model.mu)
    # ... and that IS the winner, not merely self-consistent.
    winner = max(range(len(SEEDS)), key=lambda i: model.restart_lls_[i])
    assert winner == 0, "SEEDS puts the winner first; see the SEEDS comment"
    reference = _model(tmp_path / "ref", seed=SEEDS[winner])
    reference.fit(real_data)
    assert _same(saved["A"], reference.A)


def test_the_winners_tuned_block_size_is_restored(real_data, tmp_path, monkeypatch):
    """``do_opt_block`` re-times the block size inside every restart, so the
    tuned value is per-restart state and the winner's must survive the restore.

    The search's RETURN VALUE is stubbed rather than its timings: which size
    wins a real sweep is documented as machine-dependent (``blocktune``), so
    pinning it deterministically is the only way to assert this at all.
    """
    from pamica.numpy_impl import core as np_core

    sizes = iter([4096, 8192, 16384])
    monkeypatch.setattr(np_core.blocktune, "search", lambda **kwargs: next(sizes))

    model = _model(
        tmp_path, seed=0, n_restarts=len(SEEDS), restart_seeds=SEEDS, do_opt_block=True
    )
    model.fit(real_data)

    assert max(range(len(SEEDS)), key=lambda i: model.restart_lls_[i]) == 0
    assert model.block_size == 4096
