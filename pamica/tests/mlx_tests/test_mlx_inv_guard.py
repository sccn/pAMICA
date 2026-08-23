"""Condition-number guard in ``_update_unmixing_matrices`` -- issue #274.

MLX 0.32's CPU-stream ``mx.linalg.inv`` does not raise a catchable Python
exception on a singular matrix: LAPACK's LU failure aborts the whole process
(``libc++abi: ... [Inverse::eval_cpu] LU factorization failed``), which no
``try``/``except`` around ``fit`` can catch. ``_update_unmixing_matrices`` now
condition-checks each per-model matrix host-side immediately before calling
``inv`` and raises a catchable ``RuntimeError`` instead.

This module covers the guard mechanics: a genuinely singular ``A`` (and a
near-singular one) raise ``RuntimeError`` rather than aborting the process,
and the guard is read-only -- it reproduces the exact ``W``/``_logdet_W`` a
call with no guard at all would have produced, on the same real fitted state.
Real sample EEG throughout, following the manipulate-real-fitted-state pattern
used by ``test_mlx_sharing.py``'s ``_force_merged_column``.
"""

from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "sample_data"
DATA_FILE = SAMPLE_DIR / "eeglab_data.fdt"
NW = 32
FIELD = 30504
NMIX = 3
SEED = 42
BLOCK = 1024

pytestmark = [
    pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing"),
    pytest.mark.skipif(
        mx.default_device().type != mx.DeviceType.gpu, reason="no Apple GPU"
    ),
]


def _real_data(n_samples: int = 4096) -> np.ndarray:
    from pamica.torch_impl.utils import load_eeglab_data

    data = load_eeglab_data(str(DATA_FILE), data_dim=NW, field_dim=FIELD)
    return data[:, :n_samples].astype(np.float64)


def _warm_model(n_models: int = 1, warmup: int = 5, **kwargs):
    """A model driven ``warmup`` M-steps past init on real sample EEG, mirroring
    ``test_mlx_sharing.py``'s ``_warm_model`` -- hand-driving the loop is what
    lets the caller manipulate ``A`` between two specific, known-good states.
    """
    from pamica.mlx_impl import AMICAMLXNG

    model = AMICAMLXNG(
        n_channels=NW,
        n_models=n_models,
        n_mix=NMIX,
        seed=SEED,
        block_size=BLOCK,
        **kwargs,
    )
    x_t = model._preprocess(_real_data())
    model._initialize_parameters()
    for it in range(warmup):
        model.iteration = it
        model._update_parameters(model._accumulate_blocks(x_t), x_t.shape[1])
    mx.eval(model.A, model.mu, model.alpha, model.beta, model.rho, model.gm, model.c)
    return model, x_t


# --- singular / near-singular A raises, does not abort -----------------------


def test_duplicated_column_raises_runtime_error_not_abort():
    """Duplicating a column of a real fitted ``A`` makes model 0's per-model
    submatrix exactly singular. Before the guard this reached MLX's CPU-stream
    ``inv`` and aborted the whole process; now it must raise a catchable
    ``RuntimeError`` naming the model and iteration -- the test process itself
    surviving to assert on it is part of what is being verified."""
    model, _ = _warm_model()
    a_np = np.array(model.A)
    a_np[:, 1] = a_np[:, 0]  # exact duplicate column -> singular submatrix
    model.A = mx.array(a_np)
    model.iteration = 7

    with pytest.raises(RuntimeError, match=r"Singular unmixing matrix for model 0"):
        model._update_unmixing_matrices()


def test_duplicated_column_error_names_iteration_and_condition():
    """The RuntimeError message is informative: model index, iteration, and a
    huge (not merely 'not finite') condition number, not just a generic
    failure -- so a user hitting this in the wild can act on it."""
    from pamica.mlx_impl.core import _INV_COND_THRESHOLD

    model, _ = _warm_model()
    a_np = np.array(model.A)
    a_np[:, 1] = a_np[:, 0]
    model.A = mx.array(a_np)
    model.iteration = 13

    with pytest.raises(RuntimeError) as exc_info:
        model._update_unmixing_matrices()

    msg = str(exc_info.value)
    assert "model 0" in msg
    assert "iteration 13" in msg
    assert f"{_INV_COND_THRESHOLD:.1e}" in msg


def test_near_singular_column_raises():
    """A near-singular (not bit-exact-duplicate, not exactly zero) column is
    also caught: scaling one column of a real fitted ``A`` down by 1e-13
    leaves it finite and nonzero -- MLX's CPU-stream ``inv`` actually
    SUCCEEDS on this exact construction even at far more extreme scales (see
    ``_INV_COND_THRESHOLD``'s comment), so this is not a matrix that would
    necessarily abort on its own -- yet the guard still catches it, because
    it is a condition check, not an ``isfinite``/exact-singularity check."""
    from pamica.mlx_impl.core import _INV_COND_THRESHOLD

    model, _ = _warm_model()
    a_np = np.array(model.A)
    a_np[:, 3] = (a_np[:, 3].astype(np.float64) * 1e-13).astype(np.float32)
    assert not np.any(a_np[:, 3] == 0.0)  # scaled small, not zeroed out
    assert np.all(np.isfinite(a_np))

    # Sanity: the perturbation really is near-singular in float32 (this is
    # what the guard computes internally, independently reproduced here).
    cond = np.linalg.cond(a_np)
    assert np.isfinite(cond) and cond > _INV_COND_THRESHOLD

    model.A = mx.array(a_np)
    model.iteration = 0

    with pytest.raises(RuntimeError, match=r"Singular unmixing matrix"):
        model._update_unmixing_matrices()


def test_second_model_singular_names_model_one():
    """A 2-model fit where only model 1's submatrix is singular is named
    correctly -- the guard must not just blame model 0 by default."""
    model, _ = _warm_model(n_models=2)
    cl = np.array(model.comp_list)
    a_np = np.array(model.A)
    # Duplicate two columns WITHIN model 1's own comp_list selection.
    rows = cl[:, 1]
    a_np[:, rows[1]] = a_np[:, rows[0]]
    model.A = mx.array(a_np)
    model.iteration = 2

    with pytest.raises(RuntimeError, match=r"Singular unmixing matrix for model 1"):
        model._update_unmixing_matrices()


# --- guard is read-only: bit-identical to an unguarded inv/slogdet -----------


def test_guard_is_bit_identical_on_healthy_state():
    """On a healthy (well-conditioned) real fitted state, the guarded method
    must produce the exact same ``W``/``_logdet_W`` as calling
    ``mx.linalg.inv``/``slogdet`` directly with no guard at all -- proving the
    condition check is read-only and does not perturb the numerics it
    precedes."""
    from pamica.mlx_impl.core import _CPU

    model, _ = _warm_model(n_models=2)

    ws_ref, logdets_ref = [], []
    for h in range(model.n_models):
        wh = mx.linalg.inv(model.A[:, model.comp_list[:, h]], stream=_CPU)
        ws_ref.append(wh)
        logdets_ref.append(mx.linalg.slogdet(wh, stream=_CPU)[1])
    w_ref = mx.stack(ws_ref, axis=0)
    logdet_ref = mx.stack(logdets_ref)

    model._update_unmixing_matrices()

    np.testing.assert_array_equal(np.array(model.W), np.array(w_ref))
    np.testing.assert_array_equal(np.array(model._logdet_W), np.array(logdet_ref))


def test_healthy_full_fit_completes_unaffected():
    """A standard fit on the bundled sample never comes near the guard
    threshold (measured max condition number ~3.8 across single- and 2-model
    fits, do_newton and share_comps included -- eleven orders of magnitude
    below ``_INV_COND_THRESHOLD``; see the issue #274 PR body) and completes
    exactly as it did before the guard: finite log-likelihood, finite W, a
    non-degenerate stop reason. (A deliberately adversarial existing test,
    ``test_mlx_newton.py::test_fallback_ramps_toward_lrate_cap_and_counts``,
    legitimately reaches cond~4.4e9 by repeatedly stepping from the same
    under-determined block -- still ~225x under the threshold; that test's
    own pass/fail is the guard's regression check for it, not this one.)"""
    from pamica.mlx_impl import AMICAMLXNG

    m = AMICAMLXNG(n_channels=NW, n_models=2, n_mix=NMIX, seed=SEED, block_size=BLOCK)
    m.fit(_real_data(), max_iter=30, verbose=False)

    assert m.stop_reason not in AMICAMLXNG._DEGENERATE_STOP_REASONS
    assert m.final_ll_ is not None
    assert np.isfinite(m.final_ll_)
    assert np.all(np.isfinite(np.array(m.W)))
