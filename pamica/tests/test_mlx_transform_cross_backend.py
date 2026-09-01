"""``transform`` and the mixing/unmixing accessors: MLX (float32) vs PyTorch
(float64) agreement (issue #287, epic #278 Phase 1; ``variance_order`` added
in the epic's post-Phase-3 polish round, issue #92).

Cross-backend by design, so this lives in ``pamica/tests/`` rather than
``pamica/tests/mlx_tests/`` (``.rules/backend_parity.md``): the same split as
``test_mlx_newton_cross_backend.py``/``test_mlx_pdf_families_cross_backend.py``/
``test_mlx_sharing_cross_backend.py``. Each of those isolates the question
"does float32 MLX survive against a float64 oracle holding IDENTICAL fitted
parameters", not a fitting-trajectory comparison -- and that is exactly the
question here for ``transform``/the ``get_*`` accessors. The MLX-only
mechanics (model_idx validation, unfitted errors, the training-data bit-level
check, the fit-path no-op pin) stay in ``mlx_tests/test_mlx_transform.py``.

Every comparison starts from ONE real fitted MLX state copied into a float64
``AMICATorchNG`` twin, so only the arithmetic differs (the same construction
as ``test_mlx_newton_cross_backend.py::_torch_twin``).

Real bundled sample EEG only, no synthetic data or mocks (``.rules/testing.md``).
MLX is an optional Apple-Silicon backend, so the module self-skips via
``importorskip`` plus an Apple-GPU guard; PyTorch always runs.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from pamica.torch_impl.core import AMICATorchNG

mx = pytest.importorskip("mlx.core", reason="MLX not installed (Apple Silicon only)")
mlx_core = pytest.importorskip(
    "pamica.mlx_impl.core", reason="MLX not installed (Apple Silicon only)"
)
AMICAMLXNG = mlx_core.AMICAMLXNG

SAMPLE_DIR = Path(__file__).resolve().parents[1] / "sample_data"
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


def _torch_twin(model: "AMICAMLXNG", dtype=torch.float64) -> "AMICATorchNG":
    """A float64 ``AMICATorchNG`` holding the fitted MLX model's exact state
    (params + preprocessing), so the two differ only in the arithmetic they
    run for transform()/the get_* accessors. Same construction as
    ``test_mlx_newton_cross_backend.py::_torch_twin``, without the
    Newton-specific fields (not needed here)."""
    ng = AMICATorchNG(
        n_channels=model.n_channels,
        n_models=model.n_models,
        n_mix=NMIX,
        device="cpu",
        dtype=dtype,
        seed=SEED,
        keep_best=False,
    )
    ng._initialize_parameters()
    for name in ("A", "mu", "alpha", "beta", "rho", "gm", "c"):
        value = np.array(getattr(model, name)).astype(np.float64)
        setattr(ng, name, torch.from_numpy(value).to(dtype))
    ng.comp_list = torch.from_numpy(np.array(model.comp_list).astype(np.int64))
    ng.pdtype = torch.from_numpy(np.array(model.pdtype).astype(np.int64))
    ng.mean = torch.from_numpy(np.array(model.mean).astype(np.float64)).to(dtype)
    assert model._sphere_np is not None
    ng.sphere = torch.from_numpy(model._sphere_np.copy()).to(dtype)
    ng._sphere_pinv = None
    ng.sldet = model.sldet
    ng._update_unmixing_matrices()
    return ng


def _relerr(a: np.ndarray, b: np.ndarray) -> float:
    """Max relative error against the float64 reference ``b``, floored so a
    near-zero reference entry cannot manufacture a huge ratio (same
    construction as ``test_mlx_newton_cross_backend.py::_relerr``). Suited to
    the mixing/unmixing MATRICES below (``get_mixing_matrix`` etc.), whose
    entries do not pervasively pass through zero."""
    scale = np.maximum(np.abs(b), np.abs(b).max() * 1e-6)
    return float(np.max(np.abs(a - b) / scale))


def _max_rel_disagreement(a: np.ndarray, b: np.ndarray) -> float:
    """Max entrywise absolute disagreement normalized by the reference
    array's own dynamic range: ``||a - b||_inf / ||b||_inf``.

    Used for ``transform``'s output specifically, which is NOT a good fit for
    :func:`_relerr`'s per-entry floored ratio: transform's per-sample source
    activations are signed and legitimately pass through zero (they are
    roughly mean-zero source signals), so a per-entry relative error is
    dominated by noise at those near-zero crossings rather than by the actual
    float32-vs-float64 disagreement (measured: entrywise ratios up to ~3%
    driven entirely by ~5e-6-scale entries, while the max absolute
    disagreement across the whole array is ~5e-6). This max-norm form is how
    a reconstruction/unmixing error is normally reported.
    """
    scale = np.abs(b).max()
    return float(np.abs(a - b).max() / scale)


def _fit_model(n_models: int = 1, max_iter: int = 5, **kwargs) -> "AMICAMLXNG":
    model = AMICAMLXNG(
        n_channels=NW,
        n_models=n_models,
        n_mix=NMIX,
        seed=SEED,
        block_size=BLOCK,
        **kwargs,
    )
    data = _real_data()
    model.fit(data, max_iter=max_iter, verbose=False)
    assert model.stop_reason not in AMICAMLXNG._DEGENERATE_STOP_REASONS
    return model


def test_transform_matches_float64_torch_twin():
    """G1 for transform: the fitted MLX sources agree with the float64 twin.

    Measured on the bundled sample (5-iteration single-model fit, 4096
    samples): max absolute disagreement ~4.7e-6 (float32 rounding scale);
    ``_max_rel_disagreement`` (normalized by the twin's own dynamic range,
    ~15.9) reports that as ~3e-7. The threshold is set two orders above the
    measured value, not at it, so this stays a real regression guard rather
    than a pin to today's exact float32 noise.
    """
    model = _fit_model(n_models=1)
    ng = _torch_twin(model)
    data = _real_data()

    S_mlx = model.transform(data, model_idx=0)
    S_ng = ng.transform(data, model_idx=0)

    err = _max_rel_disagreement(S_mlx.astype(np.float64), S_ng)
    assert err < 1e-5, f"transform differs from the float64 twin by {err:.3e}"


def test_multimodel_transform_matches_float64_twin_with_c_centering():
    """Multi-model ``model_idx`` routing, including the nonzero per-model bias
    ``c`` (issue #27): Newton on with ``newt_start=0`` so ``c`` actually moves
    off zero within a few iterations, mirroring
    ``test_ng_backend.py::test_multimodel_transform_applies_bias_c``."""
    model = _fit_model(n_models=2, max_iter=4, do_newton=True, newt_start=0)
    c = np.array(model.c)
    assert not np.allclose(c, 0.0), "c never moved; this test would be vacuous"
    assert not np.allclose(c[:, 0], c[:, 1]), "the two models' c did not differ"

    ng = _torch_twin(model)
    data = _real_data()

    for h in range(2):
        S_mlx = model.transform(data, model_idx=h)
        S_ng = ng.transform(data, model_idx=h)
        err = _max_rel_disagreement(S_mlx.astype(np.float64), S_ng)
        assert err < 1e-5, f"model {h}: transform differs from twin by {err:.3e}"


def test_accessors_match_float64_torch_twin():
    """``get_mixing_matrix``/``get_unmixing_matrix``/``get_rho`` agree with the
    float64 twin on a full-rank multi-model fit.

    ``get_unmixing_matrix`` alone gets the looser 1e-3 bound: ``get_mixing_matrix``
    and ``get_rho`` are pure reindex/transpose views of ``A``/``rho`` (which are
    copied bit-for-bit into the twin, see ``_torch_twin``), so they agree to
    float32 precision (~1e-4). ``W`` is instead the output of two INDEPENDENT
    matrix inversions -- MLX's float32 CPU-stream ``mx.linalg.inv`` in
    ``_update_unmixing_matrices`` vs. torch's float64 ``torch.linalg.inv`` -- so
    it carries real (if still small) numerical disagreement on top of the
    float32 gap; measured ~1e-5 to ~1e-4 across runs of this test, comfortably
    inside the 1e-3 bound.
    """
    model = _fit_model(n_models=2, max_iter=5)
    ng = _torch_twin(model)

    for h in range(2):
        a_mlx = model.get_mixing_matrix(h)
        a_ng = ng.get_mixing_matrix(h)
        err = _relerr(a_mlx.astype(np.float64), a_ng)
        assert err < 1e-4, f"model {h}: get_mixing_matrix differs by {err:.3e}"

        w_mlx = model.get_unmixing_matrix(h)
        w_ng = ng.get_unmixing_matrix(h)
        err = _relerr(w_mlx.astype(np.float64), w_ng)
        assert err < 1e-3, f"model {h}: get_unmixing_matrix differs by {err:.3e}"

        rho_mlx = model.get_rho(h)
        rho_ng = ng.get_rho(h)
        err = _relerr(rho_mlx.astype(np.float64), rho_ng)
        assert err < 1e-4, f"model {h}: get_rho differs by {err:.3e}"


def test_variance_order_matches_float64_twin():
    """``variance_order`` (issue #92, epic #278 polish round) agrees with the
    float64 twin on the actual ORDER, not just close variance values.

    A near-tie between two sources' back-projected variance would make the
    order genuinely ambiguous between float32 MLX and float64 torch even
    though both computed the quantity correctly -- that risk is checked for,
    not silently assumed away: the assertion below requires every consecutive
    gap in the twin's own (float64) descending-sorted variances to clear
    1e-3 relative, well above the ~1e-4 to 1e-3 float32-vs-float64
    disagreement measured for ``get_mixing_matrix``/``get_rho`` above, so an
    order mismatch here is a real regression rather than a coin flip on a
    near-tied pair. A short (5-iteration) fit was tried first and rejected
    for this reason -- its minimum gap was ~2e-4, inside the float32 noise
    band -- so this test runs the fit longer (30 iterations) specifically to
    reach a spectrum with real separation (measured minimum gap ~0.47%,
    comfortably above the bound below); a shorter/noisier config is exactly
    the kind of near-tie this assertion exists to catch and reject rather
    than let the order check pass by luck. If a future data/config change
    shrinks the gap back down, this assertion is designed to fail loudly (not
    the order check) so the weaker config gets replaced rather than the order
    check getting silently loosened. Multi-model, so ``model_idx`` routing is
    exercised.
    """
    model = _fit_model(n_models=2, max_iter=30)
    ng = _torch_twin(model)

    for h in range(2):
        order_mlx, svar_mlx = model.variance_order(h, return_svar=True)
        order_ng, svar_ng = ng.variance_order(h, return_svar=True)

        gaps = -np.diff(svar_ng) / np.maximum(svar_ng[:-1], svar_ng.max() * 1e-6)
        assert gaps.min() > 1e-3, (
            f"model {h}: variance gaps too tight ({gaps.min():.2e}) for an "
            "order comparison to be meaningful on this fit/config"
        )

        assert np.array_equal(order_mlx, order_ng), (
            f"model {h}: variance_order disagrees: {order_mlx} vs {order_ng}"
        )
        err = _relerr(svar_mlx.astype(np.float64), svar_ng)
        assert err < 1e-3, f"model {h}: variance_order svar differs by {err:.3e}"


def test_sensor_mixing_matrix_matches_twin_on_rank_reduced_fit():
    """``get_sensor_mixing_matrix`` -- the ``pinv(sphere) @ A`` back-map -- on a
    genuinely rank-reduced real fit (real EEG projected onto a rank-20
    subspace, the same construction as
    ``mlx_tests/test_mlx_newton.py::test_rank_reduced_newton_fit_completes``).
    This is the only accessor that reads the sphere PSEUDO-inverse (issue
    #223), so it is the one where a rank/shape bug would show."""
    x = _real_data(4096)
    x = x - x.mean(axis=1, keepdims=True)
    rank = 20
    U_r = np.linalg.svd(x, full_matrices=False)[0][:, :rank]
    x_low = U_r @ (U_r.T @ x)

    model = AMICAMLXNG(n_channels=NW, n_mix=NMIX, seed=SEED, block_size=BLOCK)
    model.fit(x_low, max_iter=5, verbose=False)
    assert model.stop_reason not in AMICAMLXNG._DEGENERATE_STOP_REASONS
    assert model.n_channels == rank
    assert model._sphere_np is not None and model._sphere_np.shape == (rank, NW)

    ng = _torch_twin(model)
    assert ng.sphere is not None and tuple(ng.sphere.shape) == (rank, NW)

    a_mlx = model.get_sensor_mixing_matrix(0)
    a_ng = ng.get_sensor_mixing_matrix(0)
    assert a_mlx.shape == (NW, rank)
    err = _relerr(a_mlx.astype(np.float64), a_ng)
    assert err < 1e-3, f"get_sensor_mixing_matrix differs from twin by {err:.3e}"


def test_transform_matches_forward_activations_via_twin_composition():
    """Cross-check the transpose identity ``transform`` relies on
    (``S = W^T (x - c)`` equals ``_forward``'s ``b = (x - c)^T W`` transposed)
    against the float64 twin's own ``_forward``, not just against MLX's own
    ``_forward`` (that in-backend check lives in
    ``mlx_tests/test_mlx_transform.py``)."""
    model = _fit_model(n_models=1, max_iter=5)
    ng = _torch_twin(model)
    data = _real_data()

    S_ng = ng.transform(data, model_idx=0)
    assert ng.sphere is not None and ng.mean is not None
    X_t = ng.sphere @ (
        torch.from_numpy(np.ascontiguousarray(data)).to(ng.device, ng.dtype) - ng.mean
    )
    logV, b_list, z_list, y_list, azrho_list = ng._forward(X_t)
    b0 = b_list[0].cpu().numpy()
    assert np.allclose(S_ng.T, b0, atol=1e-9), "torch twin: transform != _forward b"

    S_mlx = model.transform(data, model_idx=0)
    err = _max_rel_disagreement(S_mlx.astype(np.float64), S_ng)
    assert err < 1e-5, f"transform differs from the float64 twin by {err:.3e}"
