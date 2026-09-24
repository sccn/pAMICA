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


# Float32 perturbation study behind the variance_order agreement check below:
# the same 2-model, 30-iteration MLX fit on the input and on 48 copies of it,
# each element multiplied by (1 + eps * z) with eps the float32 machine
# epsilon (1.19e-7) and z standard normal (numpy default_rng(20260922)), each
# compared with its float64 twin (98 per-model orders, Apple M4 Pro). Both
# backends compute the order host-side in float64, so the per-component
# variance difference comes only from MLX's float32 parameters (W and the
# sphere): at most 1.65e-6 relative. No pair of components fell inside the
# resulting band in any fit (the closest adjacent pair sat 34x outside its
# band), so no order ever differed; 35 of the 3038 adjacent gaps were under
# 1e-3 relative, none under 3e-5.
_SVAR_STUDY_MAX_REL = 1.65e-6
_SVAR_SAFETY = 3.0
# At most one in-band swap per model. From the study's gap density near zero
# (35 of 3038 adjacent gaps under 1e-3) and its widest band (2.9e-6), an
# adjacent pair lands inside the band with probability about 3e-5, so about
# 1e-3 in-band pairs are expected per 32-component model: one swap covers a
# rare near-tie on other hardware, while two in one model (about 5e-7) would
# signal a real ordering change, as would any swap outside the band.
_MAX_ORDER_SWAPS = 1


def _per_component(order: np.ndarray, svar_sorted: np.ndarray) -> np.ndarray:
    """Undo ``variance_order``'s sort: the variance of each source index."""
    assert np.array_equal(np.sort(order), np.arange(order.size))
    v = np.empty(order.size, dtype=np.float64)
    v[order] = svar_sorted
    return v


def _order_violations(
    order32: np.ndarray, v32: np.ndarray, order64: np.ndarray, v64: np.ndarray
) -> list[str]:
    """Every way the MLX order (``order32``, per-source variances ``v32``)
    departs from the float64 twin's (``order64``, ``v64``) beyond float32
    rounding; empty when the two agree.

    A pair of sources may swap order only when the gap between their twin
    variances lies inside the float32 band, the sum of the two sources'
    measured ``|v32 - v64|``. The band's width is itself capped at
    ``_SVAR_SAFETY`` times the study's largest per-source difference, and the
    number of swapped pairs at ``_MAX_ORDER_SWAPS``.
    """
    n = v64.size
    d = np.abs(v32 - v64)
    violations = []
    rel = d / v64
    cap = _SVAR_SAFETY * _SVAR_STUDY_MAX_REL
    if rel.max() > cap:
        violations.append(
            f"source {int(rel.argmax())}'s variance differs by {rel.max():.2e} "
            f"relative, over the {cap:.2e} float32 cap"
        )
    rank32 = np.empty(n, dtype=np.int64)
    rank32[order32] = np.arange(n)
    rank64 = np.empty(n, dtype=np.int64)
    rank64[order64] = np.arange(n)
    i, j = np.triu_indices(n, k=1)
    swapped = (rank32[i] - rank32[j]) * (rank64[i] - rank64[j]) < 0
    gap = np.abs(v64[i] - v64[j])
    band = d[i] + d[j]
    wide = np.flatnonzero(swapped & (gap > band))
    if wide.size:
        k = wide[np.argmax(gap[wide] / band[wide])]
        violations.append(
            f"{wide.size} swapped pairs lie outside the float32 band, worst "
            f"sources {int(i[k])} and {int(j[k])}: gap {gap[k]:.3e} > band "
            f"{band[k]:.3e}"
        )
    n_swaps = int(swapped.sum())
    if n_swaps > _MAX_ORDER_SWAPS:
        violations.append(
            f"{n_swaps} swapped pairs, over the cap of {_MAX_ORDER_SWAPS}"
        )
    return violations


def _variance_orders(model, ng, h: int):
    """Both backends' order and per-source variances for model ``h``, after
    checking that each order sorts its own variances in descending order."""
    order32, svar32 = model.variance_order(h, return_svar=True)
    order64, svar64 = ng.variance_order(h, return_svar=True)
    assert np.all(np.diff(svar32) <= 0) and np.all(np.diff(svar64) <= 0)
    v32 = _per_component(order32, np.asarray(svar32, dtype=np.float64))
    v64 = _per_component(order64, np.asarray(svar64, dtype=np.float64))
    return order32, v32, order64, v64


@pytest.fixture(scope="module")
def variance_order_fit():
    """One 2-model MLX fit and its float64 twin, shared by the
    ``variance_order`` agreement test and its negative controls."""
    model = _fit_model(n_models=2, max_iter=30)
    return model, _torch_twin(model)


def test_variance_order_matches_float64_twin(variance_order_fit):
    """``variance_order`` (issue #92, epic #278 polish round) agrees with the
    float64 twin on the actual ORDER, not just close variance values.

    The two orders must agree exactly except for swaps between sources whose
    variance gap lies inside the float32 band measured here, with the band's
    width and the number of swaps capped from the perturbation study recorded
    above (see :func:`_order_violations`). This replaced a fixed 1e-3
    near-tie precondition on the twin's gaps: it assumed a float32 band about
    300 times wider than the measured one, and failed on a CI runner whose
    GPU trajectory produced a 1.65e-4 gap, well clear of any float32
    ambiguity. ``test_variance_order_check_catches_a_divergence`` shows the
    check fails on real divergences. Multi-model, so ``model_idx`` routing is
    exercised.
    """
    model, ng = variance_order_fit
    for h in range(2):
        violations = _order_violations(*_variance_orders(model, ng, h))
        assert not violations, f"model {h}: " + "; ".join(violations)


@pytest.mark.parametrize(
    ("corruption", "expected"),
    [
        ("reversed", "outside the float32 band"),
        ("shuffled", "outside the float32 band"),
        ("closest_pair_swap", "outside the float32 band"),
        ("inflated_variance", "float32 cap"),
    ],
)
def test_variance_order_check_catches_a_divergence(
    variance_order_fit, corruption, expected
):
    """Negative controls for :func:`_order_violations` on the real fit: the
    MLX side's order reversed or shuffled; the closest adjacent pair whose
    gap lies outside the float32 band swapped (the hardest real swap to
    catch, and a single one, under the swap cap, so only the band can catch
    it); or one source's variance moved by twice the float32 cap."""
    model, ng = variance_order_fit
    order32, v32, order64, v64 = _variance_orders(model, ng, 0)
    assert not _order_violations(order32, v32, order64, v64)

    if corruption == "reversed":
        order32 = order32[::-1].copy()
    elif corruption == "shuffled":
        order32 = np.random.default_rng(0).permutation(order32)
    elif corruption == "closest_pair_swap":
        d = np.abs(v32 - v64)[order64]
        gap = -np.diff(v64[order64])
        separated = np.flatnonzero(gap > d[:-1] + d[1:])
        k = int(separated[np.argmin(gap[separated])])
        pa, pb = (int(np.flatnonzero(order32 == src)[0]) for src in order64[k : k + 2])
        order32 = order32.copy()
        order32[[pa, pb]] = order32[[pb, pa]]
    else:
        v32 = v32.copy()
        v32[order32[0]] *= 1.0 + 2.0 * _SVAR_SAFETY * _SVAR_STUDY_MAX_REL

    violations = _order_violations(order32, v32, order64, v64)
    assert any(expected in v for v in violations), violations


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
