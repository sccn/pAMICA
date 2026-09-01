"""EEGLAB export (``write_amica_output``) on the MLX backend -- issue #92,
epic #278 Phase 3/#289 (port of ``AMICATorchNG.write_amica_output``,
torch_impl/core.py:3285-3366).

Round-trips through the shared :func:`pamica.numpy_impl.load.loadmodout`
reader (the same reader real EEGLAB output is validated against, issue
#159), plus a torch-twin byte comparison: a float64 ``AMICATorchNG`` built
from the exact fitted MLX state (the ``_torch_twin`` pattern from
``pamica/tests/test_mlx_sharing_cross_backend.py``) exports the same
parameters, so the two on-disk files can be diffed directly. This embeds an
inline torch comparison in an ``mlx_tests`` file, the same precedent
``test_mlx_backend.py``/``test_mlx_sharing.py`` already set for non-drift-
guard torch comparisons; the anti-drift AGREEMENT pin for ``do_reject``
itself lives in the one new cross-backend file
(``pamica/tests/test_mlx_reject_cross_backend.py``) per ``.rules/
backend_parity.md``.

Apple-Silicon only, real sample EEG (no synthetic/mock).
"""

import logging
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from pamica.mlx_impl import AMICAMLXNG  # noqa: E402  (after the MLX importorskip)
from pamica.numpy_impl.load import loadmodout  # noqa: E402

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "sample_data"
DATA_FILE = SAMPLE_DIR / "eeglab_data.fdt"
NW = 32
FIELD = 30504
NMIX = 3
BLOCK = 1024

pytestmark = [
    pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing"),
    pytest.mark.skipif(
        mx.default_device().type != mx.DeviceType.gpu, reason="no Apple GPU"
    ),
]


@pytest.fixture(scope="module")
def real_data() -> np.ndarray:
    from pamica.torch_impl.utils import load_eeglab_data

    return load_eeglab_data(str(DATA_FILE), data_dim=NW, field_dim=FIELD).astype(
        np.float64
    )[:, :4096]


def _model(**kwargs: Any) -> AMICAMLXNG:
    params: dict[str, Any] = dict(n_channels=NW, n_mix=NMIX, block_size=BLOCK)
    params.update(kwargs)
    return AMICAMLXNG(**params)


def _llt_invariant(Lt: np.ndarray, n_good: int, nw: int) -> float:
    return float(Lt.sum()) / (n_good * nw)


_LL_TOL = 1e-5


@pytest.mark.parametrize("n_models", [1, 2])
def test_write_amica_output_round_trips_through_loadmodout(
    real_data, tmp_path, n_models
):
    m = _model(n_models=n_models, seed=42, keep_best=False)
    m.fit(real_data, max_iter=6, verbose=False)
    outdir = tmp_path / f"amicaout{n_models}"
    m.write_amica_output(outdir)
    out = loadmodout(outdir)

    assert out.W.shape == (NW, NW, n_models)
    assert out.LL is not None
    # loadmodout sorts models by descending gm (mod_prob) AND re-permutes
    # components by descending explained variance within each model (issue
    # #159), so only order-independent quantities are checked through this
    # reader -- exact byte-level reproduction (raw files, no reordering) is
    # pinned separately below, against a torch twin
    # (test_export_is_byte_compatible_with_a_torch_twin), mirroring
    # torch_tests/test_amica_ng_wrapper.py's split between
    # test_write_amica_output_bytes (raw bytes) and
    # test_write_amica_output_loadmodout_readable (order-independent, via
    # loadmodout).
    np.testing.assert_allclose(
        sorted(out.mod_prob.tolist()), sorted(np.array(m.gm).tolist()), atol=1e-6
    )
    np.testing.assert_allclose(out.LL, np.array(m.ll_history), atol=1e-6)
    # Mixture proportions per component must sum to 1: a meaningful check
    # that the (num_mix, n_comp) params were read back with the correct
    # column-major layout (a C-order write would scramble them).
    np.testing.assert_allclose(out.alpha.sum(axis=0), 1.0, atol=1e-6)

    assert out.Lt is not None and out.Lht is not None
    n_good = out.Lt.size
    inv = _llt_invariant(out.Lt, n_good, NW)
    assert abs(inv - out.LL[-1]) <= _LL_TOL


def test_share_comps_export_round_trips_through_loadmodout(real_data, tmp_path):
    """A ``share_comps`` fit that genuinely merges components must still
    export and round-trip through ``loadmodout`` -- PR #318 review item 9.

    Same forcing recipe as ``test_mlx_sharing.py``'s
    ``test_two_model_share_fit_completes_and_merges`` (``share_start=4``,
    ``share_iter=8``, ``comp_thresh=0.9``, ``max_iter=25``): deterministic
    on this fixture, not a "if it happens to merge" check.
    """
    m = _model(
        n_models=2,
        seed=3,
        share_comps=True,
        share_start=4,
        share_iter=8,
        comp_thresh=0.9,
        keep_best=False,
    )
    m.fit(real_data, max_iter=25, verbose=False)
    assert m.stop_reason not in AMICAMLXNG._DEGENERATE_STOP_REASONS
    used = int(np.array(m.comp_used).sum())
    assert used < m.n_comps, "no merge survived the fit"
    groups = m.shared_components()
    assert groups

    outdir = tmp_path / "shared_amicaout"
    m.write_amica_output(outdir)
    out = loadmodout(outdir)
    assert out.W.shape == (NW, NW, m.n_models)
    assert out.LL is not None and np.isfinite(out.LL[-1])
    assert np.all(np.isfinite(out.W))


def test_written_ll_ends_at_the_keep_best_restored_iterate(real_data):
    """Under a genuine keep_best restore (test_mlx_llt_stash.py's forced
    recipe), the written LL trajectory ends at the restored iterate, not
    the discarded later one."""
    kwargs: dict[str, Any] = dict(
        n_models=2,
        n_mix=NMIX,
        seed=0,
        block_size=BLOCK,
        do_newton=True,
        newt_start=1,
        lrate=0.5,
        use_min_dll=True,
        min_dll=1e-4,
        maxincs=2,
        use_grad_norm=False,
    )
    m = AMICAMLXNG(n_channels=NW, **kwargs)
    m.fit(real_data, max_iter=60, verbose=False)
    if m.stop_reason in AMICAMLXNG._DEGENERATE_STOP_REASONS:
        pytest.skip("aggressive run ended degenerate; not the case under test")
    assert m.final_ll_ is not None
    if np.isclose(m.ll_history[-1], m.final_ll_):
        pytest.skip("run was monotone; keep_best restore did not fire")

    with tempfile.TemporaryDirectory() as d:
        m.write_amica_output(d)
        out = loadmodout(d)
        assert out.LL is not None
        assert np.isclose(out.LL[-1], m.final_ll_)
        assert len(out.LL) < len(m.ll_history)


def test_do_reject_zero_sentinel_is_written(real_data, tmp_path):
    """Under do_reject, a rejected sample's LLt entry is written as exactly
    0.0 -- the load-bearing sentinel load_rej reconstructs from
    (amica15.f90:2231-2234)."""
    m = _model(
        seed=42,
        do_reject=True,
        rejsig=2.0,
        rejstart=2,
        rejint=3,
        maxrej=2,
        keep_best=False,
    )
    m.fit(real_data, max_iter=10, verbose=False)
    assert m.good_idx is not None
    n_rejected = real_data.shape[1] - int(m.good_idx.size)
    assert n_rejected > 0

    outdir = tmp_path / "reject_amicaout"
    m.write_amica_output(outdir)
    out = loadmodout(outdir)
    assert out.Lt is not None
    assert int((out.Lt == 0.0).sum()) == n_rejected


def test_from_state_dict_model_warns_and_omits_llt(real_data, tmp_path, caplog):
    m = _model(seed=1, keep_best=False)
    m.fit(real_data, max_iter=4, verbose=False)
    state = m.state_dict()
    restored = AMICAMLXNG.from_state_dict(state)
    assert restored._llt_lht is None and restored._llt_lt is None

    outdir = tmp_path / "restored_amicaout"
    with caplog.at_level(logging.WARNING, logger="pamica.mlx_impl.core"):
        restored.write_amica_output(outdir)
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "No LLt data available" in text
    assert not (outdir / "LLt").exists()
    assert (outdir / "W").exists()


def test_write_amica_output_requires_a_fitted_model(tmp_path):
    m = _model(seed=1)
    with pytest.raises(RuntimeError, match="fitted"):
        m.write_amica_output(tmp_path / "should-not-be-created")


def test_write_amica_output_refuses_a_degenerate_fit(tmp_path, real_data):
    """PR #311 review: write_amica_output had no refusal guard at all for a
    degenerate (non-finite-LL) model -- unlike state_dict()/save(), which
    already refuse (test_mlx_persistence.py's
    test_state_dict_refuses_degenerate_fit). AMICAMLXNG has no
    scikit-learn-style wrapper in front of it, so a caller using it
    directly had no gate whatsoever. Same NaN-injection recipe as that
    test (do_sphere=False/do_mean=False so the NaN reaches the fit
    directly, not absorbed by sphering)."""
    data = real_data.copy()
    data[0, 0] = np.nan
    m = AMICAMLXNG(n_channels=NW, n_mix=NMIX, seed=1, do_sphere=False, do_mean=False)
    m.fit(data, max_iter=5, verbose=False)
    assert m.stop_reason in AMICAMLXNG._DEGENERATE_STOP_REASONS
    with pytest.raises(RuntimeError, match="degenerate"):
        m.write_amica_output(tmp_path / "degenerate_out")
    assert not (tmp_path / "degenerate_out").exists()


def test_write_amica_output_refuses_force_set_nonfinite_param(tmp_path, real_data):
    """The defense-in-depth isfinite sweep fires independent of
    stop_reason bookkeeping, mirroring
    test_mlx_persistence.py::test_state_dict_refuses_force_set_nonfinite_param_naming_it.
    A real fit with a HEALTHY stop_reason has one param force-corrupted
    afterward (direct-attribute pattern, not a mock), so only the isfinite
    sweep -- not the stop_reason check above it -- can catch this. Also
    confirms the stale-stash note in the fix's commit message: the model's
    _llt_lht/_llt_lt are still the healthy fit's real stash (nothing wrong
    with them here), yet the write is refused anyway because a corrupted
    PARAMETER must never reach disk regardless of the stash's own health.
    """
    m = _model(seed=1, keep_best=False)
    m.fit(real_data, max_iter=4, verbose=False)
    assert m.stop_reason not in AMICAMLXNG._DEGENERATE_STOP_REASONS
    mu_np = np.array(m.mu)
    mu_np[0, 0] = np.nan
    m.mu = mx.array(mu_np)
    with pytest.raises(RuntimeError, match="non-finite parameters.*mu"):
        m.write_amica_output(tmp_path / "corrupt_out")
    assert not (tmp_path / "corrupt_out").exists()


def test_write_amica_output_makes_no_extra_forward_pass(
    real_data, tmp_path, monkeypatch
):
    """write_amica_output performs zero E-step forward passes: it consumes
    the stash instead of walking the data again (the point of #157)."""
    m = _model(n_models=2, seed=42)
    m.fit(real_data, max_iter=5, verbose=False)
    calls = []
    real_forward = m._forward
    monkeypatch.setattr(
        m, "_forward", lambda X: (calls.append(X.shape[1]), real_forward(X))[1]
    )
    m.write_amica_output(tmp_path / "amicaout")
    assert calls == []
    assert (tmp_path / "amicaout" / "LLt").exists()


# --- torch-twin byte comparison (W layout, single-model Fortran parity) ----
def _torch_twin(model, sphere_np: np.ndarray):
    """A float64 AMICATorchNG holding ``model``'s exact fitted state, for a
    direct file-level export comparison. Mirrors
    ``test_mlx_sharing_cross_backend.py``'s ``_torch_twin`` builder."""
    import torch

    from pamica.torch_impl.core import AMICATorchNG

    ng = AMICATorchNG(
        n_channels=model.n_channels,
        n_models=model.n_models,
        n_mix=model.n_mix,
        device="cpu",
        dtype=torch.float64,
        block_size=model.block_size,
        seed=model.seed,
    )
    ng._initialize_parameters()
    for name in ("A", "mu", "alpha", "beta", "rho", "gm", "c"):
        value = np.array(getattr(model, name)).astype(np.float64)
        setattr(ng, name, torch.from_numpy(value).to(torch.float64))
    ng.comp_list = torch.from_numpy(np.array(model.comp_list).astype(np.int64))
    ng.sphere = torch.from_numpy(sphere_np.copy()).to(torch.float64)
    ng.mean = torch.from_numpy(np.array(model.mean).astype(np.float64)).to(
        torch.float64
    )
    ng.sldet = model.sldet
    ng._update_unmixing_matrices()
    ng.ll_history = list(model.ll_history)
    ng.final_ll_ = model.final_ll_
    ng.stop_reason = model.stop_reason
    assert model._llt_lht is not None and model._llt_lt is not None
    ng._llt_lht = model._llt_lht.astype(np.float64)
    ng._llt_lt = model._llt_lt.astype(np.float64)
    return ng


# Files whose bytes are exact-dtype-comparable once both sides are read back
# as float64 (comp_list is int32 on disk either way).
_FLOAT_FILES = (
    "gm",
    "W",
    "S",
    "mean",
    "c",
    "alpha",
    "mu",
    "sbeta",
    "rho",
    "LL",
    "A",
    "LLt",
)


@pytest.mark.parametrize("n_models", [1, 2])
def test_export_is_byte_compatible_with_a_torch_twin(real_data, tmp_path, n_models):
    m = _model(n_models=n_models, seed=42, keep_best=False)
    m.fit(real_data, max_iter=6, verbose=False)
    assert m._sphere_np is not None
    ng = _torch_twin(m, m._sphere_np)

    mdir, tdir = tmp_path / "mlx", tmp_path / "torch"
    m.write_amica_output(mdir)
    ng.write_amica_output(tdir)

    for name in _FLOAT_FILES:
        mp, tp = mdir / name, tdir / name
        assert mp.exists() and tp.exists(), f"{name} missing on one side"
        ma = np.fromfile(mp, dtype=np.float64)
        ta = np.fromfile(tp, dtype=np.float64)
        assert ma.shape == ta.shape, f"{name}: shape mismatch {ma.shape} vs {ta.shape}"
        # float32-native MLX vs a float64 torch twin recomputing W=inv(A)/S
        # from the same A/sphere: agreement to float32 precision, not bit-exact.
        max_diff = float(np.abs(ma - ta).max())
        assert max_diff < 1e-5, f"{name}: max abs diff {max_diff:.3e}"

    mc = np.fromfile(mdir / "comp_list", dtype=np.int32)
    tc = np.fromfile(tdir / "comp_list", dtype=np.int32)
    np.testing.assert_array_equal(mc, tc)

    # PR #311 review: pin the (n_models, n, n) -> (n, n, num_models) transpose
    # on a MEANINGFUL axis, not just aggregate flat bytes -- reshape the raw
    # W file back to write_amicaout's own on-disk contract (the writer does
    # ``np.asarray(W).transpose(2, 0, 1).ravel(order="C")``, so the inverse
    # is ``reshape(num_models, nw, nw).transpose(1, 2, 0)``) and diff each
    # model's (nw, nw) slice separately, so a per-model swap or a wrong axis
    # in the transpose shows up as a named model index, not just "W differs
    # somewhere". n_models=2 is what actually exercises a non-trivial model
    # axis; n_models=1 is the degenerate case where this collapses to the
    # aggregate check above.
    mw = np.fromfile(mdir / "W", dtype=np.float64).reshape(n_models, NW, NW)
    mw = mw.transpose(1, 2, 0)
    tw = np.fromfile(tdir / "W", dtype=np.float64).reshape(n_models, NW, NW)
    tw = tw.transpose(1, 2, 0)
    assert mw.shape == (NW, NW, n_models)
    for h in range(n_models):
        max_diff = float(np.abs(mw[:, :, h] - tw[:, :, h]).max())
        assert max_diff < 1e-5, f"model {h}: W max abs diff {max_diff:.3e}"
