"""Explicit PCA reduction (``pcakeep``/``pcadb``) across backends (issue #323,
epic #324 Phase 1).

The validation, the ``pcakeep``-over-``pcadb`` precedence and the "is a
reduction requested?" predicate are one shared decision (``pamica/rank.py``,
unit-tested in ``test_rank_policy.py``). This module is the anti-drift guard
required by ``.rules/backend_parity.md``. It fails if any backend applies that
decision differently, whether as a different kept rank, a different sphere, a
different upfront ``mir_step`` verdict, or a request that one backend accepts
and another rejects.

PyTorch and NumPy always run, so a divergence between them cannot land. MLX is
an optional Apple-Silicon backend: its checks skip individually (via
``pytest.importorskip`` plus an Apple-GPU guard) instead of skipping the
module, so the unconditional comparisons still run everywhere.

Real bundled sample EEG only (32 channels x 30504 frames), no synthetic data or
mocks (``.rules/testing.md``). Every fit is a few iterations: these are
preprocessing and gate tests, not convergence tests.
"""

import logging
import math
from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np
import pytest
import torch
from scipy.optimize import linear_sum_assignment

from pamica import AMICA, AMICA_NumPy
from pamica.torch_impl.core import AMICATorchNG
from pamica.torch_impl.utils import load_eeglab_data

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_data"
DATA_FILE = SAMPLE_DIR / "eeglab_data.fdt"
NW = 32
FIELD = 30504
NMIX = 3
SEED = 42

PCAKEEP = 20
# Chosen from the sample's covariance spectrum (population covariance, dB below
# the largest eigenvalue): the 10th eigenvalue sits at -17.21 dB and the 11th at
# -19.92 dB, one of the widest gaps in the spectrum, so pcadb=18.5 keeps
# exactly 10 dimensions with more than 1.2 dB of margin on either side.
PCADB = 18.5
PCADB_RANK = 10
# The bundled input.param/sample_params.json value. On its own it keeps 26 of
# the sample's 32 dimensions (test_rank_policy.py pins that), so it visibly
# changes the model unless pcakeep's precedence discards it.
BUNDLED_PCADB = 30.0

CONFIGS: Dict[str, dict] = {
    "pcakeep": {"pcakeep": PCAKEEP},
    "pcadb": {"pcadb": PCADB},
    "both": {"pcakeep": PCAKEEP, "pcadb": BUNDLED_PCADB},
}
EXPECTED_RANK = {"pcakeep": PCAKEEP, "pcadb": PCADB_RANK, "both": PCAKEEP}

# Torch and NumPy build the sphere from the same float64 covariance with
# different LAPACK eigensolvers (torch.linalg.eigh vs scipy.linalg.eigh), so
# agreement is to roundoff, not bit-exact. Measured on the sample: max abs
# difference 2.2e-14 (entries up to 0.12), max entrywise relative 2.0e-11.
SPHERE_RTOL = 1e-10
SPHERE_ATOL = 1e-13
# MLX stores the float32 cast of its float64 host sphere (_sphere_np), so the
# GPU copy agrees to float32 rounding (relative 2**-24 ~ 6e-8; measured max
# abs 3.6e-9) and the host copy to float64 roundoff (measured 3.5e-15).
MLX_F32_RTOL = 1e-6
MLX_F32_ATOL = 1e-12

pytestmark = pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing")


@pytest.fixture(scope="module")
def real_data() -> np.ndarray:
    return load_eeglab_data(str(DATA_FILE), data_dim=NW, field_dim=FIELD).astype(
        np.float64
    )


def _mlx_core():
    """The MLX backend module, or skip this test (never the whole module)."""
    mlx_core = pytest.importorskip(
        "pamica.mlx_impl.core", reason="MLX not installed (Apple Silicon only)"
    )
    mx = mlx_core.mx
    if mx.default_device().type != mx.DeviceType.gpu:
        pytest.skip("no Apple GPU")
    return mlx_core


def _torch(**kwargs) -> AMICATorchNG:
    return AMICATorchNG(
        n_channels=NW,
        n_mix=NMIX,
        seed=SEED,
        device="cpu",
        dtype=torch.float64,
        **kwargs,
    )


def _numpy(outdir: Path, max_iter: int = 2, **kwargs):
    return AMICA_NumPy(
        num_models=1,
        num_mix=NMIX,
        max_iter=max_iter,
        seed=SEED,
        use_tqdm=False,
        outdir=str(outdir),
        **kwargs,
    )


def _mlx(**kwargs):
    return _mlx_core().AMICAMLXNG(n_channels=NW, n_mix=NMIX, seed=SEED, **kwargs)


# A fitted backend reduced to what these tests compare: the kept source count,
# the input width, the float64 sphere and sldet (plus the model itself).
Fit = Dict[str, object]


def _fit_torch(X: np.ndarray, _tmp: Path, **kwargs) -> Fit:
    m = _torch(**kwargs)
    m.fit(X, max_iter=2, verbose=False)
    assert m.stop_reason not in AMICATorchNG._DEGENERATE_STOP_REASONS
    assert m.sphere is not None
    return {
        "n": m.n_channels,
        "n_in": m.n_channels_in,
        "sphere": m.sphere.cpu().numpy(),
        "sldet": m.sldet,
        "model": m,
    }


def _fit_numpy(X: np.ndarray, tmp: Path, **kwargs) -> Fit:
    m = _numpy(tmp, **kwargs)
    m.fit(X)
    assert m.converged is True
    return {
        "n": m.data_dim,
        "n_in": m.data_dim_in,
        "sphere": np.asarray(m.sphere),
        "sldet": m.sldet,
        "model": m,
    }


def _fit_mlx(X: np.ndarray, _tmp: Path, **kwargs) -> Fit:
    m = _mlx(**kwargs)
    m.fit(X, max_iter=2, verbose=False)
    assert m.stop_reason not in m._DEGENERATE_STOP_REASONS
    return {
        "n": m.n_channels,
        "n_in": m.n_channels_in,
        "sphere": np.array(m.sphere).astype(np.float64),
        "sphere_host": m._sphere_np,
        "sldet": m.sldet,
        "model": m,
    }


FITTERS: Dict[str, Callable[..., Fit]] = {
    "torch": _fit_torch,
    "numpy": _fit_numpy,
    "mlx": _fit_mlx,
}


@pytest.fixture(scope="module")
def fits(real_data, tmp_path_factory) -> Callable[[str, str], Fit]:
    """Fit each (backend, config) pair once per module and share it."""
    cache: Dict[Tuple[str, str], Fit] = {}

    def get(backend: str, config: str) -> Fit:
        if backend == "mlx":
            _mlx_core()  # skip before fitting, not after
        key = (backend, config)
        if key not in cache:
            tmp = tmp_path_factory.mktemp(f"{backend}_{config}")
            cache[key] = FITTERS[backend](real_data, tmp, **CONFIGS[config])
        return cache[key]

    return get


# --- (a)/(b): every backend keeps the same rank and builds the same sphere ---
@pytest.mark.parametrize("config", ["pcakeep", "pcadb"])
def test_torch_and_numpy_agree_on_the_reduced_sphere(fits, config):
    """Unconditional anti-drift pair: runs wherever the test suite runs."""
    rank = EXPECTED_RANK[config]
    t = fits("torch", config)
    n = fits("numpy", config)
    for fit in (t, n):
        assert fit["n"] == rank
        assert fit["n_in"] == NW
        assert fit["sphere"].shape == (rank, NW)
    np.testing.assert_allclose(
        n["sphere"], t["sphere"], rtol=SPHERE_RTOL, atol=SPHERE_ATOL
    )
    assert n["sldet"] == pytest.approx(t["sldet"], rel=1e-12)


@pytest.mark.parametrize("config", ["pcakeep", "pcadb"])
def test_mlx_agrees_with_torch_on_the_reduced_sphere(fits, config):
    rank = EXPECTED_RANK[config]
    t = fits("torch", config)
    m = fits("mlx", config)
    assert m["n"] == rank
    assert m["n_in"] == NW
    assert m["sphere"].shape == (rank, NW)
    # The GPU copy is float32; the host copy is the float64 computation.
    np.testing.assert_allclose(
        m["sphere"], t["sphere"], rtol=MLX_F32_RTOL, atol=MLX_F32_ATOL
    )
    np.testing.assert_allclose(
        m["sphere_host"], t["sphere"], rtol=SPHERE_RTOL, atol=SPHERE_ATOL
    )
    assert m["sldet"] == pytest.approx(t["sldet"], rel=1e-12)


# --- (c): pcakeep wins over pcadb, identically on every backend --------------
@pytest.mark.parametrize("backend", ["torch", "numpy", "mlx"])
def test_pcakeep_and_pcadb_together_equal_pcakeep_alone(fits, backend):
    """BUNDLED_PCADB alone would keep 26 dimensions; with pcakeep set it is
    ignored, so the sphere is bit-for-bit the pcakeep-only one."""
    both = fits(backend, "both")
    alone = fits(backend, "pcakeep")
    assert both["n"] == alone["n"] == PCAKEEP
    np.testing.assert_array_equal(both["sphere"], alone["sphere"])
    assert both["sldet"] == alone["sldet"]


@pytest.mark.parametrize("backend", ["torch", "numpy", "mlx"])
def test_precedence_is_logged_once_per_model(real_data, tmp_path, caplog, backend):
    """One INFO line at construction; the fit-time re-check in
    numerical_rank stays silent, so a fit does not repeat it."""
    X = real_data[:, :4096]
    with caplog.at_level(logging.INFO, logger="pamica.rank"):
        if backend == "torch":
            _torch(pcakeep=PCAKEEP, pcadb=BUNDLED_PCADB).fit(
                X, max_iter=1, verbose=False
            )
        elif backend == "numpy":
            _numpy(tmp_path, max_iter=1, pcakeep=PCAKEEP, pcadb=BUNDLED_PCADB).fit(X)
        else:
            _mlx(pcakeep=PCAKEEP, pcadb=BUNDLED_PCADB).fit(X, max_iter=1, verbose=False)
    notes = [r for r in caplog.records if "takes precedence" in r.getMessage()]
    assert len(notes) == 1, [r.getMessage() for r in notes]


@pytest.mark.parametrize("backend", ["torch", "numpy", "mlx"])
def test_do_sphere_false_ignores_the_request_with_one_warning(
    real_data, tmp_path, caplog, backend
):
    """No backend reduces without sphering (the reference's no-sphere branch
    keeps every dimension, numeigs = nx at amica15.f90:527), so pcakeep=20 is
    ignored: the fit is full width, and the constructor says so exactly once."""
    X = real_data[:, :4096]
    kwargs = {"pcakeep": PCAKEEP, "do_sphere": False}
    with caplog.at_level(logging.INFO, logger="pamica.rank"):
        if backend == "torch":
            m = _torch(**kwargs)
            m.fit(X, max_iter=1, verbose=False)
            assert m.sphere is not None
            n, sphere = m.n_channels, m.sphere.cpu().numpy()
        elif backend == "numpy":
            m = _numpy(tmp_path, max_iter=1, **kwargs)
            m.fit(X)
            n, sphere = m.data_dim, np.asarray(m.sphere)
        else:
            m = _mlx(**kwargs)
            m.fit(X, max_iter=1, verbose=False)
            n, sphere = m.n_channels, np.array(m.sphere)
    warnings = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and "do_sphere=False" in r.getMessage()
    ]
    assert len(warnings) == 1, [r.getMessage() for r in caplog.records]
    assert n == NW
    np.testing.assert_array_equal(sphere, np.eye(NW))


# --- (d): an invalid request fails at construction on every backend ----------
INVALID = [
    ({"pcakeep": -3}, "pcakeep"),
    ({"pcakeep": 0}, "pcakeep"),
    ({"pcakeep": 2.7}, "pcakeep"),
    ({"pcakeep": True}, "pcakeep"),
    ({"pcakeep": "20"}, "pcakeep"),
    ({"pcadb": 0.0}, "pcadb"),
    ({"pcadb": -5.0}, "pcadb"),
    ({"pcadb": math.nan}, "pcadb"),
    ({"pcadb": math.inf}, "pcadb"),
    ({"pcadb": True}, "pcadb"),
    ({"pcakeep": PCAKEEP, "pcadb": -5.0}, "pcadb"),
]


@pytest.mark.parametrize("backend", ["torch", "numpy", "mlx"])
@pytest.mark.parametrize(("kwargs", "name"), INVALID)
def test_invalid_request_raises_at_construction(tmp_path, backend, kwargs, name):
    """Before #323, torch and NumPy accepted every one of these silently:
    pcakeep=-3 fitted 29 sources, pcakeep=2.7 fitted 2, and pcakeep=0 or
    pcadb<=0 ran to a degenerate nan_ll fit."""
    with pytest.raises(ValueError, match=name):
        if backend == "torch":
            _torch(**kwargs)
        elif backend == "numpy":
            _numpy(tmp_path, **kwargs)
        else:
            _mlx(**kwargs)


# --- (e): the upfront mir_step gate gives the same verdict on torch and MLX --
REDUCING = [
    {"pcakeep": PCAKEEP},
    {"pcakeep": NW - 1},
    {"pcadb": PCADB},
    {"pcakeep": PCAKEEP, "pcadb": BUNDLED_PCADB},
]
NOT_REDUCING = [
    {"pcakeep": NW},
    {"pcakeep": 2 * NW},
    {"pcakeep": NW, "pcadb": BUNDLED_PCADB},  # the bundled input.param
    # Without sphering nothing is reduced, so neither is a request.
    {"pcakeep": PCAKEEP, "do_sphere": False},
    {"pcadb": PCADB, "do_sphere": False},
]


def _backend(name: str, **kwargs):
    return _torch(**kwargs) if name == "torch" else _mlx(**kwargs)


@pytest.mark.parametrize("backend", ["torch", "mlx"])
@pytest.mark.parametrize("kwargs", REDUCING)
def test_mir_step_rejects_a_reduction_request_up_front(real_data, backend, kwargs):
    m = _backend(backend, **kwargs)
    with pytest.raises(ValueError, match="Rejected up front"):
        m.fit(real_data[:, :4096], max_iter=2, verbose=False, mir_step=1)
    # Up front means before preprocessing: no sphere was built.
    assert m.sphere is None


@pytest.mark.parametrize("backend", ["torch", "mlx"])
@pytest.mark.parametrize("kwargs", NOT_REDUCING)
def test_mir_step_accepts_a_request_that_reduces_nothing(real_data, backend, kwargs):
    """pcakeep >= n_channels keeps every dimension (Fortran's min()), and
    do_sphere=False reduces nothing at all, so neither is a reduction request;
    the pre-#323 torch gate refused both anyway."""
    m = _backend(backend, **kwargs)
    m.fit(real_data[:, :4096], max_iter=2, verbose=False, mir_step=1)
    assert m.n_channels == m.n_channels_in == NW
    assert [row[0] for row in m.mir_history_] == [0, 1]
    assert all(math.isfinite(row[1]) for row in m.mir_history_)


# --- (f): the reduced fits find the same components --------------------------
def _matched_abs_corr(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Hungarian-matched |corr| between the rows (sources) of ``a`` and ``b``:
    best-|corr| matching is invariant to the per-component sign, scale and
    order. Same construction as ``torch_tests/test_amica_ng_wrapper.py``'s
    loadmodout round-trip test (none of the MLX cross-backend suites has a
    matching helper to reuse)."""
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    corr = np.abs(a @ b.T)
    rows, cols = linear_sum_assignment(1.0 - corr)
    return corr[rows, cols], cols


def test_torch_and_mlx_reduced_fits_find_the_same_components(real_data):
    """Same seed, pcakeep=20, 10 iterations: float64 torch vs float32 MLX.

    Threshold: the MLX cross-backend suites (``test_mlx_*_cross_backend.py``)
    compare fitted states at float32 tolerance and have no Hungarian-matched
    correlation bar, so this uses the strictest one in the suite, min matched
    |corr| > 0.999 (``torch_tests/test_amica_ng_wrapper.py``), well above the
    0.95 cross-implementation gate (``torch_tests/test_ng_backend.py::
    test_end_to_end_correlation_vs_fortran``). The log-likelihood bar is
    ``mlx_tests/test_mlx_backend.py::test_converged_ll_matches_torch_float32``'s
    1e-2. Measured: min matched |corr| 0.99999994, identity matching, LL
    difference 1.9e-7.
    """
    mlx_core = _mlx_core()
    t = _torch(pcakeep=PCAKEEP)
    t.fit(real_data, max_iter=10, verbose=False)
    m = mlx_core.AMICAMLXNG(n_channels=NW, n_mix=NMIX, seed=SEED, pcakeep=PCAKEEP)
    m.fit(real_data, max_iter=10, verbose=False)
    assert t.n_channels == m.n_channels == PCAKEEP

    matched, cols = _matched_abs_corr(t.transform(real_data), m.transform(real_data))
    assert matched.min() > 0.999, f"min matched |corr| {matched.min():.6f}"
    # Same seed, same sphere: the components come out in the same order.
    np.testing.assert_array_equal(cols, np.arange(PCAKEEP))
    assert t.final_ll_ is not None and m.final_ll_ is not None
    assert abs(t.final_ll_ - m.final_ll_) < 1e-2


# --- (g): persistence keeps the request and the reduced sphere --------------
def test_mlx_save_load_round_trips_the_reduction(real_data, tmp_path):
    """numpy scalars are accepted by the validator, so they are the case that
    exercises state_dict's cast to JSON-serializable int/float."""
    mlx_core = _mlx_core()
    m = mlx_core.AMICAMLXNG(
        n_channels=NW,
        n_mix=NMIX,
        seed=SEED,
        pcakeep=np.int64(PCAKEEP),
        pcadb=np.float64(BUNDLED_PCADB),
    )
    m.fit(real_data, max_iter=2, verbose=False)
    path = tmp_path / "model.npz"
    m.save(str(path))
    loaded = mlx_core.AMICAMLXNG.load(str(path))

    assert loaded.pcakeep == PCAKEEP and type(loaded.pcakeep) is int
    assert loaded.pcadb == BUNDLED_PCADB and type(loaded.pcadb) is float
    assert loaded.n_channels == PCAKEEP and loaded.n_channels_in == NW
    np.testing.assert_array_equal(np.array(loaded.sphere), np.array(m.sphere))
    np.testing.assert_array_equal(loaded.transform(real_data), m.transform(real_data))


def test_mlx_payload_without_the_keys_loads_with_defaults(real_data):
    """A payload written before #323 has no pcakeep/pcadb config keys; it must
    load with the constructor defaults (None), and its sphere still comes from
    params, so a reduced model stays reduced."""
    mlx_core = _mlx_core()
    m = mlx_core.AMICAMLXNG(n_channels=NW, n_mix=NMIX, seed=SEED, pcakeep=PCAKEEP)
    m.fit(real_data, max_iter=2, verbose=False)
    state = m.state_dict()
    del state["config"]["pcakeep"]
    del state["config"]["pcadb"]
    loaded = mlx_core.AMICAMLXNG.from_state_dict(state)

    assert loaded.pcakeep is None and loaded.pcadb is None
    assert loaded.n_channels == PCAKEEP and loaded.n_channels_in == NW
    np.testing.assert_array_equal(np.array(loaded.sphere), np.array(m.sphere))


def test_torch_save_load_round_trips_numpy_scalar_request(real_data, tmp_path):
    """AMICA.load reads with torch.load(weights_only=True), which refuses numpy
    scalars; state_dict persists the validated request as plain int/float."""
    model = AMICA(device="cpu", verbose=False)
    model.fit(
        real_data,
        max_iter=2,
        seed=SEED,
        pcakeep=np.int64(PCAKEEP),
        pcadb=np.float64(BUNDLED_PCADB),
    )
    path = tmp_path / "model.pt"
    model.save(str(path))
    loaded = AMICA.load(str(path), device="cpu")

    assert loaded.model_ is not None and model.model_ is not None
    assert loaded.model_.pcakeep == PCAKEEP and type(loaded.model_.pcakeep) is int
    assert loaded.model_.pcadb == BUNDLED_PCADB
    assert type(loaded.model_.pcadb) is float
    assert loaded.model_.sphere is not None and model.model_.sphere is not None
    assert tuple(loaded.model_.sphere.shape) == (PCAKEEP, NW)
    torch.testing.assert_close(
        loaded.model_.sphere, model.model_.sphere, rtol=0.0, atol=0.0
    )


# --- refits and restarts start from the input geometry -----------------------
# _preprocess shrinks n_channels to the kept rank. torch and MLX used to
# validate the next fit's X against that shrunk count, so a refit, or the
# second of n_restarts (which _fit_restarts deliberately does not catch), died
# with "X has 32 channels, model expects 20". NumPy re-derives its sizes from
# the data on every fit and is the always-on control.
AUTO_RANK = 20
BACKENDS = ["torch", "numpy", "mlx"]


@pytest.fixture(scope="module")
def rank_deficient(real_data) -> np.ndarray:
    """The real sample projected onto its leading 20 principal directions (the
    construction of test_rank_policy.py's fixture), so automatic mineig_rel
    detection reduces it to rank 20 with no explicit request."""
    x = real_data - real_data.mean(axis=1, keepdims=True)
    U = np.linalg.svd(x, full_matrices=False)[0][:, :AUTO_RANK]
    return U @ (U.T @ x)


def _reduction(real_data, rank_deficient, reduction: str):
    if reduction == "pcakeep":
        return real_data, {"pcakeep": PCAKEEP}
    return rank_deficient, {}


def _new(backend: str, tmp: Path, **kwargs):
    if backend == "torch":
        return _torch(**kwargs)
    if backend == "numpy":
        return _numpy(tmp, **kwargs)
    return _mlx(**kwargs)


def _fit(backend: str, m, X: np.ndarray) -> None:
    if backend == "numpy":
        m.fit(X)  # max_iter is a constructor argument there (2, see _numpy)
    else:
        m.fit(X, max_iter=2, verbose=False)


def _geometry(backend: str, m) -> Tuple[int, int]:
    if backend == "numpy":
        return m.data_dim, m.data_dim_in
    return m.n_channels, m.n_channels_in


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("reduction", ["pcakeep", "automatic"])
def test_refit_after_a_rank_reduction(
    real_data, rank_deficient, tmp_path, backend, reduction
):
    """A second fit on the same instance starts from the constructor's
    geometry, so it reduces to the same rank; on torch and MLX, which
    re-initialize every parameter per fit, it also reproduces the first fit.
    NumPy's refit warm-starts from the fitted A (_initialize_parameters keeps
    a non-None A), so only its geometry is compared."""
    X, kwargs = _reduction(real_data, rank_deficient, reduction)
    if backend == "mlx":
        _mlx_core()
    m = _new(backend, tmp_path, **kwargs)
    _fit(backend, m, X)
    assert _geometry(backend, m) == (PCAKEEP, NW)
    first_ll = None if backend == "numpy" else list(m.ll_history)

    _fit(backend, m, X)
    assert _geometry(backend, m) == (PCAKEEP, NW)
    if first_ll is not None:
        assert list(m.ll_history) == first_ll


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("reduction", ["pcakeep", "automatic"])
def test_restarts_survive_a_rank_reduction(
    real_data, rank_deficient, tmp_path, backend, reduction
):
    X, kwargs = _reduction(real_data, rank_deficient, reduction)
    if backend == "mlx":
        _mlx_core()
    seeds = [SEED, SEED + 1]
    m = _new(backend, tmp_path, n_restarts=2, restart_seeds=seeds, **kwargs)
    _fit(backend, m, X)
    assert _geometry(backend, m) == (PCAKEEP, NW)
    assert m.restart_seeds_ == seeds
    assert len(m.restart_lls_) == 2
    assert all(math.isfinite(ll) for ll in m.restart_lls_), m.restart_stop_reasons_


@pytest.mark.parametrize("backend", ["torch", "mlx"])
def test_channel_check_names_the_input_count(real_data, backend):
    """Before and after a reduced fit the model accepts, and names, the input
    width; the shrunk rank is never mistaken for it (a 20-channel X used to be
    accepted by a model fitted to 32 channels and reduced to 20)."""
    m = _backend(backend, pcakeep=PCAKEEP)
    assert m.n_channels_in == NW
    m.fit(real_data, max_iter=1, verbose=False)
    assert (m.n_channels, m.n_channels_in) == (PCAKEEP, NW)
    with pytest.raises(
        ValueError, match=f"X has {PCAKEEP} channels, model expects {NW}"
    ):
        m.fit(real_data[:PCAKEEP], max_iter=1, verbose=False)


@pytest.mark.parametrize("backend", ["torch", "mlx"])
def test_reloaded_reduced_model_transforms_and_refits(real_data, tmp_path, backend):
    """The input count is derived from the restored sphere's width, since
    config's n_channels is the fitted rank. NumPy is not covered: it persists
    an EEGLAB amicaout directory, not a model state it can load and refit."""
    X = real_data
    if backend == "torch":
        m = _torch(pcakeep=PCAKEEP)
        m.fit(X, max_iter=2, verbose=False)
        path = tmp_path / "state.pt"
        torch.save(m.state_dict(), path)
        loaded = AMICATorchNG.from_state_dict(
            torch.load(path, weights_only=True), device="cpu"
        )
    else:
        mlx_core = _mlx_core()
        m = _mlx(pcakeep=PCAKEEP)
        m.fit(X, max_iter=2, verbose=False)
        path = tmp_path / "model.npz"
        m.save(str(path))
        loaded = mlx_core.AMICAMLXNG.load(str(path))

    assert (loaded.n_channels, loaded.n_channels_in) == (PCAKEEP, NW)
    np.testing.assert_array_equal(loaded.transform(X), m.transform(X))
    loaded.fit(X, max_iter=2, verbose=False)
    assert (loaded.n_channels, loaded.n_channels_in) == (PCAKEEP, NW)
    assert loaded.final_ll_ is not None and math.isfinite(loaded.final_ll_)
