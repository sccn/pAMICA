"""The ``AMICA`` wrapper end to end on every backend it can build (issue #313).

``AMICA(backend="torch")`` (the default) builds ``AMICATorchNG`` and
``AMICA(backend="mlx")`` builds ``AMICAMLXNG``; every wrapper method must work
on both. PyTorch always runs, so a torch regression cannot land; the MLX tests
skip individually (``pytest.importorskip`` plus an Apple-GPU guard) instead of
skipping the module, and compare MLX against PyTorch where the two should agree
(``.rules/backend_parity.md``).

Real bundled sample EEG only (32 channels x 30504 frames), no synthetic data or
mocks (``.rules/testing.md``). Every fit is a few iterations: these are wiring,
persistence and agreement tests, not convergence tests.
"""

import inspect
import logging
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from pamica import AMICA
from pamica.fortran_params import read_params_file
from pamica.torch_impl.core import AMICATorchNG
from pamica.torch_impl.utils import load_eeglab_data

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_data"
DATA_FILE = SAMPLE_DIR / "eeglab_data.fdt"
PARAM_FILE = SAMPLE_DIR / "input.param"
NW = 32
FIELD = 30504
SEED = 42
PCAKEEP = 20
N_FRAMES = 8192  # enough for a stable few-iteration fit, cheap on every backend
MAX_ITER = 3
BACKENDS = ["torch", "mlx"]

pytestmark = pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing")


@pytest.fixture(scope="module")
def real_data() -> np.ndarray:
    return load_eeglab_data(str(DATA_FILE), data_dim=NW, field_dim=FIELD).astype(
        np.float64
    )


@pytest.fixture(scope="module")
def X(real_data) -> np.ndarray:
    return real_data[:, :N_FRAMES]


def _require_mlx():
    """The MLX backend class, or skip this test (never the whole module)."""
    mlx_core = pytest.importorskip(
        "pamica.mlx_impl.core", reason="MLX not installed (Apple Silicon only)"
    )
    mx = mlx_core.mx
    if mx.default_device().type != mx.DeviceType.gpu:
        pytest.skip("no Apple GPU")
    return mlx_core.AMICAMLXNG


def _backend_class(backend: str) -> type:
    return AMICATorchNG if backend == "torch" else _require_mlx()


def _wrapper(backend: str, **kwargs) -> AMICA:
    """An unfitted wrapper; torch pinned to the CPU so its fits are
    deterministic (MLX has no device choice)."""
    if backend == "torch":
        kwargs.setdefault("device", "cpu")
    else:
        _require_mlx()
    return AMICA(verbose=False, backend=backend, **kwargs)


@pytest.fixture(scope="module")
def fitted(X):
    """One few-iteration wrapper fit per backend, shared across the module."""
    cache: dict = {}

    def get(backend: str) -> AMICA:
        if backend not in cache:
            model = _wrapper(backend)
            model.fit(X, max_iter=MAX_ITER, seed=SEED)
            cache[backend] = model
        return cache[backend]

    return get


# --- pin: a format_version 1 save (written before issue #313) still loads -----
def _v1_payload(model: AMICA) -> dict:
    """The payload ``AMICA.save`` wrote before issue #313, key for key.

    Version 1 carried no backend name: the wrapper could only build
    ``AMICATorchNG``, so every version 1 file holds a torch state dict.
    """
    assert model.model_ is not None
    return {
        "format_version": 1,
        "wrapper": {
            "n_models": model.n_models,
            "n_mix": model.n_mix,
            "verbose": model.verbose,
        },
        "backend": model.model_.state_dict(),
    }


def test_v1_payload_still_loads_and_transforms_identically(X, tmp_path):
    model = AMICA(device="cpu", verbose=False)
    model.fit(X, max_iter=MAX_ITER, block_size=1024, seed=SEED)
    path = tmp_path / "v1.pt"
    torch.save(_v1_payload(model), path)

    loaded = AMICA.load(str(path), device="cpu")

    assert type(loaded.model_) is AMICATorchNG
    assert loaded.is_fitted_ and loaded.converged_
    assert loaded.stop_reason_ == model.stop_reason_
    assert loaded.final_ll_ == model.final_ll_
    assert loaded.ll_history_ == model.ll_history_
    np.testing.assert_array_equal(loaded.transform(X), model.transform(X))


# --- backend selection and its errors --------------------------------------------
def test_unknown_backend_name_lists_the_valid_ones():
    with pytest.raises(ValueError, match="backend must be one of 'torch', 'mlx'"):
        AMICA(backend="jax")


def test_device_with_the_mlx_backend_raises_at_construction():
    """Checked before MLX's availability, so it raises the same everywhere."""
    with pytest.raises(
        ValueError, match="device='cpu' applies only to backend='torch'"
    ):
        AMICA(backend="mlx", device="cpu")


def test_mlx_backend_without_mlx_raises_import_error(monkeypatch):
    """``sys.modules[name] = None`` is Python's own marker for a module that
    cannot be imported: ``import mlx`` then raises ImportError and
    ``importlib.util.find_spec("mlx")`` returns None, exactly as on a machine
    without MLX (every CI runner but the macOS one). The wrapper's real
    availability check runs unmodified."""
    monkeypatch.setitem(sys.modules, "mlx", None)
    with pytest.raises(ImportError, match=r"uv pip install mlx.*`mlx` extra"):
        AMICA(backend="mlx")
    # The PyTorch default never looks for MLX.
    assert AMICA().backend == "torch"


def test_import_pamica_never_imports_mlx():
    """Neither ``import pamica`` nor building the wrapper (either backend)
    imports MLX; only fitting an MLX model does. A fresh interpreter, since
    this process's other tests may already have imported it."""
    code = (
        "import importlib.util, sys\n"
        "from pamica import AMICA\n"
        "AMICA()\n"
        "if importlib.util.find_spec('mlx') is not None:\n"
        "    AMICA(backend='mlx')\n"
        "leaked = sorted(m for m in sys.modules if m == 'mlx' or m.startswith('mlx.'))\n"
        "assert not leaked, leaked\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_dtype_with_the_mlx_backend_raises_before_fitting(X):
    model = _wrapper("mlx")
    with pytest.raises(ValueError, match=r"\['dtype'\] apply only to backend='torch'"):
        model.fit(X, max_iter=1, dtype=torch.float32)
    assert model.model_ is None  # nothing was built


@pytest.mark.parametrize("backend", BACKENDS)
def test_unknown_fit_keyword_names_the_backend(X, backend):
    model = _wrapper(backend)
    name = _backend_class(backend).__name__
    with pytest.raises(TypeError, match=rf"\['blocksize'\].*{name}"):
        model.fit(X, max_iter=1, blocksize=512)


def test_backend_attribute_is_rechecked_at_fit(X):
    """A backend changed after construction goes through the same checks."""
    model = AMICA(verbose=False)
    model.backend = "jax"
    with pytest.raises(ValueError, match="backend must be one of"):
        model.fit(X, max_iter=1)


# --- the fitted surface, both backends -------------------------------------------
@pytest.mark.parametrize("backend", BACKENDS)
def test_fit_publishes_the_backend_record(fitted, backend):
    model = fitted(backend)
    backend_model = model.model_
    assert type(backend_model) is _backend_class(backend)
    assert model.is_fitted_ is True and model.converged_ is True
    assert model.stop_reason_ == backend_model.stop_reason == "max_iter"
    assert model.ll_history_ == backend_model.ll_history
    assert len(model.ll_history_) == MAX_ITER
    assert model.final_ll_ == backend_model.final_ll_
    assert model.restart_seeds_ == [SEED]
    assert model.restart_lls_ == [model.final_ll_]
    assert model.restart_stop_reasons_ == ["max_iter"]
    assert model.mir_history_ == []


@pytest.mark.parametrize("backend", BACKENDS)
def test_every_delegating_method_works(fitted, X, backend, tmp_path):
    """Each wrapper method returns the backend's own answer, shaped as
    documented, on both backends."""
    model = fitted(backend)
    b = model.model_
    assert b is not None
    np.testing.assert_array_equal(model.transform(X), b.transform(X))
    assert model.transform(X).shape == (NW, N_FRAMES)
    for name in (
        "get_mixing_matrix",
        "get_unmixing_matrix",
        "get_sensor_mixing_matrix",
        "get_pdftype",
        "get_rho",
        "get_model_center",
        "variance_order",
    ):
        np.testing.assert_array_equal(
            getattr(model, name)(), getattr(b, name)(), err_msg=name
        )
    np.testing.assert_array_equal(model.get_sphere(), b.get_sphere())
    np.testing.assert_array_equal(model.get_mean(), b.get_mean())
    assert model.get_sensor_mixing_matrix().shape == (NW, NW)
    assert model.shared_components() == []
    np.testing.assert_array_equal(model.model_loglik(X), b.model_loglik(X))
    np.testing.assert_array_equal(model.model_probability(X), np.ones((1, N_FRAMES)))
    mir_nats, _ = model.mir(X)
    assert mir_nats == b.mir(X)[0] and np.isfinite(mir_nats)
    np.testing.assert_array_equal(model.pmi(X), b.pmi(X))
    model.write_amica_output(str(tmp_path / "amicaout"))
    assert (tmp_path / "amicaout" / "W").exists()


@pytest.mark.parametrize("backend", BACKENDS)
def test_unfitted_accessors_raise_not_fitted(backend):
    model = _wrapper(backend)
    for accessor in (
        model.get_sphere,
        model.get_mean,
        model.get_model_center,
        model.get_sensor_mixing_matrix,
    ):
        with pytest.raises(ValueError, match="fitted"):
            accessor()


def test_degenerate_mlx_fit_is_refused_like_torch(real_data):
    """The #50 contract on MLX: a NaN in real EEG drives a genuine degenerate
    stop, which the wrapper classifies with AMICAMLXNG's own
    ``_DEGENERATE_STOP_REASONS`` and refuses to use."""
    AMICAMLXNG = _require_mlx()
    bad = real_data[:, :4096].copy()
    bad[0, 0] = np.nan
    model = _wrapper("mlx")
    model.fit(bad, max_iter=3, seed=0)
    assert model.stop_reason_ in AMICAMLXNG._DEGENERATE_STOP_REASONS
    assert model.converged_ is False and model.is_fitted_ is False
    for action in (model.transform, model.model_loglik):
        with pytest.raises(RuntimeError, match="degenerate"):
            action(real_data[:, :512])
    for accessor in (model.get_sphere, model.get_mean, model.get_model_center):
        with pytest.raises(RuntimeError, match="degenerate"):
            accessor()


# --- from_params_file on MLX (issue #304's MLX path) ---------------------------------
def _not_applied_keys(caplog) -> tuple[str, list]:
    """The backend named in, and the keys listed by, fit()'s one "not
    applied" warning."""
    messages = [
        r.getMessage() for r in caplog.records if "NOT applied" in r.getMessage()
    ]
    assert len(messages) == 1, messages
    match = re.search(r"fit\(\)/(\w+) parameter.*: \[(.*)\]$", messages[0])
    assert match is not None, messages[0]
    keys = [k.strip().strip("'") for k in match.group(2).split(",")]
    return match.group(1), keys


def test_params_file_drives_the_mlx_backend(X, caplog):
    AMICAMLXNG = _require_mlx()
    model = AMICA.from_params_file(str(PARAM_FILE), backend="mlx", verbose=False)
    assert model.backend == "mlx"
    with caplog.at_level(logging.WARNING, logger="pamica.amica"):
        model.fit(X, max_iter=3, seed=SEED)

    b = model.model_
    assert b is not None and type(b) is AMICAMLXNG
    # The file's values, chosen where they differ from AMICAMLXNG's defaults.
    assert b.pcakeep == 32 and b.pcadb == 30.0
    assert b.block_size == 512
    assert b.do_newton is True and b.newt_start == 50 and b.newtrate0 == 1.0
    assert b.lrate0 == 0.05 and b.minlrate == 1e-8
    assert b.maxdecs == 3 and b.rholratefact == 0.5 and b.invsigmax == 100.0
    assert b.mineig == 1e-12
    assert b.n_channels == NW  # pcakeep 32 on 32 channels reduces nothing
    assert model.is_fitted_

    backend_name, keys = _not_applied_keys(caplog)
    assert backend_name == "AMICAMLXNG"
    mlx_params = set(inspect.signature(AMICAMLXNG).parameters)
    assert keys and not set(keys) & mlx_params, keys
    # Every file key MLX lacks is named: the partition is AMICAMLXNG's own.
    file_keys = set(read_params_file(PARAM_FILE))
    fit_named = {"max_iter", "lrate", "do_mean", "do_sphere", "do_newton"}
    expected = file_keys - mlx_params - fit_named - {"num_models", "num_mix"}
    assert set(keys) == expected


def test_params_file_warning_names_the_torch_backend(X, caplog):
    """The same file on the default backend: partitioned by AMICATorchNG."""
    model = AMICA.from_params_file(str(PARAM_FILE), device="cpu", verbose=False)
    with caplog.at_level(logging.WARNING, logger="pamica.amica"):
        model.fit(X, max_iter=1, seed=SEED)
    backend_name, keys = _not_applied_keys(caplog)
    assert backend_name == "AMICATorchNG"
    assert not set(keys) & set(inspect.signature(AMICATorchNG).parameters)


# --- pcakeep through the wrapper on MLX (#323's remaining bullet) ---------------------
def test_pcakeep_through_the_mlx_wrapper(X):
    _require_mlx()
    model = _wrapper("mlx")
    model.fit(X, max_iter=MAX_ITER, seed=SEED, pcakeep=PCAKEEP)
    assert model.model_ is not None and model.model_.n_channels == PCAKEEP
    assert model.transform(X).shape == (PCAKEEP, N_FRAMES)
    assert model.get_sensor_mixing_matrix().shape == (NW, PCAKEEP)
    assert model.get_sphere().shape == (PCAKEEP, NW)
    assert model.get_model_center().shape == (PCAKEEP,)


def test_mir_step_gate_through_the_mlx_wrapper(X):
    """The upfront gate rejects a real reduction request and accepts the
    bundled files' pcakeep=32 on 32 channels, which reduces nothing."""
    _require_mlx()
    model = _wrapper("mlx")
    with pytest.raises(ValueError, match="Rejected up front"):
        model.fit(X, max_iter=2, seed=SEED, pcakeep=PCAKEEP, mir_step=1)
    assert model.model_ is None  # a failed fit publishes nothing
    model.fit(X, max_iter=2, seed=SEED, pcakeep=NW, mir_step=1)
    assert [row[0] for row in model.mir_history_] == [0, 1]
    assert all(np.isfinite(row[1]) for row in model.mir_history_)
