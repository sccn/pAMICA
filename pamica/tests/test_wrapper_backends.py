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
from typing import Any

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
JSON_FILE = SAMPLE_DIR / "sample_params.json"
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


def _backend_class(backend: str) -> Any:
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
    assert loaded.backend == "torch"
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
    message = rf"\['blocksize'\]: neither a fit\(\) parameter nor a constructor keyword of {name}\."
    with pytest.raises(TypeError, match=message):
        model.fit(X, max_iter=1, blocksize=512)


def test_torch_only_and_unknown_together_name_both(X):
    """PR #347 review item 4: a first version of this check raised on the
    torch-only category alone, so ``fit(dtype=..., blocksize=...)`` with
    ``backend='mlx'`` named only ``dtype`` and silently dropped
    ``blocksize`` from the message -- the same drop the NumPy backend's own
    kwarg check had (issue #346). Both categories are now named in ONE
    ``TypeError``, and the two single-category messages
    (``test_dtype_with_the_mlx_backend_raises_before_fitting``,
    ``test_unknown_fit_keyword_names_the_backend``) are unchanged."""
    model = _wrapper("mlx")
    with pytest.raises(TypeError) as exc:
        model.fit(X, max_iter=1, dtype=torch.float32, blocksize=512)
    message = str(exc.value)
    assert "dtype" in message
    assert "apply only to backend='torch'" in message
    assert "blocksize" in message
    assert "neither a fit() parameter nor a constructor keyword" in message
    assert model.model_ is None  # nothing was built


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

    # fit_transform on a fresh wrapper: the fit it runs is this fixture's, and
    # it returns that model's own transform of the same data.
    fresh = _wrapper(backend)
    sources = fresh.fit_transform(X, max_iter=MAX_ITER, seed=SEED)
    assert type(fresh.model_) is type(b) and fresh.is_fitted_
    np.testing.assert_array_equal(sources, fresh.transform(X))
    reference = model.transform(X)
    if backend == "torch":
        np.testing.assert_array_equal(sources, reference)  # CPU: bit for bit
    else:
        assert np.abs(sources - reference).max() <= _float32_tol(reference)


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


@pytest.mark.parametrize("backend", BACKENDS)
def test_degenerate_fit_is_refused_on_both_backends(real_data, backend):
    """The #50 contract: a NaN in real EEG drives a genuine degenerate stop
    (torch ``nan_ll``, MLX ``nan_params``), which the wrapper classifies with
    the backend class's own ``_DEGENERATE_STOP_REASONS`` and refuses to use."""
    backend_cls = _backend_class(backend)
    bad = real_data[:, :4096].copy()
    bad[0, 0] = np.nan
    model = _wrapper(backend)
    model.fit(bad, max_iter=3, seed=0)
    assert model.stop_reason_ in backend_cls._DEGENERATE_STOP_REASONS
    assert model.converged_ is False and model.is_fitted_ is False
    assert repr(model) == (
        f"<AMICA (degenerate fit, stop_reason={model.stop_reason_!r}, "
        f"backend={backend!r}, n_models=1, n_mix=3)>"
    )
    for action in (model.transform, model.model_loglik):
        with pytest.raises(RuntimeError, match="degenerate"):
            action(real_data[:, :512])
    for accessor in (model.get_sphere, model.get_mean, model.get_model_center):
        with pytest.raises(RuntimeError, match="degenerate"):
            accessor()


@pytest.mark.parametrize("backend", BACKENDS)
def test_repr_names_the_backend_and_state(fitted, backend):
    assert repr(_wrapper(backend, n_models=2, n_mix=4)) == (
        f"<AMICA (unfitted, backend={backend!r}, n_models=2, n_mix=4)>"
    )
    assert repr(fitted(backend)) == (
        f"<AMICA (fitted: {NW} sources, backend={backend!r}, n_models=1, n_mix=3)>"
    )


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


def test_json_params_file_drives_the_mlx_backend(X, caplog):
    """pamica's JSON schema through the MLX backend, including the JSON-only
    spellings the shared reader translates (max_decs -> maxdecs,
    min_grad_norm -> min_nd, share_int -> share_iter, issue #304)."""
    AMICAMLXNG = _require_mlx()
    model = AMICA.from_params_file(str(JSON_FILE), backend="mlx", verbose=False)
    assert model.backend == "mlx"
    with caplog.at_level(logging.WARNING, logger="pamica.amica"):
        model.fit(X, max_iter=3, seed=SEED)

    b = model.model_
    assert b is not None and type(b) is AMICAMLXNG
    assert b.pcakeep == 32 and b.pcadb == 30.0 and b.block_size == 512
    assert b.do_newton is True and b.newt_start == 50 and b.newtrate0 == 1.0
    assert b.maxdecs == 3 and b.min_nd == 1e-7 and b.share_iter == 100
    assert b.maxrej == 3 and b.rholratefact == 0.5 and b.invsigmax == 100.0
    assert model.is_fitted_

    backend_name, keys = _not_applied_keys(caplog)
    assert backend_name == "AMICAMLXNG"
    mlx_params = set(inspect.signature(AMICAMLXNG).parameters)
    assert keys and not set(keys) & mlx_params, keys
    assert {"files", "outdir", "data_dim", "num_comps"} <= set(keys)
    for translated in ("max_decs", "min_grad_norm", "share_int"):
        assert translated not in keys


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


# --- save/load: format_version 2 records the backend -----------------------------
def _float32_tol(reference: np.ndarray) -> float:
    """Normwise float32 tolerance: a few units of float32 rounding (2**-24
    relative) on the largest entry."""
    return 8 * 2.0**-24 * float(np.abs(reference).max())


@pytest.mark.parametrize("backend", BACKENDS)
def test_save_load_round_trip(fitted, X, backend, tmp_path):
    """fit -> transform -> save -> load -> transform, bit for bit on both
    backends: the loaded parameters are the saved ones exactly, and each
    backend's transform is deterministic on the same parameters."""
    model = fitted(backend)
    path = tmp_path / f"{backend}.pt"
    model.save(str(path))

    payload = torch.load(path, weights_only=True)
    assert payload["format_version"] == 2
    assert payload["wrapper"]["backend"] == backend
    assert all(
        isinstance(t, torch.Tensor) for t in payload["backend"]["params"].values()
    )

    loaded = AMICA.load(str(path))
    assert loaded.backend == backend
    assert type(loaded.model_) is _backend_class(backend)
    assert loaded.is_fitted_ is True and loaded.converged_ is True
    assert loaded.stop_reason_ == model.stop_reason_
    assert loaded.final_ll_ == model.final_ll_
    assert loaded.ll_history_ == model.ll_history_
    assert loaded.restart_seeds_ == model.restart_seeds_
    assert loaded.restart_lls_ == model.restart_lls_
    assert loaded.restart_stop_reasons_ == model.restart_stop_reasons_
    np.testing.assert_array_equal(loaded.transform(X), model.transform(X))
    for name in (
        "get_mixing_matrix",
        "get_unmixing_matrix",
        "get_mean",
        "get_model_center",
    ):
        np.testing.assert_array_equal(
            getattr(loaded, name)(), getattr(model, name)(), err_msg=name
        )
    if backend == "torch":
        np.testing.assert_array_equal(loaded.get_sphere(), model.get_sphere())
    else:
        # Only MLX's float32 sphere is persisted: a reloaded model's float64
        # host copy is that sphere upcast, not the fit's float64 original.
        mlx_sphere = np.array(model.model_.sphere, dtype=np.float64)
        np.testing.assert_array_equal(loaded.get_sphere(), mlx_sphere)
        original = model.get_sphere()
        assert np.abs(loaded.get_sphere() - original).max() <= _float32_tol(original)


def test_mlx_params_round_trip_with_their_dtypes(fitted, tmp_path):
    """Every MLX param comes back with its original dtype and values
    (float32, and the integer comp_list/pdtype)."""
    model = fitted("mlx")
    path = tmp_path / "mlx.pt"
    model.save(str(path))
    loaded = AMICA.load(str(path))
    assert model.model_ is not None and loaded.model_ is not None
    before = model.model_.state_dict()
    after = loaded.model_.state_dict()
    assert {a.dtype for a in before["params"].values()} >= {
        np.dtype(np.float32),
        np.dtype(np.int64),
        np.dtype(np.int32),
    }
    for name, array in before["params"].items():
        assert after["params"][name].dtype == array.dtype, name
        np.testing.assert_array_equal(after["params"][name], array, err_msg=name)
    assert after["config"] == before["config"]
    assert after["extra"] == before["extra"]


def test_mlx_bool_array_survives_the_payload_conversion(fitted, tmp_path):
    """No MLX param is boolean today; this pins the conversion for one anyway,
    using MLX's own real boolean mask (comp_used), through a real
    torch.save/torch.load(weights_only=True) round trip."""
    from pamica.amica import _mlx_state_from_payload, _state_to_payload

    b = fitted("mlx").model_
    state = b.state_dict()
    comp_used = np.array(b.comp_used)
    assert comp_used.dtype == np.bool_
    state["params"]["comp_used"] = comp_used
    path = tmp_path / "state.pt"
    torch.save(_state_to_payload(state), path)
    restored = _mlx_state_from_payload(torch.load(path, weights_only=True), str(path))
    for name, array in state["params"].items():
        assert restored["params"][name].dtype == array.dtype, name
        np.testing.assert_array_equal(restored["params"][name], array, err_msg=name)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("config", "restart_seeds", {SEED, SEED + 1}),  # a set
        ("extra", "good_idx", np.arange(4)),  # a bare array outside params
    ],
)
def test_unsavable_state_value_names_its_key_path(fitted, section, key, value):
    """A value torch.load(weights_only=True) could not read back fails at
    save time, naming where it sits, instead of writing an unloadable file.
    The state is a real fitted torch model's; one entry is replaced."""
    from pamica.amica import _state_to_payload

    b = fitted("torch").model_
    assert b is not None
    state = b.state_dict()
    state[section][key] = value
    path = re.escape(f"state[{section!r}][{key!r}]")
    with pytest.raises(TypeError, match=rf"cannot write {path} = .*weights_only"):
        _state_to_payload(state)


def test_weights_only_safe_converts_scalars_and_names_nested_paths():
    """Called directly: numpy scalars become Python numbers of the matching
    type, and a rejected value deep in a list is named down to its index."""
    from pamica.amica import _weights_only_safe

    converted = _weights_only_safe(
        {"seed": np.int64(3), "tol": np.float32(0.5), "flag": np.bool_(True)},
        "state['config']",
    )
    assert converted == {"seed": 3, "tol": 0.5, "flag": True}
    assert [type(v) for v in converted.values()] == [int, float, bool]
    path = re.escape("state['extra']['restart_seeds_'][1] = {3} (set)")
    with pytest.raises(TypeError, match=path):
        _weights_only_safe({"restart_seeds_": [0, {3}]}, "state['extra']")


@pytest.mark.parametrize("backend", BACKENDS)
def test_numpy_scalar_settings_save_and_load(X, backend, tmp_path):
    """A numpy-integer seed lands in the backend's config and fit record,
    which the weights_only unpickler refuses; save stores it as a Python int
    so the file loads."""
    model = _wrapper(backend)
    model.fit(X[:, :4096], max_iter=2, seed=np.int64(SEED))
    path = tmp_path / "np_seed.pt"
    model.save(str(path))
    loaded = AMICA.load(str(path))
    assert loaded.model_ is not None
    assert loaded.model_.seed == SEED and type(loaded.model_.seed) is int
    assert loaded.restart_seeds_ == [SEED]
    np.testing.assert_array_equal(loaded.transform(X), model.transform(X))


def test_v2_payload_without_a_backend_is_malformed(fitted, tmp_path):
    path = tmp_path / "model.pt"
    fitted("torch").save(str(path))
    payload = torch.load(path, weights_only=True)
    del payload["wrapper"]["backend"]
    torch.save(payload, path)
    with pytest.raises(ValueError, match=r"malformed.*wrapper\['backend'\] is None"):
        AMICA.load(str(path))


def test_unknown_format_version_names_the_readable_ones(fitted, tmp_path):
    path = tmp_path / "model.pt"
    fitted("torch").save(str(path))
    payload = torch.load(path, weights_only=True)
    payload["format_version"] = 3
    torch.save(payload, path)
    with pytest.raises(
        ValueError, match=r"format_version: 3 \(expected one of \[1, 2\]\)"
    ):
        AMICA.load(str(path))


def _labeled_mlx_payload(fitted, tmp_path) -> Path:
    """A real torch-fitted save relabeled ``wrapper["backend"] = "mlx"``.

    Both checks below fire on the recorded backend name before any parameter
    is read, so the relabeled file exercises them on every machine, MLX or not.
    """
    path = tmp_path / "labeled_mlx.pt"
    fitted("torch").save(str(path))
    payload = torch.load(path, weights_only=True)
    payload["wrapper"]["backend"] = "mlx"
    torch.save(payload, path)
    return path


def test_mlx_payload_with_a_device_raises(fitted, tmp_path):
    path = _labeled_mlx_payload(fitted, tmp_path)
    with pytest.raises(ValueError, match="holds an MLX-backend model.*device=None"):
        AMICA.load(str(path), device="cpu")


def test_mlx_payload_without_mlx_names_the_backend(fitted, tmp_path, monkeypatch):
    """Written on a machine with MLX, opened on one without (the
    ``sys.modules`` marker, as in the construction test above)."""
    path = _labeled_mlx_payload(fitted, tmp_path)
    monkeypatch.setitem(sys.modules, "mlx", None)
    with pytest.raises(ImportError, match="holds an MLX-backend model.*not installed"):
        AMICA.load(str(path))


# --- EEGLAB export from an MLX fit --------------------------------------------------
@pytest.mark.parametrize("pcakeep", [None, PCAKEEP])
def test_mlx_eeglab_export_reloads_through_loadmodout(X, tmp_path, pcakeep):
    """The wrapper's write_amica_output on an MLX fit is what EEGLAB reads:
    shapes as loadmodout15 expects them (a reduced sphere padded to the
    Fortran nx x nx record), the written sphere and mean exactly the model's,
    and the reloaded sources and maps the live ones up to loadmodout's
    variance order and per-component scale."""
    from pamica.numpy_impl.load import loadmodout

    model = _wrapper("mlx")
    model.fit(X, max_iter=MAX_ITER, seed=SEED, pcakeep=pcakeep)
    n = NW if pcakeep is None else pcakeep
    outdir = tmp_path / "amicaout"
    model.write_amica_output(str(outdir))
    out = loadmodout(outdir)

    assert (out.num_pcs, out.data_dim, out.num_models) == (n, NW, 1)
    assert out.W.shape == (n, n, 1)
    assert out.S.shape == (NW, NW)
    assert out.A.shape == (NW, n, 1)
    assert model.model_ is not None
    np.testing.assert_array_equal(
        out.S[:n], np.array(model.model_.sphere, dtype=np.float64)
    )
    np.testing.assert_array_equal(out.S[n:], 0.0)
    np.testing.assert_array_equal(np.ravel(out.data_mean), model.get_mean())
    np.testing.assert_array_equal(out.LL, model.ll_history_)

    # loadmodout reorders by back-projected variance (origord) and rescales
    # each component; within that, sources and sensor maps must be the live
    # ones to float32 tolerance (measured 2.8e-7 / 9.4e-8 full rank).
    order = np.asarray(out.origord).ravel()
    assert sorted(order.tolist()) == list(range(n))
    for loaded, live in (
        (out.sources(X), model.transform(X).astype(np.float64)[order]),
        (out.A[:, :, 0].T, model.get_sensor_mixing_matrix()[:, order].T),
    ):
        scale = (loaded * live).sum(axis=1) / (live**2).sum(axis=1)
        residual = loaded - scale[:, None] * live
        assert np.linalg.norm(residual) <= 1e-5 * np.linalg.norm(loaded)


# --- torch and MLX through the wrapper agree ------------------------------------------
@pytest.mark.parametrize("pcakeep", [None, PCAKEEP])
def test_torch_and_mlx_wrappers_find_the_same_components(real_data, pcakeep):
    """Same data, seed and configuration through AMICA(backend=...): the same
    number of components, the same sources (Hungarian-matched |corr| at the
    bar of test_pca_reduction_cross_backend.py, which this reuses), and the
    same log-likelihood to MLX's float32 bar. Measured on the full sample, 10
    iterations: min matched |corr| 0.99999996 full rank / 0.99999993 at
    pcakeep=20, identity matching, LL difference 1.3e-6 / 4.8e-6."""
    from pamica.tests.test_pca_reduction_cross_backend import _matched_abs_corr

    kwargs = {} if pcakeep is None else {"pcakeep": pcakeep}
    t = _wrapper("torch")
    t.fit(real_data, max_iter=10, seed=SEED, **kwargs)
    m = _wrapper("mlx")
    m.fit(real_data, max_iter=10, seed=SEED, **kwargs)

    n = NW if pcakeep is None else pcakeep
    assert t.model_ is not None and m.model_ is not None
    assert t.model_.n_channels == m.model_.n_channels == n
    s_t, s_m = t.transform(real_data), m.transform(real_data)
    assert s_t.shape == s_m.shape == (n, FIELD)
    matched, cols = _matched_abs_corr(s_t, s_m)
    assert matched.min() >= 0.999, f"min matched |corr| {matched.min():.6f}"
    np.testing.assert_array_equal(cols, np.arange(n))  # same seed, same order
    assert t.final_ll_ is not None and m.final_ll_ is not None
    assert abs(t.final_ll_ - m.final_ll_) < 1e-2


def test_wrapper_accessors_agree_across_backends(fitted):
    """The wrapper's get_sphere/get_mean/get_model_center return the same
    shapes on both backends and agree within float32 tolerance (the sphere
    closer still: both are float64 computations of the same eigenproblem)."""
    t, m = fitted("torch"), fitted("mlx")
    np.testing.assert_allclose(m.get_sphere(), t.get_sphere(), rtol=1e-10, atol=1e-13)
    t_mean = t.get_mean()
    assert m.get_mean().shape == t_mean.shape == (NW,)
    assert np.abs(m.get_mean() - t_mean).max() <= _float32_tol(t_mean)
    np.testing.assert_array_equal(m.get_model_center(), t.get_model_center())
    assert all(
        a.dtype == np.float64
        for model in (t, m)
        for a in (model.get_sphere(), model.get_mean(), model.get_model_center())
    )


@pytest.mark.parametrize("damage", ["missing_params", "non_tensor_param"])
def test_malformed_mlx_payload_raises_a_named_error(fitted, tmp_path, damage):
    path = tmp_path / "mlx.pt"
    fitted("mlx").save(str(path))
    payload = torch.load(path, weights_only=True)
    if damage == "missing_params":
        del payload["backend"]["params"]
        expected = "no 'params' section"
    else:
        payload["backend"]["params"]["A"] = [1.0, 2.0]
        expected = "MLX param 'A' is a list, not a tensor"
    torch.save(payload, path)
    with pytest.raises(ValueError, match=f"malformed AMICA save file.*{expected}"):
        AMICA.load(str(path))
