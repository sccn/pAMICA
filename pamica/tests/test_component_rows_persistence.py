"""Persistence of the component-row ``A`` (issue #334, epic #324 Phase 8, ADR 0007).

The PyTorch ``state_dict`` is ``format_version`` 4 and the MLX save format 2:
both store ``A`` as ``(n_comps, n)``, one component per row. A save from before
the change (PyTorch 3, MLX 1: ``A`` as ``(n, n_comps)`` with ``comp_list``
indexing stored columns) is converted without loss when unmerged, and refused
with a message to refit when ``share_comps`` had merged components, because
that merge compared and tied stored columns. The ``AMICA`` wrapper's own save
(``format_version`` 2) wraps the backend payloads, so it converts and refuses
the same way.

Pinned here, per backend and through the wrapper: new saves round-trip bit for
bit (single model, unmerged and merged multi-model); old saves, written by the
pre-change classes loaded from git (never hand-built), load into models whose
``transform`` and accessors reproduce the saved model exactly; an old merged
save is refused. Real bundled sample EEG only, no synthetic data or mocks.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
import torch

from pamica import AMICA
from pamica.tests.pre_change import load_pre_change_package
from pamica.torch_impl.core import AMICATorchNG
from pamica.torch_impl.utils import load_eeglab_data

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_data"
DATA_FILE = SAMPLE_DIR / "eeglab_data.fdt"
NW = 32
FIELD = 30504
NMIX = 3
SEED = 42
N_SAMPLES = 4096
ITERS = 10
# The epic #324 head Phase 8 merges onto: the last component-column layout.
PRE_CHANGE_COMMIT = "0930c0e68ec9e2bbef3d20d51cff4c03029248f2"

# Fit recipes. "merged" merges a few pairs at iteration 8 under either layout's
# metric (both models are still near-collinear there); the refusal test checks
# that the pre-change fit really merged.
_RECIPES: Dict[str, Dict[str, Any]] = {
    "single": dict(n_models=1),
    "unmerged": dict(n_models=2),
    "merged": dict(
        n_models=2,
        share_comps=True,
        share_start=8,
        share_iter=100,
        comp_thresh=0.99,
    ),
}
_REFIT = "Refit it with this version of pamica"

pytestmark = pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing")


@pytest.fixture(scope="module")
def real_data() -> np.ndarray:
    X = load_eeglab_data(str(DATA_FILE), data_dim=NW, field_dim=FIELD)
    return X.astype(np.float64)[:, :N_SAMPLES]


@pytest.fixture(scope="module")
def pre(tmp_path_factory) -> Any:
    return load_pre_change_package(
        PRE_CHANGE_COMMIT, "pamica_pre334", tmp_path_factory.mktemp("pre334")
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


def _backend_fit(cls: Any, backend: str, recipe: str, X: np.ndarray) -> Any:
    kwargs: Dict[str, Any] = dict(
        n_channels=NW, n_mix=NMIX, seed=SEED, block_size=N_SAMPLES, **_RECIPES[recipe]
    )
    if backend == "torch":
        kwargs.update(device="cpu", dtype=torch.float64)
    model = cls(**kwargs)
    model.fit(X, max_iter=ITERS, verbose=False)
    return model


def _wrapper_fit(cls: Any, backend: str, recipe: str, X: np.ndarray) -> Any:
    """A wrapper fit; an ``"mlx"`` fit skips the calling test without MLX or an
    Apple GPU, so no wrapper route can reach the backend's ImportError."""
    if backend == "mlx":
        _mlx_core()
    recipe_kwargs = dict(_RECIPES[recipe])
    n_models = recipe_kwargs.pop("n_models")
    device = "cpu" if backend == "torch" else None
    model = cls(
        n_models=n_models, n_mix=NMIX, device=device, verbose=False, backend=backend
    )
    model.fit(X, max_iter=ITERS, seed=SEED, block_size=N_SAMPLES, **recipe_kwargs)
    return model


def _outputs(model: Any, X: np.ndarray, n_models: int) -> Dict[str, np.ndarray]:
    """What a caller reads off a fitted model, per model: sources and the
    three mixing/unmixing accessors, plus the sharing groups."""
    out: Dict[str, Any] = {}
    for h in range(n_models):
        out[f"transform{h}"] = np.asarray(model.transform(X, model_idx=h))
        out[f"mixing{h}"] = np.asarray(model.get_mixing_matrix(h))
        out[f"sensor{h}"] = np.asarray(model.get_sensor_mixing_matrix(h))
        out[f"unmixing{h}"] = np.asarray(model.get_unmixing_matrix(h))
    out["shared"] = model.shared_components()
    return out


def _assert_same_outputs(got: Dict[str, Any], want: Dict[str, Any]) -> None:
    assert got.keys() == want.keys()
    for key in want:
        if key == "shared":
            assert got[key] == want[key]
        else:
            np.testing.assert_array_equal(got[key], want[key], err_msg=key)


def _merged(model: Any) -> bool:
    cl = model.comp_list
    cl = cl.cpu().numpy() if isinstance(cl, torch.Tensor) else np.array(cl)
    return np.unique(cl).size < cl.size


# --- new-format round trips --------------------------------------------------
@pytest.mark.parametrize("recipe", list(_RECIPES))
def test_torch_state_dict_round_trip(real_data, recipe, tmp_path):
    m = _backend_fit(AMICATorchNG, "torch", recipe, real_data)
    assert _merged(m) == (recipe == "merged"), "setup: merge state"
    state = m.state_dict()
    assert state["format_version"] == 4
    assert tuple(state["params"]["A"].shape) == (m.n_comps, m.n_channels)
    torch.save(state, tmp_path / "state.pt")
    loaded = AMICATorchNG.from_state_dict(
        torch.load(tmp_path / "state.pt", weights_only=True), device="cpu"
    )
    assert loaded.A is not None and m.A is not None
    assert torch.equal(loaded.A, m.A)
    n_models = _RECIPES[recipe]["n_models"]
    _assert_same_outputs(
        _outputs(loaded, real_data, n_models), _outputs(m, real_data, n_models)
    )


@pytest.mark.parametrize("recipe", list(_RECIPES))
def test_mlx_save_round_trip(real_data, recipe, tmp_path):
    cls = _mlx_core().AMICAMLXNG
    m = _backend_fit(cls, "mlx", recipe, real_data)
    assert _merged(m) == (recipe == "merged"), "setup: merge state"
    state = m.state_dict()
    assert state["format_version"] == 2
    assert state["params"]["A"].shape == (m.n_comps, m.n_channels)
    m.save(str(tmp_path / "model.npz"))
    n_models = _RECIPES[recipe]["n_models"]
    via_file = cls.load(str(tmp_path / "model.npz"))
    via_dict = cls.from_state_dict(state)
    want = _outputs(via_dict, real_data, n_models)
    _assert_same_outputs(_outputs(via_file, real_data, n_models), want)
    np.testing.assert_array_equal(np.array(via_file.A), np.array(m.A))
    # transform and the sphered-space accessors read persisted arrays only, so
    # they match the in-memory model too (the sensor maps use the float32
    # sphere after a reload; see AMICAMLXNG._load_params).
    for h in range(n_models):
        np.testing.assert_array_equal(
            want[f"transform{h}"], np.asarray(m.transform(real_data, model_idx=h))
        )
        np.testing.assert_array_equal(want[f"mixing{h}"], m.get_mixing_matrix(h))


@pytest.mark.parametrize("recipe", list(_RECIPES))
@pytest.mark.parametrize("backend", ["torch", "mlx"])
def test_wrapper_save_round_trip(real_data, backend, recipe, tmp_path):
    m = _wrapper_fit(AMICA, backend, recipe, real_data)
    assert _merged(m.model_) == (recipe == "merged"), "setup: merge state"
    path = str(tmp_path / "model.pt")
    m.save(path)
    loaded = AMICA.load(path, device="cpu" if backend == "torch" else None)
    n_models = _RECIPES[recipe]["n_models"]
    got = _outputs(loaded, real_data, n_models)
    if backend == "torch":
        _assert_same_outputs(got, _outputs(m, real_data, n_models))
    else:
        # The MLX sensor maps use the float32 sphere after a reload, so compare
        # against a second reload rather than the in-memory model.
        _assert_same_outputs(got, _outputs(AMICA.load(path), real_data, n_models))


# --- old saves: converted when unmerged, refused when merged ------------------
def _old_class(pre: Any, backend: str) -> Any:
    if backend == "torch":
        return pre.torch_impl.core.AMICATorchNG
    _mlx_core()
    return importlib.import_module(f"{pre.__name__}.mlx_impl.core").AMICAMLXNG


@pytest.mark.parametrize("recipe", ["single", "unmerged"])
def test_old_torch_state_dict_converts_losslessly(pre, real_data, recipe, tmp_path):
    """A format_version 3 state (components as columns) loads into a model that
    reproduces the saved model's sources and accessors exactly."""
    old = _backend_fit(_old_class(pre, "torch"), "torch", recipe, real_data)
    state = old.state_dict()
    assert state["format_version"] == 3
    torch.save(state, tmp_path / "old.pt")
    loaded = AMICATorchNG.from_state_dict(
        torch.load(tmp_path / "old.pt", weights_only=True), device="cpu"
    )
    assert loaded.A is not None
    assert tuple(loaded.A.shape) == (loaded.n_comps, loaded.n_channels)
    n_models = _RECIPES[recipe]["n_models"]
    _assert_same_outputs(
        _outputs(loaded, real_data, n_models), _outputs(old, real_data, n_models)
    )
    # And it saves again in the current format, round-tripping unchanged.
    again = AMICATorchNG.from_state_dict(loaded.state_dict(), device="cpu")
    assert loaded.state_dict()["format_version"] == 4
    assert again.A is not None
    assert torch.equal(again.A, loaded.A)


@pytest.mark.parametrize("recipe", ["single", "unmerged"])
def test_old_mlx_save_converts_losslessly(pre, real_data, recipe, tmp_path):
    """A format_version 1 ``.npz`` (components as columns) loads into a model
    that reproduces the saved model exactly: compared with the pre-change class
    loading the same file, so both read the same float32 sphere."""
    old_cls = _old_class(pre, "mlx")
    new_cls = _mlx_core().AMICAMLXNG
    old = _backend_fit(old_cls, "mlx", recipe, real_data)
    path = str(tmp_path / "old.npz")
    old.save(path)
    with np.load(path) as raw:
        assert int(raw["format_version"]) == 1
    n_models = _RECIPES[recipe]["n_models"]
    want = _outputs(old_cls.load(path), real_data, n_models)
    _assert_same_outputs(_outputs(new_cls.load(path), real_data, n_models), want)
    converted = new_cls.from_state_dict(old.state_dict())
    _assert_same_outputs(_outputs(converted, real_data, n_models), want)


@pytest.mark.parametrize("recipe", ["single", "unmerged"])
@pytest.mark.parametrize("backend", ["torch", "mlx"])
def test_old_wrapper_save_converts_losslessly(
    pre, real_data, backend, recipe, tmp_path
):
    """The wrapper's own save wraps the backend payload, so an old wrapper file
    goes through the same conversion."""
    old = _wrapper_fit(pre.AMICA, backend, recipe, real_data)
    path = str(tmp_path / "old.pt")
    old.save(path)
    device = "cpu" if backend == "torch" else None
    n_models = _RECIPES[recipe]["n_models"]
    want = _outputs(pre.AMICA.load(path, device=device), real_data, n_models)
    loaded = AMICA.load(path, device=device)
    _assert_same_outputs(_outputs(loaded, real_data, n_models), want)


@pytest.mark.parametrize("route", ["state_dict", "wrapper"])
@pytest.mark.parametrize("backend", ["torch", "mlx"])
def test_old_merged_save_is_refused(pre, real_data, backend, route, tmp_path):
    """A pre-change save whose ``share_comps`` merged components tied stored
    columns across models; it has no component-row equivalent, so every load
    route refuses it and says to refit."""
    if route == "wrapper":
        old = _wrapper_fit(pre.AMICA, backend, "merged", real_data)
        assert _merged(old.model_), "setup: the pre-change fit merged nothing"
        path = str(tmp_path / "old.pt")
        old.save(path)
        with pytest.raises(ValueError, match=_REFIT):
            AMICA.load(path, device="cpu" if backend == "torch" else None)
        return
    old = _backend_fit(_old_class(pre, backend), backend, "merged", real_data)
    assert _merged(old), "setup: the pre-change fit merged nothing"
    if backend == "torch":
        with pytest.raises(ValueError, match=_REFIT):
            AMICATorchNG.from_state_dict(old.state_dict(), device="cpu")
        return
    new_cls = _mlx_core().AMICAMLXNG
    with pytest.raises(ValueError, match=_REFIT):
        new_cls.from_state_dict(old.state_dict())
    path = str(tmp_path / "old.npz")
    old.save(path)
    with pytest.raises(ValueError, match=_REFIT):
        new_cls.load(path)
