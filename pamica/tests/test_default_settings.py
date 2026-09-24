"""Every pamica backend defaults to the same settings (issue #354).

The defaults table in ``docs/guides/amica-differences.md`` gives one pamica
column for ``AMICATorchNG``, ``AMICAMLXNG``, ``AMICA_NumPy`` and the wrappers
that build the first two. These tests hold the backends to it: every setting a
backend shares with ``AMICATorchNG`` has ``AMICATorchNG``'s default.

``AMICATorchNG`` and ``AMICAMLXNG`` declare their defaults in their
signatures. ``AMICA_NumPy`` reads its own from the bundled
``pamica/numpy_impl/params.json`` in its constructor (falling back to values in
code for keys the file does not carry), so it is compared through a default
construction, with its own spelling tables mapping the names. NumPy and torch
always run (``.rules/backend_parity.md``); the MLX test skips without MLX or an
Apple GPU.

``AMICANative`` runs the reference binary, so its defaults are the
``input.param`` it writes. That file is built without a binary and read back
through ``pamica.fortran_params.read_params_file``, the reader every pamica
parameter-file path uses, so each binary keyword is compared under the pamica
name it translates to.
"""

import inspect

import pytest

from pamica import AMICA_NumPy, AMICANative, AMICATorchNG
from pamica.fortran_params import FORTRAN_TO_PAMICA_KEY, read_params_file
from pamica.native.engine import _render_param
from pamica.numpy_impl import core as numpy_core

# Settings AMICA.fit names itself; max_iter is a fit() argument on the
# torch and MLX backends and a constructor setting on NumPy.
_FIT_MAX_ITER = "max_iter"


def _torch_defaults() -> dict:
    """AMICATorchNG's defaults, plus its fit()'s max_iter."""
    defaults = {
        name: param.default
        for name, param in inspect.signature(AMICATorchNG).parameters.items()
        if name != "n_channels"
    }
    defaults[_FIT_MAX_ITER] = (
        inspect.signature(AMICATorchNG.fit).parameters[_FIT_MAX_ITER].default
    )
    return defaults


def _numpy_name(name: str) -> str:
    """This setting's spelling on the NumPy backend, from its own tables."""
    return numpy_core._ALTERNATE_SPELLING.get(
        name, numpy_core._CANONICAL_TO_NUMPY_KEY.get(name, name)
    )


def test_numpy_shares_every_torch_default():
    """A default ``AMICA_NumPy()``, which reads the bundled params.json, holds
    AMICATorchNG's default for every setting the two share. Before issue #354
    params.json turned Newton on and ran 2000 iterations."""
    torch_defaults = _torch_defaults()
    unsupported = numpy_core._torch_only_options()
    shared = sorted(set(torch_defaults) - unsupported)
    model = AMICA_NumPy()

    missing = [name for name in shared if not hasattr(model, _numpy_name(name))]
    assert missing == [], f"NumPy has no attribute for {missing}"
    differing = {
        name: (getattr(model, _numpy_name(name)), torch_defaults[name])
        for name in shared
        if getattr(model, _numpy_name(name)) != torch_defaults[name]
    }
    assert differing == {}, "NumPy default != torch default: " + repr(differing)
    # The comparison reaches the settings this issue moved.
    assert {"do_newton", _FIT_MAX_ITER, "lrate"} <= set(shared)


def test_numpy_params_json_carries_the_shared_defaults():
    """The bundled file itself, read by the backend's own loader, sets the
    moved values: a code fallback cannot mask a stale entry in the file."""
    params = numpy_core.load_default_params()
    torch_defaults = _torch_defaults()
    for name in ("do_newton", _FIT_MAX_ITER, "lrate"):
        assert params[_numpy_name(name)] == torch_defaults[name], name


def _require_mlx():
    mlx_core = pytest.importorskip(
        "pamica.mlx_impl.core", reason="MLX not installed (Apple Silicon only)"
    )
    mx = mlx_core.mx
    if mx.default_device().type != mx.DeviceType.gpu:
        pytest.skip("no Apple GPU")
    return mlx_core.AMICAMLXNG


def test_mlx_shares_every_torch_default():
    """Every keyword AMICATorchNG and AMICAMLXNG share has the same default
    (only torch has device/dtype), and so does fit()'s max_iter."""
    mlx_cls = _require_mlx()
    torch_params = inspect.signature(AMICATorchNG).parameters
    mlx_params = inspect.signature(mlx_cls).parameters
    assert set(torch_params) - set(mlx_params) == {"device", "dtype"}
    assert set(mlx_params) <= set(torch_params)
    differing = {
        name: (torch_params[name].default, mlx_params[name].default)
        for name in mlx_params
        if torch_params[name].default != mlx_params[name].default
    }
    assert differing == {}
    mlx_max_iter = inspect.signature(mlx_cls.fit).parameters[_FIT_MAX_ITER].default
    assert mlx_max_iter == _torch_defaults()[_FIT_MAX_ITER]


# The canonical pamica keys (read_params_file's output) that AMICATorchNG
# spells differently in its constructor.
_TORCH_SPELLING = {"num_models": "n_models", "num_mix": "n_mix"}
NW, N_SAMPLES = 32, 30504


def _native_settings(tmp_path) -> dict:
    """The input.param a default ``AMICANative().fit`` writes for 32-channel
    data, read back under pamica's canonical names."""
    param = AMICANative()._input_params(NW, N_SAMPLES, {})
    path = tmp_path / "input.param"
    path.write_text(_render_param(param))
    return read_params_file(path)


def test_native_input_param_shares_every_torch_default(tmp_path):
    """Before issue #354 the engine wrote the bundled input.param's values
    (lrate 0.05, Newton on, max_iter 2000, block_size 512, ...)."""
    torch_defaults = _torch_defaults()
    settings = _native_settings(tmp_path)
    compared = {}
    for key, value in settings.items():
        name = _TORCH_SPELLING.get(key, key)
        # pcakeep is the data's channel count, which keeps every dimension:
        # what pamica's default of None means. block_size is one thread's
        # share of a block in the binary, checked below.
        if name in torch_defaults and name not in ("pcakeep", "block_size"):
            compared[name] = (value, torch_defaults[name])
    differing = {k: v for k, v in compared.items() if v[0] != v[1]}
    assert differing == {}, "native default != torch default: " + repr(differing)
    assert settings["pcakeep"] == NW
    assert {"lrate", "do_newton", "max_iter", "pdftype"} <= set(compared)


def test_native_block_is_pamicas_block_split_over_the_threads():
    """The binary processes n_samples // (max_threads * block_size) blocks
    of max_threads * block_size samples (amica15.f90:1215-1227), so the
    engine writes pamica's block divided by the thread count; one capped at
    the data's length still leaves the binary a block (issue #292)."""
    block = _torch_defaults()["block_size"]
    param = AMICANative()._input_params(NW, N_SAMPLES, {})
    threads, per_thread = param["max_threads"], param["block_size"]
    assert isinstance(threads, int) and isinstance(per_thread, int)
    assert block - threads < per_thread * threads <= block
    short = AMICANative()._input_params(NW, 2048, {})["block_size"]
    assert isinstance(short, int) and 0 < short * threads <= 2048
    given = AMICANative(block_size=512)._input_params(NW, N_SAMPLES, {})
    assert given["block_size"] == 512  # a given block_size is written as is


def test_native_input_param_writes_every_setting_it_can(tmp_path):
    """Every pamica setting with a binary keyword and a non-None default is
    written; the ones without a keyword (keep_best, mineig_rel, ...) cannot
    be, and the None defaults (pcadb, seed) are left to the binary."""
    torch_defaults = _torch_defaults()
    settings = _native_settings(tmp_path)
    writable = {
        _TORCH_SPELLING.get(key, key) for key in set(FORTRAN_TO_PAMICA_KEY.values())
    } & set(torch_defaults)
    expected = {name for name in writable if torch_defaults[name] is not None}
    written = {_TORCH_SPELLING.get(key, key) for key in settings}
    assert expected <= written, sorted(expected - written)
    for name in ("keep_best", "mineig_rel", "n_restarts", "pcadb", "seed"):
        assert name not in written, name
