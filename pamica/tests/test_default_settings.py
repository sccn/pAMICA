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

``AMICANative`` is not covered: it writes its own ``input.param`` defaults
(``pamica/native/engine.py``), recorded in the defaults section of the
differences guide.
"""

import inspect

import pytest

from pamica import AMICA_NumPy, AMICATorchNG
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
