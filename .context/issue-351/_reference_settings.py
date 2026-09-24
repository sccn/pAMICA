"""The reference settings of ``benchmarks/reproduce_table1.py``, shared by this
directory's scripts so that every native-binary run passes its protocol's
settings explicitly (a change of ``AMICANative``'s defaults cannot move them).

``REFERENCE_SETTINGS`` is read from the script's source without importing it,
so a script that imports an older pamica tree can use it too.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from typing import Any

_TABLE1 = Path(__file__).resolve().parents[2] / "benchmarks/reproduce_table1.py"
_NAME = "reproduce_table1_351"


def _literal(name: str) -> dict[str, Any]:
    for node in ast.parse(_TABLE1.read_text()).body:
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
            and node.value is not None
        ):
            return dict(ast.literal_eval(node.value))
    raise LookupError(f"{name} not found in {_TABLE1}")


REFERENCE_SETTINGS: dict[str, Any] = _literal("REFERENCE_SETTINGS")


def _table1():
    if _NAME in sys.modules:
        return sys.modules[_NAME]
    spec = importlib.util.spec_from_file_location(_NAME, _TABLE1)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_NAME] = mod
    spec.loader.exec_module(mod)
    return mod


def single_model_reference_kwargs(max_iter: int) -> dict[str, Any]:
    return _table1().single_model_reference_kwargs(max_iter)


def multimodel_reference_kwargs(max_iter: int) -> dict[str, Any]:
    return _table1().multimodel_reference_kwargs(max_iter)
