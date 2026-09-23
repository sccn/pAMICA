"""Import pamica as it was at an earlier commit, beside the live package.

Byte-identity tests fit the live backends and the pre-change backends in one
process on one machine, so they can compare results bit for bit without recorded
constants. The whole package at that commit is extracted from git (its Python
sources and the two data files its modules read) and imported under a private
top-level name, so every relative import resolves inside the old tree: a helper
module that a later change edited (``numpy_impl/utils.py``, say) cannot leak
into the old backends, and nothing of the old tree leaks into the live one.

Not a test module (no ``test_`` prefix, so pytest does not collect it).
"""

from __future__ import annotations

import importlib
import io
import os
import subprocess
import sys
import tarfile
from pathlib import Path, PurePosixPath
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
# Non-Python files the package reads at run time: the NumPy backend's
# defaults and the reference source fortran_params.py parses its defaults from.
_DATA_FILES = frozenset({"pamica/numpy_impl/params.json", "pamica/amica15.f90"})


def load_pre_change_package(commit: str, alias: str, dest: Path) -> ModuleType:
    """Import the ``pamica`` package at ``commit`` as top-level ``alias``.

    The sources are written under ``dest / alias`` (a pytest temporary
    directory). A second call with the same ``alias`` returns the package
    already imported. A shallow clone may lack the commit: the object is then
    fetched (best effort), and if it is still unreachable the calling
    test fails under ``CI`` (whose jobs check out full history,
    ``fetch-depth: 0``, so there an unreachable pin is a real error) and is
    skipped, loudly, only locally.
    """
    if alias in sys.modules:
        return sys.modules[alias]
    present = subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=REPO_ROOT,
        capture_output=True,
    )
    if present.returncode != 0:
        # Only a clone that lacks the object fetches it: `git fetch --depth 1`
        # records the commit as a shallow boundary, which would hide the
        # history behind it in a complete clone.
        subprocess.run(
            ["git", "fetch", "origin", commit, "--depth", "1"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
    result = subprocess.run(
        ["git", "archive", "--format=tar", commit, "pamica"],
        cwd=REPO_ROOT,
        capture_output=True,
    )
    if result.returncode != 0:
        reason = (
            f"git object {commit[:7]} is not reachable in this checkout; "
            f"git archive stderr: {result.stderr.decode().strip()!r}"
        )
        if os.environ.get("CI"):
            pytest.fail(reason)
        pytest.skip(reason)

    root = dest / alias
    with tarfile.open(fileobj=io.BytesIO(result.stdout)) as tar:
        for member in tar.getmembers():
            name = PurePosixPath(member.name)
            if not member.isfile() or name.parts[1] in ("tests", "sample_data"):
                continue
            if name.suffix != ".py" and member.name not in _DATA_FILES:
                continue
            extracted = tar.extractfile(member)
            assert extracted is not None  # a regular file always has content
            target = root.joinpath(*name.parts[1:])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(extracted.read())

    # Only the top-level import needs the path: submodules resolve through the
    # package's own __path__, which already points at ``root``.
    sys.path.insert(0, str(dest))
    try:
        return importlib.import_module(alias)
    finally:
        sys.path.remove(str(dest))
