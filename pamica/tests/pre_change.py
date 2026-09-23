"""Import pamica as it was at an earlier commit, beside the live package.

Byte-identity tests fit the live backends and the pre-change backends in one
process on one machine, so they can compare results bit for bit without recorded
constants. The whole package at that commit is extracted from git (its Python
sources and the two data files its modules read) and imported under a private
top-level name, so every relative import resolves inside the old tree: a helper
module that a later change edited (``numpy_impl/utils.py``, say) cannot leak
into the old backends, and nothing of the old tree leaks into the live one.

This is the one place tests read historical code (issue #343), and it only
reads: it never fetches, and it never writes to the repository. A
``git fetch --depth 1`` run from a test once turned a developer's full clone
shallow, after which ``git gc`` could prune history. When the pinned commit is
missing, the calling test fails under ``CI`` (whose jobs check out full
history, ``fetch-depth: 0``) and is otherwise skipped with the command that
fetches it.

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


def _git(*args: str) -> subprocess.CompletedProcess:
    """Run a read-only git command in the repository."""
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True)


def require_commit(commit: str) -> None:
    """Fail under ``CI``, or skip, unless ``commit`` is in this clone.

    ``commit`` must be a full 40-character SHA, so the message can name the
    exact ``git fetch`` that brings it in.
    """
    if len(commit) != 40:
        raise ValueError(f"pin a full 40-character commit SHA, got {commit!r}")
    if _git("cat-file", "-e", f"{commit}^{{commit}}").returncode == 0:
        return
    shallow = _git("rev-parse", "--is-shallow-repository").stdout.strip() == b"true"
    remedy = (
        "this clone is shallow; run `git fetch --unshallow origin`"
        if shallow
        else f"run `git fetch origin {commit}`"
    )
    reason = (
        f"commit {commit[:7]} is not in this clone, so the test cannot load the "
        f"pre-change code; {remedy} and rerun (tests never fetch, issue #343)"
    )
    if os.environ.get("CI"):
        pytest.fail(reason)
    pytest.skip(reason)


def load_pre_change_package(commit: str, alias: str, dest: Path) -> ModuleType:
    """Import the ``pamica`` package at ``commit`` as top-level ``alias``.

    The sources are written under ``dest / alias`` (a pytest temporary
    directory). A second call with the same ``alias`` returns the package
    already imported. :func:`require_commit` decides what happens when the
    commit is missing.
    """
    if alias in sys.modules:
        return sys.modules[alias]
    require_commit(commit)
    result = _git("archive", "--format=tar", commit, "pamica")
    if result.returncode != 0:
        raise RuntimeError(
            f"git archive {commit[:7]} failed although the commit is present: "
            f"{result.stderr.decode().strip()!r}"
        )

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
