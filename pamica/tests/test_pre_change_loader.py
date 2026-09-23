"""The historical-code loader never fetches or mutates the repository (issue #343).

Tests used to run ``git fetch origin <sha> --depth 1`` to reach a pinned
commit, which in a full clone records a shallow boundary, after which ``git gc``
can prune history. ``pamica/tests/pre_change.py`` now only reads. These tests put
a ``git`` wrapper first on ``PATH`` that logs its arguments and then runs the
real git, drive the loader through its three outcomes (commit present; missing
locally; missing under ``CI``), and assert that no fetch ran and the clone's
shallow state did not change. The wrapper runs real git, so every command still
does its real work.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from pamica.tests import pre_change

# The epic #324 head Phase 8 merges onto; any commit in the clone would do.
PRESENT = "0930c0e68ec9e2bbef3d20d51cff4c03029248f2"
# A well-formed SHA that names no object (the empty-tree SHA with its first
# character changed).
MISSING = "0b825dc642cb6eb9a060e54bf8d69288fbee4904"

_REAL_GIT = shutil.which("git")
pytestmark = pytest.mark.skipif(_REAL_GIT is None, reason="git not installed")


@pytest.fixture
def git_log(tmp_path, monkeypatch) -> Path:
    """Put a logging ``git`` wrapper first on ``PATH``; return its log file."""
    assert _REAL_GIT is not None
    bindir = tmp_path / "bin"
    bindir.mkdir()
    log = tmp_path / "git.log"
    wrapper = bindir / "git"
    wrapper.write_text(
        f"#!/bin/sh\nprintf '%s\\n' \"$*\" >> '{log}'\nexec '{_REAL_GIT}' \"$@\"\n"
    )
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    return log


def _logged(log: Path) -> list[str]:
    return log.read_text().splitlines() if log.exists() else []


def _is_shallow() -> str:
    return subprocess.run(
        ["git", "rev-parse", "--is-shallow-repository"],
        cwd=pre_change.REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _assert_no_fetch(log: Path) -> None:
    calls = _logged(log)
    assert calls, "setup: the wrapper logged nothing, so git was not intercepted"
    assert not any(
        call.split()[:1] == ["fetch"] or "--depth" in call.split() for call in calls
    ), f"the loader fetched: {calls}"


def test_loading_a_present_commit_only_reads(git_log, tmp_path):
    if pre_change._git("cat-file", "-e", f"{PRESENT}^{{commit}}").returncode:
        pre_change.require_commit(PRESENT)  # fails under CI, skips locally
    shallow_before = _is_shallow()
    alias = "pamica_loader_check"
    try:
        package = pre_change.load_pre_change_package(PRESENT, alias, tmp_path)
        assert package.__file__ is not None
        assert Path(package.__file__).is_relative_to(tmp_path)
    finally:
        for name in [m for m in sys.modules if m == alias or m.startswith(alias + ".")]:
            del sys.modules[name]
    _assert_no_fetch(git_log)
    assert any(call.startswith("archive") for call in _logged(git_log))
    assert _is_shallow() == shallow_before


def test_a_missing_commit_skips_locally_with_the_fetch_command(git_log, monkeypatch):
    monkeypatch.delenv("CI", raising=False)
    shallow_before = _is_shallow()
    with pytest.raises(pytest.skip.Exception) as excinfo:
        pre_change.require_commit(MISSING)
    message = str(excinfo.value)
    if shallow_before == "true":
        assert "git fetch --unshallow origin" in message
    else:
        assert f"git fetch origin {MISSING}" in message
    _assert_no_fetch(git_log)
    assert _is_shallow() == shallow_before


def test_a_missing_commit_fails_under_ci(git_log, monkeypatch):
    monkeypatch.setenv("CI", "true")
    with pytest.raises(pytest.fail.Exception, match="is not in this clone"):
        pre_change.require_commit(MISSING)
    _assert_no_fetch(git_log)


def test_an_abbreviated_pin_is_rejected():
    with pytest.raises(ValueError, match="40-character"):
        pre_change.require_commit(PRESENT[:7])
