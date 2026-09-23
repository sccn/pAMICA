"""The historical-code loader never fetches or mutates the repository (issue #343).

Tests used to run ``git fetch origin <sha> --depth 1`` to reach a pinned
commit, which in a full clone records a shallow boundary, after which ``git gc``
can prune history. ``pamica/tests/pre_change.py`` now only reads. These tests put
a ``git`` wrapper first on ``PATH`` that logs its arguments, REFUSES any fetch
(it exits nonzero without running git, so not even a test of the guard can
touch the repository), and hands every other command to the real git, which
does its real work. They drive the loader through its outcomes (commit present;
missing locally; missing under ``CI``; no git; not a git checkout; an alias
reused for another commit) and assert that no fetch ran and the clone's
shallow state did not change.
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


# Arguments the wrapper refuses: anything that fetches or reshapes history.
_REFUSED = ("fetch", "pull", "--depth", "--deepen", "--shallow", "--unshallow")
_REFUSED_STATUS = 97


@pytest.fixture
def git_log(tmp_path, monkeypatch) -> Path:
    """Put a logging, fetch-refusing ``git`` wrapper first on ``PATH``; return
    its log file."""
    assert _REAL_GIT is not None
    bindir = tmp_path / "bin"
    bindir.mkdir()
    log = tmp_path / "git.log"
    wrapper = bindir / "git"
    wrapper.write_text(
        "#!/bin/sh\n"
        f"printf '%s\\n' \"$*\" >> '{log}'\n"
        'for arg in "$@"; do\n'
        '  case "$arg" in\n'
        "    fetch|pull|--depth*|--deepen*|--shallow*|--unshallow)\n"
        "      echo 'git wrapper: refused (tests never fetch, issue #343)' >&2\n"
        f"      exit {_REFUSED_STATUS} ;;\n"
        "  esac\n"
        "done\n"
        f"exec '{_REAL_GIT}' \"$@\"\n"
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
    fetched = [
        call for call in calls if any(arg.startswith(_REFUSED) for arg in call.split())
    ]
    assert not fetched, f"the loader fetched: {fetched}"


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


def test_the_guard_catches_a_fetch_without_running_it(git_log):
    """The negative control: a fetch sent through ``_git`` is logged, refused by
    the wrapper before any git runs, flagged by ``_assert_no_fetch``, and the
    clone's shallow state is unchanged."""
    shallow_before = _is_shallow()
    result = pre_change._git("fetch", "origin", PRESENT, "--depth", "1")
    assert result.returncode == _REFUSED_STATUS
    assert b"refused" in result.stderr
    with pytest.raises(AssertionError, match="the loader fetched"):
        _assert_no_fetch(git_log)
    assert _is_shallow() == shallow_before


@pytest.fixture
def no_git(tmp_path, monkeypatch) -> None:
    """A ``PATH`` with no ``git`` on it."""
    empty = tmp_path / "empty_bin"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))


def test_missing_git_skips_locally(no_git, monkeypatch):
    monkeypatch.delenv("CI", raising=False)
    with pytest.raises(pytest.skip.Exception, match="git is not installed"):
        pre_change.require_commit(PRESENT)


def test_missing_git_fails_under_ci(no_git, monkeypatch):
    monkeypatch.setenv("CI", "true")
    with pytest.raises(pytest.fail.Exception, match="git is not installed"):
        pre_change.require_commit(PRESENT)


@pytest.fixture
def not_a_checkout(tmp_path, monkeypatch) -> Path:
    """A directory git cannot resolve to a repository, however ``tmp_path`` is
    placed: the search for ``.git`` stops at its parent."""
    root = tmp_path / "unpacked"
    root.mkdir()
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    for var in ("GIT_DIR", "GIT_WORK_TREE"):
        monkeypatch.delenv(var, raising=False)
    return root


def test_outside_a_checkout_skips_without_fetch_advice(
    git_log, not_a_checkout, monkeypatch, tmp_path
):
    monkeypatch.delenv("CI", raising=False)
    with pytest.raises(pytest.skip.Exception) as excinfo:
        pre_change.load_pre_change_package(
            PRESENT, "pamica_not_a_checkout", tmp_path, root=not_a_checkout
        )
    message = str(excinfo.value)
    assert "is not a git checkout" in message
    assert "git fetch" not in message
    assert "pamica_not_a_checkout" not in sys.modules
    _assert_no_fetch(git_log)


def test_outside_a_checkout_fails_under_ci(not_a_checkout, monkeypatch):
    monkeypatch.setenv("CI", "true")
    with pytest.raises(pytest.fail.Exception, match="is not a git checkout"):
        pre_change.require_commit(PRESENT, root=not_a_checkout)


def test_an_alias_is_never_reused_for_another_commit(tmp_path):
    if pre_change._git("cat-file", "-e", f"{PRESENT}^{{commit}}").returncode:
        pre_change.require_commit(PRESENT)  # fails under CI, skips locally
    alias = "pamica_alias_check"
    try:
        first = pre_change.load_pre_change_package(PRESENT, alias, tmp_path)
        assert pre_change.load_pre_change_package(PRESENT, alias, tmp_path) is first
        with pytest.raises(ValueError, match="different alias"):
            pre_change.load_pre_change_package(MISSING, alias, tmp_path)
    finally:
        for name in [m for m in sys.modules if m == alias or m.startswith(alias + ".")]:
            del sys.modules[name]
