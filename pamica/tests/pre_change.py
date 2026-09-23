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
fetches it; the same holds, with its own message, when git is not installed or
the tests do not run from a git checkout.

Every commit loaded here predates the normalized initial mixing matrix of
issue #341 (epic #324 Phase 12), so :func:`with_normalized_initial_mixing`
adapts a pre-change backend class to start from the live initialization, for
tests that pin a later change against such a commit.

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
from typing import Any, NoReturn

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
# Non-Python files the package reads at run time: the NumPy backend's
# defaults and the reference source fortran_params.py parses its defaults from.
_DATA_FILES = frozenset({"pamica/numpy_impl/params.json", "pamica/amica15.f90"})


def _unavailable(reason: str) -> NoReturn:
    """Fail under ``CI`` (whose jobs check out full history with git), else skip."""
    if os.environ.get("CI"):
        pytest.fail(reason)
    pytest.skip(reason)


def _git(*args: str, root: Path = REPO_ROOT) -> subprocess.CompletedProcess:
    """Run a read-only git command in ``root``."""
    try:
        return subprocess.run(["git", *args], cwd=root, capture_output=True)
    except FileNotFoundError:
        _unavailable(
            "git is not installed; these tests need git and the pamica history"
        )


def require_commit(commit: str, root: Path = REPO_ROOT) -> None:
    """Fail under ``CI``, or skip, unless ``commit`` is in the clone at ``root``.

    ``commit`` must be a full 40-character SHA, so the message can name the
    exact ``git fetch`` that brings it in. Outside a git checkout (pamica
    installed from a wheel or an sdist, say) there is nothing to fetch into, so
    that case gets its own message.
    """
    if len(commit) != 40:
        raise ValueError(f"pin a full 40-character commit SHA, got {commit!r}")
    inside = _git("rev-parse", "--is-inside-work-tree", root=root)
    if inside.returncode != 0 or inside.stdout.strip() != b"true":
        _unavailable(
            f"{root} is not a git checkout (pamica installed from a wheel or an "
            "sdist, say), so the test cannot load the pre-change code; run it "
            "from a clone of the pamica repository with its history"
        )
    if _git("cat-file", "-e", f"{commit}^{{commit}}", root=root).returncode == 0:
        return
    shallow = _git("rev-parse", "--is-shallow-repository", root=root)
    remedy = (
        "this clone is shallow; run `git fetch --unshallow origin`"
        if shallow.stdout.strip() == b"true"
        else f"run `git fetch origin {commit}`"
    )
    _unavailable(
        f"commit {commit[:7]} is not in this clone, so the test cannot load the "
        f"pre-change code; {remedy} and rerun (tests never fetch, issue #343)"
    )


# The attribute a loaded package records its commit in, so a cached alias is
# never silently reused for a different commit.
_COMMIT_ATTR = "__pamica_pre_change_commit__"


def load_pre_change_package(
    commit: str, alias: str, dest: Path, root: Path = REPO_ROOT
) -> ModuleType:
    """Import the ``pamica`` package at ``commit`` as top-level ``alias``.

    The sources are written under ``dest / alias`` (a pytest temporary
    directory). A second call with the same ``alias`` and ``commit`` returns
    the package already imported; the same ``alias`` with another commit
    raises ``ValueError``. :func:`require_commit` decides what happens when the
    commit, git or the checkout at ``root`` is missing.
    """
    if alias in sys.modules:
        loaded = getattr(sys.modules[alias], _COMMIT_ATTR, None)
        if loaded != commit:
            raise ValueError(
                f"module {alias!r} is already imported from commit {loaded!r}; "
                f"load {commit[:7]} under a different alias"
            )
        return sys.modules[alias]
    require_commit(commit, root)
    result = _git("archive", "--format=tar", commit, "pamica", root=root)
    if result.returncode != 0:
        raise RuntimeError(
            f"git archive {commit[:7]} failed although the commit is present: "
            f"{result.stderr.decode().strip()!r}"
        )

    package_dir = dest / alias
    with tarfile.open(fileobj=io.BytesIO(result.stdout)) as tar:
        for member in tar.getmembers():
            name = PurePosixPath(member.name)
            if not member.isfile() or name.parts[1] in ("tests", "sample_data"):
                continue
            if name.suffix != ".py" and member.name not in _DATA_FILES:
                continue
            extracted = tar.extractfile(member)
            assert extracted is not None  # a regular file always has content
            target = package_dir.joinpath(*name.parts[1:])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(extracted.read())

    # Only the top-level import needs the path: submodules resolve through the
    # package's own __path__, which already points at ``package_dir``.
    sys.path.insert(0, str(dest))
    try:
        package = importlib.import_module(alias)
    finally:
        sys.path.remove(str(dest))
    setattr(package, _COMMIT_ATTR, commit)
    return package


def with_normalized_initial_mixing(cls: Any) -> Any:
    """A subclass of the pre-change backend class ``cls`` that starts from the
    normalized initial ``A`` the live backends draw (issue #341).

    Every commit the byte-identity tests load predates epic #324 Phase 12,
    whose backends normalize each drawn initial component to unit norm, as the
    reference does (amica15.f90:818-819). That changes every trajectory from its
    first E-step, so a test that pins a LATER behavior against such a commit
    (the layout of issue #334, the rescale of issue #333, ...) fits the
    pre-change class from the same initial ``A`` as the live code; everything
    else the pre-change class does is its own, unmodified. ``cls`` is the
    ``AMICATorchNG``, ``AMICAMLXNG`` or NumPy ``AMICA`` class of a package from
    :func:`load_pre_change_package`.

    The pre-change class draws exactly as it always did, so every later draw
    (``mu``, ``beta``, a restart) is unchanged; only the drawn ``A`` is then
    replaced, in the class's own storage layout (component rows, or the
    component-column layout before issue #334) and dtype:

    * PyTorch and MLX draw each fit's ``A`` first from a fresh
      ``numpy.random.RandomState(seed)``, so the replacement is the live
      :func:`pamica.initialization.initial_mixing` from that generator.
    * NumPy draws from its running generator and only while ``A`` is ``None``
      (a supplied ``A`` is kept, and the restart after a non-finite likelihood
      redraws through the same method), so the replacement normalizes the
      class's own draw: off-diagonal entries of ``I + 0.01 * (0.5 - u)`` are
      exactly the live draw's ``0.01 * (0.5 - u)``, and the diagonal is set to
      one as in :func:`pamica.initialization.draw_initial_block`.
    """
    from pamica.initialization import initial_mixing, normalize_components

    # The backend module the class (or the pre-change class it derives from,
    # for a test's own subclass) was defined in.
    backend, module = "", None
    for base in cls.__mro__:
        parts = base.__module__.split(".")
        if len(parts) == 3 and parts[1:] in (
            ["torch_impl", "core"],
            ["mlx_impl", "core"],
            ["numpy_impl", "core"],
        ):
            backend, module = parts[1], sys.modules[base.__module__]
            break
    if module is None:
        raise ValueError(f"{cls!r} is not a pamica backend class")

    def blocks(shape: tuple, n: int, m: int) -> list:
        """Each model's block of an ``A`` of ``shape`` as index tuples."""
        if shape == (m * n, n):  # component rows (issue #334 on)
            return [(slice(h * n, (h + 1) * n), slice(None)) for h in range(m)]
        if shape == (n, m * n):  # component columns (before issue #334)
            return [(slice(None), slice(h * n, (h + 1) * n)) for h in range(m)]
        raise ValueError(f"unexpected A shape {shape} for {m} model(s) of {n}")

    if backend == "numpy_impl":

        class NumPyNormalizedInit(cls):
            def _initialize_parameters(self):
                drawn = self.A is None and not getattr(self, "fix_init", False)
                super()._initialize_parameters()
                if drawn:
                    A = np.array(self.A, dtype=np.float64)
                    n, m = self.data_dim, self.num_models
                    for idx in blocks(A.shape, n, m):
                        block = A[idx].copy()
                        block[np.diag_indices(n)] = 1.0
                        A[idx] = normalize_components(block)
                    self.A = A
                    self._update_unmixing_matrices()

        return NumPyNormalizedInit

    class NormalizedInit(cls):
        def _initialize_parameters(self):
            super()._initialize_parameters()
            n, m = self.n_channels, self.n_models
            rows = initial_mixing(np.random.RandomState(self.seed), n, m)
            shape = tuple(self.A.shape)
            A = np.empty(shape)
            for h, idx in enumerate(blocks(shape, n, m)):
                A[idx] = rows[h * n : (h + 1) * n, :]
            if backend == "torch_impl":
                self.A = module.torch.from_numpy(A).to(self.A.device, self.A.dtype)
            else:
                self.A = module.mx.array(A.astype(np.float32))
            self._update_unmixing_matrices()

    return NormalizedInit
