"""Suite-wide fixtures.

Every test runs with its working directory set to a per-session temporary
directory, so nothing the suite writes to a relative path lands in the
repository. The legacy NumPy backend is the main reason: ``AMICA_NumPy``
writes ``out.txt`` into its ``outdir`` at construction and its model files at
the end of a fit, and ``outdir`` defaults to ``./output``. Tests that need a
repository-relative path (a params file whose ``files`` entry is relative to
the repository root, say) resolve it from ``__file__`` or ``monkeypatch.chdir``
explicitly.
"""

import pytest


@pytest.fixture(scope="session", autouse=True)
def _cwd_outside_the_repository(tmp_path_factory):
    """Run the session from a temporary directory, restored afterwards.

    Session-scoped, so it is in place before any module- or class-scoped
    fixture runs; under ``pytest-xdist`` each worker gets its own directory.
    """
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(tmp_path_factory.mktemp("cwd"))
        yield
