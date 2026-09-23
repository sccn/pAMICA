"""Suite-wide fixtures.

Every test runs with its working directory set to a per-session temporary
directory, so nothing the suite writes to a relative path lands in the
repository. This is defense in depth: the legacy NumPy backend used to default
``outdir`` to ``./output`` and write ``out.txt`` and its model files there
from every fit (63 tests did), and a relative ``outdir`` in a params file
(``sample_params.json`` says ``./amicaout/``) or the NumPy CLI's ``--outdir``
default of ``output`` still resolve against the working directory. Tests that
need a repository-relative path (a params file whose ``files`` entry is
relative to the repository root, say) resolve it from ``__file__`` or
``monkeypatch.chdir`` explicitly.
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
