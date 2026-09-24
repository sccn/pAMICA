"""``AMICA_NumPy`` writes files only when given an ``outdir`` (epic #324 Phase 6).

The legacy backend used to default ``outdir`` to ``./output`` and write its
``out.txt`` log at construction, its ``writestep`` checkpoints during the fit
and its results at the end, into whatever the caller's working directory was.
The PyTorch and MLX backends never write unless asked, so the NumPy default is
now ``outdir=None``: no files at all. An explicit ``outdir`` (keyword or params
file; the CLI passes its own ``--outdir``, default ``output``) writes exactly
what it did before.

Real bundled sample EEG only; each fit is a few iterations with every write
path switched on (a checkpoint and a history snapshot on every iteration).
"""

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

from pamica import AMICA_NumPy
from pamica.numpy_impl.load import loadmodout
from pamica.torch_impl.utils import load_eeglab_data

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_data"
DATA_FILE = SAMPLE_DIR / "eeglab_data.fdt"
JSON_FILE = SAMPLE_DIR / "sample_params.json"
NW = 32
N_FRAMES = 4096
MAX_ITER = 3

# Every file-writing path on: out.txt, a results checkpoint and a history
# snapshot each iteration, then the final results write.
ALL_WRITES: Dict[str, Any] = dict(
    seed=0,
    max_iter=MAX_ITER,
    block_size=N_FRAMES,
    writestep=1,
    do_history=True,
    histstep=1,
)
RESULT_FILES = ("out.txt", "W", "S", "mean", "gm", "LL", "LLt", "A")

pytestmark = pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing")


@pytest.fixture(scope="module")
def X() -> np.ndarray:
    data = load_eeglab_data(str(DATA_FILE), data_dim=NW, field_dim=30504)
    return data.astype(np.float64)[:, :N_FRAMES]


def test_default_writes_nothing(X, tmp_path, monkeypatch):
    """No ``outdir``: the fit completes and leaves the working directory
    empty, with every write path switched on."""
    monkeypatch.chdir(tmp_path)
    model = AMICA_NumPy(use_tqdm=False, **ALL_WRITES)
    assert model.outdir is None and model.file_path is None
    model.fit(X)
    assert model.converged
    assert len(model.ll) == MAX_ITER
    assert list(tmp_path.iterdir()) == []


def test_explicit_outdir_writes_as_before(X, tmp_path, monkeypatch):
    """An explicit ``outdir`` gets the log, the checkpoints, the history and
    the final results, readable by ``loadmodout``; nothing lands elsewhere."""
    monkeypatch.chdir(tmp_path)
    outdir = tmp_path / "amicaout"
    model = AMICA_NumPy(use_tqdm=False, outdir=str(outdir), **ALL_WRITES)
    assert (outdir / "out.txt").exists()  # written at construction, as before
    model.fit(X)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["amicaout"]
    for name in RESULT_FILES:
        assert (outdir / name).exists(), name
    assert len(list((outdir / "history").iterdir())) == MAX_ITER
    assert " iter " in (outdir / "out.txt").read_text()  # the per-iteration log
    np.testing.assert_array_equal(loadmodout(outdir).LL, model.ll)


def test_params_file_outdir_is_explicit(X, tmp_path, monkeypatch):
    """A params file that names an ``outdir`` (the bundled
    ``sample_params.json`` says ``./amicaout/``) writes there, relative to
    the working directory, as it always has."""
    monkeypatch.chdir(tmp_path)
    assert json.loads(JSON_FILE.read_text())["outdir"] == "./amicaout/"
    model = AMICA_NumPy(params_file=str(JSON_FILE), use_tqdm=False, **ALL_WRITES)
    assert model.outdir == Path("./amicaout/")
    model.fit(X)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["amicaout"]
    for name in RESULT_FILES:
        assert (tmp_path / "amicaout" / name).exists(), name
