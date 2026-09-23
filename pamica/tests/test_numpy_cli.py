"""NumPy CLI accepts both params-file formats (issue #304 PR #327 review).

Before this fix, ``cli.py``'s own ``load_params()`` did a bare ``json.load``,
so ``python -m pamica.numpy_impl.cli input.param`` died with a raw
``json.JSONDecodeError`` before ever reaching ``AMICA(params_file=...)`` --
even though the constructor itself already accepted the literal Fortran text
format. This exercises the CLI's actual ``main()`` entry point end to end on
real bundled data (a tiny iteration budget), for both formats, so the
regression cannot come back unnoticed.

Real bundled ``sample_data/input.param``/``sample_params.json`` and
``eeglab_data.fdt`` only (NO MOCKS); each params file is a tmp copy with
``max_iter`` lowered and ``files`` repointed at the real ``.fdt``'s absolute
path, so neither the CLI's nor the fit's working directory matters.
"""

import sys
from pathlib import Path

import pytest

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_data"
PARAM_FILE = SAMPLE_DIR / "input.param"
JSON_FILE = SAMPLE_DIR / "sample_params.json"
DATA_FILE = SAMPLE_DIR / "eeglab_data.fdt"

pytestmark = pytest.mark.skipif(
    not PARAM_FILE.exists() or not JSON_FILE.exists() or not DATA_FILE.exists(),
    reason="bundled sample_data/input.param, sample_params.json or eeglab_data.fdt missing",
)


def _fortran_text_with_tiny_iters(tmp_path: Path) -> Path:
    """A tmp copy of the real input.param: max_iter lowered to 2, files
    repointed at the real sample .fdt's absolute path."""
    dest = tmp_path / "input.param"
    lines = []
    for line in PARAM_FILE.read_text().splitlines():
        key = line.split()[0] if line.split() else None
        if key == "max_iter":
            lines.append("max_iter 2")
        elif key == "files":
            lines.append(f"files {DATA_FILE.resolve()}")
        else:
            lines.append(line)
    dest.write_text("\n".join(lines) + "\n")
    return dest


def _json_with_tiny_iters(tmp_path: Path) -> Path:
    """A tmp copy of the real sample_params.json: max_iter lowered to 2,
    files repointed at the real sample .fdt's absolute path."""
    import json

    payload = json.loads(JSON_FILE.read_text())
    payload["max_iter"] = 2
    payload["files"] = [str(DATA_FILE.resolve())]
    dest = tmp_path / "sample_params.json"
    dest.write_text(json.dumps(payload))
    return dest


def _run_cli(paramfile: Path, outdir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from pamica.numpy_impl import cli

    monkeypatch.setattr(
        sys, "argv", ["cli.py", str(paramfile), "--outdir", str(outdir), "--seed", "0"]
    )
    cli.main()


def test_cli_accepts_fortran_text_params_file(tmp_path, monkeypatch):
    """The regression this whole file guards: a literal Fortran input.param
    must drive the CLI end to end, not just AMICA(params_file=...) directly."""
    dest = _fortran_text_with_tiny_iters(tmp_path)
    outdir = tmp_path / "amicaout"
    _run_cli(dest, outdir, monkeypatch)
    assert (outdir / "W").exists()


def test_cli_still_accepts_json_params_file(tmp_path, monkeypatch):
    """Unaffected by the fix: the pre-existing JSON path keeps working."""
    dest = _json_with_tiny_iters(tmp_path)
    outdir = tmp_path / "amicaout_json"
    _run_cli(dest, outdir, monkeypatch)
    assert (outdir / "W").exists()


def test_load_params_maps_canonical_keys_for_fortran_text(tmp_path):
    """load_params() itself (not just main()) returns this backend's own
    attribute spellings for the three renamed settings, matching
    AMICA(params_file=...)'s own _read_numpy_keyed_params."""
    from pamica.numpy_impl.cli import load_params

    dest = _fortran_text_with_tiny_iters(tmp_path)
    params = load_params(str(dest))
    assert params["max_decs"] == 3  # input.param's own max_decs
    assert params["min_grad_norm"] == 1e-7  # input.param's own min_grad_norm
    assert params["share_int"] == 100  # input.param's own share_iter
    assert "maxdecs" not in params
    assert "min_nd" not in params
    assert "share_iter" not in params


def test_cli_without_outdir_still_writes_to_output(tmp_path, monkeypatch):
    """The library default no longer writes files (``outdir=None``), but the
    CLI's own ``--outdir`` default is ``output``, so a CLI run without the flag
    still writes its results to ``./output`` as before."""
    from pamica.numpy_impl import cli

    dest = _json_with_tiny_iters(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    monkeypatch.chdir(run_dir)
    monkeypatch.setattr(sys, "argv", ["cli.py", str(dest), "--seed", "0"])
    cli.main()
    for name in ("out.txt", "W", "S", "LL"):
        assert (run_dir / "output" / name).exists(), name
