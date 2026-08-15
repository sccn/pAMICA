"""Parity-harness parameter forwarding (issue #228).

Both arms of a parity run must be configured identically. The writer previously
rewrote six hardcoded keys and left everything else at the template's value while
the Python side honored it, which silently makes the comparison uncontrolled.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from validate_implementations import (  # noqa: E402
    fortran_accepted_keys,
    write_fortran_param_file,
)

TEMPLATE = ROOT / "pamica" / "sample_data" / "input.param"
SOURCE = ROOT / "pamica" / "amica15.f90"

pytestmark = pytest.mark.skipif(
    not TEMPLATE.exists() or not SOURCE.exists(),
    reason="reference param template or Fortran source missing",
)


def _written(tmp_path, params, overrides=None):
    dest = tmp_path / "input.param"
    write_fortran_param_file(
        TEMPLATE.read_text().splitlines(keepends=True), dest, params, overrides
    )
    lines = [ln for ln in dest.read_text().splitlines() if ln.split()]
    out = dict(ln.split(None, 1) for ln in lines)
    assert len(out) == len(lines), "a key was written more than once"
    return {k: v.strip() for k, v in out.items()}


def test_accepted_keys_come_from_the_reference_source():
    keys = fortran_accepted_keys(SOURCE)
    assert keys is not None
    # Spot-check across the parser rather than pinning the whole set, which
    # would just restate the source.
    for key in ("do_newton", "block_size", "do_approx_sphere", "max_decs", "lrate"):
        assert key in keys, f"{key} should be parsed out of amica15.f90"


def test_missing_source_disables_filtering():
    assert fortran_accepted_keys(Path("does-not-exist.f90")) is None


def test_do_newton_is_forwarded(tmp_path):
    """The regression that motivated #228: this key was silently dropped, so
    ``params['do_newton'] = False`` configured Python without Newton and left the
    reference running with it."""
    assert _written(tmp_path, {"do_newton": False})["do_newton"] == "0"
    assert _written(tmp_path, {"do_newton": True})["do_newton"] == "1"


def test_block_size_is_forwarded(tmp_path):
    """Needed since the default moved to 8192 (#229) while the template says 512."""
    assert _written(tmp_path, {"block_size": 8192})["block_size"] == "8192"


def test_aliased_names_are_translated(tmp_path):
    got = _written(tmp_path, {"num_mix": 4, "share_int": 50, "maxrej": 7})
    assert got["num_mix_comps"] == "4"
    assert got["share_iter"] == "50"
    assert got["numrej"] == "7"


def test_keys_absent_from_the_template_are_appended(tmp_path):
    """The binary accepts more keywords than the shipped template lists."""
    assert "do_approx_sphere" not in TEMPLATE.read_text()
    assert _written(tmp_path, {"do_approx_sphere": True})["do_approx_sphere"] == "1"


def test_overrides_beat_params(tmp_path):
    """The harness's paths must win over whatever params.json carries."""
    got = _written(
        tmp_path,
        {"files": ["some/other/path.fdt"], "outdir": "./elsewhere/"},
        overrides={"files": "./eeglab_data.fdt", "outdir": "./fortran_output/"},
    )
    assert got["files"] == "./eeglab_data.fdt"
    assert got["outdir"] == "./fortran_output/"


def test_untouched_template_values_survive(tmp_path):
    got = _written(tmp_path, {"max_iter": 300})
    assert got["max_iter"] == "300"
    assert got["data_dim"] == "32"  # not in params, must keep the template's value


def test_unsupported_key_is_reported(tmp_path, capsys):
    """A setting the reference cannot honor must never be dropped in silence."""
    _written(tmp_path, {"mineig_rel": 1e-12})
    assert "mineig_rel" in capsys.readouterr().out


def test_python_only_keys_are_not_reported(tmp_path, capsys):
    """``device``/``seed`` configure our side only, so their absence is expected."""
    _written(tmp_path, {"device": "cpu", "seed": 42})
    out = capsys.readouterr().out
    assert "device" not in out and "seed" not in out
