"""Fortran ``input.param`` reader (issue #132).

Real bundled data only (NO MOCKS): ``sample_data/input.param`` and
``sample_data/sample_params.json`` describe the same reference run, so the
round-trip test at the bottom compares the two directly rather than against a
synthetic fixture. The malformed-line cases below are the one place tiny
hand-written text is unavoidable (there is no "real" malformed parameter
file), kept to a couple of lines each, per ``.rules/testing.md``.
"""

import json
import logging
from pathlib import Path

import pytest

from pamica.amica import AMICA
from pamica.fortran_params import (
    FORTRAN_TO_PAMICA_KEY,
    FORTRAN_UNSUPPORTED_KEYS,
    _fortran_accepted_keys,
    read_fortran_param_file,
)

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_data"
PARAM_FILE = SAMPLE_DIR / "input.param"
JSON_FILE = SAMPLE_DIR / "sample_params.json"

pytestmark = pytest.mark.skipif(
    not PARAM_FILE.exists() or not JSON_FILE.exists(),
    reason="bundled sample_data/input.param or sample_params.json missing",
)


@pytest.fixture(scope="module")
def parsed():
    return read_fortran_param_file(PARAM_FILE)


def test_known_scalar_mappings_land(parsed):
    """Ground truth read directly from the bundled input.param."""
    assert parsed["block_size"] == 512
    assert parsed["pdftype"] == 0
    assert parsed["kurt_start"] == 3
    assert parsed["num_kurt"] == 5
    assert parsed["kurt_int"] == 1
    assert parsed["num_models"] == 1
    assert parsed["max_iter"] == 2000
    assert parsed["data_dim"] == 32
    assert parsed["rho0"] == 1.5
    assert parsed["lrate"] == 0.05


def test_renamed_keys_translate(parsed):
    """The three keywords the JOSS issue named as mismatched."""
    assert "num_mix_comps" not in parsed
    assert parsed["num_mix"] == 3  # Fortran num_mix_comps -> num_mix
    assert "share_iter" not in parsed
    assert parsed["share_int"] == 100  # Fortran share_iter -> share_int
    assert "numrej" not in parsed
    assert parsed["maxrej"] == 3  # Fortran numrej -> maxrej


def test_boolean_flags_are_python_bool(parsed):
    assert parsed["do_newton"] is True
    assert parsed["do_reject"] is False
    assert parsed["share_comps"] is False
    assert parsed["use_min_dll"] is True


def test_list_valued_keys(parsed):
    assert parsed["files"] == ["./eeglab_data.fdt"]
    assert parsed["field_dim"] == [30504]


def test_unrecognized_keys_warn_and_are_dropped(caplog):
    """input.param carries three keywords amica15.f90 does not actually parse
    (field_blocksize, doPCA, load_W -- none is a `case(...)` arm), so the
    reference binary silently ignores them too; this reader must not."""
    with caplog.at_level(logging.WARNING, logger="pamica.fortran_params"):
        got = read_fortran_param_file(PARAM_FILE)
    assert "field_blocksize" not in got
    assert "doPCA" not in got
    assert "load_W" not in got
    warnings = "\n".join(r.message for r in caplog.records)
    assert "field_blocksize" in warnings
    assert "doPCA" in warnings
    assert "load_W" in warnings


def test_unsupported_keys_warn_and_are_dropped(caplog):
    """blk_min etc. are real Fortran keywords with no pamica equivalent."""
    with caplog.at_level(logging.WARNING, logger="pamica.fortran_params"):
        got = read_fortran_param_file(PARAM_FILE)
    assert "blk_min" not in got
    warnings = "\n".join(r.message for r in caplog.records)
    assert "blk_min" in warnings
    assert "load_rho" in warnings


def test_mapping_tables_partition_every_known_fortran_key():
    """Every keyword amica15.f90 accepts is classified exactly once: mapped
    or deliberately unsupported, never both, never neither. This is the
    systematic-sweep guard: it fails the moment amica15.f90 grows a keyword
    neither table has classified yet."""
    accepted = _fortran_accepted_keys()
    assert accepted is not None
    mapped = set(FORTRAN_TO_PAMICA_KEY)
    unsupported = set(FORTRAN_UNSUPPORTED_KEYS)
    assert mapped & unsupported == set()
    assert accepted == mapped | unsupported


def test_missing_source_disables_unrecognized_filtering(tmp_path, monkeypatch):
    """Without the reference source, every keyword in the file is treated as
    Fortran-accepted (mirrors validate_implementations.fortran_accepted_keys);
    only the supported/unsupported split still applies."""
    import pamica.fortran_params as fp

    monkeypatch.setattr(fp, "_DEFAULT_SOURCE", tmp_path / "does-not-exist.f90")
    dest = tmp_path / "input.param"
    dest.write_text("block_size 1024\nnot_a_real_keyword 1\n")
    got = fp.read_fortran_param_file(dest)
    assert got["block_size"] == 1024
    assert "not_a_real_keyword" not in got  # still has no entry in either table


def test_repeated_key_last_line_wins(tmp_path):
    dest = tmp_path / "input.param"
    dest.write_text("num_mix_comps 3\nnum_mix 5\n")
    assert read_fortran_param_file(dest)["num_mix"] == 5


def test_comments_and_blank_lines_are_skipped(tmp_path):
    dest = tmp_path / "input.param"
    dest.write_text("# a full-line comment\n\nblock_size 2048\n   \n")
    assert read_fortran_param_file(dest) == {"block_size": 2048}


def test_out_of_range_flag_matches_fortran_semantics(tmp_path):
    """Fortran's own reader only treats an exact 1 as true (amica15.f90:
    `if (k == 1) then ... .true. else ... .false.`), so 2 is false, not an
    error."""
    dest = tmp_path / "input.param"
    dest.write_text("do_newton 2\n")
    assert read_fortran_param_file(dest)["do_newton"] is False


class TestMalformedLines:
    """Hard errors: a malformed line must never be silently dropped."""

    def test_key_with_no_value_raises(self, tmp_path):
        dest = tmp_path / "input.param"
        dest.write_text("block_size 512\ndo_newton\n")
        with pytest.raises(ValueError, match="malformed line"):
            read_fortran_param_file(dest)

    def test_unparsable_int_value_raises(self, tmp_path):
        dest = tmp_path / "input.param"
        dest.write_text("block_size not_a_number\n")
        with pytest.raises(ValueError, match="block_size"):
            read_fortran_param_file(dest)

    def test_unparsable_float_value_raises(self, tmp_path):
        dest = tmp_path / "input.param"
        dest.write_text("rho0 not_a_float\n")
        with pytest.raises(ValueError, match="rho0"):
            read_fortran_param_file(dest)


class TestFromParamsFileIntegration:
    """The `.param` format reaches the exact same construction path as JSON
    (issue #132's "least-API-invasive" integration: format auto-detection in
    AMICA.from_params_file, no new public method)."""

    def test_param_extension_is_detected(self):
        model = AMICA.from_params_file(str(PARAM_FILE))
        assert model.n_models == 1
        assert model.n_mix == 3

    def test_kwargs_still_override_the_param_path(self):
        model = AMICA.from_params_file(str(PARAM_FILE), n_mix=7)
        assert model.n_mix == 7

    def test_param_and_json_agree_on_model_construction(self):
        from_param = AMICA.from_params_file(str(PARAM_FILE))
        from_json = AMICA.from_params_file(str(JSON_FILE))
        assert from_param.n_models == from_json.n_models
        assert from_param.n_mix == from_json.n_mix


def test_param_and_json_agree_on_overlapping_settings():
    """Round trip: sample_data/input.param and sample_data/sample_params.json
    describe the same reference run, so every setting present in both must
    agree once the Fortran reader's renames are applied. `files` is excluded:
    the two files spell the same data file's path relative to different
    working directories (input.param relative to sample_data/, sample_params.json
    relative to the repo root), which is a path-context difference, not a
    parameter-translation one."""
    from_param = read_fortran_param_file(PARAM_FILE)
    from_json = json.loads(JSON_FILE.read_text())

    common = sorted((set(from_param) & set(from_json)) - {"files"})
    assert len(common) >= 40  # sanity: most settings really do overlap
    mismatched = {
        key: (from_param[key], from_json[key])
        for key in common
        if from_param[key] != from_json[key]
    }
    assert mismatched == {}
