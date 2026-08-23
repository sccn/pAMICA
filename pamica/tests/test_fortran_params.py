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

import numpy as np
import pytest

from pamica.amica import AMICA
from pamica.fortran_params import (
    FORTRAN_TO_PAMICA_KEY,
    FORTRAN_UNSUPPORTED_KEYS,
    _fortran_accepted_keys,
    read_fortran_param_file,
)
from pamica.torch_impl.utils import load_eeglab_data

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_data"
PARAM_FILE = SAMPLE_DIR / "input.param"
JSON_FILE = SAMPLE_DIR / "sample_params.json"
DATA_FILE = SAMPLE_DIR / "eeglab_data.fdt"
NW, FIELD = 32, 30504

pytestmark = pytest.mark.skipif(
    not PARAM_FILE.exists() or not JSON_FILE.exists(),
    reason="bundled sample_data/input.param or sample_params.json missing",
)


@pytest.fixture(scope="module")
def real_data() -> np.ndarray:
    if not DATA_FILE.exists():
        pytest.skip("sample data missing")
    return load_eeglab_data(str(DATA_FILE), data_dim=NW, field_dim=FIELD).astype(
        np.float64
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
    """The three keywords actually renamed to match AMICATorchNG's constructor
    spelling (review item 1): min_grad_norm/max_decs/numrej. share_iter is
    deliberately NOT renamed -- it already matches the constructor -- unlike
    sample_params.json's own (constructor-mismatched) "share_int" spelling."""
    assert "num_mix_comps" not in parsed
    assert parsed["num_mix"] == 3  # Fortran num_mix_comps -> num_mix
    assert "min_grad_norm" not in parsed
    assert parsed["min_nd"] == 1e-7  # Fortran min_grad_norm -> min_nd
    assert "max_decs" not in parsed
    assert parsed["maxdecs"] == 3  # Fortran max_decs -> maxdecs
    assert "numrej" not in parsed
    assert parsed["maxrej"] == 3  # Fortran numrej -> maxrej
    assert "share_int" not in parsed  # NOT the JSON schema's mismatched spelling
    assert parsed["share_iter"] == 100  # identity: matches the constructor


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


class TestInlineCommentsAndExponents:
    """Review item 4: deliberately more permissive than the Fortran parser,
    which only treats a line-leading `#` as a comment."""

    def test_inline_comment_is_stripped_from_scalar_value(self, tmp_path):
        dest = tmp_path / "input.param"
        dest.write_text("block_size 512 # my comment\n")
        assert read_fortran_param_file(dest)["block_size"] == 512

    def test_inline_comment_is_stripped_from_list_value(self, tmp_path):
        dest = tmp_path / "input.param"
        dest.write_text("field_dim 30504 30504 # two files\n")
        assert read_fortran_param_file(dest)["field_dim"] == [30504, 30504]

    def test_hash_without_leading_space_is_not_treated_as_a_comment(self, tmp_path):
        """Only ' #' (space then hash) is treated as an inline comment marker,
        per the module's documented, deliberately narrow rule."""
        dest = tmp_path / "input.param"
        dest.write_text("outdir ./amicaout#tag/\n")
        assert read_fortran_param_file(dest)["outdir"] == "./amicaout#tag/"

    def test_fortran_double_precision_exponent_uppercase(self, tmp_path):
        dest = tmp_path / "input.param"
        dest.write_text("rho0 1.5D+00\n")
        assert read_fortran_param_file(dest)["rho0"] == 1.5

    def test_fortran_double_precision_exponent_lowercase(self, tmp_path):
        dest = tmp_path / "input.param"
        dest.write_text("min_dll 1.0d-9\n")
        assert read_fortran_param_file(dest)["min_dll"] == 1e-9


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

    def test_value_that_is_only_an_inline_comment_raises(self, tmp_path):
        """ "do_newton  # just a comment" has no real value once the comment
        marker is stripped -- must not silently coerce "#comment" as a flag."""
        dest = tmp_path / "input.param"
        dest.write_text("do_newton  # oops, forgot the value\n")
        with pytest.raises(ValueError):
            read_fortran_param_file(dest)


class TestZeroRecognizedKeysGuard:
    """Review item 3, layer 2: never a silent all-defaults return when a
    non-empty file yields nothing usable (e.g. JSON handed to this reader by
    mistake, garbling every 'line' into an unrecognized keyword)."""

    def test_all_unrecognized_content_raises(self, tmp_path):
        dest = tmp_path / "input.param"
        payload = json.loads(JSON_FILE.read_text())
        dest.write_text(json.dumps(payload))  # single compact line, real content
        with pytest.raises(ValueError, match="not one keyword was recognized"):
            read_fortran_param_file(dest)

    def test_all_unsupported_content_does_not_raise(self, tmp_path):
        """A file of only known-but-unsupported keywords is a real (if
        useless) Fortran param file, not a format mismatch, so this stays a
        loud warning rather than an error -- only the "nothing at all was
        recognized" case is treated as likely-wrong-format."""
        dest = tmp_path / "input.param"
        dest.write_text("write_LLt 1\n")
        assert read_fortran_param_file(dest) == {}

    def test_genuinely_empty_file_does_not_raise(self, tmp_path):
        dest = tmp_path / "input.param"
        dest.write_text("")
        assert read_fortran_param_file(dest) == {}


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

    def test_param_extension_with_json_content_is_sniffed_as_json(self, tmp_path):
        """Review item 3: a file named .param but holding JSON content must
        still be parsed as JSON -- extension is never trusted over content."""
        dest = tmp_path / "sneaky.param"
        dest.write_text(JSON_FILE.read_text())
        model = AMICA.from_params_file(str(dest))
        reference = AMICA.from_params_file(str(JSON_FILE))
        assert model.n_models == reference.n_models
        assert model.n_mix == reference.n_mix

    def test_compact_json_named_param_does_not_silently_default(self, tmp_path):
        """The exact bug from review item 3: compact single-line JSON saved
        with a .param extension must not be silently mis-parsed as garbled
        Fortran text and returned as a default (num_models=1, num_mix=3)
        model when the file actually asks for something else."""
        payload = json.loads(JSON_FILE.read_text())
        payload = {**payload, "num_models": 2, "num_mix": 5}
        dest = tmp_path / "compact.param"
        dest.write_text(json.dumps(payload))
        model = AMICA.from_params_file(str(dest))
        assert model.n_models == 2
        assert model.n_mix == 5


class TestFitAppliesFileDefaults:
    """Review item 2: from_params_file's translated dict must actually reach
    the fitted backend, not just size n_models/n_mix. Evidence is read off
    the fitted AMICATorchNG instance, not just off the AMICA wrapper, so a
    default silently failing to forward would be caught here."""

    def test_file_defaults_reach_the_fitted_backend(self, real_data):
        """do_newton and block_size are chosen because they diverge from
        fit()'s/AMICATorchNG's own hard-coded defaults (do_newton=False,
        block_size=8192), so landing on the file's values (True, 512) is
        proof the file, not a coincidence, drove the run. max_iter is
        explicitly overridden to keep the test fast -- the file's own
        max_iter=2000 would not be; explicit-kwarg precedence is covered by
        the "wins" test below."""
        model = AMICA.from_params_file(str(PARAM_FILE), verbose=False)
        model.fit(real_data[:, :4096], max_iter=3, seed=0)
        backend = model.model_
        assert backend is not None
        assert backend.do_newton is True
        assert backend.block_size == 512
        assert backend.lrate0 == 0.05
        assert backend.newt_start == 50

    def test_explicit_kwargs_win_over_file_defaults(self, real_data):
        model = AMICA.from_params_file(str(PARAM_FILE), verbose=False)
        model.fit(
            real_data[:, :4096],
            max_iter=3,
            seed=0,
            do_newton=False,
            block_size=1024,
        )
        backend = model.model_
        assert backend is not None
        assert backend.do_newton is False
        assert backend.block_size == 1024

    def test_unhandled_file_settings_warn_once(self, real_data, caplog):
        """files/outdir/data_dim/... have no fit()/AMICATorchNG equivalent
        and must be named in a warning, not silently dropped, per the
        lead-decided design."""
        model = AMICA.from_params_file(str(PARAM_FILE), verbose=False)
        with caplog.at_level(logging.WARNING, logger="pamica.amica"):
            model.fit(real_data[:, :4096], max_iter=1, seed=0)
        warnings = "\n".join(r.message for r in caplog.records)
        assert "data_dim" in warnings
        assert "files" in warnings

    def test_fit_without_from_params_file_is_unaffected(self, real_data):
        """A plain AMICA(...) instance has no _file_params, so fit() must
        behave exactly as before (hard-coded defaults, no warning)."""
        model = AMICA(n_models=1, n_mix=3, device="cpu", verbose=False)
        model.fit(real_data[:, :2048], max_iter=2, block_size=1024, seed=0)
        assert model.model_ is not None
        assert model.model_.do_newton is False  # the ordinary hard default


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
    # 39 as of this writing (min_grad_norm/max_decs/share_int no longer
    # overlap by construction: this reader targets the constructor's
    # min_nd/maxdecs/share_iter spelling, which sample_params.json's own
    # schema does not use -- see FORTRAN_TO_PAMICA_KEY's module docstring).
    assert len(common) >= 35  # sanity: most settings really do overlap
    mismatched = {
        key: (from_param[key], from_json[key])
        for key in common
        if from_param[key] != from_json[key]
    }
    assert mismatched == {}
