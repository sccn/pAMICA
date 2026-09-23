"""Parity harness (``validate_implementations.py``): parameter forwarding
(issue #228) and per-backend dispatch (issue #315).

Both arms of a parity run must be configured identically. The writer previously
rewrote six hardcoded keys and left everything else at the template's value while
the Python side honored it, which silently makes the comparison uncontrolled.
The same holds across backends: each one the harness runs must receive the same
settings, which the dispatch tests below pin.
"""

import os
import sys
from pathlib import Path

import numpy as np
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


def test_canonical_names_are_also_translated(tmp_path):
    """Issue #304: ``_FORTRAN_ALIASES`` now also accepts pamica's canonical
    key spellings (``maxdecs``/``min_nd``), not just the JSON-schema aliases
    the test above covers -- ``load_sample_data`` hands ``params`` through
    already canonical-keyed (see the end-to-end test below)."""
    got = _written(tmp_path, {"maxdecs": 4, "min_nd": 0.0001})
    assert got["max_decs"] == "4"
    assert got["min_grad_norm"] == "0.0001"


def test_load_sample_data_writes_fortran_spellings_end_to_end(tmp_path):
    """The production call sequence: ``load_sample_data()`` (canonical-keyed
    via ``read_params_file``, issue #304) feeding straight into
    ``write_fortran_param_file``, exactly as ``run_fortran_amica`` does. The
    written file must carry Fortran's own spellings, never the canonical
    ones ``write_fortran_param_file`` translated them from."""
    from validate_implementations import load_sample_data

    _, params = load_sample_data()
    got = _written(tmp_path, params)
    for fortran_key in ("max_decs", "min_grad_norm", "share_iter", "numrej"):
        assert fortran_key in got
    for canonical_key in ("maxdecs", "min_nd", "share_int", "maxrej"):
        assert canonical_key not in got


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


def test_real_params_json_produces_a_parseable_file(tmp_path):
    """Regression: the full params.json, not a hand-picked subset.

    `files` and `field_dim` are lists there, and writing Python's repr put
    brackets in the file, which aborts the Fortran parser at read time. The
    unit tests above all passed while the harness could not run.
    """
    import json

    params = json.loads(
        (ROOT / "pamica" / "sample_data" / "sample_params.json").read_text()
    )
    got = _written(tmp_path, params, overrides={"seed": 42, "max_threads": 1})
    for key, value in got.items():
        assert "[" not in value and "]" not in value, f"{key} kept Python list syntax"
    assert got["field_dim"] == "30504"
    assert got["seed"] == "42"
    assert got["max_threads"] == "1"


def test_list_values_are_space_separated(tmp_path):
    got = _written(tmp_path, {"field_dim": [100, 200]})
    assert got["field_dim"] == "100 200"


def test_default_reference_binary_falls_back_loudly(capsys, monkeypatch):
    """Without a native engine the harness must say the run is uncontrolled.

    The fallback binary cannot be seeded, so silently using it would report a
    comparison against a random draw as if it were controlled (issue #228).
    """
    from validate_implementations import LEGACY_BINARY, default_reference_binary

    monkeypatch.delenv("PAMICA_NATIVE_BINARY", raising=False)
    got = default_reference_binary(download=False)
    if got == LEGACY_BINARY:
        out = capsys.readouterr().out
        assert "cannot be seeded" in out and "#228" in out
    else:  # a cached native engine is present; that is the good path
        assert got.exists()


def test_env_override_selects_the_native_engine(tmp_path, monkeypatch):
    fake = tmp_path / "amica15_native"
    fake.write_text("#!/bin/sh\n")
    monkeypatch.setenv("PAMICA_NATIVE_BINARY", str(fake))
    from validate_implementations import default_reference_binary

    assert default_reference_binary(download=False) == fake.resolve()


# --- per-backend dispatch (issue #315) ---------------------------------------
#
# The harness runs torch, NumPy and MLX against the same reference. These are
# the cheap checks (a couple of iterations on a slice of the real sample, no
# Fortran binary): argument parsing, that every runner maps the same canonical
# params onto its backend, and that the report/summary files land where the
# docs say. The Fortran-backed comparison is the AMICA_RUN_FORTRAN-gated test
# at the end of this module.

_DISPATCH_SAMPLES = 4096


def _mlx_or_skip():
    """Skip unless MLX imports and sees an Apple GPU, like the MLX suites."""
    mx = pytest.importorskip("mlx.core")
    if mx.default_device().type != mx.DeviceType.gpu:
        pytest.skip("no Apple GPU")


@pytest.fixture(scope="module")
def dispatch_input():
    """The harness's own data and params, capped at two iterations."""
    from validate_implementations import load_sample_data

    data, params = load_sample_data()
    params["max_iter"] = 2
    return data[:, :_DISPATCH_SAMPLES], params


@pytest.fixture(scope="module")
def dispatch_out(tmp_path_factory):
    """The ``output_dir`` every dispatch run below is handed."""
    return tmp_path_factory.mktemp("dispatch")


@pytest.fixture(scope="module")
def dispatch_results(dispatch_input, dispatch_out):
    """One two-iteration run per available backend, shared by the checks below."""
    from validate_implementations import BACKENDS, mlx_unavailable_reason, run_backend

    data, params = dispatch_input
    return {
        name: run_backend(name, data, dict(params), dispatch_out, 42)
        for name in BACKENDS
        if name != "mlx" or mlx_unavailable_reason() is None
    }


def test_backend_flag_parses_names_lists_and_all():
    from validate_implementations import BACKENDS, build_parser, parse_backends

    assert parse_backends("torch") == ["torch"]
    assert parse_backends("all") == list(BACKENDS)
    # Order is the caller's, duplicates dropped, whitespace tolerated.
    assert parse_backends("mlx, torch,mlx") == ["mlx", "torch"]
    parser = build_parser()
    assert parser.parse_args([]).backend is None  # the historical torch-only run
    assert parser.parse_args(["--backend", "numpy,torch"]).backend == [
        "numpy",
        "torch",
    ]


@pytest.mark.parametrize("value", ["cuda", "torch,", "", "ALL"])
def test_backend_flag_rejects_unknown_names(value, capsys):
    from validate_implementations import build_parser

    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["--backend", value])
    assert exc.value.code == 2
    assert "choose from torch, numpy, mlx" in capsys.readouterr().err


@pytest.mark.parametrize("backend", ["torch", "numpy", "mlx"])
def test_every_backend_returns_the_comparison_contract(backend, dispatch_results):
    """Each runner returns what compare_results reads, in the Fortran layout:
    ``W`` the true unmixing and ``A`` the true mixing, so ``A @ W == I`` for a
    full-rank single-model fit (a transposed ``A`` would fail this)."""
    if backend == "mlx":
        _mlx_or_skip()
    res = dispatch_results[backend]
    assert res["final_iter"] == 2 == len(res["ll_history"])
    assert np.isfinite(res["final_ll"])
    assert res["W"].shape == res["A"].shape == (32, 32)
    assert res["W"].dtype == res["A"].dtype == np.float64
    atol = 1e-5 if backend == "mlx" else 1e-10
    np.testing.assert_allclose(res["A"] @ res["W"], np.eye(32), atol=atol)
    assert res["runtime_s"] > 0


def test_torch_and_numpy_runners_apply_the_same_params(dispatch_results):
    """Anti-drift: the two float64 backends are the same trajectory to ~1e-13,
    so any setting one runner maps and the other drops shows up here."""
    torch_res, numpy_res = dispatch_results["torch"], dispatch_results["numpy"]
    np.testing.assert_allclose(
        numpy_res["ll_history"], torch_res["ll_history"], rtol=1e-10
    )
    np.testing.assert_allclose(numpy_res["W"], torch_res["W"], atol=1e-10)


def test_mlx_runner_applies_the_same_params_as_torch(dispatch_results):
    """The same anti-drift check for the float32 MLX runner, at float32
    tolerance (measured 4e-7 on the likelihood, 8e-7 on ``W``)."""
    _mlx_or_skip()
    torch_res, mlx_res = dispatch_results["torch"], dispatch_results["mlx"]
    np.testing.assert_allclose(
        mlx_res["ll_history"], torch_res["ll_history"], atol=1e-5
    )
    np.testing.assert_allclose(mlx_res["W"], torch_res["W"], atol=1e-5)


def test_numpy_runner_writes_under_the_output_dir(dispatch_results, dispatch_out):
    """The runner hands AMICA_NumPy the run's own output directory, so its
    ``out.txt`` log and final model files sit next to the reports (the backend
    writes nothing without an ``outdir``)."""
    assert "numpy" in dispatch_results
    assert (dispatch_out / "numpy_run" / "out.txt").exists()
    assert (dispatch_out / "numpy_run" / "W").exists()


def test_main_writes_one_report_per_backend_and_a_summary(tmp_path):
    """``main`` end to end without the reference: the PyTorch report keeps its
    historical name, other backends get their own, and an explicit
    ``--backend`` adds the one-row-per-backend summary."""
    from validate_implementations import main

    out = tmp_path / "out"
    argv = ["--backend", "torch,numpy", "--skip-fortran", "--max-iter", "1"]
    assert main([*argv, "--output-dir", str(out)]) == 0
    assert (out / "validation_report.txt").exists()
    assert (out / "validation_report_numpy.txt").exists()
    summary = (out / "parity_summary.md").read_text()
    assert "| PyTorch | float64 | 1 |" in summary
    assert "| NumPy | float64 | 1 |" in summary


def test_mlx_backend_without_mlx_exits_with_the_install_hint(
    tmp_path, monkeypatch, capsys
):
    """``--backend mlx`` on a host without MLX stops before any work, naming
    the extra to install. Where MLX is installed, the ImportError its absence
    raises is reproduced at the real import site (``None`` in
    ``sys.modules``); the harness code path itself is unmodified."""
    from validate_implementations import main

    try:
        import pamica.mlx_impl  # noqa: F401
    except ImportError:
        pass
    else:
        monkeypatch.setitem(sys.modules, "pamica.mlx_impl", None)
    out = tmp_path / "never"
    assert main(["--backend", "torch,mlx", "--output-dir", str(out)]) == 2
    err = capsys.readouterr().err
    assert "--backend mlx" in err and "uv sync --extra mlx" in err
    assert not out.exists(), "nothing may run before the MLX check"


# --- Fortran-backed per-backend parity (issue #315) --------------------------
#
# Opt-in, like test_ng_pdf_families.py's converged-LL parity test: it runs the
# reference binary (resolved the way the harness resolves it) and every
# available backend at the harness's default settings, then holds each to the
# bars docs/guides/validation.md records for the harness rows.


@pytest.fixture(scope="module")
def fortran_parity_rows(tmp_path_factory):
    from validate_implementations import (
        BACKENDS,
        compare_results,
        load_sample_data,
        mlx_unavailable_reason,
        run_backend,
        run_fortran_amica,
    )

    data, params = load_sample_data()
    params["max_iter"] = 100
    out = tmp_path_factory.mktemp("fortran_parity")
    fortran = run_fortran_amica(data, params, out, seed=42)
    assert fortran is not None and "W" in fortran, "reference run failed"
    rows = {}
    for name in BACKENDS:
        if name == "mlx" and mlx_unavailable_reason() is not None:
            continue
        res = run_backend(name, data, dict(params), out, 42)
        rows[name] = (res, compare_results(fortran, res))
    return rows


@pytest.mark.skipif(
    os.environ.get("AMICA_RUN_FORTRAN") != "1",
    reason="opt-in Fortran-binary integration test (set AMICA_RUN_FORTRAN=1)",
)
@pytest.mark.parametrize("backend", ["torch", "numpy", "mlx"])
def test_backend_meets_its_parity_bar_against_fortran(backend, fortran_parity_rows):
    if backend == "mlx":
        _mlx_or_skip()
    res, cmp = fortran_parity_rows[backend]
    assert cmp["mean_correlation"] > 0.95
    assert cmp["amari_distance"] < 0.05
    assert cmp["ll_difference"] < 0.005
    if backend != "torch" and "torch" in fortran_parity_rows:
        # Every backend also lands on the float64 PyTorch likelihood: to the
        # ~1e-5 a full float64 NumPy fit is documented to track it at
        # (test_full_fit_parity_numpy_vs_ng), and to about five significant
        # digits for float32 MLX.
        torch_ll = fortran_parity_rows["torch"][0]["final_ll"]
        tol = 1e-4 if backend == "mlx" else 1e-5
        assert abs(res["final_ll"] - torch_ll) < tol
