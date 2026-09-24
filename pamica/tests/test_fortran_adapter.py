"""Structural tests for the native-Fortran benchmark adapter (issue #85).

Real sample EEG only (NO MOCKS): the ``.fdt`` round-trip and the ``out.txt``
parser are checked against the committed EEGLAB data and a committed amica run.
The end-to-end run test is skipped unless a runnable native amica binary is
present (built on the CUDA host via ``benchmarks/fortran/build_amica.sh``, or
pointed at by ``AMICA_FORTRAN_BIN``), so CI and Apple-only checkouts stay green.
"""

import importlib.util
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from pamica.numpy_impl.data import load_data_file

_REPO = Path(__file__).resolve().parents[2]
SAMPLE_DIR = _REPO / "pamica" / "sample_data"
FDT = SAMPLE_DIR / "eeglab_data.fdt"
OUT_TXT = SAMPLE_DIR / "amicaout" / "out.txt"
DATA_DIM, FIELD_DIM = 32, 30504


def _load_bench():
    """Load benchmarks/benchmark_dimsweep.py (not an installed package)."""
    path = _REPO / "benchmarks" / "benchmark_dimsweep.py"
    spec = importlib.util.spec_from_file_location("benchmark_dimsweep", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bench = _load_bench()

# A runnable native amica binary if one is available (built by
# benchmarks/fortran/build_amica.sh, or pointed at by AMICA_FORTRAN_BIN). Absent
# on CI / Apple-only checkouts, which is why the run tests below are skip-gated.
_FBIN = os.environ.get("AMICA_FORTRAN_BIN")


def test_write_fdt_roundtrip(tmp_path):
    """_write_fdt is the exact inverse of the canonical load_data_file, so
    re-writing the committed .fdt reproduces it byte-for-byte -- locking the
    float32 column-major (channel-fastest) convention."""
    data = load_data_file(str(FDT), DATA_DIM, FIELD_DIM, dtype=np.float32)
    assert data.shape == (DATA_DIM, FIELD_DIM)
    out = tmp_path / "roundtrip.fdt"
    bench._write_fdt(data, out)
    assert out.read_bytes() == FDT.read_bytes()


def test_write_fdt_channel_subset(tmp_path):
    """A channel subset writes channel-fastest, so the canonical reader reads it
    back exactly (the sweep slices full[:nc, :ns])."""
    data = load_data_file(str(FDT), DATA_DIM, FIELD_DIM, dtype=np.float32)
    nc, ns = 16, 5000
    sub = np.ascontiguousarray(data[:nc, :ns])
    out = tmp_path / "subset.fdt"
    bench._write_fdt(sub, out)
    back = load_data_file(str(out), nc, ns, dtype=np.float32)
    np.testing.assert_array_equal(back, sub)


def test_parse_fortran_out():
    """Parse the committed amica out.txt -> per-iter seconds + final LL."""
    secs, ll = bench._parse_fortran_out(OUT_TXT)
    assert len(secs) == 200  # max_iter of the committed run
    assert all(s >= 0.0 for s in secs)
    assert secs[0] > 0.0
    assert ll == pytest.approx(-3.4018729891, abs=1e-6)


def test_write_fortran_param(tmp_path):
    """The rendered param carries the matched benchmark knobs for the subset,
    and forces the full fixed-length iteration budget."""
    bench._write_fortran_param(
        tmp_path, nc=16, ns=5000, iters=30, threads=4, n_models=1
    )
    text = (tmp_path / "input.param").read_text()
    for expected in (
        "files ./bench.fdt",
        "data_dim 16",
        "field_dim 5000",
        "max_iter 30",
        "max_threads 4",
        "num_mix_comps 3",
        "pdftype 0",
        "do_newton 0",
        "block_size 512",
        "pcakeep 16",  # must track nc, not the template's 32 (the load-bearing subset line)
        "use_min_dll 0",  # run the full budget, matched to torch/mlx
        "use_grad_norm 0",
    ):
        assert expected in text, f"missing {expected!r} in rendered param"


def test_write_fortran_param_multimodel(tmp_path):
    """num_models tracks n_models (amica15 supports multi-model; the adapter
    forwards it, gating only component sharing off)."""
    bench._write_fortran_param(
        tmp_path, nc=32, ns=5000, iters=10, threads=4, n_models=2
    )
    text = (tmp_path / "input.param").read_text()
    assert "num_models 2" in text
    assert "share_comps 0" in text


def test_fortran_availability_and_dispatch():
    """Availability gating is honest without a binary: a missing binary is
    unavailable, and component sharing is always gated off for this backend
    (these decide whether native-fortran-f64 joins the auto-detected sweep)."""
    assert bench._fortran_available("/nonexistent/amica15") is False
    assert (
        bench._available("native-fortran-f64", share=True, fortran_bin="/nonexistent")
        is False
    )
    # sharing is refused even if a binary exists
    assert (
        bench._available("native-fortran-f64", share=True, fortran_bin=_FBIN) is False
    )


def test_run_fortran_raises_on_nonzero_exit(tmp_path):
    """A crashed run must raise, never return a bogus fast timing. Uses the real
    system `false` (exits nonzero) as the binary -- a real failing process, not a
    mock."""
    false_bin = shutil.which("false")
    if false_bin is None:
        pytest.skip("system 'false' not found")
    data = load_data_file(str(FDT), DATA_DIM, FIELD_DIM, dtype=np.float32)[:16, :2000]
    with pytest.raises(RuntimeError, match="native fortran run failed"):
        bench._run_fortran(
            np.ascontiguousarray(data), iters=5, repeats=1, binary=false_bin, threads=1
        )


@pytest.mark.skipif(
    not bench._fortran_available(_FBIN),
    reason="no runnable native amica binary (build via "
    "benchmarks/fortran/build_amica.sh or set AMICA_FORTRAN_BIN); skipped on "
    "CI / Apple-only checkouts",
)
def test_run_fortran_smoke():
    """End-to-end: a short native fit on a real full-width subset yields finite
    per-iteration timing and a sane converged LL. Uses the full 32 channels /
    all samples so per-iteration time clears amica's ~10 ms stamp resolution."""
    # the committed .fdt is float32 on disk (byte_size 4); _write_fdt recasts to
    # float32 anyway, so read it at its real dtype.
    data = load_data_file(str(FDT), DATA_DIM, FIELD_DIM, dtype=np.float32)
    ms, ll = bench._run_fortran(data, iters=10, repeats=1, binary=_FBIN, threads=4)
    assert ms > 0.0
    assert -5.0 < ll < -2.0


@pytest.mark.skipif(
    not bench._fortran_available(_FBIN),
    reason="needs a runnable native amica binary; skipped on CI / Apple-only checkouts",
)
def test_run_fortran_never_reports_a_bogus_zero():
    """The 0.00 s stamp guard's actual contract: for a config at amica's ~10 ms
    out.txt stamp floor, ``_run_fortran`` either raises the named
    sub-resolution error or returns a genuinely nonzero mean -- never a bogus
    0.0 ms/iter. 8ch x 2000 samples sits at the floor: on fast hardware every
    stamp rounds to 0.00 and the guard fires, but on a loaded runner a single
    iteration can cross a 0.01 stamp and the mean is then a legitimate
    nonzero (the 2026-08-30 weekly-run flake, issue #308), so both outcomes
    are asserted instead of betting on host speed."""
    data = load_data_file(str(FDT), DATA_DIM, FIELD_DIM, dtype=np.float32)[:8, :2000]
    try:
        ms_per_iter, _ = bench._run_fortran(
            np.ascontiguousarray(data), iters=8, repeats=1, binary=_FBIN, threads=2
        )
    except RuntimeError as exc:
        assert "rounds to 0.00 s" in str(exc)
    else:
        # Stamps are multiples of 0.01 s, so a non-raising mean over the 7
        # timed iters is at least 10/7 ms; a returned 0.0 is exactly the bug
        # the guard exists to prevent.
        assert ms_per_iter >= 10.0 / 7 - 1e-9


@pytest.mark.skipif(
    not bench._fortran_available(_FBIN),
    reason="needs a runnable native amica binary; skipped on CI / Apple-only checkouts",
)
def test_run_fortran_requires_two_timed_iters():
    """A single-iteration run cannot yield a per-iteration mean (iter 1 is the
    dropped warmup), so it raises rather than dividing by an empty set."""
    data = load_data_file(str(FDT), DATA_DIM, FIELD_DIM, dtype=np.float32)
    with pytest.raises(RuntimeError, match="<2 timed iterations"):
        bench._run_fortran(data, iters=1, repeats=1, binary=_FBIN, threads=4)
