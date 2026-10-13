"""Tests for ``native/check_glibc_floor.sh`` (issue #370).

The release job runs the script on each Linux binary to keep the glibc floor at
2.28. The tests run it against REAL executables: the Python interpreter that is
running the tests, ``bash``, and, where a C compiler exists, tiny programs built
with the link options the script has to understand. No binary is faked: a floor
far above anything the interpreter needs must pass, and a floor below its oldest
symbol version must fail and name the offending symbols.
"""

import platform
import re
import shutil
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "native" / "check_glibc_floor.sh"

pytestmark = pytest.mark.skipif(
    platform.system() != "Linux"
    or shutil.which("bash") is None
    or (shutil.which("readelf") is None and shutil.which("objdump") is None),
    reason="needs Linux with bash and binutils (readelf or objdump)",
)


def _check(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT), *args], capture_output=True, text=True, check=False
    )


def _interpreter() -> str:
    return str(Path(sys.executable).resolve())


def _bash() -> str:
    return str(Path(shutil.which("bash") or "bash").resolve())


def _is_static(path: str) -> bool:
    return "is static" in _check(path, "99.0").stdout


def _compile(tmp_path: Path, name: str, *flags: str) -> str:
    """Build a one-line C program with the given link flags, or skip the test."""
    cc = shutil.which("gcc") or shutil.which("cc")
    if cc is None:
        pytest.skip("needs a C compiler")
    src = tmp_path / f"{name}.c"
    src.write_text("int main(void) { return 0; }\n")
    out = tmp_path / name
    r = subprocess.run(
        [cc, "-o", str(out), str(src), *flags],
        capture_output=True,
        text=True,
        check=False,
    )
    if r.returncode != 0:
        pytest.skip(
            f"this toolchain cannot link with {' '.join(flags)}: {r.stderr[-200:]}"
        )
    return str(out)


def test_floor_far_above_passes():
    r = _check(_interpreter(), "99.0")
    assert r.returncode == 0, r.stderr
    assert "glibc floor OK" in r.stdout


def test_floor_below_oldest_version_fails_and_names_symbols():
    if _is_static(_interpreter()):
        pytest.skip("interpreter is statically linked; no glibc symbols to check")
    # Every dynamically linked glibc program references at least GLIBC_2.2.5 (x86-64)
    # or GLIBC_2.17 (aarch64), so a floor of 2.0 is always exceeded.
    r = _check(_interpreter(), "2.0")
    assert r.returncode == 1
    assert "above the floor 2.0" in r.stderr
    assert re.search(r"\(GLIBC_\d+\.\d+", r.stderr), r.stderr


@pytest.mark.parametrize("binary", [_interpreter, _bash], ids=["interpreter", "bash"])
def test_failing_check_exits_1_under_load(binary):
    # Regression for a race: the symbol listing once ended in `head -5`, and under
    # pipefail the SIGPIPE the upstream stages took when head exited made the script
    # exit 141 instead of 1 -- only when scheduling let `sort` write after head had
    # gone, about 3% of runs under load on a binary with hundreds of symbols. One run
    # rarely shows it; many concurrent runs do. Two binaries, because how many symbols
    # share a version differs between them.
    path = binary()
    if _is_static(path):
        pytest.skip("statically linked; no glibc symbols to check")
    with ThreadPoolExecutor(max_workers=24) as pool:
        codes = list(pool.map(lambda _: _check(path, "2.0").returncode, range(160)))
    assert set(codes) == {1}, Counter(codes)


def test_reports_the_highest_version_and_compares_numerically():
    if _is_static(_interpreter()):
        pytest.skip("interpreter is statically linked; no glibc symbols to check")
    r = _check(_interpreter(), "99.0")
    m = re.search(r"needs glibc (\d+(?:\.\d+)+)", r.stdout)
    assert m, r.stdout
    highest = m.group(1)
    # The version the script reports is itself an accepted floor.
    assert _check(_interpreter(), highest).returncode == 0
    # Versions compare as numbers: 2.9 is older than 2.17, although the text "2.9"
    # sorts after "2.17". Every glibc interpreter build needs at least 2.10.
    if tuple(int(x) for x in highest.split(".")) >= (2, 10):
        assert _check(_interpreter(), "2.9").returncode == 1


def test_static_binary_has_no_floor_to_exceed(tmp_path):
    binary = _compile(tmp_path, "static_prog", "-static")
    r = _check(binary, "2.0")
    assert r.returncode == 0, r.stderr
    assert "is static" in r.stdout


def test_packed_relative_relocations_count_as_glibc_2_36(tmp_path):
    binary = _compile(tmp_path, "relr_prog", "-Wl,-z,pack-relative-relocs")
    # The linker marks such a binary with the version GLIBC_ABI_DT_RELR among its version
    # requirements (not among its symbols), and some linkers ignore the option.
    inspector, flags = (
        ("readelf", ["-V"]) if shutil.which("readelf") else ("objdump", ["-p"])
    )
    probe = subprocess.run(
        [inspector, *flags, binary], capture_output=True, text=True, check=False
    )
    if "GLIBC_ABI_DT_RELR" not in probe.stdout:
        pytest.skip(
            "this toolchain does not mark packed relocations with GLIBC_ABI_DT_RELR"
        )
    # The marker carries no number, so a regex for GLIBC_<digits> alone would pass it.
    r = _check(binary, "2.35")
    assert r.returncode == 1
    assert "DT_RELR" in r.stderr
    assert _check(binary, "2.36").returncode == 0


def test_truncated_dynamic_binary_is_never_reported_as_passing(tmp_path):
    if _is_static(_interpreter()):
        pytest.skip("interpreter is statically linked")
    cut = tmp_path / "truncated"
    cut.write_bytes(Path(_interpreter()).read_bytes()[:200])
    r = _check(str(cut), "99.0")
    assert r.returncode == 2, (r.stdout, r.stderr)
    assert "OK" not in r.stdout


def test_needed_allowlist_passes_when_every_library_matches():
    if _is_static(_interpreter()):
        pytest.skip("interpreter is statically linked; nothing is needed")
    r = _check(_interpreter(), "99.0", ".*")
    assert r.returncode == 0, r.stderr
    assert "shared libraries OK" in r.stdout


def test_needed_allowlist_fails_and_names_the_library():
    if _is_static(_interpreter()):
        pytest.skip("interpreter is statically linked; nothing is needed")
    r = _check(_interpreter(), "99.0", "^nothing-matches$")
    assert r.returncode == 1
    assert "outside the allowed set" in r.stderr
    assert re.search(r"lib\w+\.so", r.stderr), r.stderr


@pytest.mark.parametrize(
    "args",
    [(), ("only-one-argument",), ("/no/such/file", "2.28"), ("a", "2.28", "b", "c")],
    ids=["no-arguments", "one-argument", "missing-file", "four-arguments"],
)
def test_usage_errors_exit_2(args):
    r = _check(*args)
    assert r.returncode == 2


def test_rejects_a_malformed_floor():
    r = _check(_interpreter(), "twenty-eight")
    assert r.returncode == 2
    assert "floor must look like" in r.stderr


def test_rejects_a_file_that_is_not_elf(tmp_path):
    text = tmp_path / "not-a-binary.txt"
    text.write_text("plain text\n")
    r = _check(str(text), "2.28")
    assert r.returncode == 2
    assert "not an ELF binary" in r.stderr
