#!/usr/bin/env bash
# Build a dependency-free native amica15 (epic #165, phase 1).
#
# Extends sccn/amica PR #54's vendor-neutral recipe (gfortran, no MKL, portable
# + reproducible random_seed, vmath_shim for vrda_exp/vrda_log) with a single-rank MPI shim
# (mpi_single.f90 + mpi_single.c), so the binary links NO MPI runtime and is
# self-contained. Runtime dependencies after this are only the C/Fortran runtime
# and LAPACK/BLAS, which we static-link where possible.
#
# Usage:
#   bash native/build.sh                 # shim build (default): gfortran, no MPI
#   MPI_MODE=mpi bash native/build.sh    # reference build: mpif90 + real Open MPI
#   OUT=path LAPACK_LIBS="..." FC=... CC=...   # overrides
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$here/.." && pwd)"
src_dir="${AMICA_SRC:-$repo_root/pamica}"
mode="${MPI_MODE:-shim}"
build_dir="$here/build_$mode"
out="${OUT:-$here/amica15_$mode}"
os="$(uname -s)"
# `cc` is not always present (MSYS2 mingw ships `gcc`, not `cc`); fall back.
cc="${CC:-$(command -v cc 2>/dev/null || command -v gcc 2>/dev/null || echo cc)}"
# patch_sources.py needs a Python; MSYS2 mingw calls it `python`, not `python3`.
python="${PYTHON:-$(command -v python3 2>/dev/null || command -v python 2>/dev/null || echo python3)}"

if [[ "$mode" == "shim" ]]; then
  fc="${FC:-gfortran}"
else
  fc="${FC:-mpif90}"
fi

# LAPACK/BLAS: Accelerate on macOS (a system framework, always present -- no
# install, so it does not count against "dependency-free"); reference LAPACK on
# Linux otherwise. The release job overrides it via LAPACK_LIBS with static OpenBLAS
# on Linux and Windows.
if [[ -n "${LAPACK_LIBS:-}" ]]; then
  lapack_libs="$LAPACK_LIBS"
elif [[ "$os" == "Darwin" ]]; then
  lapack_libs="-framework Accelerate"
else
  lapack_libs="-llapack -lblas"
fi

echo "amica native build ($mode)"
echo "  platform : $os $(uname -m)"
echo "  compiler : $fc"
echo "  lapack   : $lapack_libs"
echo "  output   : $out"

command -v "$fc" >/dev/null 2>&1 || { echo "ERROR: '$fc' not found" >&2; exit 1; }

rm -rf "$build_dir"
mkdir -p "$build_dir/src"
work="$build_dir/src"
cp "$src_dir/funmod2.f90" "$src_dir/amica15.f90" "$src_dir/amica15_header.f90" "$work/"

# --- Patch the build copy only (the tracked sources stay the parity reference).
"$python" "$here/patch_sources.py" "$work/amica15.f90" "$work/amica15_header.f90"

set -x
# Vector-math shim (vrda_exp/vrda_log as libm loops); portable, no AMD LibM.
"$cc" -O3 -c "$here/vmath_shim.c" -o "$build_dir/vmath_shim.o"

extra_obj=("$build_dir/vmath_shim.o")
mpi_mod_inc=()
if [[ "$mode" == "shim" ]]; then
  # Single-rank MPI shim: the `mpi` module (constants) + the C stubs. Compile the
  # module first so `use mpi` in amica15 resolves to it (not a real MPI install).
  "$fc" -O3 -cpp -J"$build_dir" -c "$here/mpi_single.f90" -o "$build_dir/mpi_single_mod.o"
  "$cc" -O3 -c "$here/mpi_single.c" -o "$build_dir/mpi_single.o"
  extra_obj+=("$build_dir/mpi_single_mod.o" "$build_dir/mpi_single.o")
  mpi_mod_inc=(-I"$build_dir")
fi

# funmod2 first (amica15 does `use funmod2`); -J puts .mod files in build_dir.
# -cpp resolves the source's #ifdef MKL (undefined -> plain-Fortran branches);
# -std=legacy + -fallow-argument-mismatch relax this ifort-vintage F90; the
# generically-typed MPI calls need -fallow-argument-mismatch in shim mode too.
common_f=(-O3 -fopenmp -cpp -ffree-line-length-none -std=legacy
          -fallow-argument-mismatch -J"$build_dir" "${mpi_mod_inc[@]}" -I"$work")
"$fc" "${common_f[@]}" -c "$work/funmod2.f90" -o "$build_dir/funmod2.o"
# -static-libgfortran/-libgcc bundle the Fortran/GCC runtime so the binary does
# not need a matching gfortran install. EXTRA_LDFLAGS carries per-platform static
# linkage from the release matrix (e.g. -static-libquadmath, static OpenBLAS, or a
# full -static on Linux); LAPACK_LIBS is the LAPACK/BLAS linkage.
# shellcheck disable=SC2086  # $lapack_libs/$EXTRA_LDFLAGS must word-split
"$fc" "${common_f[@]}" \
  "$work/amica15.f90" "$build_dir/funmod2.o" "${extra_obj[@]}" \
  -o "$out" \
  -static-libgfortran -static-libgcc ${EXTRA_LDFLAGS:-} $lapack_libs
set +x

# Self-verify the shim build links NO real MPI runtime (the whole point). Uses
# whichever inspector the host has; a hit means a stray system MPI leaked onto
# the search path and shadowed the shim, so fail loudly rather than ship it.
if [[ "$mode" == "shim" ]]; then
  deps=""
  if command -v otool >/dev/null 2>&1; then deps="$(otool -L "$out" 2>/dev/null)"
  elif command -v ldd >/dev/null 2>&1; then deps="$(ldd "$out" 2>/dev/null)"
  elif command -v objdump >/dev/null 2>&1; then deps="$(objdump -p "$out" 2>/dev/null | grep -i 'DLL Name\|NEEDED')"
  fi
  if [[ -n "$deps" ]] && grep -qiE 'libmpi|msmpi|open-mpi|mpich' <<<"$deps"; then
    echo "ERROR: shim build links a real MPI runtime (should link none):" >&2
    grep -iE 'libmpi|msmpi|open-mpi|mpich' <<<"$deps" >&2
    exit 1
  fi
fi

echo
echo "built: $out"
