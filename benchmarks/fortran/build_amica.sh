#!/usr/bin/env bash
# Build a native amica15 for the cross-platform benchmark (epic #84, phase #85).
#
# amica15.f90 is an MPI + OpenMP + LAPACK program, so this needs an MPI Fortran
# wrapper (mpif90) plus LAPACK/BLAS -- NOT plain gfortran. The resulting binary
# runs fine as a single MPI rank (OpenMPI singleton); OpenMP width is set at run
# time via OMP_NUM_THREADS. Building from source (not the bundled x86 amica15mac,
# which only runs under Rosetta on Apple Silicon) gives an honest native-CPU
# timing reference on BOTH the x86 CUDA host and Apple Silicon.
#
# Usage:
#   bash benchmarks/fortran/build_amica.sh
#   FC=mpiifort bash benchmarks/fortran/build_amica.sh          # override compiler
#   LAPACK_LIBS="-framework Accelerate" bash .../build_amica.sh  # override linkage
#   AMICA_SRC=/path/to/src bash benchmarks/fortran/build_amica.sh
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$here/../.." && pwd)"
src_dir="${AMICA_SRC:-$repo_root/pamica}"
fc="${FC:-mpif90}"
build_dir="$here/build"
out="$here/amica15"
os="$(uname -s)"

# LAPACK/BLAS linkage: apt libs on Linux; brew lapack (falling back to the
# Accelerate framework) on macOS. Override with LAPACK_LIBS=... .
if [[ -n "${LAPACK_LIBS:-}" ]]; then
  lapack_libs="$LAPACK_LIBS"
elif [[ "$os" == "Darwin" ]]; then
  if brew_lapack="$(brew --prefix lapack 2>/dev/null)" && [[ -d "$brew_lapack/lib" ]]; then
    lapack_libs="-L$brew_lapack/lib -llapack -lblas"
  else
    lapack_libs="-framework Accelerate"
  fi
else
  lapack_libs="-llapack -lblas"
fi

echo "amica native build"
echo "  platform : $os $(uname -m)"
echo "  compiler : $fc"
echo "  lapack   : $lapack_libs"
echo "  sources  : $src_dir/{funmod2,amica15}.f90"
echo "  output   : $out"

if ! command -v "$fc" >/dev/null 2>&1; then
  if [[ "$os" == "Darwin" ]]; then
    cat >&2 <<EOF

ERROR: '$fc' not found. amica15.f90 is an MPI+OpenMP+LAPACK program and needs an
MPI Fortran wrapper plus LAPACK/BLAS. On macOS (Homebrew, no sudo):

  brew install gcc open-mpi lapack

Then re-run this script (override the compiler with FC=... if it is not mpif90).
EOF
  else
    cat >&2 <<EOF

ERROR: '$fc' not found. amica15.f90 is an MPI+OpenMP+LAPACK program and needs an
MPI Fortran wrapper plus LAPACK/BLAS. On Debian/Ubuntu:

  sudo apt-get update
  sudo apt-get install -y gfortran libopenmpi-dev openmpi-bin liblapack-dev libblas-dev

Then re-run this script (override the compiler with FC=... if it is not mpif90).
EOF
  fi
  exit 1
fi

for f in funmod2.f90 amica15.f90; do
  if [[ ! -f "$src_dir/$f" ]]; then
    echo "ERROR: source not found: $src_dir/$f" >&2
    exit 1
  fi
done

mkdir -p "$build_dir"
work_src="$build_dir/src"
mkdir -p "$work_src"
cp "$src_dir/funmod2.f90" "$src_dir/amica15.f90" "$src_dir/amica15_header.f90" "$work_src/"

# No random_seed patch is needed anymore: since #235 the tracked source seeds
# portably itself (random_seed(size=nseed) then an allocated seedvec(nseed),
# sccn/amica PR #54), which gfortran accepts and which keeps the deterministic
# `seed`-param behavior the parity tests rely on. The old sed patch targeted the
# pre-#235 size-2 ifort-only seed line and would now both fail its anchor check
# and, if it matched, destroy the deterministic seeding. Verify the portable
# seeding is present so a future source change fails loudly here, not at runtime.
if ! grep -q 'call random_seed(size = nseed)' "$work_src/amica15.f90"; then
  echo "ERROR: expected portable random_seed(size=nseed) seeding in amica15.f90 (post-#235); source drifted" >&2
  exit 1
fi

# Portable vector-math shim: the non-MKL branch calls AMD LibM's vrda_exp/vrda_log,
# which are absent in a plain gfortran+LAPACK build. vmath_shim.c provides them as
# libm exp/log loops (see its header for the ABI + accuracy note).
cc="${CC:-cc}"
set -x
"$cc" -O3 -c "$here/vmath_shim.c" -o "$build_dir/vmath_shim.o"
set +x

# funmod2 first (amica15 does `use funmod2`); -J puts the .mod in build/ so the
# tree stays clean; -cpp resolves the source's `#ifdef MKL` guards (MKL undefined
# -> the mkl_vml.f90 include is skipped, plain-Fortran exp/log branches used);
# -std=legacy + -fallow-argument-mismatch relax this vintage F90's obsolescent
# constructs to warnings; -ffree-line-length-none lifts gfortran's 132-col cap.
set -x
# shellcheck disable=SC2086  # $lapack_libs must word-split into separate flags
"$fc" -O3 -fopenmp -cpp -ffree-line-length-none -std=legacy \
  -fallow-argument-mismatch -J"$build_dir" -I"$work_src" \
  "$work_src/funmod2.f90" "$work_src/amica15.f90" "$build_dir/vmath_shim.o" \
  -o "$out" $lapack_libs
set +x

echo
echo "built: $out"
echo "quick check: in a dir holding input.param + its .fdt, run"
echo "  OMP_NUM_THREADS=4 $out input.param"
echo "then benchmark with:"
echo "  uv run python benchmarks/benchmark_dimsweep.py --data DATA.npy \\"
echo "      --backends native-fortran-f64 --iters 30"
