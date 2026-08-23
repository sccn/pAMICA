# Native-Fortran backend for the cross-platform benchmark (epic #84, phase #85)

The `native-fortran-f64` backend in `benchmark_dimsweep.py` times **amica15 compiled
from source** so the Fortran reference has an honest per-iteration number to compare
against the torch / mlx / numpy backends.

## Why build from source (not `amica15mac`)

The bundled `pamica/sample_data/amica15mac` is an **x86_64** binary. On Apple Silicon
it only runs under Rosetta 2, so its timing is not representative of native hardware.
The benchmark therefore builds `amica15` from `pamica/{funmod2,amica15}.f90` natively on
each host -- the x86 Linux/CUDA host *and* Apple Silicon -- and times *that*. Both are
honest native-CPU rows. (`amica15mac` is still fine for structural smoke-testing of the
adapter on a Mac; it is just not a timing reference.)

## Host setup

amica15.f90 is an **MPI + OpenMP + LAPACK** program, so it needs an MPI Fortran wrapper
(`mpif90`) plus LAPACK/BLAS, not plain `gfortran`.

**Debian/Ubuntu (the x86 CUDA host):**
```bash
sudo apt-get update
sudo apt-get install -y gfortran libopenmpi-dev openmpi-bin liblapack-dev libblas-dev
```

**macOS / Apple Silicon (native arm64, no sudo):**
```bash
brew install gcc open-mpi lapack
```
The build script links brew's LAPACK on macOS (falling back to `-framework Accelerate`);
override with `LAPACK_LIBS=...` if needed.

`nvcc` is **not** required: the torch CUDA backends ship their own CUDA runtime, and the
Fortran build is CPU-only (MPI+OpenMP+LAPACK).

## Build

```bash
bash benchmarks/fortran/build_amica.sh        # -> benchmarks/fortran/amica15
```

The script compiles `funmod2.f90` first (amica15 does `use funmod2`), keeps the `.mod`
in `benchmarks/fortran/build/`, and lifts gfortran's 132-column limit
(`-ffree-line-length-none`). Override the compiler with `FC=...` and the source dir with
`AMICA_SRC=...`. The built binary and `build/` are gitignored.

## Run

amica runs as a single MPI rank (OpenMPI singleton -- no `mpirun` needed); OpenMP width is
`OMP_NUM_THREADS`, which the harness sets from `--fortran-threads` (default: all cores).

```bash
# native-fortran only
uv run python benchmarks/benchmark_dimsweep.py \
  --data benchmarks/data/ds002718_sub-002_eeg70.npy \
  --backends native-fortran-f64 --iters 30

# alongside the CUDA backends on the same host
uv run python benchmarks/benchmark_dimsweep.py \
  --data benchmarks/data/ds002718_sub-002_eeg70.npy \
  --backends native-fortran-f64,torch-cuda-f64,torch-cpu-f64 --out cuda.json
```

Use the **same** `ds002718_sub-002_eeg70.npy` as every other host (fetch recipe in
`benchmarks/README_dimsweep.md`) so the comparison is apples-to-apples.

## How the timing works (startup-immune)

Wrapping the whole process in a timer would fold in MPI init, data load, and PCA/sphering.
Instead the adapter parses amica's own per-iteration stamp from `out.txt`:

```
 iter     2 lrate = 0.05 LL = -3.4685 nd = 0.0257, D = ...  (  0.01 s,   0.0 h)
```

The `( <sec> s, ...)` field is that iteration's compute time (printed after init). The
adapter drops iter 1 (first-touch warmup), takes the **mean** of the rest, and reports the
min across `--repeats`. The final `LL` is read from the same lines and is already on the
per-sample scale the torch backends report (no normalization).

**Resolution caveat:** amica prints the per-iteration time to only ~0.01 s (10 ms). When
every iteration rounds to the same stamp (e.g. a steady 0.03 s/iter prints `0.03` each
time), the mean is still that stamp -- averaging does *not* recover sub-10 ms detail. So
run at a size whose per-iteration time is well above 10 ms (the 70-ch benchmark is
~30-70 ms/iter, giving ~±15% at worst) and treat the native-Fortran number as coarser than
the in-process torch/mlx timings. A config that rounds to 0.00 s/iter is rejected outright.

If a run exits non-zero the adapter raises (a crashed run must not masquerade as a fast
one). Settings are matched to the other backends: `num_mix=3`, `pdftype=0`, `do_newton` off,
`block_size=512`, and `use_min_dll`/`use_grad_norm` forced off so it runs the full
`--iters` budget.

## Build portability (gfortran, no MKL/AMD)

`amica15.f90` targets Intel's toolchain (`ifort` + MKL), so a plain `gfortran` + LAPACK
build needs two portability fixes. `build_amica.sh` applies them **without touching the
tracked reference source** (it compiles a build copy under `build/src/`):

1. **`-cpp`** so the source's `#ifdef MKL` guards resolve (MKL undefined -> the
   `include 'mkl_vml.f90'` is skipped and the non-MKL branch is used).
   (A third fix, a `random_seed` patch for the old ifort-only size-2 seed array, became
   obsolete when #235 made the source seed portably and deterministically itself --
   `random_seed(size=nseed)` + an allocated `seedvec`, sccn/amica PR #54; the build
   script now only verifies that portable seeding is present.)
2. **`vmath_shim.c`** provides `vrda_exp`/`vrda_log` (the non-MKL branch calls AMD LibM's
   vector exp/log) as libm loops, so no vendor math library is needed. IEEE-accurate; a
   vendor SIMD math library could make the exp/log-heavy E-step somewhat faster, so treat
   this as a portable-baseline timing, not a max-tuned one.

These are generic gfortran-portability fixes (not pamica-specific) and are candidates to
upstream to [sccn/amica](https://github.com/sccn/amica).
