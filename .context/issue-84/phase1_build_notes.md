# Phase 1 (#85): native-Fortran adapter + build notes

Adds a `native-fortran-f64` backend to `benchmarks/benchmark_dimsweep.py` that times an
`amica15` compiled from source, plus a cross-platform build (`benchmarks/fortran/`).

## gfortran build portability (source targets ifort + MKL)

> Update (PR #295, 2026-08-23): item 2 below is obsolete -- #235 rewrote the source's seeding to the portable deterministic form, and build_amica.sh no longer patches it. Kept for history.

`amica15.f90` assumes Intel's toolchain. A plain `gfortran` + LAPACK build needs three
fixes, applied by `build_amica.sh` to a **build copy** (the tracked `pamica/amica15.f90`
reference is never modified):

1. `-cpp` -- resolve the `#ifdef MKL` guards so `include 'mkl_vml.f90'` is skipped.
2. `random_seed` -- the source's size-2 seed array (ifort-sized) is rejected by gfortran
   (wants 8); the build copy uses a portable default seed. Timing-neutral (seeding was
   already clock-based / non-reproducible).
3. `vmath_shim.c` -- the non-MKL branch calls AMD LibM `vrda_exp`/`vrda_log`; the shim
   provides them as libm loops so no vendor math lib is needed. IEEE-accurate; a vendor
   SIMD math lib could make the exp/log-heavy E-step somewhat faster.

Also `-std=legacy -fallow-argument-mismatch -ffree-line-length-none`. Built clean on both
**macOS arm64 (gfortran 16, brew open-mpi + lapack)** and **Ubuntu 24.04 x86_64
(gfortran 13, apt openmpi + liblapack-dev)**. Upstream target: sccn/amica #44 (compile on
M1) is an exact match; #49 asks about Ubuntu builds too, though via Intel OneAPI rather than
this gfortran path.

## Timing method

Per-iteration time is parsed from amica's own `out.txt` stamp (`( <sec> s, ...)`), which is
startup-immune (printed after MPI init / data load / PCA / sphering). Mean over the timed
iters (drop iter 1), min across repeats. **Resolution caveat:** the stamp is ~10 ms, so run
where per-iteration time is well above 10 ms (the 70-ch benchmark is); a config that rounds
to 0.00 s/iter is rejected rather than reported as bogus 0.0.

## Cross-platform sanity (real ds002718 sub-002, 30000 samples, single-model)

the CUDA workstation (Linux x86_64, RTX 4090, 32 cores, torch 2.12.1+cu130), 30 iters x 3 repeats,
fortran-threads 8:

| backend            | 32ch ms/it | 70ch ms/it | LL (32 / 70)     |
|--------------------|-----------:|-----------:|------------------|
| native-fortran-f64 |      12.07 |      44.83 | -3.308 / -3.242  |
| torch-cuda-f64     |      34.88 |      38.61 | -3.273 / -3.196  |
| torch-cpu-f64      |     217.51 |     226.48 | -3.273 / -3.196  |

Native x86 Fortran+OpenMP is fastest at 32ch and competitive with CUDA at 70ch (crossover
there), ~5-18x over torch-CPU. LL agrees to ~2 digits (Fortran differs slightly: random
init + libm math + unconverged at 30 iters). torch-cuda/torch-cpu agree to 5 digits.

Mac arm64 (native build, 8 threads) also runs as a first-class row (~30 ms/it @ 32ch,
~5x over torch-CPU-f64). Full cross-platform grid + CUDA multi-model report is Phase 3 (#87).
