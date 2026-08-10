# Cross-platform dimension-sweep benchmark (issue #77, epic #74 Phase B)

`benchmark_dimsweep.py` measures **both results (converged log-likelihood) and
performance (ms/iteration)** for every AMICA backend the host supports, sweeping
the channel count on real 70-channel EEG, to answer where an Apple/NVIDIA GPU
actually beats the CPU. Backends: `numpy-cpu-f64`, `torch-cpu-f64/f32`,
`torch-mps-f32`, `torch-cuda-f64/f32`, `mlx-f32` (the MLX backend supports single-
and multi-model but has no component sharing yet, so it is excluded only from the
`--share` configs), and `native-fortran-f64` (the Fortran reference compiled from
source, validated single-model in Phase 1 with component sharing off -- see
`benchmarks/fortran/README.md` to build it).

**`native-fortran-f64` result caveat:** every other backend fixes `seed=42`, so its
`final_ll` is reproducible and directly comparable across backends and repeats. amica seeds
its random init from the wall clock (non-reproducible run-to-run), so the native-Fortran
`final_ll` is drawn from a different basin each invocation and can differ from the others by
more than the fixed-seed backends differ among themselves. Treat that column as a sanity
check (same ballpark), not a fixed-seed parity number; the `ms/iter` timing is unaffected.

## Data (real, not committed)

Real 70-channel EEG from OpenNeuro **ds002718** (Wakeman-Henson faces), subject
sub-002. The data is not committed (`benchmarks/data/` is gitignored); fetch and
extract it once:

```bash
# 1. download one subject's EEGLAB .set (public, no credentials; ~224 MB)
aws s3 cp --no-sign-request \
  s3://openneuro.org/ds002718/sub-002/eeg/sub-002_task-FaceRecognition_eeg.set \
  /tmp/ds002718_sub-002.set

# 2. extract the 70 EEG channels to a (70, n_samples) float64 .npy (needs mne)
uv pip install mne
uv run python - <<'PY'
import mne, numpy as np
raw = mne.io.read_raw_eeglab("/tmp/ds002718_sub-002.set", preload=True, verbose="ERROR")
data = raw.get_data(picks=raw.ch_names[:70]) * 1e6   # first 70 are EEG; V -> uV
np.save("benchmarks/data/ds002718_sub-002_eeg70.npy", data[:, :60000].astype(np.float64))
PY
```

(NEMAR mirrors the same dataset at data.nemar.org / ww2.nemar.org/dataset/ds002718.)

## Run

```bash
# Local (Apple Silicon: cpu / mps / mlx auto-detected)
uv run python benchmarks/benchmark_dimsweep.py \
  --data benchmarks/data/ds002718_sub-002_eeg70.npy --out mac.json

# A CUDA host (skip the CPU backends if its CPU is busy)
uv run python benchmarks/benchmark_dimsweep.py \
  --data benchmarks/data/ds002718_sub-002_eeg70.npy \
  --backends torch-cuda-f64,torch-cuda-f32 --out cuda.json

# Multi-model + component sharing (MLX runs multi-model; auto-excluded only from --share)
uv run python benchmarks/benchmark_dimsweep.py --data DATA --n-models 2 --out mac_m2.json
uv run python benchmarks/benchmark_dimsweep.py --data DATA --n-models 2 --share --out mac_m2share.json

# CPU core-count scaling sweep (#86): run the CPU backends at each thread count
# (GPU backends run once). torch-cpu -> set_num_threads, numpy -> threadpoolctl,
# native-fortran -> OMP_NUM_THREADS. Best on a many-core host (e.g. the CUDA workstation, 32 cores).
uv run python benchmarks/benchmark_dimsweep.py \
  --data benchmarks/data/ds002718_sub-002_eeg70.npy \
  --backends torch-cpu-f64,numpy-cpu-f64,native-fortran-f64 \
  --threads 4,8,12,16,24 --out scaling.json

# Merge per-platform JSONs into ms/it + LL tables, one block per config; --threads
# rows add a "CPU scaling" block (threads x backend) with a GPU reference line.
uv run python benchmarks/benchmark_dimsweep.py --report mac.json cuda.json scaling.json ...
```

Findings live in `.context/issue-77/benchmark_findings.md` (Phase B: channel sweep, Apple GPU)
and `.context/issue-84/phase2_cpu_scaling.md` (Phase 2: CPU core-count scaling, cross-platform).

## Reproducing Table 1 (issue #144)

`benchmarks/reproduce_table1.py` reproduces the parity numbers in the JOSS paper's Table 1
against the Fortran reference, printing each measured value next to the paper's claimed one so
a reviewer can check them off directly:

```bash
uv run python benchmarks/reproduce_table1.py --tier bundled     # no download, see cost below
uv run python benchmarks/reproduce_table1.py --tier external \
  --data benchmarks/data/ds002718_sub-002_eeg70_full.npy        # needs the download below
```

This is a distinct, more expensive protocol from the `benchmark_dimsweep.py`/`benchmark_decompose.py`
sweeps above: it re-fits the reference binary and pamica several times each, at the full 2000-iteration
budget, to get numbers precise enough to compare against the paper rather than a single ms/iteration
sample. Read this section before running the external tier -- it is not a quick check.

### What each tier costs

**Bundled tier** (`--tier bundled`, default): uses `pamica/sample_data/eeglab_data.fdt`
(32 channels, 30,504 frames, already committed -- no download). Default budget is 5 single-model
seeds at 2000 iterations plus a 20-run multi-model ensemble at 100 iterations (matching
`docs/guides/validation.md`'s documented protocol). Measured end to end on a 2026 Apple Silicon
laptop (10 cores, no CUDA, MPS unusable here because these runs are float64 -- see below):

| phase | measured |
|---|---|
| single-model sweep (5 seeds x 2000 iter, Fortran + pamica) | ~24 min |
| multi-model ensemble (20 runs x 100 iter, Fortran + pamica) | ~9 min |
| score-function / sufficient-statistics check | <1 s |
| **total** | **~33 min** |

A `--n-seeds 2 --max-iter 100 --multimodel-runs 2 --multimodel-max-iter 20` smoke test (not the
paper's protocol -- noisier numbers, just a pipeline check) completes in well under a minute.

**External tier** (`--tier external`): needs OpenNeuro **ds002718** sub-002 (Wakeman-Henson
faces), the *full* recording (70 channels, 747,750 frames, k~153 -- not the 60,000-frame
truncation used for the ms/iteration sweeps above). Download and extraction:

```bash
# 1. download (public, no credentials; ~224 MB compressed .set)
aws s3 cp --no-sign-request \
  s3://openneuro.org/ds002718/sub-002/eeg/sub-002_task-FaceRecognition_eeg.set \
  /tmp/ds002718_sub-002.set

# 2. extract ALL 70 EEG channels x ALL frames (no truncation) to a float64 .npy
#    (~419 MB on disk; needs mne)
uv pip install mne
uv run python - <<'PY'
import mne, numpy as np
raw = mne.io.read_raw_eeglab("/tmp/ds002718_sub-002.set", preload=True, verbose="ERROR")
data = raw.get_data(picks=raw.ch_names[:70]) * 1e6   # first 70 are EEG; V -> uV
np.save("benchmarks/data/ds002718_sub-002_eeg70_full.npy", data.astype(np.float64))
PY
```

The protocol is 5 sequential Fortran fits at 2000 iterations on that recording, plus 5 matching
pamica fits -- **hours, not minutes**, even on capable hardware. There is no cheaper substitute:
`docs/guides/validation.md`'s own data-adequacy sweep found cross-backend agreement is
under-determined below k~60 and only plateaus at k~153 (this recording's full size), so a
shorter run would reproduce a noisier number, not a faster version of the same one -- that
tradeoff was made deliberately when this recording was chosen as the headline dataset.

Single-fit wall-clock at this exact configuration (70ch, 747,750 frames, 2000 iterations,
`do_newton=0`) was measured on a 32-core Linux workstation with an RTX 4090
(`.context/issue-90/ksweep_findings.md`):

| backend | single fit |
|---|---:|
| native-fortran-f64 | 1303 s (~22 min) |
| pamica, CUDA float64 | 1856 s (~31 min) |

Scaling those to the paper's 5-seed protocol (Fortran phases can run back to back while each
seed's GPU phase overlaps the next seed's Fortran phase, the way the original workstation script
pipelined them):

| configuration | estimated total |
|---|---|
| 32-core workstation + CUDA (pipelined: GPU total + one Fortran fit) | **~3 hours** |
| 32-core workstation + CUDA (naive, sequential) | ~4.4 hours |
| **no GPU** (pamica falls back to CPU float64; ~5x slower than CUDA-f64 at this size per the
  ms/iteration table above) | **roughly half a day** for the pamica phase alone |

A slower or fewer-core machine pushes the Fortran phase out further too (native Fortran scales
with thread count; see the CPU-scaling table in the main `README_dimsweep.md` findings). Budget
accordingly, and consider running the external tier overnight or on a shared compute node rather
than interactively.

### Device and binary resolution

Neither tier assumes a GPU or a specific reference-binary path. The compute device is
auto-detected (CUDA, then Apple MPS, then CPU); because these runs use float64 for Fortran parity
and MPS has no float64 support, an auto-selected (or explicitly requested) MPS device always
falls back to CPU, with a printed message saying so. The Fortran binary is resolved via
`pamica.native.resolver` (a platform release asset, downloaded and SHA-256-verified on first use
and cached under `~/.cache/pamica/bin/`), honoring `PAMICA_NATIVE_BINARY`/`--fortran-binary` for
a local override -- so this runs on Linux and Windows, not just the macOS `amica15mac` fixture
bundled for the older `validate_implementations.py` harness.
