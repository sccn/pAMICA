#!/usr/bin/env bash
# Driver for newton_seeds.py on the CUDA workstation (issue #351 item 4a).
# Two queues run side by side: the pamica fits on the GPU, the reference fits
# on FORTRAN_THREADS CPU threads. Launch detached:
#   nohup bash .context/issue-351/run_newton_hallu.sh OUT_DIR > OUT_DIR.log 2>&1 < /dev/null &
set -u
OUT=${1:?usage: run_newton_hallu.sh OUT_DIR}
DATA=${DATA:-benchmarks/data/ds002718_sub-002_eeg70_full.npy}
FORTRAN_THREADS=${FORTRAN_THREADS:-10}
UV=${UV:-uv}
SCRIPT=.context/issue-351/newton_seeds.py
mkdir -p "$OUT"

gpu_queue() {
  for s in 42 7 13; do
    [ -f "$OUT/pamica_seed$s.npz" ] || "$UV" run python "$SCRIPT" pamica --seed "$s" --data "$DATA" --out-dir "$OUT"
  done
  [ -f "$OUT/pamica_fixinit.npz" ] || "$UV" run python "$SCRIPT" pamica --fixinit --data "$DATA" --out-dir "$OUT"
}

cpu_queue() {
  for s in 1 2; do
    [ -f "$OUT/fortran_seed$s.npz" ] || "$UV" run python "$SCRIPT" fortran --seed "$s" --threads "$FORTRAN_THREADS" --data "$DATA" --out-dir "$OUT"
  done
  [ -f "$OUT/fortran_fixinit.npz" ] || "$UV" run python "$SCRIPT" fortran --fixinit --threads "$FORTRAN_THREADS" --data "$DATA" --out-dir "$OUT"
}

date
gpu_queue > "$OUT/gpu_queue.log" 2>&1 &
cpu_queue > "$OUT/cpu_queue.log" 2>&1 &
wait
date
"$UV" run python "$SCRIPT" compare --out-dir "$OUT"
echo DONE
