#!/bin/bash
#SBATCH -J h3d_check
#SBATCH -p <gpu-partition>
#SBATCH -N 2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH -t 00:10:00
#SBATCH -o <outdir>/%x-%j.out
#
# Correctness. Every run must print PASS: relative L2 error against the exact discrete solution below 1e-9.
# OpenMP backend: 2, 8 and 12 ranks (oversubscribed, so x and y faces are exercised), even and uneven splits.
# CUDA backend: 1, 2 and 4 GPUs (4 = two nodes, so faces also cross InfiniBand), n = 16, 17 (uneven), 64,
# host-staged and CUDA-aware halos, plus a weak-scaling (--local) run.
# Submit from the repository root after `make all bw SM=80`.
set -u
cd "${SLURM_SUBMIT_DIR:-.}"
source scripts/env.sh
OUT=${OUT:-results}
mkdir -p "$OUT"
B=./bin/halo3d_cuda
M="mpirun --bind-to none --oversubscribe"
PASS=0 FAIL=0
run() {
  echo "### $*"
  timeout 120 "$@" 2>&1 | grep -v '^\s*$'
  local rc=${PIPESTATUS[0]}
  if [ "$rc" -eq 0 ]; then PASS=$((PASS + 1)); else FAIL=$((FAIL + 1)); fi
  echo "### rc=$rc"
}
export OMP_NUM_THREADS=2
run $M -np 2 -x OMP_NUM_THREADS $B --backend omp --n 16 --steps 50 --csv "$OUT/check.csv"
run $M -np 8 -x OMP_NUM_THREADS $B --backend omp --n 16 --steps 50 --csv "$OUT/check.csv"
run $M -np 8 -x OMP_NUM_THREADS $B --backend omp --n 17 --steps 50 --csv "$OUT/check.csv"
run $M -np 12 -x OMP_NUM_THREADS $B --backend omp --n 23 --steps 30 --csv "$OUT/check.csv"
export OMP_NUM_THREADS=4
for np in 1 2 4; do
  for n in 16 17 64; do
    run $M -np $np --map-by ppr:2:node -x OMP_NUM_THREADS $B --backend cuda --n $n --steps 50 --csv "$OUT/check.csv"
    run $M -np $np --map-by ppr:2:node -x OMP_NUM_THREADS $B --backend cuda --n $n --steps 50 --cuda-aware --csv "$OUT/check.csv"
  done
done
run $M -np 4 --map-by ppr:2:node -x OMP_NUM_THREADS $B --backend cuda --local 40 --steps 40 --csv "$OUT/check.csv"
run $M -np 4 --map-by ppr:2:node -x OMP_NUM_THREADS $B --backend omp --local 40 --steps 40 --csv "$OUT/check.csv"
echo "### summary: $PASS passed, $FAIL failed"
