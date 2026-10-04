#!/usr/bin/env bash
# Laptop sweep under WSL2: i7-12650H (6P + 4E cores, 16 hardware threads) + one RTX 3070 Ti Laptop GPU (sm_86).
# Appends to results/local.csv: OpenMP backend for np x threads <= 16, then the CUDA backend for np = 1, 2, 4.
# All CUDA ranks share the single GPU (their contexts time-slice; no MPS on WSL2), so np > 1 there exercises the
# halo path and staging, it is not a scaling result. WSL2 hides the P/E-core topology, so threads are not pinned.
set -euo pipefail
cd "$(dirname "$0")/.."
N=${N:-256}
STEPS=${STEPS:-50}
OUT=${OUT:-results/local.csv}
MPIRUN=${MPIRUN:-mpirun --bind-to none}   # OpenMPI binds each rank to a single core by default: fatal for OpenMP
mkdir -p results

make -s omp
BIN=./bin/halo3d_omp
if command -v nvcc >/dev/null; then make -s cuda SM=86; BIN=./bin/halo3d_cuda; fi

for np in 1 2 4; do
  for t in 1 2 4 8 16; do
    (( np * t <= 16 )) || continue
    OMP_NUM_THREADS=$t $MPIRUN -np $np $BIN --backend omp --n "$N" --steps "$STEPS" --csv "$OUT"
  done
done

if [[ $BIN == *cuda ]]; then
  for np in 1 2 4; do
    OMP_NUM_THREADS=4 $MPIRUN -np $np $BIN --backend cuda --n "$N" --steps "$STEPS" --csv "$OUT"
  done
fi
echo "wrote $OUT  ->  python3 scripts/plot.py --strong $OUT"
