#!/bin/bash
#SBATCH -J h3d_cpuscale
#SBATCH -p <cpu-partition>
#SBATCH -N 16
#SBATCH --ntasks-per-node=48
#SBATCH --exclusive
#SBATCH --mem=170G
#SBATCH -t 00:25:00
#SBATCH -o <outdir>/%x-%j.out
#
# CPU strong and weak scaling on 1, 2, 4, 8, 16 nodes, OpenMP backend, one rank per socket
# (2 ranks/node x 24 threads, each rank bound to the 24 cores of its socket).
#   strong: fixed global N^3 (N = 1024: 8.6 GB per field, 17 GB in total)
#   weak:   L^3 cells per rank (L = 512), so the global grid grows with the rank count
# Every point runs REPS times. Submit from the repository root after `make all SM=80`.
set -u
cd "${SLURM_SUBMIT_DIR:-.}"
source scripts/env.sh
OUT=${OUT:-results}
REPS=${REPS:-3}
N=${N:-1024}
L=${L:-512}
STEPS=${STEPS:-100}
mkdir -p "$OUT"
B=./bin/halo3d_omp
MAP="--map-by ppr:1:socket:PE=24 --bind-to core"
ENVS="-x OMP_NUM_THREADS=24 -x OMP_PLACES=cores -x OMP_PROC_BIND=close"
for nodes in 1 2 4 8 16; do
  [ "$nodes" -le "$SLURM_JOB_NUM_NODES" ] || continue
  P=$((2 * nodes))
  for r in $(seq "$REPS"); do
    REPORT=""; [ "$r" -eq 1 ] && [ "$nodes" -eq 2 ] && REPORT="--report-bindings"
    mpirun -np $P $MAP $REPORT $ENVS $B --n "$N" --steps "$STEPS" --csv "$OUT/cpu_strong.csv"
    mpirun -np $P $MAP $ENVS $B --local "$L" --steps "$STEPS" --csv "$OUT/cpu_weak.csv"
  done
done
