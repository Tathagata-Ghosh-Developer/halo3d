#!/bin/bash
#SBATCH -J h3d_gpuweakL
#SBATCH -p <gpu-partition>
#SBATCH -N 8
#SBATCH --ntasks-per-node=48
#SBATCH --gres=gpu:2
#SBATCH --exclusive
#SBATCH --mem=170G
#SBATCH -t 00:20:00
#SBATCH -o <outdir>/%x-%j.out
#
# GPU weak scaling with a larger block per GPU (L = 1024: 8x the volume, 4x the face area of L = 512),
# to test the surface-to-volume explanation of the L = 512 weak-scaling efficiency. Same binding as slurm_gpu.sh.
# Submit from the repository root after `make all SM=80`.
set -u
cd "${SLURM_SUBMIT_DIR:-.}"
source scripts/env.sh
OUT=${OUT:-results}
REPS=${REPS:-3}
L=${L:-1024}
STEPS=${STEPS:-100}
mkdir -p "$OUT"
MAP="--map-by ppr:2:node:PE=12 --bind-to core"
ENVS="-x OMP_NUM_THREADS=12 -x OMP_PLACES=cores -x OMP_PROC_BIND=close"
for P in 1 2 4 8 16; do
  [ "$P" -le $((2 * SLURM_JOB_NUM_NODES)) ] || continue
  for r in $(seq "$REPS"); do
    for aware in "" "--cuda-aware"; do
      mpirun -np $P $MAP $ENVS ./bin/halo3d_cuda --backend cuda --local "$L" --steps "$STEPS" $aware \
        --csv "$OUT/gpu_weak_L$L.csv"
    done
  done
done
