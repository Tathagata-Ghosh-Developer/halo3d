#!/bin/bash
#SBATCH -J h3d_nsys
#SBATCH -p <gpu-partition>
#SBATCH -N 2
#SBATCH --ntasks-per-node=48
#SBATCH --gres=gpu:2
#SBATCH --exclusive
#SBATCH --mem=170G
#SBATCH -t 00:15:00
#SBATCH -o <outdir>/%x-%j.out
#
# Nsight Systems traces of 4 GPUs on 2 nodes (strong scaling, 1024^3, 20 steps, host-staged halos), one report
# per rank, for the current code and for the build before the stream-priority fix (V1_BIN). The GPU kernel and
# copy timestamps of rank 0 are exported to CSV; scripts/overlap.py turns them into per-step overlap numbers.
# Submit from the repository root after `make all SM=80`, with V1_BIN=<path to the pre-fix halo3d_cuda>.
set -u
cd "${SLURM_SUBMIT_DIR:-.}"
source scripts/env.sh
OUT=${OUT:-results}/nsys
mkdir -p "$OUT"
MAP="--map-by ppr:2:node:PE=12 --bind-to core"
for tag in v2 v1; do
  BIN=./bin/halo3d_cuda
  [ "$tag" = v1 ] && BIN=${V1_BIN:?set V1_BIN to the pre-fix binary}
  mpirun -np 4 $MAP -x OMP_NUM_THREADS=12 \
    nsys profile --trace=cuda,mpi --mpi-impl=openmpi --force-overwrite true \
    -o "$OUT/${tag}_rank%q{OMPI_COMM_WORLD_RANK}" $BIN --backend cuda --n 1024 --steps 20
  for r in 0 1; do
    nsys stats --report cuda_gpu_trace --format csv --force-export=true \
      --output "$OUT/${tag}_rank$r" "$OUT/${tag}_rank$r.nsys-rep"
  done
done
ls -la "$OUT"
