#!/bin/bash
#SBATCH -J h3d_gpu
#SBATCH -p <gpu-partition>
#SBATCH -N 8
#SBATCH --ntasks-per-node=48
#SBATCH --gres=gpu:2
#SBATCH --exclusive
#SBATCH --mem=170G
#SBATCH -t 00:30:00
#SBATCH -o <outdir>/%x-%j.out
#
# GPU nodes: 2 GPUs each, both attached to socket 0, one InfiniBand NIC.
#   1. Device copy bandwidth (the measured roof) and pinned host <-> device bandwidth.
#   2. OSU point-to-point bandwidth, host and device buffers, within a node and across nodes.
#   3. Strong (N = 1024) and weak (L = 512 per GPU) scaling on 1, 2, 4, 8, 16 GPUs, one rank per GPU,
#      host-staged halos and CUDA-aware halos. Each GPU rank gets 12 cores of socket 0.
# Every point runs REPS times. Submit from the repository root after `make all bw SM=80`.
set -u
cd "${SLURM_SUBMIT_DIR:-.}"
source scripts/env.sh
OUT=${OUT:-results}
REPS=${REPS:-3}
N=${N:-1024}
L=${L:-512}
STEPS=${STEPS:-200}
mkdir -p "$OUT"
nvidia-smi --query-gpu=name,memory.total,driver_version,pcie.link.gen.max,pcie.link.width.max --format=csv

echo "## 1. GPU bandwidth"
for d in 0 1; do CUDA_VISIBLE_DEVICES=$d ./bin/bw_gpu; done

echo "## 2. OSU bandwidth (MB/s)"
OSU=$(dirname "$(dirname "$(command -v mpicc)")")/tests/osu-micro-benchmarks-cuda
[ -x "$OSU/osu_bw" ] || OSU=$(ls -d "$NVHPC_ROOT"/comm_libs/*/hpcx/*/ompi/tests/osu-micro-benchmarks-cuda 2>/dev/null | head -1)
for where in "ppr:2:node" "ppr:1:node"; do
  for buf in "H H" "D D"; do
    echo "# osu_bw $buf, $where (2:node = same node, 1:node = across InfiniBand)"
    mpirun -np 2 --map-by $where --bind-to core "$OSU/osu_bw" -m 4096:8388608 -d cuda $buf 2>&1 | grep -vE "^\s*$|^#"
  done
done

echo "## 3. strong and weak scaling"
MAP="--map-by ppr:2:node:PE=12 --bind-to core"
ENVS="-x OMP_NUM_THREADS=12 -x OMP_PLACES=cores -x OMP_PROC_BIND=close"
B=./bin/halo3d_cuda
for P in 1 2 4 8 16; do
  [ "$P" -le $((2 * SLURM_JOB_NUM_NODES)) ] || continue
  for r in $(seq "$REPS"); do
    for aware in "" "--cuda-aware"; do
      REPORT=""; [ "$r" -eq 1 ] && [ "$P" -eq 4 ] && [ -z "$aware" ] && REPORT="--report-bindings"
      mpirun -np $P $MAP $REPORT $ENVS $B --backend cuda --n "$N" --steps "$STEPS" $aware --csv "$OUT/gpu_strong.csv"
      mpirun -np $P $MAP $ENVS $B --backend cuda --local "$L" --steps "$STEPS" $aware --csv "$OUT/gpu_weak.csv"
    done
  done
done
