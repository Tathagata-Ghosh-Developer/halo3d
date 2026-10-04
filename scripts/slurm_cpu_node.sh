#!/bin/bash
#SBATCH -J h3d_cpunode
#SBATCH -p <cpu-partition>
#SBATCH -N 1
#SBATCH --ntasks-per-node=48
#SBATCH --exclusive
#SBATCH --mem=170G
#SBATCH -t 00:30:00
#SBATCH -o <outdir>/%x-%j.out
#
# One CPU node (2 sockets x 24 cores), OpenMP backend:
#   1. STREAM-style copy/triad bandwidth for 1..48 threads: the measured roof of the CPU roofline.
#   2. Thread scaling, one rank, 512^3, threads packed onto socket 0 first (OMP_PROC_BIND=close).
#   3. Ranks x threads on the full node (1x48, 2x24 = one rank per socket, 4x12, 8x6, 48x1), 512^3 and 1024^3.
#   4. Size of the y-blocks that keep three z-planes in L2 (HALO3D_BLOCK_KB), 2x24, 1024^3.
# Every timed point runs REPS times. Submit from the repository root after `make all bw SM=80`.
set -u
cd "${SLURM_SUBMIT_DIR:-.}"
source scripts/env.sh
OUT=${OUT:-results}
REPS=${REPS:-3}
mkdir -p "$OUT"
lscpu | grep -E "Model name|Socket|Core|NUMA node|L3"
B=./bin/halo3d_omp
export OMP_PLACES=cores

echo "## 1. memory bandwidth"
for t in 1 2 4 8 12 16 24 32 48; do
  OMP_NUM_THREADS=$t OMP_PROC_BIND=close ./bin/bw_cpu $((1 << 27)) 10
done
OMP_NUM_THREADS=24 OMP_PROC_BIND=spread ./bin/bw_cpu $((1 << 27)) 10   # 12 cores per socket
OMP_NUM_THREADS=48 OMP_PROC_BIND=spread ./bin/bw_cpu $((1 << 27)) 10

echo "## 2. thread scaling, 1 rank, 512^3"
for t in 1 2 4 8 12 16 24 32 48; do
  for r in $(seq "$REPS"); do
    mpirun -np 1 --bind-to none -x OMP_NUM_THREADS=$t -x OMP_PLACES -x OMP_PROC_BIND=close \
      $B --n 512 --steps 50 --csv "$OUT/cpu_threads.csv"
  done
done

echo "## 3. ranks x threads on one node"
for n in 512 1024; do
  for rt in "1 48" "2 24" "4 12" "8 6" "48 1"; do
    set -- $rt
    R=$1 T=$2
    if [ "$R" -eq 1 ]; then MAP="--bind-to none"; else MAP="--map-by ppr:$((R / 2)):socket:PE=$T --bind-to core"; fi
    for r in $(seq "$REPS"); do
      REPORT=""; [ "$r" -eq 1 ] && [ "$n" -eq 512 ] && REPORT="--report-bindings"
      mpirun -np "$R" $MAP $REPORT -x OMP_NUM_THREADS=$T -x OMP_PLACES -x OMP_PROC_BIND=close \
        $B --n $n --steps 100 --csv "$OUT/cpu_hybrid.csv"
    done
  done
done

echo "## 4. y-block size (HALO3D_BLOCK_KB; 0 = whole planes, no blocking), 2 ranks x 24 threads, 1024^3"
for kb in 0 128 256 512 1024 4096; do
  for r in $(seq "$REPS"); do
    mpirun -np 2 --map-by ppr:1:socket:PE=24 --bind-to core -x HALO3D_BLOCK_KB=$kb -x OMP_NUM_THREADS=24 \
      -x OMP_PLACES -x OMP_PROC_BIND=close $B --n 1024 --steps 50 --csv "$OUT/cpu_block_${kb}kb.csv"
  done
done
