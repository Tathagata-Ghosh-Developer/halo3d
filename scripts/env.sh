# Toolchain used for every run in results/: NVIDIA HPC SDK 25.11 with HPC-X (Open MPI 4.1, UCX, CUDA 13.0).
# mpicxx wraps GCC (C++17, OpenMP) instead of nvc++; nvcc uses mpicxx as its host compiler.
module load nvhpc-hpcx/25.11
export OMPI_CXX=g++ OMPI_CC=gcc
export CUDA_HOME=$NVHPC_ROOT/cuda
# Added after the runs, only silences a UCX warning about a variable set by the site environment
# (the warnings were stripped from results/logs/):
export UCX_WARN_UNUSED_ENV_VARS=n
