# halo3d build.
#   make omp            -> bin/halo3d_omp    (OpenMP backend only, no CUDA needed)
#   make cuda SM=80     -> bin/halo3d_cuda   (both backends; SM=80 A100, SM=86 RTX A5000 / 3070 Ti, SM=70 V100)
#   make check          -> 16^3 grid, 2 ranks, OpenMP backend, fails unless the error is at round-off level
#   make bw SM=80       -> bin/bw_cpu, bin/bw_gpu (STREAM-style copy bandwidth: the measured roofline roofs)
# Switching SM? run `make clean` first (make does not track variable changes).
MPICXX    ?= mpicxx
NVCC      ?= nvcc
MPIRUN    ?= mpirun
SM        ?= 80
CUDA_HOME ?= /usr/local/cuda

DEFS     = -DOMPI_SKIP_MPICXX -DMPICH_SKIP_MPICXX
CXXFLAGS ?= -O3 -march=native -std=c++17 -fopenmp -Wall -Wextra
# nvcc uses mpicxx as its host compiler, which also supplies the include path for mpi.h
NVFLAGS  ?= -O3 -std=c++17 -arch=sm_$(SM) -ccbin $(MPICXX) -Xcompiler -Wall
SRC      = src/main.cpp src/stencil_omp.cpp src/halo.cpp

all: omp cuda

omp: bin/halo3d_omp

cuda: bin/halo3d_cuda

bin/halo3d_omp: $(SRC) src/common.h
	@mkdir -p bin
	$(MPICXX) $(CXXFLAGS) $(DEFS) $(SRC) -o $@

bin/stencil_cuda.o: src/stencil_cuda.cu src/common.h
	@mkdir -p bin
	$(NVCC) $(NVFLAGS) $(DEFS) -c $< -o $@

bin/halo3d_cuda: $(SRC) bin/stencil_cuda.o src/common.h
	$(MPICXX) $(CXXFLAGS) $(DEFS) -DHALO3D_CUDA $(SRC) bin/stencil_cuda.o \
		-L$(CUDA_HOME)/lib64 -Wl,-rpath,$(CUDA_HOME)/lib64 -lcudart -o $@

bin/bw_cpu: src/bw_cpu.cpp
	@mkdir -p bin
	$(MPICXX) $(CXXFLAGS) $< -o $@

bin/bw_gpu: src/bw_gpu.cu
	@mkdir -p bin
	$(NVCC) -O3 -std=c++17 -arch=sm_$(SM) $< -o $@

bw: bin/bw_cpu bin/bw_gpu

check: omp
	OMP_NUM_THREADS=2 $(MPIRUN) -np 2 ./bin/halo3d_omp --backend omp --n 16 --steps 50

clean:
	rm -rf bin

.PHONY: all omp cuda bw check clean
