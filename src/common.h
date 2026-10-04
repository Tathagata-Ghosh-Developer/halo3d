// halo3d: shared types and the few functions the backends call.
#pragma once
#include <mpi.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

// Explicit Euler for u_t = alpha * lap(u) on a unit-spaced grid: r = alpha*dt/h^2 must be <= 1/6.
constexpr double R = 0.125, C0 = 1.0 - 6.0 * R;
constexpr double PI = 3.141592653589793;
constexpr double TOL = 1e-9;        // pass threshold for the relative L2 error vs the exact discrete solution
constexpr int BYTES_PER_LUP = 16;   // ideal traffic per lattice update: read u once, write v once (FP64)

// A face of the local block, flattened: element (a, b) lives at u[base + a*sa + b*sb], 0 <= a < na, 0 <= b < nb.
struct Face { long base, sa, sb; int na, nb; };

// Half-open index box [lo, hi) in padded coordinates (interior cells are 1..n).
struct Box { int lo[3], hi[3]; };

struct Grid {
  MPI_Comm cart;
  int rank, dims[3], coords[3];
  int N[3], n[3], off[3];   // global cells, local cells, global offset of local cell 1 (per dim x, y, z)
  int nbr[6];               // neighbour across face f = 2*dim + side (side 0 = low); MPI_PROC_NULL at the wall
  long s[3], cells;         // strides of the padded (n+2)^3 array, x fastest, and its total size
  Face send[6], recv[6];    // interior layer we send / ghost layer we receive, per face
  long boff[7];             // offset of face f inside the packed halo buffers; boff[6] = total length
};

// t_loop: whole time loop; t_wait: blocked in MPI_Waitall; t_inner: interior update (OpenMP: wall time of the
// interior sweep; CUDA: interior-kernel time from cudaEvents). All in seconds, summed over steps.
struct Stats { double t_loop, t_wait, t_inner, kernel_gbs; };

[[noreturn]] inline void die(const char* msg) {
  std::fprintf(stderr, "halo3d: %s\n", msg);
  MPI_Abort(MPI_COMM_WORLD, 1);
  std::exit(1);
}

// sin(pi * global_index / (N+1)) along dim d, indexed by local padded index. The product over x, y, z is the
// slowest sine mode: zero on the Dirichlet walls and an eigenvector of the discrete 7-point Laplacian.
inline std::vector<double> mode1d(const Grid& g, int d) {
  std::vector<double> s(g.n[d] + 2);
  for (int i = 0; i < g.n[d] + 2; ++i) s[i] = std::sin(PI * (g.off[d] + i) / (g.N[d] + 1));
  return s;
}

// halo.cpp
Grid make_grid(int n_global, int n_local);
void make_boxes(const Grid& g, Box& interior, Box shell[6]);
void halo_start(const Grid& g, double* sendbuf, double* recvbuf, MPI_Request req[12]);

// stencil_omp.cpp
double* alloc_field(const Grid& g, bool with_mode);
Stats run_omp(const Grid& g, double* u, int steps);

// stencil_cuda.cu (only linked into bin/halo3d_cuda)
Stats run_cuda(const Grid& g, double* u, int steps, bool cuda_aware);
