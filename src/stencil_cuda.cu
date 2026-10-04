// CUDA backend: one rank per GPU. The interior runs on its own stream while a second stream packs faces,
// stages them to pinned host memory (or hands device pointers to a CUDA-aware MPI), unpacks the ghosts and
// updates the boundary shell.
#include "common.h"
#include <cuda_runtime.h>
#include <utility>

#define CK(call)                                                                                  \
  do {                                                                                            \
    cudaError_t e_ = (call);                                                                      \
    if (e_ != cudaSuccess) {                                                                      \
      std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
      MPI_Abort(MPI_COMM_WORLD, 1);                                                               \
    }                                                                                             \
  } while (0)

constexpr int BX = 32, BY = 8, KZ = 32;   // thread tile in x, y; z planes marched per block

// 2.5D blocking: a BX x BY tile of threads marches up KZ planes. The z neighbours stay in registers
// (below / cur / above), the x/y neighbours come from a shared-memory copy of the plane with a 1-cell halo,
// so each cell is read from DRAM about once.
__global__ void __launch_bounds__(BX * BY)
interior_kernel(const double* __restrict__ u, double* __restrict__ v, Box b, long sy, long sz) {
  __shared__ double t[BY + 2][BX + 2];
  const int tx = threadIdx.x, ty = threadIdx.y;
  const int i = b.lo[0] + blockIdx.x * BX + tx, j = b.lo[1] + blockIdx.y * BY + ty;
  const int k0 = b.lo[2] + blockIdx.z * KZ, k1 = min(k0 + KZ, b.hi[2]);
  const bool ok = i < b.hi[0] && j < b.hi[1];
  long c = k0 * sz + j * sy + i;
  double below = 0.0, cur = 0.0, above = 0.0;
  if (ok) { below = u[c - sz]; cur = u[c]; above = u[c + sz]; }
  for (int k = k0; k < k1; ++k, c += sz) {
    if (ok) {
      t[ty + 1][tx + 1] = cur;
      if (tx == 0) t[ty + 1][0] = u[c - 1];
      if (tx == BX - 1 || i == b.hi[0] - 1) t[ty + 1][tx + 2] = u[c + 1];
      if (ty == 0) t[0][tx + 1] = u[c - sy];
      if (ty == BY - 1 || j == b.hi[1] - 1) t[ty + 2][tx + 1] = u[c + sy];
    }
    __syncthreads();
    if (ok) {
      v[c] = C0 * cur + R * (t[ty + 1][tx] + t[ty + 1][tx + 2] + t[ty][tx + 1] + t[ty + 2][tx + 1] + below + above);
      below = cur;
      cur = above;
      if (k + 1 < k1) above = u[c + 2 * sz];
    }
    __syncthreads();
  }
}

// One thread per cell, for the thin boundary slabs where z-marching would leave most of the GPU idle.
__global__ void shell_kernel(const double* __restrict__ u, double* __restrict__ v, Box b, long sy, long sz) {
  const long w = b.hi[0] - b.lo[0], h = b.hi[1] - b.lo[1];
  const long t = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (t >= w * h * (b.hi[2] - b.lo[2])) return;
  const long c = (b.lo[2] + t / (w * h)) * sz + (b.lo[1] + t / w % h) * sy + b.lo[0] + t % w;
  v[c] = C0 * u[c] + R * (u[c - 1] + u[c + 1] + u[c - sy] + u[c + sy] + u[c - sz] + u[c + sz]);
}

__global__ void face_kernel(double* u, double* buf, Face F, bool unpack) {
  const int a = blockIdx.x * blockDim.x + threadIdx.x, b = blockIdx.y;
  if (a >= F.na) return;
  const long c = F.base + a * F.sa + b * F.sb, q = a + (long)F.na * b;
  if (unpack) u[c] = buf[q]; else buf[q] = u[c];
}

static long volume(const Box& b) {
  long n = 1;
  for (int d = 0; d < 3; ++d) n *= b.hi[d] > b.lo[d] ? b.hi[d] - b.lo[d] : 0;
  return n;
}

static unsigned cdiv(long a, long b) { return (unsigned)((a + b - 1) / b); }

Stats run_cuda(const Grid& g, double* h_u, int steps, bool aware) {
  // pick the GPU by node-local rank (MPI-3); a laptop with one GPU puts every rank on device 0
  MPI_Comm node;
  int lrank, ndev;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node);
  MPI_Comm_rank(node, &lrank);
  MPI_Comm_free(&node);
  CK(cudaGetDeviceCount(&ndev));
  CK(cudaSetDevice(lrank % ndev));

  const size_t bytes = g.cells * sizeof(double), hbytes = g.boff[6] * sizeof(double);
  double *u, *v, *dsend, *drecv, *hsend = nullptr, *hrecv = nullptr;
  CK(cudaMalloc(&u, bytes));
  CK(cudaMalloc(&v, bytes));
  CK(cudaMalloc(&dsend, hbytes));
  CK(cudaMalloc(&drecv, hbytes));
  if (!aware) {
    CK(cudaMallocHost(&hsend, hbytes));
    CK(cudaMallocHost(&hrecv, hbytes));
  }
  CK(cudaMemcpy(u, h_u, bytes, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(v, h_u, bytes, cudaMemcpyHostToDevice));   // v needs the zero ghost layers too
  // The boundary stream gets the highest priority. The interior kernel has tens of thousands of blocks queued;
  // without priorities, a pack, unpack or shell kernel issued after it waits for SMs behind those blocks, so the
  // halo path only starts once the interior is almost done and the overlap is lost.
  int prio_low, prio_high;
  CK(cudaDeviceGetStreamPriorityRange(&prio_low, &prio_high));
  cudaStream_t s_in, s_bd;
  CK(cudaStreamCreateWithPriority(&s_in, cudaStreamNonBlocking, prio_low));
  CK(cudaStreamCreateWithPriority(&s_bd, cudaStreamNonBlocking, prio_high));
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));

  Box in, shell[6];
  make_boxes(g, in, shell);
  const long sy = g.s[1], sz = g.s[2], nin = volume(in);
  const dim3 tile(BX, BY);
  const dim3 grid(cdiv(in.hi[0] - in.lo[0], BX), cdiv(in.hi[1] - in.lo[1], BY), cdiv(in.hi[2] - in.lo[2], KZ));
  double* sbuf = aware ? dsend : hsend;
  double* rbuf = aware ? drecv : hrecv;
  MPI_Request req[12];
  Stats st{};
  double kernel_s = 0.0;

  MPI_Barrier(g.cart);
  const double t0 = MPI_Wtime();
  for (int s = 0; s < steps; ++s) {
    // boundary stream: pack every face that has a neighbour, then stage them. All packs go first so that they are
    // queued ahead of the interior kernel, not each one behind the previous face's copy.
    for (int f = 0; f < 6; ++f) {
      if (g.nbr[f] == MPI_PROC_NULL) continue;
      const Face& F = g.send[f];
      face_kernel<<<dim3(cdiv(F.na, 128), F.nb), 128, 0, s_bd>>>(u, dsend + g.boff[f], F, false);
    }
    for (int f = 0; f < 6 && !aware; ++f)
      if (g.nbr[f] != MPI_PROC_NULL)
        CK(cudaMemcpyAsync(hsend + g.boff[f], dsend + g.boff[f], (g.boff[f + 1] - g.boff[f]) * sizeof(double),
                           cudaMemcpyDeviceToHost, s_bd));
    CK(cudaEventRecord(e0, s_in));   // interior stream: needs no ghost that is in flight
    if (nin > 0) interior_kernel<<<grid, tile, 0, s_in>>>(u, v, in, sy, sz);
    CK(cudaEventRecord(e1, s_in));

    CK(cudaStreamSynchronize(s_bd));   // send buffers are ready
    halo_start(g, sbuf, rbuf, req);
    const double tw = MPI_Wtime();
    MPI_Waitall(12, req, MPI_STATUSES_IGNORE);
    st.t_wait += MPI_Wtime() - tw;

    for (int f = 0; f < 6 && !aware; ++f)   // boundary stream: ghosts in, then the shell
      if (g.nbr[f] != MPI_PROC_NULL)
        CK(cudaMemcpyAsync(drecv + g.boff[f], hrecv + g.boff[f], (g.boff[f + 1] - g.boff[f]) * sizeof(double),
                           cudaMemcpyHostToDevice, s_bd));
    for (int f = 0; f < 6; ++f) {
      if (g.nbr[f] == MPI_PROC_NULL) continue;
      const Face& F = g.recv[f];
      face_kernel<<<dim3(cdiv(F.na, 128), F.nb), 128, 0, s_bd>>>(u, drecv + g.boff[f], F, true);
    }
    for (const Box& b : shell)
      if (volume(b) > 0) shell_kernel<<<cdiv(volume(b), 256), 256, 0, s_bd>>>(u, v, b, sy, sz);
    CK(cudaGetLastError());
    CK(cudaStreamSynchronize(s_bd));
    CK(cudaStreamSynchronize(s_in));
    float ms = 0.0f;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    kernel_s += ms * 1e-3;
    std::swap(u, v);
  }
  st.t_loop = MPI_Wtime() - t0;
  st.t_inner = kernel_s;
  if (nin > 0 && kernel_s > 0.0) st.kernel_gbs = (double)nin * steps * BYTES_PER_LUP / kernel_s / 1e9;

  CK(cudaMemcpy(h_u, u, bytes, cudaMemcpyDeviceToHost));
  CK(cudaEventDestroy(e0));
  CK(cudaEventDestroy(e1));
  CK(cudaStreamDestroy(s_in));
  CK(cudaStreamDestroy(s_bd));
  CK(cudaFree(u));
  CK(cudaFree(v));
  CK(cudaFree(dsend));
  CK(cudaFree(drecv));
  if (!aware) {
    CK(cudaFreeHost(hsend));
    CK(cudaFreeHost(hrecv));
  }
  return st;
}
