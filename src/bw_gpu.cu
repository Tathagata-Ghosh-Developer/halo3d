// GPU bandwidth: device copy (kernel and cudaMemcpy D2D), the measured roof for the GPU roofline, plus pinned
// host <-> device copies, which is the path every host-staged halo face takes.
//   ./bin/bw_gpu [bytes per array] [repetitions]
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CK(call)                                                                                    \
  do {                                                                                              \
    cudaError_t e_ = (call);                                                                        \
    if (e_ != cudaSuccess) {                                                                        \
      std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
      std::exit(1);                                                                                 \
    }                                                                                               \
  } while (0)

__global__ void copy_kernel(const double* __restrict__ a, double* __restrict__ b, long n) {
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) b[i] = a[i];
}

// best time over reps of op(), in seconds, timed with events on the default stream
template <class F> static double best(int reps, F op) {
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  op();   // warm-up
  double t = 1e30;
  for (int r = 0; r < reps; ++r) {
    CK(cudaEventRecord(e0));
    op();
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    float ms;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    if (ms * 1e-3 < t) t = ms * 1e-3;
  }
  CK(cudaEventDestroy(e0));
  CK(cudaEventDestroy(e1));
  return t;
}

int main(int argc, char** argv) {
  const size_t bytes = argc > 1 ? std::atol(argv[1]) : (size_t)2 << 30;   // 2 GiB per array
  const int reps = argc > 2 ? std::atoi(argv[2]) : 20;
  const long n = bytes / sizeof(double);
  const size_t hb = (size_t)64 << 20;   // 64 MiB host transfers
  cudaDeviceProp p;
  CK(cudaGetDeviceProperties(&p, 0));
  double *a, *b, *h;
  CK(cudaMalloc(&a, bytes));
  CK(cudaMalloc(&b, bytes));
  CK(cudaMallocHost(&h, hb));
  CK(cudaMemset(a, 0, bytes));
  CK(cudaMemset(b, 0, bytes));
  const int blocks = 4 * p.multiProcessorCount * 8;
  const double tk = best(reps, [&] { copy_kernel<<<blocks, 256>>>(a, b, n); });
  const double tm = best(reps, [&] { CK(cudaMemcpy(b, a, bytes, cudaMemcpyDeviceToDevice)); });
  const double th2d = best(reps, [&] { CK(cudaMemcpy(a, h, hb, cudaMemcpyHostToDevice)); });
  const double td2h = best(reps, [&] { CK(cudaMemcpy(h, a, hb, cudaMemcpyDeviceToHost)); });
  CK(cudaGetLastError());
  std::printf("bw_gpu %s | copy kernel %.1f GB/s | cudaMemcpy D2D %.1f GB/s (read + write counted) | "
              "pinned H2D %.1f GB/s | pinned D2H %.1f GB/s\n",
              p.name, 2.0 * bytes / tk / 1e9, 2.0 * bytes / tm / 1e9, hb / th2d / 1e9, hb / td2h / 1e9);
  CK(cudaFree(a));
  CK(cudaFree(b));
  CK(cudaFreeHost(h));
  return 0;
}
