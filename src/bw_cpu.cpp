// STREAM-style copy and triad bandwidth with OpenMP: the measured memory roof for the CPU roofline.
// Arrays are first-touched with the same static schedule as the kernels, as in the stencil.
// Byte counting follows STREAM (copy 16 B, triad 24 B per element; write-allocate traffic is not counted),
// which matches the 16 B/LUP model of the stencil, so stencil ceiling = copy GB/s / 16 B.
//   OMP_NUM_THREADS=48 OMP_PLACES=cores OMP_PROC_BIND=spread ./bin/bw_cpu [elements per array] [repetitions]
#include <omp.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv) {
  const long n = argc > 1 ? std::atol(argv[1]) : 1L << 27;   // 1 GiB per array: far beyond the caches
  const int reps = argc > 2 ? std::atoi(argv[2]) : 20;
  double *a = new double[n], *b = new double[n], *c = new double[n];
#pragma omp parallel for schedule(static)
  for (long i = 0; i < n; ++i) { a[i] = 1.0; b[i] = 2.0; c[i] = 0.0; }
  double copy = 1e30, triad = 1e30;
  for (int r = 0; r < reps; ++r) {   // best of reps, as STREAM reports
    double t = omp_get_wtime();
#pragma omp parallel for schedule(static)
    for (long i = 0; i < n; ++i) c[i] = a[i];
    copy = std::min(copy, omp_get_wtime() - t);
    t = omp_get_wtime();
#pragma omp parallel for schedule(static)
    for (long i = 0; i < n; ++i) a[i] = b[i] + 3.0 * c[i];
    triad = std::min(triad, omp_get_wtime() - t);
  }
  double sum = 0.0;
#pragma omp parallel for reduction(+ : sum)
  for (long i = 0; i < n; i += 4096) sum += a[i] + c[i];
  std::printf("bw_cpu threads %d | elements %ld | copy %.1f GB/s | triad %.1f GB/s | checksum %g\n",
              omp_get_max_threads(), n, 16.0 * n / copy / 1e9, 24.0 * n / triad / 1e9, sum);
  delete[] a;
  delete[] b;
  delete[] c;
  return 0;
}
