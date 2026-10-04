// halo3d: 3D heat diffusion (explicit 7-point Jacobi stencil), MPI + OpenMP + CUDA.
#include "common.h"
#include <omp.h>
#include <string>

static const char* USAGE =
    "usage: halo3d [--backend omp|cuda] [--n N | --local n] [--steps S] [--cuda-aware] [--csv FILE]\n"
    "  --n N         global grid N^3 (strong scaling), default 256\n"
    "  --local n     n^3 cells per rank; the global grid grows with the rank count (weak scaling)\n"
    "  --steps S     time steps, default 100\n"
    "  --cuda-aware  give device pointers to MPI (needs a CUDA-aware MPI); default stages halos via pinned host memory\n"
    "  --csv FILE    append one result row to FILE";

int main(int argc, char** argv) {
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);   // only the master thread calls MPI
  std::string backend = "omp", csv;
  int n = 256, local = 0, steps = 100;
  bool aware = false;
  for (int a = 1; a < argc; ++a) {
    const std::string k = argv[a];
    if (k == "--cuda-aware") aware = true;
    else if (a + 1 >= argc) die(USAGE);
    else if (k == "--backend") backend = argv[++a];
    else if (k == "--n") n = std::atoi(argv[++a]);
    else if (k == "--local") local = std::atoi(argv[++a]);
    else if (k == "--steps") steps = std::atoi(argv[++a]);
    else if (k == "--csv") csv = argv[++a];
    else die(USAGE);
  }
  if (steps < 1 || (local < 1 && n < 1)) die(USAGE);

  Grid g = make_grid(n, local);
  double* u = alloc_field(g, true);
  Stats st{};
  if (backend == "omp") st = run_omp(g, u, steps);
#ifdef HALO3D_CUDA
  else if (backend == "cuda") st = run_cuda(g, u, steps, aware);
#endif
  else die("unknown backend (the cuda backend lives in bin/halo3d_cuda)");

  // Verification. The sine mode is an eigenvector of the 7-point operator, so after S steps the exact
  // discrete solution is G^S * u0 with G = 1 - 4r * sum_d sin^2(pi / (2(N_d+1))). Anything above round-off
  // is a bug (wrong halo, wrong face, race). The continuous PDE decays as exp(-r S pi^2 sum_d 1/(N_d+1)^2);
  // that gap is the O(h^2 + dt) discretisation error, printed for reference only.
  double G = 1.0, lam = 0.0;
  for (int d = 0; d < 3; ++d) {
    const double th = PI / (g.N[d] + 1);
    G -= 4.0 * R * std::sin(th / 2) * std::sin(th / 2);
    lam += th * th;
  }
  const double amp = std::pow(G, steps), amp_pde = std::exp(-R * steps * lam);
  const auto sx = mode1d(g, 0), sy = mode1d(g, 1), sz = mode1d(g, 2);
  double e = 0.0, e_pde = 0.0, nrm = 0.0;
#pragma omp parallel for collapse(2) reduction(+ : e, e_pde, nrm)
  for (int k = 1; k <= g.n[2]; ++k)
    for (int j = 1; j <= g.n[1]; ++j)
      for (int i = 1; i <= g.n[0]; ++i) {
        const double m = sx[i] * sy[j] * sz[k], x = u[i + j * g.s[1] + k * g.s[2]];
        e += (x - amp * m) * (x - amp * m);
        e_pde += (x - amp_pde * m) * (x - amp_pde * m);
        nrm += amp * m * amp * m;
      }
  double loc[3] = {e, e_pde, nrm}, sum[3], t[3] = {st.t_loop, st.t_wait, st.t_inner}, tmax[3];
  MPI_Allreduce(loc, sum, 3, MPI_DOUBLE, MPI_SUM, g.cart);
  MPI_Reduce(t, tmax, 3, MPI_DOUBLE, MPI_MAX, 0, g.cart);   // the slowest rank sets the pace
  MPI_Comm node;                                            // node count = ranks that are node-local rank 0
  int lrank, leader, nodes;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node);
  MPI_Comm_rank(node, &lrank);
  MPI_Comm_free(&node);
  leader = lrank == 0;
  MPI_Allreduce(&leader, &nodes, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  const double err = std::sqrt(sum[0] / sum[2]), err_pde = std::sqrt(sum[1] / sum[2]);
  const bool pass = err < TOL;   // false for NaN too

  if (g.rank == 0) {
    const int P = g.dims[0] * g.dims[1] * g.dims[2], T = omp_get_max_threads();
    const double glups = (double)g.N[0] * g.N[1] * g.N[2] * steps / tmax[0] / 1e9, gbs = glups * BYTES_PER_LUP;
    std::printf("halo3d %s%s | nodes %d | ranks %d = %dx%dx%d | threads/rank %d | global %dx%dx%d | local(rank 0) %dx%dx%d | steps %d\n",
                backend.c_str(), aware ? " (cuda-aware)" : "", nodes, P, g.dims[0], g.dims[1], g.dims[2], T, g.N[0],
                g.N[1], g.N[2], g.n[0], g.n[1], g.n[2], steps);
    std::printf("time %.4f s (%.3f ms/step) | halo wait %.4f s | interior %.4f s | %.3f GLUP/s | %.1f GB/s at %d B/LUP",
                tmax[0], 1e3 * tmax[0] / steps, tmax[1], tmax[2], glups, gbs, BYTES_PER_LUP);
    if (st.kernel_gbs > 0) std::printf(" | interior kernel %.1f GB/s (rank 0 GPU)", st.kernel_gbs);
    std::printf("\nrel. L2 error vs exact discrete solution %.3e (%s, tol %.0e) | vs continuous PDE %.3e\n", err,
                pass ? "PASS" : "FAIL", TOL, err_pde);
    if (!csv.empty()) {
      bool fresh = true;
      if (FILE* old = std::fopen(csv.c_str(), "r")) { fresh = false; std::fclose(old); }
      FILE* f = std::fopen(csv.c_str(), "a");
      if (!f) die("cannot open the --csv file");
      if (fresh)
        std::fprintf(f, "backend,cuda_aware,nodes,ranks,threads,px,py,pz,nx,ny,nz,steps,time_s,wait_s,inner_s,glups,gbs,"
                        "kernel_gbs,err\n");
      std::fprintf(f, "%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.4f,%.2f,%.2f,%.3e\n", backend.c_str(),
                   aware ? 1 : 0, nodes, P, T, g.dims[0], g.dims[1], g.dims[2], g.N[0], g.N[1], g.N[2], steps, tmax[0],
                   tmax[1], tmax[2], glups, gbs, st.kernel_gbs, err);
      std::fclose(f);
    }
  }
  delete[] u;
  MPI_Comm_free(&g.cart);
  MPI_Finalize();
  return pass ? 0 : 1;
}
