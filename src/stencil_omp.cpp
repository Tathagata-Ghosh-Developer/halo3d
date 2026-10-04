// OpenMP backend: first-touch allocation, host face packing, time loop with comm/compute overlap.
#include "common.h"
#include <cstring>
#include <utility>

// Pages are placed by the thread that first writes them, so initialise with the same static (k, j) split the
// update loop uses: on a multi-socket node each thread then streams from its own NUMA domain.
double* alloc_field(const Grid& g, bool with_mode) {
  double* u = new double[g.cells];   // deliberately uninitialised: the loop below is the first touch
  const auto sx = mode1d(g, 0), sy = mode1d(g, 1), sz = mode1d(g, 2);
  const int X = g.n[0] + 1, Y = g.n[1] + 1, Z = g.n[2] + 1;   // index of the high ghost layer
#pragma omp parallel for collapse(2) schedule(static)
  for (int k = 0; k <= Z; ++k)
    for (int j = 0; j <= Y; ++j)
      for (int i = 0; i <= X; ++i) {
        const bool inner = with_mode && i > 0 && i < X && j > 0 && j < Y && k > 0 && k < Z;
        u[i + j * g.s[1] + k * g.s[2]] = inner ? sx[i] * sy[j] * sz[k] : 0.0;
      }
  return u;
}

static void update(const Grid& g, const double* __restrict u, double* __restrict v, const Box& b) {
  const long sy = g.s[1], sz = g.s[2];
#pragma omp parallel for collapse(2) schedule(static)
  for (int k = b.lo[2]; k < b.hi[2]; ++k)
    for (int j = b.lo[1]; j < b.hi[1]; ++j) {
      const long o = k * sz + j * sy;
#pragma omp simd
      for (int i = b.lo[0]; i < b.hi[0]; ++i) {
        const long c = o + i;
        v[c] = C0 * u[c] + R * (u[c - 1] + u[c + 1] + u[c - sy] + u[c + sy] + u[c - sz] + u[c + sz]);
      }
    }
}

// Pack interior faces into buf (unpack = false) or unpack buf into the ghost faces; wall faces are skipped.
static void copy_faces(const Grid& g, double* u, double* buf, bool unpack) {
  for (int f = 0; f < 6; ++f) {
    if (g.nbr[f] == MPI_PROC_NULL) continue;
    const Face F = unpack ? g.recv[f] : g.send[f];
    double* q = buf + g.boff[f];
#pragma omp parallel for schedule(static)
    for (int b = 0; b < F.nb; ++b)
      for (int a = 0; a < F.na; ++a) {
        double& cell = u[F.base + a * F.sa + b * F.sb];
        double& slot = q[a + (long)F.na * b];
        if (unpack) cell = slot; else slot = cell;
      }
  }
}

Stats run_omp(const Grid& g, double* u0, int steps) {
  double *u = u0, *v = alloc_field(g, false);   // v needs zero ghosts too
  std::vector<double> sbuf(g.boff[6]), rbuf(g.boff[6]);
  Box in, shell[6];
  make_boxes(g, in, shell);
  MPI_Request req[12];
  Stats st{};
  MPI_Barrier(g.cart);
  const double t0 = MPI_Wtime();
  for (int s = 0; s < steps; ++s) {
    copy_faces(g, u, sbuf.data(), false);
    halo_start(g, sbuf.data(), rbuf.data(), req);
    const double ti = MPI_Wtime();
    update(g, u, v, in);   // overlaps with the messages in flight (if the MPI library progresses them)
    const double tw = MPI_Wtime();
    st.t_inner += tw - ti;
    MPI_Waitall(12, req, MPI_STATUSES_IGNORE);
    st.t_wait += MPI_Wtime() - tw;
    copy_faces(g, u, rbuf.data(), true);
    for (const Box& b : shell) update(g, u, v, b);
    std::swap(u, v);
  }
  st.t_loop = MPI_Wtime() - t0;
  if (u != u0) std::memcpy(u0, u, g.cells * sizeof(double));
  delete[] (u == u0 ? v : u);
  return st;
}
