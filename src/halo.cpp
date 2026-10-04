// Domain decomposition and the non-blocking halo exchange (6 faces; a 7-point stencil needs no edges/corners).
#include "common.h"
#include <algorithm>

Grid make_grid(int n_global, int n_local) {
  Grid g{};
  int size, dims[3] = {0, 0, 0}, periods[3] = {0, 0, 0};
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Dims_create(size, 3, dims);                         // non-increasing: dims[0] >= dims[1] >= dims[2]
  for (int d = 0; d < 3; ++d) g.dims[d] = dims[2 - d];    // most cuts along z: z faces are contiguous planes
  MPI_Cart_create(MPI_COMM_WORLD, 3, g.dims, periods, 1, &g.cart);
  MPI_Comm_rank(g.cart, &g.rank);
  MPI_Cart_coords(g.cart, g.rank, 3, g.coords);
  for (int d = 0; d < 3; ++d) {
    g.N[d] = n_local > 0 ? n_local * g.dims[d] : n_global;
    const int q = g.N[d] / g.dims[d], rem = g.N[d] % g.dims[d], c = g.coords[d];
    g.n[d] = q + (c < rem ? 1 : 0);
    g.off[d] = c * q + std::min(c, rem);
    if (g.n[d] < 1) die("more ranks than cells along a dimension");
    MPI_Cart_shift(g.cart, d, 1, &g.nbr[2 * d], &g.nbr[2 * d + 1]);
  }
  g.s[0] = 1;
  g.s[1] = g.n[0] + 2;
  g.s[2] = g.s[1] * (g.n[1] + 2);
  g.cells = g.s[2] * (g.n[2] + 2);
  for (int f = 0; f < 6; ++f) {
    const int d = f / 2, hi = f % 2, d1 = d == 0 ? 1 : 0, d2 = d == 2 ? 1 : 2;   // d1, d2: the other two dims
    const long skip = g.s[d1] + g.s[d2];   // start at cell 1 of the other two dims (faces exclude ghosts)
    g.send[f] = {(hi ? g.n[d] : 1) * g.s[d] + skip, g.s[d1], g.s[d2], g.n[d1], g.n[d2]};
    g.recv[f] = {(hi ? g.n[d] + 1 : 0) * g.s[d] + skip, g.s[d1], g.s[d2], g.n[d1], g.n[d2]};
    g.boff[f + 1] = g.boff[f] + (long)g.n[d1] * g.n[d2];
  }
  return g;
}

// interior = cells whose stencil never reads a ghost that is being exchanged (walls are fixed zeros, so a side
// without a neighbour needs no waiting); shell = what is left, peeled as z slabs, then y, then x (<= 6 boxes,
// possibly empty; they may overlap when n < 3, which is harmless for a Jacobi update).
void make_boxes(const Grid& g, Box& in, Box sh[6]) {
  Box outer;
  for (int d = 0; d < 3; ++d) {
    outer.lo[d] = 1;
    outer.hi[d] = g.n[d] + 1;
    in.lo[d] = g.nbr[2 * d] == MPI_PROC_NULL ? 1 : 2;
    in.hi[d] = g.nbr[2 * d + 1] == MPI_PROC_NULL ? g.n[d] + 1 : g.n[d];
  }
  for (int d = 2; d >= 0; --d) {
    sh[2 * d] = sh[2 * d + 1] = outer;
    sh[2 * d].hi[d] = in.lo[d];
    sh[2 * d + 1].lo[d] = in.hi[d];
    outer.lo[d] = in.lo[d];
    outer.hi[d] = in.hi[d];
  }
}

// Face f goes to nbr[f] with tag f; that rank receives it on its opposite face f^1, so we expect tag f^1.
// Walls use MPI_PROC_NULL, which completes immediately and leaves the zero ghosts untouched.
void halo_start(const Grid& g, double* sendbuf, double* recvbuf, MPI_Request req[12]) {
  for (int f = 0; f < 6; ++f) {
    const int cnt = (int)(g.boff[f + 1] - g.boff[f]);
    MPI_Irecv(recvbuf + g.boff[f], cnt, MPI_DOUBLE, g.nbr[f], f ^ 1, g.cart, &req[f]);
    MPI_Isend(sendbuf + g.boff[f], cnt, MPI_DOUBLE, g.nbr[f], f, g.cart, &req[6 + f]);
  }
}
