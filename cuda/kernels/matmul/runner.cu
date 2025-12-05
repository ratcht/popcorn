#include "01_naive.cuh"
#include "02_naive_square.cuh"
#include "03_coalesced.cuh"
#include "04_tiled.cuh"

void run_matmul_naive(int *d_a, int *d_b, int *d_c, int m_1, int n_1, int m_2, int n_2) {
  dim3 threads(16, 16);
  dim3 grid((n_2 + 15) / 16, (m_1 + 15) / 16);
  matmulNaive<<<grid, threads>>>(d_a, d_b, d_c, m_1, n_1, m_2, n_2);
}

void run_matmul_naive_square(int *d_a, int *d_b, int *d_c, int n) {
  dim3 threads(16, 16);
  dim3 grid((n + 15) / 16, (n + 15) / 16);
  matmulNaiveSquare<<<grid, threads>>>(d_a, d_b, d_c, n);
}

void run_matmul_coalesced(int *d_a_t, int *d_b, int *d_c, int n) {
  dim3 threads(16, 16);
  dim3 grid((n + 15) / 16, (n + 15) / 16);
  matmulCoalesced<<<grid, threads>>>(d_a_t, d_b, d_c, n);
}

void run_matmul_tiled(int *d_a, int *d_b, int *d_c, int n) {
  dim3 threads(16, 16);
  dim3 grid((n + 15) / 16, (n + 15) / 16);
  matmulTiled<<<grid, threads>>>(d_a, d_b, d_c, n);
}