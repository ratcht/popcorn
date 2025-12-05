#ifndef MATMUL_NAIVE_CUH
#define MATMUL_NAIVE_CUH

#include <cuda_runtime.h>

__global__ void matmulNaive(int *a, int *b, int *c, int m_1, int n_1, int m_2, int n_2) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  int sum = 0;

  if ((row < m_2) && (col < n_1) && (n_1 == m_2)) {
    for (int k = 0; k < n_1; k++) {
      sum += a[row * n_1 + k] * b[k * n_2 + col];
    }
    c[row * n_2 + col] = sum;
  }
}

#endif