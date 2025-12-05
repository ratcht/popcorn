#ifndef MATMUL_NAIVE_SQUARE_CUH
#define MATMUL_NAIVE_SQUARE_CUH

#include <cuda_runtime.h>

__global__ void matmulNaiveSquare(int *a, int *b, int *c, int n) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  int sum = 0;

  if ((row < n) && (col < n)) {
    for (int k = 0; k < n; k++) {
      sum += a[row * n + k] * b[k * n + col];
    }
    c[row * n + col] = sum;
  }
}

#endif