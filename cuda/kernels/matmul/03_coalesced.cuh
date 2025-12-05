#ifndef MATMUL_COALESCED_CUH
#define MATMUL_COALESCED_CUH

#include <cuda_runtime.h>

__global__ void matmulCoalesced(int *a, int *b, int *c, int n) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  int sum = 0;

  if ((row < n) && (col < n)) {
    for (int k = 0; k < n; k++) {
      sum += a[k * n + row] * b[k * n + col];
    }
    c[row * n + col] = sum;
  }
}

#endif