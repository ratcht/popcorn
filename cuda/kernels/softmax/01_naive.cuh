#ifndef SOFTMAX_NAIVE_CUH
#define SOFTMAX_NAIVE_CUH

#include <cuda_runtime.h>
#include <math.h>

__global__ void softmaxNaive(float *a, float *b, int rows, int cols) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < rows) {
    float x_max = -INFINITY;
    float norm = 0.0f;

    for (int i = 0; i < cols; i++) {
      x_max = max(x_max, a[row * cols + i]);
    }

    for (int i = 0; i < cols; i++) {
      norm += expf(a[row * cols + i] - x_max);
    }

    for (int i = 0; i < cols; i++) {
      b[row * cols + i] = expf(a[row * cols + i] - x_max)/norm;
    }
  }
}

#endif