#ifndef CONV1D_NAIVE_CUH
#define CONV1D_NAIVE_CUH

#include <cuda_runtime.h>

__global__ void conv1dNaive(int *input, int *kernel, int *output, int input_size, int kernel_size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid >= input_size - kernel_size + 1) {
    return;
  }

  output[tid] = 0;

  for (int i = 0; i < kernel_size; i++) {
    output[tid] += input[tid + i] * kernel[i];
  }
}

#endif