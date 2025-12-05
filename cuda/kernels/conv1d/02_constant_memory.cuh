#ifndef CONV1D_CONSTANT_MEMORY_CUH
#define CONV1D_CONSTANT_MEMORY_CUH

#include <cuda_runtime.h>

#define CONV1D_KERNEL_SIZE 4

__constant__ int conv1d_kernel[CONV1D_KERNEL_SIZE];

__global__ void conv1dConstantMemory(int *input, int *output, int input_size, int kernel_size) {
  int g_tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (g_tid >= input_size - kernel_size + 1) {
    return;
  }

  int sum = 0;

  for (int i = 0; i < kernel_size; i++) {
    sum += input[g_tid + i] * conv1d_kernel[i];
  }

  output[g_tid] = sum;
}

#endif