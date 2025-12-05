#ifndef CONV1D_TILED_CUH
#define CONV1D_TILED_CUH

#include <cuda_runtime.h>

#define CONV1D_TILED_KERNEL_SIZE 4

__constant__ int conv1d_tiled_kernel[CONV1D_TILED_KERNEL_SIZE];

__global__ void conv1dTiled(int *input, int *output, int input_size, int kernel_size) {
  extern __shared__ int shmem[];

  int g_tid = blockIdx.x * blockDim.x + threadIdx.x;
  int l_tid = threadIdx.x;

  // load shmem
  if (g_tid < input_size) {
    shmem[l_tid] = input[g_tid];
  }

  if (l_tid < kernel_size - 1 && g_tid + blockDim.x < input_size) {
    shmem[blockDim.x + l_tid] = input[g_tid + blockDim.x];
  }

  __syncthreads();

  if (g_tid >= input_size - kernel_size + 1) {
    return;
  }

  int sum = 0;

  for (int i = 0; i < kernel_size; i++) {
    sum += shmem[l_tid + i] * conv1d_tiled_kernel[i];
  }

  output[g_tid] = sum;
}

#endif