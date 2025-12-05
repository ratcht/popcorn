#ifndef CONV1D_TILED_PADDED_CUH
#define CONV1D_TILED_PADDED_CUH

#include <cuda_runtime.h>

#define CONV1D_PADDED_KERNEL_SIZE 7

__constant__ int conv1d_padded_kernel[CONV1D_PADDED_KERNEL_SIZE];

__global__ void conv1dTiledPadded(int *input, int *output, int output_size, int kernel_size) {
  extern __shared__ int shmem[];

  int window_start = blockIdx.x * blockDim.x + threadIdx.x;
  int local_idx = threadIdx.x;

  shmem[local_idx] = input[window_start];

  if (local_idx < kernel_size - 1) {
    shmem[local_idx + blockDim.x] = input[window_start + blockDim.x];
  }

  __syncthreads();

  if (window_start >= output_size) {
    return;
  }

  int sum = 0;

  #pragma unroll
  for (int i = 0; i < kernel_size; i++) {
    sum += shmem[local_idx + i] * conv1d_padded_kernel[i];
  }

  output[window_start] = sum;
}

#endif