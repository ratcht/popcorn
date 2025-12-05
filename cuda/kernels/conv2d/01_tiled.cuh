#ifndef CONV2D_TILED_CUH
#define CONV2D_TILED_CUH

#include <cuda_runtime.h>

#define CONV2D_KERNEL_WIDTH 2
#define CONV2D_KERNEL_HEIGHT 2

__constant__ int conv2d_kernel[CONV2D_KERNEL_WIDTH * CONV2D_KERNEL_HEIGHT];

__global__ void conv2dTiled(int *input, int *output, int padded_input_width, int output_width, int output_height, int kernel_width, int kernel_height) {
  // input is pre-padded

  extern __shared__ int shmem[];

  // start values
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  int tile_width = blockDim.x + kernel_width - 1;

  // blockDim.x = number of columns = width
  // blockDim.y = number of rows = height
  // row*width + col;

  // load shmem at thread location
  shmem[threadIdx.y * tile_width + threadIdx.x] = input[row * padded_input_width + col];

  // load shmem at x + blockDim.x
  if (threadIdx.x < kernel_width - 1) {
    shmem[threadIdx.y * tile_width + threadIdx.x + blockDim.x] = input[row * padded_input_width + col + blockDim.x];
  }

  // load shmem at y + blockDim.y
  if (threadIdx.y < kernel_height - 1) {
    shmem[(threadIdx.y + blockDim.y) * tile_width + threadIdx.x] = input[(row + blockDim.y) * padded_input_width + col];
  }

  // load corner
  if (threadIdx.x < kernel_width - 1 && threadIdx.y < kernel_height - 1) {
    shmem[(threadIdx.y + blockDim.y) * tile_width + threadIdx.x + blockDim.x] = input[(row + blockDim.y) * padded_input_width + col + blockDim.x];
  }

  __syncthreads();

  if (col >= output_width || row >= output_height) {
    return;
  }

  int sum = 0;

  #pragma unroll
  for (int i = 0; i < kernel_height; i++) {
    #pragma unroll
    for (int j = 0; j < kernel_width; j++) {
      sum += shmem[(threadIdx.y + i) * tile_width + threadIdx.x + j] * conv2d_kernel[i * kernel_width + j];
    }
  }

  output[row * output_width + col] = sum;
}

#endif