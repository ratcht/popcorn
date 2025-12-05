#ifndef CONV1D_STRIDED_PADDED_CUH
#define CONV1D_STRIDED_PADDED_CUH

#include <cuda_runtime.h>

#define CONV1D_STRIDED_KERNEL_SIZE 2

__constant__ int conv1d_strided_kernel[CONV1D_STRIDED_KERNEL_SIZE];

__global__ void conv1dStridedPadded(int *input, int *output, int input_size, int kernel_size, int padding, int stride, int output_size) {
  int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int input_idx = output_idx * stride - padding;

  if (output_idx >= output_size) {
    return;
  }

  int i_start = 0;
  if (input_idx < 0) i_start = -input_idx;

  int i_end = kernel_size;
  int rem = input_size - input_idx;
  if (rem < i_end) i_end = rem;

  int sum = 0;
  #pragma unroll
  for (int i = i_start; i < i_end; i++) {
    sum += input[input_idx + i] * conv1d_strided_kernel[i];
  }

  output[output_idx] = sum;
}

#endif