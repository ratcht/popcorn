#ifndef ROPE_NAIVE_CUH
#define ROPE_NAIVE_CUH

#include <cuda_runtime.h>


__global__ void rope_naive(float *x, float *cos, float *sin, int B, int L, int D) {
  int row = blockIdx.y*blockDim.y + threadIdx.y;
  int col = blockIdx.x*blockDim.x + threadIdx.x; // = d

  int l = row % L;

  int i = row * D + col; int j = l*(D/2) + (col/2);

  if (col % 2 == 0) {
    x[i] = x[i]*cos[j] - x[i+1]*sin[j];
  } else {
    x[i] = x[i]*cos[j] - x[i-1]*sin[j];
  }
}


#endif
