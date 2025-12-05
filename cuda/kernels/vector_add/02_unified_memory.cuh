#ifndef VECTOR_ADD_UNIFIED_MEMORY_CUH
#define VECTOR_ADD_UNIFIED_MEMORY_CUH

#include <cuda_runtime.h>

__global__ void vectorAddUM(int* a, int* b, int* c, int n) {
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (tid < n) {
    c[tid] = a[tid] + b[tid];
  }
}

#endif