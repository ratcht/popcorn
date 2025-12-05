#ifndef SOFTMAX_SHARED_MEMORY_CUH
#define SOFTMAX_SHARED_MEMORY_CUH

#include <cuda_runtime.h>
#include <math.h>

__global__ void softmaxSharedMemory(float *a, float *b, int rows, int cols) {
  int row = blockIdx.x;
  int tid = threadIdx.x;

  extern __shared__ float shmem[];

  if (row < rows) {
    float x_max = -INFINITY;
    float divisor = 0.0f;

    // ===== calculate x_max ======
    for (int i = tid; i < cols; i += blockDim.x) {
      x_max = max(x_max, a[row*cols + i]);
    }

    shmem[tid] = x_max;

    for (int s = blockDim.x/2; s >= 1; s /= 2) {
      __syncthreads();
      if (tid < s) {
        shmem[tid] = max(shmem[tid], shmem[tid + s]);
      }
    }

    __syncthreads();
    x_max = shmem[0];

    // ===== calculate divisor ======
    for (int i = tid; i < cols; i += blockDim.x) {
      divisor += expf(a[row*cols + i] - x_max);
    }

    shmem[tid] = divisor;

    for (int s = blockDim.x/2; s >= 1; s /= 2) {
      __syncthreads();
      if (tid < s) {
        shmem[tid] = shmem[tid] + shmem[tid + s];
      }
    }

    __syncthreads();
    divisor = shmem[0];

    // ===== OUTPUT ======
    for (int i = tid; i < cols; i += blockDim.x) {
      b[row*cols + i] = expf(a[row*cols + i] - x_max)/divisor;
    }
  }
}

#endif