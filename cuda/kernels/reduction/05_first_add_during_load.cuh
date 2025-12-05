#ifndef REDUCTION_FIRST_ADD_DURING_LOAD_CUH
#define REDUCTION_FIRST_ADD_DURING_LOAD_CUH

#include <cuda_runtime.h>

/*
 * unroll all loops
 */

#define REDUCTION_FIRST_ADD_SIZE 128

template<unsigned int blockSize>
__device__ void warpReduce(volatile float* shmem, int tid) {
  if (blockSize >= 64) shmem[tid] += shmem[tid + 32];
  if (blockSize >= 32) shmem[tid] += shmem[tid + 16];
  if (blockSize >= 16) shmem[tid] += shmem[tid + 8];
  if (blockSize >= 8) shmem[tid] += shmem[tid + 4];
  if (blockSize >= 4) shmem[tid] += shmem[tid + 2];
  if (blockSize >= 2) shmem[tid] += shmem[tid + 1];
}

template <unsigned int blockSize>
__global__ void reductionFirstAddDuringLoad(float *v, float *v_r, int n) {
  __shared__ float psum[REDUCTION_FIRST_ADD_SIZE];

  int tid = threadIdx.x;
  int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
  int gridSize = blockSize*2*gridDim.x;

  // Bounds checking
  psum[threadIdx.x] = 0;
  while (i + blockSize < n) {
    psum[tid] += v[i] + v[i + blockSize];
    i += gridSize;
  }
  if (i < n) {
    psum[tid] += v[i];
  }

  __syncthreads();

  if (blockDim.x >= 512) {
    if (threadIdx.x < 256) {
      psum[threadIdx.x] += psum[threadIdx.x + 256];
    }
    __syncthreads();
  }
  if (blockDim.x >= 256) {
    if (threadIdx.x < 128) {
      psum[threadIdx.x] += psum[threadIdx.x + 128];
    }
    __syncthreads();
  }
  if (blockDim.x >= 128) {
    if (threadIdx.x < 64) {
      psum[threadIdx.x] += psum[threadIdx.x + 64];
    }
    __syncthreads();
  }

  if (threadIdx.x < 32) { // do last warp ourselves
    warpReduce<blockSize>(psum, threadIdx.x);
  }

  if (threadIdx.x == 0) { // set the first thread in this block
    v_r[blockIdx.x] = psum[0];
  }
}

#endif
