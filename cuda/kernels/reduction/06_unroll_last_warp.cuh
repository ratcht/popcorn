#ifndef REDUCTION_UNROLL_LAST_WARP_CUH
#define REDUCTION_UNROLL_LAST_WARP_CUH

#include <cuda_runtime.h>

/*
 * unroll just the last loop
 */

#define REDUCTION_UNROLL_LAST_SIZE 16

__device__ void warpReduceUnrollLast(volatile float* shmem, int tid) {
  shmem[tid] += shmem[tid + 32];
  shmem[tid] += shmem[tid + 16];
  shmem[tid] += shmem[tid + 8];
  shmem[tid] += shmem[tid + 4];
  shmem[tid] += shmem[tid + 2];
  shmem[tid] += shmem[tid + 1];
}

__global__ void reductionUnrollLastWarp(float *v, float *v_r) {
  __shared__ float psum[REDUCTION_UNROLL_LAST_SIZE];

  int tid = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

  psum[threadIdx.x] = v[tid] + v[tid + blockDim.x];

  __syncthreads();

  for (int s = blockDim.x/2; s > 32; s >>= 1) { // iterate strides in block

    if (threadIdx.x < s) {
      psum[threadIdx.x] += psum[threadIdx.x + s];
    }
    __syncthreads(); // wait for this step to be done
  }

  if (threadIdx.x < 32) { // do last warp ourselves
    warpReduceUnrollLast(psum, threadIdx.x);
  }

  if (threadIdx.x == 0) { // set the first thread in this block
    v_r[blockIdx.x] = psum[0];
  }
}

#endif