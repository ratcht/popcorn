#ifndef REDUCTION_DIVERGENT_CUH
#define REDUCTION_DIVERGENT_CUH

#include <cuda_runtime.h>

/*
 * warp divergence case:
 *  to figure out which thread is working, we
 *  did modulo of 2x the stride
 */

#define REDUCTION_DIVERGENT_SIZE 16

__global__ void reductionDivergent(float *v, float *v_r) {
  __shared__ float psum[REDUCTION_DIVERGENT_SIZE];

  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  psum[threadIdx.x] = v[tid];
  __syncthreads();

  for (int s = 1; s < blockDim.x; s *= 2) { // iterate strides in block
    if (threadIdx.x % (2*s) == 0) {
      psum[threadIdx.x] += psum[threadIdx.x + s];
    }
      __syncthreads(); // wait for this step to be done
  }

  if (threadIdx.x == 0) { // set the first thread in this block
    v_r[blockIdx.x] = psum[0];
  }
}

#endif