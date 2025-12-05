#ifndef REDUCTION_NO_BANK_CONFLICTS_CUH
#define REDUCTION_NO_BANK_CONFLICTS_CUH

#include <cuda_runtime.h>

/*
 * no bank conflicts reduction case:
 *  start at large stride, work down
 */

#define REDUCTION_NO_BANK_CONFLICTS_SIZE 16

__global__ void reductionNoBankConflicts(float *v, float *v_r) {
  __shared__ float psum[REDUCTION_NO_BANK_CONFLICTS_SIZE];

  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  psum[threadIdx.x] = v[tid];
  __syncthreads();

  for (int s = blockDim.x/2; s > 0; s >>= 1) { // iterate strides in block

    if (threadIdx.x < s) {
      psum[threadIdx.x] += psum[threadIdx.x + s];
    }
    __syncthreads(); // wait for this step to be done
  }

  if (threadIdx.x == 0) { // set the first thread in this block
    v_r[blockIdx.x] = psum[0];
  }
}

#endif