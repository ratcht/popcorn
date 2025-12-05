#ifndef REDUCTION_BANK_CONFLICTS_CUH
#define REDUCTION_BANK_CONFLICTS_CUH

#include <cuda_runtime.h>

/*
 * bank reduction case:
 *  to use sequential threads we calculate index
 *  as shown below
 */

#define REDUCTION_BANK_CONFLICTS_SIZE 16

__global__ void reductionBankConflicts(float *v, float *v_r) {
  __shared__ float psum[REDUCTION_BANK_CONFLICTS_SIZE];

  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  psum[threadIdx.x] = v[tid];
  __syncthreads();

  for (int s = 1; s < blockDim.x; s *= 2) { // iterate strides in block
    // ! THIS IS THE ONLY CHANGED PART
    int index = 2 * s * threadIdx.x;

    if (index < blockDim.x) {
      psum[index] += psum[index + s];
    }
    __syncthreads(); // wait for this step to be done
  }

  if (threadIdx.x == 0) { // set the first thread in this block
    v_r[blockIdx.x] = psum[0];
  }
}

#endif