#ifndef REDUCTION_COOPERATIVE_GROUPS_CUH
#define REDUCTION_COOPERATIVE_GROUPS_CUH

#include <cuda_runtime.h>
#include <cooperative_groups.h>

#define REDUCTION_COOPERATIVE_SIZE 128

using namespace cooperative_groups;


__device__ float reduce_sum(thread_block g, float* shmem, float val) {
  int lane = g.thread_rank(); // get the thread inside thread block

  for (int i = g.size() / 2; i > 0; i /= 2) {
    shmem[lane] = val;

    g.sync(); // syncthreads is legal too -> but may cause deadlocks though

    if (lane < i) {
      val += shmem[lane + i];
    }

    g.sync();
  }

  return val;
}

__device__ float thread_sum(float *input, int n) {
  float sum = 0;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = tid; i < n / 4; i += blockDim.x*gridDim.x) {
    float4 in = ((float4*)input)[i];
    sum += in.x + in.y + in.z + in.w;
  }
  return sum;
}


__global__ void reductionCooperativeGroups(float *v, float *v_r, int n) {
  float sum = thread_sum(v, n);

  extern __shared__ float shmem[];

  auto g = this_thread_block(); // unique id of this thread block

  float block_sum = reduce_sum(g, shmem, sum);

  if (g.thread_rank() == 0) {
    atomicAdd(v_r, block_sum);
  }
}

#endif