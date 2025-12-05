#ifndef SOFTMAX_WARP_SHUFFLE_CUH
#define SOFTMAX_WARP_SHUFFLE_CUH

#include <cuda_runtime.h>
#include <math.h>

#define MASK 0xFFFFFFFF

template<typename T, typename Op>
__device__ T warp_reduce(T val, Op op) {
  // warp-level reduction
  for (int i = 16; i > 0; i /= 2) {
    T other = __shfl_xor_sync(MASK, val, i);
    val = op(val, other);
  }    // now each thread in warp has the warp's maximum

  return val;
}

template<typename T, typename Op>
__device__ T block_reduce(T val, Op op, float* shmem, int tid, int warp_id) { // i.e row reduce (since each block = row in this case)

    val = warp_reduce(val, op);

    // first thread in warp stores to shmem
    if (tid % warpSize == 0) {
      shmem[warp_id] = val;
    }

    __syncthreads();

    if (warp_id == 0) {
      // first warp performs reduction across all warp maximums
      val = tid < blockDim.x/warpSize ? shmem[tid] : 0;

      // another warp shuffle using the warp maximums (loaded into warp 0)
      val = warp_reduce(val, op);
    }

    if (tid == 0) {
      shmem[0] = val;
    }

    __syncthreads();
    return shmem[0];
}

__global__ void softmaxWarpShuffle(float *a, float *b, int rows, int cols) {
  int row = blockIdx.x;
  int tid = threadIdx.x;
  int warp_id = tid / 32;

  extern __shared__ float shmem[];

  if (row < rows) {
    float x_max = -INFINITY;
    float divisor = 0.0f;

    // ===== calculate x_max ======

    // thread reduction on assigned elems
    for (int i = tid; i < cols; i += blockDim.x) {
      x_max = max(x_max, a[row*cols + i]);
    }

    x_max = block_reduce(x_max, fmaxf, shmem, tid, warp_id);

    // ===== calculate divisor ======
    for (int i = tid; i < cols; i += blockDim.x) {
      divisor += expf(a[row*cols + i] - x_max);
    }

    divisor = block_reduce(divisor, [](float a, float b){return a+b;}, shmem, tid, warp_id);

    // ===== OUTPUT ======
    for (int i = tid; i < cols; i += blockDim.x) {
      b[row*cols + i] = expf(a[row*cols + i] - x_max)/divisor;
    }
  }
}

#endif