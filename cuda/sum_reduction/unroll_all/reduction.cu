/*
 * unroll all loops
 */

#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <cuda_runtime_api.h>
#include <curand.h>
#include <driver_types.h>
#include <stdio.h>
#include <math.h>

#define SIZE 128

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
__global__ void reduction(float *v, float *v_r, int n) {
  __shared__ float psum[SIZE];

  int tid = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

  // Bounds checking
  psum[threadIdx.x] = 0;
  if (tid < n) psum[threadIdx.x] = v[tid];
  if (tid + blockDim.x < n) psum[threadIdx.x] += v[tid + blockDim.x];

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

void printVector(const float* v, int n, const char* name) {
  printf("%s = [ ", name);
  for (int i = 0; i < n; i++) {
    printf("%f ", v[i]);
  }
  printf("]\n");
}



int main() {
  int device;
  cudaGetDevice(&device);

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);

  printf("Running on GPU %d: %s\n", device, prop.name);

  int n = 1 << 22;
  int bytes = sizeof(float) * n;

  float *h_v, *h_v_r;
  h_v = (float*)malloc(bytes);
  h_v_r = (float*)malloc(bytes);

  float *d_v;
  float *d_v_r;
  cudaMalloc(&d_v, bytes);
  cudaMalloc(&d_v_r, bytes);

  // fill device
  curandGenerator_t prng;
  curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

  curandSetPseudoRandomGeneratorSeed(prng, 462);

  curandGenerateUniform(prng, d_v, n);

  // print vector
  cudaMemcpy(h_v, d_v, bytes, cudaMemcpyDeviceToHost);
  // printVector(h_v, n, "V");

  printf("\n");

  int NUM_THREADS = SIZE;
  int NUM_BLOCKS = (n + NUM_THREADS - 1) / (NUM_THREADS * 2);

  reduction<128><<<NUM_BLOCKS, NUM_THREADS>>>(d_v, d_v_r, n);

  int remaining = NUM_BLOCKS;
  while (remaining > 1) {
    int blocks = (remaining + NUM_THREADS * 2 - 1) / (NUM_THREADS * 2);
    reduction<128><<<blocks, NUM_THREADS>>>(d_v_r, d_v_r, remaining);
    remaining = blocks;
  }


  cudaMemcpy(h_v_r, d_v_r, bytes, cudaMemcpyDeviceToHost);
  // printVector(h_v_r, n, "RESULT");

  float psum = 0;
  for (int i = 0; i < n; i++) {
    psum += h_v[i];
  }
  printf("CPU sum: %f, GPU sum: %f, diff: %f\n", psum, h_v_r[0], fabsf(psum - h_v_r[0]));
  assert(fabsf(psum - h_v_r[0]) < 1e-4 * psum);
  printf("\nsuccessful\n");

  cudaFree(d_v); cudaFree(d_v_r);
  free(h_v); free(h_v_r);

}
