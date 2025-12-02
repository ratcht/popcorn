/*
 * warp divergence case:
 *  to figure out which thread is working, we
 *  did modulo of 2x the stride
 */

#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <cuda_runtime_api.h>
#include <curand.h>
#include <driver_types.h>
#include <stdio.h>
#include <math.h>

#define SIZE 16
#define SHMEM_SIZE 16 * 4 // num threads * sizeof(int)

__global__ void reduction(float *v, float *v_r) {
  __shared__ float psum[SHMEM_SIZE];

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

  int n = 1 << 6; // vector size of 2^3
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

  curandSetPseudoRandomGeneratorSeed(prng, 42);

  curandGenerateUniform(prng, d_v, n);

  // print vector
  cudaMemcpy(h_v, d_v, bytes, cudaMemcpyDeviceToHost);
  printVector(h_v, n, "V");

  printf("\n");

  int NUM_THREADS = SIZE;
  int NUM_BLOCKS = (n + NUM_THREADS - 1) / NUM_THREADS;

  reduction<<<NUM_BLOCKS, NUM_THREADS>>>(d_v, d_v_r);

  reduction<<<1, NUM_THREADS>>>(d_v_r, d_v_r);


  cudaMemcpy(h_v_r, d_v_r, bytes, cudaMemcpyDeviceToHost);
  printVector(h_v_r, n, "RESULT");

  float psum = 0;
  for (int i = 0; i < n; i++) {
    psum += h_v[i];
  }

  assert(fabsf(psum - h_v_r[0]) < 1e-3);
  printf("\nsuccessful\n");

  cudaFree(d_v); cudaFree(d_v_r);
  free(h_v); free(h_v_r);

}
