#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <cuda_runtime_api.h>
#include <cooperative_groups.h>
#include <curand.h>
#include <driver_types.h>
#include <stdio.h>
#include <math.h>

#define SIZE 128

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


__global__ void  reduction(float *v, float *v_r, int n) {
  float sum = thread_sum(v, n);

  extern __shared__ float shmem[];

  auto g = this_thread_block(); // unique id of this thread block

  float block_sum = reduce_sum(g, shmem, sum);

  if (g.thread_rank() == 0) {
    atomicAdd(v_r, block_sum);
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

  // start code

  int n = 1 << 28;
  size_t bytes = sizeof(float) * n;

  float *v;
  float *v_r;

  cudaMallocManaged(&v, bytes);
  cudaMallocManaged(&v_r, sizeof(float));
  *v_r = 0;

  // fill on device
  for (int i = 0; i < n; i++) {
    v[i] = 1.0f;
  }


  int NUM_THREADS = SIZE;
  int NUM_BLOCKS = 1024;

  reduction<<<NUM_BLOCKS, NUM_THREADS, NUM_THREADS*sizeof(float)>>>(v, v_r, n);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
      printf("Kernel launch error: %s\n", cudaGetErrorString(err));
  }
  cudaDeviceSynchronize();


  cudaDeviceSynchronize(); // wait for device
  cudaMemPrefetchAsync(v, bytes, cudaCpuDeviceId);
  cudaDeviceSynchronize();

  float psum = 0;
  for (int i = 0; i < n; i++) {
      psum += v[i];
  }

  printf("CPU sum: %f, GPU sum: %f, diff: %f\n", psum, v_r[0], fabsf(psum - v_r[0]));
  assert(fabsf(psum - v_r[0]) < 1e-4 * psum);
  printf("\nsuccessful\n");

}
