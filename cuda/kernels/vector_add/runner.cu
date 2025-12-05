#include "01_naive.cuh"
#include "02_unified_memory.cuh"

void run_vector_add_naive(int* d_a, int* d_b, int* d_c, int n) {
  int NUM_THREADS = 256;
  int NUM_BLOCKS = (n + NUM_THREADS - 1) / NUM_THREADS;
  vectorAdd<<<NUM_BLOCKS, NUM_THREADS>>>(d_a, d_b, d_c, n);
}

void run_vector_add_unified_memory(int* a, int* b, int* c, int n, int device_id) {
  int NUM_THREADS = 256;
  int NUM_BLOCKS = (n + NUM_THREADS - 1) / NUM_THREADS;

  int bytes = sizeof(int) * n;
  cudaMemPrefetchAsync(a, bytes, device_id);
  cudaMemPrefetchAsync(b, bytes, device_id);

  vectorAddUM<<<NUM_BLOCKS, NUM_THREADS>>>(a, b, c, n);

  cudaDeviceSynchronize();
  cudaMemPrefetchAsync(c, bytes, cudaCpuDeviceId);
}
