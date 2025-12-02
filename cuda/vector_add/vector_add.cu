#include <cuda_runtime.h>
#include <stdio.h>

__global__ void vectorAdd(int* a, int* b, int* c, int n) {
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (tid < n) {
    c[tid] = a[tid] + b[tid];
  }
}

void printVector(const int* v, int n, const char* name) {
  printf("%s = [ ", name);
  for (int i = 0; i < n; i++) {
    printf("%d ", v[i]);
  }
  printf("]\n");
}

int main() {
  int device;
  cudaGetDevice(&device);

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);

  printf("Running on GPU %d: %s\n", device, prop.name);

  int n = 1 << 16; // vector size of 2^16
  int bytes = sizeof(int) * n;

  int *h_a, *h_b, *h_c;
  int *d_a, *d_b, *d_c;

  h_a = (int*)malloc(bytes);
  h_b = (int*)malloc(bytes);
  h_c = (int*)malloc(bytes);

  // init host arrays
  for (int i = 0; i < n; i++) {
      h_a[i] = i;
      h_b[i] = 2 * i;
  }

  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, bytes);

  cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

  int NUM_THREADS = 256;
  int NUM_BLOCKS = (n + NUM_THREADS - 1) / NUM_THREADS;

  vectorAdd<<<NUM_BLOCKS, NUM_THREADS>>>(d_a, d_b, d_c, n);

  cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

  printVector(h_a, 10, "A");
  printVector(h_b, 10, "B");
  printVector(h_c, 10, "C = A + B");

  cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

}
