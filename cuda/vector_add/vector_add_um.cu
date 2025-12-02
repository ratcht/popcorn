#include <cuda_runtime.h>
#include <stdio.h>

__global__ void vectorAddUM(int* a, int* b, int* c, int n) {
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

  int id = cudaGetDevice(&id); // get device id

  int n = 1 << 16; // vector size of 2^16
  int bytes = sizeof(int) * n;

  int *a, *b, *c;

  // managed cuda alloc
  cudaMallocManaged(&a, bytes);
  cudaMallocManaged(&b, bytes);
  cudaMallocManaged(&c, bytes);


  // init host arrays
  for (int i = 0; i < n; i++) {
    a[i] = i;
    b[i] = 2 * i;
  }


  int NUM_THREADS = 256; // block size
  int NUM_BLOCKS = (n + NUM_THREADS - 1) / NUM_THREADS; // grid size

  cudaMemPrefetchAsync(a, bytes, id); // in the bg, start transferring data
  cudaMemPrefetchAsync(b, bytes, id); // in the bg, start transferring data

  vectorAddUM<<<NUM_BLOCKS, NUM_THREADS>>>(a, b, c, n);

  cudaDeviceSynchronize(); // wait for device

  cudaMemPrefetchAsync(c, bytes, cudaCpuDeviceId);

  printVector(a, 10, "A");
  printVector(b, 10, "B");
  printVector(c, 10, "C = A + B");

  cudaFree(a); cudaFree(b); cudaFree(c);

}
