#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <cuda_runtime_api.h>
#include <curand.h>
#include <driver_types.h>
#include <stdio.h>
#include <math.h>


__global__ void conv1d(int *input, int *kernel, int *output, int input_size, int kernel_size) {
  int tid = blockIdx.x*blockDim.x + threadIdx.x;

  if (tid >= input_size - kernel_size + 1) {
    return;
  }

  output[tid] = 0;

  for (int i = 0; i < kernel_size; i++) {
    output[tid] += input[tid+i] * kernel[i];
  }
}

void printVector(const int* v, int n, const char* name) {
  printf("%s = [ ", name);
  for (int i = 0; i < n; i++) {
    printf("%d ", v[i]);
  }
  printf("]\n");
}

bool validate_conv1d(int *input, int *kernel, int *gpu_output, int input_size, int kernel_size) {
  int output_size = input_size - kernel_size + 1;

  for (int i = 0; i < output_size; i++) {
    int expected = 0;
    for (int j = 0; j < kernel_size; j++) {
      expected += input[i+j] * kernel[j];
    }

    if (expected != gpu_output[i]) {
      printf("Mismatch at index %d: Expected=%d, GPU=%d\n", i, expected, gpu_output[i]);
      return false;
    }
  }
  return true;
}

int main() {
  int device;
  cudaGetDevice(&device);

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);

  printf("Running on GPU %d: %s\n", device, prop.name);

  int L = 1 << 12; // vector size of 2^16
  int KERNEL_SIZE = 2;
  int L_OUT = (L - KERNEL_SIZE + 1);

  int *h_input, *h_kernel, *h_output;
  int *d_input, *d_kernel, *d_output;

  h_input = (int*)malloc(L*sizeof(int));
  h_kernel = (int*)malloc(KERNEL_SIZE*sizeof(int));
  h_output = (int*)malloc(L_OUT*sizeof(int));

  // init host arrays
  for (int i = 0; i < L; i++) {
    h_input[i] = i;
  }
  h_kernel[0] = 1; h_kernel[1] = 2;

  cudaMalloc(&d_input, L*sizeof(int));
  cudaMalloc(&d_kernel, KERNEL_SIZE*sizeof(int));
  cudaMalloc(&d_output, L_OUT*sizeof(int));

  cudaMemcpy(d_input, h_input, L*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_kernel, h_kernel, KERNEL_SIZE*sizeof(int), cudaMemcpyHostToDevice);

  int NUM_THREADS = 32;
  int NUM_BLOCKS = (L_OUT + NUM_THREADS - 1) / NUM_THREADS;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  conv1d<<<NUM_BLOCKS, NUM_THREADS>>>(d_input, d_kernel, d_output, L, KERNEL_SIZE);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  cudaMemcpy(h_output, d_output, L_OUT*sizeof(int), cudaMemcpyDeviceToHost);

  // printVector(h_input, L, "INPUT");
  // printVector(h_kernel, KERNEL_SIZE, "KERNEL");
  // printVector(h_output, L_OUT, "OUTPUT = INPUT * KERNEL");

  assert(validate_conv1d(h_input, h_kernel, h_output, L, KERNEL_SIZE));

  printf("Successful!\n");
  printf("Kernel execution time: %.3f ms\n", milliseconds);

  cudaFree(d_input); cudaFree(d_kernel); cudaFree(d_output);

}
