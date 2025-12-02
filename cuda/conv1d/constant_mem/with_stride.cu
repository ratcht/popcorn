#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <cuda_runtime_api.h>
#include <curand.h>
#include <driver_types.h>
#include <stdio.h>


#define KERNEL_SIZE 2
#define PADDING 4
#define STRIDE 4
#define INPUT_SIZE 14
#define OUTPUT_SIZE ((INPUT_SIZE + 2 * PADDING - KERNEL_SIZE) / STRIDE + 1)


__constant__ int kernel[KERNEL_SIZE];

__global__ void conv1d(int *input, int *output, int input_size, int kernel_size, int padding, int stride) {
  int output_idx = blockIdx.x*blockDim.x + threadIdx.x;  // g_tid
  int input_idx = output_idx * stride - padding;

  if (output_idx >= OUTPUT_SIZE) {
    return;
  }

  int i_start = 0;
  if (input_idx < 0) i_start = -input_idx;

  int i_end = kernel_size;
  int rem = input_size - input_idx;
  if (rem < i_end) i_end = rem;

  int sum = 0;
  #pragma unroll
  for (int i = i_start; i < i_end; i++) {
    sum += input[input_idx + i] * kernel[i];
  }

  output[output_idx] = sum;
}


void printVector(const int* v, int n, const char* name) {
  printf("%s = [ ", name);
  for (int i = 0; i < n; i++) {
    printf("%d ", v[i]);
  }
  printf("]\n");
}

bool validate_conv1d(int *input, int *kernel, int *gpu_output, int input_size, int kernel_size, int padding, int stride) {
  int output_size = (input_size + (2*padding) - kernel_size)/stride + 1;

  for (int i = 0; i < output_size; i++) {
    int expected = 0;
    for (int j = 0; j < kernel_size; j++) {
      int idx = i*stride + j - padding;
      if (idx < 0 || idx >= input_size) {
        expected += 0;
      } else {
        expected += input[idx] * kernel[j];
      }

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

  int *h_input, *h_kernel, *h_output;
  int *d_input, *d_output;

  h_input = (int*)malloc(INPUT_SIZE*sizeof(int));
  h_kernel = (int*)malloc(KERNEL_SIZE*sizeof(int));
  h_output = (int*)malloc(OUTPUT_SIZE*sizeof(int));

  // init host arrays
  for (int i = 0; i < INPUT_SIZE; i++) {
    h_input[i] = i;
  }
  h_kernel[0] = 1; h_kernel[1] = 2;

  cudaMalloc(&d_input, INPUT_SIZE*sizeof(int));
  cudaMalloc(&d_output, OUTPUT_SIZE*sizeof(int));

  cudaMemcpy(d_input, h_input, INPUT_SIZE*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(kernel, h_kernel, KERNEL_SIZE*sizeof(int));


  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  /*
   * ======= KERNEL LAUNCH =======
   */

  int num_threads = 4;
  int num_blocks = (OUTPUT_SIZE + num_threads - 1) / num_threads;
  int smem_size = (num_threads + KERNEL_SIZE - 1) * sizeof(int);

  conv1d<<<num_blocks, num_threads, smem_size>>>(d_input, d_output, INPUT_SIZE, KERNEL_SIZE, PADDING, STRIDE);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  cudaMemcpy(h_output, d_output, OUTPUT_SIZE*sizeof(int), cudaMemcpyDeviceToHost);

  printVector(h_input, INPUT_SIZE, "INPUT");
  printVector(h_kernel, KERNEL_SIZE, "KERNEL");
  printVector(h_output, OUTPUT_SIZE, "OUTPUT = INPUT * KERNEL");

  assert(validate_conv1d(h_input, h_kernel, h_output, INPUT_SIZE, KERNEL_SIZE, PADDING, STRIDE));

  printf("Successful!\n");
  printf("Kernel execution time: %.3f ms\n", milliseconds);

  // Cleanup events
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cudaFree(d_input); cudaFree(d_output);

}
