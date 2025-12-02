#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <cuda_runtime_api.h>
#include <curand.h>
#include <driver_types.h>
#include <stdio.h>
#include <math.h>



#define KERNEL_SIZE 7
#define PADDING 3
#define INPUT_SIZE (1 << 16)
#define PADDED_INPUT_SIZE (INPUT_SIZE + 2 * PADDING + KERNEL_SIZE - 1)
#define OUTPUT_SIZE (INPUT_SIZE + 2 * PADDING - KERNEL_SIZE + 1)

__constant__ int kernel[KERNEL_SIZE];

__global__ void conv1d(int *input, int *output) {
  // input is pre-padded

  extern __shared__ int shmem[];

  int window_start = blockIdx.x*blockDim.x + threadIdx.x;  // g_tid
  int local_idx = threadIdx.x; // l_tid

  shmem[local_idx] = input[window_start];


  if (local_idx < KERNEL_SIZE - 1) {
    shmem[local_idx + blockDim.x] = input[window_start + blockDim.x];

  }

  __syncthreads();

  if (window_start >= OUTPUT_SIZE) {
    return;
  }

  int sum = 0;

  #pragma unroll
  for (int i = 0; i < KERNEL_SIZE; i++) {
    sum += shmem[local_idx+i] * kernel[i];
  }

  output[window_start] = sum;
}

void printVector(const int* v, int n, const char* name) {
  printf("%s = [ ", name);
  for (int i = 0; i < n; i++) {
    printf("%d ", v[i]);
  }
  printf("]\n");
}

bool validate_conv1d(int *input, int *kernel, int *gpu_output, int input_size, int kernel_size, int padding) {
  int output_size = input_size + (2*padding) - kernel_size + 1;

  for (int i = 0; i < output_size; i++) {
    int expected = 0;
    for (int j = 0; j < kernel_size; j++) {
      if (i + j < padding || padding + input_size <= i + j) {
        expected += 0;
      } else {
        expected += input[i+j-padding] * kernel[j];
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

  // pre-padded input
  int *h_padded_input = (int*)malloc(PADDED_INPUT_SIZE * sizeof(int));
  memset(h_padded_input, 0, PADDED_INPUT_SIZE * sizeof(int));

  for (int i = 0; i < INPUT_SIZE; i++) {
    h_padded_input[PADDING + i] = h_input[i];
  }

  cudaMalloc(&d_input, PADDED_INPUT_SIZE*sizeof(int));
  cudaMalloc(&d_output, OUTPUT_SIZE*sizeof(int));

  cudaMemcpy(d_input, h_padded_input, PADDED_INPUT_SIZE*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(kernel, h_kernel, KERNEL_SIZE*sizeof(int));


  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  /*
   * ======= KERNEL LAUNCH =======
   */

  int num_threads = 128;
  int num_blocks = (OUTPUT_SIZE + num_threads - 1) / num_threads;
  int smem_size = (num_threads + KERNEL_SIZE - 1) * sizeof(int);

  conv1d<<<num_blocks, num_threads, smem_size>>>(d_input, d_output);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
  }

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  cudaMemcpy(h_output, d_output, OUTPUT_SIZE*sizeof(int), cudaMemcpyDeviceToHost);

  // printVector(h_input, L, "INPUT");
  // printVector(h_kernel, KERNEL_SIZE, "KERNEL");
  // printVector(h_output, L_OUT, "OUTPUT = INPUT * KERNEL");

  assert(validate_conv1d(h_input, h_kernel, h_output, INPUT_SIZE, KERNEL_SIZE, PADDING));

  printf("Successful!\n");
  printf("Kernel execution time: %.3f ms\n", milliseconds);

  // Cleanup events
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cudaFree(d_input); cudaFree(d_output);

}
