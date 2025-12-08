#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "../kernels/matmul/runner.cu"
#include "../utils/cuda_utils.h"

void print_matmul_results(int* h_a, int* h_b, int* h_c, int n, float ms) {
  if (n <= 8) {
    print_matrix(h_a, n, n, "A");
    print_matrix(h_b, n, n, "B");
    print_matrix(h_c, n, n, "C = A x B");
  }
  printf("Matrix size: %dx%d\n", n, n);
  printf("Kernel execution time: %.3f ms\n", ms);
}

void run_naive_square_benchmark(int n) {
  int size = n * n;
  int bytes = sizeof(int) * size;
  int *h_a, *h_b, *h_c;
  int *d_a, *d_b, *d_c;

  h_a = (int*)malloc(bytes);
  h_b = (int*)malloc(bytes);
  h_c = (int*)malloc(bytes);

  for (int i = 0; i < size; i++) {
    h_a[i] = i % 10;
    h_b[i] = (i * 2) % 10;
  }

  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, bytes);

  cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  run_matmul_naive_square(d_a, d_b, d_c, n);
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

  print_matmul_results(h_a, h_b, h_c, n, milliseconds);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  free(h_a);
  free(h_b);
  free(h_c);
}

void run_coalesced_benchmark(int n) {
  int size = n * n;
  int bytes = sizeof(int) * size;
  int *h_a, *h_a_t, *h_b, *h_c;
  int *d_a_t, *d_b, *d_c;

  h_a = (int*)malloc(bytes);
  h_a_t = (int*)malloc(bytes);
  h_b = (int*)malloc(bytes);
  h_c = (int*)malloc(bytes);

  for (int i = 0; i < size; i++) {
    h_a[i] = i % 10;
    h_b[i] = (i * 2) % 10;
  }

  transpose(h_a, h_a_t, n);

  cudaMalloc(&d_a_t, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, bytes);

  cudaMemcpy(d_a_t, h_a_t, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  run_matmul_coalesced(d_a_t, d_b, d_c, n);
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

  print_matmul_results(h_a, h_b, h_c, n, milliseconds);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_a_t);
  cudaFree(d_b);
  cudaFree(d_c);
  free(h_a);
  free(h_a_t);
  free(h_b);
  free(h_c);
}

void run_tiled_benchmark(int n) {
  int size = n * n;
  int bytes = sizeof(int) * size;
  int *h_a, *h_b, *h_c;
  int *d_a, *d_b, *d_c;

  h_a = (int*)malloc(bytes);
  h_b = (int*)malloc(bytes);
  h_c = (int*)malloc(bytes);

  for (int i = 0; i < size; i++) {
    h_a[i] = i % 10;
    h_b[i] = (i * 2) % 10;
  }

  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, bytes);

  cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  run_matmul_tiled(d_a, d_b, d_c, n);
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

  print_matmul_results(h_a, h_b, h_c, n, milliseconds);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  free(h_a);
  free(h_b);
  free(h_c);
}

int main(int argc, char** argv) {
  if (argc != 3) {
    printf("Usage: %s <kernel_num> <matrix_size>\n", argv[0]);
    printf("  0 - naive square\n");
    printf("  1 - coalesced\n");
    printf("  2 - tiled\n");
    return 1;
  }

  int kernel_num = atoi(argv[1]);
  int n = atoi(argv[2]);
  print_device_info();

  switch(kernel_num) {
    case 0:
      run_naive_square_benchmark(n);
      break;
    case 1:
      run_coalesced_benchmark(n);
      break;
    case 2:
      run_tiled_benchmark(n);
      break;
    default:
      printf("Invalid kernel number\n");
      return 1;
  }

  return 0;
}
