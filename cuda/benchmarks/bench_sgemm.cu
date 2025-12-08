#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "../kernels/sgemm/runner.cu"
#include "../utils/cuda_utils.h"

typedef void (*sgemm_kernel_func)(int M, int N, int K, float alpha, float *A,
                                   float *B, float beta, float *C);

void run_benchmark(const char* kernel_name, sgemm_kernel_func kernel_func,
                   int M, int N, int K, float alpha, float beta) {
  int size_a = M * K;
  int size_b = K * N;
  int size_c = M * N;
  int bytes_a = sizeof(float) * size_a;
  int bytes_b = sizeof(float) * size_b;
  int bytes_c = sizeof(float) * size_c;

  float *h_a, *h_b, *h_c, *h_c_ref;
  float *d_a, *d_b, *d_c, *d_c_ref;

  h_a = (float*)malloc(bytes_a);
  h_b = (float*)malloc(bytes_b);
  h_c = (float*)malloc(bytes_c);
  h_c_ref = (float*)malloc(bytes_c);

  srand(42);
  for (int i = 0; i < size_a; i++) h_a[i] = (float)(rand() % 10);
  for (int i = 0; i < size_b; i++) h_b[i] = (float)(rand() % 10);
  for (int i = 0; i < size_c; i++) h_c[i] = (float)(rand() % 10);

  cudaMalloc(&d_a, bytes_a);
  cudaMalloc(&d_b, bytes_b);
  cudaMalloc(&d_c, bytes_c);
  cudaMalloc(&d_c_ref, bytes_c);

  cudaMemcpy(d_a, h_a, bytes_a, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, bytes_b, cudaMemcpyHostToDevice);

  cublasHandle_t handle;
  cublasCreate(&handle);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // warmup
  for (int i = 0; i < 3; i++) {
    cudaMemcpy(d_c_ref, h_c, bytes_c, cudaMemcpyHostToDevice);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_b, N, d_a, K, &beta, d_c_ref, N);
    cudaMemcpy(d_c, h_c, bytes_c, cudaMemcpyHostToDevice);
    kernel_func(M, N, K, alpha, d_a, d_b, beta, d_c);
  }
  cudaDeviceSynchronize();

  // correctness check
  cudaMemcpy(d_c_ref, h_c, bytes_c, cudaMemcpyHostToDevice);
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_b, N, d_a, K, &beta, d_c_ref, N);
  cudaMemcpy(h_c_ref, d_c_ref, bytes_c, cudaMemcpyDeviceToHost);

  cudaMemcpy(d_c, h_c, bytes_c, cudaMemcpyHostToDevice);
  kernel_func(M, N, K, alpha, d_a, d_b, beta, d_c);
  cudaMemcpy(h_c, d_c, bytes_c, cudaMemcpyDeviceToHost);

  bool correct = compare_matrices(h_c_ref, h_c, size_c);

  // benchmark cublas
  int iters = 50;
  cudaMemcpy(d_c_ref, h_c, bytes_c, cudaMemcpyHostToDevice);
  cudaEventRecord(start);
  for (int i = 0; i < iters; i++) {
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_b, N, d_a, K, &beta, d_c_ref, N);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float cublas_ms = 0;
  cudaEventElapsedTime(&cublas_ms, start, stop);
  cublas_ms /= iters;

  // benchmark custom kernel
  cudaMemcpy(d_c, h_c, bytes_c, cudaMemcpyHostToDevice);
  cudaEventRecord(start);
  for (int i = 0; i < iters; i++) {
    kernel_func(M, N, K, alpha, d_a, d_b, beta, d_c);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float kernel_ms = 0;
  cudaEventElapsedTime(&kernel_ms, start, stop);
  kernel_ms /= iters;

  long flops = 2L * M * N * K;
  float cublas_gflops = flops * 1e-6f / cublas_ms;
  float kernel_gflops = flops * 1e-6f / kernel_ms;

  printf("M=%d, N=%d, K=%d, alpha=%.2f, beta=%.2f\n", M, N, K, alpha, beta);
  printf("cuBLAS:  %.3f ms, %.1f GFLOPS\n", cublas_ms, cublas_gflops);
  printf("%-7s: %.3f ms, %.1f GFLOPS\n", kernel_name, kernel_ms, kernel_gflops);
  printf("speedup: %.2fx\n", cublas_ms / kernel_ms);
  printf("correctness: %s\n", correct ? "PASS" : "FAIL");

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cublasDestroy(handle);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  cudaFree(d_c_ref);
  free(h_a);
  free(h_b);
  free(h_c);
  free(h_c_ref);
}

int main(int argc, char** argv) {
  if (argc != 7) {
    printf("Usage: %s <kernel_num> <M> <N> <K> <alpha> <beta>\n", argv[0]);
    printf("  0 - naive sgemm\n");
    printf("  1 - tiled sgemm\n");
    return 1;
  }

  int kernel_num = atoi(argv[1]);
  int M = atoi(argv[2]);
  int N = atoi(argv[3]);
  int K = atoi(argv[4]);
  float alpha = atof(argv[5]);
  float beta = atof(argv[6]);

  print_device_info();

  switch(kernel_num) {
    case 1:
      run_benchmark("naive", run_sgemm_naive, M, N, K, alpha, beta);
      break;
    case 2:
      run_benchmark("coalesced", run_sgemm_coalesced, M, N, K, alpha, beta);
      break;
    case 3:
      run_benchmark("tiled", run_sgemm_tiled, M, N, K, alpha, beta);
      break;
    case 4:
      run_benchmark("1d blocktiling", run_sgemm_1d_blocktiling, M, N, K, alpha, beta);
      break;
    case 5:
      run_benchmark("2d blocktiling", run_sgemm_2d_blocktiling, M, N, K, alpha, beta);
      break;
    case 6:
      run_benchmark("vectorized", run_sgemm_vectorized, M, N, K, alpha, beta);
      break;
    case 7:
      run_benchmark("reduce bank conflicts", run_sgemm_reduce_bank_conflicts, M, N, K, alpha, beta);
      break;
    default:
      printf("Invalid kernel number\n");
      return 1;
  }

  return 0;
}
