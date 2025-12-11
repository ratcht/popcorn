#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include "../kernels/fused_qkv_proj/runner.cu"
#include "../utils/cuda_utils.h"

void run_unfused(cublasHandle_t handle, float* x, float* w, float* b, float* ones,
                 float* qkv, int seq_len, int d_model, int d_k) {
  int M = seq_len;
  int N = 3 * d_k;
  int K = d_model;
  float alpha = 1.0f;
  float beta = 0.0f;

  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
              &alpha, w, N, x, K, &beta, qkv, N);

  cublasSger(handle, N, M, &alpha, b, 1, ones, 1, qkv, N);
}

void run_cublaslt_fused(cublasLtHandle_t handle, cublasLtMatmulDesc_t matmul_desc,
                        cublasLtMatrixLayout_t w_desc, cublasLtMatrixLayout_t x_desc,
                        cublasLtMatrixLayout_t qkv_desc, cublasLtMatmulAlgo_t* algo,
                        float* x, float* w, float* qkv,
                        void* workspace, size_t workspace_size) {
  float alpha = 1.0f;
  float beta = 0.0f;

  cublasLtMatmul(handle, matmul_desc, &alpha, w, w_desc, x, x_desc,
                 &beta, qkv, qkv_desc, qkv, qkv_desc,
                 algo, workspace, workspace_size, 0);
}

void run_benchmark(int seq_len, int d_model, int d_k) {
  int M = seq_len;
  int K = d_model;
  int N = 3 * d_k;

  int size_x = M * K;
  int size_w = K * N;
  int size_b = N;
  int size_qkv = M * N;

  size_t bytes_x = sizeof(float) * size_x;
  size_t bytes_w = sizeof(float) * size_w;
  size_t bytes_b = sizeof(float) * size_b;
  size_t bytes_qkv = sizeof(float) * size_qkv;

  float *h_x, *h_w, *h_b, *h_qkv_ref, *h_qkv_lt, *h_qkv_custom;
  float *d_x, *d_w, *d_b, *d_ones, *d_qkv_ref, *d_qkv_lt, *d_qkv_custom;

  h_x = (float*)malloc(bytes_x);
  h_w = (float*)malloc(bytes_w);
  h_b = (float*)malloc(bytes_b);
  h_qkv_ref = (float*)malloc(bytes_qkv);
  h_qkv_lt = (float*)malloc(bytes_qkv);
  h_qkv_custom = (float*)malloc(bytes_qkv);

  srand(42);
  for (int i = 0; i < size_x; i++) h_x[i] = (float)(rand() % 10);
  for (int i = 0; i < size_w; i++) h_w[i] = (float)(rand() % 10);
  for (int i = 0; i < size_b; i++) h_b[i] = (float)(rand() % 10);

  float* h_ones = (float*)malloc(sizeof(float) * M);
  for (int i = 0; i < M; i++) h_ones[i] = 1.0f;

  cudaMalloc(&d_x, bytes_x);
  cudaMalloc(&d_w, bytes_w);
  cudaMalloc(&d_b, bytes_b);
  cudaMalloc(&d_ones, sizeof(float) * M);
  cudaMalloc(&d_qkv_ref, bytes_qkv);
  cudaMalloc(&d_qkv_lt, bytes_qkv);
  cudaMalloc(&d_qkv_custom, bytes_qkv);

  cudaMemcpy(d_x, h_x, bytes_x, cudaMemcpyHostToDevice);
  cudaMemcpy(d_w, h_w, bytes_w, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, bytes_b, cudaMemcpyHostToDevice);
  cudaMemcpy(d_ones, h_ones, sizeof(float) * M, cudaMemcpyHostToDevice);

  cublasHandle_t handle;
  cublasLtHandle_t lt_handle;
  cublasCreate(&handle);
  cublasLtCreate(&lt_handle);

  size_t workspace_size = 4 * 1024 * 1024;
  void* workspace;
  cudaMalloc(&workspace, workspace_size);

  // cublaslt setup
  cublasLtMatmulDesc_t matmul_desc;
  cublasLtMatrixLayout_t x_desc, w_desc, qkv_desc;
  cublasLtMatmulPreference_t preference;

  cublasLtMatmulDescCreate(&matmul_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);

  cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
  cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue));
  cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &d_b, sizeof(d_b));

  cublasLtMatrixLayoutCreate(&w_desc, CUDA_R_32F, N, K, N);
  cublasLtMatrixLayoutCreate(&x_desc, CUDA_R_32F, K, M, K);
  cublasLtMatrixLayoutCreate(&qkv_desc, CUDA_R_32F, N, M, N);

  cublasLtMatmulPreferenceCreate(&preference);
  cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                       &workspace_size, sizeof(workspace_size));

  cublasLtMatmulHeuristicResult_t heuristic;
  int returned_results = 0;
  cublasLtMatmulAlgoGetHeuristic(lt_handle, matmul_desc, w_desc, x_desc, qkv_desc, qkv_desc,
                                 preference, 1, &heuristic, &returned_results);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // warmup
  for (int i = 0; i < 3; i++) {
    run_unfused(handle, d_x, d_w, d_b, d_ones, d_qkv_ref, seq_len, d_model, d_k);
    run_cublaslt_fused(lt_handle, matmul_desc, w_desc, x_desc, qkv_desc, &heuristic.algo,
                       d_x, d_w, d_qkv_lt, workspace, workspace_size);
    run_qkv_proj_basic(d_x, d_w, d_b, d_qkv_custom, seq_len, d_model, d_k);
  }
  cudaDeviceSynchronize();

  // correctness
  run_unfused(handle, d_x, d_w, d_b, d_ones, d_qkv_ref, seq_len, d_model, d_k);
  run_cublaslt_fused(lt_handle, matmul_desc, w_desc, x_desc, qkv_desc, &heuristic.algo,
                     d_x, d_w, d_qkv_lt, workspace, workspace_size);
  run_qkv_proj_basic(d_x, d_w, d_b, d_qkv_custom, seq_len, d_model, d_k);

  cudaMemcpy(h_qkv_ref, d_qkv_ref, bytes_qkv, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_qkv_lt, d_qkv_lt, bytes_qkv, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_qkv_custom, d_qkv_custom, bytes_qkv, cudaMemcpyDeviceToHost);

  bool lt_correct = compare_matrices(h_qkv_ref, h_qkv_lt, size_qkv);
  bool custom_correct = compare_matrices(h_qkv_ref, h_qkv_custom, size_qkv);

  // benchmark
  int iters = 50;

  cudaEventRecord(start);
  for (int i = 0; i < iters; i++) {
    run_unfused(handle, d_x, d_w, d_b, d_ones, d_qkv_ref, seq_len, d_model, d_k);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float unfused_ms = 0;
  cudaEventElapsedTime(&unfused_ms, start, stop);
  unfused_ms /= iters;

  cudaEventRecord(start);
  for (int i = 0; i < iters; i++) {
    run_cublaslt_fused(lt_handle, matmul_desc, w_desc, x_desc, qkv_desc, &heuristic.algo,
                       d_x, d_w, d_qkv_lt, workspace, workspace_size);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float cublaslt_ms = 0;
  cudaEventElapsedTime(&cublaslt_ms, start, stop);
  cublaslt_ms /= iters;

  cudaEventRecord(start);
  for (int i = 0; i < iters; i++) {
    run_qkv_proj_basic(d_x, d_w, d_b, d_qkv_custom, seq_len, d_model, d_k);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float custom_ms = 0;
  cudaEventElapsedTime(&custom_ms, start, stop);
  custom_ms /= iters;

  long flops = 2L * M * N * K;
  float unfused_gflops = flops * 1e-6f / unfused_ms;
  float cublaslt_gflops = flops * 1e-6f / cublaslt_ms;
  float custom_gflops = flops * 1e-6f / custom_ms;

  printf("seq_len=%d, d_model=%d, d_k=%d (N=%d)\n", seq_len, d_model, d_k, N);
  printf("unfused (sgemm+sger): %.3f ms, %.1f GFLOPS\n", unfused_ms, unfused_gflops);
  printf("cublaslt fused:       %.3f ms, %.1f GFLOPS [%s]\n", cublaslt_ms, cublaslt_gflops, lt_correct ? "PASS" : "FAIL");
  printf("custom fused:         %.3f ms, %.1f GFLOPS [%s]\n", custom_ms, custom_gflops, custom_correct ? "PASS" : "FAIL");

  cublasLtMatmulPreferenceDestroy(preference);
  cublasLtMatrixLayoutDestroy(x_desc);
  cublasLtMatrixLayoutDestroy(w_desc);
  cublasLtMatrixLayoutDestroy(qkv_desc);
  cublasLtMatmulDescDestroy(matmul_desc);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cublasDestroy(handle);
  cublasLtDestroy(lt_handle);
  cudaFree(d_x);
  cudaFree(d_w);
  cudaFree(d_b);
  cudaFree(d_ones);
  cudaFree(d_qkv_ref);
  cudaFree(d_qkv_lt);
  cudaFree(d_qkv_custom);
  cudaFree(workspace);
  free(h_x);
  free(h_w);
  free(h_b);
  free(h_ones);
  free(h_qkv_ref);
  free(h_qkv_lt);
  free(h_qkv_custom);
}

int main(int argc, char** argv) {
  if (argc != 4) {
    printf("Usage: %s <seq_len> <d_model> <d_k>\n", argv[0]);
    return 1;
  }

  int seq_len = atoi(argv[1]);
  int d_model = atoi(argv[2]);
  int d_k = atoi(argv[3]);

  print_device_info();
  run_benchmark(seq_len, d_model, d_k);

  return 0;
}