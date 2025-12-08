#include "01_naive.cuh"
#include "02_coalesced.cuh"
#include "03_tiled.cuh"
#include "04_1d_blocktiling.cuh"
#include "05_2d_blocktiling.cuh"
#include "06_vectorized.cuh"
#include "07_reduce_bank_conflicts.cuh"

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

void run_sgemm_naive(int M, int N, int K, float alpha, float *A,
                    float *B, float beta, float *C) {
  dim3 blockDim(32, 32);
  dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
  sgemm_naive<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void run_sgemm_coalesced(int M, int N, int K, float alpha, float *A,
                    float *B, float beta, float *C) {
  dim3 blockDim(32 * 32);
  dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
  sgemm_coalesced<32><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void run_sgemm_tiled(int M, int N, int K, float alpha, float *A,
                    float *B, float beta, float *C) {
  dim3 blockDim(32 * 32);
  dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
  sgemm_tiled<32><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void run_sgemm_1d_blocktiling(int M, int N, int K, float alpha, float *A,
                    float *B, float beta, float *C) {
  const uint BM = 64;
  const uint BN = 64;
  const uint BK = 8;
  const uint TM = 8;

  dim3 blockDim((BM * BN) / TM);
  dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
  sgemm_1d_blocktiling<BM, BN, BK, TM><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void run_sgemm_2d_blocktiling(int M, int N, int K, float alpha, float *A,
                    float *B, float beta, float *C) {
  const uint BM = 128;
  const uint BN = 128;
  const uint BK = 8;
  const uint TM = 8;
  const uint TN = 8;

  dim3 blockDim((BM * BN) / (TM * TN));
  dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
  sgemm_2d_blocktiling<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void run_sgemm_vectorized(int M, int N, int K, float alpha, float *A,
                    float *B, float beta, float *C) {
  const uint BM = 128;
  const uint BN = 128;
  const uint BK = 8;
  const uint TM = 8;
  const uint TN = 8;

  dim3 blockDim((BM * BN) / (TM * TN));
  dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
  sgemm_vectorized<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void run_sgemm_reduce_bank_conflicts(int M, int N, int K, float alpha, float *A,
                    float *B, float beta, float *C) {
  const uint BM = 128;
  const uint BN = 128;
  const uint BK = 8;
  const uint TM = 8;
  const uint TN = 8;

  dim3 blockDim((BM * BN) / (TM * TN));
  dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
  sgemm_reduce_bank_conflicts<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}
