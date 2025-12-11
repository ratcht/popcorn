#include "01_basic.cuh"

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

void run_qkv_proj_basic(float* x, float* w_qkv, float* b, float* qkv,
                        int seq_len, int d_model, int d_k) {
  const uint BM = 128;
  const uint BN = 128;
  const uint BK = 8;
  const uint TM = 8;
  const uint TN = 8;

  int M = seq_len;
  int N = 3 * d_k;

  dim3 blockDim((BM * BN) / (TM * TN));
  dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
  qkv_proj_basic<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(x, w_qkv, b, qkv, d_model, d_k);
}