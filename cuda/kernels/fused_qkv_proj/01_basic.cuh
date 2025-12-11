#ifndef QKV_PROJ_BASIC_CUH
#define QKV_PROJ_BASIC_CUH

#include <cuda_runtime.h>

template <uint BM, uint BN, uint BK, uint TM, uint TN>
__global__ void qkv_proj_basic(float* x, float* w_qkv, float* b, float *qkv, int d_model, int d_k) {
  int N = 3 * d_k;
  int K = d_model;

  /* ====================================================
   * SGEMM IMPLEMENTATION (sgemm/06_vectorized.cuh)
   * ====================================================
   */

  int ty = threadIdx.x / (BN / TN); // row
  int tx = threadIdx.x % (BN / TN); // col

  int tyA = threadIdx.x / (BK / 4); // row A
  int txA = threadIdx.x % (BK / 4); // col A
  int tyB = threadIdx.x / (BN / 4); // row B
  int txB = threadIdx.x % (BN / 4); // col B

  __shared__ float As[BK][BM]; // transposed
  __shared__ float Bs[BK][BN];

  x += blockIdx.y * BM * K;                            // row=blockIdx.y, col=0
  w_qkv += blockIdx.x * BN;                                // row=0, col=blockIdx.x
  qkv += blockIdx.y * BM * N + blockIdx.x * BN;   // row=blockIdx.y, col=blockIdx.x

  // traverse row of A and down col of B
  float results[TM][TN] = {0.0};
  float regM[TM] = {0.0};
  float regN[TN] = {0.0};

  for (int block = 0; block < K; block+=BK) {
    // load shmem. each thread loads multiple elems
    // transpose A while loading

    float4 tmp = reinterpret_cast<float4*>(&x[tyA * K + txA * 4])[0];
    As[txA*4 + 0][tyA] = tmp.x;
    As[txA*4 + 1][tyA] = tmp.y;
    As[txA*4 + 2][tyA] = tmp.z;
    As[txA*4 + 3][tyA] = tmp.w;

    reinterpret_cast<float4*>(&Bs[tyB][txB * 4])[0] =
      reinterpret_cast<float4*>(&w_qkv[tyB * N + txB * 4])[0];

    __syncthreads();

    // advance to next chunk
    x += BK;
    w_qkv += BK * N;

    for (int i = 0; i < BK; i++) {
      for (int t = 0; t < TM; t++) {
        regM[t] = As[i][ty*TM + t];
      }
      for (int t = 0; t < TN; t++) {
        regN[t] = Bs[i][tx*TN + t];
      }

      // perform outer product on register cache
      // accum. into results
      for (int tM = 0; tM < TM; tM++) {
        for (int tN = 0; tN < TN; tN++) {
          results[tM][tN] += regM[tM] * regN[tN];
        }
      }

    }

    __syncthreads();
  }

  float regBias[TN];
  for (int tN = 0; tN < TN; tN += 4) {
    int col = blockIdx.x*BN + tx*TN + tN;
    float4 bias = reinterpret_cast<float4*>(&b[col])[0];
    regBias[tN + 0] = bias.x;
    regBias[tN + 1] = bias.y;
    regBias[tN + 2] = bias.z;
    regBias[tN + 3] = bias.w;
  }

  for (int tM = 0; tM < TM; tM++) {
    for (int tN = 0; tN < TN; tN+=4) {
      float4 out;
      out.x = results[tM][tN + 0] + regBias[tN + 0];
      out.y = results[tM][tN + 1] + regBias[tN + 1];
      out.z = results[tM][tN + 2] + regBias[tN + 2];
      out.w = results[tM][tN + 3] + regBias[tN + 3];

      // write back
      reinterpret_cast<float4*>(&qkv[(ty*TM + tM)*N + (tx*TN + tN)])[0] = out;
    }
  }
}

#endif
