#ifndef SGEMM_REDUCE_BANK_CONFLICTS_CUH
#define SGEMM_REDUCE_BANK_CONFLICTS_CUH

#include <cuda_runtime.h>


// (M x K) @ (K, N)

template <uint BM, uint BN, uint BK, uint TM, uint TN>
__global__ void sgemm_reduce_bank_conflicts(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
  int ty = threadIdx.x / (BN / TN); // row
  int tx = threadIdx.x % (BN / TN); // col

  int tyA = threadIdx.x / (BK / 4); // row A
  int txA = threadIdx.x % (BK / 4); // col A
  int tyB = threadIdx.x / (BN / 4); // row B
  int txB = threadIdx.x % (BN / 4); // col B

  __shared__ float As[BK][BM]; // transposed
  __shared__ float Bs[BK * 8][16];

  A += blockIdx.y * BM * K;                            // row=blockIdx.x, col=0
  B += blockIdx.x * BN;                                // row=0, col=blockIdx.y
  C += blockIdx.y * BM * N + blockIdx.x * BN;   // row=blockIdx.x, col=blockIdx.y

  // traverse row of A and down col of B
  float results[TM][TN] = {0.0};
  float regM[TM] = {0.0};
  float regN[TN] = {0.0};

  for (int block = 0; block < K; block+=BK) {
    // load shmem. each thread loads multiple elems
    // transpose A while loading

    float4 tmp = reinterpret_cast<float4*>(&A[tyA * K + txA * 4])[0];
    As[txA*4 + 0][tyA] = tmp.x;
    As[txA*4 + 1][tyA] = tmp.y;
    As[txA*4 + 2][tyA] = tmp.z;
    As[txA*4 + 3][tyA] = tmp.w;

    tmp = reinterpret_cast<float4*>(&B[tyB * N + txB * 4])[0];
    Bs[(txB % 2)*4 + tyB*8 + 0][txB/2] = tmp.x;
    Bs[(txB % 2)*4 + tyB*8 + 1][txB/2] = tmp.y;
    Bs[(txB % 2)*4 + tyB*8 + 2][txB/2] = tmp.z;
    Bs[(txB % 2)*4 + tyB*8 + 3][txB/2] = tmp.w;

    __syncthreads();

    // advance to next chunk
    A += BK;
    B += BK * N;

    for (int i = 0; i < BK; i++) {
      for (int t = 0; t < TM; t++) {
        regM[t] = As[i][ty*TM + t];
      }
      for (int t = 0; t < TN; t++) {
        regN[t] = Bs[i*8 + t][tx];
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

  for (int tM = 0; tM < TM; tM++) {
    for (int tN = 0; tN < TN; tN+=4) {
      float4 tmp = reinterpret_cast<float4*>(&C[(ty*TM + tM)*N + (tx*TN + tN)])[0];

      tmp.x = alpha * results[tM][tN] + beta * tmp.x;
      tmp.y = alpha * results[tM][tN + 1] + beta * tmp.y;
      tmp.z = alpha * results[tM][tN + 2] + beta * tmp.z;
      tmp.w = alpha * results[tM][tN + 3] + beta * tmp.w;

      // write back
      reinterpret_cast<float4*>(&C[(ty*TM + tM)*N + (tx*TN + tN)])[0] = tmp;
    }
  }


}

#endif
