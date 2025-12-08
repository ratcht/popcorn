#ifndef SGEMM_2D_BLOCKTILING_CUH
#define SGEMM_2D_BLOCKTILING_CUH

#include <cuda_runtime.h>


// (M x K) @ (K, N)

template <uint BM, uint BN, uint BK, uint TM, uint TN>
__global__ void sgemm_2d_blocktiling(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
  int ty = threadIdx.x / (BN / TN); // row
  int tx = threadIdx.x % (BN / TN); // col

  int tyA = threadIdx.x / BK; // row A
  int txA = threadIdx.x % BK; // col A
  int tyB = threadIdx.x / BN; // row B
  int txB = threadIdx.x % BN; // col B

  uint totalResultsBlocktile = BM * BN;
  uint numThreadsBlocktile = totalResultsBlocktile / (TM * TN);

  uint strideA = numThreadsBlocktile / BK;
  uint strideB = numThreadsBlocktile / BN;

  __shared__ float As[BM][BK];
  __shared__ float Bs[BK][BN];

  A += blockIdx.y * BM * K;                            // row=blockIdx.x, col=0
  B += blockIdx.x * BN;                                // row=0, col=blockIdx.y
  C += blockIdx.y * BM * N + blockIdx.x * BN;   // row=blockIdx.x, col=blockIdx.y

  // traverse row of A and down col of B
  float results[TM][TN] = {0.0};
  float regM[TM] = {0.0};
  float regN[TN] = {0.0};

  for (int block = 0; block < K; block+=BK) {
    // load shmem. each thread loads multiple elems
    for (int offset = 0; offset < BM; offset += strideA) {
      As[tyA + offset][txA] = A[(tyA + offset) * K + txA];
    }
    for (int offset = 0; offset < BK; offset += strideB) {
      Bs[tyB + offset][txB] = B[(tyB + offset) * N + txB];
    }

    __syncthreads();

    // advance to next chunk
    A += BK;
    B += BK * N;

    for (int i = 0; i < BK; i++) {
      for (int t = 0; t < TM; t++) {
        regM[t] = As[ty*TM + t][i];
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

  for (int tM = 0; tM < TM; tM++) {
    for (int tN = 0; tN < TN; tN++) {
      C[(ty*TM + tM)*N + (tx*TN + tN)] = alpha * results[tM][tN] + beta * C[(ty*TM + tM)*N + (tx*TN + tN)];
    }
  }


}

#endif
