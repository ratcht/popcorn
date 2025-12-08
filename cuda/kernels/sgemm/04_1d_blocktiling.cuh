#ifndef SGEMM_1D_BLOCKTILING_CUH
#define SGEMM_1D_BLOCKTILING_CUH

#include <cuda_runtime.h>


// (M x K) @ (K, N)

template <uint BM, uint BN, uint BK, uint TM>
__global__ void sgemm_1d_blocktiling(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
  int ty = threadIdx.x / BN; // row
  int tx = threadIdx.x % BN; // col

  int tyA = threadIdx.x / BK; // row A
  int txA = threadIdx.x % BK; // col A

  __shared__ float As[BM][BK];
  __shared__ float Bs[BK][BN];

  A += blockIdx.y * BM * K;                            // row=blockIdx.x, col=0
  B += blockIdx.x * BN;                                // row=0, col=blockIdx.y
  C += blockIdx.y * BM * N + blockIdx.x * BN;   // row=blockIdx.x, col=blockIdx.y

  // traverse row of A and down col of B
  float results[TM] = {0.0};
  for (int block = 0; block < K; block+=BK) {
    // load shmem. each thread loads 1 elem
    As[tyA][txA] = A[tyA * K + txA];
    Bs[ty][tx] = B[ty * N + tx];

    __syncthreads();

    // advance to next chunk
    A += BK;
    B += BK * N;

    for (int i = 0; i < BK; i++) {
      float tmpB = Bs[i][tx]; // cache

      for (int t = 0; t < TM; t++) {
        results[t] += As[ty*TM + t][i] * tmpB;
      }
    }

    __syncthreads();
  }

  for (int t = 0; t < TM; t++) {
    C[(ty*TM + t)*N + tx] = alpha * results[t] + beta * C[(ty*TM + t)*N + tx];
  }


}

#endif
