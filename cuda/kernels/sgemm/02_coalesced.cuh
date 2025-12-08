#ifndef SGEMM_COALESCED_CUH
#define SGEMM_COALESCED_CUH

#include <cuda_runtime.h>


// (M x K) @ (K, N)

template <uint BLOCKSIZE>
__global__ void sgemm_coalesced(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
  int x = blockIdx.x*BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  int y = blockIdx.y*BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

  // traverse row of A and down col of B
  if (x < M && y < N) {
    float tmp = 0.f;
    for (int i = 0; i < K; i++) {
      tmp += A[x*K + i] * B[i*N + y];
    }
    C[x*N + y] = alpha * tmp + beta * C[x*N + y];
  }
}

#endif
