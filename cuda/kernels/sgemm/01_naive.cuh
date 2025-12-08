#ifndef SGEMM_NAIVE_CUH
#define SGEMM_NAIVE_CUH

#include <cuda_runtime.h>

// (M x K) @ (K, N)

__global__ void sgemm_naive(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
  uint x = blockIdx.x*blockDim.x + threadIdx.x;
  uint y = blockIdx.y*blockDim.y + threadIdx.y;

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
