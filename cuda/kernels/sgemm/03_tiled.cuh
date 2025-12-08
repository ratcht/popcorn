#ifndef SGEMM_TILED_CUH
#define SGEMM_TILED_CUH

#include <cuda_runtime.h>


// (M x K) @ (K, N)

template <uint BLOCKSIZE>
__global__ void sgemm_tiled(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
  int tx = threadIdx.x / BLOCKSIZE; // row
  int ty = threadIdx.x % BLOCKSIZE; // col

  __shared__ float As[BLOCKSIZE][BLOCKSIZE];
  __shared__ float Bs[BLOCKSIZE][BLOCKSIZE];

  A += blockIdx.x * BLOCKSIZE * K;                            // row=blockIdx.x, col=0
  B += blockIdx.y * BLOCKSIZE;                                // row=0, col=blockIdx.y
  C += blockIdx.x * BLOCKSIZE * N + blockIdx.y * BLOCKSIZE;   // row=blockIdx.x, col=blockIdx.y

  // traverse row of A and down col of B
  float tmp = 0.f;
  for (int block = 0; block < K; block+=BLOCKSIZE) {
    // load shmem. each thread loads 1 elem
    As[tx][ty] = A[tx * K + ty];
    Bs[tx][ty] = B[tx * N + ty];

    __syncthreads();

    // advance to next chunk
    A += BLOCKSIZE;
    B += BLOCKSIZE * N;

    for (int i = 0; i < BLOCKSIZE; i++) {
      tmp += As[tx][i] * Bs[i][ty];
    }

    __syncthreads();
  }
  C[tx*N + ty] = alpha * tmp + beta * C[tx*N + ty];

}

#endif
