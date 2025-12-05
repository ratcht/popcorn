#ifndef MATMUL_TILED_CUH
#define MATMUL_TILED_CUH

#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void matmulTiled(int *a, int *b, int *c, int n) {
  __shared__ int A[TILE_SIZE * TILE_SIZE];
  __shared__ int B[TILE_SIZE * TILE_SIZE];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;
  int by = blockIdx.y;

  int row = by * TILE_SIZE + ty;
  int col = bx * TILE_SIZE + tx;

  int sum = 0;

  for (int k = 0; k < (n + TILE_SIZE - 1) / TILE_SIZE; k++) {

    // load shmem
    /*
      for A (row invariant):
      row*n : global row for this thread.
      k*tile_size : col tile index
      tx : col within the tile

      for B (col invariant):
      col : global column (remember its col invariant)
      k*tile_size*n : row tile index
      ty*n : row within the set
     */
    // for B (col invariant):  : global row for this thread. i*tile_size : new set of cols. tx: col within the tile
    A[(ty * TILE_SIZE) + tx] = a[row * n + (k * TILE_SIZE + tx)];
    B[(ty * TILE_SIZE) + tx] = b[(k * TILE_SIZE * n + ty * n) + col];

    __syncthreads(); // barrier. ensure all threads loaded their data

    for (int i = 0; i < TILE_SIZE; i++) {
      sum += A[ty * TILE_SIZE + i] * B[i * TILE_SIZE + tx];
    }

    __syncthreads(); // ensure some threads dont progress and stomp current shmem
  }
  c[(row * n) + col] = sum;
}

#endif