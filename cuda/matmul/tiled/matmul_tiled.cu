// !IMPORTANT : this doesnt work properly. the square does. there is an indexing issue here.

#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <stdio.h>

#define SHMEM_SIZE 16*16*4

__global__ void tiledMatmul(int *a, int *b, int *c, int m_1, int n_1, int m_2, int n_2, int tile_size) {
  assert(n_1 == m_2);

  int n = n_1;

  __shared__ int A[SHMEM_SIZE];
  __shared__ int B[SHMEM_SIZE];


  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;
  int by = blockIdx.y;

  int row = by * tile_size + ty;
  int col = bx * tile_size + tx;

  int sum = 0;

  for (int k = 0; k < (n + tile_size - 1) / tile_size; k++) {

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
    A[(ty * tile_size) + tx] = a[(row*n) + (k*tile_size) + (tx)];
    B[(ty * tile_size) + tx] = b[(col) + (k*tile_size*n_2) + (ty*n_2)];

    __syncthreads(); //barrier. ensure all threads loaded their data

    for (int i = 0; i < tile_size; i++) {
      sum += A[ty * tile_size + i] * B[i * tile_size + tx];
    }

    __syncthreads(); // ensure some threads dont progress and stom current shmem
  }
  c[(row * n_2) + col] = sum;

}

void printMatrix(const int* M, int rows, int cols, const char* name) {
  printf("%s =\n", name);
  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      printf("%4d ", M[r * cols + c]);
    }
    printf("\n");
  }
  printf("\n");
}




int main() {
  int id = cudaGetDevice(&id);

  int m_1 = 2;
  int n_1 = 3;
  int m_2 = 3;
  int n_2 = 4;

  assert(n_1 == m_2);

  // define
  int *h_a, *h_b, *h_c;
  int *d_a, *d_b, *d_c;

  h_a = (int*)malloc( sizeof(int) * m_1 * n_1);
  h_b = (int*)malloc(sizeof(int) * m_2 * n_2);
  h_c = (int*)malloc(sizeof(int) * m_1 * n_2);

  cudaMalloc(&d_a, sizeof(int) * m_1 * n_1);
  cudaMalloc(&d_b, sizeof(int) * m_2 * n_2);
  cudaMalloc(&d_c, sizeof(int) * m_1 * n_2);

  // fill in data
  for (int i = 0; i < m_1 * n_1; i++) {
    h_a[i] = i;
  }

  for (int i = 0; i < m_2 * n_2; i++) {
    h_b[i] = i * 2;
  }

  cudaMemcpy(d_a, h_a, sizeof(int) * m_1 * n_1, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, sizeof(int) * m_2 * n_2, cudaMemcpyHostToDevice);

  dim3 threads(16, 16);
  dim3 grid((n_2 + 15)/16, (m_1 + 15)/16);


  tiledMatmul<<<grid, threads>>>(d_a, d_b, d_c, m_1, n_1, m_2, n_2, 16);

  cudaMemcpy(h_c, d_c, sizeof(int) * m_1 * n_2, cudaMemcpyDeviceToHost);

  printMatrix(h_a, m_1, n_1, "A");
  printMatrix(h_b, m_2, n_2, "B");
  printMatrix(h_c, m_1, n_2, "C = A x B");

  free(h_a);
  free(h_b);
  free(h_c);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

}
