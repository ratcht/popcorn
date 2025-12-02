#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <stdio.h>

#define SHMEM_SIZE 16*16*sizeof(int) // 1 tile * 4 bytes per int

__global__ void tiledMatmul(int *a, int *b, int *c, int n, int tile_size) {
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
    A[(ty * tile_size) + tx] = a[row*n + (k * tile_size + tx)];
    B[(ty * tile_size) + tx] = b[(k * tile_size * n + ty * n) + col];

    __syncthreads(); //barrier. ensure all threads loaded their data

    for (int i = 0; i < tile_size; i++) {
      sum += A[ty * tile_size + i] * B[i * tile_size + tx];
    }

    __syncthreads(); // ensure some threads dont progress and stom current shmem
  }
  c[(row * n) + col] = sum;

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

void verify_result(int *a, int *b, int *c, int n) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      int tmp = 0;
      for (int k = 0; k < n; k++) {
        tmp += a[i * n + k] * b[k * n + j];
      }

      assert(tmp == c[i * n + j]);
    }
  }
}

int main() {
  int id = cudaGetDevice(&id);

  int n = 1024;

  // define
  int *h_a, *h_b, *h_c;
  int *d_a, *d_b, *d_c;

  h_a = (int*)malloc( sizeof(int) * n * n);
  h_b = (int*)malloc(sizeof(int) * n * n);
  h_c = (int*)malloc(sizeof(int) * n * n);

  cudaMalloc(&d_a, sizeof(int) * n * n);
  cudaMalloc(&d_b, sizeof(int) * n * n);
  cudaMalloc(&d_c, sizeof(int) * n * n);

  // fill in data
  for (int i = 0; i < n * n; i++) {
    h_a[i] = i;
    h_b[i] = i * 2;
  }

  cudaMemcpy(d_a, h_a, sizeof(int) * n * n, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, sizeof(int) * n * n, cudaMemcpyHostToDevice);

  dim3 threads(16, 16);
  dim3 grid((n + 15)/16, (n + 15)/16);


  tiledMatmul<<<grid, threads>>>(d_a, d_b, d_c, n, 16);

  cudaMemcpy(h_c, d_c, sizeof(int) * n * n, cudaMemcpyDeviceToHost);

  // verify_result(h_a, h_b, h_c, n);

  // printf("successful\n");

  free(h_a);
  free(h_b);
  free(h_c);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

}
