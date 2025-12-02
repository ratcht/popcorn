#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <stdio.h>

__global__ void matmul(int *a, int *b, int *c, int n) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  int sum = 0;

  if ((row < n) && (col < n)) {
    for (int k = 0; k < n; k++) {
      sum += a[row * n + k] * b[k * n + col];
    }
    c[row * n + col] = sum;
  }
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

  int n = 4096;

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


  matmul<<<grid, threads>>>(d_a, d_b, d_c, n);

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
