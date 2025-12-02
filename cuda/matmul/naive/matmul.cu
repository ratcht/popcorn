#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <stdio.h>

__global__ void matmul(int *a, int *b, int *c, int m_1, int n_1, int m_2, int n_2) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  int sum = 0;

  if ((row < m_2) && (col < n_1) && (n_1 == m_2)) {
    for (int k = 0; k < n_1; k++) {
      sum += a[row * n_1 + k] * b[k * n_2 + col];
    }
    c[row * n_2 + col] = sum;
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


int main() {
  int id = cudaGetDevice(&id);

  int m_1 = 2;
  int n_1 = 3;
  int m_2 = 3;
  int n_2 = 4;

  assert(n_1 == m_2);

  // define
  int *a, *b, *c;
  cudaMallocManaged(&a, sizeof(int) * m_1 * n_1);
  cudaMallocManaged(&b, sizeof(int) * m_2 * n_2);
  cudaMallocManaged(&c, sizeof(int) * n_1 * m_2);

  // fill in data
  for (int i = 0; i < m_1 * n_1; i++) {
    a[i] = i;
  }

  for (int i = 0; i < m_2 * n_2; i++) {
    b[i] = i * 2;
  }

  cudaMemPrefetchAsync(a, sizeof(int)*m_1*n_1, id);
  cudaMemPrefetchAsync(b, sizeof(int)*m_2*n_2, id);

  dim3 threads(16, 16);
  dim3 grid((n_2 + 15)/16, (m_1 + 15)/16);


  matmul<<<grid, threads>>>(a, b, c, m_1, n_1, m_2, n_2);

  cudaDeviceSynchronize();

  cudaMemPrefetchAsync(c, sizeof(int)*n_1*m_2, cudaCpuDeviceId);

  printMatrix(a, m_1, n_1, "A");
  printMatrix(b, m_2, n_2, "B");
  printMatrix(c, m_1, n_2, "C = A x B");
}
