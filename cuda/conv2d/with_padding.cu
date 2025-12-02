#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <cuda_runtime_api.h>
#include <curand.h>
#include <driver_types.h>
#include <stdio.h>


#define KERNEL_WIDTH 2
#define KERNEL_HEIGHT 2

#define PADDING 1

#define INPUT_WIDTH (1 << 3)
#define INPUT_HEIGHT (1 << 3)

#define PADDED_INPUT_WIDTH (INPUT_WIDTH + 2 * PADDING + KERNEL_WIDTH - 1)
#define PADDED_INPUT_HEIGHT (INPUT_HEIGHT + 2 * PADDING + KERNEL_HEIGHT - 1)

#define OUTPUT_WIDTH (INPUT_WIDTH + 2 * PADDING - KERNEL_WIDTH + 1)
#define OUTPUT_HEIGHT (INPUT_HEIGHT + 2 * PADDING - KERNEL_HEIGHT + 1)

__constant__ int kernel[KERNEL_WIDTH * KERNEL_HEIGHT];

__global__ void conv2d(int *input, int *output) {
  // input is pre-padded

  extern __shared__ int shmem[];

  // start values
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  int tile_width = blockDim.x + KERNEL_WIDTH - 1;

  // blockDim.x = number of columns = width
  // blockDim.y = number of rows = height
  // row*width + col;

  // load shmem at thread location
  shmem[threadIdx.y*tile_width + threadIdx.x] = input[row*PADDED_INPUT_WIDTH + col];

  // load shmem at x + blockDim.x
  if (threadIdx.x < KERNEL_WIDTH- 1) {
    shmem[threadIdx.y*tile_width + threadIdx.x + blockDim.x] = input[row*PADDED_INPUT_WIDTH + col + blockDim.x];
  }

  // load shmem at y + blockDim.y
  if (threadIdx.y < KERNEL_HEIGHT- 1) {
    shmem[(threadIdx.y + blockDim.y)*tile_width + threadIdx.x ] = input[(row + blockDim.y)*PADDED_INPUT_WIDTH + col];
  }

  // load corner
  if (threadIdx.x < KERNEL_WIDTH - 1 && threadIdx.y < KERNEL_HEIGHT- 1) {
    shmem[(threadIdx.y + blockDim.y)*tile_width + threadIdx.x + blockDim.x] = input[(row + blockDim.y)*PADDED_INPUT_WIDTH + col + blockDim.x];
  }


  __syncthreads();

  if (col >= OUTPUT_WIDTH) {
    return;
  }
  if (row >= OUTPUT_HEIGHT) {
    return;
  }

  int sum = 0;

  #pragma unroll
  for (int i = 0; i < KERNEL_HEIGHT; i++) {
    #pragma unroll
    for (int j = 0; j < KERNEL_WIDTH; j++) {
      sum += shmem[(threadIdx.y+i)*tile_width + threadIdx.x + j] * kernel[i*KERNEL_WIDTH + j];
    }
  }

  output[row*OUTPUT_WIDTH + col] = sum;
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

void save_tensor(const char* filename, int* data, int size) {
  FILE *f = fopen(filename, "wb");
  if (f == NULL) {
    printf("Error opening file %s\n", filename);
    return;
  }
  fwrite(data, sizeof(int), size, f);
  fclose(f);
}

int main() {
  int device;
  cudaGetDevice(&device);

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);

  printf("Running on GPU %d: %s\n", device, prop.name);

  int *h_input, *h_kernel, *h_output;
  int *d_input, *d_output;

  h_input = (int*)malloc(INPUT_WIDTH*INPUT_HEIGHT*sizeof(int));
  h_kernel = (int*)malloc(KERNEL_WIDTH*KERNEL_HEIGHT*sizeof(int));
  h_output = (int*)malloc(OUTPUT_WIDTH*OUTPUT_HEIGHT*sizeof(int));

  // init host arrays
  for (int i = 0; i < INPUT_WIDTH*INPUT_HEIGHT; i++) {
    h_input[i] = i;
  }
  for (int i = 0; i < KERNEL_WIDTH*KERNEL_HEIGHT; i++) {
    h_kernel[i] = i+1;
  }

  // pre-padded input
  int *h_padded_input = (int*)malloc(PADDED_INPUT_WIDTH*PADDED_INPUT_HEIGHT*sizeof(int));
  memset(h_padded_input, 0, PADDED_INPUT_WIDTH*PADDED_INPUT_HEIGHT * sizeof(int));

  for (int i = 0; i < INPUT_HEIGHT; i++) {
    for (int j = 0; j < INPUT_WIDTH; j++) {
      h_padded_input[(PADDING + i)*PADDED_INPUT_WIDTH + PADDING + j] = h_input[i*INPUT_WIDTH + j];
    }
  }
  printMatrix(h_padded_input, PADDED_INPUT_HEIGHT, PADDED_INPUT_WIDTH, "PADDED INPUT");

  cudaMalloc(&d_input, PADDED_INPUT_WIDTH*PADDED_INPUT_HEIGHT*sizeof(int));
  cudaMalloc(&d_output, OUTPUT_WIDTH*OUTPUT_HEIGHT*sizeof(int));

  cudaMemcpy(d_input, h_padded_input, PADDED_INPUT_WIDTH*PADDED_INPUT_HEIGHT*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(kernel, h_kernel, KERNEL_WIDTH*KERNEL_HEIGHT*sizeof(int));


  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  /*
   * ======= KERNEL LAUNCH =======
   */

  dim3 threads(2, 2);
  dim3 blocks((OUTPUT_WIDTH + threads.x - 1) / threads.x, (OUTPUT_HEIGHT + threads.y - 1) / threads.y);
  int smem_size = (threads.x + KERNEL_WIDTH - 1) * (threads.y + KERNEL_HEIGHT - 1) * sizeof(int);

  conv2d<<<blocks, threads, smem_size>>>(d_input, d_output);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
  }

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  cudaMemcpy(h_output, d_output, OUTPUT_WIDTH*OUTPUT_HEIGHT*sizeof(int), cudaMemcpyDeviceToHost);

  printMatrix(h_input, INPUT_HEIGHT, INPUT_WIDTH, "INPUT");
  printMatrix(h_kernel, KERNEL_HEIGHT, KERNEL_WIDTH, "KERNEL");
  printMatrix(h_output, OUTPUT_HEIGHT, OUTPUT_WIDTH, "OUTPUT = A * B");

  // assert(validate_conv1d(h_input, h_kernel, h_output, INPUT_SIZE, KERNEL_SIZE, PADDING));

  printf("Successful!\n");
  printf("Kernel execution time: %.3f ms\n", milliseconds);

  save_tensor("input.bin", h_input, INPUT_WIDTH * INPUT_HEIGHT);
  save_tensor("kernel.bin", h_kernel, KERNEL_WIDTH * KERNEL_HEIGHT);
  save_tensor("output.bin", h_output, OUTPUT_WIDTH * OUTPUT_HEIGHT);
  printf("Saved tensors to input.bin, kernel.bin, output.bin\n");

  // Cleanup events
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cudaFree(d_input); cudaFree(d_output);
  free(h_input);
  free(h_kernel);
  free(h_output);
}
