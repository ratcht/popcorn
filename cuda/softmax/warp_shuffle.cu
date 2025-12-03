#include <math.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <stdio.h>
#include <curand.h>

#define ROWS 1024
#define COLS 1024
#define BYTES sizeof(float) * ROWS * COLS

#define MASK 0xFFFFFFFF

template<typename T, typename Op>
__device__ T warp_reduce(T val, Op op) {
  // warp-level reduction
  for (int i = 16; i > 0; i /= 2) {
    T other = __shfl_xor_sync(MASK, val, i);
    val = op(val, other);
  }    // now each thread in warp has the warp's maximum

  return val;
}

template<typename T, typename Op>
__device__ T block_reduce(T val, Op op, float* shmem, int tid, int warp_id) { // i.e row reduce (since each block = row in this case)

    val = warp_reduce(val, op);

    // first thread in warp stores to shmem
    if (tid % warpSize == 0) {
      shmem[warp_id] = val;
    }

    __syncthreads();

    if (warp_id == 0) {
      // first warp performs reduction across all warp maximums
      val = tid < blockDim.x/warpSize ? shmem[tid] : 0;

      // another warp shuffle using the warp maximums (loaded into warp 0)
      val = warp_reduce(val, op);
    }

    if (tid == 0) {
      shmem[0] = val;
    }

    __syncthreads();
    return shmem[0];
}

__global__ void softmax(float *a, float *b) {
  int row = blockIdx.x;
  int tid = threadIdx.x;
  int warp_id = tid / 32;

  extern __shared__ float shmem[];

  if (row < ROWS) {
    float x_max = -INFINITY;
    float divisor = 0.0f;

    // ===== calculate x_max ======

    // thread reduction on assigned elems
    for (int i = tid; i < COLS; i += blockDim.x) {
      x_max = max(x_max, a[row*COLS + i]);
    }

    x_max = block_reduce(x_max, fmaxf, shmem, tid, warp_id);

    // ===== calculate divisor ======
    for (int i = tid; i < COLS; i += blockDim.x) {
      divisor += expf(a[row*COLS + i] - x_max);
    }

    divisor = block_reduce(divisor, [](float a, float b){return a+b;}, shmem, tid, warp_id);

    // ===== OUTPUT ======
    for (int i = tid; i < COLS; i += blockDim.x) {
      b[row*COLS + i] = expf(a[row*COLS + i] - x_max)/divisor;
    }
  }
}

void printMatrix(const float* M, int rows, int cols, const char* name) {
  printf("%s =\n", name);
  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      printf("%4f ", M[r * cols + c]);
    }
    printf("\n");
  }
  printf("\n");
}

void save_tensor(const char* filename, float* data, int size) {
  FILE *f = fopen(filename, "wb");
  if (f == NULL) {
    printf("Error opening file %s\n", filename);
    return;
  }
  fwrite(data, sizeof(float), size, f);
  fclose(f);
}

int main() {
  int id = cudaGetDevice(&id);


  // define
  float *h_a, *h_b;
  float *d_a, *d_b;

  h_a = (float*)malloc(BYTES);
  h_b = (float*)malloc(BYTES);

  cudaMalloc(&d_a, BYTES);
  cudaMalloc(&d_b, BYTES);

  // fill in data
  curandGenerator_t prng;
  curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(prng, 462);
  curandGenerateUniform(prng, d_a, ROWS*COLS);

  int NUM_THREADS = 64;
  int NUM_BLOCKS = ROWS;
  size_t shmem_size = NUM_THREADS * sizeof(float);

  // ===== START TRACKING
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  // LAUNCH KERNEL
  softmax<<<NUM_BLOCKS, NUM_THREADS, shmem_size>>>(d_a, d_b);

  // ===== STOP TRACKING
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  cudaMemcpy(h_a, d_a, BYTES, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_b, d_b, BYTES, cudaMemcpyDeviceToHost);

  // printMatrix(h_a, ROWS, COLS, "INPUT");
  // printMatrix(h_b, ROWS, COLS, "OUTPUT");

  // assert(validate_conv1d(h_input, h_kernel, h_output, INPUT_SIZE, KERNEL_SIZE, PADDING));

  printf("Successful!\n");
  printf("Kernel execution time: %.3f ms\n", milliseconds);

  save_tensor("input.bin", h_a, ROWS*COLS);
  save_tensor("output.bin", h_b, ROWS*COLS);
  printf("Saved tensors to input.bin, output.bin\n");

  // Cleanup events
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cudaFree(d_a); cudaFree(d_b);
  free(h_a); free(h_b);
}
