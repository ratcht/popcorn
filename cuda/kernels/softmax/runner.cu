#include "01_naive.cuh"
#include "02_shared_memory.cuh"
#include "03_warp_shuffle.cuh"

void run_softmax_naive(float *d_a, float *d_b, int rows, int cols) {
  int num_threads = 4;
  int num_blocks = (rows + num_threads - 1) / num_threads;
  softmaxNaive<<<num_blocks, num_threads>>>(d_a, d_b, rows, cols);
}

void run_softmax_shared_memory(float *d_a, float *d_b, int rows, int cols) {
  int num_threads = 4;
  int num_blocks = rows;
  size_t shmem_size = num_threads * sizeof(float);
  softmaxSharedMemory<<<num_blocks, num_threads, shmem_size>>>(d_a, d_b, rows, cols);
}

void run_softmax_warp_shuffle(float *d_a, float *d_b, int rows, int cols) {
  int num_threads = 64;
  int num_blocks = rows;
  size_t shmem_size = num_threads * sizeof(float);
  softmaxWarpShuffle<<<num_blocks, num_threads, shmem_size>>>(d_a, d_b, rows, cols);
}