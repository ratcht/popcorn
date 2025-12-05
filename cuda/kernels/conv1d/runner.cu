#include "01_naive.cuh"
#include "02_constant_memory.cuh"
#include "03_tiled.cuh"
#include "04_tiled_padded.cuh"
#include "05_strided_padded.cuh"

void run_conv1d_naive(int *d_input, int *d_kernel, int *d_output, int input_size, int kernel_size) {
  int output_size = input_size - kernel_size + 1;
  int num_threads = 32;
  int num_blocks = (output_size + num_threads - 1) / num_threads;
  conv1dNaive<<<num_blocks, num_threads>>>(d_input, d_kernel, d_output, input_size, kernel_size);
}

void run_conv1d_constant_memory(int *d_input, int *d_output, int input_size, int kernel_size) {
  int output_size = input_size - kernel_size + 1;
  int num_threads = 32;
  int num_blocks = (output_size + num_threads - 1) / num_threads;
  conv1dConstantMemory<<<num_blocks, num_threads>>>(d_input, d_output, input_size, kernel_size);
}

void run_conv1d_tiled(int *d_input, int *d_output, int input_size, int kernel_size) {
  int output_size = input_size - kernel_size + 1;
  int num_threads = 32;
  int num_blocks = (output_size + num_threads - 1) / num_threads;
  int smem_size = (num_threads + kernel_size - 1) * sizeof(int);
  conv1dTiled<<<num_blocks, num_threads, smem_size>>>(d_input, d_output, input_size, kernel_size);
}

void run_conv1d_tiled_padded(int *d_padded_input, int *d_output, int output_size, int kernel_size) {
  int num_threads = 128;
  int num_blocks = (output_size + num_threads - 1) / num_threads;
  int smem_size = (num_threads + kernel_size - 1) * sizeof(int);
  conv1dTiledPadded<<<num_blocks, num_threads, smem_size>>>(d_padded_input, d_output, output_size, kernel_size);
}

void run_conv1d_strided_padded(int *d_input, int *d_output, int input_size, int kernel_size, int padding, int stride, int output_size) {
  int num_threads = 4;
  int num_blocks = (output_size + num_threads - 1) / num_threads;
  int smem_size = (num_threads + kernel_size - 1) * sizeof(int);
  conv1dStridedPadded<<<num_blocks, num_threads, smem_size>>>(d_input, d_output, input_size, kernel_size, padding, stride, output_size);
}