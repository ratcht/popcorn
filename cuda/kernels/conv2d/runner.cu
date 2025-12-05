#include "01_tiled.cuh"

void run_conv2d_tiled(int *d_padded_input, int *d_output, int padded_input_width, int output_width, int output_height, int kernel_width, int kernel_height) {
  dim3 threads(2, 2);
  dim3 blocks((output_width + threads.x - 1) / threads.x, (output_height + threads.y - 1) / threads.y);
  int smem_size = (threads.x + kernel_width - 1) * (threads.y + kernel_height - 1) * sizeof(int);
  conv2dTiled<<<blocks, threads, smem_size>>>(d_padded_input, d_output, padded_input_width, output_width, output_height, kernel_width, kernel_height);
}