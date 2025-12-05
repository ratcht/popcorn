#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <stdio.h>
#include <cuda_runtime.h>

template<typename T>
void print_vector(const T* v, int n, const char* name) {
  printf("%s = [ ", name);
  for (int i = 0; i < n; i++) {
    printf("%g ", (double)v[i]);
  }
  printf("]\n");
}

template<typename T>
void print_matrix(const T* M, int rows, int cols, const char* name) {
  printf("%s =\n", name);
  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      printf("%4g ", (double)M[r * cols + c]);
    }
    printf("\n");
  }
  printf("\n");
}

template<typename T>
void save_tensor(const char* filename, const T* data, int size) {
  FILE *f = fopen(filename, "wb");
  if (f == NULL) {
    printf("Error opening file %s\n", filename);
    return;
  }
  fwrite(data, sizeof(T), size, f);
  fclose(f);
}

void print_device_info() {
  int device;
  cudaGetDevice(&device);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);
  printf("Running on GPU %d: %s\n", device, prop.name);
}

void verify_matmul(int *a, int *b, int *c, int n) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      int tmp = 0;
      for (int k = 0; k < n; k++) {
        tmp += a[i * n + k] * b[k * n + j];
      }
      if (tmp != c[i * n + j]) {
        printf("Verification failed at (%d, %d): expected %d, got %d\n", i, j, tmp, c[i * n + j]);
        return;
      }
    }
  }
  printf("Verification passed!\n");
}

void transpose(int *a, int *a_t, int n) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      a_t[j * n + i] = a[i * n + j];
    }
  }
}

bool validate_conv1d(int *input, int *kernel, int *gpu_output, int input_size, int kernel_size) {
  int output_size = input_size - kernel_size + 1;

  for (int i = 0; i < output_size; i++) {
    int expected = 0;
    for (int j = 0; j < kernel_size; j++) {
      expected += input[i + j] * kernel[j];
    }

    if (expected != gpu_output[i]) {
      printf("Conv1d validation failed at index %d: expected %d, got %d\n", i, expected, gpu_output[i]);
      return false;
    }
  }
  printf("Conv1d validation passed!\n");
  return true;
}

bool validate_conv1d_padded(int *input, int *kernel, int *gpu_output, int input_size, int kernel_size, int padding) {
  int output_size = input_size + (2 * padding) - kernel_size + 1;

  for (int i = 0; i < output_size; i++) {
    int expected = 0;
    for (int j = 0; j < kernel_size; j++) {
      if (i + j < padding || padding + input_size <= i + j) {
        expected += 0;
      } else {
        expected += input[i + j - padding] * kernel[j];
      }
    }

    if (expected != gpu_output[i]) {
      printf("Conv1d padded validation failed at index %d: expected %d, got %d\n", i, expected, gpu_output[i]);
      return false;
    }
  }
  printf("Conv1d padded validation passed!\n");
  return true;
}

bool validate_conv1d_strided(int *input, int *kernel, int *gpu_output, int input_size, int kernel_size, int padding, int stride) {
  int output_size = (input_size + (2 * padding) - kernel_size) / stride + 1;

  for (int i = 0; i < output_size; i++) {
    int expected = 0;
    for (int j = 0; j < kernel_size; j++) {
      int idx = i * stride + j - padding;
      if (idx < 0 || idx >= input_size) {
        expected += 0;
      } else {
        expected += input[idx] * kernel[j];
      }
    }

    if (expected != gpu_output[i]) {
      printf("Conv1d strided validation failed at index %d: expected %d, got %d\n", i, expected, gpu_output[i]);
      return false;
    }
  }
  printf("Conv1d strided validation passed!\n");
  return true;
}

#endif