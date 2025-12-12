#include "01_naive.cuh"

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

void run_rope_naive(float* x, float* cos, float* sin,
                        int B, int L, int D) {

  rope_naive<<<B*L, D>>>(x, cos, sin, B, L, D);
}
