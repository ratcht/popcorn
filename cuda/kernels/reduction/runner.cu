#include "01_divergent.cuh"
#include "02_bank_conflicts.cuh"
#include "03_no_bank_conflicts.cuh"
#include "04_sequential_addressing.cuh"
#include "05_first_add_during_load.cuh"
#include "06_unroll_last_warp.cuh"
#include "07_completely_unrolled.cuh"
#include "08_cooperative_groups.cuh"

void run_reduction_divergent(float *d_v, float *d_v_r, int n) {
  int num_threads = REDUCTION_DIVERGENT_SIZE;
  int num_blocks = (n + num_threads - 1) / num_threads;
  reductionDivergent<<<num_blocks, num_threads>>>(d_v, d_v_r);
}

void run_reduction_bank_conflicts(float *d_v, float *d_v_r, int n) {
  int num_threads = REDUCTION_BANK_CONFLICTS_SIZE;
  int num_blocks = (n + num_threads - 1) / num_threads;
  reductionBankConflicts<<<num_blocks, num_threads>>>(d_v, d_v_r);
}

void run_reduction_no_bank_conflicts(float *d_v, float *d_v_r, int n) {
  int num_threads = REDUCTION_NO_BANK_CONFLICTS_SIZE;
  int num_blocks = (n + num_threads - 1) / num_threads;
  reductionNoBankConflicts<<<num_blocks, num_threads>>>(d_v, d_v_r);
}

void run_reduction_sequential_addressing(float *d_v, float *d_v_r, int n) {
  int num_threads = REDUCTION_SEQUENTIAL_SIZE;
  int num_blocks = (n / 2 + num_threads - 1) / num_threads;
  reductionSequentialAddressing<<<num_blocks, num_threads>>>(d_v, d_v_r);
}

void run_reduction_first_add_during_load(float *d_v, float *d_v_r, int n) {
  int num_threads = REDUCTION_FIRST_ADD_SIZE;
  int num_blocks = 1024;
  reductionFirstAddDuringLoad<REDUCTION_FIRST_ADD_SIZE><<<num_blocks, num_threads>>>(d_v, d_v_r, n);
}

void run_reduction_unroll_last_warp(float *d_v, float *d_v_r, int n) {
  int num_threads = REDUCTION_UNROLL_LAST_SIZE;
  int num_blocks = (n / 2 + num_threads - 1) / num_threads;
  reductionUnrollLastWarp<<<num_blocks, num_threads>>>(d_v, d_v_r);
}

void run_reduction_completely_unrolled(float *d_v, float *d_v_r, int n) {
  int num_threads = REDUCTION_COMPLETELY_UNROLLED_SIZE;
  int num_blocks = 1024;
  reductionCompletelyUnrolled<REDUCTION_COMPLETELY_UNROLLED_SIZE><<<num_blocks, num_threads>>>(d_v, d_v_r, n);
}

void run_reduction_cooperative_groups(float *d_v, float *d_v_r, int n) {
  int num_threads = REDUCTION_COOPERATIVE_SIZE;
  int num_blocks = 1024;
  size_t shmem_size = num_threads * sizeof(float);
  reductionCooperativeGroups<<<num_blocks, num_threads, shmem_size>>>(d_v, d_v_r, n);
}