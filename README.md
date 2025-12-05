# Popcorn - A collection of kernels & operations

GPU kernel implementations (+ assembled torch operations) in CUDA & Triton. This is a learning project exploring different optimization techniques for common operations.

## Repository Structure

```
kernel/
├── cuda/              # CUDA kernels
├── triton/            # Triton kernels (planned)
├── torch_op/          # PyTorch implementations
├── validation/        # Kernel correctness validation scripts
└── benchmarks/        # Performance comparison tools (planned)
```

## What's Implemented

**CUDA kernels** (in `cuda/kernels/`):
- Vector addition
- Matrix multiplication
- 1D convolution
- 2D convolution
- Sum reduction
- Softmax

Each operation has multiple implementations demonstrating different optimization techniques: naive implementations, shared memory usage, memory coalescing, warp-level primitives, cooperative groups, etc.

**PyTorch implementations** (in `torch_op/`):
- Conv1d
- Conv2d
- SelfAttention

## Building and Running

See `cuda/README.md` for detailed instructions on building and running CUDA benchmarks.

Quick start:
```bash
cd cuda
make                                    # compile all benchmarks
./benchmarks/bench_matmul 2 1024        # run tiled matmul on 1024x1024 matrix
./benchmarks/bench_reduction 7 1048576  # run cooperative groups reduction
```

## Goals

- Learn GPU programming and optimization techniques
- Compare custom implementations against optimized libraries (cuBLAS, cuDNN)
- Implement same operations in different frameworks (CUDA, Triton, PyTorch)
- Document performance characteristics and optimization strategies
