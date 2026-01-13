[Link to LLM implementations](https://github.com/ratcht/llm)

# Popcorn - A collection of kernels & operations

GPU kernel implementations (+ assembled torch operations) in CUDA & Triton. This is a learning project exploring different optimization techniques for common operations.

## Repository Structure

```
popcorn/
├── cuda/              # CUDA kernels
├── tl/                # Triton kernels
├── pytorch/          # PyTorch implementations
└── validation/        # Kernel correctness validation scripts
```

## What's Implemented

**CUDA kernels** (in `cuda/kernels/`):
- Vector addition
- Matrix multiplication (+ SGEMM)
- 1D convolution
- 2D convolution
- Sum reduction
- Softmax
- Fused QKV Projection
- RoPE

Each operation has multiple implementations demonstrating different optimization techniques: naive implementations, shared memory usage, memory coalescing, warp-level primitives, cooperative groups, etc.

**Triton kernels** (in `tl/kernels/`):
- Vector addition
- Softmax
- Layer normalization
- Matrix multiplication

**PyTorch implementations** (in `pytorch/`):
- Conv1d
- Conv2d
- Self-Attention
- Layer Normalization
- RMS Normalization
- RoPE

## `cuda`

See `cuda/README.md` for detailed instructions on building and running CUDA benchmarks.

Quick start:
```bash
cd cuda
make                                    # compile all benchmarks
./benchmarks/bench_matmul 2 1024        # run tiled matmul on 1024x1024 matrix
./benchmarks/bench_reduction 7 1048576  # run cooperative groups reduction
```

## `tl`

See `tl/README.md` for detailed instructions on building and running Triton benchmarks.

Quick start:
```bash
cd tl
python -m benchmarks.bench_softmax
```

## `pytorch`

To run tests:
```bash
cd pytorch
python -m pytest __tests__/test_rope.py   # run RoPE tests
```


## Goals

- Learn GPU programming and optimization techniques
- Compare custom implementations against optimized libraries (cuBLAS, cuDNN)
- Implement same operations in different frameworks (CUDA, Triton, PyTorch)
- Document performance characteristics and optimization strategies
