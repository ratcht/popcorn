# Triton

Triton kernel implementations for common operations. Each kernel demonstrates Triton's high-level GPU programming model.

## Structure

```
tl/
├── kernels/           # kernel implementations
├── benchmarks/        # benchmark programs comparing triton vs torch
└── utils.py           # helper functions
```

## Operations

**vector_add**
- Basic element-wise addition using 1D grid
- Demonstrates SPMD launch pattern and masking

**softmax**
- Row-wise softmax with persistent kernels
- Uses online softmax algorithm (subtract max for numerical stability)
- Dynamic occupancy calculation based on register/shared memory usage

**layer_norm**
- Forward pass layer normalization
- Chunked mean and variance computation
- Supports arbitrary normalized shapes

**matmul**
- Matrix multiplication (planned)

## Setup

Requires a virtual environment with Triton installed:
```bash
python -m venv .venv
source .venv/bin/activate
pip install triton torch
```

## Running Kernels

Each kernel can be run directly to verify correctness:
```bash
python -m kernels.softmax      # run softmax with test input
python -m kernels.vector_add   # run vector_add with test input
python -m kernels.layer_norm   # run layer_norm with test input
```

## Running Benchmarks

Benchmarks compare Triton implementations against PyTorch:
```bash
python -m benchmarks.bench_softmax      # benchmark softmax
python -m benchmarks.bench_vector_add   # benchmark vector_add
```

Benchmarks output performance data and save plots to the current directory.