# lctn-cuda
kernels for tensor contraction, writing an N-dimensional tensor class, and implementing straggler mitigation for tensor network contraction from the ground up. Inspired by UCSB Science of Information Lab work.

## Current starter program

The current `main.cu` is a naive CUDA matrix multiplication example with a CPU
correctness check. It is meant to compile on a Linux machine with an NVIDIA GPU
and CUDA installed.

See `REMOTE_GPU.md` for the rented-GPU setup and run commands.

## Current Progress

- Implemented a naive CUDA matrix multiplication kernel as the initial contraction primitive.
- Developed an order-3 tensor contraction kernel:
  $$
  C_{ijlm} = \sum_k A_{ijk}B_{klm}
  $$
- Developed an order-4 multi-index tensor contraction:
  $$
  C_{ijmn} = \sum_{k,l} A_{ijkl}B_{klmn}
  $$
- Implemented row-major flattening for multidimensional tensors stored as contiguous arrays.
- Mapped free tensor indices across CUDA thread/block dimensions while iterating over contracted indices within each thread.
- Added CUDA host-side memory allocation, host/device transfers, kernel dispatch, and output retrieval.
- Currently performing static/code-level verification while GPU hardware is unavailable. Runtime correctness and performance have not yet been benchmarked.

## Future Work

### Tensor Network Contraction

- Compose multiple CUDA contractions so intermediate GPU tensors become inputs to subsequent contractions.
- Build an N-dimensional tensor abstraction containing shape and stride metadata.
- Generalize contraction beyond hard-coded tensor orders and contraction axes.
- Explore contraction ordering and intermediate tensor management.

### Validation

- Add CPU and NumPy reference implementations.
- Perform numerical CPU/GPU correctness testing.
- Test kernels with CUDA debugging and memory-checking tools.

### CUDA Optimization

- Introduce shared-memory tiling.
- Improve global-memory access and memory coalescing.
- Analyze arithmetic intensity and data reuse.
- Profile kernels and investigate occupancy, memory traffic, and computational bottlenecks.

### Python Interface

Develop a high-level Python interface over the C++/CUDA backend:

```text
Python API
    |
    v
Tensor / Network Abstraction
    |
    v
C++ / CUDA Backend
    |
    v
GPU Contraction Kernels
```

### Lagrange-Coded Computation

- Implement polynomial encoding of tensor inputs.
- Execute contractions on coded worker inputs.
- Recover tensor-network contraction results through interpolation-based decoding.
- Simulate straggling workers and measure recovery behavior.
- Compare coded and uncoded contraction pipelines.

## Long-Term Architecture

```text
Python API
    |
    v
Tensor / Network Abstraction
    |
    v
Contraction Planner
    |
    v
C++ / CUDA Backend
    |
    v
Optimized GPU Contraction Kernels
    |
    v
Coded Worker Execution
    |
    v
Interpolation / Recovery
```

## Goal

The long-term goal is to connect the mathematical structure of coded tensor-network contraction with an executable GPU implementation and study both recovery behavior and GPU performance.
