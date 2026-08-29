# lctn-cuda

CUDA kernels for experimenting with tensor contractions and small tensor-network contraction pipelines. The current code focuses on comparing straightforward naive kernels against shared-memory tiled kernels for contractions that can be viewed as matrix multiplication after row-major flattening.

This is an early research/learning project. The implemented code is intentionally small and explicit: tensors are stored as flat `float*` arrays, CPU reference implementations are included for correctness checks, and the benchmark driver prints CPU time, GPU wrapper time, kernel-only tiled-vs-naive ratios, and correctness status.

## Implemented

- Naive CUDA matrix multiplication.
- Shared-memory tiled CUDA matrix multiplication.
- Naive order-3 tensor contraction:

  ```text
  C[i,j,l,m] = sum_k A[i,j,k] * B[k,l,m]
  ```

- Shared-memory tiled order-3 contraction by flattening:

  ```text
  A[I,J,K] -> A[(I*J), K]
  B[K,L,M] -> B[K, (L*M)]
  C[I,J,L,M] -> C[(I*J), (L*M)]
  ```

- Naive order-4 contraction over two contracted indices:

  ```text
  C[i,j,m,n] = sum_k sum_l A[i,j,k,l] * B[k,l,m,n]
  ```

- A two-step naive tensor-network contraction:

  ```text
  T[i,j,l,m] = sum_k A[i,j,k] * B[k,l,m]
  C[i,j,n]   = sum_l sum_m T[i,j,l,m] * D[l,m,n]
  ```

- A tiled version of the second tensor-network contraction step, treating:

  ```text
  T[(I*J), (L*M)] * D[(L*M), N] -> C[(I*J), N]
  ```

- CPU implementations for the same contractions used by the benchmark driver to check GPU output.
- Kernel-only tiled-vs-naive timing using CUDA events for the implemented device-pointer launchers.

## Work In Progress

- General N-dimensional tensor indexing with reusable shape/stride metadata.
- General order-N tensor contraction with arbitrary free and contracted axes.
- NumPy reference validation inside this repository.
- Larger benchmark sweeps across tensor shapes and contraction patterns.
- Lagrange/polynomial coded-computation integration for straggler mitigation.
- Distributed or multi-worker execution.

The repository does not currently implement or measure distributed coded computation, memory bandwidth saturation, occupancy improvements, or full contraction-planner behavior.

## Project Layout

```text
.
├── CMakeLists.txt
├── main.cu
├── REMOTE_GPU.md
└── kernels
    ├── matrix_multiply.cu/.cuh
    ├── tilted_matrixmult.cu/.cuh
    ├── order3contraction.cu/.cuh
    ├── order4contraction.cu/.cuh
    ├── naivenetworkcontraction.cu/.cuh
    └── tiltedcontraction.cu/.cuh
```

## Build And Run

This project requires a Linux machine with an NVIDIA GPU, CUDA, CMake, and a C++/CUDA compiler. It will not compile natively on macOS without a CUDA-capable NVIDIA GPU environment.

```bash
git clone https://github.com/shaurya-sb/lctn-cuda.git
cd lctn-cuda
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/CUDAStuff
```

If Ninja is not installed, either install it or omit `-G Ninja` and use the default CMake generator.

## Benchmark Driver

`main.cu` runs fixed-size correctness and timing checks. `GPU ms` includes the current host wrapper cost: allocation, host-to-device copies, kernel launch, device synchronization, device-to-host copy, and frees. `kernel vs naive` is different: it uses CUDA events around already-allocated device-pointer launchers and compares the tiled kernel path against the matching naive GPU kernel path.

A value greater than `1.0` in `kernel vs naive` means the tiled kernel path was faster than the naive GPU path for that problem size. A value below `1.0` means the tiled path was slower.

Example benchmark output from an RTX 3090 RunPod instance:

```text
Kernel                            Problem                 CPU ms      GPU ms     Speedup   kernel vs naive   Correct
--------------------------------------------------------------------------------------------------------------------
Matrix multiply naive             256x256x256             29.622       0.532      55.659             1.000       yes
Matrix multiply tiled             256x256x256             29.622       0.179     165.451             0.942       yes
Order-3 contraction naive         16,16,64,16,16           5.500       0.536      10.268             1.000       yes
Order-3 contraction tiled         16,16,64,16,16           5.500       0.087      62.871            11.639       yes
Order-4 contraction naive         8,8,32,16,8,32          12.114       3.038       3.988                 -       yes
Network contraction naive         16,16,64,16,16,64        8.725       0.423      20.626             1.000       yes
Network contraction tiled         16,16,64,16,16,64        8.725       0.095      91.589             2.039       yes
```

These numbers are shape-dependent and include only the benchmark cases currently hard-coded in `main.cu`.

## Correctness Validation

The CUDA outputs are compared against C++ CPU reference implementations using floating-point tolerances in `main.cu`. The completed correctness checks are CPU-vs-GPU checks for the implemented C++/CUDA contractions.

NumPy validation is not currently implemented in this repository. It is planned as a separate reference path for future order-N tensor indexing and contraction work.

## Notes On Tiling

The tiled kernels use shared memory for contraction patterns that can be expressed as matrix multiplication, where tiles of the left and right operands can be reused across multiple multiply-add operations.

This does not mean tiling is always better. For operations with little or no reusable reduction structure, such as a Hadamard product or a simple inner product case, the extra shared-memory loads and synchronization can be mathematically unnecessary overhead. In those cases a direct naive kernel can be the better baseline, and any tiled version should be justified by measurement.

No claims are made here about memory coalescing, occupancy, bandwidth saturation, or hardware-level optimality. The current benchmark only reports the measured behavior of these specific kernels and shapes.

## Development Direction

The intended next step is to replace hard-coded tensor orders with a general shape/stride indexing layer, then use that layer to express order-N tensor contractions and tensor-network contraction sequences. Lagrange-coded computation and straggler-mitigation exist as prototypes built with NumPy elsewhere and are not yet implemented in the CUDA benchmark path.
