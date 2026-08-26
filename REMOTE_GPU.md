# Running on a rented CUDA GPU

This project is CUDA code. Your Mac can edit it, but it cannot compile or run
it locally because CUDA needs an NVIDIA GPU plus NVIDIA's CUDA compiler
(`nvcc`). A rented Linux GPU machine is the right target.

## 1. Rent a CUDA-capable Linux machine

Pick an Ubuntu/Linux instance with an NVIDIA GPU. After connecting with SSH,
check that the GPU is visible:

```bash
nvidia-smi
```

If that command fails, the rented machine is not ready for CUDA yet.

## 2. Install build tools if needed

Some GPU images already include these. If not:

```bash
sudo apt update
sudo apt install -y build-essential cmake ninja-build nvidia-cuda-toolkit
```

Then check:

```bash
nvcc --version
cmake --version
```

## 3. Clone this repo on the GPU machine

```bash
git clone https://github.com/shaupeda07/lctn-cuda.git
cd lctn-cuda
```

## 4. Build and run

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build
./build/CUDAStuff
```

Expected final line:

```text
Matches CPU result: yes
```

## Where the CUDA API lives

The host-side CUDA API calls are inside `matrixMultiplicationGpu` in `main.cu`:

- `cudaMalloc`: allocate memory on the GPU
- `cudaMemcpyHostToDevice`: copy CPU data to GPU memory
- `naiveMatrixMultiplyKernel<<<...>>>`: launch GPU work
- `cudaDeviceSynchronize`: wait for the GPU to finish
- `cudaMemcpyDeviceToHost`: copy the GPU result back to CPU memory
- `cudaFree`: release GPU memory

The `__global__` function is the actual GPU kernel.

## Using CLion from your Mac

For a real remote workflow, configure a CLion SSH remote toolchain that points
at the rented Linux GPU machine. CLion should run CMake, `nvcc`, and the built
program on the rented machine, not on macOS.
