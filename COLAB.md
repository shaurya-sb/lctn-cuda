# Running this CUDA project in Google Colab

Your Mac edits the code, but Colab compiles and runs it on a remote NVIDIA GPU.
There is no CUDA API call you put in the source file to create a virtual GPU.
The GPU is selected by the runtime environment before the program starts.

## 1. Turn on a GPU in Colab

In Colab:

Runtime -> Change runtime type -> Hardware accelerator -> GPU

Then run:

```bash
!nvidia-smi
!nvcc --version
```

If those commands work, Colab gave you a CUDA-capable GPU.

## 2. Upload `main.cu`

Run this cell:

```python
from google.colab import files
files.upload()
```

Choose this project's `main.cu`.

## 3. Compile and run

Run:

```bash
!nvcc -std=c++20 main.cu -o CUDAStuff
!./CUDAStuff
```

Expected final line:

```text
Matches CPU result: yes
```

## Where the CUDA API is in the code

The host-side CUDA API calls are inside `matrixMultiplicationGpu` in `main.cu`:

- `cudaMalloc`: allocate memory on the GPU
- `cudaMemcpyHostToDevice`: copy CPU data to GPU memory
- `naiveMatrixMultiplyKernel<<<...>>>`: launch GPU work
- `cudaDeviceSynchronize`: wait for the GPU to finish
- `cudaMemcpyDeviceToHost`: copy the GPU result back to CPU memory
- `cudaFree`: release GPU memory

The `__global__` function is the actual GPU kernel.
