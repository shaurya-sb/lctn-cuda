# lctn-cuda
kernels for tensor contraction, writing an N-dimensional tensor class, and implementing straggler mitigation for tensor network contraction, inspired by UCSB Science of Information Lab work.

## Current starter program

The current `main.cu` is a naive CUDA matrix multiplication example with a CPU
correctness check. It is meant to compile on a Linux machine with an NVIDIA GPU
and CUDA installed.

See `REMOTE_GPU.md` for the rented-GPU setup and run commands.
