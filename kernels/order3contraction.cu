#include "order3contraction.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdlib>
#include <iostream>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__      \
                      << ": " << cudaGetErrorString(err) << std::endl;         \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

__global__ void tensorContractionKernel(const float* A, const float* B,
                                        float* C, int I, int J, int K, int L,
                                        int M) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int l = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= I || j >= J || l >= L) {
        return;
    }

    for (int m = 0; m < M; ++m) {
        float value = 0.0f;
        for (int k = 0; k < K; ++k) {
            value += A[(i * J + j) * K + k] * B[(k * L + l) * M + m];
        }
        C[((i * J + j) * L + l) * M + m] = value;
    }
}

void order3contraction(const float* A, const float* B, float* C, int I, int J,
                       int K, int L, int M) {
    size_t bytesA = static_cast<size_t>(I) * J * K * sizeof(float);
    size_t bytesB = static_cast<size_t>(K) * L * M * sizeof(float);
    size_t bytesC = static_cast<size_t>(I) * J * L * M * sizeof(float);

    float* dA = nullptr;
    float* dB = nullptr;
    float* dC = nullptr;

    CUDA_CHECK(cudaMalloc(&dA, bytesA));
    CUDA_CHECK(cudaMalloc(&dB, bytesB));
    CUDA_CHECK(cudaMalloc(&dC, bytesC));

    CUDA_CHECK(cudaMemcpy(dA, A, bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B, bytesB, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(8, 8, 4);
    dim3 blocksPerGrid((I + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (J + threadsPerBlock.y - 1) / threadsPerBlock.y,
                       (L + threadsPerBlock.z - 1) / threadsPerBlock.z);

    tensorContractionKernel<<<blocksPerGrid, threadsPerBlock>>>(dA, dB, dC, I,
                                                                J, K, L, M);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(C, dC, bytesC, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
}
