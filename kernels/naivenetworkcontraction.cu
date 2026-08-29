#include "naivenetworkcontraction.cuh"

#include "order3contraction.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__      \
                      << ": " << cudaGetErrorString(err) << std::endl;         \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

__global__ void networkSecondContractionKernel(const float* T, const float* D,
                                               float* C, int I, int J, int L,
                                               int M, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= I || j >= J || n >= N) {
        return;
    }

    float value = 0.0f;
    for (int l = 0; l < L; ++l) {
        for (int m = 0; m < M; ++m) {
            value += T[((i * J + j) * L + l) * M + m] *
                     D[(l * M + m) * N + n];
        }
    }

    C[(i * J + j) * N + n] = value;
}

void naiveNetworkContractionGpu(const float* A, const float* B, const float* D,
                                float* C, int I, int J, int K, int L, int M,
                                int N) {
    size_t bytesA = static_cast<size_t>(I) * J * K * sizeof(float);
    size_t bytesB = static_cast<size_t>(K) * L * M * sizeof(float);
    size_t bytesT = static_cast<size_t>(I) * J * L * M * sizeof(float);
    size_t bytesD = static_cast<size_t>(L) * M * N * sizeof(float);
    size_t bytesC = static_cast<size_t>(I) * J * N * sizeof(float);

    float* dA = nullptr;
    float* dB = nullptr;
    float* dT = nullptr;
    float* dD = nullptr;
    float* dC = nullptr;

    CUDA_CHECK(cudaMalloc(&dA, bytesA));
    CUDA_CHECK(cudaMalloc(&dB, bytesB));
    CUDA_CHECK(cudaMalloc(&dT, bytesT));
    CUDA_CHECK(cudaMalloc(&dD, bytesD));
    CUDA_CHECK(cudaMalloc(&dC, bytesC));

    CUDA_CHECK(cudaMemcpy(dA, A, bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B, bytesB, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dD, D, bytesD, cudaMemcpyHostToDevice));

    order3contractionDevice(dA, dB, dT, I, J, K, L, M);

    dim3 threadsPerBlock(8, 8, 4);
    dim3 blocksPerGrid((I + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (J + threadsPerBlock.y - 1) / threadsPerBlock.y,
                       (N + threadsPerBlock.z - 1) / threadsPerBlock.z);

    networkSecondContractionKernel<<<blocksPerGrid, threadsPerBlock>>>(
        dT, dD, dC, I, J, L, M, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(C, dC, bytesC, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dT));
    CUDA_CHECK(cudaFree(dD));
    CUDA_CHECK(cudaFree(dC));
}

void naiveNetworkContractionCpu(const float* A, const float* B, const float* D,
                                float* C, int I, int J, int K, int L, int M,
                                int N) {
    std::vector<float> T(static_cast<size_t>(I) * J * L * M);

    for (int i = 0; i < I; ++i) {
        for (int j = 0; j < J; ++j) {
            for (int l = 0; l < L; ++l) {
                for (int m = 0; m < M; ++m) {
                    float value = 0.0f;
                    for (int k = 0; k < K; ++k) {
                        value += A[(i * J + j) * K + k] *
                                 B[(k * L + l) * M + m];
                    }
                    T[((i * J + j) * L + l) * M + m] = value;
                }
            }
        }
    }

    for (int i = 0; i < I; ++i) {
        for (int j = 0; j < J; ++j) {
            for (int n = 0; n < N; ++n) {
                float value = 0.0f;
                for (int l = 0; l < L; ++l) {
                    for (int m = 0; m < M; ++m) {
                        value += T[((i * J + j) * L + l) * M + m] *
                                 D[(l * M + m) * N + n];
                    }
                }
                C[(i * J + j) * N + n] = value;
            }
        }
    }
}

bool naiveNetworkOutputsMatch(const std::vector<float>& a,
                              const std::vector<float>& b) {
    if (a.size() != b.size()) {
        return false;
    }

    for (size_t i = 0; i < a.size(); ++i) {
        if (std::fabs(a[i] - b[i]) > 1.0e-5f) {
            return false;
        }
    }

    return true;
}
