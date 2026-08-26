#include "order4contraction.cuh"

#include <cuda_runtime.h>

#include <cmath>
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

__global__ void order4ContractionKernel(const float* A, const float* B,
                                        float* C, int I, int J, int K, int L,
                                        int M, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int m = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= I || j >= J || m >= M) {
        return;
    }

    for (int n = 0; n < N; ++n) {
        float value = 0.0f;
        for (int k = 0; k < K; ++k) {
            for (int l = 0; l < L; ++l) {
                value += A[((i * J + j) * K + k) * L + l] *
                         B[((k * L + l) * M + m) * N + n];
            }
        }
        C[((i * J + j) * M + m) * N + n] = value;
    }
}

void order4contractionGpu(const float* A, const float* B, float* C, int I,
                          int J, int K, int L, int M, int N) {
    size_t bytesA = static_cast<size_t>(I) * J * K * L * sizeof(float);
    size_t bytesB = static_cast<size_t>(K) * L * M * N * sizeof(float);
    size_t bytesC = static_cast<size_t>(I) * J * M * N * sizeof(float);

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
                       (M + threadsPerBlock.z - 1) / threadsPerBlock.z);

    order4ContractionKernel<<<blocksPerGrid, threadsPerBlock>>>(dA, dB, dC, I,
                                                                J, K, L, M, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(C, dC, bytesC, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
}

void order4contractionCpu(const float* A, const float* B, float* C, int I,
                          int J, int K, int L, int M, int N) {
    for (int i = 0; i < I; ++i) {
        for (int j = 0; j < J; ++j) {
            for (int m = 0; m < M; ++m) {
                for (int n = 0; n < N; ++n) {
                    float value = 0.0f;
                    for (int k = 0; k < K; ++k) {
                        for (int l = 0; l < L; ++l) {
                            value += A[((i * J + j) * K + k) * L + l] *
                                     B[((k * L + l) * M + m) * N + n];
                        }
                    }
                    C[((i * J + j) * M + m) * N + n] = value;
                }
            }
        }
    }
}

bool order4OutputsMatch(const std::vector<float>& a,
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
