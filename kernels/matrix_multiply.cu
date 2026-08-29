#include "matrix_multiply.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call) do { cudaError_t err = (call); if (err != cudaSuccess) { std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; std::exit(EXIT_FAILURE); } } while (0)

__global__ void naiveMatrixMultiplyKernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= M || col >= N) return;
    float value = 0.0f;
    for (int i = 0; i < K; ++i) value += A[row * K + i] * B[i * N + col];
    C[row * N + col] = value;
}

void matrixMultiplicationGpu(const float* A, const float* B, float* C, int M, int N, int K) {
    size_t bytesA = static_cast<size_t>(M) * K * sizeof(float);
    size_t bytesB = static_cast<size_t>(K) * N * sizeof(float);
    size_t bytesC = static_cast<size_t>(M) * N * sizeof(float);
    float* dA = nullptr;
    float* dB = nullptr;
    float* dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, bytesA));
    CUDA_CHECK(cudaMalloc(&dB, bytesB));
    CUDA_CHECK(cudaMalloc(&dC, bytesC));
    CUDA_CHECK(cudaMemcpy(dA, A, bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B, bytesB, cudaMemcpyHostToDevice));
    matrixMultiplicationDevice(dA, dB, dC, M, N, K);
    CUDA_CHECK(cudaMemcpy(C, dC, bytesC, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
}

void matrixMultiplicationDevice(const float* dA, const float* dB, float* dC,
                                int M, int N, int K) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    naiveMatrixMultiplyKernel<<<blocksPerGrid, threadsPerBlock>>>(dA, dB, dC, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void matrixMultiplicationCpu(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            float value = 0.0f;
            for (int i = 0; i < K; ++i) value += A[row * K + i] * B[i * N + col];
            C[row * N + col] = value;
        }
    }
}

bool matricesMatch(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (std::fabs(a[i] - b[i]) > 1.0e-5f) return false;
    return true;
}
