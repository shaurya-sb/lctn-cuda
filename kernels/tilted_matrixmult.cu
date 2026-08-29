#include "tilted_matrixmult.cuh"

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

constexpr int TILE_SIZE = 16;

__global__ void tiltedMatrixMultiplyKernel(const float* A, const float* B,
                                           float* C, int M, int N, int K) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float sum = 0.0f;
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int tile = 0; tile < numTiles; ++tile) {
        int tiledColA = tile * TILE_SIZE + tx;
        int tiledRowB = tile * TILE_SIZE + ty;

        if (row < M && tiledColA < K) {
            tileA[ty][tx] = A[row * K + tiledColA];
        } else {
            tileA[ty][tx] = 0.0f;
        }

        if (tiledRowB < K && col < N) {
            tileB[ty][tx] = B[tiledRowB * N + col];
        } else {
            tileB[ty][tx] = 0.0f;
        }

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += tileA[ty][i] * tileB[i][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void tiltedMatrixMultiplicationGpu(const float* A, const float* B, float* C,
                                   int M, int N, int K) {
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

    tiltedMatrixMultiplicationDevice(dA, dB, dC, M, N, K);

    CUDA_CHECK(cudaMemcpy(C, dC, bytesC, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
}

void tiltedMatrixMultiplicationDevice(const float* dA, const float* dB,
                                      float* dC, int M, int N, int K) {
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    tiltedMatrixMultiplyKernel<<<blocksPerGrid, threadsPerBlock>>>(dA, dB, dC,
                                                                   M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void tiltedMatrixMultiplicationCpu(const float* A, const float* B, float* C,
                                   int M, int N, int K) {
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            float value = 0.0f;
            for (int i = 0; i < K; ++i) {
                value += A[row * K + i] * B[i * N + col];
            }
            C[row * N + col] = value;
        }
    }
}

bool tiltedMatricesMatch(const std::vector<float>& a,
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
