#include "tiltedcontraction.cuh"

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

constexpr int CONTRACTION_TILE_SIZE = 16;

__global__ void tiledSecondContractionKernel(const float* T, const float* D,
                                             float* C, int IJ, int L, int M,
                                             int N) {
    __shared__ float tileT[CONTRACTION_TILE_SIZE][CONTRACTION_TILE_SIZE];
    __shared__ float tileD[CONTRACTION_TILE_SIZE][CONTRACTION_TILE_SIZE];

    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int LM = L * M;

    float sum = 0.0f;
    int numTiles = (LM + CONTRACTION_TILE_SIZE - 1) / CONTRACTION_TILE_SIZE;

    for (int tile = 0; tile < numTiles; ++tile) {
        int tiledColT = tile * CONTRACTION_TILE_SIZE + tx;
        int tiledRowD = tile * CONTRACTION_TILE_SIZE + ty;

        if (row < IJ && tiledColT < LM) {
            tileT[ty][tx] = T[row * LM + tiledColT];
        } else {
            tileT[ty][tx] = 0.0f;
        }

        if (tiledRowD < LM && n < N) {
            tileD[ty][tx] = D[tiledRowD * N + n];
        } else {
            tileD[ty][tx] = 0.0f;
        }

        __syncthreads();

        for (int q = 0; q < CONTRACTION_TILE_SIZE; ++q) {
            sum += tileT[ty][q] * tileD[q][tx];
        }

        __syncthreads();
    }

    if (row < IJ && n < N) {
        C[row * N + n] = sum;
    }
}

void tiltedContractionGpu(const float* A, const float* B, const float* D,
                          float* C, int I, int J, int K, int L, int M, int N) {
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

    tiltedContractionDevice(dA, dB, dT, dD, dC, I, J, K, L, M, N);

    CUDA_CHECK(cudaMemcpy(C, dC, bytesC, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dT));
    CUDA_CHECK(cudaFree(dD));
    CUDA_CHECK(cudaFree(dC));
}

void tiltedContractionDevice(const float* dA, const float* dB, float* dT,
                             const float* dD, float* dC, int I, int J, int K,
                             int L, int M, int N) {
    order3contractionDevice(dA, dB, dT, I, J, K, L, M);

    dim3 threadsPerBlock(CONTRACTION_TILE_SIZE, CONTRACTION_TILE_SIZE);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (I * J + threadsPerBlock.y - 1) / threadsPerBlock.y);

    tiledSecondContractionKernel<<<blocksPerGrid, threadsPerBlock>>>(
        dT, dD, dC, I * J, L, M, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void tiltedContractionCpu(const float* A, const float* B, const float* D,
                          float* C, int I, int J, int K, int L, int M, int N) {
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

    for (int row = 0; row < I * J; ++row) {
        for (int n = 0; n < N; ++n) {
            float value = 0.0f;
            for (int q = 0; q < L * M; ++q) {
                value += T[row * L * M + q] * D[q * N + n];
            }
            C[row * N + n] = value;
        }
    }
}

bool tiltedContractionOutputsMatch(const std::vector<float>& a,
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
