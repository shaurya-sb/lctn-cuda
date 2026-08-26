#include <cuda_runtime.h>

#include <cmath>
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

// GPU code: this function runs on the GPU.
__global__ void naiveMatrixMultiplyKernel(const float* M, const float* N,
                                          float* P, int width) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= width || col >= width) {
        return;
    }

    float value = 0.0f;
    for (int i = 0; i < width; ++i) {
        value += M[row * width + i] * N[i * width + col];
    }

    P[row * width + col] = value;
}

// CPU code: this function runs on the CPU and uses CUDA API calls to talk to
// the GPU. In Colab, these calls talk to Colab's remote NVIDIA GPU.
void matrixMultiplicationGpu(const float* M, const float* N, float* P,
                             int width) {
    size_t bytes = static_cast<size_t>(width) * width * sizeof(float);

    float* d_M = nullptr;
    float* d_N = nullptr;
    float* d_P = nullptr;

    CUDA_CHECK(cudaMalloc(&d_M, bytes));
    CUDA_CHECK(cudaMalloc(&d_N, bytes));
    CUDA_CHECK(cudaMalloc(&d_P, bytes));

    // Host RAM -> GPU RAM.
    CUDA_CHECK(cudaMemcpy(d_M, M, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_N, N, bytes, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (width + threadsPerBlock.y - 1) / threadsPerBlock.y);

    naiveMatrixMultiplyKernel<<<blocksPerGrid, threadsPerBlock>>>(d_M, d_N, d_P,
                                                                  width);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // GPU RAM -> host RAM.
    CUDA_CHECK(cudaMemcpy(P, d_P, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_M));
    CUDA_CHECK(cudaFree(d_N));
    CUDA_CHECK(cudaFree(d_P));
}

void matrixMultiplicationCpu(const float* M, const float* N, float* P,
                             int width) {
    for (int row = 0; row < width; ++row) {
        for (int col = 0; col < width; ++col) {
            float value = 0.0f;
            for (int i = 0; i < width; ++i) {
                value += M[row * width + i] * N[i * width + col];
            }
            P[row * width + col] = value;
        }
    }
}

bool matricesMatch(const std::vector<float>& a, const std::vector<float>& b) {
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

int main() {
    constexpr int width = 4;
    std::vector<float> M(width * width);
    std::vector<float> N(width * width);
    std::vector<float> gpuResult(width * width);
    std::vector<float> cpuResult(width * width);

    for (int row = 0; row < width; ++row) {
        for (int col = 0; col < width; ++col) {
            M[row * width + col] = static_cast<float>(row + 1);
            N[row * width + col] = static_cast<float>(col + 1);
        }
    }

    matrixMultiplicationGpu(M.data(), N.data(), gpuResult.data(), width);
    matrixMultiplicationCpu(M.data(), N.data(), cpuResult.data(), width);

    std::cout << "GPU result:" << std::endl;
    for (int row = 0; row < width; ++row) {
        for (int col = 0; col < width; ++col) {
            std::cout << gpuResult[row * width + col] << '\t';
        }
        std::cout << '\n';
    }

    std::cout << "\nMatches CPU result: "
              << (matricesMatch(gpuResult, cpuResult) ? "yes" : "no")
              << std::endl;

    return 0;
}
