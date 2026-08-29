#include "kernels/matrix_multiply.cuh"
#include "kernels/naivenetworkcontraction.cuh"
#include "kernels/order3contraction.cuh"
#include "kernels/order4contraction.cuh"
#include "kernels/tilted_matrixmult.cuh"
#include "kernels/tiltedcontraction.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
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

struct BenchmarkResult {
    std::string name;
    std::string problem;
    double cpuMs;
    double gpuMs;
    double gpuVsNaive;
    bool correct;
};

double timeMs(const std::function<void()>& fn) {
    auto start = std::chrono::high_resolution_clock::now();
    fn();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

double averageGpuMs(const std::function<void()>& fn, int repetitions = 5) {
    fn();

    double total = 0.0;
    for (int i = 0; i < repetitions; ++i) {
        total += timeMs(fn);
    }
    return total / repetitions;
}

double averageKernelMs(const std::function<void()>& fn, int repetitions = 20) {
    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    fn();

    float total = 0.0f;
    for (int i = 0; i < repetitions; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        fn();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsed = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
        total += elapsed;
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return total / repetitions;
}

bool outputsMatch(const std::vector<float>& a, const std::vector<float>& b,
                  float absoluteTolerance = 1.0e-3f,
                  float relativeTolerance = 1.0e-4f) {
    if (a.size() != b.size()) {
        return false;
    }

    for (size_t i = 0; i < a.size(); ++i) {
        float scale = std::max(1.0f, std::fabs(a[i]));
        if (std::fabs(a[i] - b[i]) >
            absoluteTolerance + relativeTolerance * scale) {
            return false;
        }
    }

    return true;
}

void fillInput(std::vector<float>& values) {
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = static_cast<float>((i % 17) + 1) / 17.0f;
    }
}

void printResults(const std::vector<BenchmarkResult>& results) {
    std::cout << std::left << std::setw(34) << "Kernel" << std::setw(18)
              << "Problem" << std::right << std::setw(12) << "CPU ms"
              << std::setw(12) << "GPU ms" << std::setw(12) << "Speedup"
              << std::setw(18) << "kernel vs naive" << std::setw(10)
              << "Correct" << '\n';
    std::cout << std::string(116, '-') << '\n';

    for (const BenchmarkResult& result : results) {
        double speedup = result.gpuMs > 0.0 ? result.cpuMs / result.gpuMs : 0.0;
        std::cout << std::left << std::setw(34) << result.name
                  << std::setw(18) << result.problem << std::right
                  << std::setw(12) << std::fixed << std::setprecision(3)
                  << result.cpuMs << std::setw(12) << result.gpuMs
                  << std::setw(12) << speedup;
        if (result.gpuVsNaive > 0.0) {
            std::cout << std::setw(18) << result.gpuVsNaive;
        } else {
            std::cout << std::setw(18) << "-";
        }
        std::cout << std::setw(10) << (result.correct ? "yes" : "no") << '\n';
    }
}

int main() {
    std::vector<BenchmarkResult> results;

    {
        constexpr int rows = 256;
        constexpr int cols = 256;
        constexpr int inner = 256;
        std::vector<float> A(rows * inner);
        std::vector<float> B(inner * cols);
        std::vector<float> cpu(rows * cols);
        std::vector<float> gpu(rows * cols);

        fillInput(A);
        fillInput(B);

        double cpuMs =
            timeMs([&] { matrixMultiplicationCpu(A.data(), B.data(), cpu.data(),
                                                 rows, cols, inner); });
        double gpuMs = averageGpuMs([&] {
            matrixMultiplicationGpu(A.data(), B.data(), gpu.data(), rows, cols,
                                    inner);
        });
        size_t bytesA = static_cast<size_t>(rows) * inner * sizeof(float);
        size_t bytesB = static_cast<size_t>(inner) * cols * sizeof(float);
        size_t bytesC = static_cast<size_t>(rows) * cols * sizeof(float);
        float* dA = nullptr;
        float* dB = nullptr;
        float* dC = nullptr;
        CUDA_CHECK(cudaMalloc(&dA, bytesA));
        CUDA_CHECK(cudaMalloc(&dB, bytesB));
        CUDA_CHECK(cudaMalloc(&dC, bytesC));
        CUDA_CHECK(cudaMemcpy(dA, A.data(), bytesA, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, B.data(), bytesB, cudaMemcpyHostToDevice));
        double naiveKernelMs = averageKernelMs([&] {
            matrixMultiplicationDevice(dA, dB, dC, rows, cols, inner);
        });

        results.push_back({"Matrix multiply naive", "256x256x256", cpuMs,
                           gpuMs, 1.0, outputsMatch(cpu, gpu)});

        gpu.assign(gpu.size(), 0.0f);
        gpuMs = averageGpuMs([&] {
            tiltedMatrixMultiplicationGpu(A.data(), B.data(), gpu.data(), rows,
                                          cols, inner);
        });
        double tiledKernelMs = averageKernelMs([&] {
            tiltedMatrixMultiplicationDevice(dA, dB, dC, rows, cols, inner);
        });

        CUDA_CHECK(cudaFree(dA));
        CUDA_CHECK(cudaFree(dB));
        CUDA_CHECK(cudaFree(dC));

        results.push_back({"Matrix multiply tiled", "256x256x256", cpuMs,
                           gpuMs, naiveKernelMs / tiledKernelMs,
                           outputsMatch(cpu, gpu)});
    }

    {
        constexpr int I = 16;
        constexpr int J = 16;
        constexpr int K = 64;
        constexpr int L = 16;
        constexpr int M = 16;
        std::vector<float> A(I * J * K);
        std::vector<float> B(K * L * M);
        std::vector<float> cpu(I * J * L * M);
        std::vector<float> gpu(I * J * L * M);

        fillInput(A);
        fillInput(B);

        double cpuMs = timeMs([&] {
            naiveOrder3ContractionCpu(A.data(), B.data(), cpu.data(), I, J, K,
                                      L, M);
        });
        double gpuMs = averageGpuMs([&] {
            naiveOrder3ContractionGpu(A.data(), B.data(), gpu.data(), I, J, K,
                                      L, M);
        });
        size_t bytesA = static_cast<size_t>(I) * J * K * sizeof(float);
        size_t bytesB = static_cast<size_t>(K) * L * M * sizeof(float);
        size_t bytesC = static_cast<size_t>(I) * J * L * M * sizeof(float);
        float* dA = nullptr;
        float* dB = nullptr;
        float* dC = nullptr;
        CUDA_CHECK(cudaMalloc(&dA, bytesA));
        CUDA_CHECK(cudaMalloc(&dB, bytesB));
        CUDA_CHECK(cudaMalloc(&dC, bytesC));
        CUDA_CHECK(cudaMemcpy(dA, A.data(), bytesA, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, B.data(), bytesB, cudaMemcpyHostToDevice));
        double naiveKernelMs = averageKernelMs([&] {
            naiveOrder3ContractionDevice(dA, dB, dC, I, J, K, L, M);
        });

        results.push_back({"Order-3 contraction naive", "16,16,64,16,16",
                           cpuMs, gpuMs, 1.0, outputsMatch(cpu, gpu)});

        gpu.assign(gpu.size(), 0.0f);
        gpuMs = averageGpuMs([&] {
            order3contraction(A.data(), B.data(), gpu.data(), I, J, K, L, M);
        });
        double tiledKernelMs = averageKernelMs([&] {
            order3contractionDevice(dA, dB, dC, I, J, K, L, M);
        });

        CUDA_CHECK(cudaFree(dA));
        CUDA_CHECK(cudaFree(dB));
        CUDA_CHECK(cudaFree(dC));

        results.push_back({"Order-3 contraction tiled", "16,16,64,16,16",
                           cpuMs, gpuMs, naiveKernelMs / tiledKernelMs,
                           outputsMatch(cpu, gpu)});
    }

    {
        constexpr int I = 8;
        constexpr int J = 8;
        constexpr int K = 32;
        constexpr int L = 16;
        constexpr int M = 8;
        constexpr int N = 32;
        std::vector<float> A(I * J * K * L);
        std::vector<float> B(K * L * M * N);
        std::vector<float> cpu(I * J * M * N);
        std::vector<float> gpu(I * J * M * N);

        fillInput(A);
        fillInput(B);

        double cpuMs = timeMs([&] {
            order4contractionCpu(A.data(), B.data(), cpu.data(), I, J, K, L, M,
                                 N);
        });
        double gpuMs = averageGpuMs([&] {
            order4contractionGpu(A.data(), B.data(), gpu.data(), I, J, K, L, M,
                                 N);
        });

        results.push_back({"Order-4 contraction naive", "8,8,32,16,8,32",
                           cpuMs, gpuMs, 0.0, outputsMatch(cpu, gpu)});
    }

    {
        constexpr int I = 16;
        constexpr int J = 16;
        constexpr int K = 64;
        constexpr int L = 16;
        constexpr int M = 16;
        constexpr int N = 64;
        std::vector<float> A(I * J * K);
        std::vector<float> B(K * L * M);
        std::vector<float> D(L * M * N);
        std::vector<float> cpu(I * J * N);
        std::vector<float> gpu(I * J * N);

        fillInput(A);
        fillInput(B);
        fillInput(D);

        double cpuMs = timeMs([&] {
            naiveNetworkContractionCpu(A.data(), B.data(), D.data(), cpu.data(),
                                       I, J, K, L, M, N);
        });
        double gpuMs = averageGpuMs([&] {
            naiveNetworkContractionGpu(A.data(), B.data(), D.data(), gpu.data(),
                                       I, J, K, L, M, N);
        });
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
        CUDA_CHECK(cudaMemcpy(dA, A.data(), bytesA, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, B.data(), bytesB, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dD, D.data(), bytesD, cudaMemcpyHostToDevice));
        double naiveKernelMs = averageKernelMs([&] {
            naiveNetworkContractionDevice(dA, dB, dT, dD, dC, I, J, K, L, M,
                                          N);
        });

        results.push_back({"Network contraction naive", "16,16,64,16,16,64",
                           cpuMs, gpuMs, 1.0, outputsMatch(cpu, gpu)});

        gpu.assign(gpu.size(), 0.0f);
        gpuMs = averageGpuMs([&] {
            tiltedContractionGpu(A.data(), B.data(), D.data(), gpu.data(), I, J,
                                 K, L, M, N);
        });
        double tiledKernelMs = averageKernelMs([&] {
            tiltedContractionDevice(dA, dB, dT, dD, dC, I, J, K, L, M, N);
        });

        CUDA_CHECK(cudaFree(dA));
        CUDA_CHECK(cudaFree(dB));
        CUDA_CHECK(cudaFree(dT));
        CUDA_CHECK(cudaFree(dD));
        CUDA_CHECK(cudaFree(dC));

        results.push_back({"Network contraction tiled", "16,16,64,16,16,64",
                           cpuMs, gpuMs, naiveKernelMs / tiledKernelMs,
                           outputsMatch(cpu, gpu)});
    }

    printResults(results);

    return 0;
}
