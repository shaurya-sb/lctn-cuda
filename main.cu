#include "kernels/matrix_multiply.cuh"
#include "kernels/naivenetworkcontraction.cuh"
#include "kernels/order3contraction.cuh"
#include "kernels/order4contraction.cuh"
#include "kernels/tilted_matrixmult.cuh"
#include "kernels/tiltedcontraction.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

struct BenchmarkResult {
    std::string name;
    std::string problem;
    double cpuMs;
    double gpuMs;
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
              << std::setw(10) << "Correct" << '\n';
    std::cout << std::string(98, '-') << '\n';

    for (const BenchmarkResult& result : results) {
        double speedup = result.gpuMs > 0.0 ? result.cpuMs / result.gpuMs : 0.0;
        std::cout << std::left << std::setw(34) << result.name
                  << std::setw(18) << result.problem << std::right
                  << std::setw(12) << std::fixed << std::setprecision(3)
                  << result.cpuMs << std::setw(12) << result.gpuMs
                  << std::setw(12) << speedup << std::setw(10)
                  << (result.correct ? "yes" : "no") << '\n';
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

        results.push_back({"Matrix multiply naive", "256x256x256", cpuMs,
                           gpuMs, outputsMatch(cpu, gpu)});

        gpu.assign(gpu.size(), 0.0f);
        gpuMs = averageGpuMs([&] {
            tiltedMatrixMultiplicationGpu(A.data(), B.data(), gpu.data(), rows,
                                          cols, inner);
        });

        results.push_back({"Matrix multiply tiled", "256x256x256", cpuMs,
                           gpuMs, outputsMatch(cpu, gpu)});
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

        results.push_back({"Order-3 contraction naive", "16,16,64,16,16",
                           cpuMs, gpuMs, outputsMatch(cpu, gpu)});

        gpu.assign(gpu.size(), 0.0f);
        gpuMs = averageGpuMs([&] {
            order3contraction(A.data(), B.data(), gpu.data(), I, J, K, L, M);
        });

        results.push_back({"Order-3 contraction tiled", "16,16,64,16,16",
                           cpuMs, gpuMs, outputsMatch(cpu, gpu)});
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
                           cpuMs, gpuMs, outputsMatch(cpu, gpu)});
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

        results.push_back({"Network contraction naive", "16,16,64,16,16,64",
                           cpuMs, gpuMs, outputsMatch(cpu, gpu)});

        gpu.assign(gpu.size(), 0.0f);
        gpuMs = averageGpuMs([&] {
            tiltedContractionGpu(A.data(), B.data(), D.data(), gpu.data(), I, J,
                                 K, L, M, N);
        });

        results.push_back({"Network contraction tiled", "16,16,64,16,16,64",
                           cpuMs, gpuMs, outputsMatch(cpu, gpu)});
    }

    printResults(results);

    return 0;
}
