#include "kernels/matrix_multiply.cuh"

#include <iostream>
#include <vector>

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

    matrixMultiplicationGpu(M.data(), N.data(), gpuResult.data(), width, width,
                            width);
    matrixMultiplicationCpu(M.data(), N.data(), cpuResult.data(), width, width,
                            width);

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
