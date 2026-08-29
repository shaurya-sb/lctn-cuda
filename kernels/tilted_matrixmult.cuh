#pragma once

#include <vector>

void tiltedMatrixMultiplicationGpu(const float* A, const float* B, float* C,
                                   int M, int N, int K);

void tiltedMatrixMultiplicationDevice(const float* dA, const float* dB,
                                      float* dC, int M, int N, int K);

void tiltedMatrixMultiplicationCpu(const float* A, const float* B, float* C,
                                   int M, int N, int K);

bool tiltedMatricesMatch(const std::vector<float>& a,
                         const std::vector<float>& b);
