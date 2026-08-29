#pragma once

#include <vector>

void matrixMultiplicationGpu(const float* A, const float* B, float* C, int M,
                             int N, int K);

void matrixMultiplicationDevice(const float* dA, const float* dB, float* dC,
                                int M, int N, int K);

void matrixMultiplicationCpu(const float* A, const float* B, float* C, int M,
                             int N, int K);

bool matricesMatch(const std::vector<float>& a, const std::vector<float>& b);
