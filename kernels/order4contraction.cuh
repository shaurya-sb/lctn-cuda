#pragma once

#include <vector>

void order4contractionGpu(const float* A, const float* B, float* C, int I,
                          int J, int K, int L, int M, int N);

void order4contractionCpu(const float* A, const float* B, float* C, int I,
                          int J, int K, int L, int M, int N);

bool order4OutputsMatch(const std::vector<float>& a,
                        const std::vector<float>& b);
