#pragma once

#include <vector>

void naiveNetworkContractionGpu(const float* A, const float* B, const float* D,
                                float* C, int I, int J, int K, int L, int M,
                                int N);

void naiveNetworkContractionCpu(const float* A, const float* B, const float* D,
                                float* C, int I, int J, int K, int L, int M,
                                int N);

bool naiveNetworkOutputsMatch(const std::vector<float>& a,
                              const std::vector<float>& b);
