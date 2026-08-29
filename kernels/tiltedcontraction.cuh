#pragma once

#include <vector>

void tiltedContractionGpu(const float* A, const float* B, const float* D,
                          float* C, int I, int J, int K, int L, int M, int N);

void tiltedContractionCpu(const float* A, const float* B, const float* D,
                          float* C, int I, int J, int K, int L, int M, int N);

bool tiltedContractionOutputsMatch(const std::vector<float>& a,
                                   const std::vector<float>& b);
