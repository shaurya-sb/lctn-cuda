#pragma once

// Contracts A[I, J, K] with B[K, L, M] over K, producing C[I, J, L, M].
void order3contraction(const float* A, const float* B, float* C, int I, int J,
                       int K, int L, int M);

void order3contractionDevice(const float* dA, const float* dB, float* dC,
                             int I, int J, int K, int L, int M);

void naiveOrder3ContractionGpu(const float* A, const float* B, float* C, int I,
                               int J, int K, int L, int M);

void naiveOrder3ContractionDevice(const float* dA, const float* dB, float* dC,
                                  int I, int J, int K, int L, int M);

void naiveOrder3ContractionCpu(const float* A, const float* B, float* C, int I,
                               int J, int K, int L, int M);
