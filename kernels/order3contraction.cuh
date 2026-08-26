#pragma once

// Contracts A[I, J, K] with B[K, L, M] over K, producing C[I, J, L, M].
void order3contraction(const float* A, const float* B, float* C, int I, int J,
                       int K, int L, int M);
