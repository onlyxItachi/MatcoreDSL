// Diagnostic Case: gemm_f32_epilogue_relu.cpp
// Purpose: Evaluates multi-op fusion of GEMM followed immediately by ReLU activation.
#include <cmath>

extern "C" void gemm_f32_epilogue_relu(
    int M, int N, int K,
    const float * __restrict__ A,
    const float * __restrict__ B,
    float * __restrict__ C) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float sum = 0.0f;
      for (int k = 0; k < K; ++k) {
        sum += A[i * K + k] * B[k * N + j];
      }
      // Fused ReLU epilogue
      C[i * N + j] = (sum > 0.0f) ? sum : 0.0f;
    }
  }
}
