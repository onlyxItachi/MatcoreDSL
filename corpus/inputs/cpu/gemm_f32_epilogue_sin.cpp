// Diagnostic Case: gemm_f32_epilogue_sin.cpp
// Purpose: Evaluates transcendental elementwise map fusion (mdsl.sin / std::sin) on GEMM output.
#include <cmath>

extern "C" void gemm_f32_epilogue_sin(
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
      // Fused SIN elementwise epilogue
      C[i * N + j] = std::sin(sum);
    }
  }
}
