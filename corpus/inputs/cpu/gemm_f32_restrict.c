/* Diagnostic Case: gemm_f32_restrict.c
 * Purpose: Restrict-annotated FP32 GEMM with 32-byte alignment assumptions.
 * Contract: Proves how Clang/LLVM consumes alias-freedom and alignment preconditions.
 */
#include <stddef.h>

void gemm_f32_restrict(
    int M, int N, int K,
    float alpha,
    const float * __restrict__ A, int lda,
    const float * __restrict__ B, int ldb,
    float beta,
    float * __restrict__ C, int ldc) {
  const float * __restrict__ a_ptr = (const float *)__builtin_assume_aligned(A, 32);
  const float * __restrict__ b_ptr = (const float *)__builtin_assume_aligned(B, 32);
  float * __restrict__ c_ptr = (float *)__builtin_assume_aligned(C, 32);

  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float sum = 0.0f;
      for (int k = 0; k < K; ++k) {
        sum += a_ptr[i * lda + k] * b_ptr[k * ldb + j];
      }
      c_ptr[i * ldc + j] = alpha * sum + beta * c_ptr[i * ldc + j];
    }
  }
}
