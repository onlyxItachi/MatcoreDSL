/* Diagnostic Case: gemm_f32_naive.c
 * Purpose: Baseline unannotated C99 row-major FP32 GEMM without alias/alignment hints.
 * Contract: C[M, N] = alpha * A[M, K] * B[K, N] + beta * C[M, N]
 */
void gemm_f32_naive(
    int M, int N, int K,
    float alpha,
    const float *A, int lda,
    const float *B, int ldb,
    float beta,
    float *C, int ldc) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float sum = 0.0f;
      for (int k = 0; k < K; ++k) {
        sum += A[i * lda + k] * B[k * ldb + j];
      }
      C[i * ldc + j] = alpha * sum + beta * C[i * ldc + j];
    }
  }
}
