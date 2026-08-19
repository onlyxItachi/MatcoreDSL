#include <immintrin.h>

// A. SEPARATE EPILOGUE: Accumulate -> Store C -> Reload C -> ReLU -> Store C
__declspec(noinline)
void microkernel_separate_relu(
    int K,
    const float * __restrict__ A,
    const float * __restrict__ B,
    float * __restrict__ C,
    int ldc) {
    
    // 8 accumulators for 16x4 tile (2 YMM x 4 cols)
    __m256 acc0 = _mm256_setzero_ps(); __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps(); __m256 acc3 = _mm256_setzero_ps();
    __m256 acc4 = _mm256_setzero_ps(); __m256 acc5 = _mm256_setzero_ps();
    __m256 acc6 = _mm256_setzero_ps(); __m256 acc7 = _mm256_setzero_ps();

    for (int k = 0; k < K; ++k) {
        __m256 a0 = _mm256_loadu_ps(&A[k * 16 + 0]);
        __m256 a1 = _mm256_loadu_ps(&A[k * 16 + 8]);

        __m256 b0 = _mm256_set1_ps(B[k * 4 + 0]);
        acc0 = _mm256_fmadd_ps(a0, b0, acc0);
        acc1 = _mm256_fmadd_ps(a1, b0, acc1);

        __m256 b1 = _mm256_set1_ps(B[k * 4 + 1]);
        acc2 = _mm256_fmadd_ps(a0, b1, acc2);
        acc3 = _mm256_fmadd_ps(a1, b1, acc3);

        __m256 b2 = _mm256_set1_ps(B[k * 4 + 2]);
        acc4 = _mm256_fmadd_ps(a0, b2, acc4);
        acc5 = _mm256_fmadd_ps(a1, b2, acc5);

        __m256 b3 = _mm256_set1_ps(B[k * 4 + 3]);
        acc6 = _mm256_fmadd_ps(a0, b3, acc6);
        acc7 = _mm256_fmadd_ps(a1, b3, acc7);
    }

    // 1. First Store to C (GEMM completion)
    _mm256_storeu_ps(&C[0 * ldc + 0], acc0); _mm256_storeu_ps(&C[0 * ldc + 8], acc1);
    _mm256_storeu_ps(&C[1 * ldc + 0], acc2); _mm256_storeu_ps(&C[1 * ldc + 8], acc3);
    _mm256_storeu_ps(&C[2 * ldc + 0], acc4); _mm256_storeu_ps(&C[2 * ldc + 8], acc5);
    _mm256_storeu_ps(&C[3 * ldc + 0], acc6); _mm256_storeu_ps(&C[3 * ldc + 8], acc7);

    // 2. Separate ReLU pass: Reload C, Apply Max, Store C
    __m256 zero = _mm256_setzero_ps();
    __m256 r0 = _mm256_max_ps(_mm256_loadu_ps(&C[0 * ldc + 0]), zero);
    __m256 r1 = _mm256_max_ps(_mm256_loadu_ps(&C[0 * ldc + 8]), zero);
    __m256 r2 = _mm256_max_ps(_mm256_loadu_ps(&C[1 * ldc + 0]), zero);
    __m256 r3 = _mm256_max_ps(_mm256_loadu_ps(&C[1 * ldc + 8]), zero);
    __m256 r4 = _mm256_max_ps(_mm256_loadu_ps(&C[2 * ldc + 0]), zero);
    __m256 r5 = _mm256_max_ps(_mm256_loadu_ps(&C[2 * ldc + 8]), zero);
    __m256 r6 = _mm256_max_ps(_mm256_loadu_ps(&C[3 * ldc + 0]), zero);
    __m256 r7 = _mm256_max_ps(_mm256_loadu_ps(&C[3 * ldc + 8]), zero);

    _mm256_storeu_ps(&C[0 * ldc + 0], r0); _mm256_storeu_ps(&C[0 * ldc + 8], r1);
    _mm256_storeu_ps(&C[1 * ldc + 0], r2); _mm256_storeu_ps(&C[1 * ldc + 8], r3);
    _mm256_storeu_ps(&C[2 * ldc + 0], r4); _mm256_storeu_ps(&C[2 * ldc + 8], r5);
    _mm256_storeu_ps(&C[3 * ldc + 0], r6); _mm256_storeu_ps(&C[3 * ldc + 8], r7);
}

// B. TRUE FUSED EPILOGUE: Accumulate in registers -> ReLU directly in registers -> Single Final Store to C
__declspec(noinline)
void microkernel_true_fused_relu(
    int K,
    const float * __restrict__ A,
    const float * __restrict__ B,
    float * __restrict__ C,
    int ldc) {
    
    // 8 accumulators for 16x4 tile
    __m256 acc0 = _mm256_setzero_ps(); __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps(); __m256 acc3 = _mm256_setzero_ps();
    __m256 acc4 = _mm256_setzero_ps(); __m256 acc5 = _mm256_setzero_ps();
    __m256 acc6 = _mm256_setzero_ps(); __m256 acc7 = _mm256_setzero_ps();

    for (int k = 0; k < K; ++k) {
        __m256 a0 = _mm256_loadu_ps(&A[k * 16 + 0]);
        __m256 a1 = _mm256_loadu_ps(&A[k * 16 + 8]);

        __m256 b0 = _mm256_set1_ps(B[k * 4 + 0]);
        acc0 = _mm256_fmadd_ps(a0, b0, acc0);
        acc1 = _mm256_fmadd_ps(a1, b0, acc1);

        __m256 b1 = _mm256_set1_ps(B[k * 4 + 1]);
        acc2 = _mm256_fmadd_ps(a0, b1, acc2);
        acc3 = _mm256_fmadd_ps(a1, b1, acc3);

        __m256 b2 = _mm256_set1_ps(B[k * 4 + 2]);
        acc4 = _mm256_fmadd_ps(a0, b2, acc4);
        acc5 = _mm256_fmadd_ps(a1, b2, acc5);

        __m256 b3 = _mm256_set1_ps(B[k * 4 + 3]);
        acc6 = _mm256_fmadd_ps(a0, b3, acc6);
        acc7 = _mm256_fmadd_ps(a1, b3, acc7);
    }

    // Direct in-register ReLU before store (ZERO intermediate C store, ZERO C reload)
    __m256 zero = _mm256_setzero_ps();
    acc0 = _mm256_max_ps(acc0, zero);
    acc1 = _mm256_max_ps(acc1, zero);
    acc2 = _mm256_max_ps(acc2, zero);
    acc3 = _mm256_max_ps(acc3, zero);
    acc4 = _mm256_max_ps(acc4, zero);
    acc5 = _mm256_max_ps(acc5, zero);
    acc6 = _mm256_max_ps(acc6, zero);
    acc7 = _mm256_max_ps(acc7, zero);

    // Single and only Store to C
    _mm256_storeu_ps(&C[0 * ldc + 0], acc0); _mm256_storeu_ps(&C[0 * ldc + 8], acc1);
    _mm256_storeu_ps(&C[1 * ldc + 0], acc2); _mm256_storeu_ps(&C[1 * ldc + 8], acc3);
    _mm256_storeu_ps(&C[2 * ldc + 0], acc4); _mm256_storeu_ps(&C[2 * ldc + 8], acc5);
    _mm256_storeu_ps(&C[3 * ldc + 0], acc6); _mm256_storeu_ps(&C[3 * ldc + 8], acc7);
}
