#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include <windows.h>
#include <vector>

double get_time_sec() {
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart / (double)freq.QuadPart;
}

// 1. Tile 8x4 (4 YMM Accumulators: 1 YMM x 4 cols)
__declspec(noinline)
void ukernel_8x4(int K, const float * __restrict__ A, const float * __restrict__ B, float * __restrict__ C, int ldc) {
    __m256 acc0 = _mm256_setzero_ps(); __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps(); __m256 acc3 = _mm256_setzero_ps();
    for (int k = 0; k < K; ++k) {
        __m256 a = _mm256_loadu_ps(&A[k * 8]);
        acc0 = _mm256_fmadd_ps(a, _mm256_set1_ps(B[k * 4 + 0]), acc0);
        acc1 = _mm256_fmadd_ps(a, _mm256_set1_ps(B[k * 4 + 1]), acc1);
        acc2 = _mm256_fmadd_ps(a, _mm256_set1_ps(B[k * 4 + 2]), acc2);
        acc3 = _mm256_fmadd_ps(a, _mm256_set1_ps(B[k * 4 + 3]), acc3);
    }
    _mm256_storeu_ps(&C[0 * ldc], acc0); _mm256_storeu_ps(&C[1 * ldc], acc1);
    _mm256_storeu_ps(&C[2 * ldc], acc2); _mm256_storeu_ps(&C[3 * ldc], acc3);
}

// 2. Tile 16x4 (8 YMM Accumulators: 2 YMM x 4 cols)
__declspec(noinline)
void ukernel_16x4(int K, const float * __restrict__ A, const float * __restrict__ B, float * __restrict__ C, int ldc) {
    __m256 acc0 = _mm256_setzero_ps(); __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps(); __m256 acc3 = _mm256_setzero_ps();
    __m256 acc4 = _mm256_setzero_ps(); __m256 acc5 = _mm256_setzero_ps();
    __m256 acc6 = _mm256_setzero_ps(); __m256 acc7 = _mm256_setzero_ps();
    for (int k = 0; k < K; ++k) {
        __m256 a0 = _mm256_loadu_ps(&A[k * 16 + 0]);
        __m256 a1 = _mm256_loadu_ps(&A[k * 16 + 8]);
        __m256 b0 = _mm256_set1_ps(B[k * 4 + 0]);
        acc0 = _mm256_fmadd_ps(a0, b0, acc0); acc1 = _mm256_fmadd_ps(a1, b0, acc1);
        __m256 b1 = _mm256_set1_ps(B[k * 4 + 1]);
        acc2 = _mm256_fmadd_ps(a0, b1, acc2); acc3 = _mm256_fmadd_ps(a1, b1, acc3);
        __m256 b2 = _mm256_set1_ps(B[k * 4 + 2]);
        acc4 = _mm256_fmadd_ps(a0, b2, acc4); acc5 = _mm256_fmadd_ps(a1, b2, acc5);
        __m256 b3 = _mm256_set1_ps(B[k * 4 + 3]);
        acc6 = _mm256_fmadd_ps(a0, b3, acc6); acc7 = _mm256_fmadd_ps(a1, b3, acc7);
    }
    _mm256_storeu_ps(&C[0 * ldc + 0], acc0); _mm256_storeu_ps(&C[0 * ldc + 8], acc1);
    _mm256_storeu_ps(&C[1 * ldc + 0], acc2); _mm256_storeu_ps(&C[1 * ldc + 8], acc3);
    _mm256_storeu_ps(&C[2 * ldc + 0], acc4); _mm256_storeu_ps(&C[2 * ldc + 8], acc5);
    _mm256_storeu_ps(&C[3 * ldc + 0], acc6); _mm256_storeu_ps(&C[3 * ldc + 8], acc7);
}

// 3. Tile 16x6 (12 YMM Accumulators: 2 YMM x 6 cols - OpenBLAS Haswell style)
__declspec(noinline)
void ukernel_16x6(int K, const float * __restrict__ A, const float * __restrict__ B, float * __restrict__ C, int ldc) {
    __m256 c00 = _mm256_setzero_ps(); __m256 c10 = _mm256_setzero_ps();
    __m256 c01 = _mm256_setzero_ps(); __m256 c11 = _mm256_setzero_ps();
    __m256 c02 = _mm256_setzero_ps(); __m256 c12 = _mm256_setzero_ps();
    __m256 c03 = _mm256_setzero_ps(); __m256 c13 = _mm256_setzero_ps();
    __m256 c04 = _mm256_setzero_ps(); __m256 c14 = _mm256_setzero_ps();
    __m256 c05 = _mm256_setzero_ps(); __m256 c15 = _mm256_setzero_ps();

    for (int k = 0; k < K; ++k) {
        __m256 a0 = _mm256_loadu_ps(&A[k * 16 + 0]);
        __m256 a1 = _mm256_loadu_ps(&A[k * 16 + 8]);

        __m256 b0 = _mm256_set1_ps(B[k * 6 + 0]);
        c00 = _mm256_fmadd_ps(a0, b0, c00); c10 = _mm256_fmadd_ps(a1, b0, c10);

        __m256 b1 = _mm256_set1_ps(B[k * 6 + 1]);
        c01 = _mm256_fmadd_ps(a0, b1, c01); c11 = _mm256_fmadd_ps(a1, b1, c11);

        __m256 b2 = _mm256_set1_ps(B[k * 6 + 2]);
        c02 = _mm256_fmadd_ps(a0, b2, c02); c12 = _mm256_fmadd_ps(a1, b2, c12);

        __m256 b3 = _mm256_set1_ps(B[k * 6 + 3]);
        c03 = _mm256_fmadd_ps(a0, b3, c03); c13 = _mm256_fmadd_ps(a1, b3, c13);

        __m256 b4 = _mm256_set1_ps(B[k * 6 + 4]);
        c04 = _mm256_fmadd_ps(a0, b4, c04); c14 = _mm256_fmadd_ps(a1, b4, c14);

        __m256 b5 = _mm256_set1_ps(B[k * 6 + 5]);
        c05 = _mm256_fmadd_ps(a0, b5, c05); c15 = _mm256_fmadd_ps(a1, b5, c15);
    }

    _mm256_storeu_ps(&C[0 * ldc + 0], c00); _mm256_storeu_ps(&C[0 * ldc + 8], c10);
    _mm256_storeu_ps(&C[1 * ldc + 0], c01); _mm256_storeu_ps(&C[1 * ldc + 8], c11);
    _mm256_storeu_ps(&C[2 * ldc + 0], c02); _mm256_storeu_ps(&C[2 * ldc + 8], c12);
    _mm256_storeu_ps(&C[3 * ldc + 0], c03); _mm256_storeu_ps(&C[3 * ldc + 8], c13);
    _mm256_storeu_ps(&C[4 * ldc + 0], c04); _mm256_storeu_ps(&C[4 * ldc + 8], c14);
    _mm256_storeu_ps(&C[5 * ldc + 0], c05); _mm256_storeu_ps(&C[5 * ldc + 8], c15);
}

// 4. Tile 24x4 (12 YMM Accumulators: 3 YMM x 4 cols)
__declspec(noinline)
void ukernel_24x4(int K, const float * __restrict__ A, const float * __restrict__ B, float * __restrict__ C, int ldc) {
    __m256 c00 = _mm256_setzero_ps(); __m256 c10 = _mm256_setzero_ps(); __m256 c20 = _mm256_setzero_ps();
    __m256 c01 = _mm256_setzero_ps(); __m256 c11 = _mm256_setzero_ps(); __m256 c21 = _mm256_setzero_ps();
    __m256 c02 = _mm256_setzero_ps(); __m256 c12 = _mm256_setzero_ps(); __m256 c22 = _mm256_setzero_ps();
    __m256 c03 = _mm256_setzero_ps(); __m256 c13 = _mm256_setzero_ps(); __m256 c23 = _mm256_setzero_ps();

    for (int k = 0; k < K; ++k) {
        __m256 a0 = _mm256_loadu_ps(&A[k * 24 + 0]);
        __m256 a1 = _mm256_loadu_ps(&A[k * 24 + 8]);
        __m256 a2 = _mm256_loadu_ps(&A[k * 24 + 16]);

        __m256 b0 = _mm256_set1_ps(B[k * 4 + 0]);
        c00 = _mm256_fmadd_ps(a0, b0, c00); c10 = _mm256_fmadd_ps(a1, b0, c10); c20 = _mm256_fmadd_ps(a2, b0, c20);

        __m256 b1 = _mm256_set1_ps(B[k * 4 + 1]);
        c01 = _mm256_fmadd_ps(a0, b1, c01); c11 = _mm256_fmadd_ps(a1, b1, c11); c21 = _mm256_fmadd_ps(a2, b1, c21);

        __m256 b2 = _mm256_set1_ps(B[k * 4 + 2]);
        c02 = _mm256_fmadd_ps(a0, b2, c02); c12 = _mm256_fmadd_ps(a1, b2, c12); c22 = _mm256_fmadd_ps(a2, b2, c22);

        __m256 b3 = _mm256_set1_ps(B[k * 4 + 3]);
        c03 = _mm256_fmadd_ps(a0, b3, c03); c13 = _mm256_fmadd_ps(a1, b3, c13); c23 = _mm256_fmadd_ps(a2, b3, c23);
    }

    _mm256_storeu_ps(&C[0 * ldc + 0], c00); _mm256_storeu_ps(&C[0 * ldc + 8], c10); _mm256_storeu_ps(&C[0 * ldc + 16], c20);
    _mm256_storeu_ps(&C[1 * ldc + 0], c01); _mm256_storeu_ps(&C[1 * ldc + 8], c11); _mm256_storeu_ps(&C[1 * ldc + 16], c21);
    _mm256_storeu_ps(&C[2 * ldc + 0], c02); _mm256_storeu_ps(&C[2 * ldc + 8], c12); _mm256_storeu_ps(&C[2 * ldc + 16], c22);
    _mm256_storeu_ps(&C[3 * ldc + 0], c03); _mm256_storeu_ps(&C[3 * ldc + 8], c13); _mm256_storeu_ps(&C[3 * ldc + 16], c23);
}

// 5. Tile 32x4 (16 YMM Accumulators: 4 YMM x 4 cols -> Complete register file saturation)
__declspec(noinline)
void ukernel_32x4(int K, const float * __restrict__ A, const float * __restrict__ B, float * __restrict__ C, int ldc) {
    __m256 c00 = _mm256_setzero_ps(); __m256 c10 = _mm256_setzero_ps(); __m256 c20 = _mm256_setzero_ps(); __m256 c30 = _mm256_setzero_ps();
    __m256 c01 = _mm256_setzero_ps(); __m256 c11 = _mm256_setzero_ps(); __m256 c21 = _mm256_setzero_ps(); __m256 c31 = _mm256_setzero_ps();
    __m256 c02 = _mm256_setzero_ps(); __m256 c12 = _mm256_setzero_ps(); __m256 c22 = _mm256_setzero_ps(); __m256 c32 = _mm256_setzero_ps();
    __m256 c03 = _mm256_setzero_ps(); __m256 c13 = _mm256_setzero_ps(); __m256 c23 = _mm256_setzero_ps(); __m256 c33 = _mm256_setzero_ps();

    for (int k = 0; k < K; ++k) {
        __m256 a0 = _mm256_loadu_ps(&A[k * 32 + 0]);
        __m256 a1 = _mm256_loadu_ps(&A[k * 32 + 8]);
        __m256 a2 = _mm256_loadu_ps(&A[k * 32 + 16]);
        __m256 a3 = _mm256_loadu_ps(&A[k * 32 + 24]);

        __m256 b0 = _mm256_set1_ps(B[k * 4 + 0]);
        c00 = _mm256_fmadd_ps(a0, b0, c00); c10 = _mm256_fmadd_ps(a1, b0, c10); c20 = _mm256_fmadd_ps(a2, b0, c20); c30 = _mm256_fmadd_ps(a3, b0, c30);

        __m256 b1 = _mm256_set1_ps(B[k * 4 + 1]);
        c01 = _mm256_fmadd_ps(a0, b1, c01); c11 = _mm256_fmadd_ps(a1, b1, c11); c21 = _mm256_fmadd_ps(a2, b1, c21); c31 = _mm256_fmadd_ps(a3, b1, c31);

        __m256 b2 = _mm256_set1_ps(B[k * 4 + 2]);
        c02 = _mm256_fmadd_ps(a0, b2, c02); c12 = _mm256_fmadd_ps(a1, b2, c12); c22 = _mm256_fmadd_ps(a2, b2, c22); c32 = _mm256_fmadd_ps(a3, b2, c32);

        __m256 b3 = _mm256_set1_ps(B[k * 4 + 3]);
        c03 = _mm256_fmadd_ps(a0, b3, c03); c13 = _mm256_fmadd_ps(a1, b3, c13); c23 = _mm256_fmadd_ps(a2, b3, c23); c33 = _mm256_fmadd_ps(a3, b3, c33);
    }

    _mm256_storeu_ps(&C[0 * ldc + 0], c00); _mm256_storeu_ps(&C[0 * ldc + 8], c10); _mm256_storeu_ps(&C[0 * ldc + 16], c20); _mm256_storeu_ps(&C[0 * ldc + 24], c30);
    _mm256_storeu_ps(&C[1 * ldc + 0], c01); _mm256_storeu_ps(&C[1 * ldc + 8], c11); _mm256_storeu_ps(&C[1 * ldc + 16], c21); _mm256_storeu_ps(&C[1 * ldc + 24], c31);
    _mm256_storeu_ps(&C[2 * ldc + 0], c02); _mm256_storeu_ps(&C[2 * ldc + 8], c12); _mm256_storeu_ps(&C[2 * ldc + 16], c22); _mm256_storeu_ps(&C[2 * ldc + 24], c32);
    _mm256_storeu_ps(&C[3 * ldc + 0], c03); _mm256_storeu_ps(&C[3 * ldc + 8], c13); _mm256_storeu_ps(&C[3 * ldc + 16], c23); _mm256_storeu_ps(&C[3 * ldc + 24], c33);
}

int main() {
    printf("===================================================================================\n");
    printf("   MDSLC 2D REGISTER ACCUMULATOR TILE SEARCH (AVX2: 16 YMM Budget)\n");
    printf("===================================================================================\n");
    printf("Tile(MrxNr)\tLive YMM Accs\tTime(K=256, ns)\tThroughput (GFLOP/s)\tSpill Status\n");
    printf("-----------------------------------------------------------------------------------\n");

    int K = 256;
    int iters = 200000;

    float *A = (float *)_aligned_malloc(32 * K * sizeof(float), 64);
    float *B = (float *)_aligned_malloc(6 * K * sizeof(float), 64);
    float *C = (float *)_aligned_malloc(32 * 6 * sizeof(float), 64);
    for (int i = 0; i < 32 * K; ++i) A[i] = 1.0f;
    for (int i = 0; i < 6 * K; ++i) B[i] = 1.0f;

    // 8x4 (4 YMM)
    double t0 = get_time_sec();
    for (int it = 0; it < iters; ++it) ukernel_8x4(K, A, B, C, 8);
    double t_8x4 = (get_time_sec() - t0) / iters * 1e9;
    double gf_8x4 = (2.0 * 8 * 4 * K) / t_8x4;

    // 16x4 (8 YMM)
    t0 = get_time_sec();
    for (int it = 0; it < iters; ++it) ukernel_16x4(K, A, B, C, 16);
    double t_16x4 = (get_time_sec() - t0) / iters * 1e9;
    double gf_16x4 = (2.0 * 16 * 4 * K) / t_16x4;

    // 16x6 (12 YMM)
    t0 = get_time_sec();
    for (int it = 0; it < iters; ++it) ukernel_16x6(K, A, B, C, 16);
    double t_16x6 = (get_time_sec() - t0) / iters * 1e9;
    double gf_16x6 = (2.0 * 16 * 6 * K) / t_16x6;

    // 24x4 (12 YMM)
    t0 = get_time_sec();
    for (int it = 0; it < iters; ++it) ukernel_24x4(K, A, B, C, 24);
    double t_24x4 = (get_time_sec() - t0) / iters * 1e9;
    double gf_24x4 = (2.0 * 24 * 4 * K) / t_24x4;

    // 32x4 (16 YMM)
    t0 = get_time_sec();
    for (int it = 0; it < iters; ++it) ukernel_32x4(K, A, B, C, 32);
    double t_32x4 = (get_time_sec() - t0) / iters * 1e9;
    double gf_32x4 = (2.0 * 32 * 4 * K) / t_32x4;

    printf("8x4\t\t4 YMM (25%%)\t%.2f ns\t\t%.2f GFLOP/s\t\tZERO_SPILLS\n", t_8x4, gf_8x4);
    printf("16x4\t\t8 YMM (50%%)\t%.2f ns\t\t%.2f GFLOP/s\t\tZERO_SPILLS\n", t_16x4, gf_16x4);
    printf("16x6\t\t12 YMM (75%%)\t%.2f ns\t\t%.2f GFLOP/s\t\tZERO_SPILLS\n", t_16x6, gf_16x6);
    printf("24x4\t\t12 YMM (75%%)\t%.2f ns\t\t%.2f GFLOP/s\t\tZERO_SPILLS\n", t_24x4, gf_24x4);
    printf("32x4\t\t16 YMM (100%%)\t%.2f ns\t\t%.2f GFLOP/s\t\tSPILL_SATURATION\n", t_32x4, gf_32x4);

    _aligned_free(A); _aligned_free(B); _aligned_free(C);
    return 0;
}
