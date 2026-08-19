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

// Strided Matrix-Vector multiplication (GEMV) testing cache set conflict
__declspec(noinline)
void gemv_strided(int M, int K, int lda, const float * __restrict__ A, const float * __restrict__ x, float * __restrict__ y) {
    for (int i = 0; i < M; ++i) {
        __m256 acc = _mm256_setzero_ps();
        int k = 0;
        for (; k + 8 <= K; k += 8) {
            __m256 a_vec = _mm256_loadu_ps(&A[i * lda + k]);
            __m256 x_vec = _mm256_loadu_ps(&x[k]);
            acc = _mm256_fmadd_ps(a_vec, x_vec, acc);
        }
        __m128 lo = _mm256_castps256_ps128(acc);
        __m128 hi = _mm256_extractf128_ps(acc, 1);
        __m128 sum128 = _mm_add_ps(lo, hi);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        float sum = _mm_cvtss_f32(sum128);
        for (; k < K; ++k) sum += A[i * lda + k] * x[k];
        y[i] = sum;
    }
}

int main() {
    printf("===================================================================================\n");
    printf("   MDSLC CACHE SET CONFLICT & LEADING DIMENSION PADDING SWEEP\n");
    printf("===================================================================================\n");
    printf("Matrix Dimension\tUnpadded lda (Time, ms)\tPadded lda (Time, ms)\tCache Thrashing Penalty\n");
    printf("-----------------------------------------------------------------------------------\n");

    int dims[] = { 512, 1024, 2048, 4096 };
    for (int s = 0; s < 4; ++s) {
        int N = dims[s];
        int lda_unpadded = N;
        int lda_padded = N + 8; // Offset by 32 bytes (1 AVX2 vector) to break cache-set collision

        size_t sz_unpadded = (size_t)N * lda_unpadded * sizeof(float);
        size_t sz_padded = (size_t)N * lda_padded * sizeof(float);

        float *A_unpad = (float *)_aligned_malloc(sz_unpadded, 64);
        float *A_pad = (float *)_aligned_malloc(sz_padded, 64);
        float *x = (float *)_aligned_malloc(N * sizeof(float), 64);
        float *y = (float *)_aligned_malloc(N * sizeof(float), 64);

        for (size_t i = 0; i < (size_t)N * lda_unpadded; ++i) A_unpad[i] = 1.0f;
        for (size_t i = 0; i < (size_t)N * lda_padded; ++i) A_pad[i] = 1.0f;
        for (int i = 0; i < N; ++i) x[i] = 1.0f;

        int iters = (N <= 1024) ? 2000 : 200;

        // Unpadded
        double t0 = get_time_sec();
        for (int it = 0; it < iters; ++it) gemv_strided(N, N, lda_unpadded, A_unpad, x, y);
        double t_unpad = (get_time_sec() - t0) / iters * 1000.0;

        // Padded
        double t1 = get_time_sec();
        for (int it = 0; it < iters; ++it) gemv_strided(N, N, lda_padded, A_pad, x, y);
        double t_pad = (get_time_sec() - t1) / iters * 1000.0;

        double penalty = (t_unpad / t_pad);

        printf("%dx%d\t\t%.3f ms\t\t\t%.3f ms\t\t\t%.2fx %s\n",
               N, N, t_unpad, t_pad, penalty, (penalty > 1.05 ? "PENALTY_DETECTED" : "NO_PENALTY"));

        _aligned_free(A_unpad); _aligned_free(A_pad); _aligned_free(x); _aligned_free(y);
    }

    return 0;
}
