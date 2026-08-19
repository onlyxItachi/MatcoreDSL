#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <immintrin.h>
#include <windows.h>
#include <vector>
#include <Eigen/Dense>

double get_time_sec() {
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart / (double)freq.QuadPart;
}

// 1. FP64 GEMM (AVX2: 4 doubles per YMM register)
__declspec(noinline)
void gemm_ikj_f64(int M, int N, int K, const double * __restrict__ A, const double * __restrict__ B, double * __restrict__ C) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) C[i * N + j] = 0.0;
        for (int k = 0; k < K; ++k) {
            double a_val = A[i * K + k];
            __m256d a_vec = _mm256_set1_pd(a_val);
            int j = 0;
            for (; j + 4 <= N; j += 4) {
                __m256d b_vec = _mm256_loadu_pd(&B[k * N + j]);
                __m256d c_vec = _mm256_loadu_pd(&C[i * N + j]);
                c_vec = _mm256_fmadd_pd(a_vec, b_vec, c_vec);
                _mm256_storeu_pd(&C[i * N + j], c_vec);
            }
            for (; j < N; ++j) C[i * N + j] += a_val * B[k * N + j];
        }
    }
}

// 2. FP64 GEMV (Row Reduction)
__declspec(noinline)
void gemv_simd_f64(int M, int K, const double * __restrict__ A, const double * __restrict__ x, double * __restrict__ y) {
    for (int i = 0; i < M; ++i) {
        __m256d acc = _mm256_setzero_pd();
        int k = 0;
        for (; k + 4 <= K; k += 4) {
            __m256d a_vec = _mm256_loadu_pd(&A[i * K + k]);
            __m256d x_vec = _mm256_loadu_pd(&x[k]);
            acc = _mm256_fmadd_pd(a_vec, x_vec, acc);
        }
        __m128d lo = _mm256_extractf128_pd(acc, 0);
        __m128d hi = _mm256_extractf128_pd(acc, 1);
        __m128d sum128 = _mm_add_pd(lo, hi);
        sum128 = _mm_hadd_pd(sum128, sum128);
        double sum = _mm_cvtsd_f64(sum128);
        for (; k < K; ++k) sum += A[i * K + k] * x[k];
        y[i] = sum;
    }
}

// 3. FP64 GEVM (Column Accumulation)
__declspec(noinline)
void gevm_simd_f64(int M, int N, const double * __restrict__ x, const double * __restrict__ A, double * __restrict__ y) {
    for (int j = 0; j < N; ++j) y[j] = 0.0;
    for (int i = 0; i < M; ++i) {
        double x_val = x[i];
        __m256d x_vec = _mm256_set1_pd(x_val);
        int j = 0;
        for (; j + 4 <= N; j += 4) {
            __m256d a_vec = _mm256_loadu_pd(&A[i * N + j]);
            __m256d y_vec = _mm256_loadu_pd(&y[j]);
            y_vec = _mm256_fmadd_pd(x_vec, a_vec, y_vec);
            _mm256_storeu_pd(&y[j], y_vec);
        }
        for (; j < N; ++j) y[j] += x_val * A[i * N + j];
    }
}

// 4. FP64 Dot Product
__declspec(noinline)
double dot_simd_f64(int N, const double * __restrict__ x, const double * __restrict__ y) {
    __m256d acc = _mm256_setzero_pd();
    int i = 0;
    for (; i + 4 <= N; i += 4) {
        __m256d vx = _mm256_loadu_pd(&x[i]);
        __m256d vy = _mm256_loadu_pd(&y[i]);
        acc = _mm256_fmadd_pd(vx, vy, acc);
    }
    __m128d lo = _mm256_extractf128_pd(acc, 0);
    __m128d hi = _mm256_extractf128_pd(acc, 1);
    __m128d sum128 = _mm_add_pd(lo, hi);
    sum128 = _mm_hadd_pd(sum128, sum128);
    double sum = _mm_cvtsd_f64(sum128);
    for (; i < N; ++i) sum += x[i] * y[i];
    return sum;
}

int main() {
    printf("===================================================================================\n");
    printf("   MDSLC DOUBLE PRECISION (FP64) 5-OPERATION RUNTIME CAMPAIGN\n");
    printf("===================================================================================\n");
    printf("Operation\tShape\t\tMedian Time(ns)\tThroughput (GFLOP/s)\tBandwidth (GB/s)\n");
    printf("-----------------------------------------------------------------------------------\n");

    // 1. FP64 GEMM Sweep
    int shapes_f64[] = { 16, 32, 64, 128, 256 };
    for (int s = 0; s < 5; ++s) {
        int N = shapes_f64[s];
        size_t sz = N * N * sizeof(double);
        double *A = (double *)_aligned_malloc(sz, 64);
        double *B = (double *)_aligned_malloc(sz, 64);
        double *C = (double *)_aligned_malloc(sz, 64);
        for (int i = 0; i < N * N; ++i) { A[i] = 1.0; B[i] = 1.0; }

        int iters = (N <= 32) ? 50000 : (N <= 128 ? 1000 : 50);
        double t0 = get_time_sec();
        for (int it = 0; it < iters; ++it) gemm_ikj_f64(N, N, N, A, B, C);
        double t_ns = (get_time_sec() - t0) / iters * 1e9;
        double gf = (2.0 * N * N * N) / t_ns;
        double bw = (3.0 * N * N * sizeof(double)) / t_ns;

        printf("GEMM_FP64\t%dx%d\t\t%.2f ns\t\t%.2f GFLOP/s\t\t%.2f GB/s\n", N, N, t_ns, gf, bw);
        _aligned_free(A); _aligned_free(B); _aligned_free(C);
    }

    // 2. FP64 GEMV vs GEVM Sweep
    int v_shapes[] = { 64, 256, 1024, 2048 };
    for (int s = 0; s < 4; ++s) {
        int N = v_shapes[s];
        size_t mat_sz = N * N * sizeof(double);
        double *A = (double *)_aligned_malloc(mat_sz, 64);
        double *x = (double *)_aligned_malloc(N * sizeof(double), 64);
        double *y1 = (double *)_aligned_malloc(N * sizeof(double), 64);
        double *y2 = (double *)_aligned_malloc(N * sizeof(double), 64);
        for (int i = 0; i < N * N; ++i) A[i] = 1.0;
        for (int i = 0; i < N; ++i) x[i] = 1.0;

        int iters = (N <= 256) ? 50000 : 1000;

        // GEMV
        double t0 = get_time_sec();
        for (int it = 0; it < iters; ++it) gemv_simd_f64(N, N, A, x, y1);
        double t_gemv = (get_time_sec() - t0) / iters * 1e9;

        // GEVM
        double t1 = get_time_sec();
        for (int it = 0; it < iters; ++it) gevm_simd_f64(N, N, x, A, y2);
        double t_gevm = (get_time_sec() - t1) / iters * 1e9;

        double total_bytes = (double)N * N * sizeof(double);
        printf("GEMV_FP64\t%dx%d\t\t%.2f ns\t\t%.2f GFLOP/s\t\t%.2f GB/s\n", N, N, t_gemv, (2.0 * N * N) / t_gemv, total_bytes / t_gemv);
        printf("GEVM_FP64\t%dx%d\t\t%.2f ns\t\t%.2f GFLOP/s\t\t%.2f GB/s\n", N, N, t_gevm, (2.0 * N * N) / t_gevm, total_bytes / t_gevm);

        _aligned_free(A); _aligned_free(x); _aligned_free(y1); _aligned_free(y2);
    }

    // 3. FP64 Dot Product
    int dot_sizes[] = { 1024, 16384, 65536 };
    for (int s = 0; s < 3; ++s) {
        int N = dot_sizes[s];
        double *x = (double *)_aligned_malloc(N * sizeof(double), 64);
        double *y = (double *)_aligned_malloc(N * sizeof(double), 64);
        for (int i = 0; i < N; ++i) { x[i] = 1.0; y[i] = 1.0; }

        int iters = 50000;
        double t0 = get_time_sec();
        for (int it = 0; it < iters; ++it) dot_simd_f64(N, x, y);
        double t_dot = (get_time_sec() - t0) / iters * 1e9;

        double total_bytes = (2.0 * N * sizeof(double));
        printf("DOT_FP64\tN=%d\t\t%.2f ns\t\t%.2f GFLOP/s\t\t%.2f GB/s\n", N, t_dot, (2.0 * N) / t_dot, total_bytes / t_dot);
        _aligned_free(x); _aligned_free(y);
    }

    return 0;
}
