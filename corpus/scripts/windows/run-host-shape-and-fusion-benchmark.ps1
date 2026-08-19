<#
.SYNOPSIS
    Controlled host benchmark measuring GEMM shape scaling and fused vs separate epilogue performance.
#>

param(
    [string]$ClangPath = "C:\Users\hamza\tools\llvm-21.1.8\bin\clang.exe",
    [string]$OutputDir = "$PSScriptRoot\..\..\..\raw-corpus\host_benchmarks"
)

$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$benchSrc = Join-Path $OutputDir "host_gemm_benchmark.c"
@"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <immintrin.h>
#include <windows.h>

// 1. Naive C GEMM (IJK)
void gemm_naive(int M, int N, int K, const float * __restrict__ A, const float * __restrict__ B, float * __restrict__ C) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// 2. Vector-Friendly Tiled GEMM (IKJ with FMA)
void gemm_tiled_ikj(int M, int N, int K, const float * __restrict__ A, const float * __restrict__ B, float * __restrict__ C) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) C[i * N + j] = 0.0f;
        for (int k = 0; k < K; ++k) {
            float a_val = A[i * K + k];
            __m256 a_vec = _mm256_set1_ps(a_val);
            int j = 0;
            for (; j + 8 <= N; j += 8) {
                __m256 b_vec = _mm256_loadu_ps(&B[k * N + j]);
                __m256 c_vec = _mm256_loadu_ps(&C[i * N + j]);
                c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
                _mm256_storeu_ps(&C[i * N + j], c_vec);
            }
            for (; j < N; ++j) {
                C[i * N + j] += a_val * B[k * N + j];
            }
        }
    }
}

// 3. Separate Epilogue: GEMM followed by ReLU pass
void epilogue_relu_separate(int M, int N, float * __restrict__ C) {
    for (int i = 0; i < M * N; ++i) {
        if (C[i] < 0.0f) C[i] = 0.0f;
    }
}

// 4. Fused Epilogue: GEMM with inline ReLU during store
void gemm_fused_relu(int M, int N, int K, const float * __restrict__ A, const float * __restrict__ B, float * __restrict__ C) {
    __m256 zero_vec = _mm256_setzero_ps();
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) C[i * N + j] = 0.0f;
        for (int k = 0; k < K; ++k) {
            float a_val = A[i * K + k];
            __m256 a_vec = _mm256_set1_ps(a_val);
            int j = 0;
            for (; j + 8 <= N; j += 8) {
                __m256 b_vec = _mm256_loadu_ps(&B[k * N + j]);
                __m256 c_vec = _mm256_loadu_ps(&C[i * N + j]);
                c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
                _mm256_storeu_ps(&C[i * N + j], c_vec);
            }
            for (; j < N; ++j) {
                C[i * N + j] += a_val * B[k * N + j];
            }
        }
        // In-register ReLU applied to row before final store
        int j = 0;
        for (; j + 8 <= N; j += 8) {
            __m256 c_vec = _mm256_loadu_ps(&C[i * N + j]);
            c_vec = _mm256_max_ps(c_vec, zero_vec);
            _mm256_storeu_ps(&C[i * N + j], c_vec);
        }
        for (; j < N; ++j) {
            if (C[i * N + j] < 0.0f) C[i * N + j] = 0.0f;
        }
    }
}

double get_time_sec() {
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart / (double)freq.QuadPart;
}

int main() {
    int shapes[] = { 16, 32, 64, 128, 256 };
    int num_shapes = 5;

    printf("=== Host CPU GEMM & Fusion Benchmark Results ===\n");
    printf("Shape(N)\tTiled_IKJ(ms)\tSeparate_ReLU(ms)\tFused_ReLU(ms)\tFusion_Speedup\n");

    for (int s = 0; s < num_shapes; ++s) {
        int N = shapes[s];
        int M = N, K = N;
        size_t bytes = (size_t)N * N * sizeof(float);
        float *A = (float *)_aligned_malloc(bytes, 64);
        float *B = (float *)_aligned_malloc(bytes, 64);
        float *C = (float *)_aligned_malloc(bytes, 64);

        for (int i = 0; i < N * N; ++i) {
            A[i] = ((float)rand() / RAND_MAX) - 0.5f;
            B[i] = ((float)rand() / RAND_MAX) - 0.5f;
        }

        int iters = (N <= 64) ? 200 : (N <= 128 ? 50 : 10);

        // Warmup
        gemm_tiled_ikj(M, N, K, A, B, C);

        // 1. Tiled IKJ
        double t0 = get_time_sec();
        for (int it = 0; it < iters; ++it) {
            gemm_tiled_ikj(M, N, K, A, B, C);
        }
        double t1 = get_time_sec();
        double time_tiled_ms = ((t1 - t0) / iters) * 1000.0;

        // 2. Separate ReLU
        double t2 = get_time_sec();
        for (int it = 0; it < iters; ++it) {
            gemm_tiled_ikj(M, N, K, A, B, C);
            epilogue_relu_separate(M, N, C);
        }
        double t3 = get_time_sec();
        double time_sep_ms = ((t3 - t2) / iters) * 1000.0;

        // 3. Fused ReLU
        double t4 = get_time_sec();
        for (int it = 0; it < iters; ++it) {
            gemm_fused_relu(M, N, K, A, B, C);
        }
        double t5 = get_time_sec();
        double time_fused_ms = ((t5 - t4) / iters) * 1000.0;

        double speedup = time_sep_ms / time_fused_ms;

        printf("%dx%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.2fx\n", N, N, time_tiled_ms, time_sep_ms, time_fused_ms, speedup);

        _aligned_free(A);
        _aligned_free(B);
        _aligned_free(C);
    }
    return 0;
}
"@ | Out-File -FilePath $benchSrc -Encoding utf8

$benchExe = Join-Path $OutputDir "host_gemm_benchmark.exe"
& $ClangPath -O3 -mavx2 -mfma $benchSrc -o $benchExe

& $benchExe
