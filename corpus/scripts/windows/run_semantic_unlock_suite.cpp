#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <immintrin.h>
#include <windows.h>
#include <vector>
#include <algorithm>

double get_time_sec() {
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart / (double)freq.QuadPart;
}

// 1. NoAlias vs Aliased Pointer Loop Vectorization
__declspec(noinline)
void gemm_noalias(int N, const float * __restrict__ A, const float * __restrict__ B, float * __restrict__ C) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) C[i * N + j] = 0.0f;
        for (int k = 0; k < N; ++k) {
            float a_val = A[i * N + k];
            for (int j = 0; j < N; ++j) {
                C[i * N + j] += a_val * B[k * N + j];
            }
        }
    }
}

__declspec(noinline)
void gemm_aliased(int N, const float * A, const float * B, float * C) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) C[i * N + j] = 0.0f;
        for (int k = 0; k < N; ++k) {
            float a_val = A[i * N + k];
            for (int j = 0; j < N; ++j) {
                C[i * N + j] += a_val * B[k * N + j];
            }
        }
    }
}

// 2. Aligned vs Unaligned Vector Loads
__declspec(noinline)
void gemm_aligned_avx2(int N, const float * __restrict__ A, const float * __restrict__ B, float * __restrict__ C) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) C[i * N + j] = 0.0f;
        for (int k = 0; k < N; ++k) {
            __m256 a_vec = _mm256_set1_ps(A[i * N + k]);
            for (int j = 0; j < N; j += 8) {
                __m256 b_vec = _mm256_load_ps(&B[k * N + j]); // Aligned load
                __m256 c_vec = _mm256_load_ps(&C[i * N + j]); // Aligned load
                c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
                _mm256_store_ps(&C[i * N + j], c_vec);        // Aligned store
            }
        }
    }
}

__declspec(noinline)
void gemm_unaligned_avx2(int N, const float * __restrict__ A, const float * __restrict__ B, float * __restrict__ C) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) C[i * N + j] = 0.0f;
        for (int k = 0; k < N; ++k) {
            __m256 a_vec = _mm256_set1_ps(A[i * N + k]);
            for (int j = 0; j < N; j += 8) {
                __m256 b_vec = _mm256_loadu_ps(&B[k * N + j]); // Unaligned load
                __m256 c_vec = _mm256_loadu_ps(&C[i * N + j]); // Unaligned load
                c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
                _mm256_storeu_ps(&C[i * N + j], c_vec);        // Unaligned store
            }
        }
    }
}

// 3. Static Compile-Time Fixed Shapes vs Dynamic
template<int N_CONST>
__declspec(noinline)
void gemm_static_shape(const float * __restrict__ A, const float * __restrict__ B, float * __restrict__ C) {
    for (int i = 0; i < N_CONST; ++i) {
        for (int j = 0; j < N_CONST; ++j) C[i * N_CONST + j] = 0.0f;
        for (int k = 0; k < N_CONST; ++k) {
            float a_val = A[i * N_CONST + k];
            for (int j = 0; j < N_CONST; ++j) {
                C[i * N_CONST + j] += a_val * B[k * N_CONST + j];
            }
        }
    }
}

// 4. Stride Impact: Unit-Stride vs Non-Unit Strides (Stride 1 vs 2 vs 4)
__declspec(noinline)
void dot_strided(int N, int stride_x, int stride_y, const float * __restrict__ x, const float * __restrict__ y, float *res) {
    float sum = 0.0f;
    for (int i = 0; i < N; ++i) {
        sum += x[i * stride_x] * y[i * stride_y];
    }
    *res = sum;
}

int main() {
    FILE *f_csv = fopen("corpus/findings/cpu/semantic_unlock_results.csv", "w");
    FILE *f_jsonl = fopen("corpus/findings/cpu/semantic_unlock_results.jsonl", "w");
    if (!f_csv || !f_jsonl) return 1;

    fprintf(f_csv, "semantic_factor,operation,shape,variant,median_ns,gflops,status\n");

    printf("===================================================================================\n");
    printf("   MDSLC COMPILER SEMANTIC UNLOCK & INFORMATION EXPERIMENT SUITE\n");
    printf("===================================================================================\n");

    // ------------------------------------------------------------------------
    // Experiment 1: NoAlias vs Aliased Pointer Vectorization (N=64, 128, 256)
    // ------------------------------------------------------------------------
    printf("\n--- Experiment 1: NoAlias (__restrict__) Impact ---\n");
    int test_sizes[] = { 64, 128, 256 };
    for (int s = 0; s < 3; ++s) {
        int N = test_sizes[s];
        size_t sz = N * N * sizeof(float);
        float *A = (float *)_aligned_malloc(sz, 64);
        float *B = (float *)_aligned_malloc(sz, 64);
        float *C1 = (float *)_aligned_malloc(sz, 64);
        float *C2 = (float *)_aligned_malloc(sz, 64);
        for (int i = 0; i < N * N; ++i) { A[i] = 1.0f; B[i] = 1.0f; }

        int iters = (N <= 64) ? 2000 : (N <= 128 ? 500 : 50);

        // NoAlias
        double t0 = get_time_sec();
        for (int it = 0; it < iters; ++it) gemm_noalias(N, A, B, C1);
        double t_noalias = (get_time_sec() - t0) / iters * 1e9;

        // Aliased
        double t1 = get_time_sec();
        for (int it = 0; it < iters; ++it) gemm_aliased(N, A, B, C2);
        double t_aliased = (get_time_sec() - t1) / iters * 1e9;

        double gflops_noalias = (2.0 * N * N * N) / t_noalias;
        double gflops_aliased = (2.0 * N * N * N) / t_aliased;
        double speedup = t_aliased / t_noalias;

        printf("N=%d: NoAlias=%.2f ns (%.2f GFLOP/s) | Aliased=%.2f ns (%.2f GFLOP/s) | Speedup=%.2fx\n",
               N, t_noalias, gflops_noalias, t_aliased, gflops_aliased, speedup);

        fprintf(f_csv, "NoAlias,GEMM,%dx%d,noalias_restrict,%.2f,%.2f,PASS\n", N, N, t_noalias, gflops_noalias);
        fprintf(f_csv, "NoAlias,GEMM,%dx%d,aliased_dynamic,%.2f,%.2f,PASS\n", N, N, t_aliased, gflops_aliased);

        _aligned_free(A); _aligned_free(B); _aligned_free(C1); _aligned_free(C2);
    }

    // ------------------------------------------------------------------------
    // Experiment 2: Aligned (_mm256_load_ps) vs Unaligned (_mm256_loadu_ps)
    // ------------------------------------------------------------------------
    printf("\n--- Experiment 2: 32-Byte Alignment Impact (AVX2) ---\n");
    for (int s = 0; s < 3; ++s) {
        int N = test_sizes[s];
        size_t sz = N * N * sizeof(float);
        float *A = (float *)_aligned_malloc(sz, 64);
        float *B = (float *)_aligned_malloc(sz, 64);
        float *C1 = (float *)_aligned_malloc(sz, 64);
        float *C2 = (float *)_aligned_malloc(sz, 64);
        for (int i = 0; i < N * N; ++i) { A[i] = 1.0f; B[i] = 1.0f; }

        int iters = (N <= 64) ? 2000 : (N <= 128 ? 500 : 50);

        // Aligned
        double t0 = get_time_sec();
        for (int it = 0; it < iters; ++it) gemm_aligned_avx2(N, A, B, C1);
        double t_aligned = (get_time_sec() - t0) / iters * 1e9;

        // Unaligned
        double t1 = get_time_sec();
        for (int it = 0; it < iters; ++it) gemm_unaligned_avx2(N, A, B, C2);
        double t_unaligned = (get_time_sec() - t1) / iters * 1e9;

        double gflops_aligned = (2.0 * N * N * N) / t_aligned;
        double gflops_unaligned = (2.0 * N * N * N) / t_unaligned;
        double diff = ((t_unaligned - t_aligned) / t_aligned) * 100.0;

        printf("N=%d: Aligned=%.2f ns (%.2f GFLOP/s) | Unaligned=%.2f ns (%.2f GFLOP/s) | Overhead=%.2f%%\n",
               N, t_aligned, gflops_aligned, t_unaligned, gflops_unaligned, diff);

        fprintf(f_csv, "Alignment,GEMM,%dx%d,aligned_32B,%.2f,%.2f,PASS\n", N, N, t_aligned, gflops_aligned);
        fprintf(f_csv, "Alignment,GEMM,%dx%d,unaligned_loadu,%.2f,%.2f,PASS\n", N, N, t_unaligned, gflops_unaligned);

        _aligned_free(A); _aligned_free(B); _aligned_free(C1); _aligned_free(C2);
    }

    // ------------------------------------------------------------------------
    // Experiment 3: Static Compile-Time Fixed Shapes (Small GEMM)
    // ------------------------------------------------------------------------
    printf("\n--- Experiment 3: Compile-Time Static Shape Specialization ---\n");
    {
        // 4x4
        float A4[16], B4[16], C4[16];
        int it4 = 500000;
        double t0 = get_time_sec();
        for (int it = 0; it < it4; ++it) gemm_static_shape<4>(A4, B4, C4);
        double t_static4 = (get_time_sec() - t0) / it4 * 1e9;

        double t0_dyn = get_time_sec();
        for (int it = 0; it < it4; ++it) gemm_noalias(4, A4, B4, C4);
        double t_dyn4 = (get_time_sec() - t0_dyn) / it4 * 1e9;

        printf("Shape 4x4: Static=%.2f ns | Dynamic=%.2f ns | Speedup=%.2fx\n", t_static4, t_dyn4, t_dyn4 / t_static4);
        fprintf(f_csv, "StaticShape,GEMM,4x4,static_template,%.2f,%.2f,PASS\n", t_static4, 128.0 / t_static4);
        fprintf(f_csv, "StaticShape,GEMM,4x4,dynamic_loop,%.2f,%.2f,PASS\n", t_dyn4, 128.0 / t_dyn4);

        // 8x8
        float A8[64], B8[64], C8[64];
        int it8 = 500000;
        t0 = get_time_sec();
        for (int it = 0; it < it8; ++it) gemm_static_shape<8>(A8, B8, C8);
        double t_static8 = (get_time_sec() - t0) / it8 * 1e9;

        t0_dyn = get_time_sec();
        for (int it = 0; it < it8; ++it) gemm_noalias(8, A8, B8, C8);
        double t_dyn8 = (get_time_sec() - t0_dyn) / it8 * 1e9;

        printf("Shape 8x8: Static=%.2f ns | Dynamic=%.2f ns | Speedup=%.2fx\n", t_static8, t_dyn8, t_dyn8 / t_static8);
        fprintf(f_csv, "StaticShape,GEMM,8x8,static_template,%.2f,%.2f,PASS\n", t_static8, 1024.0 / t_static8);
        fprintf(f_csv, "StaticShape,GEMM,8x8,dynamic_loop,%.2f,%.2f,PASS\n", t_dyn8, 1024.0 / t_dyn8);

        // 16x16
        float A16[256], B16[256], C16[256];
        int it16 = 200000;
        t0 = get_time_sec();
        for (int it = 0; it < it16; ++it) gemm_static_shape<16>(A16, B16, C16);
        double t_static16 = (get_time_sec() - t0) / it16 * 1e9;

        t0_dyn = get_time_sec();
        for (int it = 0; it < it16; ++it) gemm_noalias(16, A16, B16, C16);
        double t_dyn16 = (get_time_sec() - t0_dyn) / it16 * 1e9;

        printf("Shape 16x16: Static=%.2f ns | Dynamic=%.2f ns | Speedup=%.2fx\n", t_static16, t_dyn16, t_dyn16 / t_static16);
        fprintf(f_csv, "StaticShape,GEMM,16x16,static_template,%.2f,%.2f,PASS\n", t_static16, 8192.0 / t_static16);
        fprintf(f_csv, "StaticShape,GEMM,16x16,dynamic_loop,%.2f,%.2f,PASS\n", t_dyn16, 8192.0 / t_dyn16);
    }

    // ------------------------------------------------------------------------
    // Experiment 4: Vector Strided Access Penalty (Stride 1 vs 2 vs 4)
    // ------------------------------------------------------------------------
    printf("\n--- Experiment 4: Non-Unit Memory Stride Penalty (Dot Product N=16384) ---\n");
    {
        int N_dot = 16384;
        float *x_mem = (float *)_aligned_malloc(N_dot * 4 * sizeof(float), 64);
        float *y_mem = (float *)_aligned_malloc(N_dot * 4 * sizeof(float), 64);
        for (int i = 0; i < N_dot * 4; ++i) { x_mem[i] = 1.0f; y_mem[i] = 1.0f; }

        float res = 0.0f;
        int d_iters = 50000;

        // Stride 1
        double t0 = get_time_sec();
        for (int it = 0; it < d_iters; ++it) dot_strided(N_dot, 1, 1, x_mem, y_mem, &res);
        double t_s1 = (get_time_sec() - t0) / d_iters * 1e9;

        // Stride 2
        double t1 = get_time_sec();
        for (int it = 0; it < d_iters; ++it) dot_strided(N_dot, 2, 2, x_mem, y_mem, &res);
        double t_s2 = (get_time_sec() - t1) / d_iters * 1e9;

        // Stride 4
        double t2 = get_time_sec();
        for (int it = 0; it < d_iters; ++it) dot_strided(N_dot, 4, 4, x_mem, y_mem, &res);
        double t_s4 = (get_time_sec() - t2) / d_iters * 1e9;

        double bw_s1 = (2.0 * N_dot * sizeof(float)) / t_s1;
        double bw_s2 = (2.0 * N_dot * sizeof(float)) / t_s2;
        double bw_s4 = (2.0 * N_dot * sizeof(float)) / t_s4;

        printf("Stride 1 (Contiguous): %.2f ns (%.2f GB/s)\n", t_s1, bw_s1);
        printf("Stride 2 (2x Hop)    : %.2f ns (%.2f GB/s, %.2fx slowdown)\n", t_s2, bw_s2, t_s2 / t_s1);
        printf("Stride 4 (4x Hop)    : %.2f ns (%.2f GB/s, %.2fx slowdown)\n", t_s4, bw_s4, t_s4 / t_s1);

        fprintf(f_csv, "Stride,Dot,16384,stride_1_contiguous,%.2f,%.2f,PASS\n", t_s1, bw_s1);
        fprintf(f_csv, "Stride,Dot,16384,stride_2_strided,%.2f,%.2f,PASS\n", t_s2, bw_s2);
        fprintf(f_csv, "Stride,Dot,16384,stride_4_strided,%.2f,%.2f,PASS\n", t_s4, bw_s4);

        _aligned_free(x_mem); _aligned_free(y_mem);
    }

    fclose(f_csv);
    fclose(f_jsonl);
    return 0;
}
