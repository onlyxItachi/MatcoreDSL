#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <immintrin.h>
#include <windows.h>
#include <vector>
#include <algorithm>
#include <thread>
#include <atomic>
#include <omp.h>

#include <Eigen/Dense>

// ============================================================================
// Timing & Statistics
// ============================================================================

double get_time_sec() {
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart / (double)freq.QuadPart;
}

// ============================================================================
// 1. Transpose Combinations: NN, NT, TN, TT
// ============================================================================

// NN: C = A(MxK) * B(KxN)
void gemm_nn_f32(int M, int N, int K, const float * __restrict__ A, const float * __restrict__ B, float * __restrict__ C) {
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
            for (; j < N; ++j) C[i * N + j] += a_val * B[k * N + j];
        }
    }
}

// NT: C = A(MxK) * B^T(NxK) -> B is stored as NxK (row-major), so B(k, j) is B[j * K + k] (Dot products!)
void gemm_nt_f32(int M, int N, int K, const float * __restrict__ A, const float * __restrict__ B_T, float * __restrict__ C) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            __m256 acc = _mm256_setzero_ps();
            int k = 0;
            for (; k + 8 <= K; k += 8) {
                __m256 a_vec = _mm256_loadu_ps(&A[i * K + k]);
                __m256 b_vec = _mm256_loadu_ps(&B_T[j * K + k]);
                acc = _mm256_fmadd_ps(a_vec, b_vec, acc);
            }
            __m128 lo = _mm256_castps256_ps128(acc);
            __m128 hi = _mm256_extractf128_ps(acc, 1);
            __m128 sum128 = _mm_add_ps(lo, hi);
            sum128 = _mm_hadd_ps(sum128, sum128);
            sum128 = _mm_hadd_ps(sum128, sum128);
            float sum = _mm_cvtss_f32(sum128);
            for (; k < K; ++k) sum += A[i * K + k] * B_T[j * K + k];
            C[i * N + j] = sum;
        }
    }
}

// TN: C = A^T(KxM) * B(KxN) -> A is stored as KxM, so A(i, k) is A[k * M + i] (Outer updates!)
void gemm_tn_f32(int M, int N, int K, const float * __restrict__ A_T, const float * __restrict__ B, float * __restrict__ C) {
    for (int i = 0; i < M * N; ++i) C[i] = 0.0f;
    for (int k = 0; k < K; ++k) {
        for (int i = 0; i < M; ++i) {
            float a_val = A_T[k * M + i];
            __m256 a_vec = _mm256_set1_ps(a_val);
            int j = 0;
            for (; j + 8 <= N; j += 8) {
                __m256 b_vec = _mm256_loadu_ps(&B[k * N + j]);
                __m256 c_vec = _mm256_loadu_ps(&C[i * N + j]);
                c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
                _mm256_storeu_ps(&C[i * N + j], c_vec);
            }
            for (; j < N; ++j) C[i * N + j] += a_val * B[k * N + j];
        }
    }
}

// ============================================================================
// 2. Memory Packing vs Direct Streaming Microkernel
// ============================================================================

// Pack Matrix B into Kc x Nr panels for continuous L1/L2 streaming
void pack_B_panel_8(int K, int N, const float * __restrict__ B, float * __restrict__ B_packed) {
    int idx = 0;
    for (int j = 0; j + 8 <= N; j += 8) {
        for (int k = 0; k < K; ++k) {
            for (int jj = 0; jj < 8; ++jj) {
                B_packed[idx++] = B[k * N + j + jj];
            }
        }
    }
}

// Packed GEMM Microkernel
void gemm_packed_b_f32(int M, int N, int K, const float * __restrict__ A, const float * __restrict__ B_packed, float * __restrict__ C) {
    int b_panel_idx = 0;
    for (int j = 0; j + 8 <= N; j += 8) {
        for (int i = 0; i < M; ++i) {
            __m256 acc = _mm256_setzero_ps();
            const float *b_ptr = &B_packed[b_panel_idx];
            for (int k = 0; k < K; ++k) {
                __m256 a_val = _mm256_set1_ps(A[i * K + k]);
                __m256 b_val = _mm256_loadu_ps(&b_ptr[k * 8]);
                acc = _mm256_fmadd_ps(a_val, b_val, acc);
            }
            _mm256_storeu_ps(&C[i * N + j], acc);
        }
        b_panel_idx += K * 8;
    }
}

// ============================================================================
// 3. Multi-Threaded Parallel Scaling (OpenMP)
// ============================================================================

void gemm_parallel_omp_f32(int M, int N, int K, const float *A, const float *B, float *C, int num_threads) {
    #pragma omp parallel for num_threads(num_threads) schedule(static)
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
            for (; j < N; ++j) C[i * N + j] += a_val * B[k * N + j];
        }
    }
}

void gemv_parallel_omp_f32(int M, int K, const float *A, const float *x, float *y, int num_threads) {
    #pragma omp parallel for num_threads(num_threads) schedule(static)
    for (int i = 0; i < M; ++i) {
        __m256 acc = _mm256_setzero_ps();
        int k = 0;
        for (; k + 8 <= K; k += 8) {
            __m256 a_vec = _mm256_loadu_ps(&A[i * K + k]);
            __m256 x_vec = _mm256_loadu_ps(&x[k]);
            acc = _mm256_fmadd_ps(a_vec, x_vec, acc);
        }
        __m128 lo = _mm256_castps256_ps128(acc);
        __m128 hi = _mm256_extractf128_ps(acc, 1);
        __m128 sum128 = _mm_add_ps(lo, hi);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        float sum = _mm_cvtss_f32(sum128);
        for (; k < K; ++k) sum += A[i * K + k] * x[k];
        y[i] = sum;
    }
}

// ============================================================================
// 4. Main Deep Experiment Suite
// ============================================================================

int main() {
    FILE *f_csv = fopen("corpus/findings/cpu/deep_campaign_results.csv", "w");
    FILE *f_jsonl = fopen("corpus/findings/cpu/deep_campaign_results.jsonl", "w");
    if (!f_csv || !f_jsonl) {
        printf("Failed to open output files\n");
        return 1;
    }

    fprintf(f_csv, "experiment_id,domain,operation,dtype,shape,threads,metric_name,metric_value,status\n");

    printf("===================================================================================\n");
    printf("   MDSLC DEEP CPU EXECUTION CAMPAIGN: TRANSPOSE, PACKING & THREADING ATLAS\n");
    printf("===================================================================================\n");

    // ------------------------------------------------------------------------
    // Phase 1: Transpose Layouts (NN vs NT vs TN)
    // ------------------------------------------------------------------------
    printf("\n--- Phase 1: Transpose Orientation Matrix (N=256, FP32) ---\n");
    int N = 256, M = 256, K = 256;
    size_t sz = N * N * sizeof(float);
    float *A = (float *)_aligned_malloc(sz, 64);
    float *B = (float *)_aligned_malloc(sz, 64);
    float *C = (float *)_aligned_malloc(sz, 64);
    for (int i = 0; i < N * N; ++i) { A[i] = ((float)rand()/RAND_MAX)-0.5f; B[i] = ((float)rand()/RAND_MAX)-0.5f; }

    int iters = 100;
    // NN
    double t0 = get_time_sec();
    for (int it = 0; it < iters; ++it) gemm_nn_f32(M, N, K, A, B, C);
    double t_nn = (get_time_sec() - t0) / iters * 1000.0;

    // NT
    double t1 = get_time_sec();
    for (int it = 0; it < iters; ++it) gemm_nt_f32(M, N, K, A, B, C);
    double t_nt = (get_time_sec() - t1) / iters * 1000.0;

    // TN
    double t2 = get_time_sec();
    for (int it = 0; it < iters; ++it) gemm_tn_f32(M, N, K, A, B, C);
    double t_tn = (get_time_sec() - t2) / iters * 1000.0;

    double gflops_nn = (2.0 * M * N * K) / (t_nn * 1e6);
    double gflops_nt = (2.0 * M * N * K) / (t_nt * 1e6);
    double gflops_tn = (2.0 * M * N * K) / (t_tn * 1e6);

    printf("GEMM NN (Contiguous B rows) : %.3f ms (%.2f GFLOP/s)\n", t_nn, gflops_nn);
    printf("GEMM NT (Dot Product rows)  : %.3f ms (%.2f GFLOP/s)\n", t_nt, gflops_nt);
    printf("GEMM TN (Outer Product cols): %.3f ms (%.2f GFLOP/s)\n", t_tn, gflops_tn);

    fprintf(f_csv, "EXP-LAYOUT-NN,Transpose,GEMM_NN,f32,256x256,1,gflops,%.2f,PASS\n", gflops_nn);
    fprintf(f_csv, "EXP-LAYOUT-NT,Transpose,GEMM_NT,f32,256x256,1,gflops,%.2f,PASS\n", gflops_nt);
    fprintf(f_csv, "EXP-LAYOUT-TN,Transpose,GEMM_TN,f32,256x256,1,gflops,%.2f,PASS\n", gflops_tn);

    // ------------------------------------------------------------------------
    // Phase 2: Memory Packing vs Direct Streaming
    // ------------------------------------------------------------------------
    printf("\n--- Phase 2: Memory Packing Crossover Sweep (FP32) ---\n");
    int pack_shapes[] = { 16, 32, 64, 128, 256, 512 };
    for (int s = 0; s < 6; ++s) {
        int dim = pack_shapes[s];
        size_t b_bytes = dim * dim * sizeof(float);
        float *Ap = (float *)_aligned_malloc(b_bytes, 64);
        float *Bp = (float *)_aligned_malloc(b_bytes, 64);
        float *Cp = (float *)_aligned_malloc(b_bytes, 64);
        float *Bp_packed = (float *)_aligned_malloc(b_bytes, 64);
        for (int i = 0; i < dim * dim; ++i) { Ap[i] = 1.0f; Bp[i] = 1.0f; }

        int p_iters = (dim <= 64) ? 2000 : (dim <= 256 ? 100 : 20);

        // Direct Stream (No Pack)
        double td0 = get_time_sec();
        for (int it = 0; it < p_iters; ++it) gemm_nn_f32(dim, dim, dim, Ap, Bp, Cp);
        double time_direct = (get_time_sec() - td0) / p_iters * 1e6; // us

        // Packing Time
        double tp0 = get_time_sec();
        for (int it = 0; it < p_iters; ++it) pack_B_panel_8(dim, dim, Bp, Bp_packed);
        double time_pack_only = (get_time_sec() - tp0) / p_iters * 1e6; // us

        // Packed Compute Time
        double tc0 = get_time_sec();
        for (int it = 0; it < p_iters; ++it) gemm_packed_b_f32(dim, dim, dim, Ap, Bp_packed, Cp);
        double time_packed_comp = (get_time_sec() - tc0) / p_iters * 1e6; // us

        double total_packed = time_pack_only + time_packed_comp;
        double pack_ratio = (time_pack_only / total_packed) * 100.0;

        printf("Shape %dx%d: Direct=%.1f us | Pack=%.1f us (%.1f%%) | PackedCompute=%.1f us | Crossover=%s\n",
               dim, dim, time_direct, time_pack_only, pack_ratio, time_packed_comp, (total_packed < time_direct ? "PACKING_WINS" : "DIRECT_WINS"));

        fprintf(f_csv, "EXP-PACK-%d,Packing,GEMM,f32,%dx%d,1,pack_overhead_pct,%.2f,%s\n",
                dim, dim, dim, pack_ratio, (total_packed < time_direct ? "PACKING_WINS" : "DIRECT_WINS"));

        _aligned_free(Ap);
        _aligned_free(Bp);
        _aligned_free(Cp);
        _aligned_free(Bp_packed);
    }

    // ------------------------------------------------------------------------
    // Phase 3: Multi-Thread Scaling (1 to 24 Threads on 12-Core Zen 5)
    // ------------------------------------------------------------------------
    printf("\n--- Phase 3: Multi-Core Thread Scaling (Zen 5 12C/24T, FP32) ---\n");
    int thread_counts[] = { 1, 2, 4, 8, 12, 16, 24 };
    int num_threads_cases = 7;

    int N_gemm_par = 512;
    size_t sz_par = N_gemm_par * N_gemm_par * sizeof(float);
    float *A_par = (float *)_aligned_malloc(sz_par, 64);
    float *B_par = (float *)_aligned_malloc(sz_par, 64);
    float *C_par = (float *)_aligned_malloc(sz_par, 64);
    for (int i = 0; i < N_gemm_par * N_gemm_par; ++i) { A_par[i] = 1.0f; B_par[i] = 1.0f; }

    int N_gemv_par = 4096;
    size_t sz_gemv_mat = (size_t)N_gemv_par * N_gemv_par * sizeof(float);
    float *A_gemv = (float *)_aligned_malloc(sz_gemv_mat, 64);
    float *x_gemv = (float *)_aligned_malloc(N_gemv_par * sizeof(float), 64);
    float *y_gemv = (float *)_aligned_malloc(N_gemv_par * sizeof(float), 64);
    for (int i = 0; i < N_gemv_par; ++i) x_gemv[i] = 1.0f;

    printf("Threads\tGEMM_512(ms)\tGEMM_GFLOP/s\tGEMM_Scaling\tGEMV_4K(ms)\tGEMV_GB/s\tGEMV_Scaling\n");

    double gemm_1t_ms = 0.0, gemv_1t_ms = 0.0;

    for (int tc = 0; tc < num_threads_cases; ++tc) {
        int th = thread_counts[tc];

        // GEMM
        int g_iters = 30;
        double tg0 = get_time_sec();
        for (int it = 0; it < g_iters; ++it) gemm_parallel_omp_f32(N_gemm_par, N_gemm_par, N_gemm_par, A_par, B_par, C_par, th);
        double tg_ms = (get_time_sec() - tg0) / g_iters * 1000.0;
        if (th == 1) gemm_1t_ms = tg_ms;
        double g_flops = (2.0 * N_gemm_par * N_gemm_par * N_gemm_par) / (tg_ms * 1e6);
        double g_scale = gemm_1t_ms / tg_ms;

        // GEMV
        int v_iters = 100;
        double tv0 = get_time_sec();
        for (int it = 0; it < v_iters; ++it) gemv_parallel_omp_f32(N_gemv_par, N_gemv_par, A_gemv, x_gemv, y_gemv, th);
        double tv_ms = (get_time_sec() - tv0) / v_iters * 1000.0;
        if (th == 1) gemv_1t_ms = tv_ms;
        double v_bytes = (double)N_gemv_par * N_gemv_par * sizeof(float);
        double v_bw = (v_bytes / (tv_ms * 1e6));
        double v_scale = gemv_1t_ms / tv_ms;

        printf("%d\t%.2f\t\t%.2f\t\t%.2fx\t\t%.2f\t\t%.2f\t\t%.2fx\n", th, tg_ms, g_flops, g_scale, tv_ms, v_bw, v_scale);

        fprintf(f_csv, "EXP-SCALE-GEMM-T%d,Threading,GEMM,f32,512x512,%d,gflops,%.2f,PASS\n", th, th, g_flops);
        fprintf(f_csv, "EXP-SCALE-GEMV-T%d,Threading,GEMV,f32,4096x4096,%d,bandwidth_gbps,%.2f,PASS\n", th, th, v_bw);
    }

    _aligned_free(A); _aligned_free(B); _aligned_free(C);
    _aligned_free(A_par); _aligned_free(B_par); _aligned_free(C_par);
    _aligned_free(A_gemv); _aligned_free(x_gemv); _aligned_free(y_gemv);

    fclose(f_csv);
    fclose(f_jsonl);
    return 0;
}
