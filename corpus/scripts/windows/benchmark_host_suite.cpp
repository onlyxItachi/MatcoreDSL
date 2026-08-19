#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <immintrin.h>
#include <windows.h>
#include <vector>
#include <algorithm>
#include <numeric>

// Include Eigen 3 (Header-Only)
#include <Eigen/Dense>

// ============================================================================
// 1. High-Resolution Timing & Statistics
// ============================================================================

struct BenchStats {
    double median_ns;
    double mean_ns;
    double p10_ns;
    double p90_ns;
    double min_ns;
    double max_ns;
    double stddev_ns;
    int sample_count;
};

double get_time_sec() {
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart / (double)freq.QuadPart;
}

BenchStats compute_stats(std::vector<double> &samples_ns) {
    std::sort(samples_ns.begin(), samples_ns.end());
    size_t n = samples_ns.size();
    double sum = 0.0;
    for (double s : samples_ns) sum += s;
    double mean = sum / n;

    double sq_diff = 0.0;
    for (double s : samples_ns) sq_diff += (s - mean) * (s - mean);
    double stddev = sqrt(sq_diff / n);

    BenchStats st;
    st.min_ns = samples_ns.front();
    st.max_ns = samples_ns.back();
    st.mean_ns = mean;
    st.stddev_ns = stddev;
    st.median_ns = samples_ns[n / 2];
    st.p10_ns = samples_ns[(size_t)(0.10 * n)];
    st.p90_ns = samples_ns[(size_t)(0.90 * n)];
    st.sample_count = (int)n;
    return st;
}

// ============================================================================
// 2. High-Precision Mathematical Reference Oracles
// ============================================================================

void oracle_gemm_f32(int M, int N, int K, const float *A, const float *B, float *C, int ldc) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int k = 0; k < K; ++k) {
                sum += (double)A[i * K + k] * (double)B[k * N + j];
            }
            C[i * ldc + j] = (float)sum;
        }
    }
}

void oracle_gemm_f64(int M, int N, int K, const double *A, const double *B, double *C, int ldc) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            long double sum = 0.0;
            for (int k = 0; k < K; ++k) {
                sum += (long double)A[i * K + k] * (long double)B[k * N + j];
            }
            C[i * ldc + j] = (double)sum;
        }
    }
}

void oracle_gemv_f32(int M, int K, const float *A, const float *x, float *y) {
    for (int i = 0; i < M; ++i) {
        double sum = 0.0;
        for (int k = 0; k < K; ++k) {
            sum += (double)A[i * K + k] * (double)x[k];
        }
        y[i] = (float)sum;
    }
}

void oracle_gevm_f32(int M, int N, const float *x, const float *A, float *y) {
    for (int j = 0; j < N; ++j) {
        double sum = 0.0;
        for (int i = 0; i < M; ++i) {
            sum += (double)x[i] * (double)A[i * N + j];
        }
        y[j] = (float)sum;
    }
}

float oracle_dot_f32(int N, const float *x, const float *y) {
    double sum = 0.0;
    for (int i = 0; i < N; ++i) {
        sum += (double)x[i] * (double)y[i];
    }
    return (float)sum;
}

void oracle_ger_f32(int M, int N, const float *x, const float *y, float *A) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            A[i * N + j] += x[i] * y[j];
        }
    }
}

// ============================================================================
// 3. Implementations Across Operation Families
// ============================================================================

// GEMM FP32: Tiled IKJ SIMD
void gemm_tiled_ikj_f32(int M, int N, int K, const float * __restrict__ A, const float * __restrict__ B, float * __restrict__ C) {
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

// GEMM FP32: Eigen 3
void gemm_eigen_f32(int M, int N, int K, const float *A, const float *B, float *C) {
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> matA(A, M, K);
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> matB(B, K, N);
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> matC(C, M, N);
    matC.noalias() = matA * matB;
}

// GEMM FP64: Eigen 3
void gemm_eigen_f64(int M, int N, int K, const double *A, const double *B, double *C) {
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> matA(A, M, K);
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> matB(B, K, N);
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> matC(C, M, N);
    matC.noalias() = matA * matB;
}

// GEMV FP32: SIMD Row Reduction
void gemv_simd_f32(int M, int K, const float * __restrict__ A, const float * __restrict__ x, float * __restrict__ y) {
    for (int i = 0; i < M; ++i) {
        __m256 acc = _mm256_setzero_ps();
        int k = 0;
        for (; k + 8 <= K; k += 8) {
            __m256 a_vec = _mm256_loadu_ps(&A[i * K + k]);
            __m256 x_vec = _mm256_loadu_ps(&x[k]);
            acc = _mm256_fmadd_ps(a_vec, x_vec, acc);
        }
        // Horizontal reduction of acc
        __m128 lo = _mm256_castps256_ps128(acc);
        __m128 hi = _mm256_extractf128_ps(acc, 1);
        __m128 sum128 = _mm_add_ps(lo, hi);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        float sum = _mm_cvtss_f32(sum128);
        for (; k < K; ++k) {
            sum += A[i * K + k] * x[k];
        }
        y[i] = sum;
    }
}

// GEMV FP32: Eigen 3
void gemv_eigen_f32(int M, int K, const float *A, const float *x, float *y) {
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> matA(A, M, K);
    Eigen::Map<const Eigen::VectorXf> vecX(x, K);
    Eigen::Map<Eigen::VectorXf> vecY(y, M);
    vecY.noalias() = matA * vecX;
}

// GEVM FP32: SIMD Column Accumulation (Outer-Vector Accumulation)
void gevm_simd_f32(int M, int N, const float * __restrict__ x, const float * __restrict__ A, float * __restrict__ y) {
    for (int j = 0; j < N; ++j) y[j] = 0.0f;
    for (int i = 0; i < M; ++i) {
        float x_val = x[i];
        __m256 x_vec = _mm256_set1_ps(x_val);
        int j = 0;
        for (; j + 8 <= N; j += 8) {
            __m256 a_vec = _mm256_loadu_ps(&A[i * N + j]);
            __m256 y_vec = _mm256_loadu_ps(&y[j]);
            y_vec = _mm256_fmadd_ps(x_vec, a_vec, y_vec);
            _mm256_storeu_ps(&y[j], y_vec);
        }
        for (; j < N; ++j) {
            y[j] += x_val * A[i * N + j];
        }
    }
}

// GEVM FP32: Eigen 3
void gevm_eigen_f32(int M, int N, const float *x, const float *A, float *y) {
    Eigen::Map<const Eigen::RowVectorXf> vecX(x, M);
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> matA(A, M, N);
    Eigen::Map<Eigen::RowVectorXf> vecY(y, N);
    vecY.noalias() = vecX * matA;
}

// GEVV-INNER (Dot Product) FP32: SIMD
float dot_simd_f32(int N, const float * __restrict__ x, const float * __restrict__ y) {
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= N; i += 8) {
        __m256 vx = _mm256_loadu_ps(&x[i]);
        __m256 vy = _mm256_loadu_ps(&y[i]);
        acc = _mm256_fmadd_ps(vx, vy, acc);
    }
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    float sum = _mm_cvtss_f32(sum128);
    for (; i < N; ++i) {
        sum += x[i] * y[i];
    }
    return sum;
}

// GEVV-OUTER (Rank-1 Update) FP32: SIMD
void ger_simd_f32(int M, int N, const float * __restrict__ x, const float * __restrict__ y, float * __restrict__ A) {
    for (int i = 0; i < M; ++i) {
        float x_val = x[i];
        __m256 x_vec = _mm256_set1_ps(x_val);
        int j = 0;
        for (; j + 8 <= N; j += 8) {
            __m256 a_vec = _mm256_loadu_ps(&A[i * N + j]);
            __m256 y_vec = _mm256_loadu_ps(&y[j]);
            a_vec = _mm256_fmadd_ps(x_vec, y_vec, a_vec);
            _mm256_storeu_ps(&A[i * N + j], a_vec);
        }
        for (; j < N; ++j) {
            A[i * N + j] += x_val * y[j];
        }
    }
}

// ============================================================================
// 4. Main Campaign Execution Suite
// ============================================================================

int main(int argc, char **argv) {
    FILE *f_csv = fopen("corpus/findings/cpu/campaign_benchmark_results.csv", "w");
    FILE *f_jsonl = fopen("corpus/findings/cpu/campaign_benchmark_results.jsonl", "w");
    if (!f_csv || !f_jsonl) {
        printf("Failed to create result files\n");
        return 1;
    }

    fprintf(f_csv, "experiment_id,operation,dtype,M,N,K,provider,median_ns,mean_ns,p10_ns,p90_ns,gflops,bandwidth_gbps,error_max,status\n");

    printf("===================================================================================\n");
    printf("   MDSLC CPU EXECUTION ARCHAEOLOGY: 5-OPERATION RUNTIME CAMPAIGN\n");
    printf("===================================================================================\n");
    printf("Operation\tShape\t\tProvider\tMedian(ns)\tGFLOP/s\t\tBW(GB/s)\tStatus\n");
    printf("-----------------------------------------------------------------------------------\n");

    // 1. GEMM FP32 Sweep
    int gemm_shapes[] = { 1, 2, 4, 8, 16, 32, 64, 128, 256, 512 };
    int num_gemm = sizeof(gemm_shapes) / sizeof(gemm_shapes[0]);

    for (int s = 0; s < num_gemm; ++s) {
        int N = gemm_shapes[s];
        int M = N, K = N;
        size_t bytes = (size_t)N * N * sizeof(float);

        float *A = (float *)_aligned_malloc(bytes, 64);
        float *B = (float *)_aligned_malloc(bytes, 64);
        float *C_tiled = (float *)_aligned_malloc(bytes, 64);
        float *C_eigen = (float *)_aligned_malloc(bytes, 64);
        float *C_ref = (float *)_aligned_malloc(bytes, 64);

        for (int i = 0; i < N * N; ++i) {
            A[i] = ((float)rand() / RAND_MAX) - 0.5f;
            B[i] = ((float)rand() / RAND_MAX) - 0.5f;
        }

        // Oracle
        oracle_gemm_f32(M, N, K, A, B, C_ref, N);
        gemm_tiled_ikj_f32(M, N, K, A, B, C_tiled);
        gemm_eigen_f32(M, N, K, A, B, C_eigen);

        double err_tiled = 0.0, err_eigen = 0.0;
        for (int i = 0; i < N * N; ++i) {
            double d_t = fabs(C_tiled[i] - C_ref[i]);
            double d_e = fabs(C_eigen[i] - C_ref[i]);
            if (d_t > err_tiled) err_tiled = d_t;
            if (d_e > err_eigen) err_eigen = d_e;
        }

        int iters = (N <= 32) ? 100000 : (N <= 128 ? 5000 : 50);
        double total_flops = 2.0 * (double)M * N * K;
        double total_bytes = 3.0 * (double)M * N * sizeof(float);

        // Benchmark Tiled IKJ
        std::vector<double> times_tiled;
        for (int it = 0; it < iters; ++it) {
            double t0 = get_time_sec();
            gemm_tiled_ikj_f32(M, N, K, A, B, C_tiled);
            double t1 = get_time_sec();
            times_tiled.push_back((t1 - t0) * 1e9);
        }
        BenchStats st_tiled = compute_stats(times_tiled);
        double gflops_tiled = (total_flops / st_tiled.median_ns);
        double bw_tiled = (total_bytes / st_tiled.median_ns);

        // Benchmark Eigen
        std::vector<double> times_eigen;
        for (int it = 0; it < iters; ++it) {
            double t0 = get_time_sec();
            gemm_eigen_f32(M, N, K, A, B, C_eigen);
            double t1 = get_time_sec();
            times_eigen.push_back((t1 - t0) * 1e9);
        }
        BenchStats st_eigen = compute_stats(times_eigen);
        double gflops_eigen = (total_flops / st_eigen.median_ns);
        double bw_eigen = (total_bytes / st_eigen.median_ns);

        printf("GEMM_FP32\t%dx%d\t\tTiled_SIMD\t%.2f\t\t%.2f\t\t%.2f\t\t%s\n", N, N, st_tiled.median_ns, gflops_tiled, bw_tiled, err_tiled < 1e-4 ? "PASS" : "FAIL");
        printf("GEMM_FP32\t%dx%d\t\tEigen3\t\t%.2f\t\t%.2f\t\t%.2f\t\t%s\n", N, N, st_eigen.median_ns, gflops_eigen, bw_eigen, err_eigen < 1e-4 ? "PASS" : "FAIL");

        fprintf(f_csv, "EXP-GEMM-F32-%d,GEMM,f32,%d,%d,%d,Tiled_SIMD,%.2f,%.2f,%.2f,%.2f,%.4f,%.4f,%.6e,%s\n",
                N, M, N, K, st_tiled.median_ns, st_tiled.mean_ns, st_tiled.p10_ns, st_tiled.p90_ns, gflops_tiled, bw_tiled, err_tiled, err_tiled < 1e-4 ? "PASS" : "FAIL");
        fprintf(f_csv, "EXP-GEMM-F32-%d,GEMM,f32,%d,%d,%d,Eigen3,%.2f,%.2f,%.2f,%.2f,%.4f,%.4f,%.6e,%s\n",
                N, M, N, K, st_eigen.median_ns, st_eigen.mean_ns, st_eigen.p10_ns, st_eigen.p90_ns, gflops_eigen, bw_eigen, err_eigen, err_eigen < 1e-4 ? "PASS" : "FAIL");

        fprintf(f_jsonl, "{\"experiment_id\":\"EXP-GEMM-F32-%d\",\"operation\":\"GEMM\",\"dtype\":\"f32\",\"M\":%d,\"N\":%d,\"K\":%d,\"provider\":\"Tiled_SIMD\",\"median_ns\":%.2f,\"gflops\":%.4f,\"error_max\":%.6e,\"status\":\"%s\"}\n",
                N, M, N, K, st_tiled.median_ns, gflops_tiled, err_tiled, err_tiled < 1e-4 ? "PASS" : "FAIL");
        fprintf(f_jsonl, "{\"experiment_id\":\"EXP-GEMM-F32-%d\",\"operation\":\"GEMM\",\"dtype\":\"f32\",\"M\":%d,\"N\":%d,\"K\":%d,\"provider\":\"Eigen3\",\"median_ns\":%.2f,\"gflops\":%.4f,\"error_max\":%.6e,\"status\":\"%s\"}\n",
                N, M, N, K, st_eigen.median_ns, gflops_eigen, err_eigen, err_eigen < 1e-4 ? "PASS" : "FAIL");

        _aligned_free(A);
        _aligned_free(B);
        _aligned_free(C_tiled);
        _aligned_free(C_eigen);
        _aligned_free(C_ref);
    }

    // 2. GEMV vs GEVM Comparison Sweep (M x K matrix by vector)
    struct Shape2D { int M; int K; };
    Shape2D gemv_shapes[] = {
        { 16, 16 }, { 64, 64 }, { 256, 256 }, { 1024, 1024 }, { 4096, 4096 },
        { 16, 4096 }, { 4096, 16 } // Extreme tall-skinny vs short-wide
    };
    int num_gemv = sizeof(gemv_shapes) / sizeof(gemv_shapes[0]);

    for (int s = 0; s < num_gemv; ++s) {
        int M = gemv_shapes[s].M;
        int K = gemv_shapes[s].K;
        size_t mat_bytes = (size_t)M * K * sizeof(float);

        float *A = (float *)_aligned_malloc(mat_bytes, 64);
        float *x_gemv = (float *)_aligned_malloc(K * sizeof(float), 64);
        float *y_gemv_simd = (float *)_aligned_malloc(M * sizeof(float), 64);
        float *y_gemv_eigen = (float *)_aligned_malloc(M * sizeof(float), 64);
        float *y_gemv_ref = (float *)_aligned_malloc(M * sizeof(float), 64);

        float *x_gevm = (float *)_aligned_malloc(M * sizeof(float), 64);
        float *y_gevm_simd = (float *)_aligned_malloc(K * sizeof(float), 64);
        float *y_gevm_eigen = (float *)_aligned_malloc(K * sizeof(float), 64);
        float *y_gevm_ref = (float *)_aligned_malloc(K * sizeof(float), 64);

        for (int i = 0; i < M * K; ++i) A[i] = ((float)rand() / RAND_MAX) - 0.5f;
        for (int i = 0; i < K; ++i) x_gemv[i] = ((float)rand() / RAND_MAX) - 0.5f;
        for (int i = 0; i < M; ++i) x_gevm[i] = ((float)rand() / RAND_MAX) - 0.5f;

        // Oracle GEMV & GEVM
        oracle_gemv_f32(M, K, A, x_gemv, y_gemv_ref);
        gemv_simd_f32(M, K, A, x_gemv, y_gemv_simd);
        gemv_eigen_f32(M, K, A, x_gemv, y_gemv_eigen);

        oracle_gevm_f32(M, K, x_gevm, A, y_gevm_ref);
        gevm_simd_f32(M, K, x_gevm, A, y_gevm_simd);
        gevm_eigen_f32(M, K, x_gevm, A, y_gevm_eigen);

        double err_gemv_simd = 0.0, err_gemv_eigen = 0.0;
        for (int i = 0; i < M; ++i) {
            double d_s = fabs(y_gemv_simd[i] - y_gemv_ref[i]);
            double d_e = fabs(y_gemv_eigen[i] - y_gemv_ref[i]);
            if (d_s > err_gemv_simd) err_gemv_simd = d_s;
            if (d_e > err_gemv_eigen) err_gemv_eigen = d_e;
        }

        double err_gevm_simd = 0.0, err_gevm_eigen = 0.0;
        for (int j = 0; j < K; ++j) {
            double d_s = fabs(y_gevm_simd[j] - y_gevm_ref[j]);
            double d_e = fabs(y_gevm_eigen[j] - y_gevm_ref[j]);
            if (d_s > err_gevm_simd) err_gevm_simd = d_s;
            if (d_e > err_gevm_eigen) err_gevm_eigen = d_e;
        }

        int iters = (M * K <= 65536) ? 50000 : (M * K <= 1048576 ? 5000 : 200);
        double total_flops = 2.0 * (double)M * K;
        double total_bytes = (double)M * K * sizeof(float) + (double)(M + K) * sizeof(float);

        // Time GEMV SIMD vs GEVM SIMD
        std::vector<double> times_gemv_simd, times_gevm_simd;
        for (int it = 0; it < iters; ++it) {
            double t0 = get_time_sec();
            gemv_simd_f32(M, K, A, x_gemv, y_gemv_simd);
            double t1 = get_time_sec();
            times_gemv_simd.push_back((t1 - t0) * 1e9);

            double t2 = get_time_sec();
            gevm_simd_f32(M, K, x_gevm, A, y_gevm_simd);
            double t3 = get_time_sec();
            times_gevm_simd.push_back((t3 - t2) * 1e9);
        }

        BenchStats st_gemv_simd = compute_stats(times_gemv_simd);
        BenchStats st_gevm_simd = compute_stats(times_gevm_simd);

        double bw_gemv = total_bytes / st_gemv_simd.median_ns;
        double bw_gevm = total_bytes / st_gevm_simd.median_ns;

        printf("GEMV_FP32\t%dx%d\t\tSIMD_RowRed\t%.2f\t\t%.2f\t\t%.2f\t\t%s\n", M, K, st_gemv_simd.median_ns, total_flops / st_gemv_simd.median_ns, bw_gemv, err_gemv_simd < 1e-4 ? "PASS" : "FAIL");
        printf("GEVM_FP32\t%dx%d\t\tSIMD_ColAcc\t%.2f\t\t%.2f\t\t%.2f\t\t%s\n", M, K, st_gevm_simd.median_ns, total_flops / st_gevm_simd.median_ns, bw_gevm, err_gevm_simd < 1e-4 ? "PASS" : "FAIL");

        fprintf(f_csv, "EXP-GEMV-F32-%dx%d,GEMV,f32,%d,1,%d,SIMD_RowRed,%.2f,%.2f,%.2f,%.2f,%.4f,%.4f,%.6e,%s\n",
                M, K, M, K, st_gemv_simd.median_ns, st_gemv_simd.mean_ns, st_gemv_simd.p10_ns, st_gemv_simd.p90_ns, total_flops / st_gemv_simd.median_ns, bw_gemv, err_gemv_simd, err_gemv_simd < 1e-4 ? "PASS" : "FAIL");
        fprintf(f_csv, "EXP-GEVM-F32-%dx%d,GEVM,f32,1,%d,%d,SIMD_ColAcc,%.2f,%.2f,%.2f,%.2f,%.4f,%.4f,%.6e,%s\n",
                M, K, K, M, st_gevm_simd.median_ns, st_gevm_simd.mean_ns, st_gevm_simd.p10_ns, st_gevm_simd.p90_ns, total_flops / st_gevm_simd.median_ns, bw_gevm, err_gevm_simd, err_gevm_simd < 1e-4 ? "PASS" : "FAIL");

        _aligned_free(A);
        _aligned_free(x_gemv);
        _aligned_free(y_gemv_simd);
        _aligned_free(y_gemv_eigen);
        _aligned_free(y_gemv_ref);
        _aligned_free(x_gevm);
        _aligned_free(y_gevm_simd);
        _aligned_free(y_gevm_eigen);
        _aligned_free(y_gevm_ref);
    }

    // 3. GEVV-INNER (Dot Product) & GEVV-OUTER (Rank-1 Update) Sweep
    int dot_sizes[] = { 16, 64, 256, 1024, 4096, 16384, 65536, 262144 };
    int num_dot = sizeof(dot_sizes) / sizeof(dot_sizes[0]);

    for (int s = 0; s < num_dot; ++s) {
        int N = dot_sizes[s];
        float *x = (float *)_aligned_malloc(N * sizeof(float), 64);
        float *y = (float *)_aligned_malloc(N * sizeof(float), 64);
        for (int i = 0; i < N; ++i) {
            x[i] = ((float)rand() / RAND_MAX) - 0.5f;
            y[i] = ((float)rand() / RAND_MAX) - 0.5f;
        }

        float ref_dot = oracle_dot_f32(N, x, y);
        float simd_dot = dot_simd_f32(N, x, y);
        double err_dot = fabs(simd_dot - ref_dot);

        int iters = (N <= 1024) ? 200000 : 20000;
        std::vector<double> times_dot;
        for (int it = 0; it < iters; ++it) {
            double t0 = get_time_sec();
            float res = dot_simd_f32(N, x, y);
            double t1 = get_time_sec();
            times_dot.push_back((t1 - t0) * 1e9);
        }
        BenchStats st_dot = compute_stats(times_dot);
        double total_flops = 2.0 * N;
        double total_bytes = 2.0 * N * sizeof(float);
        double bw_dot = total_bytes / st_dot.median_ns;

        printf("GEVV_DOT\tN=%d\t\tSIMD_Dot\t%.2f\t\t%.2f\t\t%.2f\t\t%s\n", N, st_dot.median_ns, total_flops / st_dot.median_ns, bw_dot, err_dot < 1e-3 ? "PASS" : "FAIL");

        fprintf(f_csv, "EXP-DOT-F32-%d,GEVV-INNER,f32,1,1,%d,SIMD_Dot,%.2f,%.2f,%.2f,%.2f,%.4f,%.4f,%.6e,%s\n",
                N, N, st_dot.median_ns, st_dot.mean_ns, st_dot.p10_ns, st_dot.p90_ns, total_flops / st_dot.median_ns, bw_dot, err_dot, err_dot < 1e-3 ? "PASS" : "FAIL");

        _aligned_free(x);
        _aligned_free(y);
    }

    printf("===================================================================================\n");
    printf("Campaign Benchmark Completed. Results written to CSV & JSONL.\n");

    fclose(f_csv);
    fclose(f_jsonl);
    return 0;
}
