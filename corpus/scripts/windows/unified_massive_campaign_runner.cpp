#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <immintrin.h>
#include <windows.h>
#include <vector>
#include <string>
#include <chrono>
#include <iostream>
#include <omp.h>
#include <Eigen/Dense>

// ============================================================================
// Timer
// ============================================================================
static double get_time_sec() {
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart / (double)freq.QuadPart;
}

// ============================================================================
// Configuration Data Structure
// ============================================================================
enum class OpType { GEMM, GEMV, GEVM, GEVV_DOT, GEVV_GER };
enum class DType { FP32, FP64 };
enum class Layout { NN, NT, TN, TT };

struct ExperimentConfig {
    int id;
    OpType op;
    DType dtype;
    int M, N, K;
    Layout layout;
    int stride;
    int threads;
    std::string provider;
    std::string semantic;
};

// ============================================================================
// Double Precision Mathematical Reference Oracles
// ============================================================================
static void oracle_gemm(int M, int N, int K, Layout layout, const double *A, const double *B, double *C) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int k = 0; k < K; ++k) {
                double a_val = (layout == Layout::TN || layout == Layout::TT) ? A[k * M + i] : A[i * K + k];
                double b_val = (layout == Layout::NT || layout == Layout::TT) ? B[j * K + k] : B[k * N + j];
                sum += a_val * b_val;
            }
            C[i * N + j] = sum;
        }
    }
}

static void oracle_gemv(int M, int K, int stride, const double *A, const double *x, double *y) {
    for (int i = 0; i < M; ++i) {
        double sum = 0.0;
        for (int k = 0; k < K; ++k) {
            sum += A[i * K + k] * x[k * stride];
        }
        y[i] = sum;
    }
}

static void oracle_gevm(int M, int N, int stride, const double *x, const double *A, double *y) {
    for (int j = 0; j < N; ++j) y[j] = 0.0;
    for (int i = 0; i < M; ++i) {
        double x_val = x[i * stride];
        for (int j = 0; j < N; ++j) {
            y[j] += x_val * A[i * N + j];
        }
    }
}

static double oracle_dot(int N, int stride, const double *x, const double *y) {
    double sum = 0.0;
    for (int i = 0; i < N; ++i) sum += x[i * stride] * y[i * stride];
    return sum;
}

static void oracle_ger(int M, int N, int stride, const double *x, const double *y, double *A) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            A[i * N + j] += x[i * stride] * y[j * stride];
        }
    }
}

// ============================================================================
// Computational Kernels: FP32
// ============================================================================

// FP32 GEMM SIMD Multi-threaded
static void run_gemm_f32_simd(int M, int N, int K, Layout layout, const float *A, const float *B, float *C, int threads) {
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) C[i * N + j] = 0.0f;
        for (int k = 0; k < K; ++k) {
            float a_val = (layout == Layout::TN || layout == Layout::TT) ? A[k * M + i] : A[i * K + k];
            __m256 a_vec = _mm256_set1_ps(a_val);
            int j = 0;
            if (layout == Layout::NN || layout == Layout::TN) {
                for (; j + 8 <= N; j += 8) {
                    __m256 b_vec = _mm256_loadu_ps(&B[k * N + j]);
                    __m256 c_vec = _mm256_loadu_ps(&C[i * N + j]);
                    c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
                    _mm256_storeu_ps(&C[i * N + j], c_vec);
                }
                for (; j < N; ++j) C[i * N + j] += a_val * B[k * N + j];
            } else {
                for (; j < N; ++j) {
                    float b_val = (layout == Layout::NT || layout == Layout::TT) ? B[j * K + k] : B[k * N + j];
                    C[i * N + j] += a_val * b_val;
                }
            }
        }
    }
}

// FP32 GEMM Eigen
static void run_gemm_f32_eigen(int M, int N, int K, const float *A, const float *B, float *C, int threads) {
    Eigen::setNbThreads(threads);
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> matA(A, M, K);
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> matB(B, K, N);
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> matC(C, M, N);
    matC.noalias() = matA * matB;
}

// FP32 GEMV SIMD
static void run_gemv_f32_simd(int M, int K, int stride, const float *A, const float *x, float *y, int threads) {
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (int i = 0; i < M; ++i) {
        float sum = 0.0f;
        int k = 0;
        if (stride == 1) {
            __m256 acc = _mm256_setzero_ps();
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
            sum = _mm_cvtss_f32(sum128);
        }
        for (; k < K; ++k) sum += A[i * K + k] * x[k * stride];
        y[i] = sum;
    }
}

// FP32 GEVM SIMD
static void run_gevm_f32_simd(int M, int N, int stride, const float *x, const float *A, float *y, int threads) {
    for (int j = 0; j < N; ++j) y[j] = 0.0f;
    for (int i = 0; i < M; ++i) {
        float x_val = x[i * stride];
        __m256 x_vec = _mm256_set1_ps(x_val);
        int j = 0;
        for (; j + 8 <= N; j += 8) {
            __m256 a_vec = _mm256_loadu_ps(&A[i * N + j]);
            __m256 y_vec = _mm256_loadu_ps(&y[j]);
            y_vec = _mm256_fmadd_ps(x_vec, a_vec, y_vec);
            _mm256_storeu_ps(&y[j], y_vec);
        }
        for (; j < N; ++j) y[j] += x_val * A[i * N + j];
    }
}

// FP32 Dot SIMD
static float run_dot_f32_simd(int N, int stride, const float *x, const float *y) {
    float sum = 0.0f;
    int i = 0;
    if (stride == 1) {
        __m256 acc = _mm256_setzero_ps();
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
        sum = _mm_cvtss_f32(sum128);
    }
    for (; i < N; ++i) sum += x[i * stride] * y[i * stride];
    return sum;
}

// FP32 GER SIMD
static void run_ger_f32_simd(int M, int N, int stride, const float *x, const float *y, float *A, int threads) {
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (int i = 0; i < M; ++i) {
        float x_val = x[i * stride];
        __m256 x_vec = _mm256_set1_ps(x_val);
        int j = 0;
        if (stride == 1) {
            for (; j + 8 <= N; j += 8) {
                __m256 y_vec = _mm256_loadu_ps(&y[j]);
                __m256 a_vec = _mm256_loadu_ps(&A[i * N + j]);
                a_vec = _mm256_fmadd_ps(x_vec, y_vec, a_vec);
                _mm256_storeu_ps(&A[i * N + j], a_vec);
            }
        }
        for (; j < N; ++j) A[i * N + j] += x_val * y[j * stride];
    }
}

// ============================================================================
// Computational Kernels: FP64
// ============================================================================

static void run_gemm_f64_simd(int M, int N, int K, Layout layout, const double *A, const double *B, double *C, int threads) {
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) C[i * N + j] = 0.0;
        for (int k = 0; k < K; ++k) {
            double a_val = (layout == Layout::TN || layout == Layout::TT) ? A[k * M + i] : A[i * K + k];
            __m256d a_vec = _mm256_set1_pd(a_val);
            int j = 0;
            if (layout == Layout::NN || layout == Layout::TN) {
                for (; j + 4 <= N; j += 4) {
                    __m256d b_vec = _mm256_loadu_pd(&B[k * N + j]);
                    __m256d c_vec = _mm256_loadu_pd(&C[i * N + j]);
                    c_vec = _mm256_fmadd_pd(a_vec, b_vec, c_vec);
                    _mm256_storeu_pd(&C[i * N + j], c_vec);
                }
                for (; j < N; ++j) C[i * N + j] += a_val * B[k * N + j];
            } else {
                for (; j < N; ++j) {
                    double b_val = (layout == Layout::NT || layout == Layout::TT) ? B[j * K + k] : B[k * N + j];
                    C[i * N + j] += a_val * b_val;
                }
            }
        }
    }
}

static void run_gemm_f64_eigen(int M, int N, int K, const double *A, const double *B, double *C, int threads) {
    Eigen::setNbThreads(threads);
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> matA(A, M, K);
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> matB(B, K, N);
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> matC(C, M, N);
    matC.noalias() = matA * matB;
}

static void run_gemv_f64_simd(int M, int K, int stride, const double *A, const double *x, double *y, int threads) {
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (int i = 0; i < M; ++i) {
        double sum = 0.0;
        int k = 0;
        if (stride == 1) {
            __m256d acc = _mm256_setzero_pd();
            for (; k + 4 <= K; k += 4) {
                __m256d a_vec = _mm256_loadu_pd(&A[i * K + k]);
                __m256d x_vec = _mm256_loadu_pd(&x[k]);
                acc = _mm256_fmadd_pd(a_vec, x_vec, acc);
            }
            __m128d lo = _mm256_extractf128_pd(acc, 0);
            __m128d hi = _mm256_extractf128_pd(acc, 1);
            __m128d sum128 = _mm_add_pd(lo, hi);
            sum128 = _mm_hadd_pd(sum128, sum128);
            sum = _mm_cvtsd_f64(sum128);
        }
        for (; k < K; ++k) sum += A[i * K + k] * x[k * stride];
        y[i] = sum;
    }
}

static void run_gevm_f64_simd(int M, int N, int stride, const double *x, const double *A, double *y, int threads) {
    for (int j = 0; j < N; ++j) y[j] = 0.0;
    for (int i = 0; i < M; ++i) {
        double x_val = x[i * stride];
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

static double run_dot_f64_simd(int N, int stride, const double *x, const double *y) {
    double sum = 0.0;
    int i = 0;
    if (stride == 1) {
        __m256d acc = _mm256_setzero_pd();
        for (; i + 4 <= N; i += 4) {
            __m256d vx = _mm256_loadu_pd(&x[i]);
            __m256d vy = _mm256_loadu_pd(&y[i]);
            acc = _mm256_fmadd_pd(vx, vy, acc);
        }
        __m128d lo = _mm256_extractf128_pd(acc, 0);
        __m128d hi = _mm256_extractf128_pd(acc, 1);
        __m128d sum128 = _mm_add_pd(lo, hi);
        sum128 = _mm_hadd_pd(sum128, sum128);
        sum = _mm_cvtsd_f64(sum128);
    }
    for (; i < N; ++i) sum += x[i * stride] * y[i * stride];
    return sum;
}

static void run_ger_f64_simd(int M, int N, int stride, const double *x, const double *y, double *A, int threads) {
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (int i = 0; i < M; ++i) {
        double x_val = x[i * stride];
        __m256d x_vec = _mm256_set1_pd(x_val);
        int j = 0;
        if (stride == 1) {
            for (; j + 4 <= N; j += 4) {
                __m256d y_vec = _mm256_loadu_pd(&y[j]);
                __m256d a_vec = _mm256_loadu_pd(&A[i * N + j]);
                a_vec = _mm256_fmadd_pd(x_vec, y_vec, a_vec);
                _mm256_storeu_pd(&A[i * N + j], a_vec);
            }
        }
        for (; j < N; ++j) A[i * N + j] += x_val * y[j * stride];
    }
}

// ============================================================================
// Campaign Generator & Orchestrator
// ============================================================================

int main() {
    std::vector<ExperimentConfig> queue;
    int exp_counter = 0;

    // 1. Grid Definition
    std::vector<int> sq_shapes = { 1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512 };
    struct RectShape { int M, N, K; };
    std::vector<RectShape> rect_shapes = {
        { 4096, 16, 64 }, { 16, 4096, 64 }, { 1024, 32, 128 }, { 32, 1024, 128 },
        { 2048, 16, 16 }, { 16, 2048, 16 }
    };
    std::vector<int> dot_sizes = { 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288 };

    std::vector<DType> dtypes = { DType::FP32, DType::FP64 };
    std::vector<Layout> gemm_layouts = { Layout::NN, Layout::NT, Layout::TN, Layout::TT };
    std::vector<int> thread_options = { 1, 2, 4, 8, 12, 24 };
    std::vector<int> strides = { 1, 2, 4 };

    // --- EXPAND GEMM ---
    for (auto dt : dtypes) {
        for (int s : sq_shapes) {
            for (auto lay : gemm_layouts) {
                for (int th : { 1, 4, 12, 24 }) {
                    queue.push_back({ ++exp_counter, OpType::GEMM, dt, s, s, s, lay, 1, th, "SIMD_AVX2", "reassoc" });
                    if (lay == Layout::NN) {
                        queue.push_back({ ++exp_counter, OpType::GEMM, dt, s, s, s, lay, 1, th, "Eigen3", "strict" });
                    }
                }
            }
        }
        for (auto r : rect_shapes) {
            for (auto lay : { Layout::NN, Layout::NT }) {
                for (int th : { 1, 12 }) {
                    queue.push_back({ ++exp_counter, OpType::GEMM, dt, r.M, r.N, r.K, lay, 1, th, "SIMD_AVX2", "reassoc" });
                }
            }
        }
    }

    // --- EXPAND GEMV & GEVM ---
    for (auto dt : dtypes) {
        for (int s : sq_shapes) {
            for (int st : strides) {
                for (int th : { 1, 4, 12, 24 }) {
                    queue.push_back({ ++exp_counter, OpType::GEMV, dt, s, 1, s, Layout::NN, st, th, "SIMD_AVX2", "reassoc" });
                    queue.push_back({ ++exp_counter, OpType::GEVM, dt, 1, s, s, Layout::NN, st, th, "SIMD_AVX2", "reassoc" });
                }
            }
        }
        for (auto r : rect_shapes) {
            for (int th : { 1, 12 }) {
                queue.push_back({ ++exp_counter, OpType::GEMV, dt, r.M, 1, r.K, Layout::NN, 1, th, "SIMD_AVX2", "reassoc" });
                queue.push_back({ ++exp_counter, OpType::GEVM, dt, 1, r.N, r.K, Layout::NN, 1, th, "SIMD_AVX2", "reassoc" });
            }
        }
    }

    // --- EXPAND GEVV-DOT ---
    for (auto dt : dtypes) {
        for (int n : dot_sizes) {
            for (int st : strides) {
                queue.push_back({ ++exp_counter, OpType::GEVV_DOT, dt, 1, 1, n, Layout::NN, st, 1, "SIMD_AVX2", "reassoc" });
            }
        }
    }

    // --- EXPAND GEVV-GER ---
    for (auto dt : dtypes) {
        for (int s : sq_shapes) {
            for (int st : strides) {
                for (int th : { 1, 12 }) {
                    queue.push_back({ ++exp_counter, OpType::GEVV_GER, dt, s, s, 1, Layout::NN, st, th, "SIMD_AVX2", "reassoc" });
                }
            }
        }
    }

    int total_experiments = (int)queue.size();
    printf("===================================================================================\n");
    printf("   MDSLC UNIFIED NATIVE CAMPAIGN RUNNER (Total Pre-Expanded Configs: %d)\n", total_experiments);
    printf("===================================================================================\n");

    FILE *f_csv = fopen("corpus/findings/cpu/massive_campaign_results.csv", "w");
    FILE *f_jsonl = fopen("corpus/findings/cpu/massive_campaign_results.jsonl", "w");
    if (!f_csv || !f_jsonl) {
        printf("Error opening result files.\n");
        return 1;
    }
    fprintf(f_csv, "id,op,dtype,M,N,K,layout,stride,threads,provider,semantic,runtime_ns,gflops,bw_gbps,error,status\n");
    fflush(f_csv);

    auto start_time_all = std::chrono::high_resolution_clock::now();

    int completed = 0, failed = 0, blocked = 0;

    for (int idx = 0; idx < total_experiments; ++idx) {
        const auto &cfg = queue[idx];

        double runtime_ns = 0.0;
        double gflops = 0.0;
        double bw_gbps = 0.0;
        double max_err = 0.0;
        std::string status = "COMPLETED";

        // Determine loop count dynamically to maintain timing precision while executing fast
        int iters = 100;
        if (cfg.op == OpType::GEMM) {
            if (cfg.N <= 16) iters = 10000;
            else if (cfg.N <= 64) iters = 2000;
            else if (cfg.N <= 128) iters = 500;
            else iters = 20;
        } else if (cfg.op == OpType::GEMV || cfg.op == OpType::GEVM) {
            if (cfg.M * cfg.K <= 1024) iters = 20000;
            else if (cfg.M * cfg.K <= 65536) iters = 2000;
            else iters = 100;
        } else {
            iters = 5000;
        }

        try {
            if (cfg.dtype == DType::FP32) {
                if (cfg.op == OpType::GEMM) {
                    size_t szA = (size_t)cfg.M * cfg.K * sizeof(float);
                    size_t szB = (size_t)cfg.K * cfg.N * sizeof(float);
                    size_t szC = (size_t)cfg.M * cfg.N * sizeof(float);
                    float *A = (float *)_aligned_malloc(szA, 64);
                    float *B = (float *)_aligned_malloc(szB, 64);
                    float *C = (float *)_aligned_malloc(szC, 64);

                    if (!A || !B || !C) {
                        status = "BLOCKED_OOM";
                        blocked++;
                    } else {
                        for (int i = 0; i < cfg.M * cfg.K; ++i) A[i] = 1.0f;
                        for (int i = 0; i < cfg.K * cfg.N; ++i) B[i] = 1.0f;

                        // Warmup
                        if (cfg.provider == "Eigen3") run_gemm_f32_eigen(cfg.M, cfg.N, cfg.K, A, B, C, cfg.threads);
                        else run_gemm_f32_simd(cfg.M, cfg.N, cfg.K, cfg.layout, A, B, C, cfg.threads);

                        // Benchmark
                        double t0 = get_time_sec();
                        for (int it = 0; it < iters; ++it) {
                            if (cfg.provider == "Eigen3") run_gemm_f32_eigen(cfg.M, cfg.N, cfg.K, A, B, C, cfg.threads);
                            else run_gemm_f32_simd(cfg.M, cfg.N, cfg.K, cfg.layout, A, B, C, cfg.threads);
                        }
                        runtime_ns = (get_time_sec() - t0) / iters * 1e9;
                        gflops = (2.0 * cfg.M * cfg.N * cfg.K) / runtime_ns;
                        bw_gbps = ((double)(szA + szB + szC)) / runtime_ns;

                        // Oracle Check
                        double expected = (double)cfg.K;
                        max_err = fabs((double)C[0] - expected);
                        if (max_err > 1e-3) { status = "FAILED_ORACLE"; failed++; }
                        else completed++;

                        _aligned_free(A); _aligned_free(B); _aligned_free(C);
                    }
                } else if (cfg.op == OpType::GEMV) {
                    size_t szA = (size_t)cfg.M * cfg.K * sizeof(float);
                    size_t szx = (size_t)cfg.K * cfg.stride * sizeof(float);
                    float *A = (float *)_aligned_malloc(szA, 64);
                    float *x = (float *)_aligned_malloc(szx, 64);
                    float *y = (float *)_aligned_malloc(cfg.M * sizeof(float), 64);
                    if (!A || !x || !y) { status = "BLOCKED_OOM"; blocked++; }
                    else {
                        for (int i = 0; i < cfg.M * cfg.K; ++i) A[i] = 1.0f;
                        for (int i = 0; i < cfg.K * cfg.stride; ++i) x[i] = 1.0f;

                        double t0 = get_time_sec();
                        for (int it = 0; it < iters; ++it) run_gemv_f32_simd(cfg.M, cfg.K, cfg.stride, A, x, y, cfg.threads);
                        runtime_ns = (get_time_sec() - t0) / iters * 1e9;
                        gflops = (2.0 * cfg.M * cfg.K) / runtime_ns;
                        bw_gbps = (double)(szA + cfg.K * sizeof(float) + cfg.M * sizeof(float)) / runtime_ns;

                        max_err = fabs((double)y[0] - (double)cfg.K);
                        if (max_err > 1e-3) { status = "FAILED_ORACLE"; failed++; }
                        else completed++;

                        _aligned_free(A); _aligned_free(x); _aligned_free(y);
                    }
                } else if (cfg.op == OpType::GEVM) {
                    size_t szA = (size_t)cfg.K * cfg.N * sizeof(float);
                    size_t szx = (size_t)cfg.K * cfg.stride * sizeof(float);
                    float *A = (float *)_aligned_malloc(szA, 64);
                    float *x = (float *)_aligned_malloc(szx, 64);
                    float *y = (float *)_aligned_malloc(cfg.N * sizeof(float), 64);
                    if (!A || !x || !y) { status = "BLOCKED_OOM"; blocked++; }
                    else {
                        for (int i = 0; i < cfg.K * cfg.N; ++i) A[i] = 1.0f;
                        for (int i = 0; i < cfg.K * cfg.stride; ++i) x[i] = 1.0f;

                        double t0 = get_time_sec();
                        for (int it = 0; it < iters; ++it) run_gevm_f32_simd(cfg.K, cfg.N, cfg.stride, x, A, y, cfg.threads);
                        runtime_ns = (get_time_sec() - t0) / iters * 1e9;
                        gflops = (2.0 * cfg.K * cfg.N) / runtime_ns;
                        bw_gbps = (double)(szA + cfg.K * sizeof(float) + cfg.N * sizeof(float)) / runtime_ns;

                        max_err = fabs((double)y[0] - (double)cfg.K);
                        if (max_err > 1e-3) { status = "FAILED_ORACLE"; failed++; }
                        else completed++;

                        _aligned_free(A); _aligned_free(x); _aligned_free(y);
                    }
                } else if (cfg.op == OpType::GEVV_DOT) {
                    size_t sz = (size_t)cfg.K * cfg.stride * sizeof(float);
                    float *x = (float *)_aligned_malloc(sz, 64);
                    float *y = (float *)_aligned_malloc(sz, 64);
                    if (!x || !y) { status = "BLOCKED_OOM"; blocked++; }
                    else {
                        for (int i = 0; i < cfg.K * cfg.stride; ++i) { x[i] = 1.0f; y[i] = 1.0f; }
                        double t0 = get_time_sec();
                        float res = 0.0f;
                        for (int it = 0; it < iters; ++it) res = run_dot_f32_simd(cfg.K, cfg.stride, x, y);
                        runtime_ns = (get_time_sec() - t0) / iters * 1e9;
                        gflops = (2.0 * cfg.K) / runtime_ns;
                        bw_gbps = (2.0 * cfg.K * sizeof(float)) / runtime_ns;

                        max_err = fabs((double)res - (double)cfg.K);
                        if (max_err > 1e-3) { status = "FAILED_ORACLE"; failed++; }
                        else completed++;

                        _aligned_free(x); _aligned_free(y);
                    }
                } else if (cfg.op == OpType::GEVV_GER) {
                    size_t szx = (size_t)cfg.M * cfg.stride * sizeof(float);
                    size_t szy = (size_t)cfg.N * cfg.stride * sizeof(float);
                    size_t szA = (size_t)cfg.M * cfg.N * sizeof(float);
                    float *x = (float *)_aligned_malloc(szx, 64);
                    float *y = (float *)_aligned_malloc(szy, 64);
                    float *A = (float *)_aligned_malloc(szA, 64);
                    if (!x || !y || !A) { status = "BLOCKED_OOM"; blocked++; }
                    else {
                        for (int i = 0; i < cfg.M * cfg.stride; ++i) x[i] = 1.0f;
                        for (int i = 0; i < cfg.N * cfg.stride; ++i) y[i] = 1.0f;
                        for (int i = 0; i < cfg.M * cfg.N; ++i) A[i] = 0.0f;

                        double t0 = get_time_sec();
                        for (int it = 0; it < iters; ++it) run_ger_f32_simd(cfg.M, cfg.N, cfg.stride, x, y, A, cfg.threads);
                        runtime_ns = (get_time_sec() - t0) / iters * 1e9;
                        gflops = (2.0 * cfg.M * cfg.N) / runtime_ns;
                        bw_gbps = (double)(szx + szy + szA) / runtime_ns;

                        max_err = fabs((double)A[0] - (double)iters);
                        if (max_err > 1e-3) { status = "FAILED_ORACLE"; failed++; }
                        else completed++;

                        _aligned_free(x); _aligned_free(y); _aligned_free(A);
                    }
                }
            } else { // FP64
                if (cfg.op == OpType::GEMM) {
                    size_t szA = (size_t)cfg.M * cfg.K * sizeof(double);
                    size_t szB = (size_t)cfg.K * cfg.N * sizeof(double);
                    size_t szC = (size_t)cfg.M * cfg.N * sizeof(double);
                    double *A = (double *)_aligned_malloc(szA, 64);
                    double *B = (double *)_aligned_malloc(szB, 64);
                    double *C = (double *)_aligned_malloc(szC, 64);

                    if (!A || !B || !C) { status = "BLOCKED_OOM"; blocked++; }
                    else {
                        for (int i = 0; i < cfg.M * cfg.K; ++i) A[i] = 1.0;
                        for (int i = 0; i < cfg.K * cfg.N; ++i) B[i] = 1.0;

                        if (cfg.provider == "Eigen3") run_gemm_f64_eigen(cfg.M, cfg.N, cfg.K, A, B, C, cfg.threads);
                        else run_gemm_f64_simd(cfg.M, cfg.N, cfg.K, cfg.layout, A, B, C, cfg.threads);

                        double t0 = get_time_sec();
                        for (int it = 0; it < iters; ++it) {
                            if (cfg.provider == "Eigen3") run_gemm_f64_eigen(cfg.M, cfg.N, cfg.K, A, B, C, cfg.threads);
                            else run_gemm_f64_simd(cfg.M, cfg.N, cfg.K, cfg.layout, A, B, C, cfg.threads);
                        }
                        runtime_ns = (get_time_sec() - t0) / iters * 1e9;
                        gflops = (2.0 * cfg.M * cfg.N * cfg.K) / runtime_ns;
                        bw_gbps = ((double)(szA + szB + szC)) / runtime_ns;

                        max_err = fabs(C[0] - (double)cfg.K);
                        if (max_err > 1e-4) { status = "FAILED_ORACLE"; failed++; }
                        else completed++;

                        _aligned_free(A); _aligned_free(B); _aligned_free(C);
                    }
                } else if (cfg.op == OpType::GEMV) {
                    size_t szA = (size_t)cfg.M * cfg.K * sizeof(double);
                    size_t szx = (size_t)cfg.K * cfg.stride * sizeof(double);
                    double *A = (double *)_aligned_malloc(szA, 64);
                    double *x = (double *)_aligned_malloc(szx, 64);
                    double *y = (double *)_aligned_malloc(cfg.M * sizeof(double), 64);
                    if (!A || !x || !y) { status = "BLOCKED_OOM"; blocked++; }
                    else {
                        for (int i = 0; i < cfg.M * cfg.K; ++i) A[i] = 1.0;
                        for (int i = 0; i < cfg.K * cfg.stride; ++i) x[i] = 1.0;

                        double t0 = get_time_sec();
                        for (int it = 0; it < iters; ++it) run_gemv_f64_simd(cfg.M, cfg.K, cfg.stride, A, x, y, cfg.threads);
                        runtime_ns = (get_time_sec() - t0) / iters * 1e9;
                        gflops = (2.0 * cfg.M * cfg.K) / runtime_ns;
                        bw_gbps = (double)(szA + cfg.K * sizeof(double) + cfg.M * sizeof(double)) / runtime_ns;

                        max_err = fabs(y[0] - (double)cfg.K);
                        if (max_err > 1e-4) { status = "FAILED_ORACLE"; failed++; }
                        else completed++;

                        _aligned_free(A); _aligned_free(x); _aligned_free(y);
                    }
                } else if (cfg.op == OpType::GEVM) {
                    size_t szA = (size_t)cfg.K * cfg.N * sizeof(double);
                    size_t szx = (size_t)cfg.K * cfg.stride * sizeof(double);
                    double *A = (double *)_aligned_malloc(szA, 64);
                    double *x = (double *)_aligned_malloc(szx, 64);
                    double *y = (double *)_aligned_malloc(cfg.N * sizeof(double), 64);
                    if (!A || !x || !y) { status = "BLOCKED_OOM"; blocked++; }
                    else {
                        for (int i = 0; i < cfg.K * cfg.N; ++i) A[i] = 1.0;
                        for (int i = 0; i < cfg.K * cfg.stride; ++i) x[i] = 1.0;

                        double t0 = get_time_sec();
                        for (int it = 0; it < iters; ++it) run_gevm_f64_simd(cfg.K, cfg.N, cfg.stride, x, A, y, cfg.threads);
                        runtime_ns = (get_time_sec() - t0) / iters * 1e9;
                        gflops = (2.0 * cfg.K * cfg.N) / runtime_ns;
                        bw_gbps = (double)(szA + cfg.K * sizeof(double) + cfg.N * sizeof(double)) / runtime_ns;

                        max_err = fabs(y[0] - (double)cfg.K);
                        if (max_err > 1e-4) { status = "FAILED_ORACLE"; failed++; }
                        else completed++;

                        _aligned_free(A); _aligned_free(x); _aligned_free(y);
                    }
                } else if (cfg.op == OpType::GEVV_DOT) {
                    size_t sz = (size_t)cfg.K * cfg.stride * sizeof(double);
                    double *x = (double *)_aligned_malloc(sz, 64);
                    double *y = (double *)_aligned_malloc(sz, 64);
                    if (!x || !y) { status = "BLOCKED_OOM"; blocked++; }
                    else {
                        for (int i = 0; i < cfg.K * cfg.stride; ++i) { x[i] = 1.0; y[i] = 1.0; }
                        double t0 = get_time_sec();
                        double res = 0.0;
                        for (int it = 0; it < iters; ++it) res = run_dot_f64_simd(cfg.K, cfg.stride, x, y);
                        runtime_ns = (get_time_sec() - t0) / iters * 1e9;
                        gflops = (2.0 * cfg.K) / runtime_ns;
                        bw_gbps = (2.0 * cfg.K * sizeof(double)) / runtime_ns;

                        max_err = fabs(res - (double)cfg.K);
                        if (max_err > 1e-4) { status = "FAILED_ORACLE"; failed++; }
                        else completed++;

                        _aligned_free(x); _aligned_free(y);
                    }
                } else if (cfg.op == OpType::GEVV_GER) {
                    size_t szx = (size_t)cfg.M * cfg.stride * sizeof(double);
                    size_t szy = (size_t)cfg.N * cfg.stride * sizeof(double);
                    size_t szA = (size_t)cfg.M * cfg.N * sizeof(double);
                    double *x = (double *)_aligned_malloc(szx, 64);
                    double *y = (double *)_aligned_malloc(szy, 64);
                    double *A = (double *)_aligned_malloc(szA, 64);
                    if (!x || !y || !A) { status = "BLOCKED_OOM"; blocked++; }
                    else {
                        for (int i = 0; i < cfg.M * cfg.stride; ++i) x[i] = 1.0;
                        for (int i = 0; i < cfg.N * cfg.stride; ++i) y[i] = 1.0;
                        for (int i = 0; i < cfg.M * cfg.N; ++i) A[i] = 0.0;

                        double t0 = get_time_sec();
                        for (int it = 0; it < iters; ++it) run_ger_f64_simd(cfg.M, cfg.N, cfg.stride, x, y, A, cfg.threads);
                        runtime_ns = (get_time_sec() - t0) / iters * 1e9;
                        gflops = (2.0 * cfg.M * cfg.N) / runtime_ns;
                        bw_gbps = (double)(szx + szy + szA) / runtime_ns;

                        max_err = fabs(A[0] - (double)iters);
                        if (max_err > 1e-4) { status = "FAILED_ORACLE"; failed++; }
                        else completed++;

                        _aligned_free(x); _aligned_free(y); _aligned_free(A);
                    }
                }
            }
        } catch (...) {
            status = "FAILED_EXCEPTION";
            failed++;
        }

        // Format and flush output
        const char *op_str = (cfg.op == OpType::GEMM ? "GEMM" : (cfg.op == OpType::GEMV ? "GEMV" : (cfg.op == OpType::GEVM ? "GEVM" : (cfg.op == OpType::GEVV_DOT ? "DOT" : "GER"))));
        const char *dt_str = (cfg.dtype == DType::FP32 ? "f32" : "f64");
        const char *lay_str = (cfg.layout == Layout::NN ? "NN" : (cfg.layout == Layout::NT ? "NT" : (cfg.layout == Layout::TN ? "TN" : "TT")));

        fprintf(f_csv, "%d,%s,%s,%d,%d,%d,%s,%d,%d,%s,%s,%.2f,%.2f,%.2f,%.6f,%s\n",
                cfg.id, op_str, dt_str, cfg.M, cfg.N, cfg.K, lay_str, cfg.stride, cfg.threads, cfg.provider.c_str(), cfg.semantic.c_str(),
                runtime_ns, gflops, bw_gbps, max_err, status.c_str());
        fflush(f_csv);

        fprintf(f_jsonl, "{\"id\":%d,\"op\":\"%s\",\"dtype\":\"%s\",\"M\":%d,\"N\":%d,\"K\":%d,\"layout\":\"%s\",\"stride\":%d,\"threads\":%d,\"provider\":\"%s\",\"semantic\":\"%s\",\"runtime_ns\":%.2f,\"gflops\":%.2f,\"bw_gbps\":%.2f,\"max_error\":%.6f,\"status\":\"%s\"}\n",
                cfg.id, op_str, dt_str, cfg.M, cfg.N, cfg.K, lay_str, cfg.stride, cfg.threads, cfg.provider.c_str(), cfg.semantic.c_str(),
                runtime_ns, gflops, bw_gbps, max_err, status.c_str());
        fflush(f_jsonl);

        if ((idx + 1) % 100 == 0 || (idx + 1) == total_experiments) {
            printf("[%4d / %4d] Progress: %5.1f%% | Completed: %4d | Failed: %2d | Blocked: %2d | Current: %s %s %dx%dx%d T%d\n",
                   idx + 1, total_experiments, ((double)(idx + 1) / total_experiments) * 100.0,
                   completed, failed, blocked, op_str, dt_str, cfg.M, cfg.N, cfg.K, cfg.threads);
        }
    }

    auto total_dur = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time_all).count();
    printf("\n===================================================================================\n");
    printf("   CAMPAIGN FINISHED: %d Executions in %.2f seconds (Avg %.2f ms / experiment)\n",
           total_experiments, total_dur / 1000.0, (double)total_dur / total_experiments);
    printf("   Summary: COMPLETED=%d | FAILED=%d | BLOCKED=%d\n", completed, failed, blocked);
    printf("===================================================================================\n");

    fclose(f_csv);
    fclose(f_jsonl);
    return 0;
}
