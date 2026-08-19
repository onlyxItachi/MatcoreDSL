#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <immintrin.h>
#include <windows.h>

// Forward declarations
void microkernel_separate_relu(int K, const float * __restrict__ A, const float * __restrict__ B, float * __restrict__ C, int ldc);
void microkernel_true_fused_relu(int K, const float * __restrict__ A, const float * __restrict__ B, float * __restrict__ C, int ldc);

// Scalar Reference Oracle (Double Precision)
void oracle_gemm_relu(int M, int N, int K, const float *A, const float *B, float *C, int ldc) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int k = 0; k < K; ++k) {
                sum += (double)A[k * M + i] * (double)B[k * N + j];
            }
            double val = sum > 0.0 ? sum : 0.0;
            C[j * ldc + i] = (float)val;
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
    int test_shapes[] = { 16, 32, 64, 128, 256 };
    int num_shapes = 5;
    int k_depths[] = { 1, 4, 16, 64, 256 };
    int num_k = 5;

    FILE *f_csv = fopen("proof/fusion_epilogue/BENCHMARK_RESULTS.csv", "w");
    FILE *f_json = fopen("proof/fusion_epilogue/BENCHMARK_RESULTS.json", "w");
    if (!f_csv || !f_json) {
        printf("Failed to open output files\n");
        return 1;
    }

    fprintf(f_csv, "M,N,K,Separate_Time_ns,Fused_Time_ns,Fusion_Speedup,Oracle_Verified\n");
    fprintf(f_json, "[\n");

    printf("=== Host CPU True Register Fusion Benchmark ===\n");
    printf("Shape(MxNxK)\tSeparate(ns)\tFused(ns)\tSpeedup\tOracle\n");

    int json_first = 1;

    for (int s = 0; s < num_shapes; ++s) {
        int N = test_shapes[s];
        int M = 16; // 16x4 tile microkernel benchmarked over M=16, N=4 panel

        for (int ki = 0; ki < num_k; ++ki) {
            int K = k_depths[ki];
            int ldc = 16;

            float *A = (float *)_aligned_malloc(16 * K * sizeof(float), 64);
            float *B = (float *)_aligned_malloc(4 * K * sizeof(float), 64);
            float *C_sep = (float *)_aligned_malloc(16 * 4 * sizeof(float), 64);
            float *C_fused = (float *)_aligned_malloc(16 * 4 * sizeof(float), 64);
            float *C_ref = (float *)_aligned_malloc(16 * 4 * sizeof(float), 64);

            for (int i = 0; i < 16 * K; ++i) A[i] = ((float)rand() / RAND_MAX) - 0.5f;
            for (int i = 0; i < 4 * K; ++i) B[i] = ((float)rand() / RAND_MAX) - 0.5f;

            // 1. Oracle Verification
            oracle_gemm_relu(16, 4, K, A, B, C_ref, ldc);
            microkernel_separate_relu(K, A, B, C_sep, ldc);
            microkernel_true_fused_relu(K, A, B, C_fused, ldc);

            int verified = 1;
            for (int i = 0; i < 16 * 4; ++i) {
                if (fabs(C_sep[i] - C_ref[i]) > 1e-4f || fabs(C_fused[i] - C_ref[i]) > 1e-4f) {
                    verified = 0;
                    break;
                }
            }

            int iters = (K <= 16) ? 500000 : (K <= 64 ? 100000 : 25000);

            // Warmup
            for (int it = 0; it < 1000; ++it) {
                microkernel_separate_relu(K, A, B, C_sep, ldc);
                microkernel_true_fused_relu(K, A, B, C_fused, ldc);
            }

            // Benchmark Separate
            double t0 = get_time_sec();
            for (int it = 0; it < iters; ++it) {
                microkernel_separate_relu(K, A, B, C_sep, ldc);
            }
            double t1 = get_time_sec();
            double time_sep_ns = ((t1 - t0) / iters) * 1e9;

            // Benchmark Fused
            double t2 = get_time_sec();
            for (int it = 0; it < iters; ++it) {
                microkernel_true_fused_relu(K, A, B, C_fused, ldc);
            }
            double t3 = get_time_sec();
            double time_fused_ns = ((t3 - t2) / iters) * 1e9;

            double speedup = time_sep_ns / time_fused_ns;

            printf("16x4x%d\t\t%.2f\t\t%.2f\t\t%.2fx\t%s\n", K, time_sep_ns, time_fused_ns, speedup, verified ? "PASS" : "FAIL");
            fprintf(f_csv, "16,4,%d,%.2f,%.2f,%.4f,%s\n", K, time_sep_ns, time_fused_ns, speedup, verified ? "PASS" : "FAIL");

            if (!json_first) fprintf(f_json, ",\n");
            json_first = 0;
            fprintf(f_json, "  {\n    \"M\": 16, \"N\": 4, \"K\": %d,\n    \"separate_ns\": %.2f,\n    \"fused_ns\": %.2f,\n    \"speedup\": %.4f,\n    \"verified\": %s\n  }", K, time_sep_ns, time_fused_ns, speedup, verified ? "true" : "false");

            _aligned_free(A);
            _aligned_free(B);
            _aligned_free(C_sep);
            _aligned_free(C_fused);
            _aligned_free(C_ref);
        }
        break; // Microkernel is 16x4 tile
    }

    fprintf(f_json, "\n]\n");
    fclose(f_csv);
    fclose(f_json);

    return 0;
}
