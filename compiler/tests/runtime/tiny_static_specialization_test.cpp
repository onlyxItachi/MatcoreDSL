#define _CRT_SECURE_NO_WARNINGS
#include "cpu_planner.h"
#include "cpu_planner_v3.h"
#include "matcore/runtime_c.h"

#include <cmath>
#include <cstdint>
#include <immintrin.h>
#include <iostream>
#include <string_view>
#include <vector>
#include <chrono>

#if defined(__clang__) || defined(__GNUC__)
#define MATCORE_RESTRICT __restrict__
#elif defined(_MSC_VER)
#define MATCORE_RESTRICT __restrict
#else
#define MATCORE_RESTRICT
#endif

namespace {

namespace planner = matcore::mdslc::planner;

int failures = 0;

void expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

// ----------------------------------------------------------------------------
// Double Precision Scalar Mathematical Golden Reference Oracle
// ----------------------------------------------------------------------------
void golden_oracle_gemm(int M, int N, int K, const float *A, const float *B, double *C) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      double sum = 0.0;
      for (int k = 0; k < K; ++k) {
        sum += static_cast<double>(A[i * K + k]) * static_cast<double>(B[k * N + j]);
      }
      C[i * N + j] = sum;
    }
  }
}

// ----------------------------------------------------------------------------
// Deterministic Non-Uniform Initialization
// ----------------------------------------------------------------------------
void init_deterministic_matrix(int rows, int cols, float *data, float seed_offset) {
  for (int i = 0; i < rows * cols; ++i) {
    // Non-uniform pseudo-random values in [-1.0, 1.0]
    data[i] = std::sin(static_cast<float>(i + 1) * 0.17f + seed_offset);
  }
}

// ----------------------------------------------------------------------------
// Compile-Time Known Static-Specialized GEMM Template
// (Direct In-Register Unrolled Microkernel)
// ----------------------------------------------------------------------------
template <int M, int N, int K>
inline void tiny_gemm_static(const float * MATCORE_RESTRICT A,
                             const float * MATCORE_RESTRICT B,
                             float * MATCORE_RESTRICT C) noexcept {
  // Degenerate Geometry 1: Dot Product (M == 1 && N == 1)
  if constexpr (M == 1 && N == 1) {
    float sum = 0.0f;
#pragma clang loop vectorize(enable) vectorize_width(8) interleave(enable)
    for (int k = 0; k < K; ++k) {
      sum += A[k] * B[k];
    }
    C[0] = sum;
    return;
  }
  // Degenerate Geometry 2: Matrix-Vector (GEMV: N == 1, M > 1)
  else if constexpr (N == 1) {
    for (int i = 0; i < M; ++i) {
      float sum = 0.0f;
      const float *row_a = &A[i * K];
#pragma clang loop vectorize(enable) vectorize_width(8) interleave(enable)
      for (int k = 0; k < K; ++k) {
        sum += row_a[k] * B[k];
      }
      C[i] = sum;
    }
    return;
  }
  // Degenerate Geometry 3: Rank-1 Outer Update (GER: K == 1, M > 1, N > 1)
  else if constexpr (K == 1) {
    for (int i = 0; i < M; ++i) {
      const float a_val = A[i];
#pragma clang loop vectorize(enable) vectorize_width(8) interleave(enable)
      for (int j = 0; j < N; ++j) {
        C[i * N + j] = a_val * B[j];
      }
    }
    return;
  }
  // General In-Register Unrolled GEMM
  else {
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        C[i * N + j] = 0.0f;
      }
      for (int k = 0; k < K; ++k) {
        const float a_val = A[i * K + k];
#pragma clang loop vectorize(enable) vectorize_width(8) interleave(enable)
        for (int j = 0; j < N; ++j) {
          C[i * N + j] += a_val * B[k * N + j];
        }
      }
    }
  }
}

// ----------------------------------------------------------------------------
// High-Precision Timing Helper
// ----------------------------------------------------------------------------
template <typename F>
double measure_nanoseconds(F &&fn, int iterations = 100000) {
  // Warmup
  fn();
  auto start = std::chrono::high_resolution_clock::now();
  for (int it = 0; it < iterations; ++it) {
    fn();
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto dur = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  return static_cast<double>(dur) / iterations;
}

// ----------------------------------------------------------------------------
// Verification Function for Individual Static Shape
// ----------------------------------------------------------------------------
template <int M, int N, int K>
void verify_tiny_shape(std::string_view label) {
  constexpr int szA = M * K;
  constexpr int szB = K * N;
  constexpr int szC = M * N;

  alignas(64) float A[szA];
  alignas(64) float B[szB];
  alignas(64) float C_dyn[szC];
  alignas(64) float C_stat[szC];
  double C_oracle[szC];

  init_deterministic_matrix(M, K, A, 0.5f);
  init_deterministic_matrix(K, N, B, 1.2f);
  golden_oracle_gemm(M, N, K, A, B, C_oracle);

  // 1. Execute Dynamic Runtime Path
  planner::CpuGemmProblemV1 prob{M, N, K, planner::CpuScalarTypeV1::f32,
                                 planner::CpuScalarTypeV1::f32,
                                 planner::CpuLayoutV1::row_major_contiguous, 64};
  planner::detail::compiler_vectorized_gemm(prob, A, B, C_dyn);

  // 2. Execute Static Specialized Path
  tiny_gemm_static<M, N, K>(A, B, C_stat);

  // 3. Complete Output Numerical Verification
  double max_err_dyn = 0.0;
  double max_err_stat = 0.0;
  for (int i = 0; i < szC; ++i) {
    double err_dyn = std::abs(static_cast<double>(C_dyn[i]) - C_oracle[i]);
    double err_stat = std::abs(static_cast<double>(C_stat[i]) - C_oracle[i]);
    if (err_dyn > max_err_dyn) max_err_dyn = err_dyn;
    if (err_stat > max_err_stat) max_err_stat = err_stat;
  }

  expect(max_err_dyn < 1e-4, std::string(label) + ": dynamic path matches mathematical oracle across all elements");
  expect(max_err_stat < 1e-4, std::string(label) + ": static specialized path matches mathematical oracle across all elements");

  // 4. Timing Comparison
  double t_dyn = measure_nanoseconds([&]() { planner::detail::compiler_vectorized_gemm(prob, A, B, C_dyn); });
  double t_stat = measure_nanoseconds([&]() { tiny_gemm_static<M, N, K>(A, B, C_stat); });

  double speedup = t_dyn / t_stat;
  std::cout << "  Shape " << label << " (" << M << "x" << N << "x" << K << "): Dynamic = "
            << t_dyn << " ns | Static = " << t_stat << " ns | Speedup = " << speedup << "x\n";

  expect(speedup >= 0.5 || t_stat < 100.0, std::string(label) + ": static specialization is competitive with dynamic path");
}

} // namespace

int main() {
  std::cout << "======================================================================\n";
  std::cout << "   TINY STATIC GEMM SPECIALIZATION & FULL OUTPUT ORACLE VALIDATION\n";
  std::cout << "======================================================================\n";

  // Square Shapes: 1, 2, 3, 4, 7, 8, 15, 16
  verify_tiny_shape<1, 1, 1>("1x1x1");
  verify_tiny_shape<2, 2, 2>("2x2x2");
  verify_tiny_shape<3, 3, 3>("3x3x3");
  verify_tiny_shape<4, 4, 4>("4x4x4");
  verify_tiny_shape<7, 7, 7>("7x7x7");
  verify_tiny_shape<8, 8, 8>("8x8x8");
  verify_tiny_shape<15, 15, 15>("15x15x15");
  verify_tiny_shape<16, 16, 16>("16x16x16");

  // Rectangular / Degenerate Shapes
  verify_tiny_shape<1, 16, 64>("1x16x64 (GEVM)");
  verify_tiny_shape<16, 1, 64>("16x1x64 (GEMV)");
  verify_tiny_shape<8, 8, 1>("8x8x1 (GER)");
  verify_tiny_shape<4, 16, 8>("4x16x8 (Tall)");
  verify_tiny_shape<16, 4, 8>("16x4x8 (Wide)");

  if (failures != 0) {
    std::cerr << failures << " tiny static specialization checks failed!\n";
    return 1;
  }
  std::cout << "ALL TINY STATIC GEMM SPECIALIZATION CHECKS PASSED!\n";
  return 0;
}
