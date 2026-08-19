#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#define MATCORE_RESTRICT __restrict
#else
#define MATCORE_RESTRICT __restrict__
#endif

#include <matcore/runtime_c.h>

// Memory barrier to prevent Dead Code Elimination
static void doNotOptimize(void* p) {
#if defined(__clang__) || defined(__GNUC__)
  asm volatile("" : : "g"(p) : "memory");
#elif defined(_MSC_VER)
  _ReadWriteBarrier();
#endif
}

// -------------------------------------------------------------
// Reference Double-Precision Oracle
// -------------------------------------------------------------
void gemm_oracle(int64_t m, int64_t n, int64_t k,
                 const float* a, const float* b, float* c_ref) {
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      double acc = 0.0;
      for (int64_t p = 0; p < k; ++p) {
        acc += static_cast<double>(a[i * k + p]) * static_cast<double>(b[p * n + j]);
      }
      c_ref[i * n + j] = static_cast<float>(acc);
    }
  }
}

// -------------------------------------------------------------
// 1. Runtime Dispatch Wrapper
// -------------------------------------------------------------
void run_runtime_dispatch(int64_t m, int64_t n, int64_t k,
                          const float* a, const float* b, float* c) {
  matcore_tensor_desc_v0 desc_a{};
  desc_a.abi_version = MATCORE_RUNTIME_ABI_VERSION_V0;
  desc_a.struct_size = sizeof(matcore_tensor_desc_v0);
  desc_a.data = const_cast<float*>(a);
  desc_a.dtype = MATCORE_DTYPE_F32_V0;
  desc_a.rank = 2;
  desc_a.dims[0] = m;
  desc_a.dims[1] = k;
  desc_a.strides[0] = k;
  desc_a.strides[1] = 1;
  desc_a.memory_space = MATCORE_MEMORY_SPACE_HOST_V0;
  desc_a.mutability = MATCORE_MUTABILITY_READ_ONLY_V0;

  matcore_tensor_desc_v0 desc_b{};
  desc_b.abi_version = MATCORE_RUNTIME_ABI_VERSION_V0;
  desc_b.struct_size = sizeof(matcore_tensor_desc_v0);
  desc_b.data = const_cast<float*>(b);
  desc_b.dtype = MATCORE_DTYPE_F32_V0;
  desc_b.rank = 2;
  desc_b.dims[0] = k;
  desc_b.dims[1] = n;
  desc_b.strides[0] = n;
  desc_b.strides[1] = 1;
  desc_b.memory_space = MATCORE_MEMORY_SPACE_HOST_V0;
  desc_b.mutability = MATCORE_MUTABILITY_READ_ONLY_V0;

  matcore_tensor_desc_v0 desc_c{};
  desc_c.abi_version = MATCORE_RUNTIME_ABI_VERSION_V0;
  desc_c.struct_size = sizeof(matcore_tensor_desc_v0);
  desc_c.data = c;
  desc_c.dtype = MATCORE_DTYPE_F32_V0;
  desc_c.rank = 2;
  desc_c.dims[0] = m;
  desc_c.dims[1] = n;
  desc_c.strides[0] = n;
  desc_c.strides[1] = 1;
  desc_c.memory_space = MATCORE_MEMORY_SPACE_HOST_V0;
  desc_c.mutability = MATCORE_MUTABILITY_READ_WRITE_V0;

  matcore_policy_v0 policy{};
  policy.abi_version = MATCORE_RUNTIME_ABI_VERSION_V0;
  policy.struct_size = sizeof(matcore_policy_v0);
  policy.target = MATCORE_TARGET_CPU_V0;
  policy.fallback = MATCORE_FALLBACK_ERROR_V0;

  matcore_status_v0 st = matcore_runtime_gemm_f32_v0(&desc_c, &desc_a, &desc_b, &policy);
  if (st.code != MATCORE_STATUS_OK_V0) {
    std::cerr << "matcore_runtime_gemm_f32_v0 failed with code: " << st.code << " (" << st.message << ")\n";
  }
}

// -------------------------------------------------------------
// 2. Generated-C++ AOT Specialization Microkernels
// -------------------------------------------------------------
template <int64_t M, int64_t N, int64_t K>
void generated_cpp_aot_gemm(float* MATCORE_RESTRICT c,
                            const float* MATCORE_RESTRICT a,
                            const float* MATCORE_RESTRICT b) {
  for (int64_t i = 0; i < M; ++i) {
    for (int64_t j = 0; j < N; ++j) {
      float acc = 0.0f;
      for (int64_t p = 0; p < K; ++p) {
        acc += a[i * K + p] * b[p * N + j];
      }
      c[i * N + j] = acc;
    }
  }
}

// -------------------------------------------------------------
// 3. MLIR-Native AOT Specialization Microkernels
// -------------------------------------------------------------
template <int64_t M, int64_t N, int64_t K>
void mlir_native_aot_gemm(float* MATCORE_RESTRICT c,
                          const float* MATCORE_RESTRICT a,
                          const float* MATCORE_RESTRICT b) {
  if constexpr (M == 1 && N == 1) {
    float acc = 0.0f;
    for (int64_t p = 0; p < K; ++p) {
      acc += a[p] * b[p];
    }
    c[0] = acc;
  } else if constexpr (M == 1) {
    for (int64_t j = 0; j < N; ++j) c[j] = 0.0f;
    for (int64_t p = 0; p < K; ++p) {
      const float a_val = a[p];
      const float* b_row = b + p * N;
      for (int64_t j = 0; j < N; ++j) {
        c[j] += a_val * b_row[j];
      }
    }
  } else if constexpr (N == 1) {
    for (int64_t i = 0; i < M; ++i) {
      const float* a_row = a + i * K;
      float acc = 0.0f;
      for (int64_t p = 0; p < K; ++p) {
        acc += a_row[p] * b[p];
      }
      c[i] = acc;
    }
  } else {
    for (int64_t i = 0; i < M; ++i) {
      float* c_row = c + i * N;
      for (int64_t j = 0; j < N; ++j) {
        c_row[j] = 0.0f;
      }
    }
    for (int64_t i = 0; i < M; ++i) {
      const float* a_row = a + i * K;
      float* c_row = c + i * N;
      for (int64_t p = 0; p < K; ++p) {
        const float a_val = a_row[p];
        const float* b_row = b + p * N;
        for (int64_t j = 0; j < N; ++j) {
          c_row[j] += a_val * b_row[j];
        }
      }
    }
  }
}

// -------------------------------------------------------------
// Harness for Shape Comparison
// -------------------------------------------------------------
int g_checks = 0;
int g_failures = 0;

template <int64_t M, int64_t N, int64_t K>
void evaluateShape(const std::string& name, int iterations = 1000) {
  std::vector<float> a(M * K);
  std::vector<float> b(K * N);
  std::vector<float> c_ref(M * N);
  std::vector<float> c_runtime(M * N);
  std::vector<float> c_cpp(M * N);
  std::vector<float> c_mlir(M * N);

  // Initialize with pseudo-random non-trivial floats
  for (int64_t i = 0; i < M * K; ++i) {
    a[i] = static_cast<float>(std::sin(static_cast<double>(i + 1) * 0.73) * 1.5);
  }
  for (int64_t i = 0; i < K * N; ++i) {
    b[i] = static_cast<float>(std::cos(static_cast<double>(i + 1) * 0.47) * 2.1);
  }

  // 1. Correctness against Oracle
  gemm_oracle(M, N, K, a.data(), b.data(), c_ref.data());
  run_runtime_dispatch(M, N, K, a.data(), b.data(), c_runtime.data());
  generated_cpp_aot_gemm<M, N, K>(c_cpp.data(), a.data(), b.data());
  mlir_native_aot_gemm<M, N, K>(c_mlir.data(), a.data(), b.data());

  float max_diff_runtime = 0.0f;
  float max_diff_cpp = 0.0f;
  float max_diff_mlir = 0.0f;
  for (int64_t i = 0; i < M * N; ++i) {
    max_diff_runtime = std::max(max_diff_runtime, std::abs(c_runtime[i] - c_ref[i]));
    max_diff_cpp = std::max(max_diff_cpp, std::abs(c_cpp[i] - c_ref[i]));
    max_diff_mlir = std::max(max_diff_mlir, std::abs(c_mlir[i] - c_ref[i]));
  }

  ++g_checks;
  if (max_diff_runtime > 1e-4f) {
    std::cerr << "FAIL [" << name << "] Runtime mismatch: " << max_diff_runtime << '\n';
    ++g_failures;
  }
  ++g_checks;
  if (max_diff_cpp > 1e-4f) {
    std::cerr << "FAIL [" << name << "] C++ AOT mismatch: " << max_diff_cpp << '\n';
    ++g_failures;
  }
  ++g_checks;
  if (max_diff_mlir > 1e-4f) {
    std::cerr << "FAIL [" << name << "] MLIR AOT mismatch: " << max_diff_mlir << '\n';
    ++g_failures;
  }

  // 2. Timing benchmark
  // Warmup
  for (int w = 0; w < 100; ++w) {
    run_runtime_dispatch(M, N, K, a.data(), b.data(), c_runtime.data());
    generated_cpp_aot_gemm<M, N, K>(c_cpp.data(), a.data(), b.data());
    mlir_native_aot_gemm<M, N, K>(c_mlir.data(), a.data(), b.data());
  }

  // Runtime Dispatch Timing
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int it = 0; it < iterations; ++it) {
    run_runtime_dispatch(M, N, K, a.data(), b.data(), c_runtime.data());
    doNotOptimize(c_runtime.data());
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  double runtime_ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / iterations;

  // Generated-C++ AOT Timing
  t0 = std::chrono::high_resolution_clock::now();
  for (int it = 0; it < iterations; ++it) {
    generated_cpp_aot_gemm<M, N, K>(c_cpp.data(), a.data(), b.data());
    doNotOptimize(c_cpp.data());
  }
  t1 = std::chrono::high_resolution_clock::now();
  double cpp_ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / iterations;

  // MLIR-Native AOT Timing
  t0 = std::chrono::high_resolution_clock::now();
  for (int it = 0; it < iterations; ++it) {
    mlir_native_aot_gemm<M, N, K>(c_mlir.data(), a.data(), b.data());
    doNotOptimize(c_mlir.data());
  }
  t1 = std::chrono::high_resolution_clock::now();
  double mlir_ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / iterations;

  double flops = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
  double runtime_gflops = (flops / runtime_ns);
  double cpp_gflops = (flops / cpp_ns);
  double mlir_gflops = (flops / mlir_ns);

  std::cout << std::left << std::setw(16) << name
            << std::right << std::fixed << std::setprecision(1)
            << std::setw(12) << runtime_ns << " ns (" << std::setw(6) << runtime_gflops << " GF) | "
            << std::setw(10) << cpp_ns << " ns (" << std::setw(6) << cpp_gflops << " GF) | "
            << std::setw(10) << mlir_ns << " ns (" << std::setw(6) << mlir_gflops << " GF) | "
            << "Speedup: " << std::setw(6) << (runtime_ns / mlir_ns) << "x\n";
}

int main() {
  std::cout << "=========================================================================================================\n";
  std::cout << "TRI-WAY SPECIALIZATION COMPARISON: RUNTIME DISPATCH vs C++ AOT vs MLIR-NATIVE AOT\n";
  std::cout << "=========================================================================================================\n";
  std::cout << std::left << std::setw(16) << "Shape (MxNxK)"
            << std::setw(28) << "Runtime Dispatch"
            << std::setw(26) << "Generated-C++ AOT"
            << std::setw(26) << "MLIR-Native AOT"
            << "MLIR Speedup\n";
  std::cout << "---------------------------------------------------------------------------------------------------------\n";

  evaluateShape<1, 1, 1>("1x1x1 (DOT-1)", 10000);
  evaluateShape<1, 1, 64>("1x1x64 (DOT-64)", 10000);
  evaluateShape<1, 64, 64>("1x64x64 (GEVM)", 5000);
  evaluateShape<64, 1, 64>("64x1x64 (GEMV)", 5000);
  evaluateShape<4, 4, 4>("4x4x4 (Square)", 10000);
  evaluateShape<8, 8, 8>("8x8x8 (Square)", 5000);
  evaluateShape<16, 16, 16>("16x16x16 (Square)", 5000);
  evaluateShape<32, 32, 32>("32x32x32 (Square)", 2000);
  evaluateShape<64, 64, 64>("64x64x64 (Square)", 1000);
  evaluateShape<16, 64, 32>("16x64x32 (Rect)", 2000);
  evaluateShape<64, 16, 32>("64x16x32 (Rect)", 2000);

  std::cout << "=========================================================================================================\n";
  std::cout << "Tri-way Specialization Test: " << g_checks << " checks, " << g_failures << " failures\n";
  return g_failures == 0 ? 0 : 1;
}
