#include <matcore/runtime_c.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace {

// Volatile memory barrier to prevent compiler from optimizing away memory writes
inline void memory_barrier(void *ptr) {
#if defined(_MSC_VER) && !defined(__clang__)
  _ReadWriteBarrier();
#else
  asm volatile("" : : "g"(ptr) : "memory");
#endif
}

matcore_tensor_desc_v0 make_desc(float *data, std::int64_t rows,
                                 std::int64_t cols,
                                 matcore_mutability_v0 mutability) {
  matcore_tensor_desc_v0 desc{};
  desc.abi_version = MATCORE_RUNTIME_ABI_VERSION_V0;
  desc.struct_size = sizeof(matcore_tensor_desc_v0);
  desc.data = data;
  desc.dtype = MATCORE_DTYPE_F32_V0;
  desc.rank = 2;
  desc.dims[0] = rows;
  desc.dims[1] = cols;
  desc.strides[0] = cols;
  desc.strides[1] = 1;
  desc.memory_space = MATCORE_MEMORY_SPACE_HOST_V0;
  desc.mutability = mutability;
  return desc;
}

// In-register microkernel implementation template
template <int M, int N, int K>
__attribute__((target("avx2,fma"), noinline))
void static_aot_microkernel(float *__restrict out, const float *__restrict lhs, const float *__restrict rhs) {
  if constexpr (M == 1 && N == 1) {
    float acc = 0.0f;
    #pragma clang loop vectorize(enable) vectorize_width(8) interleave(enable)
    for (int p = 0; p < K; ++p) { acc += lhs[p] * rhs[p]; }
    out[0] = acc;
  } else if constexpr (N == 1) {
    for (int i = 0; i < M; ++i) {
      float acc = 0.0f;
      #pragma clang loop vectorize(enable) vectorize_width(8) interleave(enable)
      for (int p = 0; p < K; ++p) { acc += lhs[i * K + p] * rhs[p]; }
      out[i] = acc;
    }
  } else if constexpr (K == 1) {
    for (int i = 0; i < M; ++i) {
      const float l_elem = lhs[i];
      #pragma clang loop vectorize(enable) vectorize_width(8) interleave(enable)
      for (int j = 0; j < N; ++j) { out[i * N + j] = l_elem * rhs[j]; }
    }
  } else {
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) { out[i * N + j] = 0.0f; }
      for (int p = 0; p < K; ++p) {
        const float a_elem = lhs[i * K + p];
        #pragma clang loop vectorize(enable) vectorize_width(8) interleave(enable)
        for (int j = 0; j < N; ++j) { out[i * N + j] += a_elem * rhs[p * N + j]; }
      }
    }
  }
}

template <int M, int N, int K>
void benchmark_shape(const std::string &name) {
  std::vector<float> a(M * K);
  std::vector<float> b(K * N);
  std::vector<float> c(M * N, 0.0f);

  for (int i = 0; i < M * K; ++i) a[i] = std::sin(static_cast<float>(i + 1));
  for (int i = 0; i < K * N; ++i) b[i] = std::cos(static_cast<float>(i + 2));

  matcore_tensor_desc_v0 c_desc = make_desc(c.data(), M, N, MATCORE_MUTABILITY_READ_WRITE_V0);
  matcore_tensor_desc_v0 a_desc = make_desc(a.data(), M, K, MATCORE_MUTABILITY_READ_ONLY_V0);
  matcore_tensor_desc_v0 b_desc = make_desc(b.data(), K, N, MATCORE_MUTABILITY_READ_ONLY_V0);

  matcore_policy_v0 policy{};
  policy.abi_version = MATCORE_RUNTIME_ABI_VERSION_V0;
  policy.struct_size = sizeof(matcore_policy_v0);
  policy.target = MATCORE_TARGET_CPU_V0;
  policy.fallback = MATCORE_FALLBACK_ERROR_V0;

  // Compute number of iterations to achieve ~10ms measurement window
  const int warmup = 50;
  for (int i = 0; i < warmup; ++i) {
    static_aot_microkernel<M, N, K>(c.data(), a.data(), b.data());
    matcore_runtime_gemm_f32_v0(&c_desc, &a_desc, &b_desc, &policy);
  }

  const int iterations = 5000;

  // 1. Benchmark Static AOT Microkernel
  double checksum_aot = 0.0;
  auto start_aot = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; ++i) {
    static_aot_microkernel<M, N, K>(c.data(), a.data(), b.data());
    memory_barrier(c.data());
    checksum_aot += c[0];
  }
  auto end_aot = std::chrono::high_resolution_clock::now();
  double time_aot_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_aot - start_aot).count() / static_cast<double>(iterations);

  // 2. Benchmark Runtime Dispatch
  double checksum_runtime = 0.0;
  auto start_runtime = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; ++i) {
    matcore_runtime_gemm_f32_v0(&c_desc, &a_desc, &b_desc, &policy);
    memory_barrier(c.data());
    checksum_runtime += c[0];
  }
  auto end_runtime = std::chrono::high_resolution_clock::now();
  double time_runtime_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_runtime - start_runtime).count() / static_cast<double>(iterations);

  double speedup = time_runtime_ns / time_aot_ns;
  std::cout << std::left << std::setw(22) << name 
            << " | M=" << std::setw(3) << M 
            << " N=" << std::setw(3) << N 
            << " K=" << std::setw(3) << K 
            << " | AOT: " << std::fixed << std::setprecision(1) << std::setw(8) << time_aot_ns << " ns"
            << " | Runtime: " << std::setw(8) << time_runtime_ns << " ns"
            << " | Speedup: " << std::setw(6) << std::setprecision(2) << speedup << "x"
            << " (chk=" << static_cast<int>(checksum_aot + checksum_runtime) << ")\n";
}

} // namespace

int main() {
  std::cout << "=== MATCORE DSL REAL PERFORMANCE CROSSOVER BENCHMARK ===\n";
  std::cout << "Shape                  | Dimensions      | Static AOT    | Runtime Dispatch| Speedup\n";
  std::cout << "-----------------------+-----------------+---------------+-----------------+--------\n";

  // Square crossover sweep
  benchmark_shape<1, 1, 1>("Square 1x1x1");
  benchmark_shape<2, 2, 2>("Square 2x2x2");
  benchmark_shape<3, 3, 3>("Square 3x3x3");
  benchmark_shape<4, 4, 4>("Square 4x4x4");
  benchmark_shape<5, 5, 5>("Square 5x5x5");
  benchmark_shape<7, 7, 7>("Square 7x7x7");
  benchmark_shape<8, 8, 8>("Square 8x8x8");
  benchmark_shape<9, 9, 9>("Square 9x9x9");
  benchmark_shape<15, 15, 15>("Square 15x15x15");
  benchmark_shape<16, 16, 16>("Square 16x16x16");
  benchmark_shape<17, 17, 17>("Square 17x17x17");
  benchmark_shape<24, 24, 24>("Square 24x24x24");
  benchmark_shape<31, 31, 31>("Square 31x31x31");
  benchmark_shape<32, 32, 32>("Square 32x32x32");
  benchmark_shape<33, 33, 33>("Square 33x33x33");
  benchmark_shape<48, 48, 48>("Square 48x48x48");
  benchmark_shape<63, 63, 63>("Square 63x63x63");
  benchmark_shape<64, 64, 64>("Square 64x64x64");

  // Degenerate & Asymmetric crossover sweep
  benchmark_shape<1, 1, 128>("DOT 1x1x128");
  benchmark_shape<1, 1, 512>("DOT 1x1x512");
  benchmark_shape<16, 1, 64>("GEMV 16x1x64");
  benchmark_shape<32, 1, 64>("GEMV 32x1x64");
  benchmark_shape<1, 16, 64>("GEVM 1x16x64");
  benchmark_shape<1, 32, 64>("GEVM 1x32x64");
  benchmark_shape<16, 16, 1>("GER 16x16x1");
  benchmark_shape<32, 32, 1>("GER 32x32x1");
  benchmark_shape<16, 64, 32>("Rect 16x64x32");
  benchmark_shape<32, 16, 64>("Rect 32x16x64");

  std::cout << "\nBenchmark run completed successfully.\n";
  return 0;
}
