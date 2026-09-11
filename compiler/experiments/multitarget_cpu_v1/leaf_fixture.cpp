// Experimental leaf-only caller. No source admission, runtime registry or
// publication authority is created by linking an issued private primitive.
#include <cfenv>
#include <cstring>
#if defined(__x86_64__)
#include <xmmintrin.h>
#endif

// Reuse the existing permanent numerical discriminators without changing them.
#define main existing_strict_fixture_main
#include "../../tests/generated_cpu/execution_test.cpp"
#undef main

namespace {
bool legal(const char *mode) {
#if defined(__linux__) && defined(__x86_64__)
  if (std::fegetround() != FE_TONEAREST ||
      (_mm_getcsr() & 0xffc0U) != 0x1f80U)
    return false;
  __builtin_cpu_init();
  if (std::strcmp(mode, "baseline") == 0) return true;
  if (std::strcmp(mode, "avx") == 0) return __builtin_cpu_supports("avx");
  if (std::strcmp(mode, "avx2") == 0) return __builtin_cpu_supports("avx2");
  if (std::strcmp(mode, "avx2-fma") == 0)
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
  if (std::strcmp(mode, "avx512f") == 0)
    return __builtin_cpu_supports("avx512f");
#elif defined(__linux__) && defined(__aarch64__)
  unsigned long long fpcr = 0;
  __asm__ volatile("mrs %0, fpcr" : "=r"(fpcr));
  // Fail closed on all nondefault controls, including FZ, rounding, traps and
  // newer alternative FP behavior. This is only a fresh-process test gate.
  return std::strcmp(mode, "aarch64") == 0 && fpcr == 0 &&
         std::fegetround() == FE_TONEAREST;
#endif
  return false;
}
}

int main(int argc, char **argv) {
  if (argc != 2) return 2;
  if (!legal(argv[1])) {
    std::printf("SKIP: requested ISA or default FP environment unavailable\n");
    return 77;
  }
  char fixture_name[] = "strict-fixture";
  char *fixture_argv[] = {fixture_name};
  if (existing_strict_fixture_main(1, fixture_argv) != 0) return 1;

  // Additional independent double-precision oracle; power-of-two scaling
  // makes this finite fixture exactly representable at every f32 boundary.
  unsigned checks = 0;
  for (int m : {1, 3, 7})
    for (int n : {1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 65})
      for (int k : {1, 2, 3, 15, 16, 17}) {
        std::vector<float> a(m * k), b(k * n), c(m * n, -99);
        for (int i = 0; i < m * k; ++i) a[i] = (i % 13 - 6) / 8.0f;
        for (int i = 0; i < k * n; ++i) b[i] = (i % 11 - 5) / 16.0f;
        run(a.data(), b.data(), c.data(), m, n, k);
        for (int i = 0; i < m; ++i)
          for (int j = 0; j < n; ++j) {
            double expected = 0;
            for (int p = 0; p < k; ++p)
              expected += double(a[i * k + p]) * double(b[p * n + j]);
            ++checks;
            if (!equal(float(expected), c[i * n + j])) return 1;
          }
      }
  std::printf("independent exact double oracle: %u checks, 0 failures\n", checks);
}
