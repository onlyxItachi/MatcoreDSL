#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string_view>
#include <vector>

// Private upstream C wrapper layout, intentionally not an installed ABI.
struct Memref {
  float *allocated;
  float *aligned;
  std::int64_t offset;
  std::int64_t sizes[2];
  std::int64_t strides[2];
};
static_assert(sizeof(void *) == 8 && sizeof(Memref) == 56 &&
              alignof(Memref) == 8);
extern "C" void _mlir_ciface___matcore_strict_gemm_f32_v1(Memref *, Memref *,
                                                          Memref *);
Memref view(float *p, std::int64_t r, std::int64_t c) {
  return {p, p, 0, {r, c}, {c, 1}};
}
void run(float *a, float *b, float *c, int m, int n, int k) {
  auto av = view(a, m, k), bv = view(b, k, n), cv = view(c, m, n);
  _mlir_ciface___matcore_strict_gemm_f32_v1(&av, &bv, &cv);
}
bool equal(float a, float b) {
  return std::isnan(a) ? std::isnan(b)
                       : std::bit_cast<std::uint32_t>(a) ==
                             std::bit_cast<std::uint32_t>(b);
}
int main(int argc, char **argv) {
  if (argc == 2 && std::string_view(argv[1]) == "--oob") {
    // Deliberately violates the private leaf capacity precondition. This
    // negative control must fail inside generated code, not in its caller.
    auto *a = new float[1]{1};
    auto *b = new float[2]{1, 1};
    auto *c = new float[1]{0};
    run(a, b, c, 1, 1, 2);
    delete[] a;
    delete[] b;
    delete[] c;
    return 0;
  }
  int checks = 0, failures = 0;
  auto check = [&](bool ok) {
    ++checks;
    failures += !ok;
  };
  for (const auto shape : {std::array<int, 3>{2, 3, 4},
                           {3, 2, 5},
                           {1, 1, 2},
                           {5, 1, 3},
                           {1, 5, 3},
                           {0, 3, 4},
                           {3, 0, 4},
                           {3, 4, 0},
                           {0, 0, 0},
                           {7, 11, 13},
                           {9, 5, 17}}) {
    auto [m, n, k] = shape;
    std::vector<float> a(std::max(1, m * k)), b(std::max(1, k * n)),
        c(std::max(1, m * n), -19);
    for (int i = 0; i < m * k; ++i)
      a[i] = float(i % 7 - 3) / 3;
    for (int i = 0; i < k * n; ++i)
      b[i] = float(i % 5 - 2) / 7;
    run(a.data(), b.data(), c.data(), m, n, k);
    for (int i = 0; i < m; ++i)
      for (int j = 0; j < n; ++j) {
        float sum = 0;
        for (int p = 0; p < k; ++p) {
          volatile float product = a[i * k + p] * b[p * n + j];
          sum = sum + product;
        }
        check(equal(sum, c[i * n + j]));
      }
    if (!m || !n)
      check(c[0] == -19);
  }
  for (const auto pair :
       {std::array<float, 2>{std::numeric_limits<float>::denorm_min(), 1},
        {std::numeric_limits<float>::infinity(), 0},
        {std::numeric_limits<float>::quiet_NaN(), 1},
        {-0.0f, 1}}) {
    float a = pair[0], b = pair[1], c = 99;
    run(&a, &b, &c, 1, 1, 1);
    volatile float product = a * b;
    check(equal(0.0f + product, c));
  }
  float a[2]{-1, 1.00000011920928955078125f};
  float b[2]{1, 0.99999988079071044921875f};
  float c = 99;
  run(a, b, &c, 1, 1, 2);
  check(c == 0 && std::fma(a[1], b[1], -1.0f) != c);
  float ordered[4]{16777216.0f, 1.0f, -16777216.0f, 1.0f};
  float ones[4]{1, 1, 1, 1};
  run(ordered, ones, &c, 1, 1, 3);
  check(c == 0.0f); // Reordering cancellation before the small add gives 1.
  run(ordered, ones, &c, 1, 1, 4);
  check(c == 1.0f); // Also exercise the residual path after a reduction chunk.
  float same[4]{1, 2, 3, 4}, square[4]{};
  run(same, same, square, 2, 2, 2); // Input/input alias is legal.
  check(square[0] == 7 && square[1] == 10 && square[2] == 15 &&
        square[3] == 22);
  float emptyReduction[6]{1, 2, 3, 4, 5, 6};
  run(nullptr, nullptr, emptyReduction, 2, 3, 0);
  for (float value : emptyReduction)
    check(std::bit_cast<std::uint32_t>(value) == 0);
  std::printf("generated strict CPU execution: %d checks, %d failures\n",
              checks, failures);
  return failures != 0;
}
