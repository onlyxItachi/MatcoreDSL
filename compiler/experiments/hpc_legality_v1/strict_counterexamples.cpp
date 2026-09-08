// Independent counterexamples, not a candidate implementation or benchmark.
// Build with Clang 21, C++20, -ffp-contract=off and without fast-math.
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <initializer_list>

static float add(float a, float b) { volatile float r = a + b; return r; }
static float mul(float a, float b) { volatile float r = a * b; return r; }
static bool bits(float a, float b) {
  return std::bit_cast<std::uint32_t>(a) == std::bit_cast<std::uint32_t>(b);
}

int main() {
  if (std::fesetround(FE_TONEAREST)) return 2;
  int failures = 0;
  auto check = [&](bool condition, const char *label) {
    if (!condition) { std::fprintf(stderr, "FAIL %s\n", label); ++failures; }
  };
  const float a = 0x1.000002p0f, b = 0x1.fffffcp-1f;
  const float separated = add(-1.0f, mul(a, b));
  const float fused = std::fma(a, b, -1.0f);
  check(bits(separated, 0.0f), "separate f32 multiply/add gives positive zero");
  check(bits(fused, -0x1p-46f), "fused multiply/add gives negative 2^-46");
  check(!bits(separated, fused), "FMA is not a strict realization");

  // A two-accumulator or K-tile partial-sum reduction is not increasing K.
  float sequential = 0.0f;
  for (float value : {0x1p25f, 1.0f, -0x1p25f, 1.0f})
    sequential = add(sequential, value);
  const float split = add(add(0x1p25f, 1.0f), add(-0x1p25f, 1.0f));
  check(bits(sequential, 1.0f), "increasing K gives one");
  check(bits(split, 0.0f), "split partial sums give zero");
  check(!bits(sequential, split), "parallel K accumulators change rounding");

  const float boundary = mul(mul(0x1p100f, 0x1p100f), 0x1p-100f);
  const float reassociated = mul(0x1p100f, mul(0x1p100f, 0x1p-100f));
  check(std::isinf(boundary) && !std::signbit(boundary), "GEMM boundary overflows");
  check(bits(reassociated, 0x1p100f), "cross-GEMM reassociation stays finite");
  if (!failures) std::puts("PASS strict_counterexamples 8 checks");
  return failures ? 1 : 0;
}
