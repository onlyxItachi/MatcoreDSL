// Physical research oracle only: descriptors/FP controls are pre-established.
#include <algorithm>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

struct Descriptor {
  float *allocated;
  float *aligned;
  std::int64_t offset;
  std::int64_t sizes[2];
  std::int64_t strides[2];
};
extern "C" void _mlir_ciface_research_strict_fused_pair(
    Descriptor *, Descriptor *, Descriptor *, Descriptor *, Descriptor *);
int checks = 0, failures = 0;
void check(bool value, const std::string &name) {
  ++checks;
  if (!value) {
    ++failures;
    std::cerr << "FAIL " << name << '\n';
  }
}
bool same(float a, float b) {
  return (std::isnan(a) && std::isnan(b)) ||
      std::bit_cast<std::uint32_t>(a) == std::bit_cast<std::uint32_t>(b);
}
std::vector<float> strict(const float *a, const float *b, int m, int k, int n) {
  std::vector<float> c(m * n);
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j) {
      volatile float sum = 0.0f;
      for (int h = 0; h < k; ++h) {
        volatile float product = a[i * k + h] * b[h * n + j];
        sum = sum + product;
      }
      c[i * n + j] = sum;
    }
  return c;
}
Descriptor descriptor(float *data, int rows, int cols) {
  return {data, data, 0, {rows, cols}, {cols, 1}};
}
std::vector<float> run(const float *a, const float *b, const float *d,
    int m, int k, int n, int p, const std::string &label) {
  constexpr float sentinel = -14321.25f;
  const int panelRows = std::min(4, m);
  std::vector<float> output(m * p + 2, sentinel);
  std::vector<float> panel(panelRows * n + 2, sentinel);
  auto ad = descriptor(const_cast<float *>(a), m, k);
  auto bd = descriptor(const_cast<float *>(b), k, n);
  auto dd = descriptor(const_cast<float *>(d), n, p);
  auto ed = descriptor(m * p ? output.data() + 1 : nullptr, m, p);
  auto wd = descriptor(panelRows * n ? panel.data() + 1 : nullptr, panelRows, n);
  _mlir_ciface_research_strict_fused_pair(&ad, &bd, &dd, &ed, &wd);
  auto c = strict(a, b, m, k, n);
  auto expected = strict(c.data(), d, m, n, p);
  bool equal = true;
  for (int x = 0; x < m * p; ++x)
    equal &= same(output[x + 1], expected[x]);
  check(equal, label + " exact strict f32 result");
  check(output.front() == sentinel && output.back() == sentinel &&
      panel.front() == sentinel && panel.back() == sentinel,
      label + " exact output/panel bounds");
  return {output.begin() + 1, output.end() - 1};
}
int main() {
  if (std::fesetround(FE_TONEAREST))
    return 2;
  std::mt19937 rng(0x4128);
  auto data = [&](int size) {
    std::vector<float> result(size);
    for (float &x : result)
      x = std::ldexp(float(int(rng() % 8193) - 4096), -10);
    return result;
  };
  for (int m = 0; m <= 9; ++m)
    for (int k = 0; k <= 6; ++k)
      for (int n = 0; n <= 7; ++n)
        for (int p = 0; p <= 5; ++p) {
          auto a = data(m * k), b = data(k * n), d = data(n * p);
          run(a.empty() ? nullptr : a.data(), b.empty() ? nullptr : b.data(),
              d.empty() ? nullptr : d.data(), m, k, n, p,
              "geometry " + std::to_string(m) + "," + std::to_string(k) +
              "," + std::to_string(n) + "," + std::to_string(p));
        }
  for (int size : {1, 3, 4, 5, 9}) {
    auto a = data(size * size + 1);
    run(a.data(), a.data(), a.data(), size, size, size, size, "identical input alias");
    run(a.data(), a.data() + 1, a.data(), size, size, size, size, "partial input alias");
  }
  const float eps = std::ldexp(1.0f, -23);
  float aFma[] = {-1.0f, 1.0f + eps}, bFma[] = {1.0f, 1.0f - eps}, one[] = {1.0f};
  auto fma = run(aFma, bFma, one, 1, 2, 1, 1, "FMA discriminator");
  check(same(fma[0], 0.0f) && std::fma(1.0f + eps, 1.0f - eps, -1.0f) != fma[0],
      "separate arithmetic differs from FMA");
  float aOrder[] = {1, 1, 1}, bOrder[] = {16777216, 1, -16777216};
  auto order = run(aOrder, bOrder, one, 1, 3, 1, 1, "reduction-order discriminator");
  check(same(order[0], 0.0f) && order[0] != 1.0f, "increasing first reduction retained");
  float aSecondOrder[] = {1}, bSecondOrder[] = {16777216, 1, -16777216};
  float dSecondOrder[] = {1, 1, 1};
  auto secondOrder = run(aSecondOrder, bSecondOrder, dSecondOrder, 1, 1, 3, 1,
      "second reduction-order discriminator");
  check(same(secondOrder[0], 0.0f), "increasing second reduction retained");
  float aRound[] = {1.0f + eps}, bRound[] = {1.0f - eps}, dRound[] = {8388608.0f};
  auto rounded = run(aRound, bRound, dRound, 1, 1, 1, 1, "intermediate f32 round");
  check(same(rounded[0], 8388608.0f), "first product rounded before second GEMM");
  float bBoundary[] = {1.0f - eps, 1.0f}, dBoundary[] = {1.0f, -1.0f};
  auto boundary = run(aRound, bBoundary, dBoundary, 1, 1, 2, 1,
      "intermediate f32 boundary discriminator");
  const float unrounded = float(double(aRound[0]) * double(bBoundary[0]) - double(aRound[0]));
  check(same(boundary[0], -eps) && !same(boundary[0], unrounded),
      "removing intermediate f32 rounding changes output");
  const float inf = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float tiny = std::numeric_limits<float>::denorm_min();
  float huge[] = {std::numeric_limits<float>::max()}, zero[] = {0};
  auto overflowBoundary = run(huge, huge, zero, 1, 1, 1, 1, "cross-GEMM reassociation");
  check(std::isnan(overflowBoundary[0]) && same(huge[0] * (huge[0] * zero[0]), 0.0f),
      "(A*B)*D is not A*(B*D)");
  for (float special : {0.0f, -0.0f, inf, -inf, nan, tiny, -tiny}) {
    float a[] = {special, 1.0f, -1.0f}, b[] = {1.0f, 0.0f, -0.0f};
    run(a, b, one, 1, 3, 1, 1, "special first reduction");
    float d[] = {special};
    run(one, one, d, 1, 1, 1, 1, "special second reduction");
  }
  std::cout << checks << " checks, " << failures << " failures\n";
  return failures ? 1 : 0;
}
