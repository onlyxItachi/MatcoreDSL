// Leaf-only physical proof, not a source/runtime authority or FP-state adapter.
#include <algorithm>
#include <array>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>
#include <xmmintrin.h>

struct Descriptor {
  float *allocated, *aligned;
  std::int64_t offset, sizes[2], strides[2];
};
static_assert(std::is_standard_layout_v<Descriptor> && sizeof(Descriptor) == 56 && alignof(Descriptor) == 8);
static_assert(offsetof(Descriptor, aligned) == 8 && offsetof(Descriptor, offset) == 16 &&
    offsetof(Descriptor, sizes) == 24 && offsetof(Descriptor, strides) == 40);
extern "C" void _mlir_ciface___matcore_strict_fused_gemm_f32_v1(
    Descriptor *, Descriptor *, Descriptor *, Descriptor *, Descriptor *);
int checks = 0, failures = 0;
void check(bool value, const std::string &name) {
  ++checks;
  if (!value) { ++failures; std::cerr << "FAIL " << name << '\n'; }
}
bool same(float a, float b) {
  return (std::isnan(a) && std::isnan(b)) || std::bit_cast<std::uint32_t>(a) == std::bit_cast<std::uint32_t>(b);
}
std::vector<float> strict(const float *a, const float *b, int m, int k, int n) {
  std::vector<float> c(m * n);
  for (int i = 0; i < m; ++i) for (int j = 0; j < n; ++j) {
    volatile float sum = 0.0f;
    for (int h = 0; h < k; ++h) { volatile float product = a[i * k + h] * b[h * n + j]; sum = sum + product; }
    c[i * n + j] = sum;
  }
  return c;
}
Descriptor view(float *data, int rows, int cols) { return {data, data, 0, {rows, cols}, {cols, 1}}; }
std::vector<float> run(const float *a, const float *b, const float *d, int m, int k, int n, int p,
    const std::string &label) {
  constexpr float sentinel = -14321.25f;
  const int panelRows = std::min(4, m);
  std::vector<float> output(m * p + 2, sentinel), panel(panelRows * n + 2, sentinel);
  auto c = strict(a, b, m, k, n), expected = strict(c.data(), d, m, n, p);
  const std::vector<float> originalA = m * k ? std::vector<float>(a, a + m * k) : std::vector<float>{};
  const std::vector<float> originalB = k * n ? std::vector<float>(b, b + k * n) : std::vector<float>{};
  const std::vector<float> originalD = n * p ? std::vector<float>(d, d + n * p) : std::vector<float>{};
  auto ad = view(const_cast<float *>(a), m, k), bd = view(const_cast<float *>(b), k, n), dd = view(const_cast<float *>(d), n, p);
  auto ed = view(m * p ? output.data() + 1 : nullptr, m, p);
  auto wd = view(panelRows * n ? panel.data() + 1 : nullptr, panelRows, n);
  const auto fpControls = _mm_getcsr() & ~0x3fU;
  const auto rounding = std::fegetround();
  // The closed leaf deliberately requires nonempty E; original source guards
  // and empty-result retirement belong to the later source adapter. Avoid an
  // empty outer row loop here, including the INT64_MAX empty-product hazard.
  if (m && p) _mlir_ciface___matcore_strict_fused_gemm_f32_v1(&ad, &bd, &dd, &ed, &wd);
  check((_mm_getcsr() & ~0x3fU) == fpControls && std::fegetround() == rounding,
      label + " leaf/memset preserves FP controls");
  bool equal = true, inputs = true;
  for (int x = 0; x < m * p; ++x) equal &= same(output[x + 1], expected[x]);
  for (int x = 0; x < m * k; ++x) inputs &= same(a[x], originalA[x]);
  for (int x = 0; x < k * n; ++x) inputs &= same(b[x], originalB[x]);
  for (int x = 0; x < n * p; ++x) inputs &= same(d[x], originalD[x]);
  check(equal, label + " exact strict result");
  check(inputs, label + " input storage preserved");
  check(output.front() == sentinel && output.back() == sentinel && panel.front() == sentinel && panel.back() == sentinel,
      label + " exact output/panel bounds");
  return {output.begin() + 1, output.end() - 1};
}
int main(int argc, char **argv) {
  std::fenv_t saved;
  if (std::fegetenv(&saved) || std::fesetenv(FE_DFL_ENV)) return 2;
  _mm_setcsr((_mm_getcsr() | 0x1f80U) & ~(0x8040U | 0x6000U));
  if (argc == 2) {
    // Isolated ASan negative controls deliberately violate a leaf capacity.
    // Caller performs no OOB access; each input mode must fail on a generated
    // scalar float READ, proving sanitize_address instruments emitted loads.
    const std::string_view mode(argv[1]);
    int m = 1, k = mode == "read-d" ? 1 : 2, n = mode == "read-d" ? 2 : 1, p = 1;
    auto *a = new float[mode == "read-a" ? 1 : m * k]{};
    auto *b = new float[mode == "read-b" ? 1 : k * n]{};
    auto *d = new float[mode == "read-d" ? 1 : n * p]{};
    auto *e = new float[m * p]{}, *scratch = new float[std::min(4, m) * n]{};
    auto ad = view(a, m, k), bd = view(b, k, n), dd = view(d, n, p), ed = view(e, m, p), wd = view(scratch, std::min(4, m), n);
    if (mode != "read-a" && mode != "read-b" && mode != "read-d") return 3;
    _mlir_ciface___matcore_strict_fused_gemm_f32_v1(&ad, &bd, &dd, &ed, &wd);
    delete[] a; delete[] b; delete[] d; delete[] e; delete[] scratch;
    std::fesetenv(&saved); return 0;
  }
  if (argc != 1) return 3;
  std::mt19937 rng(0x4128);
  auto data = [&](int size) {
    std::vector<float> result(size);
    for (float &x : result) x = std::ldexp(float(int(rng() % 8193) - 4096), -10);
    return result;
  };
  for (int m = 0; m <= 9; ++m) for (int k = 0; k <= 6; ++k) for (int n = 0; n <= 7; ++n) for (int p = 0; p <= 5; ++p) {
    auto a = data(m * k), b = data(k * n), d = data(n * p);
    run(a.empty() ? nullptr : a.data(), b.empty() ? nullptr : b.data(), d.empty() ? nullptr : d.data(), m, k, n, p, "exhaustive geometry");
  }
  for (const auto &shape : {std::array<int, 4>{8,13,11,7}, {65,63,97,31}, {257,64,96,73}, {3,128,256,64}}) {
    auto [m,k,n,p] = shape; auto a = data(m*k), b = data(k*n), d = data(n*p);
    run(a.data(), b.data(), d.data(), m,k,n,p, "representative correctness, no timing");
  }
  for (int size : {1,3,4,5,9}) {
    auto a = data(size * size + 1);
    run(a.data(), a.data(), a.data(), size,size,size,size, "identical input aliases");
    run(a.data(), a.data()+1, a.data(), size,size,size,size, "partial input aliases");
  }
  const float eps = std::ldexp(1.0f, -23);
  float aFma[] = {-1.0f,1.0f+eps}, bFma[] = {1.0f,1.0f-eps}, one[] = {1.0f};
  auto fma = run(aFma,bFma,one,1,2,1,1,"FMA discriminator");
  check(same(fma[0],0.0f) && std::fma(1.0f+eps,1.0f-eps,-1.0f) != fma[0], "separate f32 operations");
  float aOrder[] = {1,1,1}, bOrder[] = {16777216,1,-16777216};
  check(same(run(aOrder,bOrder,one,1,3,1,1,"first reduction order")[0],0.0f), "increasing K");
  check(same(run(one,bOrder,aOrder,1,1,3,1,"second reduction order")[0],0.0f), "increasing N");
  float aBoundary[] = {1.0f+eps}, bBoundary[] = {1.0f-eps,1.0f}, dBoundary[] = {1.0f,-1.0f};
  auto boundary = run(aBoundary,bBoundary,dBoundary,1,1,2,1,"intermediate f32 boundary");
  check(same(boundary[0],-eps) && !same(boundary[0],float(double(aBoundary[0])*double(bBoundary[0])-double(aBoundary[0]))), "round first result before second GEMM");
  float huge[] = {std::numeric_limits<float>::max()}, zero[] = {0};
  check(std::isnan(run(huge,huge,zero,1,1,1,1,"cross-GEMM reassociation")[0]) && same(huge[0]*(huge[0]*zero[0]),0.0f), "no algebraic reassociation");
  const float inf = std::numeric_limits<float>::infinity(), nan = std::numeric_limits<float>::quiet_NaN(), tiny = std::numeric_limits<float>::denorm_min();
  for (float special : {0.0f,-0.0f,inf,-inf,nan,tiny,-tiny}) {
    float a[] = {special,1.0f,-1.0f}, b[] = {1.0f,0.0f,-0.0f}, d[] = {special};
    run(a,b,one,1,3,1,1,"special first stage"); run(one,one,d,1,1,1,1,"special second stage");
    auto zeroK = run(nullptr,nullptr,d,1,0,1,1,"zero K with nonempty C/E and special D");
    check((std::isinf(special) || std::isnan(special)) ? std::isnan(zeroK[0]) : same(zeroK[0],0.0f), "K0 does not incorrectly zero final nonempty output");
  }
  float zeroN[2] = {-0.0f,-0.0f};
  auto n0 = run(nullptr,nullptr,nullptr,1,0,0,2,"N0 positive-zero final seed");
  check(same(n0[0],0.0f) && same(n0[1],0.0f) && std::signbit(zeroN[0]), "zero N is distinct from zero K");
  std::fesetenv(&saved);
  std::cout << checks << " checks, " << failures << " failures\n";
  return failures ? 1 : 0;
}
