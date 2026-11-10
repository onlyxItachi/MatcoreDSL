// Reuse the independent scalar oracle, raw ABI, canaries and alias controls;
// the original source is unchanged and links to this target's *issued peeled*
// object. This is neither a source-driver test nor a copied execution route.
#define main retainedOutputPatternControls
#include "output_pattern_execution_test.cpp"
#undef main

namespace {
struct Witness {
  const char *name;
  std::array<std::uint32_t, 3> a, b;
  unsigned terms;
  std::uint32_t expected;
  bool allNegativeZero = false;
};
// Frozen independently calculated RNE bit patterns, not results from the leaf.
// NaN payload/sign are intentionally unspecified; all other results are exact.
constexpr Witness witnesses[]{
    {"non-FMA", {0xbf800000, 0x3f800001, 0},
     {0x3f800000, 0x3f7ffffe, 0}, 2, 0},
    {"increasing-K", {0x4b800000, 0x3f800000, 0xcb800000},
     {0x3f800000, 0x3f800000, 0x3f800000}, 3, 0},
    {"RNE-add", {0x3f800000, 0x33c00000, 0},
     {0x3f800000, 0x3f800000, 0}, 2, 0x3f800001},
    {"minimum-subnormal", {1, 0, 0}, {0x3f800000, 0, 0}, 1, 1},
    {"negative-subnormal", {0x80000001, 0, 0}, {0x3f800000, 0, 0}, 1, 0x80000001},
    {"subnormal-product", {0x00800000, 0, 0}, {0x3f000000, 0, 0}, 1, 0x00400000},
    {"underflow-RNE", {0x00800000, 0, 0}, {0x3f000003, 0, 0}, 1, 0x00400002},
    {"overflow", {0x7f7fffff, 0, 0}, {0x40000000, 0, 0}, 1, 0x7f800000},
    {"Inf-times-zero", {0x7f800000, 0, 0}, {0, 0, 0}, 1, 0x7fc00000},
    {"opposed-Infs", {0x7f800000, 0xff800000, 0},
     {0x3f800000, 0x3f800000, 0}, 2, 0x7fc00000},
    {"quiet-NaN", {0x7fc01234, 0, 0}, {0x3f800000, 0, 0}, 1, 0x7fc00000},
    // EVERY product is -0. Padding with +0 would mask a wrong negative fill.
    {"positive-zero-seed", {0x80000000, 0, 0}, {0x3f800000, 0, 0}, 1, 0, true}};
std::uint64_t fullWitnesses = 0, tailOneWitnesses = 0, tailTwoWitnesses = 0;
std::uint64_t inapplicableK = 0;

void analytic(int m, int n, int k, const Witness &witness) {
  if (k < int(witness.terms)) { ++inapplicableK; return; }
  std::vector<float> a(static_cast<std::size_t>(m * k),
      std::bit_cast<float>(witness.allNegativeZero ? 0x80000000U : 0U));
  std::vector<float> b(static_cast<std::size_t>(k * n), 1.0F);
  // Put sensitive terms at the END: even/odd K unroll epilogues must execute.
  for (int i = 0; i < m; ++i)
    for (unsigned q = 0; q < witness.terms; ++q)
      a[static_cast<std::size_t>(i * k + k - witness.terms + q)] =
          std::bit_cast<float>(witness.a[q]);
  for (unsigned q = 0; q < witness.terms; ++q)
    for (int j = 0; j < n; ++j)
      b[static_cast<std::size_t>((k - witness.terms + q) * n + j)] =
          std::bit_cast<float>(witness.b[q]);
  const auto expected = oracle(a.data(), b.data(), m, n, k);
  for (float value : expected)
    check(same(value, std::bit_cast<float>(witness.expected)),
          "independent oracle matches frozen analytic constant", m, n, k);
  run(m, n, k, a, b, witness.name);
  fullWitnesses += n >= tileN;
  tailOneWitnesses += tileN > 1 && n % tileN == 1;
  tailTwoWitnesses += tileN > 2 && n % tileN == 2;
}

void lastK(int m, int n, int k) {
  std::vector<float> a(static_cast<std::size_t>(m * k), 0.0F);
  std::vector<float> b(static_cast<std::size_t>(k * n), 1.0F);
  for (int i = 0; i < m; ++i)
    a[static_cast<std::size_t>(i * k + k - 1)] = float(i + 1);
  for (int j = 0; j < n; ++j)
    b[static_cast<std::size_t>((k - 1) * n + j)] = float(j + 1);
  const auto expected = oracle(a.data(), b.data(), m, n, k);
  // Integer product <= 65*130: exact f32, with zero preceding prefixes.
  for (int i = 0; i < m; ++i) for (int j = 0; j < n; ++j)
    check(bits(expected[static_cast<std::size_t>(i * n + j)]) == bits(float((i + 1) * (j + 1))),
          "last-K-only lane-distinct analytic constant", m, n, k);
  run(m, n, k, a, b, "last-K-only contribution");
}

void nullEmptyOutput(int m, int n, int k) {
  ++cases;
  std::vector<float> a(static_cast<std::size_t>(m * k), 2.0F);
  std::vector<float> b(static_cast<std::size_t>(k * n), 3.0F);
  auto ad = descriptor(a.empty() ? nullptr : a.data(), m, k);
  auto bd = descriptor(b.empty() ? nullptr : b.data(), k, n);
  auto cd = descriptor(nullptr, m, n);
  const auto beforeA = ad, beforeB = bd, beforeC = cd;
  // Unlike run(), this gives empty C no backing canary allocation. A/B are
  // null only when their OWN count is zero; descriptor objects remain real.
  _mlir_ciface___matcore_strict_gemm_f32_v1(&ad, &bd, &cd);
  check(std::memcmp(&ad, &beforeA, sizeof(ad)) == 0 &&
            std::memcmp(&bd, &beforeB, sizeof(bd)) == 0 &&
            std::memcmp(&cd, &beforeC, sizeof(cd)) == 0,
        "null-empty output preserves descriptors", m, n, k);
  for (float value : a) check(bits(value) == bits(2.0F), "null-empty A unchanged", m, n, k);
  for (float value : b) check(bits(value) == bits(3.0F), "null-empty B unchanged", m, n, k);
}

int fullTileOutOfBounds() {
  if (!strictIsaTestAvailable()) return 77;
  // Intentional raw-private capacity violation, not a supported-call test.
  // All N columns are a full tile; only the last B object is absent. ASan must
  // report inside the generated leaf, not in this fixture's construction.
  constexpr int k = 2, n = tileN;
  auto *a = new float[k]{1, 1};
  auto *b = new float[k * n - 1];
  auto *c = new float[n];
  std::fill_n(b, k * n - 1, 1.0F);
  std::fill_n(c, n, -99.0F);
  auto ad = descriptor(a, 1, k), bd = descriptor(b, k, n), cd = descriptor(c, 1, n);
  _mlir_ciface___matcore_strict_gemm_f32_v1(&ad, &bd, &cd);
  delete[] a; delete[] b; delete[] c;
  return 0;
}
} // namespace

int main(int argc, char **argv) {
  if (argc == 2 && std::string_view(argv[1]) == "--oob-full")
    return fullTileOutOfBounds();
  // Includes --corrupt-output and the retained --oob tail capacity probe.
  const int original = retainedOutputPatternControls(argc, argv);
  if (original || argc != 1) return original;
  std::fenv_t caller;
  if (std::fegetenv(&caller) || std::fesetenv(FE_DFL_ENV) ||
      std::fegetround() != FE_TONEAREST) return 2;
  nullEmptyOutput(0, tileN + 1, 3);
  nullEmptyOutput(tileM + 1, 0, 3);
  nullEmptyOutput(0, 0, 3);
  nullEmptyOutput(0, tileN + 1, 0);
  nullEmptyOutput(tileM + 1, 0, 0);
  std::vector<int> widths{1, 2, std::max(1, tileN - 1), tileN, tileN + 1, tileN + 2,
                          2 * tileN - 1, 2 * tileN, 2 * tileN + 1, 2 * tileN + 2,
                          4, 5, 7, 8};
  std::sort(widths.begin(), widths.end());
  widths.erase(std::unique(widths.begin(), widths.end()), widths.end());
  for (int m : {tileM, tileM + 1}) for (int n : widths) {
    for (int k : {1, 2, 3, 4, 5, 6}) {
      // Uniform strict witnesses cannot replace lane-distinct addressing tests.
      ordinary(m, n, k);
      lastK(m, n, k);
      for (const auto &witness : witnesses) analytic(m, n, k, witness);
    }
    for (int k : {33, 65}) {
      analytic(m, n, k, witnesses[0]);
      analytic(m, n, k, witnesses[1]);
    }
    ordinary(m, n, 0); // Raw K=0 must fill C even though source adapters bypass it.
  }
  check(fullWitnesses > 0, "strict witnesses reach full N bodies", tileM, tileN, 3);
  check(tileN == 1 || tailOneWitnesses > 0,
        "strict witnesses reach width-one N tails", tileM, tileN, 3);
  check(tileN <= 2 || tailTwoWitnesses > 0,
        "strict witnesses reach width-two N tails", tileM, tileN, 3);
  if (std::fesetenv(&caller)) return 2;
  std::printf("output peel raw leaf tile=%dx%d: %llu cases, %llu checks, %llu failures; "
              "strict full=%llu tail1=%llu tail2=%llu; inapplicable-K skips=%llu; "
              "tail1 applicable=%d tail2 applicable=%d\n",
      tileM, tileN, static_cast<unsigned long long>(cases),
      static_cast<unsigned long long>(checks), static_cast<unsigned long long>(failures),
      static_cast<unsigned long long>(fullWitnesses),
      static_cast<unsigned long long>(tailOneWitnesses),
      static_cast<unsigned long long>(tailTwoWitnesses),
      static_cast<unsigned long long>(inapplicableK), tileN > 1, tileN > 2);
  return failures != 0;
}
