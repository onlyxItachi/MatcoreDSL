#pragma once
// Experiment-only oracle and timing protocol; no production ABI or optimizer authority.
#include <algorithm>
#include <bit>
#include <cfenv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

namespace hpc {
enum class Error { none, shape, access, incompatible, other };
struct Outcome {
  bool ok;
  Error error;
  unsigned long long failed, completed, effect, publications, observations;
};
Outcome invoke(float *, float *, float *, unsigned long long,
               unsigned long long, unsigned long long, int);
extern const bool is_region;
extern const char *backend;
struct Case { const char *name; int m, n, k; int pattern = 0; bool timed = true; };
inline const std::vector<Case> cases{
    {"scalar", 1, 1, 1}, {"square16", 16, 16, 16},
    {"square64", 64, 64, 64}, {"square128", 128, 128, 128},
    {"square256", 256, 256, 256}, {"square512", 512, 512, 512},
    {"rect31x97x19", 31, 97, 19}, {"rect128x256x64", 128, 256, 64},
    {"skinny16x512x256", 16, 512, 256}, {"tall384x48x96", 384, 48, 96},
    {"tail65x67x63", 65, 67, 63}, {"tail257x129x65", 257, 129, 65},
    {"zero_m", 0, 13, 7}, {"zero_n", 11, 0, 7},
    {"zero_k", 11, 13, 0}, {"all_zero", 0, 0, 0},
    {"cancellation", 1, 1, 3, 1, false},
    {"product_rounding", 1, 1, 2, 2, false},
    {"gradual_underflow", 1, 1, 1, 3, false},
    {"signed_zero", 1, 1, 1, 4, false},
    {"tail_cancellation", 7, 13, 3, 1, false},
    {"tail_product_rounding", 7, 13, 2, 2, false},
    {"tail_gradual_underflow", 7, 13, 1, 3, false},
    {"tail_signed_zero", 7, 13, 1, 4, false}};
constexpr std::size_t guard = 16;
constexpr float sentinel = -12345.25f;
inline std::uint32_t bits(float x) { return std::bit_cast<std::uint32_t>(x); }
struct Buffer {
  std::size_t count;
  std::vector<float> data;
  explicit Buffer(std::size_t n) : count(n), data(n + 2 * guard, sentinel) {}
  float *ptr() { return count ? data.data() + guard : nullptr; }
  bool guarded() const {
    for (std::size_t i = 0; i < guard; ++i)
      if (bits(data[i]) != bits(sentinel) ||
          bits(data[guard + count + i]) != bits(sentinel)) return false;
    return true;
  }
};
struct Data {
  Buffer a, b, c;
  std::vector<float> expected, original_a, original_b;
  std::vector<double> double_oracle;
  explicit Data(const Case &shape)
      : a(std::size_t(shape.m) * shape.k), b(std::size_t(shape.k) * shape.n),
        c(std::size_t(shape.m) * shape.n), expected(c.count), double_oracle(c.count) {
    // Deterministic small dyadics: all ordinary-case products/sums are exact f32.
    for (std::size_t i = 0; i < a.count; ++i)
      a.ptr()[i] = float(int((i * 17 + 3) % 31) - 15) / 16;
    for (std::size_t i = 0; i < b.count; ++i)
      b.ptr()[i] = float(int((i * 13 + 7) % 29) - 14) / 16;
    if (shape.pattern == 1) {
      for (int row = 0; row < shape.m; ++row) {
        a.ptr()[3 * row] = 0x1p24f; a.ptr()[3 * row + 1] = 1;
        a.ptr()[3 * row + 2] = -0x1p24f;
      }
      std::fill_n(b.ptr(), b.count, 1);
    } else if (shape.pattern == 2) {
      for (int row = 0; row < shape.m; ++row) {
        a.ptr()[2 * row] = -1; a.ptr()[2 * row + 1] = 1 + 0x1p-23f;
      }
      std::fill_n(b.ptr(), shape.n, 1);
      std::fill_n(b.ptr() + shape.n, shape.n, 1 - 0x1p-23f);
      // A contracted second update gives -2^-46, separate product/add gives +0.
      if (std::fma(1 + 0x1p-23f, 1 - 0x1p-23f, -1.0f) != -0x1p-46f)
        std::exit(13);
    } else if (shape.pattern == 3) {
      std::fill_n(a.ptr(), a.count, 0x1p-126f);
      std::fill_n(b.ptr(), b.count, 0.5f);
    } else if (shape.pattern == 4) {
      std::fill_n(a.ptr(), a.count, -0.0f);
      std::fill_n(b.ptr(), b.count, 1);
    }
    original_a = a.data; original_b = b.data;
    for (int i = 0; i < shape.m; ++i) for (int j = 0; j < shape.n; ++j) {
      volatile float strict_sum = 0;
      double sum = 0;
      for (int k = 0; k < shape.k; ++k) {
        const float av = a.ptr()[std::size_t(i) * shape.k + k];
        const float bv = b.ptr()[std::size_t(k) * shape.n + j];
        volatile float product = av * bv;
        strict_sum = strict_sum + product;
        sum += double(av) * double(bv);
      }
      const auto p = std::size_t(i) * shape.n + j;
      expected[p] = strict_sum; double_oracle[p] = sum;
      if (shape.pattern == 0 && double(expected[p]) != sum) {
        std::fprintf(stderr, "ordinary fixture lost exact double/f32 agreement\n");
        std::exit(10);
      }
    }
    if ((shape.pattern == 1 || shape.pattern == 2 || shape.pattern == 4) &&
        bits(expected[0]) != bits(0.0f)) std::exit(11);
    if (shape.pattern == 3 && bits(expected[0]) != bits(0x1p-127f)) std::exit(12);
  }
  bool verify(bool unchanged_output) const {
    const auto exact = [](float a, float b) { return bits(a) == bits(b); };
    if (!a.guarded() || !b.guarded() || !c.guarded() ||
        !std::equal(a.data.begin(), a.data.end(), original_a.begin(), exact) ||
        !std::equal(b.data.begin(), b.data.end(), original_b.begin(), exact)) return false;
    for (std::size_t i = 0; i < c.count; ++i)
      if (bits(c.data[guard + i]) != bits(unchanged_output ? sentinel : expected[i]))
        return false;
    return true;
  }
};
inline bool valid_outcome(const Outcome &o, int fault, bool incompatible) {
  if (!is_region) return o.ok && !fault && !incompatible;
  if (incompatible) return !o.ok && o.error == Error::incompatible &&
      o.failed == 3 && o.completed == 2 && o.effect == 0 &&
      o.publications == 0 && o.observations == 0;
  if (fault) return !o.ok && o.error == (fault == 1 ? Error::shape : Error::access) &&
      o.failed == (fault == 1 ? 1ULL : 4ULL) &&
      o.completed == (fault == 1 ? 0ULL : 3ULL) && o.effect == 0 &&
      o.publications == 0 && o.observations == 0;
  return o.ok && o.failed == 0 && o.completed == 5 && o.effect == 4 &&
      o.publications == 1 && o.observations == 0;
}
inline int main(int argc, char **argv) {
  std::string selected;
  bool timing = false, corrupt = false, incompatible = false, relaxed = false;
  int fault = 0, samples = 5;
  double target_ms = 30;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--case" && i + 1 < argc) selected = argv[++i];
    else if (arg == "--timing") timing = true;
    else if (arg == "--corrupt-output") corrupt = true;
    else if (arg == "--expect-incompatible") incompatible = true;
    else if (arg == "--relaxed") relaxed = true;
    else if (arg == "--bad-input-shape") fault = 1;
    else if (arg == "--readonly-output") fault = 2;
    else if (arg == "--samples" && i + 1 < argc) samples = std::atoi(argv[++i]);
    else if (arg == "--target-ms" && i + 1 < argc) target_ms = std::atof(argv[++i]);
    else { std::fprintf(stderr, "unknown/incomplete argument: %s\n", argv[i]); return 2; }
  }
  if (samples < 1 || samples > 100 || !(target_ms > 0 && target_ms <= 1000)) return 2;
  if (std::fesetround(FE_TONEAREST) != 0) return 2;
  int executed = 0;
  volatile float sink = 0;
  for (const auto &shape : cases) {
    if (!selected.empty() && selected != shape.name) continue;
    // Reassociation controls are separate: do not impose strict bit results on BLAS.
    if (relaxed && shape.pattern != 0) continue;
    ++executed;
    Data data(shape);
    auto call = [&]() {
      const auto o = invoke(data.a.ptr(), data.b.ptr(), data.c.ptr(),
                            shape.m, shape.n, shape.k, fault);
      if (!valid_outcome(o, fault, incompatible)) {
        std::fprintf(stderr, "bad status %s: code=%d failed=%llu completed=%llu effect=%llu pubs=%llu obs=%llu\n",
                     shape.name, int(o.error), o.failed, o.completed, o.effect,
                     o.publications, o.observations);
        std::exit(3);
      }
      if (data.c.count) sink = data.c.ptr()[0];
    };
    call();
    if (corrupt && data.c.count) data.c.ptr()[0] = std::nextafter(data.c.ptr()[0], INFINITY);
    if (!data.verify(fault || incompatible)) {
      std::fprintf(stderr, "oracle/guard/input mismatch: %s\n", shape.name); return 4;
    }
    std::printf("{\"kind\":\"correctness\",\"backend\":\"%s\",\"case\":\"%s\",\"m\":%d,\"n\":%d,\"k\":%d,\"elements\":%zu,\"pass\":true}\n",
                backend, shape.name, shape.m, shape.n, shape.k, data.c.count);
    if (!timing || !shape.timed || fault || incompatible) continue;
    call(); call();
    auto batch = [&](int repetitions) {
      const auto start = std::chrono::steady_clock::now();
      for (int i = 0; i < repetitions; ++i) call();
      return std::chrono::duration<double, std::nano>(
          std::chrono::steady_clock::now() - start).count();
    };
    int repetitions = 1;
    while (repetitions < 1048576 && batch(repetitions) < target_ms * 1e6)
      repetitions *= 2;
    for (int sample = 0; sample < samples; ++sample) {
      const auto ns = batch(repetitions);
      if (!data.verify(false)) return 5;
      std::printf("{\"kind\":\"timing\",\"backend\":\"%s\",\"case\":\"%s\",\"sample\":%d,\"repetitions\":%d,\"total_ns\":%.0f,\"ns_per_call\":%.3f}\n",
                  backend, shape.name, sample, repetitions, ns, ns / repetitions);
    }
  }
  (void)sink;
  if (!executed) { std::fprintf(stderr, "no selected cases\n"); return 2; }
  return 0;
}
} // namespace hpc
