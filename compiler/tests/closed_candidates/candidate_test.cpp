#include "closed_host_v1.h"
#include <array>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>
#include <xmmintrin.h>

#pragma STDC FENV_ACCESS ON
#pragma STDC FP_CONTRACT OFF
namespace ch = matcore::mdslc::runtime::closed_host_v1;
namespace {
int checks = 0, failures = 0;
void check(bool condition, const char *label) {
  ++checks;
  if (!condition) {
    ++failures;
    std::cerr << "FAIL: " << label << '\n';
  }
}
bool equal(float a, float b) {
  return (std::isnan(a) && std::isnan(b)) ||
         std::bit_cast<std::uint32_t>(a) == std::bit_cast<std::uint32_t>(b);
}
float strict(const float *a, const float *b, std::uint64_t k, std::uint64_t n) {
  float sum = 0;
  for (std::uint64_t p = 0; p < k; ++p) {
    const float product = a[p] * b[p * n];
    sum = sum + product;
  }
  return sum;
}
ch::Code expected(ch::Candidate candidate, ch::Numeric numeric) {
  if (candidate == ch::Candidate::existing_native ||
      candidate == ch::Candidate::authenticated_openblas) {
    if (numeric == ch::Numeric::strict_f32)
      return ch::Code::candidate_incompatible;
#ifdef EXPECT_NATIVE_ONLY
    return ch::Code::candidate_unavailable;
#endif
#ifndef EXPECT_OPENBLAS
    if (candidate == ch::Candidate::authenticated_openblas)
      return ch::Code::candidate_unavailable;
#endif
  }
#ifdef EXPECT_NATIVE_ONLY
  if (candidate == ch::Candidate::generated_strict)
    return ch::Code::candidate_unavailable;
#endif
  return ch::Code::ok;
}
ch::Implementation actual(ch::Candidate candidate) {
  if (candidate == ch::Candidate::automatic) {
#ifdef EXPECT_NATIVE_ONLY
    return ch::Implementation::native_strict;
#else
    return ch::Implementation::generated_strict;
#endif
  }
  if (candidate == ch::Candidate::native_strict)
    return ch::Implementation::native_strict;
  if (candidate == ch::Candidate::generated_strict)
    return ch::Implementation::generated_strict;
  if (candidate == ch::Candidate::existing_native)
    return ch::Implementation::existing_reference;
  return ch::Implementation::authenticated_openblas;
}
void run(ch::Candidate candidate, ch::Numeric numeric, std::uint64_t m,
         std::uint64_t n, std::uint64_t k) {
  std::vector<float> a(m * k), b(k * n), out(m * n, -999);
  for (std::size_t i = 0; i < a.size(); ++i)
    a[i] = static_cast<float>(static_cast<int>(i % 7) - 3);
  for (std::size_t i = 0; i < b.size(); ++i)
    b[i] = static_cast<float>(static_cast<int>(i % 5) - 2);
  ch::Session session(ch::Options{candidate});
  ch::Value lhs, rhs, result;
  check(bool(session.read(1, {a.data(), m, k, a.size()}, lhs)), "read lhs");
  check(bool(session.read(2, {b.data(), k, n, b.size()}, rhs)), "read rhs");
  const auto status = session.gemm(3, lhs, rhs, numeric, result);
  check(status.code == expected(candidate, numeric), "forced selection status");
  const auto report = session.candidateReport();
  check(report.requested == candidate && report.numeric == numeric &&
            report.frontier == 3,
        "report binds request/profile/frontier");
  check(report.code == status.code, "report code");
  if (!status) {
    check(!result.valid() && !report.value_issued &&
              !report.invocation_attempted,
          "illegal request did not invoke/issue");
    check(
        session.publish(4, result,
                        {out.data(), m, n, out.size(), ch::Access::read_write})
                .code == status.code,
        "sticky forced failure");
    for (float v : out)
      check(v == -999, "failed force preserves output");
    return;
  }
  const auto implementation = m == 0 || n == 0
                                  ? ch::Implementation::empty_output
                              : k == 0 ? ch::Implementation::zero_reduction
                                       : actual(candidate);
  check(report.actual == implementation && report.value_issued,
        "actual implementation accurately reported");
  check(report.invocation_attempted == (m != 0 && n != 0 && k != 0),
        "empty computation did not invoke leaf/provider");
  check(bool(session.publish(
            4, result, {out.data(), m, n, out.size(), ch::Access::read_write})),
        "publish");
  for (std::uint64_t i = 0; i < m; ++i)
    for (std::uint64_t j = 0; j < n; ++j)
      check(equal(out[i * n + j],
                  k == 0 ? 0.0f : strict(a.data() + i * k, b.data() + j, k, n)),
            "rectangular exact math");
  check(session.candidateReport().actual == report.actual,
        "report stable across publication");
}
// Exhaust the legal K=2 reduction/FMA expression family for sensitive values.
// NaN payload identity is not a contract. This does not assert broad BLAS
// parity.
bool allowed(float result, float a, float b, float c, float d) {
  const float p = a * b, q = c * d;
  const float z = 0.0f, zp = z + p, zq = z + q, pq = p + q;
  const std::array<float, 9> values{zp + q,
                                    zq + p,
                                    z + pq,
                                    std::fma(a, b, zq),
                                    std::fma(c, d, zp),
                                    std::fma(a, b, std::fma(c, d, z)),
                                    std::fma(c, d, std::fma(a, b, z)),
                                    z + std::fma(a, b, q),
                                    z + std::fma(c, d, p)};
  for (float v : values)
    if (equal(result, v))
      return true;
  return false;
}
void numerical(ch::Candidate candidate) {
  const auto numeric =
      candidate == ch::Candidate::existing_native ||
              candidate == ch::Candidate::authenticated_openblas
          ? ch::Numeric::reassociate_f32
          : ch::Numeric::strict_f32;
  if (expected(candidate, numeric) != ch::Code::ok)
    return;
  const float inf = std::numeric_limits<float>::infinity(),
              tiny = std::numeric_limits<float>::denorm_min();
  const std::array<float, 12> numbers{0.0f,
                                      -0.0f,
                                      1,
                                      -1,
                                      inf,
                                      -inf,
                                      std::numeric_limits<float>::quiet_NaN(),
                                      tiny,
                                      -tiny,
                                      0x1.000002p0f,
                                      0x1.fffffep-1f,
                                      0x1p-126f};
  // 144 independent reductions per GEMM exercise provider nontrivial geometry.
  std::array<float, 288> a{}, b{};
  std::array<float, 20736> output{};
  for (std::size_t i = 0; i < 144; ++i) {
    a[2 * i] = numbers[i / 12];
    a[2 * i + 1] = numbers[i % 12];
    b[i] = numbers[i / 12];
    b[144 + i] = numbers[i % 12];
  }
  ch::Session session(ch::Options{candidate});
  ch::Value lhs, rhs, value;
  check(bool(session.read(1, {a.data(), 144, 2, a.size()}, lhs)),
        "numeric lhs");
  check(bool(session.read(2, {b.data(), 2, 144, b.size()}, rhs)),
        "numeric rhs");
  check(bool(session.gemm(3, lhs, rhs, numeric, value)),
        "numerical candidate success");
  check(bool(session.publish(
            4, value,
            {output.data(), 144, 144, output.size(), ch::Access::read_write})),
        "numeric publish");
  for (std::size_t i = 0; i < 144; ++i)
    for (std::size_t j = 0; j < 144; ++j)
      check(numeric == ch::Numeric::strict_f32
                ? equal(output[i * 144 + j],
                        strict(a.data() + 2 * i, b.data() + j, 2, 144))
                : allowed(output[i * 144 + j], a[2 * i], b[j], a[2 * i + 1],
                          b[144 + j]),
            "K2 IEEE permitted expression membership");
  // FMA discriminator must distinguish strict from a fused implementation.
  float af[2] = {-1, 0x1.000002p0f}, bf[2] = {1, 0x1.fffffep-1f}, of = 99;
  ch::Session discriminator(ch::Options{candidate});
  ch::Value av, bv, cv;
  discriminator.read(1, {af, 1, 2, 2}, av);
  discriminator.read(2, {bf, 2, 1, 2}, bv);
  discriminator.gemm(3, av, bv, numeric, cv);
  discriminator.publish(4, cv, {&of, 1, 1, 1, ch::Access::read_write});
  check(numeric == ch::Numeric::strict_f32
            ? equal(of, 0.0f)
            : allowed(of, af[0], bf[0], af[1], bf[1]),
        "FMA numerical permission");
  for (float x : numbers)
    for (float y : numbers) {
      ch::Session scalar(ch::Options{candidate});
      ch::Value av, bv, cv;
      float output = 99;
      scalar.read(1, {&x, 1, 1, 1}, av);
      scalar.read(2, {&y, 1, 1, 1}, bv);
      check(bool(scalar.gemm(3, av, bv, numeric, cv)), "K1 candidate");
      scalar.publish(4, cv, {&output, 1, 1, 1, ch::Access::read_write});
      const float product = x * y, separate = 0.0f + product;
      check(equal(output, separate) ||
                (numeric == ch::Numeric::reassociate_f32 &&
                 equal(output, std::fma(x, y, 0.0f))),
            "K1 signed zero and nonfinite expression membership");
    }
  // Boundaries through small/tail and blocked provider paths. Here all
  // permitted orders/FMA choices have the same value class and zero sign.
  for (std::uint64_t k : {3, 15, 16, 17, 255, 256, 257})
    for (int kind : {0, 1, 2, 3}) {
      constexpr std::uint64_t m = 17, n = 19;
      std::vector<float> a(m * k, kind == 3 ? -0.0f : 0.0f), b(k * n, 1.0f),
          out(m * n, 99);
      for (std::uint64_t i = 0; i < m; ++i)
        if (kind != 3)
          a[i * k] = inf;
      if (kind == 1)
        for (std::uint64_t j = 0; j < n; ++j)
          b[j] = 0;
      if (kind == 2)
        for (std::uint64_t i = 0; i < m; ++i)
          a[i * k + 1] = -inf;
      ch::Session tail(ch::Options{candidate});
      ch::Value av, bv, cv;
      tail.read(1, {a.data(), m, k, a.size()}, av);
      tail.read(2, {b.data(), k, n, b.size()}, bv);
      check(bool(tail.gemm(3, av, bv, numeric, cv)),
            "nonfinite tail candidate");
      tail.publish(4, cv,
                   {out.data(), m, n, out.size(), ch::Access::read_write});
      const float want = kind == 0   ? inf
                         : kind == 3 ? 0.0f
                                     : std::numeric_limits<float>::quiet_NaN();
      for (float v : out)
        check(
            equal(v, want),
            "tail padding must not suppress or invent nonfinite/zero behavior");
    }
}
void connected(ch::Candidate candidate) {
  const auto numeric =
      candidate == ch::Candidate::existing_native ||
              candidate == ch::Candidate::authenticated_openblas
          ? ch::Numeric::reassociate_f32
          : ch::Numeric::strict_f32;
  if (expected(candidate, numeric) != ch::Code::ok)
    return;
  {
    // Independently suggested: both inputs and the result may name one handle.
    // The input snapshot survives until computation completes; publication is
    // still separate from replacing a C++ handle's immutable value identity.
    float data[4] = {1, 2, 3, 4};
    ch::Session session(ch::Options{candidate});
    ch::Value value;
    session.read(1, {data, 2, 2, 4}, value);
    check(bool(session.gemm(2, value, value, numeric, value)),
          "same immutable handle as lhs, rhs and result");
    session.publish(3, value, {data, 2, 2, 4, ch::Access::read_write});
    check(data[0] == 7 && data[1] == 10 && data[2] == 15 && data[3] == 22,
          "handle replacement did not overwrite live inputs");
  }
  {
    // Independently suggested: valid zero-footprint operands do not imply a
    // representable output. Overflow must precede allocation or any provider
    // probe/leaf call, preserving the previous result handle and effect prefix.
    constexpr auto huge =
        static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
    float old = 17;
    ch::Session session(ch::Options{candidate});
    ch::Value lhs, rhs, result;
    session.read(1, {&old, 1, 1, 1}, result);
    session.read(2, {nullptr, huge, 0, 0}, lhs);
    session.read(3, {nullptr, 0, huge, 0}, rhs);
    check(session.gemm(4, lhs, rhs, numeric, result).code ==
              ch::Code::extent_overflow,
          "zero-footprint inputs do not bypass output extent overflow");
    const auto report = session.candidateReport();
    check(!report.invocation_attempted && !report.provider_contract_checked &&
              report.actual == ch::Implementation::none && !report.value_issued,
          "overflow precedes all candidate and probe calls");
    check(result.valid() && result.rows() == 1 && result.columns() == 1,
          "overflow preserves old result handle");
    ch::Session reader;
    float observed = 0;
    reader.publish(1, result, {&observed, 1, 1, 1, ch::Access::read_write});
    check(observed == 17, "overflow preserves old immutable value bytes");
  }
  for (bool lhsCarry : {false, true}) {
    float a[6] = {1, 2, 3, 4, 5, 6}, b[6] = {2, 3, 1, 4, 2, 1},
          d[4] = {1, 2, 3, 4}, out[4] = {};
    ch::Session session(ch::Options{candidate});
    ch::Value av, bv, cv, dv, ev, late;
    session.read(1, {a, 2, 3, 6}, av);
    session.read(2, {b, 3, 2, 6}, bv);
    check(bool(session.gemm(3, av, bv, numeric, cv)), "first dependent GEMM");
    session.read(4, {d, 2, 2, 4}, dv);
    session.publish(5, cv, {d, 2, 2, 4, ch::Access::read_write});
    session.read(6, {d, 2, 2, 4}, late);
    check(bool(session.gemm(7, lhsCarry ? cv : dv, lhsCarry ? dv : cv, numeric,
                            ev)),
          "lhs/rhs carry GEMM");
    session.publish(8, ev, {out, 2, 2, 4, ch::Access::read_write});
    const std::array<float, 4> want =
        lhsCarry ? std::array<float, 4>{52, 76, 139, 202}
                 : std::array<float, 4>{60, 90, 130, 194};
    for (int i = 0; i < 4; ++i)
      check(equal(out[i], want[i]),
            "noncommuting carry, immutable prepublication read");
    session.publish(9, late, {out, 2, 2, 4, ch::Access::read_write});
    check(out[0] == 10 && out[1] == 14 && out[2] == 25 && out[3] == 38,
          "late read sees publication");
    const auto copied = session.candidateReport();
    check(copied.frontier == 7 && copied.value_issued,
          "last GEMM report binds second frontier");
    // Later failure must preserve earlier publications and immutable handle.
    check(session.gemm(10, av, cv, numeric, ev).code ==
              ch::Code::shape_mismatch,
          "late shape failure");
    check(out[0] == 10 && session.status().publications == 3 && ev.valid(),
          "failure preserves completed effect prefix and old value");
    check(session.candidateReport().actual == ch::Implementation::none &&
              !session.candidateReport().value_issued,
          "failure report does not claim previous candidate ran");
    check(copied.frontier == 7 && copied.value_issued,
          "report snapshot remains immutable");
  }
}
void environment(ch::Candidate candidate) {
  auto numeric = candidate == ch::Candidate::existing_native ||
                         candidate == ch::Candidate::authenticated_openblas
                     ? ch::Numeric::reassociate_f32
                     : ch::Numeric::strict_f32;
  if (expected(candidate, numeric) != ch::Code::ok)
    return;
  std::fenv_t original;
  std::fegetenv(&original);
  std::fesetround(FE_DOWNWARD);
  std::feraiseexcept(FE_INEXACT | FE_UNDERFLOW);
  _mm_setcsr(_mm_getcsr() | 0x8040U);
  const auto mxcsr = _mm_getcsr();
  std::uint16_t cw = 0, sw = 0;
  __asm__ volatile("fnstcw %0" : "=m"(cw));
  __asm__ volatile("fnstsw %0" : "=am"(sw));
  float a = std::numeric_limits<float>::infinity(), b = 0;
  ch::Session session(ch::Options{candidate});
  ch::Value lhs, rhs, value;
  session.read(1, {&a, 1, 1, 1}, lhs);
  session.read(2, {&b, 1, 1, 1}, rhs);
  const auto status = session.gemm(3, lhs, rhs, numeric, value);
  std::uint16_t acw = 0, asw = 0;
  __asm__ volatile("fnstcw %0" : "=m"(acw));
  __asm__ volatile("fnstsw %0" : "=am"(asw));
  check(bool(status), "altered caller FP environment supported");
  check(mxcsr == _mm_getcsr() && cw == acw && sw == asw,
        "complete FP environment restored");
  std::fesetenv(&original);
}
} // namespace
int main() {
  const std::array candidates{
      ch::Candidate::automatic, ch::Candidate::native_strict,
      ch::Candidate::generated_strict, ch::Candidate::existing_native,
      ch::Candidate::authenticated_openblas};
  for (auto candidate : candidates) {
    for (auto numeric :
         {ch::Numeric::strict_f32, ch::Numeric::reassociate_f32}) {
      run(candidate, numeric, 2, 4, 3);
      run(candidate, numeric, 3, 2, 5);
      run(candidate, numeric, 0, 4, 3);
      run(candidate, numeric, 4, 0, 3);
      run(candidate, numeric, 3, 4, 0);
    }
    numerical(candidate);
    environment(candidate);
    connected(candidate);
  }
  {
    ch::Session session(ch::Options{static_cast<ch::Candidate>(255)});
    float a = 1;
    ch::Value lhs, rhs, out;
    session.read(1, {&a, 1, 1, 1}, lhs);
    session.read(2, {&a, 1, 1, 1}, rhs);
    check(session.gemm(3, lhs, rhs, ch::Numeric::strict_f32, out).code ==
              ch::Code::invalid_candidate,
          "forged enum rejected");
  }
  { // Huge empty descriptor must not invoke generated M-loop; all capacities
    // zero.
    ch::Session session(ch::Options{ch::Candidate::automatic});
    ch::Value a, b, c;
    constexpr auto huge =
        static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
    session.read(1, {nullptr, huge, 0, 0}, a);
    session.read(2, {nullptr, 0, 0, 0}, b);
    check(bool(session.gemm(3, a, b, ch::Numeric::strict_f32, c)),
          "huge empty shape terminates");
    check(session.candidateReport().actual == ch::Implementation::empty_output,
          "huge empty no leaf invocation");
  }
  std::cout << checks << " candidate checks, " << failures << " failures\n";
  return failures == 0 ? 0 : 1;
}
