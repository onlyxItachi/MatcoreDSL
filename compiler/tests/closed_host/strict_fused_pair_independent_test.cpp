#include "closed_host_v1.h"
#include "../support/closed_fp_fixture.h"

#include <array>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#pragma STDC FENV_ACCESS ON
#pragma STDC FP_CONTRACT OFF

// Contract-first independent tests, authored without inspecting the fused
// runtime implementation. This target must link the real issued fused leaf and
// its matched test adapter; a native or callback fallback is not test evidence.
// Register a finite CTest timeout: the huge-empty case deliberately detects an
// incorrect traversal of INT64_MAX logical rows.
namespace h = matcore::mdslc::runtime::closed_host_v1;

namespace {
unsigned checks = 0, failures = 0;
constexpr auto strict = h::Numeric::strict_f32;
constexpr auto generated = h::Candidate::generated_strict;

void expect(bool condition, const char *label) {
  ++checks;
  if (!condition) {
    ++failures;
    std::fprintf(stderr, "FAIL: %s\n", label);
  }
}

h::ResourceView view(float *data, std::uint64_t rows, std::uint64_t columns,
                     std::uint64_t capacity) {
  return {data, rows, columns, capacity, h::Access::read_write};
}

h::Value seed(float *data, std::uint64_t rows, std::uint64_t columns,
              std::uint64_t capacity) {
  h::Session owner;
  h::Value value;
  expect(static_cast<bool>(owner.read(1, view(data, rows, columns, capacity), value)),
         "independent immutable input construction succeeds");
  return value;
}

bool sameStatus(const h::Status &a, const h::Status &b) {
  return a.code == b.code && a.failed_frontier == b.failed_frontier &&
         a.completed_frontier == b.completed_frontier &&
         a.completed_effect_frontier == b.completed_effect_frontier &&
         a.publications == b.publications && a.observations == b.observations &&
         a.completed == b.completed;
}

bool sameNumber(float a, float b) {
  // Strict semantics preserve signed zero but do not specify NaN payload bits.
  return (std::isnan(a) && std::isnan(b)) ||
         std::bit_cast<std::uint32_t>(a) == std::bit_cast<std::uint32_t>(b);
}

template <typename Sequence>
bool contents(const h::Value &value, const Sequence &expected) {
  if (!value.valid() || value.rows() * value.columns() != expected.size()) return false;
  for (std::size_t i = 0; i < expected.size(); ++i)
    if (!sameNumber(value.data()[i], expected[i])) return false;
  return true;
}

// Independent explicit f32 oracle, not another candidate or fused schedule.
// Each source GEMM completes in a separate returned vector. Volatile products
// and accumulators retain both rounding points and increasing reduction order.
std::vector<float> reference(const float *a, const float *b,
                             std::size_t m, std::size_t k, std::size_t n) {
  std::vector<float> result(m * n);
  for (std::size_t i = 0; i < m; ++i)
    for (std::size_t j = 0; j < n; ++j) {
      volatile float sum = 0.0F;
      for (std::size_t x = 0; x < k; ++x) {
        volatile float product = a[i * k + x] * b[x * n + j];
        sum = sum + product;
      }
      result[i * n + j] = sum;
    }
  return result;
}

void originalGuardOrder() {
  float one = 1, marker = 791;
  std::array<float, 2> two{1, 2};
  const auto scalar = seed(&one, 1, 1, 1);
  const auto wide = seed(two.data(), 1, 2, 2);
  const auto tall = seed(two.data(), 2, 1, 2);
  const auto sentinel = seed(&marker, 1, 1, 1);
  const h::Value invalid;
  const auto max = static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
  const auto byteOverflow = static_cast<std::uint64_t>(
      std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(float) + 1;
  const auto hugeRows = seed(nullptr, max, 0, 0);
  const auto byteRows = seed(nullptr, byteOverflow, 0, 0);
  const auto zeroZero = seed(nullptr, 0, 0, 0);
  const auto zeroOne = seed(nullptr, 0, 1, 0);
  const auto zeroTwo = seed(nullptr, 0, 2, 0);
  const auto twoZero = seed(nullptr, 2, 0, 0);
  const auto badNumeric = static_cast<h::Numeric>(255);
  const auto badCandidate = static_cast<h::Candidate>(255);
  struct Case {
    const h::Value *a, *b, *d;
    h::Numeric first, second;
    h::Candidate candidate;
    h::Code expected;
    h::Frontier failed;
  };
  const std::array cases{
      Case{&invalid, &scalar, &invalid, badNumeric, badNumeric, badCandidate,
           h::Code::invalid_value, 7},
      Case{&wide, &scalar, &invalid, badNumeric, strict, generated,
           h::Code::invalid_value, 7},
      Case{&wide, &scalar, &invalid, strict, badNumeric, generated,
           h::Code::shape_mismatch, 7},
      Case{&scalar, &scalar, &invalid, strict, badNumeric, badCandidate,
           h::Code::invalid_candidate, 7},
      Case{&hugeRows, &zeroTwo, &twoZero, strict, strict, badCandidate,
           h::Code::invalid_candidate, 7},
      // Every input and final output is empty; logical C[MAX,2] still fails f1.
      Case{&hugeRows, &zeroTwo, &twoZero, strict, strict, generated,
           h::Code::extent_overflow, 7},
      // M*N fits signed index, but C's byte span is not representable.
      Case{&byteRows, &zeroOne, &invalid, strict, strict, generated,
           h::Code::extent_overflow, 7},
      Case{&scalar, &scalar, &invalid, strict, strict, generated,
           h::Code::invalid_value, 8},
      Case{&scalar, &scalar, &tall, strict, badNumeric, generated,
           h::Code::invalid_value, 8},
      Case{&scalar, &scalar, &tall, strict, strict, generated,
           h::Code::shape_mismatch, 8},
      // C[MAX,0] is legal; E[MAX,2] is a second-frontier extent failure.
      Case{&hugeRows, &zeroZero, &zeroTwo, strict, strict, generated,
           h::Code::extent_overflow, 8},
      Case{&byteRows, &zeroZero, &zeroOne, strict, strict, generated,
           h::Code::extent_overflow, 8},
      Case{&scalar, &scalar, &scalar, strict, strict, h::Candidate::native_strict,
           h::Code::candidate_incompatible, 7},
      Case{&scalar, &scalar, &scalar, strict, strict, h::Candidate::automatic,
           h::Code::candidate_incompatible, 7},
      Case{&scalar, &scalar, &scalar, h::Numeric::reassociate_f32, strict, generated,
           h::Code::candidate_incompatible, 7},
      Case{&scalar, &scalar, &scalar, strict, h::Numeric::reassociate_f32, generated,
           h::Code::candidate_incompatible, 8},
  };
  for (const auto &item : cases) {
    h::Session session(h::Options{item.candidate});
    // A second required failure must not be replaced by removed-C allocation.
    session.configureForTesting({1, nullptr, nullptr});
    h::Value result = sentinel;
    const auto status = session.gemmStrictFusedPair(
        7, 8, *item.a, *item.b, *item.d, item.first, item.second, result);
    expect(status.code == item.expected && status.failed_frontier == item.failed &&
               status.completed_frontier == (item.failed == 7 ? 0U : 7U) &&
               status.completed_effect_frontier == 0 && !status.completed,
           "original pair guards fail in their original frontier and precedence");
    const auto report = session.candidateReport();
    expect(result.data() == sentinel.data() &&
               session.allocationAttemptsForTesting() == 0 &&
               !report.invocation_attempted && !report.value_issued,
           "required pair rejection allocates no intermediate and issues no value");
    session.gemmStrictFusedPair(90, 91, invalid, invalid, invalid,
                               badNumeric, badNumeric, result);
    session.complete(99);
    expect(sameStatus(status, session.status()) && result.data() == sentinel.data(),
           "first pair failure remains sticky across later malformed calls");
  }
}

void emptyAndZeroReductions() {
  const auto max = static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
  for (const auto &shape : std::array<std::array<std::uint64_t, 4>, 3>{
           std::array<std::uint64_t, 4>{max, 0, 0, 0}, {0, max, 0, max}, {9, 0, 3, 0}}) {
    const auto [m, k, n, p] = shape;
    auto a = seed(nullptr, m, k, 0), b = seed(nullptr, k, n, 0);
    auto d = seed(nullptr, n, p, 0);
    h::Session session(h::Options{generated});
    h::Value result;
    const auto status = session.gemmStrictFusedPair(1, 2, a, b, d, strict, strict, result);
    const auto report = session.candidateReport();
    expect(status && status.completed_frontier == 2 && result.valid() &&
               result.rows() == m && result.columns() == p,
           "huge legal empty result checks both source products and returns promptly");
    expect(report.actual == h::Implementation::empty_output &&
               !report.invocation_attempted && report.value_issued,
           "empty final output never enters generated row loops");
  }
  {
    auto a = seed(nullptr, 5, 0, 0), b = seed(nullptr, 0, 0, 0);
    auto d = seed(nullptr, 0, 3, 0);
    h::Session session(h::Options{generated});
    h::Value result;
    const auto status = session.gemmStrictFusedPair(1, 2, a, b, d, strict, strict, result);
    expect(status && contents(result, std::array<float, 15>{}) &&
               session.candidateReport().actual == h::Implementation::zero_reduction &&
               !session.candidateReport().invocation_attempted,
           "zero N produces positive-zero E only after both required guards");
  }
  {
    const float inf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    std::array<float, 4> input{inf, nan, -inf, -0.0F};
    auto a = seed(nullptr, 5, 0, 0), b = seed(nullptr, 0, 1, 0);
    auto d = seed(input.data(), 1, 4, 4);
    h::Session session(h::Options{generated});
    h::Value result;
    const auto status = session.gemmStrictFusedPair(1, 2, a, b, d, strict, strict, result);
    bool correct = status && result.valid() && result.rows() == 5 && result.columns() == 4;
    if (correct)
      for (std::size_t i = 0; i < 5; ++i)
        correct = correct && std::isnan(result.data()[4 * i]) &&
                  std::isnan(result.data()[4 * i + 1]) &&
                  std::isnan(result.data()[4 * i + 2]) &&
                  sameNumber(result.data()[4 * i + 3], 0.0F);
    expect(correct, "K0 still multiplies +0 by Inf and NaN in every consumer row");
    const auto report = session.candidateReport();
    expect(report.frontier == 2 && report.invocation_attempted && report.value_issued &&
               report.actual_threads == 1 && report.actual != h::Implementation::zero_reduction,
           "K0 with nonempty E honestly reports actual fused candidate invocation");
  }
}

void strictArithmeticAndAliases() {
  std::array<float, 4> aa{1, 2, 3, 4}, bb{2, 0, 1, 3}, dd{1, 2, 4, 1};
  const std::array<float, 4> expected{28, 14, 58, 32};
  const auto a = seed(aa.data(), 2, 2, 4), b = seed(bb.data(), 2, 2, 4);
  const auto d = seed(dd.data(), 2, 2, 4);
  for (int outputAlias = -1; outputAlias < 3; ++outputAlias) {
    auto av = a, bv = b, dv = d;
    h::Value separate;
    auto &result = outputAlias == 0 ? av : outputAlias == 1 ? bv :
                   outputAlias == 2 ? dv : separate;
    h::Session session(h::Options{generated});
    const auto status = session.gemmStrictFusedPair(1, 2, av, bv, dv, strict, strict, result);
    expect(status && contents(result, expected) && contents(a, aa) &&
               contents(b, bb) && contents(d, dd),
           "noncommuting pair order survives commit into any input-handle variable");
    expect(result.data() != a.data() && result.data() != b.data() && result.data() != d.data(),
           "fused final output is private rather than an input overwrite");
  }
  {
    const auto square = reference(aa.data(), aa.data(), 2, 2, 2);
    const auto cube = reference(square.data(), aa.data(), 2, 2, 2);
    h::Session session(h::Options{generated});
    h::Value result;
    expect(static_cast<bool>(session.gemmStrictFusedPair(1, 2, a, a, a, strict, strict, result)) &&
               contents(result, cube) && contents(a, aa),
           "all three immutable input handles may share identical physical storage");
  }
  for (const auto &geometry : std::array<std::array<std::size_t, 4>, 4>{
           std::array<std::size_t, 4>{1, 3, 2, 4}, {4, 5, 3, 2}, {5, 2, 5, 3}, {9, 7, 6, 5}}) {
    const auto [m, k, n, p] = geometry;
    std::vector<float> ar(m * k), br(k * n), dr(n * p);
    for (std::size_t i = 0; i < ar.size(); ++i) ar[i] = float(int(i % 11) - 5) * 0.25F;
    for (std::size_t i = 0; i < br.size(); ++i) br[i] = float(int(i % 7) - 3) * 0.5F;
    for (std::size_t i = 0; i < dr.size(); ++i) dr[i] = float(int(i % 13) - 6) * 0.125F;
    const auto c = reference(ar.data(), br.data(), m, k, n);
    const auto e = reference(c.data(), dr.data(), m, n, p);
    const auto av = seed(ar.data(), m, k, ar.size()), bv = seed(br.data(), k, n, br.size());
    const auto dv = seed(dr.data(), n, p, dr.size());
    h::Session session(h::Options{generated});
    h::Value result;
    expect(static_cast<bool>(session.gemmStrictFusedPair(1, 2, av, bv, dv, strict, strict, result)) &&
               contents(result, e) && contents(av, ar) && contents(bv, br) && contents(dv, dr),
           "rectangular full and tail row panels match independent two-stage strict oracle");
  }
  const float eps = 0x1p-23F;
  struct ScalarCase { std::vector<float> a, b, d; std::size_t k, n; float expected; };
  std::array cases{
      ScalarCase{{-1, 1 + eps}, {1, 1 - eps}, {1}, 2, 1, 0},
      ScalarCase{{1, 1, 1}, {16777216, 1, -16777216}, {1}, 3, 1, 0},
      ScalarCase{{1}, {16777216, 1, -16777216}, {1, 1, 1}, 1, 3, 0},
      ScalarCase{{1 + eps}, {1 - eps, 1}, {1, -1}, 1, 2, -eps},
      ScalarCase{{std::numeric_limits<float>::max()},
                 {std::numeric_limits<float>::max()}, {0}, 1, 1,
                 std::numeric_limits<float>::quiet_NaN()},
  };
  for (auto &item : cases) {
    auto av = seed(item.a.data(), 1, item.k, item.a.size());
    auto bv = seed(item.b.data(), item.k, item.n, item.b.size());
    auto dv = seed(item.d.data(), item.n, 1, item.d.size());
    h::Session session(h::Options{generated});
    h::Value result;
    expect(static_cast<bool>(session.gemmStrictFusedPair(1, 2, av, bv, dv, strict, strict, result)) &&
               contents(result, std::array<float, 1>{item.expected}),
           "FMA, both reduction orders, intermediate rounding and cross-GEMM rewrites are falsified");
  }
}

void retainedPrefixAndPublicationAliases() {
  std::array<float, 4> aData{1, 2, 3, 4}, bData{2, 0, 1, 3}, dData{1, 2, 4, 1};
  const auto a = seed(aData.data(), 2, 2, 4), b = seed(bData.data(), 2, 2, 4);
  const auto d = seed(dData.data(), 2, 2, 4);
  const auto wrongD = seed(nullptr, 3, 0, 0);
  const std::array<float, 4> originalA = aData, originalB = bData;
  std::array<float, 4> prior{-1, -1, -1, -1}, later{-2, -2, -2, -2};
  h::Value result = b, retained;
  {
    h::Session session(h::Options{generated});
    session.publish(1, a, view(prior.data(), 2, 2, 4));
    session.observe(2, view(prior.data(), 2, 2, 4));
    retained = session.observation(0);
    const auto status = session.gemmStrictFusedPair(3, 4, a, b, wrongD, strict, strict, result);
    expect(status.code == h::Code::shape_mismatch && status.failed_frontier == 4 &&
               status.completed_frontier == 3 && status.completed_effect_frontier == 2 &&
               status.publications == 1 && status.observations == 1 &&
               prior == originalA && result.data() == b.data(),
           "f2 failure retains f1 required completion and exact earlier effect prefix");
    const auto attempts = session.allocationAttemptsForTesting();
    session.publish(5, b, view(later.data(), 2, 2, 4));
    session.observe(6, view(nullptr, 1, 1, 1));
    session.gemmStrictFusedPair(7, 8, a, b, d, strict, strict, result);
    session.complete(9);
    expect(sameStatus(status, session.status()) &&
               session.allocationAttemptsForTesting() == attempts &&
               later == std::array<float, 4>{-2, -2, -2, -2},
           "sticky pair failure blocks every later read, candidate and publication");
  }
  prior.fill(44);
  expect(contents(retained, originalA), "pre-pair observation survives failure and session destruction");
  {
    h::Session session(h::Options{generated});
    // External A changes before the pair, but already-read A must not change.
    session.publish(1, b, view(aData.data(), 2, 2, 4));
    expect(static_cast<bool>(session.gemmStrictFusedPair(2, 3, a, b, d, strict, strict, result)) &&
               contents(result, std::array<float, 4>{28, 14, 58, 32}) && aData == originalB,
           "fused inputs remain immutable across a preceding MAY-alias publication");
    expect(static_cast<bool>(session.publish(4, result, view(aData.data(), 2, 2, 4))) &&
               aData == std::array<float, 4>{28, 14, 58, 32} && contents(a, originalA),
           "final publication may alias original external inputs without mutating old Values");
    session.complete(5);
  }
  aData.fill(33);
  expect(contents(result, std::array<float, 4>{28, 14, 58, 32}),
         "private fused result survives caller storage mutation and session destruction");
}

struct AllocationRun {
  h::Status status;
  std::uint64_t prefixAttempts = 0, totalAttempts = 0;
  bool unchangedResult = false, validResult = false, observationRetained = false;
  std::array<float, 4> prior{}, final{};
};

AllocationRun allocationRun(const h::Value &a, const h::Value &b, const h::Value &d,
                             std::uint64_t failAt) {
  AllocationRun run;
  run.prior.fill(-1); run.final.fill(-2);
  h::Session session(h::Options{generated});
  session.configureForTesting({failAt, nullptr, nullptr});
  h::Value result = b;
  session.publish(1, a, view(run.prior.data(), 2, 2, 4));
  session.observe(2, view(run.prior.data(), 2, 2, 4));
  run.prefixAttempts = session.allocationAttemptsForTesting();
  session.gemmStrictFusedPair(3, 4, a, b, d, strict, strict, result);
  run.unchangedResult = result.data() == b.data();
  run.validResult = contents(result, std::array<float, 4>{28, 14, 58, 32});
  run.observationRetained = contents(session.observation(0), std::array<float, 4>{1, 2, 3, 4});
  session.publish(5, result, view(run.final.data(), 2, 2, 4));
  session.complete(6);
  run.status = session.status();
  run.totalAttempts = session.allocationAttemptsForTesting();
  return run;
}

void allocationFailurePrefixes() {
  std::array<float, 4> ad{1, 2, 3, 4}, bd{2, 0, 1, 3}, dd{1, 2, 4, 1};
  auto a = seed(ad.data(), 2, 2, 4), b = seed(bd.data(), 2, 2, 4);
  auto d = seed(dd.data(), 2, 2, 4);
  const auto success = allocationRun(a, b, d, 0);
  expect(success.status && success.status.completed && success.validResult &&
             success.observationRetained && success.totalAttempts > success.prefixAttempts,
         "successful fused allocation schedule establishes its own measured fault-injection range");
  for (std::uint64_t failAt = 1; failAt <= success.totalAttempts; ++failAt) {
    const auto failed = allocationRun(a, b, d, failAt);
    const bool inPrefix = failAt <= success.prefixAttempts;
    expect(failed.status.code == h::Code::allocation_failure &&
               failed.status.failed_frontier == (inPrefix ? 2U : 4U) &&
               failed.status.completed_frontier == (inPrefix ? 1U : 3U) &&
               failed.status.completed_effect_frontier == (inPrefix ? 1U : 2U) &&
               failed.status.publications == 1 && failed.status.observations == (inPrefix ? 0U : 1U) &&
               !failed.status.completed && failed.unchangedResult &&
               failed.prior == ad && failed.final == std::array<float, 4>{-2, -2, -2, -2},
           "each actual allocation fault retires only an allowed original effect prefix");
    expect(failed.totalAttempts == failAt &&
               (inPrefix || failed.observationRetained),
           "allocation failure stays sticky without later allocations or lost observations");
  }
  const auto beyond = allocationRun(a, b, d, success.totalAttempts + 1);
  expect(sameStatus(success.status, beyond.status) && beyond.validResult,
         "fault index beyond actual allocations does not invent a removed allocation opportunity");
}

h::Code forbiddenCallback(h::detail::CandidateInput, h::detail::CandidateInput,
                           h::detail::CandidateOutput, void *opaque) {
  ++*static_cast<unsigned *>(opaque);
  return h::Code::ok;
}

struct Reentry {
  h::Session *session;
  h::Value input, result;
  h::Status nested;
};

h::Code reenterPair(h::detail::CandidateInput, h::detail::CandidateInput,
                     h::detail::CandidateOutput output, void *opaque) {
  auto &context = *static_cast<Reentry *>(opaque);
  context.nested = context.session->gemmStrictFusedPair(
      100, 101, context.input, context.input, context.input, strict, strict, context.result);
  if (output.rows && output.columns) output.data[0] = 937;
  return h::Code::ok;
}

void frontierAndCandidateBoundaries() {
  float one = 1;
  const auto value = seed(&one, 1, 1, 1);
  const auto maximum = std::numeric_limits<h::Frontier>::max();
  for (const auto &frontiers : std::array<std::array<h::Frontier, 2>, 4>{
           std::array<h::Frontier, 2>{0, 1}, {7, 7}, {7, 9}, {maximum, 0}}) {
    h::Session session(h::Options{generated});
    h::Value result = value;
    const auto status = session.gemmStrictFusedPair(
        frontiers[0], frontiers[1], {}, {}, {}, strict, strict, result);
    expect(status.code == h::Code::invalid_frontier &&
               status.failed_frontier == frontiers[0] &&
               session.allocationAttemptsForTesting() == 0 && result.data() == value.data(),
           "invalid or nonadjacent pair frontiers reject before values and allocation");
  }
  {
    h::Session session(h::Options{generated});
    unsigned invoked = 0;
    session.configureForTesting({0, forbiddenCallback, &invoked});
    h::Value result = value;
    const auto status = session.gemmStrictFusedPair(1, 2, value, value, value,
                                                  strict, strict, result);
    expect(status.code == h::Code::candidate_incompatible && status.failed_frontier == 1 &&
               invoked == 0 && result.data() == value.data() &&
               session.allocationAttemptsForTesting() == 0,
           "an injected callback cannot inherit trusted fused completion or be silently ignored");
  }
  {
    h::Session session;
    Reentry context{&session, value, value, {}};
    session.configureForTesting({0, reenterPair, &context});
    h::Value result = value;
    const auto status = session.gemm(1, value, value, strict, result);
    expect(status.code == h::Code::reentrant_use && status.failed_frontier == 1 &&
               sameStatus(status, context.nested) && status.completed_frontier == 0 &&
               result.data() == value.data() && context.result.data() == value.data(),
           "fused reentry preserves the active outer frontier and never exposes partial private output");
  }
  {
    h::Session session(h::Options{generated});
    session.complete(1);
    auto retired = std::move(session).takeResult();
    h::Value result = value;
    const auto status = session.gemmStrictFusedPair(2, 3, value, value, value,
                                                  strict, strict, result);
    expect(retired.ok() && status.code == h::Code::already_complete &&
               status.completed_frontier == 1 && result.data() == value.data(),
           "fused entry cannot restart a retired Session");
  }
}

void completeCallerFpState() {
  const float tiny = std::numeric_limits<float>::denorm_min();
  float marker = 81;
  std::array<float, 2> ad{tiny, 0x1.000002p0F};
  std::array<float, 4> bd{1, 0, 0, 0x1.fffffcp-1F}, dd{1, 0, 0, 1};
  const auto a = seed(ad.data(), 1, 2, 2), b = seed(bd.data(), 2, 2, 4);
  const auto d = seed(dd.data(), 2, 2, 4);
  const auto sentinel = seed(&marker, 1, 1, 1), wrongD = seed(nullptr, 3, 0, 0);
  const h::Value invalid;
  h::Session allocationProbe(h::Options{generated});
  h::Value probeResult;
  expect(static_cast<bool>(allocationProbe.gemmStrictFusedPair(
             1, 2, a, b, d, strict, strict, probeResult)),
         "FP allocation sweep derives its range from successful actual pair allocation");
  const auto allocationAttempts = allocationProbe.allocationAttemptsForTesting();
  for (std::uint64_t kind = 0; kind < 3 + allocationAttempts; ++kind) {
    h::Session session(h::Options{generated});
    if (kind >= 3) session.configureForTesting({kind - 2, nullptr, nullptr});
    h::Value result = sentinel;
    std::fenv_t saved;
    expect(std::fegetenv(&saved) == 0 && std::fesetround(FE_DOWNWARD) == 0 &&
               std::feraiseexcept(FE_DIVBYZERO | FE_INEXACT | FE_UNDERFLOW) == 0,
           "nondefault caller rounding and sticky flags are established");
    closed_fp_fixture::enableFlush();
    const auto before = closed_fp_fixture::snapshot();
    const auto status = session.gemmStrictFusedPair(
        1, 2, kind == 1 ? invalid : a, b, kind == 2 ? wrongD : d,
        strict, strict, result);
    const auto after = closed_fp_fixture::snapshot();
    expect(std::fesetenv(&saved) == 0, "test caller FP state restored");
    expect(before == after,
           "success, f1/f2 guard failure and every allocation failure restore full caller FP state");
    if (kind == 0)
      expect(status && contents(result, std::array<float, 2>{tiny, 1.0F}),
             "fused execution uses gradual underflow and nearest-even despite caller controls");
    else
      expect(status.code == (kind == 1 ? h::Code::invalid_value :
                             kind == 2 ? h::Code::shape_mismatch : h::Code::allocation_failure) &&
                 status.failed_frontier == (kind == 1 ? 1U : 2U) &&
                 result.data() == sentinel.data(),
             "FP-preserving failing return retains original frontier and output handle");
  }
}
} // namespace

int main() {
  std::fenv_t original;
  if (std::fegetenv(&original) != 0 || std::fesetenv(FE_DFL_ENV) != 0) return 2;
  originalGuardOrder();
  emptyAndZeroReductions();
  strictArithmeticAndAliases();
  retainedPrefixAndPublicationAliases();
  allocationFailurePrefixes();
  frontierAndCandidateBoundaries();
  completeCallerFpState();
  expect(std::fesetenv(&original) == 0, "original process FP environment restored");
  std::printf("Independent strict fused-pair checks: %u; failures: %u\n", checks, failures);
  return failures ? 1 : 0;
}
