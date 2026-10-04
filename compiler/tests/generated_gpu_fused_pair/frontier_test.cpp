#include "closed_host_v1.h"
#include "closed_cuda_candidate_v1.h"
#include "closed_rocdl_candidate_v1.h"
#include "../support/closed_fp_fixture.h"

#include <array>
#include <bit>
#include <cerrno>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

#pragma STDC FENV_ACCESS ON
#pragma STDC FP_CONTRACT OFF

// Contract-first independent test: REAL Session implementation, MOCKED device
// entry points. Written from interface headers and the declared combined law,
// before reading the new Session/adapter implementation. This proves neither
// generated-device arithmetic nor driver resource/quarantine behavior. Link the
// matched test Session with both GPU macros enabled, not the actual GPU workers.
namespace h = matcore::mdslc::runtime::closed_host_v1;
namespace {
constexpr auto strict = h::Numeric::strict_f32;
unsigned checks = 0, failures = 0;

void expect(bool yes, const char *what) {
  ++checks;
  if (!yes) { ++failures; std::fprintf(stderr, "FAIL mocked GPU frontier: %s\n", what); }
}

struct Mock {
  std::array<h::Code, 2> available{h::Code::ok, h::Code::ok};
  std::array<h::Code, 2> image{h::Code::ok, h::Code::ok};
  unsigned discoveries = 0, imageQueries = 0, pairs = 0, singles = 0;
  h::Code outcome = h::Code::ok;
  bool corruptControls = false, strictScope = false, privateOutput = false;
  std::array<closed_fp_fixture::Snapshot, 2> discoveryFp{};
};
Mock cuda, rocdl;
void reset() { cuda = Mock{}; rocdl = Mock{}; }
Mock &selected(h::Candidate c) { return c == h::Candidate::generated_nvvm ? cuda : rocdl; }
Mock &other(h::Candidate c) { return c == h::Candidate::generated_nvvm ? rocdl : cuda; }
h::Implementation actual(h::Candidate c) {
  return c == h::Candidate::generated_nvvm ? h::Implementation::generated_nvvm_fused_pair
                                         : h::Implementation::generated_rocdl_fused_pair;
}
struct ErrnoRestore { int saved = errno; ~ErrnoRestore() { errno = saved; } };
h::Code discovery(Mock &mock) noexcept {
  ErrnoRestore preserve;
  const auto i = mock.discoveries++;
  if (i < mock.discoveryFp.size()) mock.discoveryFp[i] = closed_fp_fixture::snapshot();
  errno = ERANGE; // Fake honors the real isolated worker's errno guarantee.
  return i < mock.available.size() ? mock.available[i] : h::Code::candidate_failure;
}
h::Code image(Mock &mock) noexcept {
  const auto i = mock.imageQueries++;
  return i < mock.image.size() ? mock.image[i] : h::Code::candidate_failure;
}
h::Code pair(Mock &mock, h::detail::CandidateInput a, h::detail::CandidateInput b,
             h::detail::CandidateInput d, h::detail::CandidateOutput e) noexcept {
  ErrnoRestore preserve;
  ++mock.pairs;
  errno = EBUSY;
  mock.strictScope = std::fegetround() == FE_TONEAREST;
#if defined(__x86_64__)
  const auto fp = closed_fp_fixture::snapshot();
  mock.strictScope = mock.strictScope && (fp.mxcsr & 0x8040U) == 0 &&
                     (fp.mxcsr & 0x1f80U) == 0x1f80U && (fp.control & 0x3fU) == 0x3fU;
#endif
  mock.privateOutput = e.data && e.data != a.data && e.data != b.data && e.data != d.data;
  if (mock.outcome != h::Code::ok) {
    if (e.rows && e.columns) e.data[0] = 937.0F; // Must never reach old output handle.
    return mock.outcome;
  }
  // Mock behavior only: an independent full-C calculation, not a GPU kernel or
  // the row-panel schedule. Allocation faults below apply to Session, not here.
  try {
    std::vector<float> c(a.rows * b.columns);
    for (std::uint64_t i = 0; i < a.rows; ++i)
      for (std::uint64_t j = 0; j < b.columns; ++j) {
        volatile float sum = 0.0F;
        for (std::uint64_t k = 0; k < a.columns; ++k) {
          volatile float product = a.data[i * a.columns + k] * b.data[k * b.columns + j];
          sum = sum + product;
        }
        c[i * b.columns + j] = sum;
      }
    for (std::uint64_t i = 0; i < e.rows; ++i)
      for (std::uint64_t j = 0; j < e.columns; ++j) {
        volatile float sum = 0.0F;
        for (std::uint64_t k = 0; k < b.columns; ++k) {
          volatile float product = c[i * b.columns + k] * d.data[k * d.columns + j];
          sum = sum + product;
        }
        e.data[i * e.columns + j] = sum;
      }
  } catch (...) { return h::Code::allocation_failure; }
  if (mock.corruptControls) {
    std::fesetround(FE_UPWARD);
    closed_fp_fixture::enableFlush();
  }
  return h::Code::ok;
}

h::ResourceView view(float *p, std::uint64_t r, std::uint64_t c, std::uint64_t cap) {
  return {p, r, c, cap, h::Access::read_write};
}
h::Value seed(std::uint64_t rows, std::uint64_t columns, const std::vector<float> &data = {}) {
  h::Session session;
  h::Value value;
  auto *pointer = data.empty() ? nullptr : const_cast<float *>(data.data());
  expect(static_cast<bool>(session.read(1, view(pointer, rows, columns, data.size()), value)),
         "input immutable Value constructed independently");
  return value;
}
h::Value ones(std::uint64_t rows, std::uint64_t columns) {
  return seed(rows, columns, std::vector<float>(rows * columns, 1.0F));
}
bool sameStatus(const h::Status &a, const h::Status &b) {
  return a.code == b.code && a.failed_frontier == b.failed_frontier &&
         a.completed_frontier == b.completed_frontier &&
         a.completed_effect_frontier == b.completed_effect_frontier &&
         a.publications == b.publications && a.observations == b.observations &&
         a.completed == b.completed;
}
template <typename Sequence> bool contents(const h::Value &v, const Sequence &expected) {
  if (!v.valid() || v.rows() * v.columns() != expected.size()) return false;
  for (std::size_t i = 0; i < expected.size(); ++i)
    if (!(std::isnan(v.data()[i]) && std::isnan(expected[i])) &&
        std::bit_cast<std::uint32_t>(v.data()[i]) != std::bit_cast<std::uint32_t>(expected[i]))
      return false;
  return true;
}
void noFallback(h::Candidate candidate) {
  expect(selected(candidate).singles == 0 && other(candidate).singles == 0 &&
             other(candidate).pairs == 0 && other(candidate).discoveries == 0 &&
             other(candidate).imageQueries == 0,
         "no single-GEMM or other-target fallback entered");
}

void guards(h::Candidate candidate) {
  const auto scalar = ones(1, 1), wide = ones(1, 2), tall = ones(2, 1);
  const auto huge = seed(INT64_MAX, 0), z1 = seed(0, 1), z2 = seed(0, 2);
  const auto byteRows = seed(static_cast<std::uint64_t>(PTRDIFF_MAX) / sizeof(float) + 1, 0);
  const auto hugeColumns = seed(0, INT64_MAX);
  const auto z0 = seed(0, 0), twoZero = seed(2, 0), overM = seed(65536, 0);
  const auto outputM = seed(1025, 0), outputN = seed(0, 1024), endZero = seed(1024, 0);
  const auto firstWorkA = ones(256, 1025), firstWorkB = ones(1025, 256);
  const auto firstWorkD = seed(256, 0);
  const auto secondWorkA = seed(256, 0), secondWorkB = seed(0, 1025);
  const auto secondWorkD = ones(1025, 256);
  const auto secondExtentD = ones(1, 65536);
  const auto capA = ones(33, 64), capB = ones(64, 64), capD = ones(64, 64);
  const auto capEmptyA = ones(33, 128), capEmptyB = ones(128, 64), capEmptyD = seed(64, 0);
  const h::Value invalid;
  const auto bad = static_cast<h::Numeric>(255);
  struct Case {
    const h::Value *a, *b, *d;
    h::Numeric first, second;
    h::Code code;
    h::Frontier frontier;
    unsigned discoveries;
  };
  const std::array cases{
    Case{&invalid, &scalar, &invalid, bad, bad, h::Code::invalid_value, 7, 0},
    Case{&wide, &scalar, &invalid, bad, strict, h::Code::invalid_value, 7, 0},
    Case{&wide, &scalar, &invalid, strict, bad, h::Code::shape_mismatch, 7, 0},
    Case{&huge, &z2, &twoZero, strict, strict, h::Code::extent_overflow, 7, 1},
    Case{&byteRows, &z1, &invalid, strict, strict, h::Code::extent_overflow, 7, 1},
    Case{&overM, &z1, &invalid, strict, bad, h::Code::candidate_incompatible, 7, 1},
    Case{&outputM, &outputN, &endZero, strict, strict, h::Code::candidate_incompatible, 7, 1},
    Case{&firstWorkA, &firstWorkB, &firstWorkD, strict, strict, h::Code::candidate_incompatible, 7, 1},
    Case{&scalar, &scalar, &invalid, strict, bad, h::Code::invalid_value, 8, 1},
    Case{&scalar, &scalar, &tall, strict, bad, h::Code::invalid_value, 8, 1},
    Case{&scalar, &scalar, &tall, strict, strict, h::Code::shape_mismatch, 8, 1},
    Case{&scalar, &scalar, &scalar, h::Numeric::reassociate_f32, strict,
         h::Code::candidate_incompatible, 7, 1},
    Case{&scalar, &scalar, &scalar, strict, h::Numeric::reassociate_f32,
         h::Code::candidate_incompatible, 8, 2},
    Case{&z0, &z1, &secondExtentD, strict, strict, h::Code::candidate_incompatible, 8, 2},
    Case{&twoZero, &z0, &hugeColumns, strict, strict, h::Code::extent_overflow, 8, 2},
    Case{&outputM, &z0, &outputN, strict, strict, h::Code::candidate_incompatible, 8, 2},
    Case{&secondWorkA, &secondWorkB, &secondWorkD, strict, strict, h::Code::candidate_incompatible, 8, 2},
    Case{&capA, &capB, &capD, strict, strict, h::Code::candidate_incompatible, 8, 2},
    Case{&capEmptyA, &capEmptyB, &capEmptyD, strict, strict, h::Code::candidate_incompatible, 8, 2},
  };
  for (const auto &item : cases) {
    reset();
    h::Session session(h::Options{candidate});
    session.configureForTesting({1, nullptr, nullptr});
    h::Value result = scalar;
    auto status = session.gemmStrictFusedPair(7, 8, *item.a, *item.b, *item.d,
                                             item.first, item.second, result);
    const auto report = session.candidateReport();
    expect(status.code == item.code && status.failed_frontier == item.frontier &&
               status.completed_frontier == (item.frontier == 7 ? 0U : 7U) &&
               status.completed_effect_frontier == 0,
           "original guard priority and logical f1/f2 frontier retained");
    expect(selected(candidate).discoveries == item.discoveries &&
               selected(candidate).pairs == 0 && session.allocationAttemptsForTesting() == 0 &&
               result.data() == scalar.data(),
           "guard failure precedes all private result allocation and combined execution");
    expect(report.frontier == item.frontier && report.code == item.code &&
               report.actual == h::Implementation::none &&
               !report.invocation_attempted && !report.value_issued,
           "guard-only report never claims producer invocation or C Value");
    const auto discoveries = selected(candidate).discoveries;
    session.gemmStrictFusedPair(90, 91, invalid, invalid, invalid, bad, bad, result);
    session.complete(99);
    expect(sameStatus(status, session.status()) && selected(candidate).discoveries == discoveries,
           "sticky failure prevents later guard probes and completion");
    noFallback(candidate);
  }

  for (unsigned failAt : {0U, 1U}) for (h::Code failure : {
           h::Code::candidate_unavailable, h::Code::candidate_failure}) {
    reset(); selected(candidate).available[failAt] = failure;
    h::Session session(h::Options{candidate});
    h::Value result = scalar;
    const auto &d = failAt == 0 ? invalid : scalar;
    auto status = session.gemmStrictFusedPair(1, 2, scalar, scalar, d, strict, strict, result);
    expect(status.code == failure && status.failed_frontier == failAt + 1 &&
               status.completed_frontier == failAt && selected(candidate).pairs == 0 &&
               result.data() == scalar.data() && session.allocationAttemptsForTesting() == 0,
           "actual f1/f2 discovery failure is not removed or reattributed");
  }
  // Pair image availability is independent of ordinary single-image discovery,
  // including empty math. It is checked at both original logical operations.
  for (unsigned failAt : {0U, 1U}) {
    reset(); selected(candidate).image[failAt] = h::Code::candidate_unavailable;
    h::Session session(h::Options{candidate});
    h::Value result = scalar;
    const auto status = session.gemmStrictFusedPair(1, 2, z0, z1, scalar, strict, strict, result);
    expect(status.code == h::Code::candidate_unavailable && status.failed_frontier == failAt + 1 &&
               status.completed_frontier == failAt && selected(candidate).pairs == 0 &&
               result.data() == scalar.data() && session.allocationAttemptsForTesting() == 0,
           "missing pair image rejects even when final E is empty");
  }
  // Full C error must still follow original first discovery, not precede it.
  reset(); selected(candidate).available[0] = h::Code::candidate_unavailable;
  h::Session missing(h::Options{candidate}); h::Value result = scalar;
  const auto status = missing.gemmStrictFusedPair(1, 2, huge, z2, twoZero, strict, strict, result);
  expect(status.code == h::Code::candidate_unavailable && status.failed_frontier == 1,
         "first discovery wins over later logical C extent failure");
}

void mathAndBypass(h::Candidate candidate) {
  const auto a = seed(2, 2, {1, 2, 3, 4}), b = seed(2, 2, {2, 0, 1, 3});
  const auto d = seed(2, 2, {1, 2, 4, 1});
  for (int alias = -1; alias < 3; ++alias) {
    reset(); auto av = a, bv = b, dv = d; h::Value separate;
    auto &result = alias == 0 ? av : alias == 1 ? bv : alias == 2 ? dv : separate;
    h::Session session(h::Options{candidate});
    const auto status = session.gemmStrictFusedPair(1, 2, av, bv, dv, strict, strict, result);
    const auto report = session.candidateReport();
    expect(status && contents(result, std::array<float, 4>{28, 14, 58, 32}) &&
               contents(a, std::array<float, 4>{1, 2, 3, 4}) && selected(candidate).privateOutput,
           "mock combined path commits private E safely into any input-handle variable");
    expect(selected(candidate).pairs == 1 && selected(candidate).discoveries == 2 &&
               report.frontier == 2 && report.actual == actual(candidate) &&
               report.invocation_attempted && report.value_issued &&
               session.allocationAttemptsForTesting() == 2,
           "one actual combined invocation, two E allocations, no host C/panel allocation");
    noFallback(candidate);
  }
  reset();
  { h::Session session(h::Options{candidate}); h::Value result;
    expect(session.gemmStrictFusedPair(1, 2, a, a, a, strict, strict, result) &&
               contents(result, std::array<float, 4>{37, 54, 81, 118}),
           "identical immutable A/B/D storage remains legal"); }

  const float inf = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const auto emptyA = seed(5, 0), emptyB = seed(0, 1), specialD = seed(1, 4, {inf, nan, -inf, -0.0F});
  reset();
  { h::Session session(h::Options{candidate}); h::Value result;
    const auto status = session.gemmStrictFusedPair(1, 2, emptyA, emptyB, specialD, strict, strict, result);
    bool correct = status && result.valid() && result.rows() == 5 && result.columns() == 4;
    if (correct) for (unsigned i = 0; i < 5; ++i)
      correct = correct && std::isnan(result.data()[4*i]) && std::isnan(result.data()[4*i+1]) &&
                std::isnan(result.data()[4*i+2]) && std::bit_cast<std::uint32_t>(result.data()[4*i+3]) == 0;
    expect(correct && selected(candidate).pairs == 1,
           "first K0 does not suppress nonempty consumer 0*Inf/NaN"); }

  for (const auto &shape : std::array<std::array<std::uint64_t, 4>, 3>{
           std::array<std::uint64_t, 4>{0, 0, 1, 0}, {5, 0, 0, 3}, {5, 0, 1, 0}}) {
    const auto [m, k, n, p] = shape;
    const auto av = seed(m, k), bv = seed(k, n), dv = seed(n, p);
    reset(); h::Session session(h::Options{candidate}); h::Value result;
    const auto status = session.gemmStrictFusedPair(1, 2, av, bv, dv, strict, strict, result);
    const bool zero = m != 0 && p != 0;
    expect(status && result.valid() && result.rows() == m && result.columns() == p &&
               selected(candidate).discoveries == 2 && selected(candidate).imageQueries == 2 &&
               selected(candidate).pairs == 0 && session.candidateReport().value_issued &&
               session.candidateReport().actual == (zero ? h::Implementation::zero_reduction
                                                          : h::Implementation::empty_output),
           "empty/N0 bypass preserves both candidate/image checks and truthful report");
    if (zero) expect(contents(result, std::array<float, 15>{}), "N0 produces positive zero");
  }
  const auto edgeA = ones(32, 64), edgeB = ones(64, 64), edgeD = ones(64, 64);
  reset(); h::Session edge(h::Options{candidate}); h::Value result;
  expect(edge.gemmStrictFusedPair(1, 2, edgeA, edgeB, edgeD, strict, strict, result) &&
             selected(candidate).pairs == 1 && contents(result, std::vector<float>(32*64, 4096.0F)),
         "exact shared work cap 2^18 is admitted, not off by one");
}

void failureAndPrefix(h::Candidate candidate) {
  const auto a = seed(2, 2, {1, 2, 3, 4}), b = seed(2, 2, {2, 0, 1, 3});
  const auto d = seed(2, 2, {1, 2, 4, 1});
  for (unsigned failure = 1; failure <= 2; ++failure) {
    reset(); h::Session session(h::Options{candidate});
    session.configureForTesting({failure, nullptr, nullptr}); h::Value result = b;
    const auto status = session.gemmStrictFusedPair(1, 2, a, b, d, strict, strict, result);
    expect(status.code == h::Code::allocation_failure && status.failed_frontier == 2 &&
               status.completed_frontier == 1 && result.data() == b.data() &&
               session.allocationAttemptsForTesting() == failure && selected(candidate).pairs == 0 &&
               !session.candidateReport().invocation_attempted,
           "each private E allocation fails at f2 after guard-only f1 without issuing C/E");
  }
  // Measure the existing observation realization independently, then inject
  // the same two E failures after its already-retired effect prefix.
  std::array<float, 4> prefix{};
  h::Session prefixProbe(h::Options{candidate});
  expect(prefixProbe.publish(1, a, view(prefix.data(), 2, 2, 4)) &&
             prefixProbe.observe(2, view(prefix.data(), 2, 2, 4)),
         "prefix allocation schedule measured from successful observation");
  const auto prefixAllocations = prefixProbe.allocationAttemptsForTesting();
  for (unsigned failure = 1; failure <= 2; ++failure) {
    reset(); h::Session session(h::Options{candidate});
    session.configureForTesting({prefixAllocations + failure, nullptr, nullptr});
    h::Value result = b;
    expect(session.publish(1, a, view(prefix.data(), 2, 2, 4)) &&
               session.observe(2, view(prefix.data(), 2, 2, 4)),
           "earlier effects are not displaced by speculative pair allocation");
    const auto status = session.gemmStrictFusedPair(3, 4, a, b, d, strict, strict, result);
    expect(status.code == h::Code::allocation_failure && status.failed_frontier == 4 &&
               status.completed_frontier == 3 && status.completed_effect_frontier == 2 &&
               status.publications == 1 && status.observations == 1 && result.data() == b.data() &&
               contents(session.observation(0), std::array<float, 4>{1, 2, 3, 4}) &&
               selected(candidate).pairs == 0,
           "either private E allocation failure retains original publication/observation prefix");
  }
  for (h::Code code : {h::Code::candidate_failure, h::Code::allocation_failure}) {
    reset(); selected(candidate).outcome = code;
    std::array<float, 4> published{-1, -1, -1, -1}, untouched{-2, -2, -2, -2};
    h::Value observed;
    { h::Session session(h::Options{candidate}); h::Value result = b;
      expect(session.publish(1, a, view(published.data(), 2, 2, 4)) &&
                 session.observe(2, view(published.data(), 2, 2, 4)),
             "earlier publication and owning observation complete");
      observed = session.observation(0);
      const auto status = session.gemmStrictFusedPair(3, 4, a, b, d, strict, strict, result);
      const auto report = session.candidateReport();
      expect(status.code == code && status.failed_frontier == 4 && status.completed_frontier == 3 &&
                 status.completed_effect_frontier == 2 && status.publications == 1 && status.observations == 1 &&
                 result.data() == b.data() && contents(result, std::array<float, 4>{2, 0, 1, 3}) &&
                 report.actual == actual(candidate) && report.invocation_attempted && !report.value_issued,
             "actual shared partial-private failure belongs to f2 and retains old result/prefix");
      const auto allocations = session.allocationAttemptsForTesting();
      const auto calls = selected(candidate).discoveries;
      session.publish(5, result, view(untouched.data(), 2, 2, 4));
      session.read(6, view(nullptr, 99, 99, 0), result);
      session.gemmStrictFusedPair(7, 8, a, b, d, strict, strict, result);
      session.complete(9);
      expect(sameStatus(status, session.status()) && allocations == session.allocationAttemptsForTesting() &&
                 calls == selected(candidate).discoveries && selected(candidate).pairs == 1 &&
                 untouched == std::array<float, 4>{-2, -2, -2, -2},
             "shared failure is sticky before every later access/allocation/publication"); }
    published.fill(44);
    expect(contents(observed, std::array<float, 4>{1, 2, 3, 4}),
           "earlier observation survives failing Session destruction and caller mutation");
  }
}

h::Code callback(h::detail::CandidateInput, h::detail::CandidateInput,
                 h::detail::CandidateOutput, void *opaque) {
  ++*static_cast<unsigned *>(opaque); return h::Code::ok;
}
void protocol(h::Candidate candidate) {
  const auto scalar = ones(1, 1);
  for (const auto &frontiers : std::array<std::array<h::Frontier, 2>, 4>{
           std::array<h::Frontier, 2>{0, 1}, {7, 7}, {7, 9}, {UINT64_MAX, 0}}) {
    reset(); h::Session session(h::Options{candidate}); h::Value result = scalar;
    const auto status = session.gemmStrictFusedPair(frontiers[0], frontiers[1], {}, {}, {}, strict, strict, result);
    expect(status.code == h::Code::invalid_frontier && status.failed_frontier == frontiers[0] &&
               result.data() == scalar.data() && selected(candidate).discoveries == 0 &&
               session.allocationAttemptsForTesting() == 0,
           "malformed pair protocol rejects before candidates and Values");
  }
  reset(); unsigned called = 0; h::Session session(h::Options{candidate});
  session.configureForTesting({0, callback, &called}); h::Value result = scalar;
  const auto status = session.gemmStrictFusedPair(1, 2, scalar, scalar, scalar, strict, strict, result);
  expect(status.code == h::Code::candidate_incompatible && status.failed_frontier == 1 &&
             called == 0 && selected(candidate).pairs == 0 && result.data() == scalar.data(),
         "test callback has no authority to impersonate a combined GPU realization");
  noFallback(candidate);
}

void fpState(h::Candidate candidate) {
  const auto scalar = ones(1, 1), wrong = seed(2, 0);
  const h::Value invalid;
  for (unsigned kind = 0; kind < 7; ++kind) {
    reset(); h::Session session(h::Options{candidate}); h::Value result = scalar;
    if (kind == 3 || kind == 4) session.configureForTesting({kind - 2, nullptr, nullptr});
    if (kind == 5) selected(candidate).outcome = h::Code::candidate_failure;
    if (kind == 6) selected(candidate).corruptControls = true;
    std::fenv_t saved;
    expect(std::fegetenv(&saved) == 0 && std::fesetround(FE_DOWNWARD) == 0 &&
               std::feraiseexcept(FE_DIVBYZERO | FE_INEXACT | FE_UNDERFLOW) == 0,
           "hostile caller FP fixture established");
    closed_fp_fixture::enableFlush();
    const auto before = closed_fp_fixture::snapshot();
    errno = EDOM;
    const auto status = session.gemmStrictFusedPair(1, 2, kind == 1 ? invalid : scalar,
        scalar, kind == 2 ? wrong : scalar, strict, strict, result);
    const auto after = closed_fp_fixture::snapshot();
    const auto savedErrno = errno;
    expect(std::fesetenv(&saved) == 0, "test caller FP environment restored");
    expect(before == after && savedErrno == EDOM,
           "success/guard/allocation/shared/control-corruption exits preserve caller FP and mock-worker errno");
    if (kind == 0 || kind >= 5)
      expect(selected(candidate).strictScope && selected(candidate).discoveryFp[0] == before &&
                 selected(candidate).discoveryFp[1] == before,
             "f2 arithmetic gets strict scope; f1 scope was restored before second discovery");
    expect(kind == 0 ? static_cast<bool>(status) :
           (status.code == (kind == 1 ? h::Code::invalid_value : kind == 2 ? h::Code::shape_mismatch :
                            kind <= 4 ? h::Code::allocation_failure : h::Code::candidate_failure) &&
            status.failed_frontier == (kind == 1 ? 1U : 2U) && result.data() == scalar.data()),
           "FP-path failure classification and output retention are truthful");
  }
}
} // namespace

namespace matcore::mdslc::runtime::closed_host_v1::detail {
Code cudaCandidateAvailable() noexcept { return discovery(cuda); }
Code rocdlCandidateAvailable() noexcept { return discovery(rocdl); }
Code cudaFusedPairImageAvailable() noexcept { return image(cuda); }
Code rocdlFusedPairImageAvailable() noexcept { return image(rocdl); }
Code cudaFusedPairCandidate(CandidateInput a, CandidateInput b, CandidateInput d, CandidateOutput e) noexcept {
  return pair(cuda, a, b, d, e);
}
Code rocdlFusedPairCandidate(CandidateInput a, CandidateInput b, CandidateInput d, CandidateOutput e) noexcept {
  return pair(rocdl, a, b, d, e);
}
Code cudaGemmCandidate(CandidateInput, CandidateInput, CandidateOutput) noexcept {
  ++cuda.singles; return Code::candidate_failure;
}
Code rocdlGemmCandidate(CandidateInput, CandidateInput, CandidateOutput) noexcept {
  ++rocdl.singles; return Code::candidate_failure;
}
} // namespace matcore::mdslc::runtime::closed_host_v1::detail

int main() {
  std::fenv_t saved;
  if (std::fegetenv(&saved) != 0 || std::fesetenv(FE_DFL_ENV) != 0) return 2;
  for (auto candidate : {h::Candidate::generated_nvvm, h::Candidate::generated_rocdl}) {
    guards(candidate); mathAndBypass(candidate); failureAndPrefix(candidate);
    protocol(candidate); fpState(candidate);
  }
  expect(std::fesetenv(&saved) == 0, "process FP environment restored");
  std::printf("REAL Session / MOCKED GPU fused frontier: %u checks; %u failures; no physical device execution\n",
              checks, failures);
  return failures ? 1 : 0;
}
