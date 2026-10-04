#include "closed_host_v1.h"
#include "../support/closed_fp_fixture.h"

#include <array>
#include <bit>
#include <cfenv>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <utility>

#pragma STDC FENV_ACCESS ON

namespace h = matcore::mdslc::runtime::closed_host_v1;

namespace {
unsigned checks = 0, failures = 0;

void expect(bool condition, const char *label) {
  ++checks;
  if (!condition) {
    ++failures;
    std::fprintf(stderr, "FAIL: %s\n", label);
  }
}

h::ResourceView view(float *data, std::uint64_t rows, std::uint64_t columns,
                     std::uint64_t capacity,
                     h::Access access = h::Access::read_write) {
  return {data, rows, columns, capacity, access};
}

bool sameStatus(const h::Status &a, const h::Status &b) {
  return a.code == b.code && a.failed_frontier == b.failed_frontier &&
         a.completed_frontier == b.completed_frontier &&
         a.completed_effect_frontier == b.completed_effect_frontier &&
         a.publications == b.publications && a.observations == b.observations &&
         a.completed == b.completed;
}

template <std::size_t N>
bool contents(const h::Value &value, const std::array<float, N> &expected) {
  return value.valid() && value.rows() * value.columns() == N &&
         (N == 0 || std::memcmp(value.data(), expected.data(), sizeof(expected)) == 0);
}

h::Value seed(float *data, std::uint64_t rows, std::uint64_t columns,
              std::uint64_t capacity) {
  h::Session session;
  h::Value value;
  expect(static_cast<bool>(session.read(1, view(data, rows, columns, capacity), value)),
         "independent seed read succeeds");
  return value;
}

// These are deliberate private-adapter malformed-input controls. They test
// the repeated runtime checks, not permission to forge a compiler derivation.
void originalReadGuardOrder() {
  std::array<float, 6> data{1, 2, 3, 4, 5, 6};
  float sentinelData = 71;
  const auto sentinel = seed(&sentinelData, 1, 1, 1);
  const h::Value invalidPublished;
  const auto invalidAccess = static_cast<h::Access>(255);
  const auto maximum = std::numeric_limits<std::uint64_t>::max();
  const auto lastAligned = std::numeric_limits<std::uintptr_t>::max() -
                           (alignof(float) - 1);
  auto *misaligned = reinterpret_cast<float *>(
      reinterpret_cast<unsigned char *>(data.data()) + 1);
  struct Case {
    h::ResourceView resource;
    std::uint64_t rows, columns;
    h::Code expected;
  };
  const std::array cases{
      Case{view(nullptr, 0, 0, 0, invalidAccess), maximum, 0, h::Code::extent_overflow},
      Case{view(nullptr, 0, 0, 0, invalidAccess), 0, maximum, h::Code::extent_overflow},
      Case{view(nullptr, 1, 1, 0, invalidAccess), 2, 2, h::Code::shape_mismatch},
      Case{view(nullptr, maximum, 1, 0, invalidAccess), 1, 1, h::Code::shape_mismatch},
      Case{view(nullptr, 2, 2, 0, invalidAccess), 2, 2, h::Code::invalid_view},
      Case{view(nullptr, 2, 2, 3), 2, 2, h::Code::insufficient_capacity},
      Case{view(nullptr, 2, 2, 4), 2, 2, h::Code::invalid_view},
      Case{view(misaligned, 2, 2, 4), 2, 2, h::Code::invalid_view},
      Case{view(reinterpret_cast<float *>(lastAligned), 1, 2, 2),
           1, 2, h::Code::extent_overflow},
      Case{view(nullptr, 0, 7, 0, invalidAccess), 0, 7, h::Code::invalid_view},
  };
  for (const auto &item : cases) {
    h::Session ordinary, forwarded;
    h::Value ordinaryResult = sentinel, forwardedResult = sentinel;
    const auto original = ordinary.read(11, item.resource, item.rows, item.columns,
                                        ordinaryResult);
    const auto optimized = forwarded.readForwarded(
        11, item.resource, item.rows, item.columns, invalidPublished, forwardedResult);
    expect(original.code == item.expected && sameStatus(original, optimized),
           "forwarding preserves original required-check precedence and status");
    expect(ordinaryResult.data() == sentinel.data() &&
               forwardedResult.data() == sentinel.data(),
           "failed original or forwarded read does not overwrite its result");
    expect(ordinary.allocationAttemptsForTesting() == 0 &&
               forwarded.allocationAttemptsForTesting() == 0,
           "invalid reads reject before any owned allocation");
  }

  {
    h::Session session;
    h::Value result = sentinel;
    const auto status = session.readForwarded(
        1, view(data.data(), 2, 2, 4), 2, 2, invalidPublished, result);
    expect(status.code == h::Code::invalid_value && result.data() == sentinel.data(),
           "retained-value validity is still required after original read checks");
  }
  {
    const auto wrongShape = seed(data.data(), 2, 3, 6);
    h::Session session;
    h::Value result = sentinel;
    const auto status = session.readForwarded(
        1, view(data.data(), 2, 2, 4), 2, 2, wrongShape, result);
    expect(status.code == h::Code::shape_mismatch && result.data() == sentinel.data(),
           "retained-value shape is checked before issuing the forwarded result");
  }
}

void orderedFailurePrefix() {
  std::array<float, 4> source{1, 2, 3, 4}, output{-1, -1, -1, -1};
  std::array<float, 4> later{-2, -2, -2, -2};
  h::Session session;
  h::Value published, result;
  expect(static_cast<bool>(session.read(1, view(source.data(), 2, 2, 4), published)),
         "prefix source read succeeds");
  expect(static_cast<bool>(session.publish(2, published, view(output.data(), 2, 2, 4))),
         "first publication succeeds before late read failure");
  expect(static_cast<bool>(session.observe(3, view(output.data(), 2, 2, 4))),
         "first owning observation succeeds before late read failure");
  const auto attempts = session.allocationAttemptsForTesting();
  const auto failed = session.readForwarded(
      4, view(output.data(), 2, 2, 4), 1, 4, published, result);
  expect(failed.code == h::Code::shape_mismatch && failed.failed_frontier == 4 &&
             failed.completed_frontier == 3 && failed.completed_effect_frontier == 3 &&
             failed.publications == 1 && failed.observations == 1 && !failed.completed,
         "late forwarded-read failure retains exact original completed effect prefix");
  expect(output == source && contents(session.observation(0), source) &&
             session.observationFrontier(0) == 3 && !result.valid(),
         "late failure keeps first publication and immutable owning observation");
  session.publish(5, published, view(later.data(), 2, 2, 4));
  session.readForwarded(6, view(nullptr, 2, 2, 4), 2, 2, {}, result);
  session.observe(7, view(nullptr, 2, 2, 4));
  session.complete(8);
  expect(sameStatus(failed, session.status()) &&
             session.allocationAttemptsForTesting() == attempts &&
             later == std::array<float, 4>{-2, -2, -2, -2},
         "sticky failure blocks all later forwarded reads and resource effects");
}

struct AllocationRun {
  h::Status status;
  std::uint64_t prefixAttempts = 0, finalAttempts = 0;
  bool finalValue = false, retainedStorage = false;
  std::array<float, 4> output{};
};

AllocationRun allocationRun(bool useForwarding, std::uint64_t failAt) {
  std::array<float, 4> data{1, 2, 3, 4};
  AllocationRun result;
  result.output.fill(-1);
  h::Session session;
  session.configureForTesting({failAt, nullptr, nullptr});
  h::Value published, late;
  session.read(1, view(data.data(), 2, 2, 4), published);
  session.publish(2, published, view(result.output.data(), 2, 2, 4));
  session.observe(3, view(result.output.data(), 2, 2, 4));
  result.prefixAttempts = session.allocationAttemptsForTesting();
  if (useForwarding)
    session.readForwarded(4, view(result.output.data(), 2, 2, 4), 2, 2,
                          published, late);
  else
    session.read(4, view(result.output.data(), 2, 2, 4), 2, 2, late);
  result.finalValue = contents(late, data);
  result.retainedStorage = late.valid() && late.data() == published.data();
  session.complete(5);
  result.status = session.status();
  result.finalAttempts = session.allocationAttemptsForTesting();
  return result;
}

void removedAllocationOpportunity() {
  const auto forwarded = allocationRun(true, 0);
  const auto ordinary = allocationRun(false, 0);
  expect(forwarded.status && ordinary.status &&
             sameStatus(forwarded.status, ordinary.status) &&
             forwarded.finalValue && ordinary.finalValue,
         "successful forwarding preserves ordinary result and frontier trace");
  expect(forwarded.prefixAttempts == ordinary.prefixAttempts &&
             forwarded.finalAttempts == forwarded.prefixAttempts &&
             ordinary.finalAttempts > ordinary.prefixAttempts &&
             forwarded.retainedStorage && !ordinary.retainedStorage,
         "forwarded read retains storage without the ordinary snapshot allocation");

  const auto firstRemoved = forwarded.prefixAttempts + 1;
  const auto protectedRun = allocationRun(true, firstRemoved);
  const auto failingRun = allocationRun(false, firstRemoved);
  expect(protectedRun.status && protectedRun.status.completed && protectedRun.finalValue,
         "eliminated allocation opportunity cannot fail the forwarded read");
  expect(failingRun.status.code == h::Code::allocation_failure &&
             failingRun.status.failed_frontier == 4 &&
             failingRun.status.completed_frontier == 3 &&
             failingRun.status.completed_effect_frontier == 3 &&
             failingRun.status.publications == 1 && failingRun.status.observations == 1 &&
             failingRun.output == protectedRun.output,
         "ordinary allocation failure and optimized success retain permitted effect prefixes");

  for (std::uint64_t failAt = 1; failAt <= forwarded.prefixAttempts; ++failAt) {
    const auto a = allocationRun(false, failAt);
    const auto b = allocationRun(true, failAt);
    expect(a.status.code == h::Code::allocation_failure && sameStatus(a.status, b.status) &&
               a.output == b.output && a.finalAttempts == b.finalAttempts,
           "every earlier allocation failure remains sticky with identical effect prefix");
  }
}

void immutableOwnershipAndOverlap() {
  const std::array<std::uint32_t, 8> bits{
      0x80000000U, 0x00000000U, 0x00000001U, 0x7fc12345U,
      0x7f800000U, 0xff800000U, 0x3f800000U, 0x40000000U};
  std::array<float, 8> expected{};
  for (std::size_t i = 0; i < bits.size(); ++i)
    expected[i] = std::bit_cast<float>(bits[i]);
  std::array<float, 9> external{};
  const std::array<float, 8> replacement{11, 12, 13, 14, 15, 16, 17, 18};
  h::Value forwarded, observed;
  {
    h::Session session;
    h::Value published, next, late;
    session.read(1, view(expected.data(), 2, 4, 8), published);
    session.read(2, view(const_cast<float *>(replacement.data()), 2, 4, 8), next);
    session.publish(3, published, view(external.data(), 2, 4, 9));
    session.observe(4, view(external.data(), 2, 4, 9));
    const auto attempts = session.allocationAttemptsForTesting();
    expect(static_cast<bool>(session.readForwarded(
               5, view(external.data(), 2, 4, 9), 2, 4, published, forwarded)) &&
               session.allocationAttemptsForTesting() == attempts &&
               forwarded.data() == published.data() && contents(forwarded, expected),
           "forwarding preserves exact published bits in shared immutable storage");
    observed = session.observation(0);
    session.publish(6, next, view(external.data() + 1, 2, 4, 8));
    // A compiler must invalidate forwarding after this MAY-alias publication.
    // The ordinary late read is intentional; this test does not pretend the
    // private runtime entry can authenticate an arbitrary source derivation.
    session.read(7, view(external.data(), 2, 4, 9), late);
    std::array<float, 8> lateExpected{};
    lateExpected[0] = expected[0];
    for (std::size_t i = 1; i < lateExpected.size(); ++i)
      lateExpected[i] = replacement[i - 1];
    expect(contents(late, lateExpected) && contents(forwarded, expected) &&
               contents(observed, expected),
           "partial-overlap publication changes late read without changing old retained values");
    expect(static_cast<bool>(session.complete(8)), "ownership example completes");
  }
  external.fill(99);
  expect(contents(forwarded, expected) && contents(observed, expected),
         "forwarded and observed values outlive session, producer handles and external storage changes");
}

void emptyAndSelfAliasing() {
  for (const auto geometry : std::array<std::array<std::uint64_t, 2>, 3>{
           std::array<std::uint64_t, 2>{0, 7}, {7, 0}, {0, 0}}) {
    h::Session session;
    h::Value published, forwarded;
    const auto resource = view(nullptr, geometry[0], geometry[1], 0);
    session.read(1, resource, published);
    session.publish(2, published, resource);
    const auto attempts = session.allocationAttemptsForTesting();
    const auto status = session.readForwarded(
        3, resource, geometry[0], geometry[1], published, forwarded);
    expect(status && forwarded.valid() && forwarded.rows() == geometry[0] &&
               forwarded.columns() == geometry[1] && status.completed_frontier == 3 &&
               status.completed_effect_frontier == 2 && status.publications == 1 &&
               session.allocationAttemptsForTesting() == attempts,
           "empty forwarded read preserves dimensions and never dereferences null storage");
  }
  std::array<float, 4> data{1, 2, 3, 4}, output{};
  h::Session session;
  h::Value published;
  session.read(1, view(data.data(), 2, 2, 4), published);
  session.publish(2, published, view(output.data(), 2, 2, 4));
  const auto *original = published.data();
  const auto status = session.readForwarded(
      3, view(output.data(), 2, 2, 4), 2, 2, published, published);
  expect(status && published.data() == original && contents(published, data),
         "retained source and forwarded result may be the same private Value handle");
}

void orderedNoncommutingInputs() {
  std::array<float, 4> a{1, 2, 3, 4}, b{2, 0, 1, 3}, d{1, 2, 4, 1}, external{};
  h::Session session;
  h::Value av, bv, dv, c, forwarded, left, right, both;
  session.read(1, view(a.data(), 2, 2, 4), av);
  session.read(2, view(b.data(), 2, 2, 4), bv);
  session.read(3, view(d.data(), 2, 2, 4), dv);
  session.gemm(4, av, bv, h::Numeric::strict_f32, c);
  session.publish(5, c, view(external.data(), 2, 2, 4));
  session.readForwarded(6, view(external.data(), 2, 2, 4), 2, 2, c, forwarded);
  session.gemm(7, forwarded, dv, h::Numeric::strict_f32, left);
  session.gemm(8, dv, forwarded, h::Numeric::strict_f32, right);
  session.gemm(9, forwarded, forwarded, h::Numeric::strict_f32, both);
  expect(session.status() && contents(c, std::array<float, 4>{4, 6, 10, 12}) &&
             contents(left, std::array<float, 4>{28, 14, 58, 32}) &&
             contents(right, std::array<float, 4>{24, 30, 26, 36}) &&
             contents(both, std::array<float, 4>{76, 96, 160, 204}),
         "forwarded lhs, rhs and same-handle C*C preserve noncommuting operand order");
}

struct Reentry {
  h::Session *session;
  h::ResourceView resource;
  h::Value published, result;
  h::Status nested;
};

h::Code reenterForwardedRead(h::detail::CandidateInput, h::detail::CandidateInput,
                            h::detail::CandidateOutput output, void *opaque) {
  auto &state = *static_cast<Reentry *>(opaque);
  state.nested = state.session->readForwarded(
      100, state.resource, state.resource.rows, state.resource.columns,
      state.published, state.result);
  if (output.rows && output.columns) output.data[0] = 913;
  return h::Code::ok;
}

void frontierAndReentryGuards() {
  std::array<float, 4> data{1, 2, 3, 4}, external{};
  const auto valid = seed(data.data(), 2, 2, 4);
  {
    h::Session session;
    h::Value result;
    const auto status = session.readForwarded(
        0, view(nullptr, 2, 2, 4), 2, 2, {}, result);
    expect(status.code == h::Code::invalid_frontier && status.failed_frontier == 0 &&
               !result.valid(),
           "zero frontier rejects before inspecting malformed forwarding inputs");
  }
  {
    h::Session session;
    h::Value result;
    session.publish(2, valid, view(external.data(), 2, 2, 4));
    const auto status = session.readForwarded(
        2, view(nullptr, 2, 2, 4), 2, 2, {}, result);
    expect(status.code == h::Code::invalid_frontier && status.failed_frontier == 2 &&
               status.completed_frontier == 2 && status.publications == 1,
           "nonincreasing forwarded frontier cannot hide behind invalid retained input");
  }
  {
    h::Session session;
    h::Value result;
    session.complete(1);
    auto retired = std::move(session).takeResult();
    const auto status = session.readForwarded(
        2, view(nullptr, 2, 2, 4), 2, 2, {}, result);
    expect(retired.ok() && status.code == h::Code::already_complete &&
               status.completed && status.completed_frontier == 1 && !result.valid(),
           "retired session cannot be restarted through forwarding");
  }
  {
    h::Session session;
    Reentry context{&session, view(external.data(), 2, 2, 4), valid, valid, {}};
    session.configureForTesting({0, reenterForwardedRead, &context});
    session.publish(1, valid, context.resource);
    h::Value result;
    const auto status = session.gemm(2, valid, valid, h::Numeric::strict_f32, result);
    expect(status.code == h::Code::reentrant_use && status.failed_frontier == 2 &&
               sameStatus(status, context.nested) && status.completed_frontier == 1 &&
               status.publications == 1 && !result.valid() &&
               context.result.data() == valid.data() && external == data,
           "forwarding reentry preserves outer frontier and prevents private partial-result exposure");
  }
}

void preservesCallerFloatingPointState() {
  std::array<float, 4> data{1, 2, 3, 4}, external{};
  h::Session session;
  h::Value published, result;
  session.read(1, view(data.data(), 2, 2, 4), published);
  session.publish(2, published, view(external.data(), 2, 2, 4));
  fenv_t saved;
  expect(fegetenv(&saved) == 0 && fesetround(FE_DOWNWARD) == 0 &&
             feraiseexcept(FE_DIVBYZERO | FE_INEXACT) == 0,
         "caller floating-point fixture is established");
  const auto before = closed_fp_fixture::snapshot();
  const auto success = session.readForwarded(
      3, view(external.data(), 2, 2, 4), 2, 2, published, result);
  const auto afterSuccess = closed_fp_fixture::snapshot();
  const auto failure = session.readForwarded(
      4, view(external.data(), 2, 2, 4), 1, 4, published, result);
  const auto afterFailure = closed_fp_fixture::snapshot();
  expect(fesetenv(&saved) == 0, "caller floating-point fixture is restored");
  expect(success && failure.code == h::Code::shape_mismatch &&
             before == afterSuccess && before == afterFailure,
         "successful and failed forwarded reads preserve complete caller FP state");
}
} // namespace

int main() {
  originalReadGuardOrder();
  orderedFailurePrefix();
  removedAllocationOpportunity();
  immutableOwnershipAndOverlap();
  emptyAndSelfAliasing();
  orderedNoncommutingInputs();
  frontierAndReentryGuards();
  preservesCallerFloatingPointState();
  std::printf("Independent forwarded-read checks: %u; failures: %u\n", checks, failures);
  return failures ? 1 : 0;
}
