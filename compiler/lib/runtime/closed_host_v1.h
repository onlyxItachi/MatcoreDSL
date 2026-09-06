#ifndef MATCORE_MDSLC_RUNTIME_CLOSED_HOST_V1_H
#define MATCORE_MDSLC_RUNTIME_CLOSED_HOST_V1_H

#include <matcore/region.h>
#include <memory>
#include <vector>

namespace matcore::mdslc::runtime::closed_host_v1 {

// Private compile-trusted registry. No source/serialized identifier or callback
// creates a candidate. Default sessions retain the original strict native path;
// automatic explicitly chooses linked strict generated, otherwise strict native.
enum class Candidate : std::uint8_t {
  automatic, native_strict, generated_strict, existing_native,
  authenticated_openblas
};
enum class Implementation : std::uint8_t {
  none, native_strict, generated_strict, existing_reference,
  authenticated_openblas, empty_output, zero_reduction, test_only
};
struct Options { Candidate candidate = Candidate::native_strict; };
struct CandidateReport {
  Frontier frontier = 0;
  Candidate requested = Candidate::native_strict;
  Implementation actual = Implementation::none;
  Numeric numeric = Numeric::strict_f32;
  Code code = Code::ok;
  // A selected implementation may fail before producing a value. For a legacy
  // route, actual is set only after its own report confirms the fixed variant.
  bool invocation_attempted = false;
  bool value_issued = false;
  // Authentication probes are distinct from the requested GEMM: a first forced
  // provider request, including empty math, may run the fixed private probe.
  bool provider_contract_checked = false;
  bool provider_probe_invoked = false;
  std::uint32_t actual_threads = 0;
};
struct ValueStorage;
class Value {
public:
  Value() noexcept = default;
  bool valid() const noexcept;
  std::uint64_t rows() const noexcept;
  std::uint64_t columns() const noexcept;
  const float *data() const noexcept;
private:
  friend class Session;
  std::shared_ptr<const ValueStorage> storage_;
};
struct Observation { Frontier frontier; Value value; };

// Private execution adapter, not a public source API or frozen ABI. This adapter
// does not authenticate source, accept serialized authority, or interpret an AST.
// These shapes describe only the private test injection ABI. No production
// method accepts a candidate function. Production uses the closed registry above.
namespace detail {
struct CandidateInput {
  const float *data;
  std::uint64_t rows;
  std::uint64_t columns;
};
struct CandidateOutput {
  float *data;
  std::uint64_t rows;
  std::uint64_t columns;
};
using TestCandidate = Code (*)(CandidateInput, CandidateInput, CandidateOutput,
                              void *context);
} // namespace detail

#if defined(MDSLC_CLOSED_HOST_TESTING)
struct TestHooks {
  // Fail this numbered attempted owned allocation, starting at one; zero off.
  std::uint64_t fail_allocation = 0;
  detail::TestCandidate candidate = nullptr;
  void *context = nullptr;
};
#endif

// Valid ordinary host float objects, declared capacity/lifetime/access and no
// concurrent conflicting access are caller preconditions, not pointer proofs.
// All external views MAY overlap. A read eagerly snapshots in this initial
// realization; immutable value semantics do not require eager copies generally.
// A successful publish replaces its complete destination under normal-return
// semantics. All fallible preparation precedes its byte copy. This is neither
// concurrent atomicity nor crash recovery, and does not cover device/file export.
// Earlier publications survive later failure; failure is sticky and later calls
// perform no effects. Frontiers are positive and strictly increasing along the
// actual executed path, supplied by an authenticated generated wrapper.
// A Session is thread-confined. Reentry detection is not a lock or permission
// for concurrent calls, including calls that use disjoint external views.
// Allocation attempts/counts and instrumentation of replaceable allocation
// functions are realization details, not stable mathematical trace events.
// Returned allocation-failure frontiers and completed effect prefixes are kept.
// This bounded adapter trusts the linked runtime allocator/deallocator contract;
// it does not sandbox arbitrary interposed hooks that mutate host/FP state.
// In particular allocation, failure and deallocation must preserve caller FP
// state and introduce no arbitrary host effects. They can occur outside the
// numerical scope, including result destruction after a candidate returns.
class Session {
public:
  Session() noexcept = default;
  explicit Session(Options options) noexcept : options_(options) {}
  ~Session() noexcept;
  Session(const Session &) = delete;
  Session &operator=(const Session &) = delete;
  Session(Session &&) = delete;
  Session &operator=(Session &&) = delete;

  Status read(Frontier, ResourceView, Value &) noexcept;
  Status read(Frontier, ResourceView, std::uint64_t requested_rows,
              std::uint64_t requested_columns, Value &) noexcept;
  Status gemm(Frontier, const Value &, const Value &, Numeric, Value &) noexcept;
  Status publish(Frontier, const Value &, ResourceView) noexcept;
  // Observation captures immutable contents at this frontier. It is not an
  // arbitrary host callback, an external export, or merely a counter increment.
  Status observe(Frontier, ResourceView) noexcept;
  Status complete(Frontier) noexcept;
  Status status() const noexcept { return status_; }
  // Value snapshot of the last attempted GEMM, not mutable registry authority.
  // Later non-GEMM operations and sticky failures do not rewrite this report.
  CandidateReport candidateReport() const noexcept { return candidate_report_; }
  // Returns a stable immutable handle, not a pointer invalidated by record growth.
  Value observation(std::uint64_t index) const noexcept;
  Frontier observationFrontier(std::uint64_t index) const noexcept;
  // Generated wrappers retire a completed/failed invocation exactly once.
  // Moving already-owned observation records cannot allocate or fail after
  // publication. Diagnostic strings have compiler-owned static lifetime.
  matcore::mdsl::Result takeResult(
      matcore::mdsl::SourceLocation failure = {}) && noexcept;

#if defined(MDSLC_CLOSED_HOST_TESTING)
  // Only exported by the separately compiled test variant, never production.
  // Configure before the first operation. Misuse fails at diagnostic frontier
  // zero (there is no source operation for test-harness configuration).
  void configureForTesting(TestHooks) noexcept;
  std::uint64_t allocationAttemptsForTesting() const noexcept;
#endif

private:
  struct ActiveCall;
  bool begin(Frontier) noexcept;
  Status fail(Code, Frontier) noexcept;
  Status succeed(Frontier) noexcept;
  bool allocationAllowed() noexcept;
  Code snapshot(ResourceView, std::shared_ptr<ValueStorage> &) noexcept;
  Code allocate(std::uint64_t rows, std::uint64_t columns,
                std::shared_ptr<ValueStorage> &) noexcept;
  Status status_;
  Options options_;
  CandidateReport candidate_report_;
  bool active_ = false;
  bool result_taken_ = false;
  Frontier active_frontier_ = 0;
  ObservationBlock *observations_ = nullptr;
  // Fixed layout across the production/test declarations; production has no
  // setter and compiles out every callback/injection use.
  [[maybe_unused]] std::uint64_t allocation_attempts_ = 0;
  [[maybe_unused]] std::uint64_t fail_allocation_ = 0;
  [[maybe_unused]] detail::TestCandidate test_candidate_ = nullptr;
  [[maybe_unused]] void *test_context_ = nullptr;
};

const char *message(Code) noexcept;
const char *implementationName(Implementation) noexcept;

} // namespace matcore::mdslc::runtime::closed_host_v1
#endif
