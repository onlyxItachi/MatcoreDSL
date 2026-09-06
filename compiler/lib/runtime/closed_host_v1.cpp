#include "closed_host_v1.h"

#include <cfenv>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <exception>
#include <limits>
#include <new>
#include <mutex>

#if defined(MDSLC_CLOSED_HOST_LEGACY_CANDIDATES)
#include "matcore/runtime_c.h"
#endif

#if defined(__FAST_MATH__)
#error "closed host strict-f32 adapter must not be compiled with fast math"
#endif

#if defined(__linux__) && defined(__x86_64__)
#include <xmmintrin.h>
#endif

#if defined(__clang__)
#pragma STDC FENV_ACCESS ON
#pragma STDC FP_CONTRACT OFF
#endif

namespace matcore::mdslc::runtime::closed_host_v1 {

static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559 &&
              std::numeric_limits<float>::digits == 24);

struct ValueStorage {
  std::uint64_t rows = 0;
  std::uint64_t columns = 0;
  std::vector<float> elements;
};

namespace {
#if defined(MDSLC_CLOSED_HOST_GENERATED_STRICT)
// Private pinned MLIR 21 x86-64 identity-memref ABI, never an installed type.
struct GeneratedMemref {
  float *allocated;
  float *aligned;
  std::int64_t offset;
  std::int64_t sizes[2];
  std::int64_t strides[2];
};
static_assert(sizeof(GeneratedMemref) == 56 && alignof(GeneratedMemref) == 8);
static_assert(offsetof(GeneratedMemref, sizes) == 24 &&
              offsetof(GeneratedMemref, strides) == 40);
extern "C" void _mlir_ciface___matcore_strict_gemm_f32_v1(
    GeneratedMemref *, GeneratedMemref *, GeneratedMemref *);
GeneratedMemref descriptor(const ValueStorage &value) noexcept {
  auto *data = const_cast<float *>(value.elements.data());
  return {data, data, 0,
          {static_cast<std::int64_t>(value.rows),
           static_cast<std::int64_t>(value.columns)},
          {static_cast<std::int64_t>(value.columns), 1}};
}
#endif

Candidate selectedCandidate(Candidate request) noexcept {
  if (request != Candidate::automatic) return request;
#if defined(MDSLC_CLOSED_HOST_GENERATED_STRICT)
  return Candidate::generated_strict;
#else
  return Candidate::native_strict;
#endif
}

Code candidateLegality(Candidate request, Numeric numeric) noexcept {
  switch (selectedCandidate(request)) {
  case Candidate::native_strict: return Code::ok;
  case Candidate::generated_strict:
#if defined(MDSLC_CLOSED_HOST_GENERATED_STRICT)
    return Code::ok;
#else
    return Code::candidate_unavailable;
#endif
  case Candidate::existing_native:
  case Candidate::authenticated_openblas:
    if (numeric != Numeric::reassociate_f32)
      return Code::candidate_incompatible;
#if !defined(MDSLC_CLOSED_HOST_LEGACY_CANDIDATES)
    return Code::candidate_unavailable;
#else
    if (request == Candidate::authenticated_openblas) {
#if !defined(MDSLC_CLOSED_HOST_OPENBLAS)
      return Code::candidate_unavailable;
#endif
    }
    return Code::ok;
#endif
  case Candidate::automatic: break;
  }
  return Code::invalid_candidate;
}

#if defined(MDSLC_CLOSED_HOST_LEGACY_CANDIDATES)
matcore_tensor_desc_v0 legacyDescriptor(const float *data, std::uint64_t rows,
                                      std::uint64_t columns,
                                      bool writable) noexcept {
  matcore_tensor_desc_v0 descriptor{};
  descriptor.struct_size = sizeof(descriptor);
  descriptor.data = const_cast<float *>(data);
  descriptor.dtype = MATCORE_DTYPE_F32_V0;
  descriptor.rank = 2;
  descriptor.dims[0] = static_cast<std::int64_t>(rows);
  descriptor.dims[1] = static_cast<std::int64_t>(columns);
  descriptor.strides[0] = descriptor.dims[1];
  descriptor.strides[1] = 1;
  descriptor.memory_space = MATCORE_MEMORY_SPACE_HOST_V0;
  descriptor.mutability = writable ? MATCORE_MUTABILITY_READ_WRITE_V0
                                   : MATCORE_MUTABILITY_READ_ONLY_V0;
  return descriptor;
}

Code legacyGemm(Candidate candidate, const float *lhs, const float *rhs,
                float *output, std::uint64_t m, std::uint64_t n,
                std::uint64_t k, CandidateReport &evidence) noexcept {
  auto a = legacyDescriptor(lhs, m, k, false);
  auto b = legacyDescriptor(rhs, k, n, false);
  auto c = legacyDescriptor(output, m, n, true);
  matcore_policy_v0 policy{};
  policy.struct_size = sizeof(policy);
  policy.target = MATCORE_TARGET_CPU_V0;
  policy.fallback = MATCORE_FALLBACK_ERROR_V0;
  matcore_cpu_gemm_execution_options_v1 options{};
  options.abi_version = MATCORE_RUNTIME_EXECUTION_ABI_VERSION_V1;
  options.struct_size = sizeof(options);
  options.requested_threads = 1;
  const bool provider = candidate == Candidate::authenticated_openblas;
  options.request = provider ? MATCORE_CPU_GEMM_REQUEST_FORCE_EXTERNAL_OPENBLAS_V2
                            : MATCORE_CPU_GEMM_REQUEST_FORCE_REFERENCE_V2;
  matcore_cpu_gemm_plan_report_v2 report{};
  report.abi_version = MATCORE_RUNTIME_PLAN_ABI_VERSION_V2;
  report.struct_size = sizeof(report);
  evidence.invocation_attempted = true;
  const auto status = matcore_runtime_gemm_f32_execute_v1(
      &c, &a, &b, &policy, &options, nullptr, 0, &report);
  const char *expected = provider ? "cpu.external.openblas.f32.v1"
                                  : "cpu.reference.f32.v1";
  const bool identified = report.selected_stable_id != nullptr &&
                         std::strcmp(report.selected_stable_id, expected) == 0 &&
                         report.selected_actual_threads == 1 &&
                         report.selected_workspace_bytes == 0;
  if (identified) {
    evidence.actual = provider ? Implementation::authenticated_openblas
                              : Implementation::existing_reference;
    evidence.actual_threads = 1;
  }
  if (status.code == MATCORE_STATUS_UNAVAILABLE_VARIANT_V0)
    return Code::candidate_unavailable;
  if (status.code != MATCORE_STATUS_OK_V0 || !identified ||
      report.plan_status != MATCORE_CPU_PLAN_STATUS_SELECTED_V1)
    return Code::candidate_failure;
  return Code::ok;
}

bool sameNumber(float a, float b) noexcept {
  return (std::isnan(a) && std::isnan(b)) ||
         std::bit_cast<std::uint32_t>(a) == std::bit_cast<std::uint32_t>(b);
}

bool allowedPair(float result, float a, float b, float c, float d) noexcept {
  const float p = a * b, q = c * d, z = 0.0F;
  const float zp = z + p, zq = z + q, pq = p + q;
  // This bounded probe checks membership, not agreement with one arbitrary
  // reassociated oracle. NaN payloads are deliberately outside the contract.
  for (float value : std::array<float, 9>{zp + q, zq + p, z + pq,
           std::fma(a, b, zq), std::fma(c, d, zp),
           std::fma(a, b, std::fma(c, d, z)),
           std::fma(c, d, std::fma(a, b, z)),
           z + std::fma(a, b, q), z + std::fma(c, d, p)})
    if (sameNumber(result, value)) return true;
  return false;
}

Code closedProviderContract(CandidateReport &evidence) noexcept {
  static std::once_flag once;
  static Code result = Code::candidate_incompatible;
  evidence.provider_contract_checked = true;
  try {
    // Runs inside the full FP scope. Stack-only Matcore fixture preparation;
    // the unchanged runtime authenticates provider identity and thread/FP state.
    // This does not prove all versions/cores or substitute for release evidence.
    std::call_once(once, [&] {
      evidence.provider_probe_invoked = true;
      const float tiny = std::numeric_limits<float>::denorm_min();
      const float inf = std::numeric_limits<float>::infinity();
      const std::array<float, 8> a{-1, 0x1.000002p0F, tiny, 0, inf, -inf, -0.0F, -0.0F};
      const std::array<float, 8> b{1, 1, 0, -1, 0x1.fffffep-1F, 1, 1, -1};
      std::array<float, 16> out{};
      CandidateReport probe;
      auto code = legacyGemm(Candidate::authenticated_openblas, a.data(), b.data(),
                             out.data(), 4, 4, 2, probe);
      if (code != Code::ok) { result = code; return; }
      for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j)
          if (!allowedPair(out[4*i+j], a[2*i], b[j], a[2*i+1], b[4+j]))
            return;
      for (const auto pair : std::array<std::array<float, 2>, 4>{
             std::array<float, 2>{-0.0F, 1.0F}, {tiny, 0.5F}, {inf, 0.0F}, {tiny, 1.0F}}) {
        float scalar = 99;
        code = legacyGemm(Candidate::authenticated_openblas, &pair[0], &pair[1],
                          &scalar, 1, 1, 1, probe);
        if (code != Code::ok) { result = code; return; }
        const float product = pair[0] * pair[1], separate = 0.0F + product;
        if (!sameNumber(scalar, separate) &&
            !sameNumber(scalar, std::fma(pair[0], pair[1], 0.0F))) return;
      }
      result = Code::ok;
    });
  } catch (...) {
    return Code::candidate_failure;
  }
  return result;
}
#endif

constexpr bool supportedPlatform() noexcept {
#if defined(__linux__) && defined(__x86_64__)
  return true;
#else
  return false;
#endif
}

Code extent(std::uint64_t rows, std::uint64_t columns,
            std::size_t &elements) noexcept {
  constexpr auto maximum = static_cast<std::uint64_t>(
      std::numeric_limits<std::int64_t>::max());
  if (rows > maximum || columns > maximum ||
      (rows != 0 && columns > maximum / rows))
    return Code::extent_overflow;
  const auto count = rows * columns;
  if (count > std::numeric_limits<std::size_t>::max() / sizeof(float) ||
      count > static_cast<std::uint64_t>(
                  std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(float))
    return Code::extent_overflow;
  elements = static_cast<std::size_t>(count);
  return Code::ok;
}

Code validate(ResourceView view, bool write, std::size_t &elements) noexcept {
  if (view.access != Access::read_only && view.access != Access::read_write)
    return Code::invalid_view;
  auto code = extent(view.rows, view.columns, elements);
  if (code != Code::ok) return code;
  if (view.capacity_elements < elements) return Code::insufficient_capacity;
  if (write && view.access != Access::read_write) return Code::access_denied;
  if (elements == 0) return Code::ok;
  const auto address = reinterpret_cast<std::uintptr_t>(view.data);
  if (view.data == nullptr || address % alignof(float) != 0)
    return Code::invalid_view;
  if (elements * sizeof(float) >
      std::numeric_limits<std::uintptr_t>::max() - address)
    return Code::extent_overflow;
  return Code::ok;
}

// Restore the complete thread environment, including sticky exception flags.
// The old runtime's control-only helper intentionally is not reused here.
// A failure to restore a previously captured supported-host environment is an
// unrecoverable adapter malfunction: returning normally would violate the
// contract. No recoverable status is claimed for that impossible-to-isolate path.
class ScopedFp {
public:
  ScopedFp() noexcept {
#if defined(__linux__) && defined(__x86_64__)
    if (std::fegetenv(&saved_) != 0) return;
    saved_mxcsr_ = _mm_getcsr();
    __asm__ volatile("fnstcw %0" : "=m"(saved_control_));
    __asm__ volatile("fnstsw %0" : "=am"(saved_status_));
    captured_ = true;
    if (std::fesetenv(FE_DFL_ENV) != 0) { restore(); return; }
    expected_mxcsr_ = _mm_getcsr();
    __asm__ volatile("fnstcw %0" : "=m"(expected_control_));
    valid_ = (expected_mxcsr_ & 0xFFC0U) == 0x1F80U &&
             (expected_control_ & 0x0C3FU) == 0x003FU;
    if (!valid_) restore();
#endif
  }
  ~ScopedFp() { restore(); }
  bool valid() const noexcept { return valid_; }
  bool controlsUnchanged() const noexcept {
#if defined(__linux__) && defined(__x86_64__)
    std::uint16_t control = 0;
    __asm__ volatile("fnstcw %0" : "=m"(control));
    return valid_ && (_mm_getcsr() & ~0x3FU) ==
                         (expected_mxcsr_ & ~0x3FU) &&
           control == expected_control_;
#else
    return false;
#endif
  }
  void restore() noexcept {
#if defined(__linux__) && defined(__x86_64__)
    if (!captured_) return;
    if (std::fesetenv(&saved_) != 0) std::terminate();
    std::uint16_t control = 0, status = 0;
    __asm__ volatile("fnstcw %0" : "=m"(control));
    __asm__ volatile("fnstsw %0" : "=am"(status));
    if (_mm_getcsr() != saved_mxcsr_ || control != saved_control_ ||
        status != saved_status_)
      std::terminate();
    captured_ = false;
#endif
  }
private:
  std::fenv_t saved_{};
  bool captured_ = false;
  bool valid_ = false;
  std::uint32_t saved_mxcsr_ = 0, expected_mxcsr_ = 0;
  std::uint16_t saved_control_ = 0, saved_status_ = 0, expected_control_ = 0;
};

void strictGemm(const ValueStorage &lhs, const ValueStorage &rhs,
                ValueStorage &result) noexcept {
  if (result.elements.empty()) return;
  for (std::uint64_t i = 0; i < lhs.rows; ++i) {
    for (std::uint64_t j = 0; j < rhs.columns; ++j) {
      float sum = 0.0F;
      for (std::uint64_t k = 0; k < lhs.columns; ++k) {
        const float product = lhs.elements[i * lhs.columns + k] *
                              rhs.elements[k * rhs.columns + j];
        sum = sum + product;
      }
      result.elements[i * rhs.columns + j] = sum;
    }
  }
}
} // namespace

bool Value::valid() const noexcept { return static_cast<bool>(storage_); }
std::uint64_t Value::rows() const noexcept { return storage_ ? storage_->rows : 0; }
std::uint64_t Value::columns() const noexcept {
  return storage_ ? storage_->columns : 0;
}

struct Session::ActiveCall {
  explicit ActiveCall(Session &session) noexcept : session(session) {}
  ~ActiveCall() { session.active_ = false; session.active_frontier_ = 0; }
  Session &session;
};

Status Session::fail(Code code, Frontier frontier) noexcept {
  if (status_.code == Code::ok) {
    status_.code = code;
    status_.failed_frontier = frontier;
  }
  return status_;
}
Status Session::succeed(Frontier frontier) noexcept {
  if (status_) status_.completed_frontier = frontier;
  return status_;
}
bool Session::begin(Frontier frontier) noexcept {
  if (!status_) return false;
  if (active_) { fail(Code::reentrant_use, active_frontier_); return false; }
  if (status_.completed) { fail(Code::already_complete, frontier); return false; }
  if (frontier == 0 || frontier <= status_.completed_frontier) {
    fail(Code::invalid_frontier, frontier);
    return false;
  }
  if (!supportedPlatform()) {
    fail(Code::unsupported_fp_environment, frontier);
    return false;
  }
  active_ = true;
  active_frontier_ = frontier;
  return true;
}
bool Session::allocationAllowed() noexcept {
#if defined(MDSLC_CLOSED_HOST_TESTING)
  ++allocation_attempts_;
  return fail_allocation_ == 0 || allocation_attempts_ != fail_allocation_;
#else
  return true;
#endif
}
Code Session::allocate(std::uint64_t rows, std::uint64_t columns,
                       std::shared_ptr<ValueStorage> &storage) noexcept {
  std::size_t count = 0;
  const auto code = extent(rows, columns, count);
  if (code != Code::ok) return code;
  try {
    if (!allocationAllowed()) return Code::allocation_failure;
    auto fresh = std::make_shared<ValueStorage>();
    fresh->rows = rows;
    fresh->columns = columns;
    if (count != 0) {
      if (!allocationAllowed()) return Code::allocation_failure;
      fresh->elements.resize(count);
    }
    storage = std::move(fresh);
    return Code::ok;
  } catch (...) {
    return Code::allocation_failure;
  }
}
Code Session::snapshot(ResourceView view,
                       std::shared_ptr<ValueStorage> &storage) noexcept {
  std::size_t count = 0;
  auto code = validate(view, false, count);
  if (code != Code::ok) return code;
  code = allocate(view.rows, view.columns, storage);
  if (code != Code::ok) return code;
  if (count != 0)
    std::memcpy(storage->elements.data(), view.data, count * sizeof(float));
  return Code::ok;
}

Status Session::read(Frontier frontier, ResourceView view, Value &result) noexcept {
  return read(frontier, view, view.rows, view.columns, result);
}
Status Session::read(Frontier frontier, ResourceView view,
                     std::uint64_t rows, std::uint64_t columns,
                     Value &result) noexcept {
  if (!begin(frontier)) return status_;
  ActiveCall active(*this);
  std::size_t count = 0;
  const auto requested = extent(rows, columns, count);
  if (requested != Code::ok) return fail(requested, frontier);
  if (view.rows != rows || view.columns != columns)
    return fail(Code::shape_mismatch, frontier);
  std::shared_ptr<ValueStorage> storage;
  const auto code = snapshot(view, storage);
  if (code != Code::ok) return fail(code, frontier);
  result.storage_ = std::move(storage);
  return succeed(frontier);
}
Status Session::gemm(Frontier frontier, const Value &lhs, const Value &rhs,
                     Numeric numeric, Value &result) noexcept {
  if (!begin(frontier)) return status_;
  ActiveCall active(*this);
  candidate_report_ = {frontier, options_.candidate, Implementation::none, numeric};
  const auto rejected = [&](Code code) noexcept {
    candidate_report_.code = code;
    return fail(code, frontier);
  };
  if (!lhs.valid() || !rhs.valid()) return rejected(Code::invalid_value);
  if (numeric != Numeric::strict_f32 && numeric != Numeric::reassociate_f32)
    return rejected(Code::invalid_value);
  if (lhs.columns() != rhs.rows()) return rejected(Code::shape_mismatch);
  auto code = candidateLegality(options_.candidate, numeric);
  if (code != Code::ok) return rejected(code);
  std::shared_ptr<ValueStorage> storage;
  code = allocate(lhs.rows(), rhs.columns(), storage);
  if (code != Code::ok) return rejected(code);
  ScopedFp environment;
  if (!environment.valid()) return rejected(Code::unsupported_fp_environment);
#if defined(MDSLC_CLOSED_HOST_LEGACY_CANDIDATES)
  if (options_.candidate == Candidate::authenticated_openblas) {
    code = closedProviderContract(candidate_report_);
    if (!environment.controlsUnchanged()) code = Code::candidate_failure;
    if (code != Code::ok) {
      environment.restore();
      return rejected(code);
    }
  }
#endif
#if defined(MDSLC_CLOSED_HOST_TESTING)
  if (test_candidate_ != nullptr) {
    candidate_report_.actual = Implementation::test_only;
    candidate_report_.invocation_attempted = true;
    try {
      code = test_candidate_(
          {lhs.storage_->elements.data(), lhs.rows(), lhs.columns()},
          {rhs.storage_->elements.data(), rhs.rows(), rhs.columns()},
          {storage->elements.data(), storage->rows, storage->columns},
          test_context_);
    } catch (...) {
      code = Code::candidate_failure;
    }
  } else
#endif
  {
    // Availability and numerical compatibility remain forced-choice obligations
    // even for empty math. Never enter generated zero-output outer loops or the
    // old runtime's positive-shape-only/BLAS quick-return contract for empties.
    if (storage->elements.empty()) {
      candidate_report_.actual = Implementation::empty_output;
    } else if (lhs.columns() == 0) {
      // vector<float> value initialization already produced positive zero.
      candidate_report_.actual = Implementation::zero_reduction;
    } else {
      switch (selectedCandidate(options_.candidate)) {
      case Candidate::native_strict:
        candidate_report_.actual = Implementation::native_strict;
        candidate_report_.invocation_attempted = true;
        candidate_report_.actual_threads = 1;
        strictGemm(*lhs.storage_, *rhs.storage_, *storage);
        break;
      case Candidate::generated_strict:
#if defined(MDSLC_CLOSED_HOST_GENERATED_STRICT)
        {
          auto a = descriptor(*lhs.storage_), b = descriptor(*rhs.storage_);
          auto c = descriptor(*storage);
          candidate_report_.actual = Implementation::generated_strict;
          candidate_report_.invocation_attempted = true;
          candidate_report_.actual_threads = 1;
          _mlir_ciface___matcore_strict_gemm_f32_v1(&a, &b, &c);
        }
#else
        code = Code::candidate_unavailable;
#endif
        break;
      case Candidate::existing_native:
      case Candidate::authenticated_openblas:
#if defined(MDSLC_CLOSED_HOST_LEGACY_CANDIDATES)
        code = legacyGemm(options_.candidate, lhs.storage_->elements.data(),
                          rhs.storage_->elements.data(), storage->elements.data(),
                          lhs.rows(), rhs.columns(), lhs.columns(), candidate_report_);
#else
        code = Code::candidate_unavailable;
#endif
        break;
      case Candidate::automatic: code = Code::invalid_candidate; break;
      }
    }
  }
  if (!environment.controlsUnchanged()) code = Code::candidate_failure;
  environment.restore();
  // Test-hook reentry may already have failed this Session. Never issue a value
  // or clear that error merely because the outer candidate returned success.
  if (!status_) { candidate_report_.code = status_.code; return status_; }
  if (code != Code::ok) return rejected(code);
  result.storage_ = std::move(storage);
  candidate_report_.value_issued = true;
  return succeed(frontier);
}
Status Session::publish(Frontier frontier, const Value &value,
                        ResourceView destination) noexcept {
  if (!begin(frontier)) return status_;
  ActiveCall active(*this);
  if (!value.valid()) return fail(Code::invalid_value, frontier);
  std::size_t count = 0;
  const auto code = validate(destination, true, count);
  if (code != Code::ok) return fail(code, frontier);
  if (value.rows() != destination.rows || value.columns() != destination.columns)
    return fail(Code::shape_mismatch, frontier);
  // No allocation, callback, numerical operation or recoverable check follows
  // the first write. Snapshots own their storage, independent of all views.
  if (count != 0)
    std::memmove(destination.data, value.storage_->elements.data(),
                 count * sizeof(float));
  ++status_.publications;
  status_.completed_effect_frontier = frontier;
  return succeed(frontier);
}
Status Session::observe(Frontier frontier, ResourceView view) noexcept {
  if (!begin(frontier)) return status_;
  ActiveCall active(*this);
  std::shared_ptr<ValueStorage> storage;
  const auto code = snapshot(view, storage);
  if (code != Code::ok) return fail(code, frontier);
  try {
    // Record-growth allocation must complete before the effect is retired.
    if (observations_.size() == observations_.capacity() && !allocationAllowed())
      return fail(Code::allocation_failure, frontier);
    Value value;
    value.storage_ = std::move(storage);
    observations_.push_back({frontier, std::move(value)});
  } catch (...) {
    return fail(Code::allocation_failure, frontier);
  }
  ++status_.observations;
  status_.completed_effect_frontier = frontier;
  return succeed(frontier);
}
Status Session::complete(Frontier frontier) noexcept {
  if (!begin(frontier)) return status_;
  ActiveCall active(*this);
  status_.completed = true;
  return succeed(frontier);
}
Value Session::observation(std::uint64_t index) const noexcept {
  return index < observations_.size() ? observations_[index].value : Value{};
}
Frontier Session::observationFrontier(std::uint64_t index) const noexcept {
  return index < observations_.size() ? observations_[index].frontier : 0;
}

#if defined(MDSLC_CLOSED_HOST_TESTING)
void Session::configureForTesting(TestHooks hooks) noexcept {
  if (active_) { fail(Code::reentrant_use, active_frontier_); return; }
  if (!status_) return;
  if (status_.completed_frontier != 0 || status_.completed) {
    fail(Code::invalid_frontier, 0);
    return;
  }
  fail_allocation_ = hooks.fail_allocation;
  allocation_attempts_ = 0;
  test_candidate_ = hooks.candidate;
  test_context_ = hooks.context;
}
std::uint64_t Session::allocationAttemptsForTesting() const noexcept {
  return allocation_attempts_;
}
#endif

const char *message(Code code) noexcept {
  switch (code) {
  case Code::ok: return "ok";
  case Code::invalid_frontier: return "frontier must increase in executed order";
  case Code::invalid_value: return "invalid immutable value or numerical profile";
  case Code::invalid_view: return "invalid dense host view";
  case Code::shape_mismatch: return "shape mismatch";
  case Code::extent_overflow: return "extent, byte size or address overflow";
  case Code::insufficient_capacity: return "insufficient declared storage capacity";
  case Code::access_denied: return "publication requires write access";
  case Code::allocation_failure: return "private storage allocation failed";
  case Code::candidate_failure: return "isolated candidate or environment validation failed";
  case Code::unsupported_fp_environment: return "unsupported full floating-point environment adapter";
  case Code::reentrant_use: return "same-session reentry is forbidden";
  case Code::already_complete: return "session is already complete";
  case Code::invalid_candidate: return "unknown compile-trusted candidate request";
  case Code::candidate_unavailable: return "forced candidate unavailable; fallback forbidden";
  case Code::candidate_incompatible: return "forced candidate violates numerical permissions";
  }
  return "unknown closed-host status";
}

const char *implementationName(Implementation implementation) noexcept {
  switch (implementation) {
  case Implementation::none: return "none";
  case Implementation::native_strict: return "closed.native.strict_f32.v1";
  case Implementation::generated_strict: return "closed.generated.strict_f32.mlir21.v1";
  case Implementation::existing_reference: return "cpu.reference.f32.v1";
  case Implementation::authenticated_openblas: return "cpu.external.openblas.f32.v1";
  case Implementation::empty_output: return "closed.semantic.empty_output.v1";
  case Implementation::zero_reduction: return "closed.semantic.zero_reduction.v1";
  case Implementation::test_only: return "test-only-injected-candidate";
  }
  return "invalid";
}

} // namespace matcore::mdslc::runtime::closed_host_v1
