#include "closed_host_v1.h"

#include <cfenv>
#include <cstddef>
#include <cstring>
#include <exception>
#include <limits>
#include <new>

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
  if (!lhs.valid() || !rhs.valid()) return fail(Code::invalid_value, frontier);
  if (numeric != Numeric::strict_f32 && numeric != Numeric::reassociate_f32)
    return fail(Code::invalid_value, frontier);
  if (lhs.columns() != rhs.rows()) return fail(Code::shape_mismatch, frontier);
  std::shared_ptr<ValueStorage> storage;
  auto code = allocate(lhs.rows(), rhs.columns(), storage);
  if (code != Code::ok) return fail(code, frontier);
  ScopedFp environment;
  if (!environment.valid()) return fail(Code::unsupported_fp_environment, frontier);
#if defined(MDSLC_CLOSED_HOST_TESTING)
  if (test_candidate_ != nullptr) {
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
    // Strict evaluation is also a legal member of the reassociation-permitted
    // profile. This is a new scalar reference candidate, not old runtime GEMM.
    strictGemm(*lhs.storage_, *rhs.storage_, *storage);
  }
  if (!environment.controlsUnchanged()) code = Code::candidate_failure;
  environment.restore();
  // Test-hook reentry may already have failed this Session. Never issue a value
  // or clear that error merely because the outer candidate returned success.
  if (!status_) return status_;
  if (code != Code::ok) return fail(Code::candidate_failure, frontier);
  result.storage_ = std::move(storage);
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
  }
  return "unknown closed-host status";
}

} // namespace matcore::mdslc::runtime::closed_host_v1
