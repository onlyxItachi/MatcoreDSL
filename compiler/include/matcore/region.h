#ifndef MATCORE_EXPERIMENTAL_REGION_H
#define MATCORE_EXPERIMENTAL_REGION_H

// Experimental C++-hosted mathematical regions. Source and ABI may evolve.
// An MDSLC-compiled function returns a checked result; ordinary C++ callers
// need no compiler internals. The intrinsics below have no runtime fallback.
#include <matcore/detail/region_storage.h>

#ifdef MATCORE_REGION
#error "MATCORE_REGION must be defined only by the canonical Matcore header"
#endif
#if defined(__clang__)
#define MATCORE_REGION [[clang::annotate("matcore.experimental.region.v1")]]
#else
#define MATCORE_REGION
#endif

namespace matcore::mdsl {
using Shape = unsigned long long;
using Access = mdslc::runtime::closed_host_v1::Access;
using Error = mdslc::runtime::closed_host_v1::Code;
enum class Numerics { strict_f32, reassociate_f32 };

// External resource descriptor, not an allocation or a proof of disjointness.
// The caller owns valid, initialized float objects and lifetime/race freedom.
struct Storage {
  float *data = nullptr;
  Shape rows = 0;
  Shape columns = 0;
  Shape capacity_elements = 0;
  Access access = Access::read_only;
};

// Source-only immutable mathematical value. No buffer ownership, operator
// overloading, expression templates, or executable proxy representation.
struct Value { Value() = delete; };

struct SourceLocation {
  const char *file = nullptr;
  unsigned int line = 0;
  unsigned int column = 0;
};

class Observation {
public:
  Observation() noexcept = default;
  Observation(const Observation &) noexcept;
  Observation &operator=(const Observation &) noexcept;
  Observation(Observation &&) noexcept;
  Observation &operator=(Observation &&) noexcept;
  ~Observation() noexcept;
  bool valid() const noexcept;
  Shape rows() const noexcept;
  Shape columns() const noexcept;
  const float *data() const noexcept;
private:
  friend class Result;
  Observation(mdslc::runtime::closed_host_v1::ObservationBlock *, Shape) noexcept;
  mdslc::runtime::closed_host_v1::ObservationBlock *block_ = nullptr;
  Shape index_ = 0;
};

// Owning observations survive the invocation and later resource mutations.
// Moving a Result and transferring it from the adapter allocate no storage.
// Observation pointers are read-only and valid while an owning handle lives.
class [[nodiscard]] Result {
public:
  Result() = delete;
  Result(const Result &) = delete;
  Result &operator=(const Result &) = delete;
  Result(Result &&other) noexcept;
  Result &operator=(Result &&other) noexcept;
  ~Result() noexcept;
  bool ok() const noexcept { return status_.code == Error::ok && status_.completed; }
  explicit operator bool() const noexcept { return ok(); }
  Error error() const noexcept { return status_.code; }
  SourceLocation failure_location() const noexcept { return failure_; }
  const char *message() const noexcept {
    return mdslc::runtime::closed_host_v1::message(status_.code);
  }
  Shape publication_count() const noexcept { return status_.publications; }
  Shape observation_count() const noexcept;
  Observation observation(Shape index) const noexcept;
  // Diagnostic execution frontiers are compiler-issued, never source inputs.
  Shape failed_frontier() const noexcept { return status_.failed_frontier; }
  Shape completed_frontier() const noexcept { return status_.completed_frontier; }
  Shape completed_effect_frontier() const noexcept {
    return status_.completed_effect_frontier;
  }
private:
  friend mdslc::runtime::closed_host_v1::Session;
  Result(mdslc::runtime::closed_host_v1::Status status,
         mdslc::runtime::closed_host_v1::ObservationBlock *observations,
         SourceLocation failure) noexcept
      : status_(status), observations_(observations), failure_(failure) {}
  mdslc::runtime::closed_host_v1::Status status_;
  mdslc::runtime::closed_host_v1::ObservationBlock *observations_ = nullptr;
  SourceLocation failure_;
};

#if defined(__clang__)
[[clang::annotate("matcore.experimental.read.v1")]]
Value read(Storage, Shape, Shape) noexcept;
[[clang::annotate("matcore.experimental.gemm.v1")]]
Value gemm(Value, Value, Numerics) noexcept;
[[clang::annotate("matcore.experimental.publish.v1")]]
void publish(Value, Storage) noexcept;
[[clang::annotate("matcore.experimental.observe.v1")]]
void observe(Storage) noexcept;
[[clang::annotate("matcore.experimental.rows.v1")]]
Shape rows(Value) noexcept;
[[clang::annotate("matcore.experimental.cols.v1")]]
Shape cols(Value) noexcept;
[[clang::annotate("matcore.experimental.complete.v1")]]
Result complete() noexcept;
#else
Value read(Storage, Shape, Shape) noexcept;
Value gemm(Value, Value, Numerics) noexcept;
void publish(Value, Storage) noexcept;
void observe(Storage) noexcept;
Shape rows(Value) noexcept;
Shape cols(Value) noexcept;
Result complete() noexcept;
#endif
} // namespace matcore::mdsl

#endif
