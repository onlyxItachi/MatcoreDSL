#ifndef MATCORE_MDSLC_CLOSED_REGION_ADMISSION_H
#define MATCORE_MDSLC_CLOSED_REGION_ADMISSION_H

#include "../mlir/MatcoreClosedRegion.h"
#include <memory>
#include <optional>
#include <string>

namespace matcore::mdslc::frontend {

struct ClosedRegionAdmissionResult;

// An in-process inspection seal, not a signature, serialized authority, or
// permission to execute. Only successful admission can construct its payload.
class AuthenticatedClosedRegionEvidence {
public:
  const closed_region::Program &program() const;
  const std::string &sourceSnapshot() const;
  const std::string &regionName() const;

private:
  struct Payload;
  explicit AuthenticatedClosedRegionEvidence(std::shared_ptr<const Payload> p);
  std::shared_ptr<const Payload> payload_;
  friend ClosedRegionAdmissionResult admitClosedRegionSource(
      const std::string &, const std::string &, const std::string &);
};

struct ClosedRegionAdmissionResult {
  // Distinguishes a well-formed C++ rejection from a Clang syntax/type error.
  bool syntax_valid = false;
  std::optional<AuthenticatedClosedRegionEvidence> evidence;
  std::string error;
  explicit operator bool() const { return evidence.has_value(); }
};

// Private inspection instrument: one selected, annotated free-function body
// in an immutable source snapshot. No compiler flags, include paths, source
// overlays, physical fixture headers, host callbacks, or execution fallback.
ClosedRegionAdmissionResult admitClosedRegionSource(
    const std::string &source, const std::string &source_identity,
    const std::string &region_name = "region");

// Re-admits sealed bytes using the same fixed compiler contract and compares
// the complete untransformed semantic graph, including source/effect bindings.
bool verifyClosedRegionMatchesEvidence(
    const AuthenticatedClosedRegionEvidence &evidence, mlir::ModuleOp module,
    std::string &error);

} // namespace matcore::mdslc::frontend
#endif
