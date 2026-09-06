#ifndef MATCORE_MDSLC_CLOSED_REGION_ADMISSION_H
#define MATCORE_MDSLC_CLOSED_REGION_ADMISSION_H

#include "../mlir/MatcoreClosedRegion.h"
#include <memory>
#include <optional>
#include <string>

namespace matcore::mdslc::frontend {

struct Options;
namespace detail { class ClosedRegionEvidenceIssuer; }
struct ClosedRegionAdmissionResult;

// An in-process inspection seal, not a signature, serialized authority, or
// permission to execute. Only successful admission can construct its payload.
class AuthenticatedClosedRegionEvidence {
public:
  // A moved-from seal has no authority: queries/pairing reject it and payload
  // accessors throw logic_error rather than dereference an empty capability.
  const closed_region::Program &program() const;
  const std::string &sourceSnapshot() const;
  const std::string &regionName() const;
  bool hasHostContext() const;
  const std::string &hostContextIdentity() const;

private:
  struct Payload;
  explicit AuthenticatedClosedRegionEvidence(std::shared_ptr<const Payload> p);
  std::shared_ptr<const Payload> payload_;
  friend ClosedRegionAdmissionResult admitClosedRegionSource(
      const std::string &, const std::string &, const std::string &);
  friend class detail::ClosedRegionEvidenceIssuer;
  friend bool verifyClosedRegionMatchesEvidence(
      const AuthenticatedClosedRegionEvidence &, mlir::ModuleOp, std::string &);
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

// Same grammar in a bounded physical host TU. Options are inputs, not trusted
// authority: the fixed compiler tuple, flags, environment, dependencies and
// source preprocessing are authenticated before an immutable seal is issued.
ClosedRegionAdmissionResult admitClosedRegionHost(
    const Options &options, const std::string &working_directory,
    const std::string &region_name = "region");

// Re-admits sealed bytes using the same fixed compiler contract and compares
// the complete untransformed semantic graph, including source/effect bindings.
bool verifyClosedRegionMatchesEvidence(
    const AuthenticatedClosedRegionEvidence &evidence, mlir::ModuleOp module,
    std::string &error);

} // namespace matcore::mdslc::frontend
#endif
