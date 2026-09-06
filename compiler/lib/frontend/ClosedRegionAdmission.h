#ifndef MATCORE_MDSLC_CLOSED_REGION_ADMISSION_H
#define MATCORE_MDSLC_CLOSED_REGION_ADMISSION_H

#include "../mlir/MatcoreClosedRegion.h"
#include <memory>
#include <optional>
#include <string>

namespace matcore::mdslc::frontend {

struct Options;
namespace detail {
class ClosedRegionEvidenceIssuer;
class ClosedRegionCompilationAccess;
}
struct ClosedRegionAdmissionResult;

// Compiler installation inputs, not a source-selected declaration override.
// Admission requires both exact expected bytes and the physically resolved
// parsed FileIDs. The driver supplies these paths from its own installation.
struct ExperimentalRegionHeaders {
  std::string region_path;
  std::string storage_path;
};
struct ClosedRegionParameterBinding {
  enum class Kind { Storage, Shape };
  std::string name;
  Kind kind = Kind::Storage;
  bool operator==(const ClosedRegionParameterBinding &) const = default;
};
struct ClosedRegionNamespaceBinding {
  std::string name;
  bool is_inline = false;
  bool operator==(const ClosedRegionNamespaceBinding &) const = default;
};
struct ClosedRegionHelperBinding {
  std::string qualified_name;
  std::string mangled_name;
  closed_region::SourceSite body;
};
// Frontend-specific rewrite witness. It is intentionally absent from semantic
// Program and cannot be supplied by a serialized graph or imported MLIR.
struct ClosedRegionEntryBinding {
  std::string qualified_name;
  std::string mangled_name;
  std::string signature_sha256;
  closed_region::SourceSite body;
  closed_region::SourceSite completion;
  std::vector<ClosedRegionNamespaceBinding> namespaces;
  std::vector<ClosedRegionParameterBinding> parameters;
  std::vector<ClosedRegionHelperBinding> value_helpers;
};

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
  const std::optional<ClosedRegionEntryBinding> &entryBinding() const;

private:
  struct Payload;
  explicit AuthenticatedClosedRegionEvidence(std::shared_ptr<const Payload> p);
  std::shared_ptr<const Payload> payload_;
  friend ClosedRegionAdmissionResult admitClosedRegionSource(
      const std::string &, const std::string &, const std::string &);
  friend class detail::ClosedRegionEvidenceIssuer;
  friend class detail::ClosedRegionCompilationAccess;
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

// Experimental Result-returning mathematical function in a real host TU.
// Private fixture entry points and their inspection behavior remain unchanged.
ClosedRegionAdmissionResult admitExperimentalRegionHost(
    const Options &options, const std::string &working_directory,
    const ExperimentalRegionHeaders &headers, const std::string &region_name);

// Re-admits sealed bytes using the same fixed compiler contract and compares
// the complete untransformed semantic graph, including source/effect bindings.
bool verifyClosedRegionMatchesEvidence(
    const AuthenticatedClosedRegionEvidence &evidence, mlir::ModuleOp module,
    std::string &error);

} // namespace matcore::mdslc::frontend
#endif
