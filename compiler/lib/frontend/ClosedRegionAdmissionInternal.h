#ifndef MATCORE_MDSLC_CLOSED_REGION_ADMISSION_INTERNAL_H
#define MATCORE_MDSLC_CLOSED_REGION_ADMISSION_INTERNAL_H

#include "ClosedRegionAdmission.h"
#include "ClosedRegionHostInputs.h"
#include <clang/Basic/SourceLocation.h>
#include <functional>
#include <vector>

namespace clang { class ASTContext; class Decl; class FileManager; }

namespace matcore::mdslc::frontend::detail {

struct ClosedRegionPreprocessorEvent {
  clang::SourceRange range;
  std::string description;
  clang::SourceLocation definition;
};
struct ClosedRegionASTPolicy {
  clang::FileID owned_header;
  clang::FileID owned_storage_header;
  bool experimental = false;
  std::vector<ClosedRegionPreprocessorEvent> events;
};

const char *closedRegionOwnedHeaderPath();
const char *closedRegionOwnedHeaderSource();
std::string closedRegionDigest(const std::string &bytes);
const char *experimentalRegionHeaderSource();
const char *experimentalRegionStorageHeaderSource();
bool authenticateExperimentalRegionHeaders(
    clang::ASTContext &, clang::FileManager &, const ExperimentalRegionHeaders &,
    ClosedRegionASTPolicy &, std::string &error);
bool sameClosedRegionEntryBinding(const ClosedRegionEntryBinding &,
                                 const ClosedRegionEntryBinding &);
clang::FileID authenticateClosedRegionHeader(clang::ASTContext &context,
                                            clang::FileManager &files,
                                            std::string &error);
bool admitClosedRegionParsedAST(clang::ASTContext &context,
                               const clang::Decl *anchor,
                               const std::string &selected,
                               closed_region::Program &program,
                               const ClosedRegionASTPolicy &policy,
                               std::string &error,
                               ClosedRegionEntryBinding *entry = nullptr);

class ClosedRegionEvidenceIssuer {
public:
  static ClosedRegionAdmissionResult host(
      closed_region::Program program,
      std::shared_ptr<const closed_region_host::HostInputSnapshot> inputs,
      const std::string &region_name,
      std::optional<ClosedRegionEntryBinding> entry = {},
      std::optional<ExperimentalRegionHeaders> headers = {});
};

// Compiler-only read access. The snapshot has no serialized constructor and
// replay cannot substitute mutable files. It grants no authority to rewritten
// bytes; code generation must separately derive them from the sealed binding.
class ClosedRegionCompilationAccess {
public:
  static std::shared_ptr<const closed_region_host::HostInputSnapshot> host(
      const AuthenticatedClosedRegionEvidence &);
};

// Deterministic race falsification seam, noninstalled and test-only. This hook
// is supplied by the compiler test harness, never by admitted source code.
ClosedRegionAdmissionResult admitClosedRegionHostForTesting(
    const Options &options, const std::string &working_directory,
    const std::string &region_name,
    const std::function<void()> &after_initial_parse);

bool replayClosedRegionHost(
    const closed_region_host::HostInputSnapshot &inputs,
    const std::string &region_name, closed_region::Program &program,
    std::string &error,
    const ExperimentalRegionHeaders *headers = nullptr,
    ClosedRegionEntryBinding *entry = nullptr);

ClosedRegionAdmissionResult admitExperimentalRegionHostForTesting(
    const Options &, const std::string &, const ExperimentalRegionHeaders &,
    const std::string &, const std::function<void()> &after_initial_parse);

} // namespace matcore::mdslc::frontend::detail
#endif
