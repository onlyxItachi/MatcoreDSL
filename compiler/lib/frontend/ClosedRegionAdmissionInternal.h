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
};
struct ClosedRegionASTPolicy {
  clang::FileID owned_header;
  std::vector<ClosedRegionPreprocessorEvent> events;
};

const char *closedRegionOwnedHeaderPath();
const char *closedRegionOwnedHeaderSource();
std::string closedRegionDigest(const std::string &bytes);
clang::FileID authenticateClosedRegionHeader(clang::ASTContext &context,
                                            clang::FileManager &files,
                                            std::string &error);
bool admitClosedRegionParsedAST(clang::ASTContext &context,
                               const clang::Decl *anchor,
                               const std::string &selected,
                               closed_region::Program &program,
                               const ClosedRegionASTPolicy &policy,
                               std::string &error);

class ClosedRegionEvidenceIssuer {
public:
  static ClosedRegionAdmissionResult host(
      closed_region::Program program,
      std::shared_ptr<const closed_region_host::HostInputSnapshot> inputs,
      const std::string &region_name);
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
    std::string &error);

} // namespace matcore::mdslc::frontend::detail
#endif
