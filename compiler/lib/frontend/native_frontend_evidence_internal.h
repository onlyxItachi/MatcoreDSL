#ifndef MATCORE_MDSLC_NATIVE_FRONTEND_EVIDENCE_INTERNAL_H
#define MATCORE_MDSLC_NATIVE_FRONTEND_EVIDENCE_INTERNAL_H

#include "frontend.h"

namespace matcore::mdslc::frontend::detail {

// Non-installed issuer seam. Only the native LibTooling implementation calls
// this after parsing/Sema, source-stability checks, and IR verification have
// all succeeded.
class NativeFrontendEvidenceIssuerV1 {
public:
  static AuthenticatedNativeFrontendEvidenceV1 issue(
      const Result &result, const Options &effective_options);
};

} // namespace matcore::mdslc::frontend::detail

#endif // MATCORE_MDSLC_NATIVE_FRONTEND_EVIDENCE_INTERNAL_H
