#ifndef MATCORE_MDSLC_RECOVERED_GEMM_RECOGNIZER_H
#define MATCORE_MDSLC_RECOVERED_GEMM_RECOGNIZER_H

#include "frontend.h"

#include <string>

namespace clang {
class ASTContext;
class ForStmt;
class FunctionDecl;
} // namespace clang

namespace matcore::mdslc::frontend {

struct RecoveredGemmInspectionInputs {
  std::string main_display_path;
  std::string main_canonical_path;
  std::string compilation_identity;
  unsigned optimization_level = 0;
  std::string denormal_mode;
  std::string fp32_denormal_mode;
  bool denormal_mode_is_ieee = false;
  bool fp32_denormal_mode_is_ieee = false;
};

RecoveredGemmCandidate inspectRecoveredGemmLoop(
    const clang::ForStmt &outer_loop, const clang::FunctionDecl &function,
    clang::ASTContext &context, const RecoveredGemmInspectionInputs &inputs);

} // namespace matcore::mdslc::frontend

#endif // MATCORE_MDSLC_RECOVERED_GEMM_RECOGNIZER_H
