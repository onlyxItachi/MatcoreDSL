#pragma once

#include <string>
#include <vector>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"

namespace matcore {

struct PassDiagnostic {
  std::string pass_name;
  int stage_index = 0;
  std::string severity;  // "error", "warning", "note", "remark"
  std::string message;
  std::string ir_before;  // IR state before the failing pass (truncated if large)
  std::string target_profile;
  std::string dtype_signature;
  std::vector<std::string> suggestions;
};

struct CapturedDiagnostic {
  std::string pass_name;
  std::string severity;
  std::string message;
};

struct StructuredDiagnosticReport {
  std::string route_name;
  std::string failing_stage;
  int stage_index = 0;
  std::vector<PassDiagnostic> diagnostics;
  std::string ir_at_failure;
  std::string formatted_message;  // Human-readable summary
};

StructuredDiagnosticReport buildDiagnosticReport(
    const std::string &route_name, const std::string &stage_name, int stage_index,
    const std::string &raw_diagnostics, mlir::ModuleOp module,
    const std::string &target_profile = {}, const std::string &dtype_signature = {},
    const std::vector<CapturedDiagnostic> &captured_diagnostics = {},
    const std::string &ir_before = {});

std::vector<std::string> suggestRemediation(const std::string &stage_name,
                                            const std::string &diagnostic_text);

std::string formatDiagnosticReport(const StructuredDiagnosticReport &report);
std::string formatDiagnosticReportJson(const StructuredDiagnosticReport &report);

}  // namespace matcore
