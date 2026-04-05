#include "matcore/diagnostics.h"

#include <algorithm>
#include <cctype>
#include <initializer_list>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/OperationSupport.h"

namespace matcore {
namespace {

constexpr std::size_t kMaxIrBytes = 50 * 1024;

std::string toLower(std::string value) {
  for (char &ch : value) {
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  }
  return value;
}

bool containsAll(const std::string &haystack,
                 std::initializer_list<std::string_view> needles) {
  for (std::string_view needle : needles) {
    if (haystack.find(needle) == std::string::npos) {
      return false;
    }
  }
  return true;
}

std::string truncateIr(const std::string &ir) {
  if (ir.size() <= kMaxIrBytes) {
    return ir;
  }
  std::string out = ir.substr(0, kMaxIrBytes);
  out += "\n... [TRUNCATED at " + std::to_string(kMaxIrBytes) + " bytes]\n";
  return out;
}

std::string moduleIrString(mlir::ModuleOp module) {
  if (!module) {
    return {};
  }
  std::string ir;
  llvm::raw_string_ostream ir_stream(ir);
  mlir::OpPrintingFlags flags;
  flags.printGenericOpForm().elideLargeElementsAttrs();
  module.print(ir_stream, flags);
  ir_stream.flush();
  return ir;
}

std::string firstLines(const std::string &text, int max_lines) {
  std::istringstream stream(text);
  std::ostringstream out;
  std::string line;
  int count = 0;
  while (count < max_lines && std::getline(stream, line)) {
    out << line << '\n';
    ++count;
  }
  if (stream.good()) {
    out << "... [TRUNCATED at " << max_lines << " lines]\n";
  }
  return out.str();
}

}  // namespace

std::vector<std::string> suggestRemediation(const std::string &stage_name,
                                            const std::string &diagnostic_text) {
  const std::string lowered_stage = toLower(stage_name);
  const std::string lowered_diag = toLower(diagnostic_text);
  const std::string lowered_all = lowered_stage + "\n" + lowered_diag;
  std::vector<std::string> suggestions;

  if (containsAll(lowered_all, {"convert-vector-to-llvm", "tensor"})) {
    suggestions.push_back(
        "Check bufferization pass ordering — unbufferized tensor op survived");
  }
  if (containsAll(lowered_all, {"gpu.launch", "static", "0"})) {
    suggestions.push_back(
        "Check tile size vs matrix dimension — empty grid dimension after "
        "tiling");
  }
  if (containsAll(lowered_all, {"nvgpu.mma.sync", "type"})) {
    suggestions.push_back(
        "Verify float16 inputs for tensor core path — wrong fragment type");
  }
  if (containsAll(lowered_all, {"convert-gpu-to-nvvm", "symbol"})) {
    suggestions.push_back(
        "Check registerGpuRuntimeSymbols() call — missing GPU runtime symbols");
  }
  if (containsAll(lowered_all, {"cumoduleload", "301"})) {
    suggestions.push_back(
        "SM version mismatch — check target profile matches GPU hardware");
  }
  if (containsAll(lowered_all, {"culaunchkernel"})) {
    suggestions.push_back(
        "Verify macro-topology grid/block dimensions calculation");
  }
  if (containsAll(lowered_all, {"padding", "out of bound"})) {
    suggestions.push_back(
        "Verify ceil_div logic in DynamicMatmulPaddingPass");
  }

  return suggestions;
}

StructuredDiagnosticReport buildDiagnosticReport(
    const std::string &route_name, const std::string &stage_name, int stage_index,
    const std::string &raw_diagnostics, mlir::ModuleOp module,
    const std::string &target_profile, const std::string &dtype_signature,
    const std::vector<CapturedDiagnostic> &captured_diagnostics,
    const std::string &ir_before) {
  StructuredDiagnosticReport report;
  report.route_name = route_name;
  report.failing_stage = stage_name;
  report.stage_index = stage_index;

  report.ir_at_failure = truncateIr(moduleIrString(module));
  const std::string pre_failure_ir =
      ir_before.empty() ? report.ir_at_failure : truncateIr(ir_before);

  if (!captured_diagnostics.empty()) {
    for (const CapturedDiagnostic &captured : captured_diagnostics) {
      if (captured.message.empty()) {
        continue;
      }
      const std::string &pass_name =
          captured.pass_name.empty() ? stage_name : captured.pass_name;
      const std::string severity =
          captured.severity.empty() ? "error" : captured.severity;
      PassDiagnostic diag;
      diag.pass_name = pass_name;
      diag.stage_index = stage_index;
      diag.severity = severity;
      diag.message = captured.message;
      diag.ir_before = pre_failure_ir;
      diag.target_profile = target_profile;
      diag.dtype_signature = dtype_signature;
      diag.suggestions = suggestRemediation(pass_name, captured.message);
      report.diagnostics.push_back(std::move(diag));
    }
  }

  if (report.diagnostics.empty() && !raw_diagnostics.empty()) {
    PassDiagnostic diag;
    diag.pass_name = stage_name;
    diag.stage_index = stage_index;
    diag.severity = "error";
    diag.message = raw_diagnostics;
    diag.ir_before = pre_failure_ir;
    diag.target_profile = target_profile;
    diag.dtype_signature = dtype_signature;
    diag.suggestions = suggestRemediation(stage_name, raw_diagnostics);
    report.diagnostics.push_back(std::move(diag));
  }

  report.formatted_message = formatDiagnosticReport(report);
  return report;
}

std::string formatDiagnosticReport(const StructuredDiagnosticReport &report) {
  std::ostringstream out;
  out << "=== MatCore Pass Failure Diagnostic ===\n";
  out << "Route: " << report.route_name << '\n';
  out << "Failed at stage: " << report.failing_stage;
  if (report.stage_index > 0) {
    out << " (stage " << report.stage_index << ")";
  }
  out << '\n';
  if (!report.diagnostics.empty()) {
    out << "Diagnostic: " << report.diagnostics.front().message << '\n';
  } else {
    out << "Diagnostic: <none>\n";
  }

  std::unordered_set<std::string> seen;
  std::vector<std::string> all_suggestions;
  for (const PassDiagnostic &diag : report.diagnostics) {
    for (const std::string &suggestion : diag.suggestions) {
      if (seen.insert(suggestion).second) {
        all_suggestions.push_back(suggestion);
      }
    }
  }

  if (!all_suggestions.empty()) {
    out << "Suggestions:\n";
    for (const std::string &suggestion : all_suggestions) {
      out << "  - " << suggestion << '\n';
    }
  }
  out << "IR at failure:\n" << firstLines(report.ir_at_failure, 200);
  out << "===\n";
  return out.str();
}

std::string formatDiagnosticReportJson(const StructuredDiagnosticReport &report) {
  llvm::json::Array diagnostics_json;
  for (const PassDiagnostic &diag : report.diagnostics) {
    llvm::json::Array suggestions_json;
    for (const std::string &suggestion : diag.suggestions) {
      suggestions_json.push_back(suggestion);
    }
    diagnostics_json.push_back(llvm::json::Object{
        {"pass_name", diag.pass_name},
        {"stage_index", diag.stage_index},
        {"severity", diag.severity},
        {"message", diag.message},
        {"ir_before", diag.ir_before},
        {"target_profile", diag.target_profile},
        {"dtype_signature", diag.dtype_signature},
        {"suggestions", std::move(suggestions_json)},
    });
  }

  llvm::json::Object report_json{
      {"route_name", report.route_name},
      {"failing_stage", report.failing_stage},
      {"stage_index", report.stage_index},
      {"diagnostics", std::move(diagnostics_json)},
      {"ir_at_failure", report.ir_at_failure},
      {"formatted_message", report.formatted_message},
  };
  return llvm::formatv("{0:2}", llvm::json::Value(std::move(report_json))).str();
}

}  // namespace matcore
