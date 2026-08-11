#include "MatcoreCpuRuntimeLowering.h"
#include "MatcoreRecoveredGemmBridge.h"
#include "MatcoreV1Bridge.h"
#include "frontend.h"
#include "matcore_ir_v1.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace bridge = matcore::mdslc::mlir_bridge;
namespace frontend = matcore::mdslc::frontend;
namespace lowering = matcore::mdslc::mlir_lowering;
namespace v1 = matcore::mdslc::ir::v1;

int checks = 0;
int failures = 0;

void check(bool condition, std::string_view message) {
  ++checks;
  if (!condition) {
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
  }
}

void checkContains(std::string_view value, std::string_view expected,
                   std::string_view message) {
  check(value.find(expected) != std::string_view::npos, message);
}

std::string readFile(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary);
  std::ostringstream contents;
  contents << input.rdbuf();
  check(input.good() || input.eof(), "test fixture must be readable");
  return contents.str();
}

frontend::Options optionsFor(const std::filesystem::path &source,
                             bool relaxed) {
  frontend::Options options;
  options.input_path = source.string();
  options.clang_path = MDSLC_RECOVERED_TEST_CLANG;
  options.clang_resource_directory = MDSLC_RECOVERED_TEST_RESOURCE_DIR;
  options.trusted_public_headers = {MDSLC_RECOVERED_TEST_PUBLIC_HEADER};
  options.inspect_recovered_cpp_gemm = true;
  options.compiler_arguments = {"-std=c++20", "-O2"};
  if (relaxed) {
    const std::vector<std::string> relaxed_flags = {
        "-ffp-contract=fast",  "-fassociative-math",
        "-fno-signed-zeros",  "-fno-trapping-math",
        "-fhonor-nans",       "-fhonor-infinities",
        "-fno-reciprocal-math", "-fno-approx-func",
        "-fno-rounding-math",
    };
    options.compiler_arguments.insert(options.compiler_arguments.end(),
                                      relaxed_flags.begin(),
                                      relaxed_flags.end());
  }
  options.compiler_arguments.push_back(source.string());
  return options;
}

bool extract(const frontend::Options &options, frontend::Result &result) {
  std::unique_ptr<frontend::Frontend> native =
      frontend::createClangLibToolingFrontend();
  const bool succeeded = native->extract(options, result);
  if (!succeeded) {
    for (const frontend::Diagnostic &diagnostic : result.diagnostics) {
      std::cerr << diagnostic.file << ':' << diagnostic.line << ':'
                << diagnostic.column << ": " << diagnostic.message << '\n';
    }
  }
  return succeeded;
}

v1::Module explicitCapture() {
  const std::string json = readFile(
      std::filesystem::path(MDSLC_RECOVERED_TEST_IR_DIR) /
      "gemm_capture.v1.golden.json");
  v1::Module module;
  std::string error;
  check(v1::parseAndVerifyJson(json, module, error),
        "explicit dynamic Matcore IR v1 fixture must verify");
  return module;
}

template <typename Mutation>
void expectAuthenticationRejected(const frontend::Result &authenticated,
                                  const frontend::Options &options,
                                  Mutation mutate,
                                  std::string_view expected,
                                  std::string_view message) {
  frontend::Result damaged = authenticated;
  frontend::Options damaged_options = options;
  mutate(damaged, damaged_options);
  mlir::MLIRContext context;
  bridge::RecoveredGemmBridgeResultV1 result =
      bridge::bridgeRecoveredGemmToMatcoreMlirV1(
          damaged, damaged_options, 0, context);
  check(!result, message);
  checkContains(result.error, expected,
                "authentication rejection must explain the failed boundary");
}

} // namespace

int main() {
  const std::filesystem::path recovered_source =
      std::filesystem::path(MDSLC_RECOVERED_TEST_FRONTEND_DIR) / "canonical.mdsl";
  const std::string source_before = readFile(recovered_source);
  frontend::Options relaxed_options = optionsFor(recovered_source, true);
  frontend::Result recovered_frontend;
  check(extract(relaxed_options, recovered_frontend),
        "real native frontend must inspect the canonical relaxed GEMM loop");
  check(readFile(recovered_source) == source_before,
        "native recovery inspection must not rewrite ordinary C++");
  check(recovered_frontend.module.operations.empty(),
        "native recovery inspection must not forge a capture DTO operation");
  check(recovered_frontend.recovered_gemm_candidates.size() == 1,
        "canonical loop must produce exactly one recovered candidate");

  mlir::MLIRContext recovered_context;
  bridge::RecoveredGemmBridgeResultV1 recovered =
      bridge::bridgeRecoveredGemmToMatcoreMlirV1(
          recovered_frontend, relaxed_options, 0, recovered_context);
  check(static_cast<bool>(recovered),
        "authenticated relaxed loop must build recovered Matcore MLIR");
  std::string error;
  if (recovered) {
    check(bridge::verifyRecoveredGemmAnalysisModuleV1(*recovered.module, error),
          "recovered module must pass its closed analysis-only envelope");
    const std::string text = bridge::serializeDeterministicMlir(*recovered.module);
    checkContains(text, "kind = \"recovered_cpp_loop\"",
                  "recovered module must retain recovered source origin");
    checkContains(text, "permission = \"source_proven_guard_required\"",
                  "recovered module must retain guard-required permission");
    checkContains(text, "target = \"generic\"",
                  "recovered module must not invent a CPU target decision");
    checkContains(text, "fallback = \"preserve_original_cpp\"",
                  "recovered module must preserve ordinary C++ fallback");
    checkContains(text, "mdsl.required_runtime_guards",
                  "recovered module must retain its exact guard obligations");
    checkContains(text, "mdsl.source_identity",
                  "recovered module must retain stable source identity");
    check(text.find("canonical_callee") == std::string::npos &&
              text.find("matcore::mdsl::gemm") == std::string::npos,
          "recovered module must not forge trusted explicit-call provenance");

    std::vector<lowering::CpuRuntimeDispatchRecordV1> records(1);
    records.front().site_id = "sentinel";
    check(!lowering::lowerExplicitGemmToCpuRuntimeDispatchV1(
              *recovered.module, records, error),
          "strict explicit CPU lowering must reject recovered analysis IR");
    check(records.empty(),
          "recovered CPU-lowering rejection must clear pending records");
    checkContains(error, "explicit Matcore IR v1 bridge envelope",
                  "CPU rejection must explain the executable boundary");

    mlir::MLIRContext second_context;
    auto second = bridge::bridgeRecoveredGemmToMatcoreMlirV1(
        recovered_frontend, relaxed_options, 0, second_context);
    check(static_cast<bool>(second),
          "a repeated authenticated recovery bridge must succeed");
    if (second) {
      check(bridge::serializeDeterministicMlir(*second.module) == text,
            "recovered analysis MLIR must be byte deterministic");
    }
  }

  mlir::MLIRContext explicit_context;
  const v1::Module capture = explicitCapture();
  auto explicit_module = bridge::bridgeV1ToMatcoreMlir(
      capture, explicit_context, bridge::explicitGemmF32V1BridgeContext());
  check(static_cast<bool>(explicit_module),
        "explicit dynamic GEMM fixture must bridge to Matcore MLIR");
  if (recovered && explicit_module) {
    bridge::MathematicalGemmFingerprintV1 explicit_fingerprint;
    bridge::MathematicalGemmFingerprintV1 recovered_fingerprint;
    check(bridge::fingerprintMathematicalGemmV1(
              *explicit_module.module, explicit_fingerprint, error),
          "explicit GEMM mathematical fingerprint must succeed");
    check(bridge::fingerprintMathematicalGemmV1(
              *recovered.module, recovered_fingerprint, error),
          "recovered GEMM mathematical fingerprint must succeed");
    check(!explicit_fingerprint.canonical_contract.empty() &&
              explicit_fingerprint.sha256.starts_with("sha256:") &&
              explicit_fingerprint.sha256.size() == 71,
          "mathematical fingerprint must be inspectable and SHA-256 identified");
    bool equivalent = false;
    check(bridge::equivalentMathematicalGemmV1(
              *explicit_module.module, *recovered.module, equivalent, error),
          "explicit/recovered mathematical comparison must succeed");
    check(equivalent &&
              explicit_fingerprint.canonical_contract ==
                  recovered_fingerprint.canonical_contract &&
              explicit_fingerprint.sha256 == recovered_fingerprint.sha256,
          "explicit and relaxed recovered GEMM must share one normalized WHAT");

    v1::Module static_capture = capture;
    auto &operation = static_capture.operations.front();
    operation.operands[0].type.shape = {v1::ScalarExpr::staticValue(2),
                                        v1::ScalarExpr::staticValue(3)};
    operation.operands[0].type.strides = {v1::ScalarExpr::staticValue(3),
                                          v1::ScalarExpr::staticValue(1)};
    operation.operands[1].type.shape = {v1::ScalarExpr::staticValue(3),
                                        v1::ScalarExpr::staticValue(4)};
    operation.operands[1].type.strides = {v1::ScalarExpr::staticValue(4),
                                          v1::ScalarExpr::staticValue(1)};
    operation.output.type.shape = {v1::ScalarExpr::staticValue(2),
                                   v1::ScalarExpr::staticValue(4)};
    operation.output.type.strides = {v1::ScalarExpr::staticValue(4),
                                     v1::ScalarExpr::staticValue(1)};
    mlir::MLIRContext static_context;
    auto static_module = bridge::bridgeV1ToMatcoreMlir(
        static_capture, static_context,
        bridge::explicitGemmF32V1BridgeContext());
    check(static_cast<bool>(static_module),
          "valid static explicit GEMM must bridge for inequality testing");
    if (static_module) {
      equivalent = true;
      check(bridge::equivalentMathematicalGemmV1(
                *static_module.module, *recovered.module, equivalent, error),
            "static/dynamic mathematical comparison must be well formed");
      check(!equivalent,
            "fingerprint must preserve static versus dynamic shape semantics");
    }
  }

  expectAuthenticationRejected(
      recovered_frontend, relaxed_options,
      [](frontend::Result &, frontend::Options &options) {
        options.compiler_arguments.insert(options.compiler_arguments.end() - 1,
                                          "-DIDENTITY_DRIFT=1");
      },
      "compilation identity",
      "changed compilation options must invalidate recovered authorization");
  expectAuthenticationRejected(
      recovered_frontend, relaxed_options,
      [](frontend::Result &result, frontend::Options &) {
        result.source_snapshot.push_back(' ');
      },
      "source digest",
      "changed source bytes must invalidate recovered authorization");
  expectAuthenticationRejected(
      recovered_frontend, relaxed_options,
      [](frontend::Result &result, frontend::Options &) {
        char &last =
            result.recovered_gemm_candidates[0].source_snapshot_sha256.back();
        last = last == '0' ? '1' : '0';
      },
      "source digest",
      "changed source digest must invalidate recovered authorization");
  expectAuthenticationRejected(
      recovered_frontend, relaxed_options,
      [](frontend::Result &result, frontend::Options &) {
        result.recovered_gemm_candidates[0].site_id =
            "mc_00000000000000000000000000000000";
      },
      "site ID",
      "forged site identity must invalidate recovered authorization");
  expectAuthenticationRejected(
      recovered_frontend, relaxed_options,
      [](frontend::Result &result, frontend::Options &) {
        ++result.recovered_gemm_candidates[0].line;
      },
      "line/column",
      "changed source line must invalidate recovered authorization");
  expectAuthenticationRejected(
      recovered_frontend, relaxed_options,
      [](frontend::Result &result, frontend::Options &) {
        ++result.recovered_gemm_candidates[0].outer_loop_range.begin;
      },
      "outer-loop range",
      "changed recovered source range must invalidate authorization");
  expectAuthenticationRejected(
      recovered_frontend, relaxed_options,
      [](frontend::Result &result, frontend::Options &) {
        std::swap(result.recovered_gemm_candidates[0].proof_ranges[0],
                  result.recovered_gemm_candidates[0].proof_ranges[1]);
      },
      "proof ranges",
      "reordered proof roles must invalidate recovered authorization");
  expectAuthenticationRejected(
      recovered_frontend, relaxed_options,
      [](frontend::Result &result, frontend::Options &) {
        std::swap(
            result.recovered_gemm_candidates[0].required_runtime_guards[0],
            result.recovered_gemm_candidates[0].required_runtime_guards[1]);
      },
      "runtime guards",
      "reordered runtime guards must invalidate recovered authorization");
  expectAuthenticationRejected(
      recovered_frontend, relaxed_options,
      [](frontend::Result &result, frontend::Options &) {
        result.recovered_gemm_candidates[0].fp_proof.preserve_signed_zero = true;
      },
      "floating-point proof",
      "changed effective FP proof must invalidate recovered authorization");
  expectAuthenticationRejected(
      recovered_frontend, relaxed_options,
      [](frontend::Result &result, frontend::Options &) {
        result.recovered_gemm_candidates[0].semantic_contract =
            "forged_contract";
      },
      "semantic contract",
      "changed semantic contract must invalidate recovered authorization");
  expectAuthenticationRejected(
      recovered_frontend, relaxed_options,
      [](frontend::Result &result, frontend::Options &) {
        result.recovered_gemm_candidates[0].rejection_reasons = {
            "injected_rejection"};
      },
      "zero-rejection",
      "a rejection reason must prevent recovered authorization");

  frontend::Options strict_options = optionsFor(recovered_source, false);
  frontend::Result strict_frontend;
  check(extract(strict_options, strict_frontend),
        "strict ordinary GEMM must still compile through the native frontend");
  check(strict_frontend.recovered_gemm_candidates.size() == 1 &&
            strict_frontend.recovered_gemm_candidates[0].state ==
                frontend::RecoveredGemmState::recognized_rejected,
        "strict ordinary GEMM must be recognized but rewrite-rejected");
  mlir::MLIRContext strict_context;
  auto strict = bridge::bridgeRecoveredGemmToMatcoreMlirV1(
      strict_frontend, strict_options, 0, strict_context);
  check(!strict,
        "strict/default C++ semantics must not produce authorized recovered IR");
  checkContains(strict.error, "recognized_guard_required",
                "strict rejection must explain the permission boundary");

  const std::filesystem::path non_gemm_source =
      std::filesystem::path(MDSLC_RECOVERED_TEST_FRONTEND_DIR) /
      "not_output_accumulate.mdsl";
  frontend::Options non_gemm_options = optionsFor(non_gemm_source, true);
  frontend::Result non_gemm_frontend;
  check(extract(non_gemm_options, non_gemm_frontend),
        "non-GEMM ordinary C++ fixture must remain compilable");
  check(non_gemm_frontend.recovered_gemm_candidates.size() == 1 &&
            non_gemm_frontend.recovered_gemm_candidates[0].state ==
                frontend::RecoveredGemmState::not_recognized,
        "non-GEMM loop must remain not recognized");
  mlir::MLIRContext non_gemm_context;
  auto non_gemm = bridge::bridgeRecoveredGemmToMatcoreMlirV1(
      non_gemm_frontend, non_gemm_options, 0, non_gemm_context);
  check(!non_gemm,
        "not-recognized ordinary C++ must not produce semantic GEMM IR");

  check(readFile(recovered_source) == source_before,
        "all recovered bridge analysis must leave the source unchanged");
  if (failures != 0) {
    std::cerr << failures << " failure(s) across " << checks << " checks\n";
    return 1;
  }
  std::cout << "Matcore recovered GEMM bridge: " << checks << '/' << checks
            << " checks passed\n";
  return 0;
}
