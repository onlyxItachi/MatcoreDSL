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
  const std::filesystem::path include_root =
      std::filesystem::path(MDSLC_RECOVERED_TEST_PUBLIC_HEADER)
          .parent_path()
          .parent_path();
  options.compiler_arguments = {"-std=c++20", "-O2",
                                "-I" + include_root.string()};
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
void expectMutableResultCannotForge(
    const frontend::Result &published_result,
    std::string_view authenticated_mlir, Mutation mutate,
    std::string_view message) {
  frontend::Result damaged = published_result;
  mutate(damaged);
  check(damaged.native_evidence.has_value() &&
            damaged.native_evidence->valid(),
        "copied sealed evidence must remain valid after public Result mutation");
  if (!damaged.native_evidence)
    return;
  mlir::MLIRContext context;
  bridge::RecoveredGemmBridgeResultV1 result =
      bridge::bridgeRecoveredGemmToMatcoreMlirV1(
          *damaged.native_evidence, 0, context);
  check(static_cast<bool>(result), message);
  if (result) {
    check(bridge::serializeDeterministicMlir(*result.module) ==
              authenticated_mlir,
          "mutable Result drift must not alter sealed recovered MLIR");
  }
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
  check(recovered_frontend.native_evidence.has_value() &&
            recovered_frontend.native_evidence->valid(),
        "successful native recovery inspection must issue sealed evidence");

  mlir::MLIRContext recovered_context;
  bridge::RecoveredGemmBridgeResultV1 recovered;
  if (recovered_frontend.native_evidence) {
    recovered = bridge::bridgeRecoveredGemmToMatcoreMlirV1(
        *recovered_frontend.native_evidence, 0, recovered_context);
  }
  check(static_cast<bool>(recovered),
        "authenticated relaxed loop must build recovered Matcore MLIR");
  std::string error;
  std::string recovered_text;
  if (recovered) {
    check(bridge::verifyRecoveredGemmAnalysisModuleV1(*recovered.module, error),
          "recovered module must pass its closed analysis-only envelope");
    recovered_text = bridge::serializeDeterministicMlir(*recovered.module);
    checkContains(recovered_text, "kind = \"recovered_cpp_loop\"",
                  "recovered module must retain recovered source origin");
    checkContains(recovered_text,
                  "permission = \"source_proven_guard_required\"",
                  "recovered module must retain guard-required permission");
    checkContains(recovered_text, "target = \"generic\"",
                  "recovered module must not invent a CPU target decision");
    checkContains(recovered_text, "fallback = \"preserve_original_cpp\"",
                  "recovered module must preserve ordinary C++ fallback");
    checkContains(recovered_text, "mdsl.required_runtime_guards",
                  "recovered module must retain its exact guard obligations");
    checkContains(recovered_text, "mdsl.source_identity",
                  "recovered module must retain stable source identity");
    check(recovered_text.find("canonical_callee") == std::string::npos &&
              recovered_text.find("matcore::mdsl::gemm") ==
                  std::string::npos,
          "recovered module must not forge trusted explicit-call provenance");

    std::vector<lowering::CpuRuntimeDispatchRecordV1> records(1);
    records.front().site_id = "sentinel";
    check(!lowering::lowerExplicitGemmToCpuRuntimeDispatchV1(
              *recovered.module, records, error),
          "strict explicit CPU lowering must reject recovered analysis IR");
    check(records.empty(),
          "recovered CPU-lowering rejection must clear pending records");
    checkContains(error, "analysis_only",
                  "CPU rejection must explain the executable boundary");

    mlir::MLIRContext second_context;
    auto second = bridge::bridgeRecoveredGemmToMatcoreMlirV1(
        *recovered_frontend.native_evidence, 0, second_context);
    check(static_cast<bool>(second),
          "a repeated authenticated recovery bridge must succeed");
    if (second) {
      check(bridge::serializeDeterministicMlir(*second.module) ==
                recovered_text,
            "recovered analysis MLIR must be byte deterministic");
    }
  }

  const std::filesystem::path explicit_source =
      std::filesystem::path(MDSLC_RECOVERED_TEST_FRONTEND_DIR).parent_path() /
      "gemm_capture.mdsl";
  const std::string explicit_source_before = readFile(explicit_source);
  frontend::Options explicit_options = optionsFor(explicit_source, false);
  frontend::Result explicit_frontend;
  check(extract(explicit_options, explicit_frontend),
        "real native frontend must capture the trusted explicit GEMM call");
  check(readFile(explicit_source) == explicit_source_before,
        "native explicit extraction must not rewrite source");
  check(explicit_frontend.module.operations.size() == 1,
        "native explicit extraction must produce one authenticated v0 site");
  check(explicit_frontend.native_evidence.has_value() &&
            explicit_frontend.native_evidence->valid(),
        "native explicit extraction must issue sealed evidence in inspection mode");
  if (explicit_frontend.native_evidence &&
      recovered_frontend.native_evidence) {
    const auto authenticated =
        bridge::compareAuthenticatedExplicitAndRecoveredGemmV1(
            *explicit_frontend.native_evidence, 0,
            *recovered_frontend.native_evidence, 0);
    check(static_cast<bool>(authenticated),
          "live native explicit/recovered comparison must authenticate");
    check(authenticated.equivalent &&
              !authenticated.explicit_fingerprint.canonical_contract.empty() &&
              authenticated.explicit_fingerprint.sha256.starts_with(
                  "sha256:") &&
              authenticated.explicit_fingerprint.sha256.size() == 71 &&
              authenticated.explicit_fingerprint.canonical_contract ==
                  authenticated.recovered_fingerprint.canonical_contract &&
              authenticated.explicit_fingerprint.sha256 ==
                  authenticated.recovered_fingerprint.sha256,
          "live explicit v0/v1/MLIR and recovered loop must share normalized WHAT");
  }

  // Parsed/golden data may use the explicitly structural diagnostic API, but
  // it never enters the authenticated equivalence API above.
  mlir::MLIRContext explicit_context;
  const v1::Module capture = explicitCapture();
  auto explicit_module = bridge::bridgeV1ToMatcoreMlir(
      capture, explicit_context, bridge::explicitGemmF32V1BridgeContext());
  check(static_cast<bool>(explicit_module),
        "untrusted explicit fixture must bridge for structural diagnostics");
  if (recovered && explicit_module) {
    bridge::MathematicalGemmFingerprintV1 explicit_fingerprint;
    bridge::MathematicalGemmFingerprintV1 recovered_fingerprint;
    check(bridge::fingerprintStructuralMathematicalGemmV1(
              *explicit_module.module, explicit_fingerprint, error),
          "explicit fixture structural fingerprint must succeed");
    check(bridge::fingerprintStructuralMathematicalGemmV1(
              *recovered.module, recovered_fingerprint, error),
          "recovered structural fingerprint must succeed");
    check(!explicit_fingerprint.canonical_contract.empty() &&
              explicit_fingerprint.sha256.starts_with("sha256:") &&
              explicit_fingerprint.sha256.size() == 71,
          "structural fingerprint must be inspectable and SHA-256 identified");
    bool equivalent = false;
    check(bridge::equivalentStructuralMathematicalGemmV1(
              *explicit_module.module, *recovered.module, equivalent, error),
          "structural explicit/recovered comparison must succeed");
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
      check(bridge::equivalentStructuralMathematicalGemmV1(
                *static_module.module, *recovered.module, equivalent, error),
            "static/dynamic mathematical comparison must be well formed");
      check(!equivalent,
            "fingerprint must preserve static versus dynamic shape semantics");
    }

    std::vector<lowering::CpuRuntimeDispatchRecordV1> records(1);
    records.front().site_id = "sentinel";
    (*explicit_module.module)
        ->setAttr("mdsl.analysis_only",
                  mlir::BoolAttr::get(&explicit_context, true));
    check(!lowering::lowerExplicitGemmToCpuRuntimeDispatchV1(
              *explicit_module.module, records, error),
          "analysis-only taint must reject an otherwise explicit envelope");
    check(records.empty(),
          "analysis-only rejection must clear pending runtime records");
    checkContains(error, "analysis_only",
                  "analysis-only rejection must explain the taint boundary");
    (*explicit_module.module)->removeAttr("mdsl.analysis_only");

    records.push_back({});
    (*explicit_module.module)
        ->setAttr("mdsl.producer",
                  mlir::StringAttr::get(&explicit_context,
                                        "clang-ast-json-bootstrap-v0"));
    check(!lowering::lowerExplicitGemmToCpuRuntimeDispatchV1(
              *explicit_module.module, records, error),
          "bootstrap producer must not authorize executable CPU lowering");
    check(records.empty(),
          "bootstrap-producer rejection must clear pending runtime records");
    checkContains(error, "clang-libtooling-v1",
                  "bootstrap rejection must name the required native producer");
  }

  if (!recovered_text.empty()) {
    expectMutableResultCannotForge(
        recovered_frontend, recovered_text,
        [](frontend::Result &result) {
          ++result.recovered_gemm_candidates[0].proof_ranges[1].range.begin;
        },
        "an in-bounds public proof-range shift cannot mutate sealed evidence");
    expectMutableResultCannotForge(
        recovered_frontend, recovered_text,
        [](frontend::Result &result) {
          --result.recovered_gemm_candidates[0].outer_loop_range.end;
          --result.recovered_gemm_candidates[0].proof_ranges[0].range.end;
        },
        "coordinated public outer-end/proof drift cannot mutate sealed evidence");
    expectMutableResultCannotForge(
        recovered_frontend, recovered_text,
        [](frontend::Result &result) {
          auto &candidate = result.recovered_gemm_candidates[0];
          candidate.output_parameter = "forged_output";
          candidate.lhs_parameter = "forged_lhs";
          candidate.rhs_parameter = "forged_rhs";
          candidate.m_parameter = "forged_m";
          candidate.n_parameter = "forged_n";
          candidate.k_parameter = "forged_k";
        },
        "public binding drift cannot mutate sealed evidence");
    expectMutableResultCannotForge(
        recovered_frontend, recovered_text,
        [](frontend::Result &result) {
          result.module.source_file = "forged-display.mdsl";
          result.module.translation_unit = "forged-display.mdsl";
          auto &candidate = result.recovered_gemm_candidates[0];
          candidate.source_file = "forged-display.mdsl";
          candidate.function_name = "forged_function";
        },
        "public function/source-display drift cannot mutate sealed evidence");
    expectMutableResultCannotForge(
        recovered_frontend, recovered_text,
        [](frontend::Result &result) {
          auto &candidate = result.recovered_gemm_candidates[0];
          candidate.state = frontend::RecoveredGemmState::recognized_rejected;
          candidate.fp_proof.preserve_signed_zero = true;
          std::swap(candidate.required_runtime_guards[0],
                    candidate.required_runtime_guards[1]);
          result.source_snapshot.push_back(' ');
          result.diagnostics.push_back(
              {.file = "forged-display.mdsl",
               .line = 1,
               .column = 1,
               .message = "forged diagnostic"});
        },
        "public state/FP/guard/source drift cannot mutate sealed evidence");

    frontend::Options drifted_options = relaxed_options;
    drifted_options.compiler_arguments.insert(
        drifted_options.compiler_arguments.end() - 1,
        "-DIDENTITY_DRIFT=1");
    check(frontend::stableCompilationIdentity(drifted_options) !=
              frontend::stableCompilationIdentity(relaxed_options),
          "option-drift fixture must change compilation identity");
    mlir::MLIRContext drifted_options_context;
    auto sealed_after_option_drift =
        bridge::bridgeRecoveredGemmToMatcoreMlirV1(
            *recovered_frontend.native_evidence, 0,
            drifted_options_context);
    check(sealed_after_option_drift &&
              bridge::serializeDeterministicMlir(
                  *sealed_after_option_drift.module) == recovered_text,
          "post-extraction Options drift cannot alter sealed compilation identity");
  }

  frontend::Options strict_options = optionsFor(recovered_source, false);
  frontend::Result strict_frontend;
  check(extract(strict_options, strict_frontend),
        "strict ordinary GEMM must still compile through the native frontend");
  check(strict_frontend.recovered_gemm_candidates.size() == 1 &&
            strict_frontend.recovered_gemm_candidates[0].state ==
                frontend::RecoveredGemmState::recognized_rejected,
        "strict ordinary GEMM must be recognized but rewrite-rejected");
  check(strict_frontend.native_evidence.has_value() &&
            strict_frontend.native_evidence->valid(),
        "strict native inspection must still seal its rejected evidence");
  mlir::MLIRContext strict_context;
  bridge::RecoveredGemmBridgeResultV1 strict;
  if (strict_frontend.native_evidence) {
    strict = bridge::bridgeRecoveredGemmToMatcoreMlirV1(
        *strict_frontend.native_evidence, 0, strict_context);
  }
  check(!strict,
        "strict/default C++ semantics must not produce authorized recovered IR");
  checkContains(strict.error, "recognized_guard_required",
                "strict rejection must explain the permission boundary");

  frontend::Result relabeled_strict = strict_frontend;
  if (!relabeled_strict.recovered_gemm_candidates.empty() &&
      !recovered_frontend.recovered_gemm_candidates.empty()) {
    relabeled_strict.recovered_gemm_candidates[0] =
        recovered_frontend.recovered_gemm_candidates[0];
  }
  if (relabeled_strict.native_evidence) {
    mlir::MLIRContext relabeled_context;
    auto relabeled = bridge::bridgeRecoveredGemmToMatcoreMlirV1(
        *relabeled_strict.native_evidence, 0, relabeled_context);
    check(!relabeled,
          "copying all relaxed literals onto a strict public Result cannot forge authorization");
    checkContains(relabeled.error, "recognized_guard_required",
                  "sealed strict evidence must retain its original rejection state");
  }
  if (explicit_frontend.native_evidence && strict_frontend.native_evidence) {
    const auto strict_comparison =
        bridge::compareAuthenticatedExplicitAndRecoveredGemmV1(
            *explicit_frontend.native_evidence, 0,
            *strict_frontend.native_evidence, 0);
    check(!strict_comparison,
          "authenticated equivalence must reject a strict recovered loop");
    checkContains(strict_comparison.error, "recognized_guard_required",
                  "authenticated strict comparison must explain permission rejection");
  }

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
  check(non_gemm_frontend.native_evidence.has_value() &&
            non_gemm_frontend.native_evidence->valid(),
        "non-GEMM native inspection must seal its negative result");
  mlir::MLIRContext non_gemm_context;
  bridge::RecoveredGemmBridgeResultV1 non_gemm;
  if (non_gemm_frontend.native_evidence) {
    non_gemm = bridge::bridgeRecoveredGemmToMatcoreMlirV1(
        *non_gemm_frontend.native_evidence, 0, non_gemm_context);
  }
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
