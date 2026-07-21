#include "matcore_ir.h"
#include "matcore_ir_v1.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>

namespace {

namespace v0 = matcore::mdslc::ir;
namespace v1 = matcore::mdslc::ir::v1;

int failures = 0;
int checks = 0;

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

std::string readFile(const std::string &path) {
  std::ifstream stream(path, std::ios::binary);
  std::ostringstream contents;
  contents << stream.rdbuf();
  check(stream.good() || stream.eof(), "test fixture must be readable");
  return contents.str();
}

std::string replaceOnce(std::string input, std::string_view needle,
                        std::string_view replacement) {
  const std::size_t position = input.find(needle);
  check(position != std::string::npos, "JSON mutation needle must exist");
  if (position != std::string::npos) {
    input.replace(position, needle.size(), replacement);
  }
  return input;
}

void makeStatic(v1::TensorValue &value, std::uint64_t rows,
                std::uint64_t columns) {
  value.type.shape = {v1::ScalarExpr::staticValue(rows),
                      v1::ScalarExpr::staticValue(columns)};
  value.type.strides = {v1::ScalarExpr::staticValue(columns),
                        v1::ScalarExpr::staticValue(1)};
}

void expectInvalid(const v1::Module &module, std::string_view expected,
                   std::string_view message) {
  std::string error;
  check(!v1::verify(module, error), message);
  checkContains(error, expected, "invalid module diagnostic must be actionable");
}

} // namespace

int main() {
  const std::string v0_json = readFile(
      std::string(MDSLC_FRONTEND_TEST_SOURCE_DIR) +
      "/gemm_capture.golden.json");

  std::string error;
  v0::Module captured;
  check(v0::parseAndVerifyJson(v0_json, captured, error),
        "existing v0 golden must parse and verify");
  check(error.empty(), "successful v0 parse must clear the diagnostic");
  check(v0::serializeDeterministicJson(captured) == v0_json,
        "v0 parse/serialize bytes must remain frozen");

  std::uint32_t version = 99;
  check(v1::probeJsonVersion(v0_json, version, error) && version == 0,
        "version probe must identify v0 without parsing it as v1");

  v1::Module typed;
  check(v1::fromV0(captured, typed, error),
        "verified v0 must upgrade to typed v1");
  check(v1::verify(typed, error), "upgraded typed v1 must verify");
  check(typed.operations.size() == 1, "upgrade must preserve operation count");
  const v1::Operation &operation = typed.operations.front();
  check(operation.kind == v1::OperationKind::Gemm,
        "operation kind must be typed GEMM");
  check(operation.accumulation_dtype == v1::DType::F32,
        "accumulation dtype must be explicit f32");
  check(operation.requirements ==
            std::vector<v1::SemanticRequirement>{
                v1::SemanticRequirement::Rank2Gemm,
                v1::SemanticRequirement::F32Arithmetic,
                v1::SemanticRequirement::HostAddressable,
                v1::SemanticRequirement::SynchronousExecution},
        "semantic requirements must be explicit and canonically ordered");
  check(operation.output.type.required_alignment_bytes == 4,
        "minimum ABI alignment must be explicit");
  check(operation.output.type.shape[0] == v1::ScalarExpr::dynamic("m") &&
            operation.output.type.shape[1] ==
                v1::ScalarExpr::dynamic("n"),
        "output shape must carry symbolic M/N");
  check(operation.operands[0].type.shape[0] ==
                v1::ScalarExpr::dynamic("m") &&
            operation.operands[0].type.shape[1] ==
                v1::ScalarExpr::dynamic("k") &&
            operation.operands[1].type.shape[0] ==
                v1::ScalarExpr::dynamic("k") &&
            operation.operands[1].type.shape[1] ==
                v1::ScalarExpr::dynamic("n"),
        "operand shapes must encode exact symbolic M/K/N relationships");
  check(operation.output.type.strides[0] ==
                v1::ScalarExpr::dynamic("n") &&
            operation.output.type.strides[1] ==
                v1::ScalarExpr::staticValue(1),
        "row-major output stride must be structurally explicit");
  check(operation.output.mutability == v1::Mutability::WriteOnly &&
            operation.operands[0].mutability == v1::Mutability::ReadOnly &&
            operation.operands[1].mutability == v1::Mutability::ReadOnly,
        "mutability contract must be typed");
  check(operation.effects.reads ==
                std::vector<v1::ValueId>{v1::ValueId::Lhs,
                                         v1::ValueId::Rhs} &&
            operation.effects.writes ==
                std::vector<v1::ValueId>{v1::ValueId::Output},
        "read/write effects must be typed and ordered");

  const std::string v1_json = v1::serializeDeterministicJson(typed);
  check(v1_json == v1::serializeDeterministicJson(typed),
        "v1 serialization must be deterministic across calls");
  checkContains(v1_json, "\"version\": 1", "v1 JSON must identify version 1");
  checkContains(v1_json, "\"requirements\"",
                "v1 JSON must serialize semantic requirements");
  check(v1::probeJsonVersion(v1_json, version, error) && version == 1,
        "version probe must identify v1");

  v1::Module reparsed;
  check(v1::parseAndVerifyJson(v1_json, reparsed, error),
        "serialized v1 must parse and verify");
  check(v1::serializeDeterministicJson(reparsed) == v1_json,
        "v1 parse/serialize must be byte stable");

  v0::Module projected;
  check(v1::projectToV0(reparsed, projected, error),
        "canonical upgraded v1 must project to v0");
  check(v0::serializeDeterministicJson(projected) == v0_json,
        "v0 -> v1 -> v0 must preserve exact v0 bytes");

  v1::Module static_shapes = typed;
  makeStatic(static_shapes.operations[0].output, 2, 4);
  makeStatic(static_shapes.operations[0].operands[0], 2, 3);
  makeStatic(static_shapes.operations[0].operands[1], 3, 4);
  check(v1::verify(static_shapes, error),
        "consistent static GEMM shapes and strides must verify");
  check(!v1::projectToV0(static_shapes, projected, error),
        "static typed shapes must not be erased by v0 projection");
  checkContains(error, "losslessly", "lossy static projection must explain why");

  v1::Module aligned = typed;
  aligned.operations[0].output.type.required_alignment_bytes = 16;
  aligned.operations[0].operands[0].type.required_alignment_bytes = 16;
  aligned.operations[0].operands[1].type.required_alignment_bytes = 16;
  check(v1::verify(aligned, error), "stronger alignment must verify in v1");
  check(!v1::projectToV0(aligned, projected, error),
        "stronger alignment must not be discarded by projection");

  v1::Module column_major = static_shapes;
  for (v1::TensorValue *value :
       {&column_major.operations[0].output,
        &column_major.operations[0].operands[0],
        &column_major.operations[0].operands[1]}) {
    value->type.layout = v1::Layout::ColumnMajorContiguous;
    value->type.strides = {v1::ScalarExpr::staticValue(1),
                           value->type.shape[0]};
  }
  check(v1::verify(column_major, error),
        "well-formed column-major typed values must verify");
  check(!v1::projectToV0(column_major, projected, error),
        "column-major semantics must not project to row-major v0");

  v1::Module strided = static_shapes;
  strided.operations[0].output.type.layout = v1::Layout::Strided;
  strided.operations[0].output.type.strides = {
      v1::ScalarExpr::staticValue(8), v1::ScalarExpr::staticValue(1)};
  check(v1::verify(strided, error), "explicit positive strides must verify");
  check(!v1::projectToV0(strided, projected, error),
        "general strided semantics must not project to contiguous v0");

  v1::Module invalid = typed;
  invalid.operations[0].operands[1].type.shape[0] =
      v1::ScalarExpr::dynamic("other_k");
  expectInvalid(invalid, "M/K/N", "mismatched contraction dimensions fail");

  invalid = typed;
  invalid.operations[0].output.type.strides[0] =
      v1::ScalarExpr::dynamic("wrong_stride");
  expectInvalid(invalid, "row-major", "invalid contiguous stride fails");

  invalid = typed;
  invalid.operations[0].output.type.required_alignment_bytes = 3;
  expectInvalid(invalid, "power of two", "non-power-of-two alignment fails");

  invalid = typed;
  invalid.operations[0].output.type.shape[0] =
      v1::ScalarExpr::staticValue(0);
  expectInvalid(invalid, "positive", "zero static dimension fails");

  invalid = typed;
  invalid.operations[0].output.type.shape[0] =
      v1::ScalarExpr::dynamic("bad-symbol");
  invalid.operations[0].operands[0].type.shape[0] =
      v1::ScalarExpr::dynamic("bad-symbol");
  expectInvalid(invalid, "canonical symbol", "invalid dynamic symbol fails");

  invalid = typed;
  invalid.operations[0].output.mutability = v1::Mutability::ReadWrite;
  expectInvalid(invalid, "mutability", "read-write output fails v1 GEMM contract");

  invalid = typed;
  invalid.operations[0].effects.writes.clear();
  expectInvalid(invalid, "effects", "missing write effect fails");

  invalid = typed;
  invalid.operations[0].alias_requirements.pop_back();
  expectInvalid(invalid, "no-alias", "missing alias precondition fails");

  invalid = typed;
  std::swap(invalid.operations[0].requirements[0],
            invalid.operations[0].requirements[1]);
  expectInvalid(invalid, "capability requirements",
                "reordered semantic requirements fail");

  invalid = typed;
  invalid.operations[0].requirements.push_back(
      v1::SemanticRequirement::Rank2Gemm);
  expectInvalid(invalid, "capability requirements",
                "extra semantic requirement fails");

  invalid = typed;
  invalid.operations[0].output.type.element_dtype = v1::DType::BF16;
  invalid.operations[0].operands[0].type.element_dtype = v1::DType::BF16;
  invalid.operations[0].operands[1].type.element_dtype = v1::DType::BF16;
  expectInvalid(invalid, "host f32", "unsupported bf16 execution contract fails");

  invalid = typed;
  invalid.operations[0].output.type.memory_space = v1::MemorySpace::Device;
  invalid.operations[0].operands[0].type.memory_space = v1::MemorySpace::Device;
  invalid.operations[0].operands[1].type.memory_space = v1::MemorySpace::Device;
  expectInvalid(invalid, "host f32", "device-resident v1 GEMM fails CPU scope");

  invalid = typed;
  invalid.operations.push_back(invalid.operations[0]);
  expectInvalid(invalid, "unique", "duplicate site IDs fail");

  v0::Module invalid_v0 = captured;
  invalid_v0.operations[0].target = "cuda";
  check(!v1::fromV0(invalid_v0, reparsed, error),
        "unverified v0 must never cross the v1 boundary");
  checkContains(error, "unverified", "invalid v0 upgrade must name the boundary");

  const std::string unknown_version =
      replaceOnce(v1_json, "\"version\": 1", "\"version\": 99");
  check(!v1::probeJsonVersion(unknown_version, version, error),
        "unknown schema version must fail dispatch");
  checkContains(error, "unsupported", "unknown version diagnostic is explicit");
  check(!v1::parseAndVerifyJson(unknown_version, reparsed, error),
        "v1 parser must reject unknown version without fallback");
  check(reparsed.operations.empty(), "failed v1 parse must reset output module");

  const std::string extra_field = replaceOnce(
      v1_json, "\"producer\": \"clang-libtooling-v1\",",
      "\"producer\": \"clang-libtooling-v1\",\n  \"extra\": true,");
  check(!v1::parseAndVerifyJson(extra_field, reparsed, error),
        "v1 exact-member parser must reject unknown fields");
  checkContains(error, "unexpected", "unknown-field diagnostic is explicit");

  const std::string bad_requirement =
      replaceOnce(v1_json, "\"f32_arithmetic\"", "\"f64_arithmetic\"");
  check(!v1::parseAndVerifyJson(bad_requirement, reparsed, error),
        "unknown semantic requirements must fail parsing");
  checkContains(error, "semantic requirement",
                "unknown requirement diagnostic names its field");

  const std::string malformed = "{\"schema\": \"matcore.ir\",";
  check(!v1::parseAndVerifyJson(malformed, reparsed, error),
        "malformed v1 JSON must fail cleanly");
  checkContains(error, "malformed JSON", "malformed JSON reports parser offset");

  if (failures != 0) {
    std::cerr << failures << " of " << checks << " IR v1 checks failed\n";
    return 1;
  }
  std::cout << "Matcore IR v1 PASS: " << checks << " checks\n";
  return 0;
}
