#include "matcore_ir.h"
#include "matcore_ir_v1.h"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

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

v1::Module withDTypeContract(const v1::Module &source, v1::DType input,
                             v1::DType output, v1::DType accumulation,
                             std::uint32_t input_alignment,
                             std::uint32_t output_alignment) {
  v1::Module result = source;
  auto &operation = result.operations[0];
  operation.output.type.element_dtype = output;
  operation.output.type.required_alignment_bytes = output_alignment;
  operation.operands[0].type.element_dtype = input;
  operation.operands[0].type.required_alignment_bytes = input_alignment;
  operation.operands[1].type.element_dtype = input;
  operation.operands[1].type.required_alignment_bytes = input_alignment;
  operation.accumulation_dtype = accumulation;
  operation.requirements[1] =
      input == v1::DType::I8 ? v1::SemanticRequirement::I32Arithmetic
                             : v1::SemanticRequirement::F32Arithmetic;
  return result;
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
  check(typed.translation_unit == captured.translation_unit &&
            typed.source_file == captured.source_file &&
            typed.producer == captured.producer,
        "upgrade must preserve module source provenance and producer");
  check(operation.site_id == captured.operations[0].site_id &&
            operation.source.file == captured.operations[0].source.file &&
            operation.source.offset == captured.operations[0].source.offset &&
            operation.call_range.begin ==
                captured.operations[0].call_range.begin &&
            operation.call_range.end == captured.operations[0].call_range.end &&
            operation.argument_ranges.size() ==
                captured.operations[0].argument_ranges.size(),
        "upgrade must preserve stable site and exact source ranges");
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

  v1::Module two_operations = typed;
  v1::Operation independent = operation;
  independent.site_id = "mc_11111111111111111111111111111111";
  const std::uint64_t range_shift = operation.call_range.end + 10 -
                                    operation.call_range.begin;
  independent.source.offset += range_shift;
  independent.source.line += 10;
  independent.call_range.begin += range_shift;
  independent.call_range.end += range_shift;
  for (v0::SourceRange &range : independent.argument_ranges) {
    range.begin += range_shift;
    range.end += range_shift;
  }
  independent.output.source_expression = "Z";
  independent.operands[0].source_expression = "X";
  independent.operands[1].source_expression = "Y";
  two_operations.operations.push_back(std::move(independent));
  check(v1::verify(two_operations, error),
        "independent operations may reuse operation-scoped m/k/n symbols");
  check(two_operations.operations[0].output.type.shape[0].symbol ==
            two_operations.operations[1].output.type.shape[0].symbol,
        "same symbol spelling across operations must remain explicitly local");

  const std::string v1_json = v1::serializeDeterministicJson(typed);
  const std::string v1_golden =
      readFile(std::string(MDSLC_IR_TEST_SOURCE_DIR) +
               "/gemm_capture.v1.golden.json");
  check(v1_json == v1_golden,
        "canonical v1 serialization must match the reviewed golden bytes");
  check(v1_json == v1::serializeDeterministicJson(typed),
        "v1 serialization must be deterministic across calls");
  checkContains(v1_json, "\"version\": 1", "v1 JSON must identify version 1");
  checkContains(v1_json, "\"requirements\"",
                "v1 JSON must serialize semantic requirements");
  check(v1::probeJsonVersion(v1_json, version, error) && version == 1,
        "version probe must identify v1");
  v1::Module reparsed;
  check(!v1::parseAndVerifyJson(v0_json, reparsed, error),
        "v1 parser must not silently parse or upgrade v0 JSON");
  checkContains(error, "expected version 1",
                "wrong exact parser diagnostic must name expected version");

  check(v1::parseAndVerifyJson(v1_golden, reparsed, error),
        "reviewed v1 golden must parse and verify");
  check(v1::serializeDeterministicJson(reparsed) == v1_golden,
        "v1 golden parse/serialize must be byte stable");

  v0::Module projected;
  check(v1::projectToV0(reparsed, projected, error),
        "canonical upgraded v1 must project to v0");
  check(v0::serializeDeterministicJson(projected) == v0_json,
        "v0 -> v1 -> v0 must preserve exact v0 bytes");

  const v1::Module bf16 =
      withDTypeContract(typed, v1::DType::BF16, v1::DType::F32,
                        v1::DType::F32, 2, 4);
  check(v1::verify(bf16, error),
        "BF16 inputs with F32 accumulation and output must verify");
  const std::string bf16_json = v1::serializeDeterministicJson(bf16);
  v1::Module bf16_reparsed;
  check(v1::parseAndVerifyJson(bf16_json, bf16_reparsed, error) &&
            v1::serializeDeterministicJson(bf16_reparsed) == bf16_json,
        "BF16/F32 semantics must round-trip deterministically through JSON");
  check(!v1::projectToV0(bf16, projected, error),
        "BF16 operands must not be erased by the F32-only v0 projection");
  checkContains(error, "losslessly",
                "BF16 projection rejection must explain the information loss");

  const v1::Module i8 =
      withDTypeContract(typed, v1::DType::I8, v1::DType::I32,
                        v1::DType::I32, 1, 4);
  check(v1::verify(i8, error),
        "I8 inputs with I32 accumulation and output must verify");
  const std::string i8_json = v1::serializeDeterministicJson(i8);
  checkContains(i8_json, "\"i32_arithmetic\"",
                "I8/I32 JSON names its exact arithmetic requirement");
  v1::Module i8_reparsed;
  check(v1::parseAndVerifyJson(i8_json, i8_reparsed, error) &&
            v1::serializeDeterministicJson(i8_reparsed) == i8_json,
        "I8/I32 semantics must round-trip deterministically through JSON");
  check(!v1::projectToV0(i8, projected, error),
        "I8/I32 semantics must not cross the F32-only v0 projection");
  checkContains(error, "cannot be represented losslessly",
                "I8 projection rejection must name the lossy boundary");

  v1::Module static_shapes = typed;
  makeStatic(static_shapes.operations[0].output, 2, 4);
  makeStatic(static_shapes.operations[0].operands[0], 2, 3);
  makeStatic(static_shapes.operations[0].operands[1], 3, 4);
  check(v1::verify(static_shapes, error),
        "consistent static GEMM shapes and strides must verify");
  const std::string static_json =
      v1::serializeDeterministicJson(static_shapes);
  v1::Module static_reparsed;
  check(v1::parseAndVerifyJson(static_json, static_reparsed, error) &&
            v1::serializeDeterministicJson(static_reparsed) == static_json,
        "static dimensions and strides must parse/serialize byte-stably");
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

  v1::Module renamed_symbols = typed;
  renamed_symbols.operations[0].output.type.shape[0] =
      v1::ScalarExpr::dynamic("rows");
  renamed_symbols.operations[0].operands[0].type.shape[0] =
      v1::ScalarExpr::dynamic("rows");
  check(v1::verify(renamed_symbols, error),
        "noncanonical but consistent dynamic symbols must verify in v1");
  check(!v1::projectToV0(renamed_symbols, projected, error),
        "v0 projection must reject noncanonical dynamic symbols");
  checkContains(error, "losslessly",
                "renamed dynamic symbols must receive a lossy diagnostic");

  v1::Module invalid = typed;
  invalid.operations[0].operands[1].type.shape[0] =
      v1::ScalarExpr::dynamic("other_k");
  expectInvalid(invalid, "M/K/N", "mismatched contraction dimensions fail");

  invalid = typed;
  invalid.translation_unit.clear();
  expectInvalid(invalid, "translation-unit", "missing translation unit fails");

  invalid = typed;
  invalid.source_file = "input.cpp";
  expectInvalid(invalid, ".mdsl", "non-MDSL source file fails");

  invalid = typed;
  invalid.producer = "untrusted-producer";
  expectInvalid(invalid, "producer", "unknown producer fails");

  invalid = typed;
  invalid.operations[0].site_id = "MC_3c5b6d5e7992fb7b249de44210c6415d";
  expectInvalid(invalid, "site IDs", "noncanonical site ID fails");

  invalid = typed;
  invalid.operations[0].kind = static_cast<v1::OperationKind>(99);
  expectInvalid(invalid, "canonical gemm", "unknown operation kind fails");

  invalid = typed;
  invalid.operations[0].canonical_callee = "matcore::mdsl::other";
  expectInvalid(invalid, "canonical gemm", "noncanonical callee fails");

  invalid = typed;
  invalid.operations[0].source.file = "other.mdsl";
  expectInvalid(invalid, "input .mdsl", "foreign source location fails");

  invalid = typed;
  invalid.operations[0].source.line = 0;
  expectInvalid(invalid, "source location", "zero source line fails");

  invalid = typed;
  ++invalid.operations[0].call_range.begin;
  expectInvalid(invalid, "half-open", "source offset/range mismatch fails");

  invalid = typed;
  invalid.operations[0].argument_ranges.resize(2);
  expectInvalid(invalid, "three or four", "wrong argument range count fails");

  invalid = typed;
  invalid.operations[0].argument_ranges[1].begin =
      invalid.operations[0].argument_ranges[0].begin;
  expectInvalid(invalid, "ordered", "overlapping argument ranges fail");

  invalid = typed;
  v1::Operation overlapping = invalid.operations[0];
  overlapping.site_id = "mc_11111111111111111111111111111111";
  invalid.operations.push_back(std::move(overlapping));
  expectInvalid(invalid, "sorted", "overlapping operation ranges fail");

  invalid = typed;
  invalid.operations[0].output.id = v1::ValueId::Lhs;
  expectInvalid(invalid, "semantic role", "incorrect output role fails");

  invalid = typed;
  invalid.operations[0].operands[0].source_expression.clear();
  expectInvalid(invalid, "source expression", "empty source expression fails");

  invalid = typed;
  invalid.operations[0].operands.clear();
  expectInvalid(invalid, "ordered lhs", "missing operands fail");

  invalid = typed;
  invalid.operations[0].output.type.rank = 3;
  expectInvalid(invalid, "rank 2", "wrong tensor rank fails");

  invalid = typed;
  invalid.operations[0].output.type.shape.pop_back();
  expectInvalid(invalid, "rank 2", "wrong shape cardinality fails");

  invalid = typed;
  invalid.operations[0].output.type.element_dtype =
      static_cast<v1::DType>(99);
  expectInvalid(invalid, "element dtype", "unknown dtype fails");

  invalid = typed;
  invalid.operations[0].operands[1].type.element_dtype = v1::DType::F64;
  invalid.operations[0].operands[1].type.required_alignment_bytes = 8;
  expectInvalid(invalid, "lhs and rhs", "mismatched input dtypes fail");

  invalid = typed;
  invalid.operations[0].accumulation_dtype = v1::DType::F64;
  expectInvalid(invalid, "dtype contract", "illegal accumulation dtype fails");

  invalid = typed;
  invalid.operations[0].output.type.layout = static_cast<v1::Layout>(99);
  expectInvalid(invalid, "layout or memory", "unknown layout fails");

  invalid = typed;
  invalid.operations[0].output.type.memory_space =
      static_cast<v1::MemorySpace>(99);
  expectInvalid(invalid, "layout or memory", "unknown memory space fails");

  invalid = typed;
  invalid.operations[0].output.type.required_alignment_bytes = 2;
  expectInvalid(invalid, "at least", "under-aligned f32 value fails");

  invalid = typed;
  invalid.operations[0].output.type.layout =
      v1::Layout::ColumnMajorContiguous;
  expectInvalid(invalid, "column-major", "invalid column-major strides fail");

  invalid = typed;
  v1::ScalarExpr malformed_static = v1::ScalarExpr::staticValue(2);
  malformed_static.symbol = "not_allowed";
  invalid.operations[0].output.type.shape[0] = malformed_static;
  expectInvalid(invalid, "no symbol", "static expression with a symbol fails");

  invalid = typed;
  v1::ScalarExpr malformed_dynamic = v1::ScalarExpr::dynamic("m");
  malformed_dynamic.value = 7;
  invalid.operations[0].output.type.shape[0] = malformed_dynamic;
  expectInvalid(invalid, "no literal", "dynamic expression with a literal fails");

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
  invalid.operations[0].effects.synchronization =
      static_cast<v1::Synchronization>(99);
  expectInvalid(invalid, "effects", "unknown synchronization fails");

  invalid = typed;
  invalid.operations[0].alias_requirements.pop_back();
  expectInvalid(invalid, "no-alias", "missing alias precondition fails");

  invalid = typed;
  invalid.operations[0].output.source_expression =
      invalid.operations[0].operands[0].source_expression;
  expectInvalid(invalid, "must not alias", "matching output/input expressions fail");

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
  invalid.operations[0].output.type.required_alignment_bytes = 2;
  invalid.operations[0].operands[0].type.element_dtype = v1::DType::BF16;
  invalid.operations[0].operands[0].type.required_alignment_bytes = 2;
  invalid.operations[0].operands[1].type.element_dtype = v1::DType::BF16;
  invalid.operations[0].operands[1].type.required_alignment_bytes = 2;
  expectInvalid(invalid, "dtype contract",
                "BF16 output is rejected because BF16 GEMM outputs F32");

  invalid = typed;
  invalid.operations[0].output.type.memory_space = v1::MemorySpace::Device;
  invalid.operations[0].operands[0].type.memory_space = v1::MemorySpace::Device;
  invalid.operations[0].operands[1].type.memory_space = v1::MemorySpace::Device;
  expectInvalid(invalid, "host-addressable",
                "device-resident v1 GEMM fails CPU scope");

  invalid = typed;
  invalid.operations[0].policy.target = static_cast<v1::Target>(99);
  expectInvalid(invalid, "target=cpu", "unknown target policy fails");

  invalid = typed;
  invalid.operations[0].policy.fallback = static_cast<v1::Fallback>(99);
  expectInvalid(invalid, "target=cpu", "unknown fallback policy fails");

  invalid = typed;
  invalid.operations.push_back(invalid.operations[0]);
  expectInvalid(invalid, "unique", "duplicate site IDs fail");

  v0::Module invalid_v0 = captured;
  invalid_v0.operations[0].target = "cuda";
  check(!v1::fromV0(invalid_v0, reparsed, error),
        "unverified v0 must never cross the v1 boundary");
  checkContains(error, "unverified", "invalid v0 upgrade must name the boundary");
  check(reparsed.operations.empty(),
        "failed v0 upgrade must not leave a stale typed module");

  const std::string unknown_version =
      replaceOnce(v1_json, "\"version\": 1", "\"version\": 99");
  check(!v1::probeJsonVersion(unknown_version, version, error),
        "unknown schema version must fail dispatch");
  checkContains(error, "unsupported", "unknown version diagnostic is explicit");
  check(!v1::parseAndVerifyJson(unknown_version, reparsed, error),
        "v1 parser must reject unknown version without fallback");
  check(reparsed.operations.empty(), "failed v1 parse must reset output module");

  const std::string unknown_schema = replaceOnce(
      v1_json, "\"schema\": \"matcore.ir\"",
      "\"schema\": \"matcore.unknown\"");
  check(!v1::probeJsonVersion(unknown_schema, version, error),
        "unknown schema must fail version dispatch");
  checkContains(error, "schema", "unknown schema diagnostic is explicit");
  check(!v1::parseAndVerifyJson(unknown_schema, reparsed, error),
        "v1 parser must reject an unknown schema");

  const std::string string_version = replaceOnce(
      v1_json, "\"version\": 1", "\"version\": \"1\"");
  check(!v1::probeJsonVersion(string_version, version, error),
        "non-integer version must fail version dispatch");
  checkContains(error, "integer", "non-integer version diagnostic is explicit");

  check(!v1::parseAndVerifyJson("[]", reparsed, error),
        "non-object JSON root must fail");
  checkContains(error, "root", "non-object root diagnostic is explicit");

  const std::string extra_field = replaceOnce(
      v1_json, "\"producer\": \"clang-libtooling-v1\",",
      "\"producer\": \"clang-libtooling-v1\",\n  \"extra\": true,");
  check(!v1::parseAndVerifyJson(extra_field, reparsed, error),
        "v1 exact-member parser must reject unknown fields");
  checkContains(error, "unexpected", "unknown-field diagnostic is explicit");

  const std::string extra_tensor_field = replaceOnce(
      v1_json, "\"required_alignment_bytes\": 4",
      "\"required_alignment_bytes\": 4, \"extra\": true");
  check(!v1::parseAndVerifyJson(extra_tensor_field, reparsed, error),
        "nested unknown tensor fields must fail exact-member parsing");
  checkContains(error, "tensor value",
                "nested unknown field diagnostic names the tensor object");

  const std::string extra_scalar_field = replaceOnce(
      v1_json, "\"symbol\": \"m\"",
      "\"symbol\": \"m\", \"value\": 1");
  check(!v1::parseAndVerifyJson(extra_scalar_field, reparsed, error),
        "scalar expressions must reject unknown members");
  checkContains(error, "scalar expression",
                "scalar exact-member diagnostic is explicit");

  const std::string bad_dtype =
      replaceOnce(v1_json, "\"dtype\": \"f32\"", "\"dtype\": \"f128\"");
  check(!v1::parseAndVerifyJson(bad_dtype, reparsed, error),
        "unknown dtype must fail JSON parsing");
  checkContains(error, "dtype", "unknown dtype diagnostic is explicit");

  const std::string renamed_operation_field = replaceOnce(
      v1_json, "\"canonical_callee\"", "\"callee\"");
  check(!v1::parseAndVerifyJson(renamed_operation_field, reparsed, error),
        "missing operation members must fail exact-member parsing");
  checkContains(error, "operation", "missing operation field names its object");

  const std::string bad_source_range = replaceOnce(
      v1_json, "\"end\": 550", "\"end\": 419");
  check(!v1::parseAndVerifyJson(bad_source_range, reparsed, error),
        "nonempty source ranges are verified after JSON parsing");
  checkContains(error, "verifier rejected", "semantic JSON rejection is wrapped");

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

  std::string invalid_utf8_v1 = v1_json;
  const std::size_t v1_expression =
      invalid_utf8_v1.find("\"expression\": \"C\"");
  check(v1_expression != std::string::npos,
        "v1 UTF-8 mutation target must exist");
  if (v1_expression != std::string::npos) {
    const std::size_t value_offset =
        v1_expression + std::string_view("\"expression\": \"").size();
    invalid_utf8_v1[value_offset] = static_cast<char>(0xff);
  }
  check(!v1::probeJsonVersion(invalid_utf8_v1, version, error),
        "version probing must reject invalid UTF-8 before dispatch");
  checkContains(error, "malformed JSON",
                "invalid UTF-8 probe diagnostic must be actionable");
  check(!v1::parseAndVerifyJson(invalid_utf8_v1, reparsed, error),
        "v1 parser must reject invalid UTF-8");
  checkContains(error, "malformed JSON",
                "invalid UTF-8 v1 diagnostic must be actionable");

  std::string invalid_utf8_v0 = v0_json;
  const std::size_t v0_expression =
      invalid_utf8_v0.find("\"expression\": \"C\"");
  check(v0_expression != std::string::npos,
        "v0 UTF-8 mutation target must exist");
  if (v0_expression != std::string::npos) {
    const std::size_t value_offset =
        v0_expression + std::string_view("\"expression\": \"").size();
    invalid_utf8_v0[value_offset] = static_cast<char>(0xff);
  }
  v0::Module invalid_utf8_module;
  check(!v0::parseAndVerifyJson(invalid_utf8_v0, invalid_utf8_module, error),
        "v0 compatibility parser must also reject invalid UTF-8");
  checkContains(error, "malformed JSON",
                "invalid UTF-8 v0 diagnostic must be actionable");

  if (failures != 0) {
    std::cerr << failures << " of " << checks << " IR v1 checks failed\n";
    return 1;
  }
  std::cout << "Matcore IR v1 PASS: " << checks << " checks\n";
  return 0;
}
