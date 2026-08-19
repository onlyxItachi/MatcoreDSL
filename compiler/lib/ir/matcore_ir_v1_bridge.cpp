#include "matcore_ir_v1.h"

#include <array>
#include <string_view>

namespace matcore::mdslc::ir::v1 {
namespace {

TensorType canonicalTensorType(const ScalarExpr &rows,
                               const ScalarExpr &columns) {
  TensorType type;
  type.element_dtype = DType::F32;
  type.rank = 2;
  type.shape = {rows, columns};
  type.strides = {columns, ScalarExpr::staticValue(1)};
  type.layout = Layout::RowMajorContiguous;
  type.memory_space = MemorySpace::Host;
  type.required_alignment_bytes = 4;
  return type;
}

TensorValue canonicalTensor(ValueId id, std::string expression,
                            Mutability mutability, const ScalarExpr &rows,
                            const ScalarExpr &columns) {
  TensorValue value;
  value.id = id;
  value.source_expression = std::move(expression);
  value.mutability = mutability;
  value.type = canonicalTensorType(rows, columns);
  return value;
}

std::vector<SemanticRequirement> canonicalRequirements() {
  return {SemanticRequirement::Rank2Gemm,
          SemanticRequirement::F32Arithmetic,
          SemanticRequirement::HostAddressable,
          SemanticRequirement::SynchronousExecution};
}

bool canonicalScalar(const ScalarExpr &actual, const ScalarExpr &expected) {
  return actual == expected;
}

bool canonicalTensorForV0(const TensorValue &value, ValueId expected_id,
                          Mutability expected_mutability,
                          const ScalarExpr &expected_rows,
                          const ScalarExpr &expected_columns,
                          std::string_view context, std::string &error) {
  const TensorType &type = value.type;
  if (value.id != expected_id || value.mutability != expected_mutability ||
      type.element_dtype != DType::F32 || type.rank != 2 ||
      type.shape.size() != 2 || type.strides.size() != 2 ||
      type.layout != Layout::RowMajorContiguous ||
      type.memory_space != MemorySpace::Host ||
      type.required_alignment_bytes != 4 ||
      !canonicalScalar(type.shape[0], expected_rows) ||
      !canonicalScalar(type.shape[1], expected_columns) ||
      !canonicalScalar(type.strides[0], expected_columns) ||
      !canonicalScalar(type.strides[1], ScalarExpr::staticValue(1))) {
    error = std::string(context) +
            " cannot be represented losslessly by Matcore IR v0";
    return false;
  }
  return true;
}

} // namespace

bool fromV0(const ir::Module &source, Module &destination,
            std::string &error) {
  destination = Module{};
  error.clear();
  std::string v0_error;
  if (!ir::verify(source, v0_error)) {
    error = "cannot upgrade unverified Matcore IR v0: " + v0_error;
    return false;
  }

  Module upgraded;
  upgraded.translation_unit = source.translation_unit;
  upgraded.source_file = source.source_file;
  upgraded.producer = source.producer;
  for (const ir::Operation &operation : source.operations) {
    const ScalarExpr m = ScalarExpr::dynamic("m");
    const ScalarExpr k = ScalarExpr::dynamic("k");
    const ScalarExpr n = ScalarExpr::dynamic("n");

    Operation converted;
    converted.site_id = operation.site_id;
    converted.kind = OperationKind::Gemm;
    converted.canonical_callee = operation.canonical_callee;
    converted.source = operation.source;
    converted.call_range = operation.call_range;
    converted.argument_ranges = operation.argument_ranges;
    converted.output = canonicalTensor(
        ValueId::Output, operation.output.expression, Mutability::WriteOnly, m,
        n);
    converted.operands = {
        canonicalTensor(ValueId::Lhs, operation.operands[0].expression,
                        Mutability::ReadOnly, m, k),
        canonicalTensor(ValueId::Rhs, operation.operands[1].expression,
                        Mutability::ReadOnly, k, n)};
    converted.accumulation_dtype = DType::F32;
    converted.requirements = canonicalRequirements();
    converted.alias_requirements = {
        {AliasRelation::NoAlias, ValueId::Output, ValueId::Lhs},
        {AliasRelation::NoAlias, ValueId::Output, ValueId::Rhs}};
    converted.effects.reads = {ValueId::Lhs, ValueId::Rhs};
    converted.effects.writes = {ValueId::Output};
    converted.effects.synchronization = Synchronization::Synchronous;
    converted.policy.target = Target::Cpu;
    converted.policy.fallback = Fallback::Error;
    upgraded.operations.push_back(std::move(converted));
  }

  if (!verify(upgraded, error)) {
    error = "internal v0-to-v1 conversion produced invalid IR: " + error;
    return false;
  }
  destination = std::move(upgraded);
  return true;
}

bool projectToV0(const Module &source, ir::Module &destination,
                 std::string &error) {
  destination = ir::Module{};
  error.clear();
  if (!verify(source, error)) {
    error = "cannot project unverified Matcore IR v1: " + error;
    return false;
  }

  ir::Module projected;
  projected.translation_unit = source.translation_unit;
  projected.source_file = source.source_file;
  projected.producer = source.producer;
  for (const Operation &operation : source.operations) {
    const ScalarExpr m = ScalarExpr::dynamic("m");
    const ScalarExpr k = ScalarExpr::dynamic("k");
    const ScalarExpr n = ScalarExpr::dynamic("n");
    if (operation.kind != OperationKind::Gemm ||
        operation.accumulation_dtype != DType::F32 ||
        operation.requirements != canonicalRequirements() ||
        !canonicalTensorForV0(operation.output, ValueId::Output,
                              Mutability::WriteOnly, m, n, "gemm output",
                              error) ||
        !canonicalTensorForV0(operation.operands[0], ValueId::Lhs,
                              Mutability::ReadOnly, m, k, "gemm lhs", error) ||
        !canonicalTensorForV0(operation.operands[1], ValueId::Rhs,
                              Mutability::ReadOnly, k, n, "gemm rhs", error)) {
      if (error.empty()) {
        error = "gemm semantics cannot be represented losslessly by Matcore "
                "IR v0";
      }
      return false;
    }

    ir::Operation converted;
    converted.site_id = operation.site_id;
    converted.kind = "gemm";
    converted.canonical_callee = operation.canonical_callee;
    converted.source = operation.source;
    converted.call_range = operation.call_range;
    converted.argument_ranges = operation.argument_ranges;
    converted.output = {"output", operation.output.source_expression, "write"};
    converted.operands = {
        {"lhs", operation.operands[0].source_expression, "read"},
        {"rhs", operation.operands[1].source_expression, "read"}};
    converted.target = "cpu";
    converted.fallback = "error";
    if (operation.output.type.shape[0].kind == ScalarExpr::Kind::Static) {
      converted.static_m = static_cast<std::int64_t>(operation.output.type.shape[0].value);
    }
    if (operation.output.type.shape[1].kind == ScalarExpr::Kind::Static) {
      converted.static_n = static_cast<std::int64_t>(operation.output.type.shape[1].value);
    }
    if (operation.operands[0].type.shape[1].kind == ScalarExpr::Kind::Static) {
      converted.static_k = static_cast<std::int64_t>(operation.operands[0].type.shape[1].value);
    }
    projected.operations.push_back(std::move(converted));
  }

  std::string v0_error;
  if (!ir::verify(projected, v0_error)) {
    error = "internal v1-to-v0 projection produced invalid IR: " + v0_error;
    return false;
  }
  destination = std::move(projected);
  return true;
}

} // namespace matcore::mdslc::ir::v1
