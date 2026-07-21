#include "matcore_ir_v1.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <unordered_set>

namespace matcore::mdslc::ir::v1 {
namespace {

bool validSiteId(std::string_view identifier) {
  return identifier.size() == 35 && identifier.starts_with("mc_") &&
         std::all_of(identifier.begin() + 3, identifier.end(), [](char value) {
           return (value >= '0' && value <= '9') ||
                  (value >= 'a' && value <= 'f');
         });
}

bool validSymbol(std::string_view symbol) {
  if (symbol.empty()) {
    return false;
  }
  const auto validFirst = [](unsigned char value) {
    return std::isalpha(value) != 0 || value == '_';
  };
  const auto validRest = [](unsigned char value) {
    return std::isalnum(value) != 0 || value == '_';
  };
  return validFirst(static_cast<unsigned char>(symbol.front())) &&
         std::all_of(symbol.begin() + 1, symbol.end(), [&](char value) {
           return validRest(static_cast<unsigned char>(value));
         });
}

bool known(std::string_view value) { return value != "invalid"; }

std::uint32_t naturalAlignment(DType dtype) {
  switch (dtype) {
  case DType::F16:
  case DType::BF16:
    return 2;
  case DType::F32:
  case DType::I32:
    return 4;
  case DType::F64:
    return 8;
  case DType::I8:
    return 1;
  }
  return 0;
}

bool legalAccumulation(DType element, DType accumulation) {
  switch (element) {
  case DType::F16:
  case DType::BF16:
    return accumulation == DType::F32;
  case DType::F32:
    return accumulation == DType::F32;
  case DType::F64:
    return accumulation == DType::F64;
  case DType::I8:
    return accumulation == DType::I32;
  case DType::I32:
    return accumulation == DType::I32;
  }
  return false;
}

bool verifyScalar(const ScalarExpr &value, std::string_view context,
                  std::string &error) {
  switch (value.kind) {
  case ScalarExpr::Kind::Static:
    if (value.value == 0 || !value.symbol.empty()) {
      error = std::string(context) +
              " static value must be positive and have no symbol";
      return false;
    }
    return true;
  case ScalarExpr::Kind::Dynamic:
    if (value.value != 0 || !validSymbol(value.symbol)) {
      error = std::string(context) +
              " dynamic value must have one canonical symbol and no literal";
      return false;
    }
    return true;
  }
  error = std::string(context) + " has an unknown expression kind";
  return false;
}

bool verifyTensorType(const TensorType &type, std::string_view context,
                      std::string &error) {
  if (!known(toString(type.element_dtype))) {
    error = std::string(context) + " has an unknown element dtype";
    return false;
  }
  if (type.rank != 2 || type.shape.size() != type.rank ||
      type.strides.size() != type.rank) {
    error = std::string(context) +
            " must be rank 2 with two shape and stride expressions";
    return false;
  }
  for (std::size_t index = 0; index < type.rank; ++index) {
    if (!verifyScalar(type.shape[index],
                      std::string(context) + " shape[" +
                          std::to_string(index) + "]",
                      error) ||
        !verifyScalar(type.strides[index],
                      std::string(context) + " stride[" +
                          std::to_string(index) + "]",
                      error)) {
      return false;
    }
  }
  if (!known(toString(type.layout)) ||
      !known(toString(type.memory_space))) {
    error = std::string(context) + " has an unknown layout or memory space";
    return false;
  }
  if (type.required_alignment_bytes == 0 ||
      (type.required_alignment_bytes & (type.required_alignment_bytes - 1)) !=
          0 ||
      type.required_alignment_bytes < naturalAlignment(type.element_dtype)) {
    error = std::string(context) +
            " alignment must be a power of two at least as large as the dtype";
    return false;
  }

  const ScalarExpr one = ScalarExpr::staticValue(1);
  if (type.layout == Layout::RowMajorContiguous &&
      (!(type.strides[1] == one) ||
       !(type.strides[0] == type.shape[1]))) {
    error = std::string(context) +
            " row-major contiguous strides must be [shape[1], 1]";
    return false;
  }
  if (type.layout == Layout::ColumnMajorContiguous &&
      (!(type.strides[0] == one) ||
       !(type.strides[1] == type.shape[0]))) {
    error = std::string(context) +
            " column-major contiguous strides must be [1, shape[0]]";
    return false;
  }
  return true;
}

bool verifySourceContract(const Module &module, const Operation &operation,
                          std::uint64_t previous_call_end,
                          std::string &error) {
  if (operation.source.file != module.source_file ||
      operation.source.line == 0 || operation.source.column == 0) {
    error = "operation source location must refer to the input .mdsl file";
    return false;
  }
  if (operation.call_range.begin != operation.source.offset ||
      operation.call_range.end <= operation.call_range.begin) {
    error = "operation call range must be a nonempty half-open source range";
    return false;
  }
  if (operation.call_range.begin < previous_call_end) {
    error = "operation call ranges must be sorted and non-overlapping";
    return false;
  }
  if (operation.argument_ranges.size() != 3 &&
      operation.argument_ranges.size() != 4) {
    error = "gemm requires three or four explicit source argument ranges";
    return false;
  }
  std::uint64_t previous_argument_end = operation.call_range.begin;
  for (const SourceRange &range : operation.argument_ranges) {
    if (range.begin < previous_argument_end || range.end <= range.begin ||
        range.end > operation.call_range.end) {
      error = "argument ranges must be ordered, nonempty, and inside the call";
      return false;
    }
    previous_argument_end = range.end;
  }
  return true;
}

bool verifyValue(const TensorValue &value, ValueId expected_id,
                 Mutability expected_mutability, std::string_view context,
                 std::string &error) {
  if (!known(toString(value.id)) || value.id != expected_id) {
    error = std::string(context) + " has the wrong semantic role";
    return false;
  }
  if (value.source_expression.empty()) {
    error = std::string(context) + " source expression must not be empty";
    return false;
  }
  if (!known(toString(value.mutability)) ||
      value.mutability != expected_mutability) {
    error = std::string(context) + " has the wrong mutability";
    return false;
  }
  return verifyTensorType(value.type, context, error);
}

bool exactAliasContract(const std::vector<AliasRequirement> &requirements) {
  return requirements.size() == 2 &&
         requirements[0].relation == AliasRelation::NoAlias &&
         requirements[0].first == ValueId::Output &&
         requirements[0].second == ValueId::Lhs &&
         requirements[1].relation == AliasRelation::NoAlias &&
         requirements[1].first == ValueId::Output &&
         requirements[1].second == ValueId::Rhs;
}

bool exactEffectsContract(const Effects &effects) {
  return effects.reads == std::vector<ValueId>{ValueId::Lhs, ValueId::Rhs} &&
         effects.writes == std::vector<ValueId>{ValueId::Output} &&
         effects.synchronization == Synchronization::Synchronous;
}

bool exactRequirementContract(
    const std::vector<SemanticRequirement> &requirements) {
  return requirements ==
         std::vector<SemanticRequirement>{
             SemanticRequirement::Rank2Gemm,
             SemanticRequirement::F32Arithmetic,
             SemanticRequirement::HostAddressable,
             SemanticRequirement::SynchronousExecution};
}

} // namespace

ScalarExpr ScalarExpr::staticValue(std::uint64_t literal) {
  ScalarExpr result;
  result.kind = Kind::Static;
  result.value = literal;
  return result;
}

ScalarExpr ScalarExpr::dynamic(std::string name) {
  ScalarExpr result;
  result.kind = Kind::Dynamic;
  result.symbol = std::move(name);
  return result;
}

bool operator==(const ScalarExpr &lhs, const ScalarExpr &rhs) {
  return lhs.kind == rhs.kind && lhs.value == rhs.value &&
         lhs.symbol == rhs.symbol;
}

std::string_view toString(OperationKind value) {
  return value == OperationKind::Gemm ? "gemm" : "invalid";
}

std::string_view toString(DType value) {
  switch (value) {
  case DType::F16:
    return "f16";
  case DType::BF16:
    return "bf16";
  case DType::F32:
    return "f32";
  case DType::F64:
    return "f64";
  case DType::I8:
    return "i8";
  case DType::I32:
    return "i32";
  }
  return "invalid";
}

std::string_view toString(Layout value) {
  switch (value) {
  case Layout::RowMajorContiguous:
    return "row_major_contiguous";
  case Layout::ColumnMajorContiguous:
    return "column_major_contiguous";
  case Layout::Strided:
    return "strided";
  }
  return "invalid";
}

std::string_view toString(MemorySpace value) {
  switch (value) {
  case MemorySpace::Host:
    return "host";
  case MemorySpace::Device:
    return "device";
  }
  return "invalid";
}

std::string_view toString(Mutability value) {
  switch (value) {
  case Mutability::ReadOnly:
    return "read";
  case Mutability::WriteOnly:
    return "write";
  case Mutability::ReadWrite:
    return "read_write";
  }
  return "invalid";
}

std::string_view toString(ValueId value) {
  switch (value) {
  case ValueId::Output:
    return "output";
  case ValueId::Lhs:
    return "lhs";
  case ValueId::Rhs:
    return "rhs";
  }
  return "invalid";
}

std::string_view toString(AliasRelation value) {
  return value == AliasRelation::NoAlias ? "no_alias" : "invalid";
}

std::string_view toString(Synchronization value) {
  return value == Synchronization::Synchronous ? "synchronous" : "invalid";
}

std::string_view toString(Target value) {
  return value == Target::Cpu ? "cpu" : "invalid";
}

std::string_view toString(Fallback value) {
  return value == Fallback::Error ? "error" : "invalid";
}

std::string_view toString(SemanticRequirement value) {
  switch (value) {
  case SemanticRequirement::Rank2Gemm:
    return "rank2_gemm";
  case SemanticRequirement::F32Arithmetic:
    return "f32_arithmetic";
  case SemanticRequirement::HostAddressable:
    return "host_addressable";
  case SemanticRequirement::SynchronousExecution:
    return "synchronous_execution";
  }
  return "invalid";
}

bool verify(const Module &module, std::string &error) {
  error.clear();
  if (module.translation_unit.empty()) {
    error = "translation-unit identity must not be empty";
    return false;
  }
  if (module.source_file.empty() ||
      !std::string_view(module.source_file).ends_with(".mdsl")) {
    error = "source file must be nonempty and use the .mdsl extension";
    return false;
  }
  if (module.producer != "clang-ast-json-bootstrap-v0" &&
      module.producer != "clang-libtooling-v1") {
    error = "unknown Matcore IR v1 producer";
    return false;
  }

  std::unordered_set<std::string> site_ids;
  std::uint64_t previous_call_end = 0;
  for (const Operation &operation : module.operations) {
    if (!validSiteId(operation.site_id) ||
        !site_ids.insert(operation.site_id).second) {
      error = "operation site IDs must be canonical, lowercase, and unique";
      return false;
    }
    if (!known(toString(operation.kind)) ||
        operation.kind != OperationKind::Gemm ||
        operation.canonical_callee != "matcore::mdsl::gemm") {
      error = "Matcore IR v1 only accepts canonical gemm operations";
      return false;
    }
    if (!verifySourceContract(module, operation, previous_call_end, error)) {
      return false;
    }
    previous_call_end = operation.call_range.end;

    if (!verifyValue(operation.output, ValueId::Output,
                     Mutability::WriteOnly, "gemm output", error)) {
      return false;
    }
    if (operation.operands.size() != 2 ||
        !verifyValue(operation.operands[0], ValueId::Lhs,
                     Mutability::ReadOnly, "gemm lhs", error) ||
        !verifyValue(operation.operands[1], ValueId::Rhs,
                     Mutability::ReadOnly, "gemm rhs", error)) {
      if (error.empty()) {
        error = "gemm requires ordered lhs and rhs operands";
      }
      return false;
    }

    const TensorType &output = operation.output.type;
    const TensorType &lhs = operation.operands[0].type;
    const TensorType &rhs = operation.operands[1].type;
    if (output.element_dtype != lhs.element_dtype ||
        output.element_dtype != rhs.element_dtype) {
      error = "gemm input and output element dtypes must match";
      return false;
    }
    if (!known(toString(operation.accumulation_dtype)) ||
        !legalAccumulation(output.element_dtype,
                           operation.accumulation_dtype)) {
      error = "gemm element and accumulation dtype combination is illegal";
      return false;
    }
    if (!exactRequirementContract(operation.requirements)) {
      error = "gemm semantic capability requirements must be the canonical "
              "rank2/f32/host/synchronous set";
      return false;
    }
    if (output.element_dtype != DType::F32 ||
        operation.accumulation_dtype != DType::F32 ||
        output.memory_space != MemorySpace::Host ||
        lhs.memory_space != MemorySpace::Host ||
        rhs.memory_space != MemorySpace::Host) {
      error = "current Matcore IR v1 GEMM semantics require host f32 values "
              "with f32 accumulation";
      return false;
    }
    if (!(lhs.shape[0] == output.shape[0]) ||
        !(lhs.shape[1] == rhs.shape[0]) ||
        !(rhs.shape[1] == output.shape[1])) {
      error = "gemm requires exact symbolic M/K/N shape relationships";
      return false;
    }
    if (operation.output.source_expression ==
            operation.operands[0].source_expression ||
        operation.output.source_expression ==
            operation.operands[1].source_expression) {
      error = "gemm output must not alias an input expression";
      return false;
    }
    if (!exactAliasContract(operation.alias_requirements)) {
      error = "gemm requires exact output/lhs and output/rhs no-alias contracts";
      return false;
    }
    if (!exactEffectsContract(operation.effects)) {
      error = "gemm effects must synchronously read lhs/rhs and write output";
      return false;
    }
    if (!known(toString(operation.policy.target)) ||
        !known(toString(operation.policy.fallback)) ||
        operation.policy.target != Target::Cpu ||
        operation.policy.fallback != Fallback::Error) {
      error = "Matcore IR v1 GEMM requires target=cpu and fallback=error";
      return false;
    }
  }
  return true;
}

} // namespace matcore::mdslc::ir::v1
