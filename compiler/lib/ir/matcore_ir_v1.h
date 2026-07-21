#ifndef MATCORE_MDSLC_MATCORE_IR_V1_H
#define MATCORE_MDSLC_MATCORE_IR_V1_H

#include "matcore_ir.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace matcore::mdslc::ir::v1 {

inline constexpr std::uint32_t kMatcoreIrVersion = 1;

enum class OperationKind { Gemm };
enum class DType { F16, BF16, F32, F64, I8, I32 };
enum class Layout { RowMajorContiguous, ColumnMajorContiguous, Strided };
enum class MemorySpace { Host, Device };
enum class Mutability { ReadOnly, WriteOnly, ReadWrite };
enum class ValueId { Output, Lhs, Rhs };
enum class AliasRelation { NoAlias };
enum class Synchronization { Synchronous };
enum class Target { Cpu };
enum class Fallback { Error };
enum class SemanticRequirement {
  Rank2Gemm,
  F32Arithmetic,
  HostAddressable,
  SynchronousExecution
};

// A dimension or stride is either a positive compile-time integer or a
// positive runtime value identified by a stable semantic symbol. Dynamic
// symbols are scoped to their enclosing Operation: reusing a symbol within
// one operation expresses equality, while the same spelling in another
// operation has no semantic relationship. No implicit unification is
// performed.
struct ScalarExpr {
  enum class Kind { Static, Dynamic };

  Kind kind = Kind::Dynamic;
  std::uint64_t value = 0;
  std::string symbol;

  static ScalarExpr staticValue(std::uint64_t value);
  static ScalarExpr dynamic(std::string symbol);
};

bool operator==(const ScalarExpr &lhs, const ScalarExpr &rhs);

struct TensorType {
  DType element_dtype = DType::F32;
  std::uint32_t rank = 2;
  std::vector<ScalarExpr> shape;
  std::vector<ScalarExpr> strides;
  Layout layout = Layout::RowMajorContiguous;
  MemorySpace memory_space = MemorySpace::Host;
  std::uint32_t required_alignment_bytes = 4;
};

struct TensorValue {
  ValueId id = ValueId::Lhs;
  std::string source_expression;
  Mutability mutability = Mutability::ReadOnly;
  TensorType type;
};

struct AliasRequirement {
  AliasRelation relation = AliasRelation::NoAlias;
  ValueId first = ValueId::Output;
  ValueId second = ValueId::Lhs;
};

struct Effects {
  std::vector<ValueId> reads;
  std::vector<ValueId> writes;
  Synchronization synchronization = Synchronization::Synchronous;
};

struct Policy {
  Target target = Target::Cpu;
  Fallback fallback = Fallback::Error;
};

struct Operation {
  // Operation is the structural scope for every dynamic ScalarExpr nested in
  // this value. Matcore IR v1 intentionally has no module-wide dimension
  // namespace.
  std::string site_id;
  OperationKind kind = OperationKind::Gemm;
  std::string canonical_callee;
  SourceLocation source;
  SourceRange call_range;
  std::vector<SourceRange> argument_ranges;
  TensorValue output;
  std::vector<TensorValue> operands;
  DType accumulation_dtype = DType::F32;
  // Target-independent semantics required of any legal implementation. These
  // are not detected device features and must remain canonically ordered.
  std::vector<SemanticRequirement> requirements;
  std::vector<AliasRequirement> alias_requirements;
  Effects effects;
  Policy policy;
};

struct Module {
  std::string translation_unit;
  std::string source_file;
  std::string producer;
  std::vector<Operation> operations;
};

// Verifies the complete typed semantic contract. It does not perform target
// capability legalization or select an implementation variant.
bool verify(const Module &module, std::string &error);

// JSON v1 has a fixed key order, exact-member parsing, and one trailing newline.
std::string serializeDeterministicJson(const Module &module);
bool parseAndVerifyJson(std::string_view json, Module &module,
                        std::string &error);

// Lightweight schema/version discovery for callers that dispatch to an exact
// parser. Unknown versions are reported, never retried as another version.
bool probeJsonVersion(std::string_view json, std::uint32_t &version,
                      std::string &error);

// The bridge never guesses. Upgrade accepts only an already valid v0 module.
// Projection accepts only the canonical dynamic host/f32 subset that the v0
// serializer, generated code, and runtime ABI represent without information
// loss.
bool fromV0(const ir::Module &source, Module &destination, std::string &error);
bool projectToV0(const Module &source, ir::Module &destination,
                 std::string &error);

std::string_view toString(OperationKind value);
std::string_view toString(DType value);
std::string_view toString(Layout value);
std::string_view toString(MemorySpace value);
std::string_view toString(Mutability value);
std::string_view toString(ValueId value);
std::string_view toString(AliasRelation value);
std::string_view toString(Synchronization value);
std::string_view toString(Target value);
std::string_view toString(Fallback value);
std::string_view toString(SemanticRequirement value);

} // namespace matcore::mdslc::ir::v1

#endif // MATCORE_MDSLC_MATCORE_IR_V1_H
