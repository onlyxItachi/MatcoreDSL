#include "matcore_ir_v1.h"

#include <rapidjson/document.h>
#include <rapidjson/error/en.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/stringbuffer.h>

#include <initializer_list>
#include <limits>
#include <string_view>

namespace matcore::mdslc::ir::v1 {
namespace {

using Writer = rapidjson::PrettyWriter<rapidjson::StringBuffer>;

void writeString(Writer &writer, std::string_view value) {
  writer.String(value.data(), static_cast<rapidjson::SizeType>(value.size()));
}

void writeRange(Writer &writer, const SourceRange &range) {
  writer.StartObject();
  writer.Key("begin");
  writer.Uint64(range.begin);
  writer.Key("end");
  writer.Uint64(range.end);
  writer.EndObject();
}

void writeScalar(Writer &writer, const ScalarExpr &value) {
  writer.StartObject();
  writer.Key("kind");
  if (value.kind == ScalarExpr::Kind::Static) {
    writer.String("static");
    writer.Key("value");
    writer.Uint64(value.value);
  } else {
    writer.String("dynamic");
    writer.Key("symbol");
    writeString(writer, value.symbol);
  }
  writer.EndObject();
}

void writeTensor(Writer &writer, const TensorValue &tensor) {
  writer.StartObject();
  writer.Key("role");
  writeString(writer, toString(tensor.id));
  writer.Key("expression");
  writeString(writer, tensor.source_expression);
  writer.Key("dtype");
  writeString(writer, toString(tensor.type.element_dtype));
  writer.Key("rank");
  writer.Uint(tensor.type.rank);
  writer.Key("shape");
  writer.StartArray();
  for (const ScalarExpr &value : tensor.type.shape) {
    writeScalar(writer, value);
  }
  writer.EndArray();
  writer.Key("strides");
  writer.StartArray();
  for (const ScalarExpr &value : tensor.type.strides) {
    writeScalar(writer, value);
  }
  writer.EndArray();
  writer.Key("layout");
  writeString(writer, toString(tensor.type.layout));
  writer.Key("mutability");
  writeString(writer, toString(tensor.mutability));
  writer.Key("memory_space");
  writeString(writer, toString(tensor.type.memory_space));
  writer.Key("required_alignment_bytes");
  writer.Uint(tensor.type.required_alignment_bytes);
  writer.EndObject();
}

const rapidjson::Value *requiredMember(const rapidjson::Value &object,
                                       const char *name,
                                       rapidjson::Type type,
                                       std::string &error) {
  if (!object.IsObject()) {
    error = "expected a JSON object while reading " + std::string(name);
    return nullptr;
  }
  const auto iterator = object.FindMember(name);
  if (iterator == object.MemberEnd() || iterator->value.GetType() != type) {
    error = "missing or invalid Matcore IR v1 field: " + std::string(name);
    return nullptr;
  }
  return &iterator->value;
}

bool exactMembers(const rapidjson::Value &object,
                  std::initializer_list<const char *> names,
                  std::string_view context, std::string &error) {
  if (!object.IsObject() || object.MemberCount() != names.size()) {
    error = std::string(context) + " has unexpected or missing fields";
    return false;
  }
  for (const char *name : names) {
    if (!object.HasMember(name)) {
      error = std::string(context) + " is missing field: " + name;
      return false;
    }
  }
  return true;
}

bool readString(const rapidjson::Value &object, const char *name,
                std::string &value, std::string &error) {
  const rapidjson::Value *encoded =
      requiredMember(object, name, rapidjson::kStringType, error);
  if (encoded == nullptr) {
    return false;
  }
  value.assign(encoded->GetString(), encoded->GetStringLength());
  return true;
}

bool readUint64(const rapidjson::Value &object, const char *name,
                std::uint64_t &value, std::string &error) {
  const rapidjson::Value *encoded = object.IsObject() && object.HasMember(name)
                                        ? &object[name]
                                        : nullptr;
  if (encoded == nullptr || !encoded->IsUint64()) {
    error = "missing or invalid Matcore IR v1 integer field: " +
            std::string(name);
    return false;
  }
  value = encoded->GetUint64();
  return true;
}

bool readUint32(const rapidjson::Value &object, const char *name,
                std::uint32_t &value, std::string &error) {
  std::uint64_t wide = 0;
  if (!readUint64(object, name, wide, error) ||
      wide > std::numeric_limits<std::uint32_t>::max()) {
    if (error.empty()) {
      error = "Matcore IR v1 integer exceeds uint32 range: " +
              std::string(name);
    }
    return false;
  }
  value = static_cast<std::uint32_t>(wide);
  return true;
}

bool parseRange(const rapidjson::Value &encoded, SourceRange &range,
                std::string &error) {
  return exactMembers(encoded, {"begin", "end"}, "source range", error) &&
         readUint64(encoded, "begin", range.begin, error) &&
         readUint64(encoded, "end", range.end, error);
}

bool parseScalar(const rapidjson::Value &encoded, ScalarExpr &value,
                 std::string &error) {
  std::string kind;
  if (!readString(encoded, "kind", kind, error)) {
    return false;
  }
  if (kind == "static") {
    if (!exactMembers(encoded, {"kind", "value"}, "scalar expression",
                      error)) {
      return false;
    }
    value = ScalarExpr{};
    value.kind = ScalarExpr::Kind::Static;
    return readUint64(encoded, "value", value.value, error);
  }
  if (kind == "dynamic") {
    if (!exactMembers(encoded, {"kind", "symbol"}, "scalar expression",
                      error)) {
      return false;
    }
    value = ScalarExpr{};
    value.kind = ScalarExpr::Kind::Dynamic;
    return readString(encoded, "symbol", value.symbol, error);
  }
  error = "unknown Matcore IR v1 scalar expression kind: " + kind;
  return false;
}

template <typename Enum>
bool parseNamedEnum(std::string_view encoded,
                    std::initializer_list<std::pair<std::string_view, Enum>>
                        choices,
                    Enum &value, std::string_view context, std::string &error) {
  for (const auto &[name, candidate] : choices) {
    if (encoded == name) {
      value = candidate;
      return true;
    }
  }
  error = "unknown Matcore IR v1 " + std::string(context) + ": " +
          std::string(encoded);
  return false;
}

bool readValueId(const rapidjson::Value &object, const char *name,
                 ValueId &value, std::string &error) {
  std::string encoded;
  return readString(object, name, encoded, error) &&
         parseNamedEnum(encoded,
                        {{"output", ValueId::Output}, {"lhs", ValueId::Lhs},
                         {"rhs", ValueId::Rhs}},
                        value, "value role", error);
}

bool parseTensor(const rapidjson::Value &encoded, TensorValue &tensor,
                 std::string &error) {
  if (!exactMembers(encoded,
                    {"role", "expression", "dtype", "rank", "shape",
                     "strides", "layout", "mutability", "memory_space",
                     "required_alignment_bytes"},
                    "tensor value", error) ||
      !readValueId(encoded, "role", tensor.id, error) ||
      !readString(encoded, "expression", tensor.source_expression, error) ||
      !readUint32(encoded, "rank", tensor.type.rank, error) ||
      !readUint32(encoded, "required_alignment_bytes",
                  tensor.type.required_alignment_bytes, error)) {
    return false;
  }

  std::string dtype;
  std::string layout;
  std::string mutability;
  std::string memory_space;
  if (!readString(encoded, "dtype", dtype, error) ||
      !parseNamedEnum(dtype,
                      {{"f16", DType::F16}, {"bf16", DType::BF16},
                       {"f32", DType::F32}, {"f64", DType::F64},
                       {"i8", DType::I8}, {"i32", DType::I32}},
                      tensor.type.element_dtype, "dtype", error) ||
      !readString(encoded, "layout", layout, error) ||
      !parseNamedEnum(layout,
                      {{"row_major_contiguous", Layout::RowMajorContiguous},
                       {"column_major_contiguous",
                        Layout::ColumnMajorContiguous},
                       {"strided", Layout::Strided}},
                      tensor.type.layout, "layout", error) ||
      !readString(encoded, "mutability", mutability, error) ||
      !parseNamedEnum(mutability,
                      {{"read", Mutability::ReadOnly},
                       {"write", Mutability::WriteOnly},
                       {"read_write", Mutability::ReadWrite}},
                      tensor.mutability, "mutability", error) ||
      !readString(encoded, "memory_space", memory_space, error) ||
      !parseNamedEnum(memory_space,
                      {{"host", MemorySpace::Host},
                       {"device", MemorySpace::Device}},
                      tensor.type.memory_space, "memory space", error)) {
    return false;
  }

  const rapidjson::Value *shape =
      requiredMember(encoded, "shape", rapidjson::kArrayType, error);
  const rapidjson::Value *strides =
      requiredMember(encoded, "strides", rapidjson::kArrayType, error);
  if (shape == nullptr || strides == nullptr) {
    return false;
  }
  for (const rapidjson::Value &item : shape->GetArray()) {
    ScalarExpr value;
    if (!parseScalar(item, value, error)) {
      return false;
    }
    tensor.type.shape.push_back(std::move(value));
  }
  for (const rapidjson::Value &item : strides->GetArray()) {
    ScalarExpr value;
    if (!parseScalar(item, value, error)) {
      return false;
    }
    tensor.type.strides.push_back(std::move(value));
  }
  return true;
}

bool parseValueIdArray(const rapidjson::Value &object, const char *name,
                       std::vector<ValueId> &values, std::string &error) {
  const rapidjson::Value *encoded =
      requiredMember(object, name, rapidjson::kArrayType, error);
  if (encoded == nullptr) {
    return false;
  }
  for (const rapidjson::Value &item : encoded->GetArray()) {
    if (!item.IsString()) {
      error = "Matcore IR v1 effect roles must be strings";
      return false;
    }
    ValueId value;
    if (!parseNamedEnum(
            std::string_view(item.GetString(), item.GetStringLength()),
            {{"output", ValueId::Output}, {"lhs", ValueId::Lhs},
             {"rhs", ValueId::Rhs}},
            value, "value role", error)) {
      return false;
    }
    values.push_back(value);
  }
  return true;
}

bool parseOperation(const rapidjson::Value &encoded, Operation &operation,
                    std::string &error) {
  if (!exactMembers(
          encoded,
          {"site_id", "kind", "canonical_callee", "source",
           "source_argument_ranges", "output", "operands",
           "accumulation_dtype", "requirements", "alias_requirements",
           "effects", "policy"},
          "operation", error) ||
      !readString(encoded, "site_id", operation.site_id, error) ||
      !readString(encoded, "canonical_callee", operation.canonical_callee,
                  error)) {
    return false;
  }

  std::string kind;
  std::string accumulation;
  if (!readString(encoded, "kind", kind, error) ||
      !parseNamedEnum(kind, {{"gemm", OperationKind::Gemm}}, operation.kind,
                      "operation kind", error) ||
      !readString(encoded, "accumulation_dtype", accumulation, error) ||
      !parseNamedEnum(accumulation,
                      {{"f16", DType::F16}, {"bf16", DType::BF16},
                       {"f32", DType::F32}, {"f64", DType::F64},
                       {"i8", DType::I8}, {"i32", DType::I32}},
                      operation.accumulation_dtype, "dtype", error)) {
    return false;
  }

  const rapidjson::Value *source =
      requiredMember(encoded, "source", rapidjson::kObjectType, error);
  if (source == nullptr ||
      !exactMembers(*source,
                    {"file", "line", "column", "byte_offset", "byte_range"},
                    "source location", error) ||
      !readString(*source, "file", operation.source.file, error) ||
      !readUint32(*source, "line", operation.source.line, error) ||
      !readUint32(*source, "column", operation.source.column, error) ||
      !readUint64(*source, "byte_offset", operation.source.offset, error)) {
    return false;
  }
  const rapidjson::Value *byte_range =
      requiredMember(*source, "byte_range", rapidjson::kObjectType, error);
  if (byte_range == nullptr ||
      !parseRange(*byte_range, operation.call_range, error)) {
    return false;
  }

  const rapidjson::Value *argument_ranges = requiredMember(
      encoded, "source_argument_ranges", rapidjson::kArrayType, error);
  if (argument_ranges == nullptr) {
    return false;
  }
  for (const rapidjson::Value &item : argument_ranges->GetArray()) {
    SourceRange range;
    if (!parseRange(item, range, error)) {
      return false;
    }
    operation.argument_ranges.push_back(range);
  }

  const rapidjson::Value *output =
      requiredMember(encoded, "output", rapidjson::kObjectType, error);
  const rapidjson::Value *operands =
      requiredMember(encoded, "operands", rapidjson::kArrayType, error);
  if (output == nullptr || operands == nullptr ||
      !parseTensor(*output, operation.output, error)) {
    return false;
  }
  for (const rapidjson::Value &item : operands->GetArray()) {
    TensorValue operand;
    if (!parseTensor(item, operand, error)) {
      return false;
    }
    operation.operands.push_back(std::move(operand));
  }

  const rapidjson::Value *requirements =
      requiredMember(encoded, "requirements", rapidjson::kArrayType, error);
  if (requirements == nullptr) {
    return false;
  }
  for (const rapidjson::Value &item : requirements->GetArray()) {
    if (!item.IsString()) {
      error = "Matcore IR v1 semantic requirements must be strings";
      return false;
    }
    SemanticRequirement requirement;
    if (!parseNamedEnum(
            std::string_view(item.GetString(), item.GetStringLength()),
            {{"rank2_gemm", SemanticRequirement::Rank2Gemm},
             {"f32_arithmetic", SemanticRequirement::F32Arithmetic},
             {"host_addressable", SemanticRequirement::HostAddressable},
             {"synchronous_execution",
              SemanticRequirement::SynchronousExecution}},
            requirement, "semantic requirement", error)) {
      return false;
    }
    operation.requirements.push_back(requirement);
  }

  const rapidjson::Value *aliases = requiredMember(
      encoded, "alias_requirements", rapidjson::kArrayType, error);
  if (aliases == nullptr) {
    return false;
  }
  for (const rapidjson::Value &item : aliases->GetArray()) {
    AliasRequirement requirement;
    std::string relation;
    if (!exactMembers(item, {"relation", "between"}, "alias requirement",
                      error) ||
        !readString(item, "relation", relation, error) ||
        !parseNamedEnum(relation, {{"no_alias", AliasRelation::NoAlias}},
                        requirement.relation, "alias relation", error)) {
      return false;
    }
    const rapidjson::Value *between =
        requiredMember(item, "between", rapidjson::kArrayType, error);
    if (between == nullptr || between->Size() != 2 ||
        !(*between)[0].IsString() || !(*between)[1].IsString()) {
      error = "Matcore IR v1 alias between must contain exactly two roles";
      return false;
    }
    if (!parseNamedEnum(
            std::string_view((*between)[0].GetString(),
                             (*between)[0].GetStringLength()),
            {{"output", ValueId::Output}, {"lhs", ValueId::Lhs},
             {"rhs", ValueId::Rhs}},
            requirement.first, "value role", error) ||
        !parseNamedEnum(
            std::string_view((*between)[1].GetString(),
                             (*between)[1].GetStringLength()),
            {{"output", ValueId::Output}, {"lhs", ValueId::Lhs},
             {"rhs", ValueId::Rhs}},
            requirement.second, "value role", error)) {
      return false;
    }
    operation.alias_requirements.push_back(requirement);
  }

  const rapidjson::Value *effects =
      requiredMember(encoded, "effects", rapidjson::kObjectType, error);
  std::string synchronization;
  if (effects == nullptr ||
      !exactMembers(*effects, {"reads", "writes", "synchronization"},
                    "effects", error) ||
      !parseValueIdArray(*effects, "reads", operation.effects.reads, error) ||
      !parseValueIdArray(*effects, "writes", operation.effects.writes,
                         error) ||
      !readString(*effects, "synchronization", synchronization, error) ||
      !parseNamedEnum(synchronization,
                      {{"synchronous", Synchronization::Synchronous}},
                      operation.effects.synchronization, "synchronization",
                      error)) {
    return false;
  }

  const rapidjson::Value *policy =
      requiredMember(encoded, "policy", rapidjson::kObjectType, error);
  std::string target;
  std::string fallback;
  return policy != nullptr &&
         exactMembers(*policy, {"target", "fallback"}, "policy", error) &&
         readString(*policy, "target", target, error) &&
         parseNamedEnum(target, {{"cpu", Target::Cpu}},
                        operation.policy.target, "target", error) &&
         readString(*policy, "fallback", fallback, error) &&
         parseNamedEnum(fallback, {{"error", Fallback::Error}},
                        operation.policy.fallback, "fallback", error);
}

bool parseDocument(std::string_view json, rapidjson::Document &document,
                   std::string &error) {
  document.Parse(json.data(), json.size());
  if (!document.HasParseError()) {
    return true;
  }
  error = "malformed JSON at byte " +
          std::to_string(document.GetErrorOffset()) + ": " +
          rapidjson::GetParseError_En(document.GetParseError());
  return false;
}

} // namespace

std::string serializeDeterministicJson(const Module &module) {
  rapidjson::StringBuffer buffer;
  Writer writer(buffer);
  writer.SetIndent(' ', 2);

  writer.StartObject();
  writer.Key("schema");
  writer.String("matcore.ir");
  writer.Key("version");
  writer.Uint(kMatcoreIrVersion);
  writer.Key("producer");
  writeString(writer, module.producer);
  writer.Key("translation_unit");
  writer.StartObject();
  writer.Key("identity");
  writeString(writer, module.translation_unit);
  writer.Key("source_file");
  writeString(writer, module.source_file);
  writer.EndObject();
  writer.Key("operations");
  writer.StartArray();
  for (const Operation &operation : module.operations) {
    writer.StartObject();
    writer.Key("site_id");
    writeString(writer, operation.site_id);
    writer.Key("kind");
    writeString(writer, toString(operation.kind));
    writer.Key("canonical_callee");
    writeString(writer, operation.canonical_callee);
    writer.Key("source");
    writer.StartObject();
    writer.Key("file");
    writeString(writer, operation.source.file);
    writer.Key("line");
    writer.Uint(operation.source.line);
    writer.Key("column");
    writer.Uint(operation.source.column);
    writer.Key("byte_offset");
    writer.Uint64(operation.source.offset);
    writer.Key("byte_range");
    writeRange(writer, operation.call_range);
    writer.EndObject();
    writer.Key("source_argument_ranges");
    writer.StartArray();
    for (const SourceRange &range : operation.argument_ranges) {
      writeRange(writer, range);
    }
    writer.EndArray();
    writer.Key("output");
    writeTensor(writer, operation.output);
    writer.Key("operands");
    writer.StartArray();
    for (const TensorValue &operand : operation.operands) {
      writeTensor(writer, operand);
    }
    writer.EndArray();
    writer.Key("accumulation_dtype");
    writeString(writer, toString(operation.accumulation_dtype));
    writer.Key("requirements");
    writer.StartArray();
    for (SemanticRequirement requirement : operation.requirements) {
      writeString(writer, toString(requirement));
    }
    writer.EndArray();
    writer.Key("alias_requirements");
    writer.StartArray();
    for (const AliasRequirement &requirement :
         operation.alias_requirements) {
      writer.StartObject();
      writer.Key("relation");
      writeString(writer, toString(requirement.relation));
      writer.Key("between");
      writer.StartArray();
      writeString(writer, toString(requirement.first));
      writeString(writer, toString(requirement.second));
      writer.EndArray();
      writer.EndObject();
    }
    writer.EndArray();
    writer.Key("effects");
    writer.StartObject();
    writer.Key("reads");
    writer.StartArray();
    for (ValueId value : operation.effects.reads) {
      writeString(writer, toString(value));
    }
    writer.EndArray();
    writer.Key("writes");
    writer.StartArray();
    for (ValueId value : operation.effects.writes) {
      writeString(writer, toString(value));
    }
    writer.EndArray();
    writer.Key("synchronization");
    writeString(writer, toString(operation.effects.synchronization));
    writer.EndObject();
    writer.Key("policy");
    writer.StartObject();
    writer.Key("target");
    writeString(writer, toString(operation.policy.target));
    writer.Key("fallback");
    writeString(writer, toString(operation.policy.fallback));
    writer.EndObject();
    writer.EndObject();
  }
  writer.EndArray();
  writer.EndObject();

  return std::string(buffer.GetString(), buffer.GetSize()) + '\n';
}

bool probeJsonVersion(std::string_view json, std::uint32_t &version,
                      std::string &error) {
  version = 0;
  error.clear();
  rapidjson::Document document;
  if (!parseDocument(json, document, error) || !document.IsObject()) {
    if (error.empty()) {
      error = "Matcore IR root must be a JSON object";
    }
    return false;
  }
  std::string schema;
  std::uint32_t parsed_version = 0;
  if (!readString(document, "schema", schema, error) ||
      schema != "matcore.ir") {
    if (error.empty()) {
      error = "unsupported Matcore IR schema: " + schema;
    }
    return false;
  }
  if (!readUint32(document, "version", parsed_version, error)) {
    return false;
  }
  if (parsed_version != ir::kMatcoreIrVersion &&
      parsed_version != kMatcoreIrVersion) {
    error = "unsupported Matcore IR version: " +
            std::to_string(parsed_version);
    return false;
  }
  version = parsed_version;
  return true;
}

bool parseAndVerifyJson(std::string_view json, Module &module,
                        std::string &error) {
  module = Module{};
  error.clear();
  rapidjson::Document document;
  if (!parseDocument(json, document, error)) {
    return false;
  }
  if (!document.IsObject()) {
    error = "Matcore IR root must be a JSON object";
    return false;
  }
  if (!exactMembers(document,
                    {"schema", "version", "producer", "translation_unit",
                     "operations"},
                    "Matcore IR root", error)) {
    return false;
  }
  std::string schema;
  std::uint32_t version = 0;
  if (!readString(document, "schema", schema, error) ||
      schema != "matcore.ir") {
    if (error.empty()) {
      error = "unsupported Matcore IR schema: " + schema;
    }
    return false;
  }
  if (!readUint32(document, "version", version, error) ||
      version != kMatcoreIrVersion) {
    if (error.empty()) {
      error = "unsupported Matcore IR version; expected version 1";
    }
    return false;
  }

  Module parsed;
  if (!readString(document, "producer", parsed.producer, error)) {
    return false;
  }
  const rapidjson::Value *translation_unit = requiredMember(
      document, "translation_unit", rapidjson::kObjectType, error);
  if (translation_unit == nullptr ||
      !exactMembers(*translation_unit, {"identity", "source_file"},
                    "translation unit", error) ||
      !readString(*translation_unit, "identity", parsed.translation_unit,
                  error) ||
      !readString(*translation_unit, "source_file", parsed.source_file,
                  error)) {
    return false;
  }
  const rapidjson::Value *operations =
      requiredMember(document, "operations", rapidjson::kArrayType, error);
  if (operations == nullptr) {
    return false;
  }
  for (const rapidjson::Value &item : operations->GetArray()) {
    if (!item.IsObject()) {
      error = "Matcore IR v1 operation must be an object";
      return false;
    }
    Operation operation;
    if (!parseOperation(item, operation, error)) {
      return false;
    }
    parsed.operations.push_back(std::move(operation));
  }
  if (!verify(parsed, error)) {
    error = "Matcore IR v1 verifier rejected input: " + error;
    return false;
  }
  module = std::move(parsed);
  return true;
}

} // namespace matcore::mdslc::ir::v1
