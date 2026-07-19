#include "matcore_ir.h"

#include <rapidjson/document.h>
#include <rapidjson/error/en.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/stringbuffer.h>

#include <string_view>
#include <unordered_set>

namespace matcore::mdslc::ir {
namespace {

using Writer = rapidjson::PrettyWriter<rapidjson::StringBuffer>;

void writeString(Writer &writer, std::string_view value) {
  writer.String(value.data(), static_cast<rapidjson::SizeType>(value.size()));
}

void writeMatrixValue(Writer &writer, const MatrixValue &value) {
  writer.StartObject();
  writer.Key("role");
  writeString(writer, value.role);
  writer.Key("expression");
  writeString(writer, value.expression);
  writer.Key("dtype");
  writer.String("f32");
  writer.Key("rank");
  writer.Uint(2);
  writer.Key("shape");
  writer.StartArray();
  writer.String("?");
  writer.String("?");
  writer.EndArray();
  writer.Key("strides");
  writer.StartArray();
  writer.String("dynamic-columns");
  writer.String("1");
  writer.EndArray();
  writer.Key("layout");
  writer.String("row_major_contiguous");
  writer.Key("mutability");
  writeString(writer, value.mutability);
  writer.Key("memory_space");
  writer.String("host");
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
    error = "missing or invalid Matcore IR field: " + std::string(name);
    return nullptr;
  }
  return &iterator->value;
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
    error = "missing or invalid Matcore IR integer field: " + std::string(name);
    return false;
  }
  value = encoded->GetUint64();
  return true;
}

bool parseRange(const rapidjson::Value &encoded, SourceRange &range,
                std::string &error) {
  return readUint64(encoded, "begin", range.begin, error) &&
         readUint64(encoded, "end", range.end, error);
}

bool parseMatrix(const rapidjson::Value &encoded, MatrixValue &matrix,
                 std::string &error) {
  if (!readString(encoded, "role", matrix.role, error) ||
      !readString(encoded, "expression", matrix.expression, error) ||
      !readString(encoded, "mutability", matrix.mutability, error)) {
    return false;
  }
  std::string dtype;
  std::string layout;
  std::string memory_space;
  if (!readString(encoded, "dtype", dtype, error) ||
      !readString(encoded, "layout", layout, error) ||
      !readString(encoded, "memory_space", memory_space, error)) {
    return false;
  }
  const rapidjson::Value *rank =
      encoded.IsObject() && encoded.HasMember("rank") ? &encoded["rank"] : nullptr;
  if (dtype != "f32" || rank == nullptr || !rank->IsUint() ||
      rank->GetUint() != 2 || layout != "row_major_contiguous" ||
      memory_space != "host") {
    error = "Matcore IR v0 matrix metadata must be host f32 rank-2 contiguous";
    return false;
  }
  return true;
}

} // namespace

bool verify(const Module &module, std::string &error) {
  if (module.translation_unit.empty()) {
    error = "translation-unit identity must not be empty";
    return false;
  }
  if (module.source_file.empty()) {
    error = "source file must not be empty";
    return false;
  }
  if (!std::string_view(module.source_file).ends_with(".mdsl")) {
    error = "source file must use the .mdsl extension";
    return false;
  }
  if (module.producer != "clang-ast-json-bootstrap-v0") {
    error = "unknown Matcore IR v0 producer";
    return false;
  }

  std::unordered_set<std::string> site_ids;
  std::uint64_t previous_call_end = 0;
  for (const Operation &operation : module.operations) {
    if (operation.site_id.empty() || !site_ids.insert(operation.site_id).second) {
      error = "operation site IDs must be nonempty and unique";
      return false;
    }
    if (operation.kind != "gemm" ||
        operation.canonical_callee != "matcore::mdsl::gemm") {
      error = "Matcore IR v0 only accepts canonical gemm operations";
      return false;
    }
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
    previous_call_end = operation.call_range.end;
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
    if (operation.output.role != "output" ||
        operation.output.mutability != "write" ||
        operation.output.expression.empty()) {
      error = "gemm requires one explicit mutable output";
      return false;
    }
    if (operation.operands.size() != 2 ||
        operation.operands[0].role != "lhs" ||
        operation.operands[1].role != "rhs") {
      error = "gemm requires ordered lhs and rhs operands";
      return false;
    }
    for (const MatrixValue &operand : operation.operands) {
      if (operand.mutability != "read" || operand.expression.empty()) {
        error = "gemm operands must be readable stable expressions";
        return false;
      }
    }
    if (operation.output.expression == operation.operands[0].expression ||
        operation.output.expression == operation.operands[1].expression) {
      error = "gemm output must not alias an input expression";
      return false;
    }
    if (operation.target != "cpu" || operation.fallback != "error") {
      error = "bootstrap Matcore IR v0 requires target=cpu and fallback=error";
      return false;
    }
  }
  return true;
}

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
    writeString(writer, operation.kind);
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
    writer.StartObject();
    writer.Key("begin");
    writer.Uint64(operation.call_range.begin);
    writer.Key("end");
    writer.Uint64(operation.call_range.end);
    writer.EndObject();
    writer.EndObject();
    writer.Key("source_argument_ranges");
    writer.StartArray();
    for (const SourceRange &range : operation.argument_ranges) {
      writer.StartObject();
      writer.Key("begin");
      writer.Uint64(range.begin);
      writer.Key("end");
      writer.Uint64(range.end);
      writer.EndObject();
    }
    writer.EndArray();
    writer.Key("output");
    writeMatrixValue(writer, operation.output);
    writer.Key("operands");
    writer.StartArray();
    for (const MatrixValue &operand : operation.operands) {
      writeMatrixValue(writer, operand);
    }
    writer.EndArray();
    writer.Key("alias_requirements");
    writer.StartArray();
    writer.StartObject();
    writer.Key("relation");
    writer.String("no_alias");
    writer.Key("between");
    writer.StartArray();
    writer.String("output");
    writer.String("lhs");
    writer.EndArray();
    writer.EndObject();
    writer.StartObject();
    writer.Key("relation");
    writer.String("no_alias");
    writer.Key("between");
    writer.StartArray();
    writer.String("output");
    writer.String("rhs");
    writer.EndArray();
    writer.EndObject();
    writer.EndArray();
    writer.Key("effects");
    writer.StartObject();
    writer.Key("reads");
    writer.StartArray();
    writer.String("lhs");
    writer.String("rhs");
    writer.EndArray();
    writer.Key("writes");
    writer.StartArray();
    writer.String("output");
    writer.EndArray();
    writer.Key("synchronization");
    writer.String("synchronous");
    writer.EndObject();
    writer.Key("policy");
    writer.StartObject();
    writer.Key("target");
    writeString(writer, operation.target);
    writer.Key("fallback");
    writeString(writer, operation.fallback);
    writer.EndObject();
    writer.EndObject();
  }
  writer.EndArray();
  writer.EndObject();

  return std::string(buffer.GetString(), buffer.GetSize()) + '\n';
}

bool parseAndVerifyJson(std::string_view json, Module &module,
                        std::string &error) {
  module = Module{};
  error.clear();
  rapidjson::Document document;
  document.Parse(json.data(), json.size());
  if (document.HasParseError()) {
    error = "malformed JSON at byte " +
            std::to_string(document.GetErrorOffset()) + ": " +
            rapidjson::GetParseError_En(document.GetParseError());
    return false;
  }
  if (!document.IsObject()) {
    error = "Matcore IR root must be a JSON object";
    return false;
  }
  std::string schema;
  if (!readString(document, "schema", schema, error) || schema != "matcore.ir") {
    if (error.empty()) {
      error = "unsupported Matcore IR schema: " + schema;
    }
    return false;
  }
  const rapidjson::Value *version =
      document.HasMember("version") ? &document["version"] : nullptr;
  if (version == nullptr || !version->IsUint() ||
      version->GetUint() != kMatcoreIrVersion) {
    error = "unsupported Matcore IR version; expected version 0";
    return false;
  }
  if (!readString(document, "producer", module.producer, error)) {
    return false;
  }
  const rapidjson::Value *translation_unit = requiredMember(
      document, "translation_unit", rapidjson::kObjectType, error);
  if (translation_unit == nullptr ||
      !readString(*translation_unit, "identity", module.translation_unit,
                  error) ||
      !readString(*translation_unit, "source_file", module.source_file,
                  error)) {
    return false;
  }
  const rapidjson::Value *operations =
      requiredMember(document, "operations", rapidjson::kArrayType, error);
  if (operations == nullptr) {
    return false;
  }

  for (const rapidjson::Value &encoded : operations->GetArray()) {
    if (!encoded.IsObject()) {
      error = "Matcore IR operation must be an object";
      return false;
    }
    Operation operation;
    if (!readString(encoded, "site_id", operation.site_id, error) ||
        !readString(encoded, "kind", operation.kind, error) ||
        !readString(encoded, "canonical_callee", operation.canonical_callee,
                    error)) {
      return false;
    }
    const rapidjson::Value *source =
        requiredMember(encoded, "source", rapidjson::kObjectType, error);
    if (source == nullptr ||
        !readString(*source, "file", operation.source.file, error) ||
        !readUint64(*source, "byte_offset", operation.source.offset, error)) {
      return false;
    }
    std::uint64_t line = 0;
    std::uint64_t column = 0;
    if (!readUint64(*source, "line", line, error) ||
        !readUint64(*source, "column", column, error) ||
        line > UINT32_MAX || column > UINT32_MAX) {
      if (error.empty()) {
        error = "source line or column exceeds the v0 range";
      }
      return false;
    }
    operation.source.line = static_cast<std::uint32_t>(line);
    operation.source.column = static_cast<std::uint32_t>(column);
    const rapidjson::Value *call_range =
        requiredMember(*source, "byte_range", rapidjson::kObjectType, error);
    if (call_range == nullptr ||
        !parseRange(*call_range, operation.call_range, error)) {
      return false;
    }
    const rapidjson::Value *argument_ranges = requiredMember(
        encoded, "source_argument_ranges", rapidjson::kArrayType, error);
    if (argument_ranges == nullptr) {
      return false;
    }
    for (const rapidjson::Value &range : argument_ranges->GetArray()) {
      SourceRange parsed;
      if (!parseRange(range, parsed, error)) {
        return false;
      }
      operation.argument_ranges.push_back(parsed);
    }
    const rapidjson::Value *output =
        requiredMember(encoded, "output", rapidjson::kObjectType, error);
    if (output == nullptr || !parseMatrix(*output, operation.output, error)) {
      return false;
    }
    const rapidjson::Value *operands =
        requiredMember(encoded, "operands", rapidjson::kArrayType, error);
    if (operands == nullptr) {
      return false;
    }
    for (const rapidjson::Value &operand : operands->GetArray()) {
      MatrixValue parsed;
      if (!parseMatrix(operand, parsed, error)) {
        return false;
      }
      operation.operands.push_back(std::move(parsed));
    }
    const rapidjson::Value *policy =
        requiredMember(encoded, "policy", rapidjson::kObjectType, error);
    if (policy == nullptr ||
        !readString(*policy, "target", operation.target, error) ||
        !readString(*policy, "fallback", operation.fallback, error)) {
      return false;
    }
    module.operations.push_back(std::move(operation));
  }
  if (!verify(module, error)) {
    error = "Matcore IR verifier rejected input: " + error;
    module = Module{};
    return false;
  }
  return true;
}

} // namespace matcore::mdslc::ir
