#include "matcore_ir.h"

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
    writer.EndObject();
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

} // namespace matcore::mdslc::ir
