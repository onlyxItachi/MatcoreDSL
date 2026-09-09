#include "ExperimentalRegionEmitter.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/SHA256.h"

#include <algorithm>
#include <map>
#include <sstream>

namespace matcore::mdslc::codegen {
namespace {
std::string literal(const std::string &bytes) {
  std::string result = "\"";
  for (const unsigned char byte : bytes) {
    if (byte >= 32 && byte <= 126 && byte != '\\' && byte != '"') {
      result += static_cast<char>(byte);
    } else {
      result += '\\';
      result += static_cast<char>('0' + (byte >> 6));
      result += static_cast<char>('0' + ((byte >> 3) & 7));
      result += static_cast<char>('0' + (byte & 7));
    }
  }
  return result + "\"";
}
const char *candidate(ClosedCpuPolicy policy) {
  switch (policy) {
  case ClosedCpuPolicy::Automatic: return "automatic";
  case ClosedCpuPolicy::NativeStrict: return "native_strict";
  case ClosedCpuPolicy::GeneratedStrict: return "generated_strict";
  case ClosedCpuPolicy::GeneratedReassociate: return "generated_reassociate";
  case ClosedCpuPolicy::ExistingNative: return "existing_native";
  case ClosedCpuPolicy::OpenBLAS: return "authenticated_openblas";
  }
  return nullptr;
}
} // namespace

ExperimentalRegionEmissionResult emitExperimentalRegion(
    const frontend::AuthenticatedClosedRegionEvidence &evidence,
    ClosedCpuPolicy policy) {
  ExperimentalRegionEmissionResult result;
  if (!evidence.hasHostContext() || !evidence.entryBinding() || !candidate(policy)) {
    result.error = "experimental implementation requires a sealed named entry and known CPU policy";
    return result;
  }
  // This replays the complete original host/semantic witness. No caller-provided
  // mutable graph, source range, symbol binding or certificate is accepted.
  auto closed = emitClosedHostV1(evidence);
  if (!closed) {
    result.error = closed.error;
    return result;
  }
  const auto &binding = *evidence.entryBinding();
  const auto &region = evidence.program().regions.front();
  if (binding.mangled_name.empty() || binding.signature_sha256.size() != 64 ||
      binding.parameters.size() != region.resources.size() + region.shape_parameters.size()) {
    result.error = "sealed named entry is missing its complete ABI/parameter binding";
    return result;
  }
  using Kind = frontend::ClosedRegionParameterBinding::Kind;
  std::map<std::uint64_t, Kind> kinds;
  for (const auto &resource : region.resources)
    kinds.emplace(resource.parameter_index, Kind::Storage);
  for (const auto &shape : region.shape_parameters)
    kinds.emplace(shape.parameter_index, Kind::Shape);
  for (std::size_t index = 0; index < binding.parameters.size(); ++index) {
    if (!kinds.contains(index) || kinds.at(index) != binding.parameters[index].kind) {
      result.error = "sealed ABI parameters differ from the verified semantic bindings";
      return result;
    }
  }

  ExperimentalRegionEmission emission;
  emission.contract = std::move(*closed.emission);
  emission.host_symbol = binding.mangled_name;
  const auto identity = emission.contract.semantic_sha256 + ":" +
                        binding.signature_sha256 + ":" + candidate(policy);
  emission.helper_symbol = "__matcore_region_" + llvm::toHex(
      llvm::SHA256::hash(llvm::arrayRefFromStringRef(identity)), true);
  // Helper removal names are copied from sealed frontend bindings, never
  // discovered by a pattern match in arbitrary LLVM IR.
  for (const auto &helper : binding.value_helpers)
    emission.retired_value_helpers.push_back(helper.mangled_name);

  std::ostringstream cpp;
  cpp << "#include <matcore/region.h>\n"
      << emission.contract.implementation
      // C linkage supplies only the deterministic internal symbol. Both sides
      // use the exact checked C++ return ABI; this is not a public C interface.
      << "#pragma clang diagnostic ignored \"-Wreturn-type-c-linkage\"\n"
      << "extern \"C\" ::matcore::mdsl::Result " << emission.helper_symbol << '(';
  for (std::size_t index = 0; index < binding.parameters.size(); ++index) {
    if (index) cpp << ", ";
    cpp << "::matcore::mdsl::"
        << (binding.parameters[index].kind == Kind::Storage ? "Storage" : "Shape")
        << " arg_" << index;
  }
  cpp << ") noexcept {\n"
      << "  namespace mch = ::matcore::mdslc::runtime::closed_host_v1;\n"
      << "  mch::Session session(mch::Options{mch::Candidate::" << candidate(policy) << "});\n"
      << "  const auto status = ::matcore::mdslc::generated_closed_host_v1::"
      << emission.contract.symbol << "(session";
  for (std::size_t index = 0; index < binding.parameters.size(); ++index) {
    const auto arg = "arg_" + std::to_string(index);
    cpp << ", ";
    if (binding.parameters[index].kind == Kind::Shape) {
      cpp << arg;
    } else {
      cpp << "mch::ResourceView{" << arg << ".data, " << arg << ".rows, "
          << arg << ".columns, " << arg << ".capacity_elements, " << arg << ".access}";
    }
  }
  cpp << ");\n"
      << "  ::matcore::mdsl::SourceLocation location{};\n"
      << "  switch (status.failed_frontier) {\n";
  const auto source_path = literal(evidence.program().source_identity);
  for (const auto &frontier : emission.contract.frontiers) {
    const auto &site = frontier.helper_calls.empty() ? frontier.source
                                                    : frontier.helper_calls.front();
    const auto file = literal(evidence.program().source_files.at(site.file_id - 1).path);
    cpp << "  case " << frontier.id << ": location = {" << file << ", "
        << site.line << ", " << site.column << "}; break;\n";
  }
  cpp << "  case " << emission.contract.completion_frontier << ": location = {"
      << source_path << ", " << binding.completion.line << ", "
      << binding.completion.column << "}; break;\n"
      << "  default: break;\n  }\n"
      << "  return static_cast<mch::Session&&>(session).takeResult(location);\n}\n";
  emission.helper_cpp = cpp.str();
  result.emission = std::move(emission);
  return result;
}
} // namespace matcore::mdslc::codegen
