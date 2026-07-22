#include "codegen.h"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <utility>

namespace matcore::mdslc::codegen {
namespace {

std::string siteFunction(const ir::Operation &operation) {
  return "__matcore_call_site_" + operation.site_id;
}

std::string backendFunction(const ir::Operation &operation) {
  return "matcore_generated_backend_" + operation.site_id + "_v0";
}

bool validIdentifier(std::string_view value) {
  if (value.empty() ||
      !(std::isalpha(static_cast<unsigned char>(value.front())) != 0 ||
        value.front() == '_')) {
    return false;
  }
  return std::all_of(value.begin() + 1, value.end(), [](const char character) {
    const unsigned char byte = static_cast<unsigned char>(character);
    return std::isalnum(byte) != 0 || character == '_';
  });
}

bool safeDirectiveText(std::string_view value) {
  return value.find('\n') == std::string_view::npos &&
         value.find('\r') == std::string_view::npos;
}

std::string escapeCppString(std::string_view value) {
  std::string escaped;
  escaped.reserve(value.size());
  for (const char character : value) {
    if (character == '\\' || character == '"') {
      escaped.push_back('\\');
    }
    escaped.push_back(character);
  }
  return escaped;
}

std::string declaration(const ir::Operation &operation, bool with_default) {
  std::ostringstream output;
  output << "void " << siteFunction(operation)
         << "(::matcore::mdsl::out_arg output, "
            "const ::matcore::mdsl::matrix_view &lhs, "
            "const ::matcore::mdsl::matrix_view &rhs, "
            "::matcore::mdsl::policy execution_policy";
  if (with_default) {
    output << " = {}";
  }
  output << ')';
  return output.str();
}

std::string generateSites(const ir::Module &module) {
  std::ostringstream output;
  output << "#pragma once\n\n"
            "#include <matcore/mdsl.h>\n\n";
  for (const ir::Operation &operation : module.operations) {
    output << declaration(operation, true) << ";\n";
  }
  return output.str();
}

std::string generateHostPreamble(const ir::Module &module) {
  std::ostringstream output;
  output << "// MDSLC generated global call-site declarations.\n"
            "namespace matcore::mdsl {\n"
            "struct matrix_view;\n"
            "struct out_arg;\n"
            "struct policy;\n"
            "} // namespace matcore::mdsl\n\n";
  for (const ir::Operation &operation : module.operations) {
    output << declaration(operation, false) << ";\n";
  }
  output << '\n';
  return output.str();
}

std::string generateStubs(const ir::Module &module,
                          std::string_view sites_include) {
  std::ostringstream output;
  output << "#include \"" << escapeCppString(sites_include) << "\"\n"
            "#include <matcore/runtime_c.h>\n";
  if (module.operations.empty()) {
    output << "\n// No Matcore call sites in this translation unit.\n";
    return output.str();
  }

  output << "\n#include <cstdint>\n"
            "#include <stdexcept>\n"
            "#include <string>\n\n"
            "#if defined(__clang__) || defined(__GNUC__)\n"
            "// Clang lowers this to ELF weak or COFF WeakExternal. The host,\n"
            "// stub, and backend objects must still be linked together.\n"
            "#define MATCORE_MDSLC_WEAK __attribute__((weak))\n"
            "#else\n"
            "#error \"MDSLC generated weak sites require Clang/GNU attributes\"\n"
            "#endif\n\n";
  for (const ir::Operation &operation : module.operations) {
    output << "extern \"C\" matcore_status_v0 "
           << backendFunction(operation)
           << "(const matcore_tensor_desc_v0 *, "
              "const matcore_tensor_desc_v0 *, "
              "const matcore_tensor_desc_v0 *, "
              "const matcore_policy_v0 *) noexcept;\n";
  }
  output << "\nnamespace {\n"
            "matcore_tensor_desc_v0 make_tensor_descriptor(\n"
            "    const matcore::mdsl::matrix_view &view,\n"
            "    matcore_mutability_v0 mutability) noexcept {\n"
            "  matcore_tensor_desc_v0 descriptor{};\n"
            "  descriptor.abi_version = MATCORE_RUNTIME_ABI_VERSION_V0;\n"
            "  descriptor.struct_size =\n"
            "      static_cast<std::uint32_t>(sizeof(matcore_tensor_desc_v0));\n"
            "  descriptor.data = view.data;\n"
            "  descriptor.dtype = MATCORE_DTYPE_F32_V0;\n"
            "  descriptor.rank = 2;\n"
            "  descriptor.dims[0] = view.rows;\n"
            "  descriptor.dims[1] = view.columns;\n"
            "  descriptor.strides[0] = view.columns;\n"
            "  descriptor.strides[1] = 1;\n"
            "  descriptor.memory_space = MATCORE_MEMORY_SPACE_HOST_V0;\n"
            "  descriptor.mutability = mutability;\n"
            "  return descriptor;\n"
            "}\n\n"
            "matcore_policy_v0 make_policy(\n"
            "    matcore::mdsl::policy execution_policy) noexcept {\n"
            "  matcore_policy_v0 policy{};\n"
            "  policy.abi_version = MATCORE_RUNTIME_ABI_VERSION_V0;\n"
            "  policy.struct_size =\n"
            "      static_cast<std::uint32_t>(sizeof(matcore_policy_v0));\n"
            "  switch (execution_policy.target) {\n"
            "  case matcore::mdsl::target::cpu:\n"
            "    policy.target = MATCORE_TARGET_CPU_V0;\n"
            "    break;\n"
            "  case matcore::mdsl::target::cuda:\n"
            "    policy.target = MATCORE_TARGET_CUDA_V0;\n"
            "    break;\n"
            "  default:\n"
            "    policy.target = MATCORE_TARGET_INVALID_V0;\n"
            "    break;\n"
            "  }\n"
            "  switch (execution_policy.fallback) {\n"
            "  case matcore::mdsl::fallback::error:\n"
            "    policy.fallback = MATCORE_FALLBACK_ERROR_V0;\n"
            "    break;\n"
            "  default:\n"
            "    policy.fallback = MATCORE_FALLBACK_INVALID_V0;\n"
            "    break;\n"
            "  }\n"
            "  return policy;\n"
            "}\n\n"
            "void throw_on_error(const matcore_status_v0 &status,\n"
            "                    const char *source_location) {\n"
            "  if (status.code != MATCORE_STATUS_OK_V0) {\n"
            "    const char *message = status.message != nullptr\n"
            "                              ? status.message\n"
            "                              : \"Matcore runtime failure\";\n"
            "    throw std::runtime_error(std::string(source_location) +\n"
            "                             \": \" + message);\n"
            "  }\n"
            "}\n"
            "} // namespace\n\n";

  for (const ir::Operation &operation : module.operations) {
    output << "MATCORE_MDSLC_WEAK " << declaration(operation, false)
           << " {\n"
              "  const ::matcore::mdsl::matrix_view empty_output{};\n"
              "  const ::matcore::mdsl::matrix_view &output_view =\n"
              "      output.value != nullptr ? *output.value : empty_output;\n"
              "  matcore_tensor_desc_v0 output_descriptor =\n"
              "      make_tensor_descriptor(output_view, "
              "MATCORE_MUTABILITY_READ_WRITE_V0);\n"
              "  matcore_tensor_desc_v0 lhs_descriptor =\n"
              "      make_tensor_descriptor(lhs, "
              "MATCORE_MUTABILITY_READ_ONLY_V0);\n"
              "  matcore_tensor_desc_v0 rhs_descriptor =\n"
              "      make_tensor_descriptor(rhs, "
              "MATCORE_MUTABILITY_READ_ONLY_V0);\n"
              "  matcore_policy_v0 runtime_policy = "
              "make_policy(execution_policy);\n"
              "  const matcore_status_v0 status = "
           << backendFunction(operation)
           << "(&output_descriptor, &lhs_descriptor, &rhs_descriptor, "
              "&runtime_policy);\n"
              "  throw_on_error(status, \""
           << escapeCppString(operation.source.file) << ':'
           << operation.source.line << ':' << operation.source.column
           << "\");\n"
              "}\n";
  }
  output << "\n#undef MATCORE_MDSLC_WEAK\n";
  return output.str();
}

std::string generateBackend(const ir::Module &module) {
  std::ostringstream output;
  output << "#include <matcore/runtime_c.h>\n";
  if (module.operations.empty()) {
    output << "\n// No Matcore backend entries in this translation unit.\n";
    return output.str();
  }
  output << "\n#if defined(__clang__) || defined(__GNUC__)\n"
            "// Clang emits a COFF WeakExternal on Windows and an ELF weak\n"
            "// definition on Linux. This is not a DLL ABI boundary.\n"
            "#define MATCORE_MDSLC_WEAK __attribute__((weak))\n"
            "#else\n"
            "#error \"MDSLC generated weak backends require Clang/GNU attributes\"\n"
            "#endif\n\n";
  for (const ir::Operation &operation : module.operations) {
    output << "extern \"C\" MATCORE_MDSLC_WEAK matcore_status_v0 "
           << backendFunction(operation)
           << "(const matcore_tensor_desc_v0 *output,\n"
              "    const matcore_tensor_desc_v0 *lhs,\n"
              "    const matcore_tensor_desc_v0 *rhs,\n"
              "    const matcore_policy_v0 *policy) noexcept {\n"
              "  return matcore_runtime_gemm_f32_v0(output, lhs, rhs, policy);\n"
              "}\n";
  }
  output << "\n#undef MATCORE_MDSLC_WEAK\n";
  return output.str();
}

} // namespace

bool generate(const ir::Module &module, std::string_view original_source,
              std::string_view sites_include, Artifacts &artifacts,
              std::string &error) {
  artifacts = Artifacts{};
  if (!ir::verify(module, error)) {
    error = "cannot generate from invalid Matcore IR: " + error;
    return false;
  }
  if (!safeDirectiveText(module.source_file) ||
      !safeDirectiveText(sites_include) || sites_include.empty()) {
    error = "source and generated include paths must be nonempty single-line text";
    return false;
  }
  for (const ir::Operation &operation : module.operations) {
    if (!validIdentifier(operation.site_id)) {
      error = "operation site ID is not a valid generated identifier";
      return false;
    }
    if (operation.call_range.end > original_source.size()) {
      error = "operation call range exceeds the original source buffer";
      return false;
    }
    for (const ir::SourceRange &range : operation.argument_ranges) {
      if (range.end > original_source.size()) {
        error = "operation argument range exceeds the original source buffer";
        return false;
      }
    }
  }

  std::string rewritten(original_source);
  for (auto iterator = module.operations.rbegin();
       iterator != module.operations.rend(); ++iterator) {
    const ir::Operation &operation = *iterator;
    std::ostringstream replacement;
    replacement << "::" << siteFunction(operation) << '(';
    const std::size_t call_begin =
        static_cast<std::size_t>(operation.call_range.begin);
    const std::size_t first_argument_begin =
        static_cast<std::size_t>(operation.argument_ranges.front().begin);
    const std::string_view discarded_call_prefix = original_source.substr(
        call_begin, first_argument_begin - call_begin);
    const std::size_t prefix_newlines = static_cast<std::size_t>(std::count(
        discarded_call_prefix.begin(), discarded_call_prefix.end(), '\n'));
    replacement << std::string(prefix_newlines, '\n');
    for (std::size_t index = 0; index < operation.argument_ranges.size();
         ++index) {
      const ir::SourceRange range = operation.argument_ranges[index];
      replacement << original_source.substr(
          static_cast<std::size_t>(range.begin),
          static_cast<std::size_t>(range.end - range.begin));
      if (index + 1 < operation.argument_ranges.size()) {
        const ir::SourceRange next = operation.argument_ranges[index + 1];
        replacement << original_source.substr(
            static_cast<std::size_t>(range.end),
            static_cast<std::size_t>(next.begin - range.end));
      }
    }
    if (operation.argument_ranges.size() == 3) {
      replacement << ", ::matcore::mdsl::policy{}";
    }
    const ir::SourceRange last_argument = operation.argument_ranges.back();
    replacement << original_source.substr(
        static_cast<std::size_t>(last_argument.end),
        static_cast<std::size_t>(operation.call_range.end -
                                 last_argument.end));
    std::string replacement_text = replacement.str();
    const std::string_view original_call = original_source.substr(
        static_cast<std::size_t>(operation.call_range.begin),
        static_cast<std::size_t>(operation.call_range.end -
                                 operation.call_range.begin));
    const std::size_t original_newlines = static_cast<std::size_t>(
        std::count(original_call.begin(), original_call.end(), '\n'));
    const std::size_t replacement_newlines = static_cast<std::size_t>(
        std::count(replacement_text.begin(), replacement_text.end(), '\n'));
    if (replacement_newlines > original_newlines) {
      error = "generated call-site replacement cannot preserve source line "
              "mapping";
      return false;
    }
    replacement_text.insert(replacement_text.size(),
                            original_newlines - replacement_newlines, '\n');
    rewritten.replace(
        static_cast<std::size_t>(operation.call_range.begin),
        static_cast<std::size_t>(operation.call_range.end -
                                 operation.call_range.begin),
        replacement_text);
  }

  if (module.operations.empty()) {
    artifacts.rewritten_host = std::move(rewritten);
  } else {
    const std::string preamble = generateHostPreamble(module);
    const std::string line_directive =
        "#line 1 \"" + escapeCppString(module.source_file) + "\"\n";
    constexpr std::string_view utf8_bom{"\xef\xbb\xbf", 3};
    if (rewritten.starts_with(utf8_bom)) {
      artifacts.rewritten_host = std::string(utf8_bom) + preamble +
                                 line_directive +
                                 rewritten.substr(utf8_bom.size());
    } else {
      artifacts.rewritten_host = preamble + line_directive + rewritten;
    }
  }
  artifacts.sites_header = generateSites(module);
  artifacts.stubs_source = generateStubs(module, sites_include);
  artifacts.backend_source = generateBackend(module);
  return true;
}

} // namespace matcore::mdslc::codegen
