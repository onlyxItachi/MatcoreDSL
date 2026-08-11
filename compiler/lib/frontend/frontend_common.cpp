#include "frontend.h"

#include <filesystem>
#include <iomanip>
#include <sstream>
#include <string_view>
#include <vector>

namespace matcore::mdslc::frontend {

std::string stableSourceIdentity(const std::string &canonical_input) {
  const std::filesystem::path input(canonical_input);
  std::filesystem::path directory = input.parent_path();
  std::error_code error;
  while (!directory.empty()) {
    if (std::filesystem::exists(directory / ".git", error) && !error) {
      const std::filesystem::path relative =
          std::filesystem::relative(input, directory, error);
      if (!error && !relative.empty()) {
        return relative.lexically_normal().generic_string();
      }
    }
    error.clear();
    const std::filesystem::path parent = directory.parent_path();
    if (parent == directory) {
      break;
    }
    directory = parent;
  }
  return input.lexically_normal().generic_string();
}

namespace {

std::string escapedInspectionValue(std::string_view value) {
  std::ostringstream output;
  output << '"';
  for (const unsigned char character : value) {
    switch (character) {
    case '\\':
      output << "\\\\";
      break;
    case '"':
      output << "\\\"";
      break;
    case '\n':
      output << "\\n";
      break;
    case '\r':
      output << "\\r";
      break;
    case '\t':
      output << "\\t";
      break;
    default:
      if (character < 0x20U || character == 0x7fU) {
        output << "\\x" << std::hex << std::setfill('0') << std::setw(2)
               << static_cast<unsigned>(character) << std::dec;
      } else {
        output << static_cast<char>(character);
      }
      break;
    }
  }
  output << '"';
  return output.str();
}

std::filesystem::path normalizedPath(const std::string &path) {
  std::error_code error;
  std::filesystem::path absolute = std::filesystem::absolute(path, error);
  if (error) {
    absolute = path;
  }
  error.clear();
  const std::filesystem::path canonical =
      std::filesystem::weakly_canonical(absolute, error);
  return error ? absolute.lexically_normal() : canonical;
}

bool samePath(const std::string &left, const std::filesystem::path &right) {
  return normalizedPath(left) == right;
}

} // namespace

std::string_view recoveredGemmStateName(RecoveredGemmState state) {
  switch (state) {
  case RecoveredGemmState::not_recognized:
    return "not_recognized";
  case RecoveredGemmState::recognized_rejected:
    return "recognized_rejected";
  case RecoveredGemmState::recognized_guard_required:
    return "recognized_guard_required";
  case RecoveredGemmState::raised:
    return "raised";
  }
  return "invalid";
}

std::string serializeRecoveredGemmInspection(const Result &result) {
  std::ostringstream output;
  output << "matcore.recovered-gemm-inspection-v1\n";
  output << "producer=clang-libtooling-v1\n";
  output << "candidate_count=" << result.recovered_gemm_candidates.size()
         << '\n';
  for (std::size_t index = 0;
       index < result.recovered_gemm_candidates.size(); ++index) {
    const RecoveredGemmCandidate &candidate =
        result.recovered_gemm_candidates[index];
    const std::string prefix = "candidate[" + std::to_string(index) + "].";
    output << prefix << "state=" << recoveredGemmStateName(candidate.state)
           << '\n';
    output << prefix << "pattern="
           << escapedInspectionValue(candidate.pattern) << '\n';
    output << prefix << "site_id="
           << escapedInspectionValue(candidate.site_id) << '\n';
    output << prefix << "source_file="
           << escapedInspectionValue(candidate.source_file) << '\n';
    output << prefix << "source_identity="
           << escapedInspectionValue(candidate.source_identity) << '\n';
    output << prefix << "compilation_identity="
           << escapedInspectionValue(candidate.compilation_identity) << '\n';
    output << prefix << "source_snapshot_sha256="
           << escapedInspectionValue(candidate.source_snapshot_sha256) << '\n';
    output << prefix << "function="
           << escapedInspectionValue(candidate.function_name) << '\n';
    output << prefix << "binding.output="
           << escapedInspectionValue(candidate.output_parameter) << '\n';
    output << prefix << "binding.lhs="
           << escapedInspectionValue(candidate.lhs_parameter) << '\n';
    output << prefix << "binding.rhs="
           << escapedInspectionValue(candidate.rhs_parameter) << '\n';
    output << prefix << "binding.m="
           << escapedInspectionValue(candidate.m_parameter) << '\n';
    output << prefix << "binding.n="
           << escapedInspectionValue(candidate.n_parameter) << '\n';
    output << prefix << "binding.k="
           << escapedInspectionValue(candidate.k_parameter) << '\n';
    output << prefix
           << "semantic_contract=f32_row_major_overwrite_m_k__k_n__m_n\n";
    output << prefix << "line=" << candidate.line << '\n';
    output << prefix << "column=" << candidate.column << '\n';
    output << prefix << "offset=" << candidate.offset << '\n';
    output << prefix << "outer_loop_range="
           << candidate.outer_loop_range.begin << ':'
           << candidate.outer_loop_range.end << '\n';
    output << prefix << "proof_range_count=" << candidate.proof_ranges.size()
           << '\n';
    for (std::size_t range_index = 0;
         range_index < candidate.proof_ranges.size(); ++range_index) {
      const RecoveredNamedRange &range = candidate.proof_ranges[range_index];
      output << prefix << "proof_range[" << range_index << "]="
             << escapedInspectionValue(range.role) << ':' << range.range.begin
             << ':' << range.range.end << '\n';
    }
    output << prefix << "fp.allow_reassociation="
           << candidate.fp_proof.allow_reassociation << '\n';
    output << prefix << "fp.contract_across_statement="
           << candidate.fp_proof.contract_across_statement << '\n';
    output << prefix << "fp.honor_nans=" << candidate.fp_proof.honor_nans
           << '\n';
    output << prefix << "fp.honor_infinities="
           << candidate.fp_proof.honor_infinities << '\n';
    output << prefix << "fp.preserve_signed_zero="
           << candidate.fp_proof.preserve_signed_zero << '\n';
    output << prefix << "fp.allow_reciprocal="
           << candidate.fp_proof.allow_reciprocal << '\n';
    output << prefix << "fp.allow_approximate_functions="
           << candidate.fp_proof.allow_approximate_functions << '\n';
    output << prefix << "fp.fenv_access="
           << candidate.fp_proof.fenv_access << '\n';
    output << prefix << "fp.fast_math_profile="
           << candidate.fp_proof.fast_math_profile << '\n';
    output << prefix << "fp.rounding_mode="
           << escapedInspectionValue(candidate.fp_proof.rounding_mode) << '\n';
    output << prefix << "fp.exception_mode="
           << escapedInspectionValue(candidate.fp_proof.exception_mode) << '\n';
    output << prefix << "fp.denormal_mode="
           << escapedInspectionValue(candidate.fp_proof.denormal_mode) << '\n';
    output << prefix << "fp.fp32_denormal_mode="
           << escapedInspectionValue(candidate.fp_proof.fp32_denormal_mode)
           << '\n';
    output << prefix << "fp.optimization_level="
           << candidate.fp_proof.optimization_level << '\n';
    output << prefix << "required_guard_count="
           << candidate.required_runtime_guards.size() << '\n';
    for (std::size_t guard_index = 0;
         guard_index < candidate.required_runtime_guards.size(); ++guard_index) {
      output << prefix << "required_guard[" << guard_index << "]="
             << escapedInspectionValue(
                    candidate.required_runtime_guards[guard_index])
             << '\n';
    }
    output << prefix << "rejection_reason_count="
           << candidate.rejection_reasons.size() << '\n';
    for (std::size_t reason_index = 0;
         reason_index < candidate.rejection_reasons.size(); ++reason_index) {
      output << prefix << "rejection_reason[" << reason_index << "]="
             << escapedInspectionValue(candidate.rejection_reasons[reason_index])
             << '\n';
    }
    output << prefix << "rewrite=preserve_original_cpp\n";
  }
  return output.str();
}

std::string stableCompilationIdentity(const Options &options) {
  const std::filesystem::path input = normalizedPath(options.input_path);
  const std::filesystem::path source_directory = input.parent_path();
  std::vector<std::filesystem::path> trusted_include_directories;
  trusted_include_directories.reserve(options.trusted_public_headers.size());
  for (const std::string &header : options.trusted_public_headers) {
    trusted_include_directories.push_back(normalizedPath(header).parent_path()
                                              .parent_path());
  }

  const auto is_implicit_include = [&](const std::string &path) {
    const std::filesystem::path normalized = normalizedPath(path);
    if (normalized == source_directory) {
      return true;
    }
    for (const std::filesystem::path &trusted : trusted_include_directories) {
      if (normalized == trusted) {
        return true;
      }
    }
    return false;
  };

  std::vector<std::string> semantic_arguments;
  for (std::size_t index = 0; index < options.compiler_arguments.size();
       ++index) {
    const std::string &argument = options.compiler_arguments[index];
    if (!argument.starts_with('-') && samePath(argument, input)) {
      continue;
    }
    if (argument == "-x") {
      if (index + 1 < options.compiler_arguments.size()) {
        ++index;
      }
      continue;
    }
    if (argument == "-fsyntax-only" ||
        argument == "-fno-color-diagnostics" || argument == "-std=c++20") {
      continue;
    }
    if ((argument == "-I" || argument == "-iquote") &&
        index + 1 < options.compiler_arguments.size() &&
        is_implicit_include(options.compiler_arguments[index + 1])) {
      ++index;
      continue;
    }
    if (argument.starts_with("-I") && argument.size() > 2 &&
        is_implicit_include(argument.substr(2))) {
      continue;
    }
    semantic_arguments.push_back(argument);
  }

  std::ostringstream identity;
  for (const std::string &argument : semantic_arguments) {
    identity << argument.size() << ':' << argument << ';';
  }
  return identity.str();
}

std::string makeStableSiteId(std::string_view source_identity,
                             std::string_view compilation_identity,
                             std::string_view source, std::uint64_t offset,
                             std::string_view kind) {
  std::uint64_t left = 14695981039346656037ULL;
  std::uint64_t right = 7809847782465536322ULL;
  const auto append = [&left, &right](std::string_view value) {
    for (const unsigned char character : value) {
      left ^= character;
      left *= 1099511628211ULL;
      right ^= static_cast<unsigned char>(character + 0x9dU);
      right *= 14029467366897019727ULL;
    }
    left ^= 0xffU;
    left *= 1099511628211ULL;
    right ^= 0x5aU;
    right *= 14029467366897019727ULL;
  };
  append("matcore-site-v0");
  append(source_identity);
  if (!compilation_identity.empty()) {
    append(compilation_identity);
  }
  append(source);
  append(std::to_string(offset));
  append(kind);
  std::ostringstream formatted;
  formatted << "mc_" << std::hex << std::setfill('0') << std::setw(16) << left
            << std::setw(16) << right;
  return formatted.str();
}

} // namespace matcore::mdslc::frontend
