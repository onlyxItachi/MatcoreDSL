#include "matcore/target_registry.h"

#include <cctype>
#include <stdexcept>

namespace matcore {
namespace {

[[noreturn]] void fail(const std::string &message) {
  throw std::runtime_error("MatCore target registry: " + message);
}

std::string toLower(std::string_view text) {
  std::string out;
  out.reserve(text.size());
  for (char c : text) {
    out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
  }
  return out;
}

std::string trim(std::string_view text) {
  std::size_t start = 0;
  while (start < text.size() &&
         std::isspace(static_cast<unsigned char>(text[start])) != 0) {
    ++start;
  }
  std::size_t end = text.size();
  while (end > start &&
         std::isspace(static_cast<unsigned char>(text[end - 1])) != 0) {
    --end;
  }
  return std::string(text.substr(start, end - start));
}

bool startsWith(std::string_view text, std::string_view prefix) {
  return text.size() >= prefix.size() &&
         text.substr(0, prefix.size()) == prefix;
}

TargetKind parseBaseTarget(std::string_view base) {
  const std::string normalized = toLower(base);

  if (normalized == "x86-auto" || normalized == "x86auto" || normalized == "x86") {
    return TargetKind::kX86Auto;
  }
  if (normalized == "x86-avx2" || normalized == "x86_avx2") {
    return TargetKind::kX86AVX2;
  }
  if (normalized == "x86-avx512" || normalized == "x86_avx512") {
    return TargetKind::kX86AVX512;
  }
  if (normalized == "nvidia-dgpu" || normalized == "nvidia_dgpu" ||
      normalized == "nvptx") {
    return TargetKind::kNvidiaDGPU;
  }
  if (normalized == "amd-igpu" || normalized == "amd_igpu" ||
      normalized == "amdgcn") {
    return TargetKind::kAmdIGPU;
  }
  if (normalized == "amd-npu" || normalized == "amd_npu" ||
      normalized == "npu") {
    return TargetKind::kAmdNPU;
  }
  if (normalized == "arm") {
    return TargetKind::kARM;
  }
  if (normalized == "tpu") {
    return TargetKind::kTPU;
  }

  if (startsWith(normalized, "sm_") || startsWith(normalized, "sm")) {
    return TargetKind::kNvidiaDGPU;
  }

  fail("unsupported base target '" + std::string(base) + "'");
}

std::optional<std::pair<int, int>> parseNvidiaSmToken(std::string_view token) {
  std::string normalized = toLower(trim(token));
  if (normalized.empty()) {
    return std::nullopt;
  }

  if (startsWith(normalized, "compute_")) {
    normalized = normalized.substr(8);
  } else if (startsWith(normalized, "sm_")) {
    normalized = normalized.substr(3);
  } else if (startsWith(normalized, "sm")) {
    normalized = normalized.substr(2);
  } else {
    return std::nullopt;
  }

  std::size_t digit_count = 0;
  while (digit_count < normalized.size() &&
         std::isdigit(static_cast<unsigned char>(normalized[digit_count])) != 0) {
    ++digit_count;
  }
  if (digit_count < 2) {
    fail("invalid NVIDIA SM token '" + std::string(token) + "'");
  }

  const std::string digits = normalized.substr(0, digit_count);
  const int major = std::stoi(digits.substr(0, digits.size() - 1));
  const int minor = std::stoi(digits.substr(digits.size() - 1, 1));
  return std::make_pair(major, minor);
}

std::string splitBase(std::string_view target, std::string *suffix_out) {
  std::string normalized = trim(target);
  if (normalized.empty()) {
    fail("empty target string");
  }

  const std::size_t split = normalized.find_first_of(":@/");
  if (split == std::string::npos) {
    *suffix_out = "";
    return normalized;
  }

  *suffix_out = normalized.substr(split + 1);
  return normalized.substr(0, split);
}

std::string canonicalTargetName(TargetKind kind) {
  switch (normalizeTarget(kind)) {
    case TargetKind::kX86Auto:
      return "x86-auto";
    case TargetKind::kX86AVX2:
      return "x86-avx2";
    case TargetKind::kX86AVX512:
      return "x86-avx512";
    case TargetKind::kNvidiaDGPU:
      return "nvidia-dgpu";
    case TargetKind::kAmdIGPU:
      return "amd-igpu";
    case TargetKind::kAmdNPU:
      return "amd-npu";
    case TargetKind::kARM:
      return "arm";
    case TargetKind::kTPU:
      return "tpu";
    case TargetKind::kNVPTX:
    case TargetKind::kAMDGCN:
    case TargetKind::kNPU:
      break;
  }
  return "unknown";
}

}  // namespace

RequestedTargetProfile ParseRequestedTargetProfile(std::string_view requested_target) {
  RequestedTargetProfile profile;
  profile.requested = trim(requested_target);

  std::string suffix;
  const std::string base = splitBase(profile.requested, &suffix);
  profile.kind = normalizeTarget(parseBaseTarget(base));

  if (profile.kind == TargetKind::kNvidiaDGPU) {
    std::optional<std::pair<int, int>> sm = parseNvidiaSmToken(base);
    if (!sm.has_value()) {
      sm = parseNvidiaSmToken(suffix);
    }
    if (sm.has_value()) {
      profile.nvidia_sm_major = sm->first;
      profile.nvidia_sm_minor = sm->second;
    }
  }

  profile.canonical = CanonicalTargetString(profile);
  return profile;
}

ExecutionRequirements BuildExecutionRequirements(
    const RequestedTargetProfile &profile) {
  ExecutionRequirements requirements;
  requirements.kind = normalizeTarget(profile.kind);

  switch (requirements.kind) {
    case TargetKind::kX86AVX2:
      requirements.required_x86_features.push_back("avx2");
      requirements.required_x86_features.push_back("fma");
      requirements.required_x86_features.push_back("f16c");
      break;
    case TargetKind::kX86AVX512:
      requirements.required_x86_features.push_back("avx512f");
      requirements.required_x86_features.push_back("avx512bw");
      requirements.required_x86_features.push_back("avx512vl");
      requirements.required_x86_features.push_back("avx512dq");
      requirements.required_x86_features.push_back("avx512cd");
      requirements.required_x86_features.push_back("avx2");
      requirements.required_x86_features.push_back("fma");
      requirements.required_x86_features.push_back("f16c");
      break;
    case TargetKind::kNvidiaDGPU:
      requirements.requires_nvidia_device = true;
      if (profile.nvidia_sm_major.has_value() && profile.nvidia_sm_minor.has_value()) {
        requirements.min_nvidia_sm_major = profile.nvidia_sm_major;
        requirements.min_nvidia_sm_minor = profile.nvidia_sm_minor;
      }
      break;
    case TargetKind::kAmdIGPU:
      requirements.requires_rocm_runtime = true;
      break;
    case TargetKind::kAmdNPU:
      requirements.requires_npu_runtime = true;
      break;
    case TargetKind::kX86Auto:
    case TargetKind::kARM:
    case TargetKind::kTPU:
    case TargetKind::kNVPTX:
    case TargetKind::kAMDGCN:
    case TargetKind::kNPU:
      break;
  }

  return requirements;
}

std::string CanonicalTargetString(const RequestedTargetProfile &profile) {
  std::string canonical = canonicalTargetName(profile.kind);
  if (normalizeTarget(profile.kind) == TargetKind::kNvidiaDGPU &&
      profile.nvidia_sm_major.has_value() && profile.nvidia_sm_minor.has_value()) {
    canonical += ":sm_";
    canonical += std::to_string(*profile.nvidia_sm_major);
    canonical += std::to_string(*profile.nvidia_sm_minor);
  }
  return canonical;
}

}  // namespace matcore
