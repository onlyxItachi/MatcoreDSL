#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "matcore/kernel_ir.h"
#include "matcore/target_registry.h"

namespace matcore {

struct DiskCacheArtifacts {
  std::filesystem::path root_dir;
  std::filesystem::path artifact_dir;
  std::filesystem::path shared_object_path;
  std::filesystem::path object_path;
};

inline constexpr std::string_view kDiskCacheVersion =
    "matcore-phase4-cache-v9-family-c-block-coop";

std::string buildExecutionCacheKey(
    const KernelIR &kernel, const RequestedTargetProfile &target,
    const std::vector<RuntimeTensorView> &tensors,
    const std::optional<std::string_view> &x86_cache_tag,
    bool graph_mode = false);

DiskCacheArtifacts buildDiskCacheArtifacts(const std::string &cache_key);
bool isDiskCacheSupported(const RequestedTargetProfile &target);
void ensureCacheDirectory(const std::filesystem::path &dir);

}  // namespace matcore
