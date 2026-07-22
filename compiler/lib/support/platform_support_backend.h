#ifndef MATCORE_MDSLC_SUPPORT_PLATFORM_SUPPORT_BACKEND_H
#define MATCORE_MDSLC_SUPPORT_PLATFORM_SUPPORT_BACKEND_H

#include "platform_support.h"

namespace matcore::mdslc::support::detail {

std::optional<std::filesystem::path> current_executable_path_native_v1(
    std::string &error);
ProcessResultV1 run_process_native_v1(const ProcessRequestV1 &request);
std::optional<std::filesystem::path> create_temp_directory_native_v1(
    std::string_view prefix, std::string &error);
std::optional<std::string> environment_utf8_native_v1(std::string_view name,
                                                      std::string &error);
std::optional<std::filesystem::path> find_executable_native_v1(
    std::string_view name, std::string &error);
FileIdentityV1 file_identity_native_v1(const std::filesystem::path &path,
                                       std::string &error);

}  // namespace matcore::mdslc::support::detail

#endif
