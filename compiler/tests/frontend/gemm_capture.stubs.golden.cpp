#include "gemm_capture.sites.h"
#include <matcore/runtime_c.h>

#include <cstdint>
#include <stdexcept>
#include <string>

extern "C" matcore_status_v0 matcore_generated_backend_mc_d02b3f5b625dc51ed089ace811832a03_v0(const matcore_tensor_desc_v0 *, const matcore_tensor_desc_v0 *, const matcore_tensor_desc_v0 *, const matcore_policy_v0 *) noexcept;

namespace {
matcore_tensor_desc_v0 make_tensor_descriptor(
    const matcore::mdsl::matrix_view &view,
    matcore_mutability_v0 mutability) noexcept {
  matcore_tensor_desc_v0 descriptor{};
  descriptor.abi_version = MATCORE_RUNTIME_ABI_VERSION_V0;
  descriptor.struct_size =
      static_cast<std::uint32_t>(sizeof(matcore_tensor_desc_v0));
  descriptor.data = view.data;
  descriptor.dtype = MATCORE_DTYPE_F32_V0;
  descriptor.rank = 2;
  descriptor.dims[0] = view.rows;
  descriptor.dims[1] = view.columns;
  descriptor.strides[0] = view.columns;
  descriptor.strides[1] = 1;
  descriptor.memory_space = MATCORE_MEMORY_SPACE_HOST_V0;
  descriptor.mutability = mutability;
  return descriptor;
}

matcore_policy_v0 make_policy(
    matcore::mdsl::policy execution_policy) noexcept {
  matcore_policy_v0 policy{};
  policy.abi_version = MATCORE_RUNTIME_ABI_VERSION_V0;
  policy.struct_size =
      static_cast<std::uint32_t>(sizeof(matcore_policy_v0));
  switch (execution_policy.target) {
  case matcore::mdsl::target::cpu:
    policy.target = MATCORE_TARGET_CPU_V0;
    break;
  case matcore::mdsl::target::cuda:
    policy.target = MATCORE_TARGET_CUDA_V0;
    break;
  default:
    policy.target = MATCORE_TARGET_INVALID_V0;
    break;
  }
  switch (execution_policy.fallback) {
  case matcore::mdsl::fallback::error:
    policy.fallback = MATCORE_FALLBACK_ERROR_V0;
    break;
  default:
    policy.fallback = MATCORE_FALLBACK_INVALID_V0;
    break;
  }
  return policy;
}

void throw_on_error(const matcore_status_v0 &status,
                    const char *source_location) {
  if (status.code != MATCORE_STATUS_OK_V0) {
    const char *message = status.message != nullptr
                              ? status.message
                              : "Matcore runtime failure";
    throw std::runtime_error(std::string(source_location) +
                             ": " + message);
  }
}
} // namespace

void __matcore_call_site_mc_d02b3f5b625dc51ed089ace811832a03(::matcore::mdsl::out_arg output, const ::matcore::mdsl::matrix_view &lhs, const ::matcore::mdsl::matrix_view &rhs, ::matcore::mdsl::policy execution_policy) {
  const ::matcore::mdsl::matrix_view empty_output{};
  const ::matcore::mdsl::matrix_view &output_view =
      output.value != nullptr ? *output.value : empty_output;
  matcore_tensor_desc_v0 output_descriptor =
      make_tensor_descriptor(output_view, MATCORE_MUTABILITY_READ_WRITE_V0);
  matcore_tensor_desc_v0 lhs_descriptor =
      make_tensor_descriptor(lhs, MATCORE_MUTABILITY_READ_ONLY_V0);
  matcore_tensor_desc_v0 rhs_descriptor =
      make_tensor_descriptor(rhs, MATCORE_MUTABILITY_READ_ONLY_V0);
  matcore_policy_v0 runtime_policy = make_policy(execution_policy);
  const matcore_status_v0 status = matcore_generated_backend_mc_d02b3f5b625dc51ed089ace811832a03_v0(&output_descriptor, &lhs_descriptor, &rhs_descriptor, &runtime_policy);
  throw_on_error(status, "compiler/tests/frontend/gemm_capture.mdsl:21:3");
}
