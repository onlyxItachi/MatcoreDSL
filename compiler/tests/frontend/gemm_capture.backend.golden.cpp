#include <matcore/runtime_c.h>

extern "C" matcore_status_v0 matcore_generated_backend_mc_2ed9c564471bcc43_v0(const matcore_tensor_desc_v0 *output,
    const matcore_tensor_desc_v0 *lhs,
    const matcore_tensor_desc_v0 *rhs,
    const matcore_policy_v0 *policy) noexcept {
  return matcore_runtime_gemm_f32_v0(output, lhs, rhs, policy);
}
