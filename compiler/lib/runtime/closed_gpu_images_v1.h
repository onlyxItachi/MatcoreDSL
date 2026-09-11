#ifndef MATCORE_MDSLC_CLOSED_GPU_IMAGES_V1_H
#define MATCORE_MDSLC_CLOSED_GPU_IMAGES_V1_H

#include <cstddef>

// Compiler-generated definitions are linked privately into the candidate DSO.
// No runtime/source API accepts replacement bytes, paths, callbacks or images.
namespace matcore::mdslc::runtime::closed_host_v1::detail {
extern const unsigned char mdslc_nvvm_fill_image_v1[];
extern const std::size_t mdslc_nvvm_fill_image_v1_size;
extern const unsigned char mdslc_nvvm_gemm_image_v1[];
extern const std::size_t mdslc_nvvm_gemm_image_v1_size;
extern const unsigned char mdslc_rocdl_fill_image_v1[];
extern const std::size_t mdslc_rocdl_fill_image_v1_size;
extern const unsigned char mdslc_rocdl_gemm_image_v1[];
extern const std::size_t mdslc_rocdl_gemm_image_v1_size;
inline constexpr char kGpuStrictGemmKernelV1[] = "__matcore_strict_gemm_f32_v1_kernel";
} // namespace matcore::mdslc::runtime::closed_host_v1::detail
#endif
