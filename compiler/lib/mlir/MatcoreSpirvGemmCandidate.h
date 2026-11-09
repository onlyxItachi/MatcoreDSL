#ifndef MATCORE_MDSLC_MLIR_SPIRV_GEMM_CANDIDATE_H
#define MATCORE_MDSLC_MLIR_SPIRV_GEMM_CANDIDATE_H

#include "MatcoreCpuGemmCandidate.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace matcore::mdslc::spirv_candidate {

using StaticGemmShapeV1 = cpu_candidate::StaticGemmShapeV1;
inline constexpr std::int64_t kMaxDispatchAxisV1 = 65535;
inline constexpr std::uint64_t kMaxStorageBufferBytesV1 = 128ULL * 1024 * 1024;
inline constexpr std::uint64_t kMaxStorageBufferElementsV1 =
    kMaxStorageBufferBytesV1 / 4;
inline constexpr char kEntryPointV1[] = "__matcore_strict_gemm_f32_v1_kernel";

// Device-independent issuance envelope, NOT device availability/qualification.
// M/N fit Vulkan 1.2 guaranteed dispatch limits; each array fits the guaranteed
// storage-buffer range and signed i32 element/byte arithmetic. K-loop's next IV
// must fit signed i32. Division-based tests reject before any product overflow.
bool validateStaticGemmShapeV1(StaticGemmShapeV1 shape, std::string &error);

enum class BufferRoleV1 { Lhs, Rhs, Output };
struct BufferBindingV1 {
  BufferRoleV1 role;
  std::uint32_t descriptor_set, binding;
  std::uint64_t bytes;
  std::uint32_t element_stride = 4, member_offset = 0;
  bool writable;
  bool operator==(const BufferBindingV1 &) const = default;
};
struct ShaderAbiV1 {
  std::string entry_point;
  std::array<std::uint32_t, 3> local_size, dispatch;
  std::vector<BufferBindingV1> buffers;
  bool operator==(const ShaderAbiV1 &) const = default;
};
struct StrictGemmArtifactV1 {
  StaticGemmShapeV1 shape{0, 0, 0};
  std::string semantic_ir, structured_ir, bufferized_ir, outlined_ir;
  std::string fill_spirv_ir, gemm_spirv_ir;
  std::vector<std::uint32_t> fill_words, gemm_words;
  ShaderAbiV1 fill_abi, gemm_abi;
  std::string manifest, error;
  explicit operator bool() const {
    return error.empty() && !fill_words.empty() && !gemm_words.empty();
  }
};

// Closed compiler-private issuer. Only checked shape is input; there is no IR,
// shader, callback, manifest or certificate import. Uses the existing strict
// primitive, One-Shot, upstream Linalg/GPU/SPIR-V lowering and serialization.
// Two dispatches are ordered fill -> GEMM with an external memory dependency.
// The ABI is inspection data, not a runtime/candidate/source authorization.
StrictGemmArtifactV1 issueStrictGemmArtifactV1(mlir::MLIRContext &context,
                                              StaticGemmShapeV1 shape);

// Inspection-only complete replay. Reissue from the checked shape and compare
// every stage/word/ABI field/manifest, not opcode counts or metadata alone.
// Rejects valid but foreign graphs as well as altered constants, CFG/indices,
// storage layout, binding, float controls and launch geometry. Never grants
// authority to supplied/serialized artifacts or executes a shader.
bool verifyStrictGemmArtifactV1(mlir::MLIRContext &context,
                               const StrictGemmArtifactV1 &artifact,
                               std::string &error);
// Independent structural inspection, supplementary to complete replay. These
// verify exact scalar reduction/address/ABI invariants; they are not issuers.
bool verifyStrictGemmOutlinedV1(mlir::ModuleOp module, StaticGemmShapeV1 shape,
                               std::string &error);
bool verifyStrictGemmShaderV1(mlir::spirv::ModuleOp module,
                             StaticGemmShapeV1 shape, bool fill,
                             std::string &error);

} // namespace matcore::mdslc::spirv_candidate
#endif
