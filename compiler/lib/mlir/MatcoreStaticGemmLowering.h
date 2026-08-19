#ifndef MATCORE_MDSLC_MLIR_STATIC_GEMM_LOWERING_H
#define MATCORE_MDSLC_MLIR_STATIC_GEMM_LOWERING_H

#include "mlir/IR/BuiltinOps.h"

#include <cstdint>
#include <string>
#include <vector>

namespace matcore::mdslc::mlir_lowering {

inline constexpr std::uint32_t kStaticGemmSpecializationRecordVersionV1 = 1;

struct StaticGemmSpecializationRecordV1 {
  std::uint32_t version = kStaticGemmSpecializationRecordVersionV1;
  std::string site_id;
  std::string function_symbol;
  std::int64_t m = 0;
  std::int64_t n = 0;
  std::int64_t k = 0;
  std::string dtype = "f32";
  std::string accumulation_dtype = "f32";
  std::uint32_t alignment = 4;
  bool no_alias = true;
  std::uint32_t vector_width = 8;
  std::string declaration_c;
  std::string definition_c;
  std::string source_file;
  std::uint64_t source_line = 0;
  std::uint64_t source_column = 0;
  std::uint64_t source_range_begin = 0;
  std::uint64_t source_range_end = 0;
};

// Lowers verified explicit Matcore IR v1 GEMM operations with static shape bounds
// to MLIR-native static specialized microkernel records.
// Returns false and populates `error` if the module contains invalid contracts,
// non-static bounds, or unsupported numerical/effect profiles.
bool lowerExplicitGemmToStaticSpecializationV1(
    mlir::ModuleOp module,
    std::vector<StaticGemmSpecializationRecordV1> &records,
    std::string &error);

} // namespace matcore::mdslc::mlir_lowering

#endif // MATCORE_MDSLC_MLIR_STATIC_GEMM_LOWERING_H
