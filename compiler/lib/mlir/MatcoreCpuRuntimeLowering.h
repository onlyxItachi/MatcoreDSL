#ifndef MATCORE_MDSLC_MLIR_CPU_RUNTIME_LOWERING_H
#define MATCORE_MDSLC_MLIR_CPU_RUNTIME_LOWERING_H

#include "mlir/IR/BuiltinOps.h"

#include <cstdint>
#include <string>
#include <vector>

namespace matcore::mdslc::mlir_lowering {

inline constexpr std::uint32_t kCpuRuntimeDispatchRecordVersionV1 = 1;
inline constexpr char kCpuRuntimeDispatchSymbolV1[] =
    "matcore_runtime_gemm_f32_v0";

// Short-lived internal handoff from a verified semantic module to generated C
// ABI source. This is not a serialized IR or a public/installable ABI type.
struct CpuRuntimeDispatchRecordV1 {
  std::uint32_t version = kCpuRuntimeDispatchRecordVersionV1;
  std::string site_id;
  std::string semantic_symbol;
  std::string operation;
  std::string dtype;
  std::string accumulation_dtype;
  std::string target;
  std::string fallback;
  std::string numerical_profile;
  std::string execution_intent;
  std::string runtime_symbol;
  std::vector<std::string> required_guards;
  std::string source_file;
  std::uint64_t source_line = 0;
  std::uint64_t source_column = 0;
  std::uint64_t source_range_begin = 0;
  std::uint64_t source_range_end = 0;
  std::int64_t static_m = 0;
  std::int64_t static_n = 0;
  std::int64_t static_k = 0;
  std::uint32_t alignment = 4;
  bool no_alias = true;
};

// Accepts only the exact explicit Matcore IR v1 bridge envelope. Recovered
// forms, map/domain composition, altered numerical policy, and malformed
// destination/effect contracts fail closed. `records` is cleared on failure.
bool lowerExplicitGemmToCpuRuntimeDispatchV1(
    mlir::ModuleOp module, std::vector<CpuRuntimeDispatchRecordV1> &records,
    std::string &error);

} // namespace matcore::mdslc::mlir_lowering

#endif // MATCORE_MDSLC_MLIR_CPU_RUNTIME_LOWERING_H
