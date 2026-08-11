#ifndef MATCORE_MDSLC_MLIR_MATCORE_V1_BRIDGE_H
#define MATCORE_MDSLC_MLIR_MATCORE_V1_BRIDGE_H

#include "matcore_ir_v1.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

#include <cstdint>
#include <optional>
#include <string>

namespace matcore::mdslc::mlir_bridge {

inline constexpr std::uint32_t kMatcoreSemanticModuleVersion = 1;
inline constexpr char kExplicitGemmF32Profile[] = "explicit-gemm-f32-v1";

enum class ReassociationSemantics {
  Invalid,
  Forbidden,
  WithinReduction,
};

enum class ContractionSemantics { Invalid, Forbidden, Allowed };

enum class ReductionOrderSemantics {
  Invalid,
  IncreasingK,
  ImplementationDefinedWithinK,
};

enum class NaNSemantics {
  Invalid,
  Strict,
  PreserveClassificationPayloadOrderUnspecified,
};

enum class SignedZeroSemantics { Invalid, Preserve, Relaxed };
enum class RoundingSemantics { Invalid, NearestTiesEven };
enum class TrappingExceptionSemantics { Invalid, Unsupported };
enum class ExceptionStatusSemantics {
  Invalid,
  IncomingNotPreservedPostCallUnspecified,
};
enum class SubnormalSemantics { Invalid, IeeeGradualFtzDazForbidden };
enum class Permission { Invalid, Forbidden, Allowed };

struct NumericalSemantics {
  std::optional<ir::v1::DType> accumulation_dtype;
  ReassociationSemantics reassociation = ReassociationSemantics::Invalid;
  ContractionSemantics contraction = ContractionSemantics::Invalid;
  ReductionOrderSemantics reduction_order =
      ReductionOrderSemantics::Invalid;
  NaNSemantics nan = NaNSemantics::Invalid;
  SignedZeroSemantics signed_zero = SignedZeroSemantics::Invalid;
  RoundingSemantics rounding = RoundingSemantics::Invalid;
  TrappingExceptionSemantics trapping_exceptions =
      TrappingExceptionSemantics::Invalid;
  ExceptionStatusSemantics exception_status =
      ExceptionStatusSemantics::Invalid;
  SubnormalSemantics subnormals = SubnormalSemantics::Invalid;
  Permission approximate_math = Permission::Invalid;
  Permission inplace = Permission::Invalid;
};

struct BridgeContext {
  // A caller must name and supply a complete reviewed profile. An empty or
  // anonymous context is invalid and never acquires permissions from a target.
  // Encoding the profile does not prove that a runtime honors its floating-
  // point environment; execution lowering must validate that separately.
  std::string numerical_profile;
  NumericalSemantics numerical;
};

BridgeContext explicitGemmF32V1BridgeContext();

struct BridgeResult {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  std::string error;

  explicit operator bool() const { return static_cast<bool>(module); }
};

// Registers only the dialects required by the semantic bridge and its text.
void registerMatcoreSemanticDialects(mlir::MLIRContext &context);

// Converts one already verified capture module into one independent function
// per source site. Dynamic symbols remain operation-local. No numerical policy
// is inferred: the explicit BridgeContext is mandatory and validated.
BridgeResult bridgeV1ToMatcoreMlir(const ir::v1::Module &source,
                                   mlir::MLIRContext &context,
                                   const BridgeContext &bridge_context);

// Verifies module-level bridge structure in addition to normal MLIR/dialect
// verification. This is used after construction and after textual parsing.
bool verifyMatcoreSemanticModule(mlir::ModuleOp module, std::string &error);

// Stable inspection text with debug locations and exactly one trailing LF.
std::string serializeDeterministicMlir(mlir::ModuleOp module);

} // namespace matcore::mdslc::mlir_bridge

#endif // MATCORE_MDSLC_MLIR_MATCORE_V1_BRIDGE_H
