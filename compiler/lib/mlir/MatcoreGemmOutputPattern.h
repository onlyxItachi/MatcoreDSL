#ifndef MATCORE_MDSLC_MLIR_GEMM_OUTPUT_PATTERN_H
#define MATCORE_MDSLC_MLIR_GEMM_OUTPUT_PATTERN_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include <cstdint>
#include <string>

namespace matcore::mdslc::gemm_pattern {

// Compiler-private HOW parameters, not source semantics, runtime policy, or a
// public ABI. The admitted family tiles independent M/N outputs by 1..64 and
// retains the complete scalar, increasing K fold. There is no K parameter.
// {0,0} denotes absence only at the candidate-selection boundary; it is not a
// valid pattern. Target/ISA facts never enter this target-neutral carrier.
struct OutputTilePatternV1 {
  std::int64_t tile_m = 0;
  std::int64_t tile_n = 0;
};

bool validateOutputTilePatternV1(const OutputTilePatternV1 &tiles,
                                 std::string &error);

// Deterministic upstream Transform instantiation from checked integer values,
// never a caller-supplied schedule or a second loop/computation language.
std::string strictGemmOutputTileTransformV1(const OutputTilePatternV1 &tiles,
                                           std::string &error);

// Inspection/derivation only. The input must be the canonical strict buffered
// primitive; failed Transform mutations are confined to an isolated clone.
// Full positive-zero fill remains outside the output tiles. Only the inner
// independent output traversal changes to M/K/N; no padding, partial reduction,
// allocation, copy, publication, alias proof, or numerical permission is added.
// Generated SCF tile stepping assumes M <= INT64_MAX-(tile_m-1) and
// N <= INT64_MAX-(tile_n-1). The existing source adapter's positive output-byte
// bounds establish this; it never invokes a generated leaf for empty output.
mlir::OwningOpRef<mlir::ModuleOp> deriveStrictGemmOutputTiledV1(
    mlir::ModuleOp canonical_bufferized, const OutputTilePatternV1 &tiles,
    std::string &error);

// Exact parameter-bound upstream replay plus a narrow independent envelope of
// actual lowered scalar order/indexing. Not a general equivalence prover,
// importer, source authenticator, or authority to execute supplied IR.
bool verifyStrictGemmOutputTiledV1(mlir::ModuleOp scheduled,
                                  const OutputTilePatternV1 &tiles,
                                  std::string &error);

// Separate compiler-owned HOW recipe: upstream N full-tile/tail peeling of
// the same output decomposition. The original v1 recipe/graph is unchanged.
// A unit N tile statically selects that existing graph (no remainder exists);
// native loop.peel is not invoked and a failed peel is never suppressed.
std::string strictGemmOutputPeelNTransformV1(const OutputTilePatternV1 &tiles,
                                           std::string &error);
mlir::OwningOpRef<mlir::ModuleOp> deriveStrictGemmOutputPeelNV1(
    mlir::ModuleOp canonical_bufferized, const OutputTilePatternV1 &tiles,
    std::string &error);

// Branch-role proof of F = tile_n*floor(N/tile_n), full [0,F) and tail [F,N),
// bounded M, both complete increasing K folds and exact A/B/C scalar dataflow,
// plus parameter-bound upstream replay. The unit-N branch retains the exact
// one-body v1 envelope, not a permissive loop/arithmetic-count range.
// As above, dimensions are nonnegative valid dense private descriptors, with
// signed tile increments bounded by M <= INT64_MAX-(tile_m-1) and
// N <= INT64_MAX-(tile_n-1). F multiplication is representable because F<=N;
// the tail has width <tile_n and does not create padding or another K fold.
// Inspection is not source authentication or authority to execute supplied IR.
bool verifyStrictGemmOutputPeelNV1(mlir::ModuleOp scheduled,
                                  const OutputTilePatternV1 &tiles,
                                  std::string &error);

} // namespace matcore::mdslc::gemm_pattern
#endif
