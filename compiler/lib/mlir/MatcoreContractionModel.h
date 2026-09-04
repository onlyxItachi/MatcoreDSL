#ifndef MATCORE_MDSLC_MLIR_CONTRACTION_MODEL_H
#define MATCORE_MDSLC_MLIR_CONTRACTION_MODEL_H

#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <string>

namespace matcore::mdslc::mlir_bridge {

// This model is an internal structured-compiler contract, not a public Matcore
// operation surface or a serialized interchange commitment. It describes only
// logical index topology. Destination update behavior, numerical permission,
// provenance, effects, and execution authority remain separate contracts.
inline constexpr std::uint32_t kContractionTopologyVersionV1 = 1;
inline constexpr char kContractionTopologySchemaV1[] =
    "matcore-contraction-topology-v1";

// Established BLAS/HPC operation identities are retained even when several
// operations use the same map carrier. In particular, GER is an outer-product
// update, not a reduction contraction disguised as GEMM with K == 1.
enum class StandardLinearAlgebraOperationV1 {
  Gemm,
  Gemv,
  Dot,
  Ger,
  BatchedGemm,
};

enum class MatrixOrientationV1 {
  Normal,
  Transpose,
};

enum class BilinearTopologyClassV1 {
  ReductionContraction,
  OuterProductUpdate,
};

struct ContractionTopologyV1 {
  StandardLinearAlgebraOperationV1 operation =
      StandardLinearAlgebraOperationV1::Gemm;
  MatrixOrientationV1 lhs_orientation = MatrixOrientationV1::Normal;
  MatrixOrientationV1 rhs_orientation = MatrixOrientationV1::Normal;
  BilinearTopologyClassV1 topology_class =
      BilinearTopologyClassV1::ReductionContraction;
  llvm::SmallVector<std::string, 4> loop_dimensions;
  llvm::SmallVector<mlir::utils::IteratorType, 4> iterator_types;
  llvm::SmallVector<mlir::AffineMap, 3> indexing_maps;
  llvm::SmallVector<unsigned, 3> operand_ranks;
  llvm::SmallVector<std::string, 1> reduction_dimensions;
};

struct ContractionTopologyResultV1 {
  ContractionTopologyV1 topology;
  std::string error;
  bool valid = false;

  explicit operator bool() const { return valid; }
};

llvm::StringRef
standardLinearAlgebraOperationNameV1(StandardLinearAlgebraOperationV1 value);
llvm::StringRef matrixOrientationNameV1(MatrixOrientationV1 value);
llvm::StringRef bilinearTopologyClassNameV1(BilinearTopologyClassV1 value);

// Returns the exact canonical logical maps for a standard operation. The maps
// are extent-neutral: static, dynamic, unit, and zero extents do not change
// index topology. Whether an extent is admitted by a source language contract
// is deliberately outside this model.
ContractionTopologyResultV1 buildCanonicalContractionTopologyV1(
    mlir::MLIRContext &context, StandardLinearAlgebraOperationV1 operation,
    MatrixOrientationV1 lhs_orientation = MatrixOrientationV1::Normal,
    MatrixOrientationV1 rhs_orientation = MatrixOrientationV1::Normal);

// Rejects any topology that is not byte-for-byte equivalent to the canonical
// model for its declared standard operation and orientations.
bool verifyCanonicalContractionTopologyV1(
    const ContractionTopologyV1 &topology, std::string &error);

// Exact internal attribute encoding used by proof-carrying structured stages.
// Encoding and decoding both fail closed unless the supplied topology is the
// canonical model in the builder/context. Decode reconstructs and compares
// with that model rather than accepting arbitrary artifact-supplied maps.
mlir::DictionaryAttr encodeContractionTopologyV1(
    mlir::Builder &builder, const ContractionTopologyV1 &topology,
    std::string &error);
ContractionTopologyResultV1 decodeContractionTopologyV1(
    mlir::DictionaryAttr attribute, mlir::MLIRContext &context);

// Shared mechanical check for an upstream structured operation. Operand ranks
// are ordered lhs, rhs, destination/result. This does not validate the scalar
// combiner or destination dataflow; operation-specific verifiers retain those
// semantic responsibilities.
bool verifyStructuredIndexingAgainstContractionTopologyV1(
    const ContractionTopologyV1 &topology,
    llvm::ArrayRef<mlir::AffineMap> indexing_maps,
    llvm::ArrayRef<mlir::utils::IteratorType> iterator_types,
    llvm::ArrayRef<unsigned> operand_ranks, std::string &error);

} // namespace matcore::mdslc::mlir_bridge

#endif // MATCORE_MDSLC_MLIR_CONTRACTION_MODEL_H
