#ifndef MATCORE_MDSLC_MLIR_REGION_BOUNDARY_OPS_H
#define MATCORE_MDSLC_MLIR_REGION_BOUNDARY_OPS_H

#include "mlir/IR/Types.h"
#include "llvm/ADT/StringRef.h"

namespace matcore::mdslc::mlir_dialect {

// A reference to an admitted source matrix_view binding. Distinct values may
// refer to overlapping physical storage. This is neither a memref ABI nor a
// proof of descriptor validity, writability, residency or disjointness.
class RegionDescriptorType
    : public mlir::Type::TypeBase<RegionDescriptorType, mlir::Type,
                                  mlir::TypeStorage> {
public:
  using Base::Base;
  static constexpr llvm::StringLiteral name = "mdsl.region_descriptor";
};

// An internal sequencing dependency for the two-call inspection. It encodes
// the observable validation/commit order, not an executable runtime token.
class RegionOrderType
    : public mlir::Type::TypeBase<RegionOrderType, mlir::Type,
                                  mlir::TypeStorage> {
public:
  using Base::Base;
  static constexpr llvm::StringLiteral name = "mdsl.region_order";
};

} // namespace matcore::mdslc::mlir_dialect
#endif
