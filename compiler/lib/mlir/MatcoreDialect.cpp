#include "MatcoreDialect.h"

#include "MatcoreOps.h"
#include "MatcoreRegionBoundaryOps.h"

#include "mlir/IR/DialectImplementation.h"

#include "MatcoreDialect.cpp.inc"

namespace matcore::mdslc::mlir_dialect {

void MatcoreDialect::initialize() {
  addTypes<RegionDescriptorType, RegionOrderType>();
  addOperations<
#define GET_OP_LIST
#include "MatcoreOps.cpp.inc"
      >();
}

mlir::Type MatcoreDialect::parseType(mlir::DialectAsmParser &parser) const {
  llvm::StringRef name;
  if (parser.parseKeyword(&name))
    return {};
  if (name == "region_descriptor")
    return RegionDescriptorType::get(getContext());
  if (name == "region_order")
    return RegionOrderType::get(getContext());
  parser.emitError(parser.getCurrentLocation(), "unknown Matcore type: ")
      << name;
  return {};
}

void MatcoreDialect::printType(mlir::Type type,
                              mlir::DialectAsmPrinter &printer) const {
  if (mlir::isa<RegionDescriptorType>(type))
    printer << "region_descriptor";
  else if (mlir::isa<RegionOrderType>(type))
    printer << "region_order";
  else
    llvm_unreachable("unregistered Matcore type");
}

} // namespace matcore::mdslc::mlir_dialect
