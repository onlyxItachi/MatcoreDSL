#include "MatcoreDialect.h"

#include "MatcoreOps.h"

#include "mlir/IR/DialectImplementation.h"

#include "MatcoreDialect.cpp.inc"

namespace matcore::mdslc::mlir_dialect {

void MatcoreDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "MatcoreOps.cpp.inc"
      >();
}

} // namespace matcore::mdslc::mlir_dialect
