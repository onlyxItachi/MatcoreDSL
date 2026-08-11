#ifndef MATCORE_MDSLC_MLIR_MATCORE_OPS_H
#define MATCORE_MDSLC_MLIR_MATCORE_OPS_H

#include "MatcoreDialect.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "MatcoreOps.h.inc"

#endif // MATCORE_MDSLC_MLIR_MATCORE_OPS_H
