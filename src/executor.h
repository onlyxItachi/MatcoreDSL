#pragma once

#include <vector>

#include "llvm/Support/Error.h"
#include "matcore/kernel_ir.h"

namespace matcore {

struct CachedExecution;

llvm::Error invokeCompiledKernel(const CachedExecution &compiled,
                                 const std::vector<RuntimeTensorView> &tensors);

}  // namespace matcore
