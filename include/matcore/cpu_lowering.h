#pragma once

#include "mlir/Pass/PassManager.h"

#include "matcore/lowering_pipeline.h"

namespace matcore {

void configureCpuPassPipeline(mlir::PassManager &pm,
                              const MatmulLoweringSignature &signature);

}  // namespace matcore
