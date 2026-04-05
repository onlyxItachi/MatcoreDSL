#pragma once

#include <string>

#include "mlir/Pass/PassManager.h"

#include "matcore/lowering_pipeline.h"

namespace matcore {

struct AmdGpuConfig {
  std::string chip;
  int wavefront_size = 64;
  bool has_mfma = false;
  bool has_wmma = false;
  int lds_size_kb = 64;
};

AmdGpuConfig detectAmdGpuConfig(const std::string &chip);
std::string amdIneligibilityReason(const MatmulLoweringSignature &sig,
                                   const AmdGpuConfig &config);
bool isEligibleForAmdGpu(const MatmulLoweringSignature &sig,
                         const AmdGpuConfig &config);
void configureAmdLoweringPipeline(mlir::PassManager &pm,
                                  const AmdGpuConfig &config);

}  // namespace matcore
