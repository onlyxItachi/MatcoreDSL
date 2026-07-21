#pragma once

#include "matcore/lowering_pipeline.h"

namespace matcore {

struct Fp8WgmmaConfig {
  int M_tile = 64;   // Must be multiple of 64
  int N_tile = 128;  // From allowed set
  int K_tile = 32;   // FP8 K step
  bool use_tma = true;
  // f32 accumulation only in MLIR 18
};

bool isEligibleForFp8Wgmma(const MatmulLoweringSignature &sig);
bool isLegalFp8WgmmaShape(int M, int N, int K);
Fp8WgmmaConfig getFp8WgmmaTileConfig(int M, int N, int K);

}  // namespace matcore
