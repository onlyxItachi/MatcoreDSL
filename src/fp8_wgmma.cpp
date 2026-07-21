#include "fp8_wgmma.h"

#include <algorithm>
#include <array>
#include <cstdint>

namespace matcore {
namespace {

bool isLegalFp8WgmmaShapeImpl(int M, int N, int K) {
  if (M <= 0 || N <= 0 || K <= 0) {
    return false;
  }
  if ((M % 64) != 0) {
    return false;
  }
  if ((K % 32) != 0) {
    return false;
  }
  if (N < 8 || N > 256 || (N % 8) != 0) {
    return false;
  }
  return true;
}

bool isFp8Operand(TensorDType dtype) {
  return dtype == TensorDType::kFloat8E4M3FN;
}

int chooseLegalN(int n) {
  constexpr std::array<int, 32> kLegalN = {
      8,   16,  24,  32,  40,  48,  56,  64,  72,  80,  88,
      96,  104, 112, 120, 128, 136, 144, 152, 160, 168, 176,
      184, 192, 200, 208, 216, 224, 232, 240, 248, 256};
  if (n <= 0) {
    return 128;
  }
  if (n >= 128 && (n % 128) == 0) {
    return 128;
  }

  int best_divisor = 0;
  for (int legal : kLegalN) {
    if (legal > n) {
      break;
    }
    if ((n % legal) == 0) {
      best_divisor = legal;
    }
  }
  if (best_divisor > 0) {
    return best_divisor;
  }

  int clamped = std::min(256, std::max(8, n));
  return ((clamped + 7) / 8) * 8;
}

}  // namespace

bool isEligibleForFp8Wgmma(const MatmulLoweringSignature &sig) {
  if (!isFp8Operand(sig.lhs_dtype) || !isFp8Operand(sig.rhs_dtype)) {
    return false;
  }
  if (sig.out_dtype != TensorDType::kFloat32) {
    return false;
  }
  if (sig.quantized_i8) {
    return false;
  }
  if (normalizeTarget(sig.target_kind) != TargetKind::kNvidiaDGPU) {
    return false;
  }
  if (sig.nvidia_sm_major < 9) {
    return false;
  }
  if (!isLegalFp8WgmmaShapeImpl(sig.matmul_m, sig.matmul_n, sig.matmul_k)) {
    return false;
  }
  return true;
}

bool isLegalFp8WgmmaShape(int M, int N, int K) {
  return isLegalFp8WgmmaShapeImpl(M, N, K);
}

Fp8WgmmaConfig getFp8WgmmaTileConfig(int M, int N, int K) {
  Fp8WgmmaConfig config;
  if (M <= 0 || N <= 0 || K <= 0) {
    config.M_tile = 64;
    config.N_tile = 128;
    config.K_tile = 32;
    config.use_tma = false;
    return config;
  }
  if (M > 0 && M >= 128 && (M % 128) == 0) {
    config.M_tile = 128;
  } else {
    config.M_tile = 64;
  }
  config.N_tile = chooseLegalN(N);
  config.K_tile = 32;
  const std::int64_t workload =
      static_cast<std::int64_t>(std::max(1, M)) * std::max(1, N) *
      std::max(1, K);
  config.use_tma = isLegalFp8WgmmaShapeImpl(M, N, K) &&
                   workload >= static_cast<std::int64_t>(64) * 64 * 32;
  return config;
}

}  // namespace matcore
