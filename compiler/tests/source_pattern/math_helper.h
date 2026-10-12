#pragma once
#include <matcore/region.h>

// A real included mathematical body. Its physical inclusion must belong to
// the frozen source closure; it is not a caller-supplied IR certificate.
namespace source_pattern_math {
inline matcore::mdsl::Value product(matcore::mdsl::Value a,
                                   matcore::mdsl::Value b) noexcept {
  return matcore::mdsl::gemm(a, b, matcore::mdsl::Numerics::strict_f32);
}
}
