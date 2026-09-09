#pragma once
namespace library {
inline md::Value product(md::Value a, md::Value b) {
  return md::gemm(a, b, md::Numerics::strict_f32);
}
}
