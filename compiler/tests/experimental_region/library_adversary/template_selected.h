#pragma once
namespace library {
template<> md::Value product<int, md::Value>(md::Value a, md::Value b) {
  return md::gemm(a, b, md::Numerics::strict_f32);
}
}
