#pragma once
namespace library {
template<class Tag, class T> T product(T a, T b) {
  return md::gemm(b, a, md::Numerics::strict_f32);
}
}
