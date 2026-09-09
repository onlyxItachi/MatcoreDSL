#pragma once
namespace library {
md::Storage hidden;
template<> md::Value product<int, md::Value>(md::Value a, md::Value b) {
  md::observe(hidden);
  return md::gemm(a, b, md::Numerics::strict_f32);
}
}
