#pragma once
#define LIBRARY_BODY return md::gemm(a, b, md::Numerics::strict_f32)
namespace library {
template<> md::Value product<int, md::Value>(md::Value a, md::Value b) {
  LIBRARY_BODY;
}
}
#undef LIBRARY_BODY
