#pragma once
#define LIBRARY_VALUE md::Value
namespace library {
template<class Tag, class T> LIBRARY_VALUE product(LIBRARY_VALUE a, LIBRARY_VALUE b) {
  return md::gemm(b, a, md::Numerics::strict_f32);
}
}
#undef LIBRARY_VALUE
