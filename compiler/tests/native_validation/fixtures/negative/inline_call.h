#pragma once

namespace native_validation {
namespace md = matcore::mdsl;

inline void header_origin(md::matrix_view &C, const md::matrix_view &A,
                          const md::matrix_view &B) {
  md::gemm(md::out(C), A, B);
}
} // namespace native_validation
