#ifndef MATCORE_TEST_HEADER_CALL_H
#define MATCORE_TEST_HEADER_CALL_H

#include <matcore/mdsl.h>

namespace md = matcore::mdsl;

inline void rejected_header_call(md::matrix_view &C,
                                 const md::matrix_view &A,
                                 const md::matrix_view &B) {
  md::gemm(md::out(C), A, B);
}

#endif
