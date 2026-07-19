#ifndef MATCORE_TEST_HEADER_INLINE_GEMM_H
#define MATCORE_TEST_HEADER_INLINE_GEMM_H

#include <matcore/mdsl.h>

inline void header_run_gemm(matcore::mdsl::matrix_view &c,
                            const matcore::mdsl::matrix_view &a,
                            const matcore::mdsl::matrix_view &b) {
  matcore::mdsl::gemm(matcore::mdsl::out(c), a, b);
}

#endif
