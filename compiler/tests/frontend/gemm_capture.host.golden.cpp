#include <matcore/mdsl.h>
#include "gemm_capture.sites.h"
#line 2 "compiler/tests/frontend/gemm_capture.mdsl"

namespace md = matcore::mdsl;

template <class T>
T host_add(T lhs, T rhs) {
  return lhs + rhs;
}

void capture_gemm() {
  float a_storage[6]{};
  float b_storage[8]{};
  float c_storage[12]{};
  md::matrix_view A{a_storage, 2, 3};
  md::matrix_view B{b_storage, 3, 4};
  md::matrix_view C{c_storage, 2, 4};

  const int ordinary_host_value = host_add(2, 3);
  (void)ordinary_host_value;

  ::matcore::mdsl::detail::__matcore_call_site_mc_d02b3f5b625dc51ed089ace811832a03(md::out(C), A, B, md::policy{.target = md::target::cpu,
                      .fallback = md::fallback::error});
}
