// MDSLC generated global call-site declarations.
namespace matcore::mdsl {
struct matrix_view;
struct out_arg;
struct policy;
} // namespace matcore::mdsl

void __matcore_call_site_mc_3c5b6d5e7992fb7b249de44210c6415d(::matcore::mdsl::out_arg output, const ::matcore::mdsl::matrix_view &lhs, const ::matcore::mdsl::matrix_view &rhs, ::matcore::mdsl::policy execution_policy);

#line 1 "compiler/tests/frontend/gemm_capture.mdsl"
#include <matcore/mdsl.h>

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

  ::__matcore_call_site_mc_3c5b6d5e7992fb7b249de44210c6415d(md::out(C), A, B,
           md::policy{.target = md::target::cpu,
                      .fallback = md::fallback::error});
}
