#ifndef MATCORE_MDSL_H
#define MATCORE_MDSL_H

#include <cstdint>

#if defined(__clang__) && __has_attribute(annotate)
#define MATCORE_MDSL_ANNOTATE(TEXT) [[clang::annotate(TEXT)]]
#else
#define MATCORE_MDSL_ANNOTATE(TEXT)
#endif

namespace matcore::mdsl {

// Bootstrap v0 deliberately exposes one concrete matrix view. It denotes a
// host-resident, row-major, contiguous f32 matrix and owns no storage.
struct matrix_view {
  float *data = nullptr;
  std::int64_t rows = 0;
  std::int64_t columns = 0;
};

struct out_arg {
  matrix_view *value = nullptr;
};

MATCORE_MDSL_ANNOTATE("matcore.wrapper.out")
[[nodiscard]] constexpr out_arg out(matrix_view &value) noexcept {
  return out_arg{&value};
}

out_arg out(const matrix_view &) = delete;
out_arg out(matrix_view &&) = delete;

enum class target : std::uint32_t {
  cpu = 0,
  cuda = 1,
};

enum class fallback : std::uint32_t {
  error = 0,
};

struct policy {
  matcore::mdsl::target target = matcore::mdsl::target::cpu;
  matcore::mdsl::fallback fallback = matcore::mdsl::fallback::error;
};

// This declaration is a compiler interception point. A source containing it
// must be processed by mdslc++ before an ordinary final link.
MATCORE_MDSL_ANNOTATE("matcore.op.gemm")
void gemm(out_arg output, const matrix_view &lhs, const matrix_view &rhs,
          policy execution_policy = {});

} // namespace matcore::mdsl

#undef MATCORE_MDSL_ANNOTATE

#endif // MATCORE_MDSL_H
