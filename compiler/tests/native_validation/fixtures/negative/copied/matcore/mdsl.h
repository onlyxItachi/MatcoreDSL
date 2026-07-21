#pragma once

#include <cstdint>

namespace matcore::mdsl {
struct matrix_view {
  float *data;
  std::int64_t rows;
  std::int64_t columns;
};
struct out_arg {
  matrix_view *value;
};
[[clang::annotate("matcore.wrapper.out")]] inline out_arg
out(matrix_view &value) noexcept {
  return {&value};
}
enum class target : std::uint32_t { cpu, cuda };
enum class fallback : std::uint32_t { error };
struct policy {
  matcore::mdsl::target target = matcore::mdsl::target::cpu;
  matcore::mdsl::fallback fallback = matcore::mdsl::fallback::error;
};
[[clang::annotate("matcore.op.gemm")]] void
gemm(out_arg, const matrix_view &, const matrix_view &, policy = {});
} // namespace matcore::mdsl
