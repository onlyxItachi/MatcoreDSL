// Private falsification vocabulary, not an installed or executable MDSL API.
// The frontend embeds these bytes; source supplied for inspection never loads
// this file from disk and cannot replace the injected declarations.
#ifndef MDSLC_INTERNAL_EMBED_CLOSED_REGION_FIXTURE
#error "This private MDSL fixture is inspection-only; ordinary compilation is unsupported."
#else
#ifndef MDSLC_CLOSED_REGION_FIXTURE_LANGUAGE_H
#define MDSLC_CLOSED_REGION_FIXTURE_LANGUAGE_H
namespace matcore::mdslc::frontend::detail {
inline constexpr char closed_region_fixture_source[] = R"mdsl_fixture(
namespace mdsl_probe {
struct Value { Value() = delete; };
struct Storage { Storage() = delete; };
using Shape = unsigned long long;
enum class Numerics { strict_f32, reassociate_f32 };
[[clang::annotate("mdsl.private.read.v1")]]
Value read(Storage, Shape, Shape) noexcept;
[[clang::annotate("mdsl.private.gemm.v1")]]
Value gemm(Value, Value, Numerics) noexcept;
[[clang::annotate("mdsl.private.publish.v1")]]
void publish(Value, Storage) noexcept;
[[clang::annotate("mdsl.private.observe.v1")]]
void observe(Storage) noexcept;
[[clang::annotate("mdsl.private.rows.v1")]]
Shape rows(Value) noexcept;
[[clang::annotate("mdsl.private.cols.v1")]]
Shape cols(Value) noexcept;
} // namespace mdsl_probe
)mdsl_fixture";
} // namespace matcore::mdslc::frontend::detail
#endif
#endif
