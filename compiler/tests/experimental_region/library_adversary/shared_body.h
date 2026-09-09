// Deliberately no include guard: included in two distinct namespaces.
// Presumed diagnostic coordinates must not become physical source authority.
#line 700 "forged-main.mdsl"
inline md::Value product(md::Value a, md::Value b) {
  return md::gemm(a, b, md::Numerics::strict_f32);
}
inline md::Value keep_after_product(md::Value a, md::Value b, md::Value old) {
  auto unused = product(a, b);
  return old;
}
inline md::Shape passthrough(md::Shape value) { return value; }
