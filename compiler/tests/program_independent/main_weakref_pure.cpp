#include "api.h"

// Force the canonical declaration to be materialized before the alias call.
// Clang then retains clean function attributes on the real symbol while the
// alias contributes its false effect promise only to the call instruction.
md::Result clean_reference(md::Storage a, md::Storage b, md::Storage c,
                           md::Shape m, md::Shape k, md::Shape n) noexcept {
  return first(a, b, c, m, k, n);
}
static md::Result alternate(md::Storage, md::Storage, md::Storage,
                            md::Shape, md::Shape, md::Shape) noexcept
    __attribute__((weakref("_Z5firstN7matcore4mdsl7StorageES1_S1_yyy"), pure));
#define first alternate
#include "main_body.h"
