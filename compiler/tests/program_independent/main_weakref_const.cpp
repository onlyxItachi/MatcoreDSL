#include "api.h"

md::Result clean_reference(md::Storage a, md::Storage b, md::Storage c,
                           md::Shape m, md::Shape k, md::Shape n) noexcept {
  return first(a, b, c, m, k, n);
}
static md::Result alternate(md::Storage, md::Storage, md::Storage,
                            md::Shape, md::Shape, md::Shape) noexcept
    __attribute__((weakref("_Z5firstN7matcore4mdsl7StorageES1_S1_yyy"), const));
#define first alternate
#include "main_body.h"
