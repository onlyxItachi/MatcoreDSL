#include "api.h"
inline md::Result first(md::Storage A, md::Storage B, md::Storage C,
    md::Shape M, md::Shape K, md::Shape N) noexcept {
  return second(A, B, C, M, K, N);
}
#include "main_body.h"
