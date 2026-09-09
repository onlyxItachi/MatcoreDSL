#include "api.h"
MATCORE_REGION md::Result not_selected(md::Storage a, md::Storage c) noexcept {
  auto value = md::read(a, 1, 1);
  md::publish(value, c);
  return md::complete();
}
#include "main_body.h"
