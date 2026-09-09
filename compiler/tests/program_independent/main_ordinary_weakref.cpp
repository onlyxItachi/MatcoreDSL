#include "api.h"
static int ordinary_alias(int) noexcept
    __attribute__((weakref("_Z7utilityi"), pure));
#define region_unit_identity() ordinary_alias(0)
#include "main_body.h"
