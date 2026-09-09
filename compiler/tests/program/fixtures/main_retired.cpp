#include "api.h"
namespace library { md::Value product(md::Value, md::Value); }
md::Value (*escaped_value_helper)(md::Value, md::Value) = &library::product;
#include "main_body.h"
