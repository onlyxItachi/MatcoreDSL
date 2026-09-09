#pragma once
#include <matcore/region.h>
namespace md = matcore::mdsl;
md::Result permissive_first(md::Storage a, md::Storage b, md::Storage c) noexcept;
md::Result strict_second(md::Storage a, md::Storage b, md::Storage c) noexcept;
