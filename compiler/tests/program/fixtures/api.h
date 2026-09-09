#pragma once
#include <matcore/region.h>
namespace md = matcore::mdsl;
md::Result first(md::Storage A, md::Storage B, md::Storage C,
                 md::Shape M, md::Shape K, md::Shape N) noexcept;
md::Result second(md::Storage A, md::Storage B, md::Storage C,
                  md::Shape M, md::Shape K, md::Shape N) noexcept;
int region_unit_identity();
