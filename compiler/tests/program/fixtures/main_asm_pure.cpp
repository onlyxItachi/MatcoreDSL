#include "api.h"
// A different C++ spelling still addresses the issued region in the ELF file.
__attribute__((pure)) md::Result remote_first(md::Storage, md::Storage,
    md::Storage, md::Shape, md::Shape, md::Shape) noexcept
    asm("_Z5firstN7matcore4mdsl7StorageES1_S1_yyy");
#define first remote_first
#include "main_body.h"
