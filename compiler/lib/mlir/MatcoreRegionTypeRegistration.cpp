#include "MatcoreDialect.h"
#include "MatcoreRegionBoundaryOps.h"

namespace matcore::mdslc::mlir_dialect {

// Keep only singleton type registration in this translation unit. Upstream
// registration instantiates LLVM's header-defined BumpPtrAllocator. Its manual
// ASan poison/unpoison protocol must match the pinned, non-ASan prebuilt MLIR
// library whose inline allocation path shares the same allocator and weak
// template symbols. The build keeps UBSan here and ASan on every semantic
// operation, parser, verifier, builder and test. No runtime semantics live here.
void MatcoreDialect::registerRegionTypes() {
  addTypes<RegionDescriptorType, RegionOrderType>();
}

} // namespace matcore::mdslc::mlir_dialect
