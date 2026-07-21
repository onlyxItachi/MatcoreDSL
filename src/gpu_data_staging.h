#ifndef MATCORE_GPU_DATA_STAGING_H_
#define MATCORE_GPU_DATA_STAGING_H_

#include <memory>

#include "mlir/Pass/Pass.h"

namespace matcore {

/// Creates a pass that stages host-side memref data to GPU device memory
/// around gpu.launch operations.
///
/// For each gpu.launch, the pass:
///   1. Identifies all memref values used inside the launch that are defined
///      outside (i.e., host-resident memory captured by the kernel).
///   2. Inserts gpu.alloc + gpu.memcpy (host → device) before the launch.
///   3. Replaces uses of host memrefs inside the launch with device counterparts.
///   4. Inserts gpu.memcpy (device → host) + gpu.dealloc after the launch for
///      memrefs that may have been written by the kernel.
///
/// This pass MUST run after the gpu.launch has been created
/// (e.g., after DynamicMacroGridMappingPass and thread mapping) but BEFORE
/// gpuKernelOutlining (e.g., before buildLowerToNVVMPassPipeline).
std::unique_ptr<mlir::Pass> CreateGpuDataStagingPass();

}  // namespace matcore

#endif  // MATCORE_GPU_DATA_STAGING_H_
