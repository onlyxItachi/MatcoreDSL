#pragma once

#include <cstdint>
#include "matcore/kernel_ir.h"

namespace mlir {
class ExecutionEngine;
}

namespace matcore {

class ObservabilityContext;

void setGpuRuntimeObservabilityContext(ObservabilityContext *obs);
void registerGpuRuntimeSymbols(mlir::ExecutionEngine &engine, TargetKind target);

// V2 Pillar 2: CUDA Graph capture/replay helpers.
// These wrap the CUDA driver API graph calls with safety checks.
// Opaque handles — callers should not dereference.

/// Begin stream capture. Returns an opaque stream handle.
/// The stream must not already be capturing.
void *matcore_graph_stream_create();

/// Begin capturing all GPU operations on the given stream.
void matcore_graph_begin_capture(void *stream);

/// End capture, returning an executable graph.
/// The caller must destroy the graph with matcore_graph_exec_destroy().
void *matcore_graph_end_capture(void *stream);

/// Replay a captured graph on the given stream.
void matcore_graph_launch(void *graph_exec, void *stream);

/// Destroy an instantiated graph execution object.
void matcore_graph_exec_destroy(void *graph_exec);

/// Destroy a stream created by matcore_graph_stream_create.
void matcore_graph_stream_destroy(void *stream);

/// Set thread-local capture stream override. While set, mgpuStreamCreate()
/// returns this stream instead of creating a new one. Pass nullptr to clear.
void matcore_set_capture_stream_override(void *stream);

/// Set warm-up mode for graph capture. During warm-up, module load results
/// are cached and module unload is skipped, preparing for capture.
void matcore_set_graph_warmup(bool enable);

/// Synchronize a CUDA stream (blocking).
void matcore_stream_synchronize(void *stream);

}  // namespace matcore
