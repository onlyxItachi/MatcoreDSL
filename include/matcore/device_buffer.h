#ifndef MATCORE_DEVICE_BUFFER_H_
#define MATCORE_DEVICE_BUFFER_H_

#include <cstdint>

namespace matcore {

/// Opaque handle for device-resident GPU memory with ownership tracking.
/// alloc_id is a monotonic counter — stale handles (after free + re-alloc of
/// same physical address) are detected and rejected.
struct DeviceBufferHandle {
  std::uint64_t ptr = 0;         // CUdeviceptr stored as uint64_t for ABI stability
  std::uint64_t size_bytes = 0;
  std::uint64_t alloc_id = 0;    // Monotonic ownership token
};

/// Allocate device memory via the existing GpuMemoryPool.
/// Returns a handle with a unique alloc_id. Throws on CUDA failure.
DeviceBufferHandle matcore_device_alloc(std::uint64_t size_bytes);

/// Release device memory back to GpuMemoryPool (not cuMemFree).
/// Validates alloc_id ownership — rejects stale or unknown handles.
/// Throws on invalid handle or double-free.
void matcore_device_free(DeviceBufferHandle handle);

/// Copy host data to device buffer. Validates size_bytes <= handle capacity.
/// Uses ScopedContext + synchronous cuMemcpy. Throws on invalid handle or
/// size overflow.
void matcore_device_upload(DeviceBufferHandle dst, const void *host_src,
                           std::uint64_t size_bytes);

/// Copy device data to host buffer. Validates size_bytes <= handle capacity.
/// Uses ScopedContext + synchronous cuMemcpy. Throws on invalid handle or
/// size overflow.
void matcore_device_download(void *host_dst, DeviceBufferHandle src,
                             std::uint64_t size_bytes);

/// Check if a handle points to a valid, live device allocation.
bool matcore_device_is_valid(DeviceBufferHandle handle);

/// Zero-fill device buffer using cuMemsetD32Async. Validates handle.
/// Uses ScopedContext + stream synchronize.
void matcore_device_zero(DeviceBufferHandle handle);

/// Zero-fill a raw device pointer. Used by execute_plan() for output zeroing
/// without needing a DeviceBufferHandle. No ownership validation — caller
/// must ensure pointer is valid.
void matcore_device_zero_raw(void *device_ptr, uint64_t size_bytes);

/// Zero-fill a raw device pointer on a specific stream. Required for CUDA
/// graph capture — zeroing must be on the capture stream to be recorded.
void matcore_device_zero_raw_on_stream(void *device_ptr, uint64_t size_bytes,
                                       void *stream);

}  // namespace matcore

#endif  // MATCORE_DEVICE_BUFFER_H_
