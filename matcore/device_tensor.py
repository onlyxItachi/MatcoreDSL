"""GPU-resident tensor support for MatcoreDSL.

Provides DeviceTensor, which keeps data on GPU between kernel launches,
and to_device(), which uploads a numpy array to GPU memory.
"""

from __future__ import annotations

import importlib
import weakref
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _get_native_module() -> Any:
    """Lazily load the _matcore_native C++ module."""
    try:
        return importlib.import_module(f"{__package__}._matcore_native")
    except Exception as exc:
        raise RuntimeError(
            "Failed to import native module '_matcore_native'. Build the extension first."
        ) from exc


_DTYPE_TO_NUMPY = {
    "float16": np.float16,
    "float32": np.float32,
    "bfloat16": np.uint16,
    "int8": np.int8,
    "int32": np.int32,
    "float8_e4m3fn": np.uint8,
}

_NUMPY_TO_DTYPE = {
    np.dtype(np.float16): "float16",
    np.dtype(np.float32): "float32",
    np.dtype(np.int8): "int8",
    np.dtype(np.int32): "int32",
}


def _numpy_dtype(matcore_dtype: str) -> np.dtype:
    """Convert a MatCore dtype string to a numpy dtype."""
    try:
        return np.dtype(_DTYPE_TO_NUMPY[matcore_dtype])
    except KeyError:
        raise ValueError(f"Unsupported MatCore dtype: {matcore_dtype!r}") from None


def _matcore_dtype(np_dtype) -> str:
    """Convert a numpy dtype to a MatCore dtype string."""
    key = np.dtype(np_dtype)
    try:
        return _NUMPY_TO_DTYPE[key]
    except KeyError:
        raise ValueError(f"Unsupported numpy dtype: {key!r}") from None


def _contiguous_strides(arr: np.ndarray) -> tuple:
    """Compute *element* strides for a C-contiguous array."""
    if arr.ndim == 0:
        return ()
    strides: list[int] = []
    acc = 1
    for dim in reversed(arr.shape):
        strides.append(acc)
        acc *= dim
    return tuple(reversed(strides))


# ---------------------------------------------------------------------------
# DeviceTensor
# ---------------------------------------------------------------------------

class DeviceTensor:
    """GPU-resident tensor. Created via to_device(), freed via .free() or context manager."""

    # ---- weak clean-up callback (static so it holds no ref to self) --------
    @staticmethod
    def _weak_free(handle: object, size_bytes: int) -> None:
        """weakref.finalize callback — must never raise."""
        try:
            native = _get_native_module()
            native.matcore_device_free(handle)
        except Exception:
            # Runtime may be torn down; swallow everything.
            pass

    # ---- construction -------------------------------------------------------
    def __init__(
        self,
        handle: object,
        shape: tuple[int, ...],
        dtype: str,
        strides: tuple[int, ...],
        size_bytes: int,
    ) -> None:
        self._handle = handle
        self._shape = tuple(shape)
        self._dtype = str(dtype)
        self._strides = tuple(strides)
        self._size_bytes = size_bytes
        self._freed = False
        self._matcore_device_tensor = True  # instance attr checked by C++ via hasattr

        # Weak-ref-based safety net for leaked tensors.
        self._weak_finalizer = weakref.finalize(
            self, DeviceTensor._weak_free, handle, size_bytes
        )

    # ---- read-only metadata properties --------------------------------------
    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> str:
        return self._dtype

    @property
    def strides(self) -> tuple[int, ...]:
        return self._strides

    @property
    def size_bytes(self) -> int:
        return self._size_bytes

    # ---- primary cleanup ----------------------------------------------------
    def free(self) -> None:
        """Explicitly free GPU memory (primary cleanup path)."""
        if self._freed:
            return
        self._freed = True
        try:
            native = _get_native_module()
            native.matcore_device_free(self._handle)
        finally:
            # Detach the weak finalizer so it won't double-free.
            self._weak_finalizer.detach()

    # ---- context manager ----------------------------------------------------
    def __enter__(self) -> "DeviceTensor":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.free()

    # ---- device_ptr property ------------------------------------------------
    @property
    def device_ptr(self) -> object:
        """Return the native handle. Raises if the tensor has been freed."""
        if self._freed:
            raise RuntimeError("DeviceTensor has already been freed.")
        return self._handle

    # ---- device-side zero fill -----------------------------------------------
    def zero_(self) -> None:
        """Zero-fill the entire device buffer (cuMemset). In-place operation."""
        if self._freed:
            raise RuntimeError("DeviceTensor has already been freed.")
        native = _get_native_module()
        native.matcore_device_zero(self._handle)

    # ---- download to host ---------------------------------------------------
    def to_host(self) -> np.ndarray:
        """Download GPU data to a numpy array."""
        if self._freed:
            raise RuntimeError("DeviceTensor has already been freed.")

        np_dt = _numpy_dtype(self.dtype)
        out = np.empty(self.shape, dtype=np_dt)
        native = _get_native_module()
        native.matcore_device_download(out.ctypes.data, self._handle, self._size_bytes)
        return out

    # ---- repr ---------------------------------------------------------------
    def __repr__(self) -> str:
        status = "freed" if self._freed else "live"
        return (
            f"DeviceTensor(shape={self.shape}, dtype={self.dtype!r}, "
            f"strides={self.strides}, size_bytes={self._size_bytes}, {status})"
        )


# ---------------------------------------------------------------------------
# Public upload helper
# ---------------------------------------------------------------------------

def to_device(array, *, target: str = "nvidia-dgpu") -> DeviceTensor:
    """Upload a numpy array to GPU memory. Returns a DeviceTensor."""
    arr = np.ascontiguousarray(array)
    size_bytes = arr.nbytes
    native = _get_native_module()
    handle = native.matcore_device_alloc(size_bytes)
    native.matcore_device_upload(handle, arr.ctypes.data, size_bytes)
    return DeviceTensor(
        handle,
        arr.shape,
        _matcore_dtype(arr.dtype),
        _contiguous_strides(arr),
        size_bytes,
    )
