"""GPU-resident tensor support for MatcoreDSL.

Provides DeviceTensor, which keeps data on GPU between kernel launches,
and to_device(), which uploads a numpy array to GPU memory.
"""

from __future__ import annotations

import importlib
import re
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


_SUPPORTED_TARGETS: tuple[str, ...] = (
    "x86-auto",
    "x86-avx2",
    "x86-avx512",
    "amd-igpu",
    "nvidia-dgpu",
    "amd-npu",
)

_TARGET_ALIASES: dict[str, str] = {
    "x86": "x86-auto",
    "x86auto": "x86-auto",
    "x86-avx2": "x86-avx2",
    "x86_avx2": "x86-avx2",
    "x86-avx512": "x86-avx512",
    "x86_avx512": "x86-avx512",
    "amdgcn": "amd-igpu",
    "amd-igpu": "amd-igpu",
    "amd_igpu": "amd-igpu",
    "nvptx": "nvidia-dgpu",
    "nvidia-dgpu": "nvidia-dgpu",
    "nvidia_dgpu": "nvidia-dgpu",
    "npu": "amd-npu",
    "amd-npu": "amd-npu",
    "amd_npu": "amd-npu",
}

_NVIDIA_SM_TOKEN = re.compile(r"^(?:compute_)?sm?_?([0-9]{2,3})$")
_AMD_GFX_TOKEN = re.compile(r"^gfx[0-9a-z]+$")

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


def _normalize_dtype_name(dtype_name: str) -> str:
    lowered = dtype_name.strip().lower()
    aliases = {
        "float32": "float32",
        "single": "float32",
        "float16": "float16",
        "half": "float16",
        "bfloat16": "bfloat16",
        "bf16": "bfloat16",
        "int8": "int8",
        "i8": "int8",
        "int32": "int32",
        "i32": "int32",
        "float8_e4m3fn": "float8_e4m3fn",
        "float8e4m3fn": "float8_e4m3fn",
        "f8e4m3fn": "float8_e4m3fn",
        "fp8_e4m3fn": "float8_e4m3fn",
        "fp8-e4m3fn": "float8_e4m3fn",
        "e4m3fn": "float8_e4m3fn",
    }
    return aliases.get(lowered, lowered)


def _normalize_target(target: str) -> str:
    if not isinstance(target, str):
        raise TypeError("target must be a string")
    normalized = target.strip().lower()
    if not normalized:
        raise ValueError("target must not be empty")

    if match := _NVIDIA_SM_TOKEN.fullmatch(normalized):
        return f"nvidia-dgpu:sm_{match.group(1)}"

    base = normalized
    suffix = ""
    for separator in (":", "@", "/"):
        if separator in normalized:
            base, suffix = normalized.split(separator, 1)
            break

    base = _TARGET_ALIASES.get(base, base)
    if base not in _SUPPORTED_TARGETS:
        raise ValueError(
            "Unsupported target "
            f"'{target}'. Supported base targets: {', '.join(_SUPPORTED_TARGETS)}. "
            "GPU profiles may be requested as nvidia-dgpu:sm_90 or amd-igpu:gfx90a."
        )

    if not suffix:
        return base

    normalized_suffix = suffix.strip()
    if base == "nvidia-dgpu":
        match = _NVIDIA_SM_TOKEN.fullmatch(normalized_suffix)
        if match is None:
            raise ValueError(
                f"Unsupported NVIDIA target profile '{target}'. Use forms like 'nvidia-dgpu:sm_90'."
            )
        return f"{base}:sm_{match.group(1)}"
    if base == "amd-igpu":
        if _AMD_GFX_TOKEN.fullmatch(normalized_suffix) is None:
            raise ValueError(
                f"Unsupported AMD target profile '{target}'. Use forms like 'amd-igpu:gfx90a'."
            )
        return f"{base}:{normalized_suffix}"

    raise ValueError(
        f"Target profile suffixes are not supported for '{base}', got '{target}'."
    )


def _logical_dtype_name(array: Any) -> str:
    override = getattr(array, "matcore_dtype", None)
    if override is not None:
        dtype_name = _normalize_dtype_name(str(override))
    else:
        dtype_obj = getattr(array, "dtype", None)
        if dtype_obj is None:
            raise TypeError("to_device expects a NumPy-like array with a dtype")
        dtype_name = _normalize_dtype_name(str(getattr(dtype_obj, "name", dtype_obj)))

    if dtype_name not in _DTYPE_TO_NUMPY:
        raise ValueError(
            f"Unsupported logical MatCore dtype: {dtype_name!r}. "
            f"Supported dtypes: {', '.join(sorted(_DTYPE_TO_NUMPY))}"
        )
    return dtype_name


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
    normalized_target = _normalize_target(target)
    if not normalized_target.startswith("nvidia-dgpu"):
        raise ValueError(
            "DeviceTensor uploads are only supported for nvidia-dgpu targets. "
            f"Got target={target!r}."
        )

    logical_dtype = _logical_dtype_name(array)
    if logical_dtype == "int8":
        quant_obj = getattr(array, "matcore_quantization", None)
        has_quant_metadata = (
            quant_obj is not None
            or hasattr(array, "matcore_scale")
            or hasattr(array, "matcore_zero_point")
            or hasattr(array, "matcore_quant_enabled")
        )
        if has_quant_metadata:
            raise ValueError(
                "to_device() does not yet support quantized int8 logical tensors. "
                "Upload plain int8 tensors or keep quantized tensors on the host path."
            )
    arr = np.ascontiguousarray(array)
    storage_itemsize = int(np.dtype(arr.dtype).itemsize)
    expected_itemsize = int(_numpy_dtype(logical_dtype).itemsize)
    if storage_itemsize != expected_itemsize:
        raise ValueError(
            "Logical MatCore dtype and runtime storage dtype do not match for to_device(): "
            f"logical dtype '{logical_dtype}' expects itemsize {expected_itemsize}, "
            f"got storage itemsize {storage_itemsize}."
        )

    size_bytes = arr.nbytes
    native = _get_native_module()
    handle = native.matcore_device_alloc(size_bytes)
    native.matcore_device_upload(handle, arr.ctypes.data, size_bytes)
    return DeviceTensor(
        handle,
        arr.shape,
        logical_dtype,
        _contiguous_strides(arr),
        size_bytes,
    )
