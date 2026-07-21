from __future__ import annotations

import argparse
import gc
import json
import os
import pathlib
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CACHE_DIR = REPO_ROOT / ".matcore_cache"
DEFAULT_SIZE = 256
DEFAULT_WARMUP = 16
DEFAULT_RUNS = 4
DEFAULT_DTYPES = ("float16", "bfloat16", "int8")
DEFAULT_TIMEOUT_SECONDS = 1800.0
DEFAULT_ARENA = "nvidia"
ARENA_FRAMEWORKS: dict[str, tuple[str, ...]] = {
    "nvidia": ("matcore", "torch", "cupy"),
    "amd-igpu": ("matcore",),
    "cpu": ("matcore", "numpy"),
}
ARENA_MATCORE_TARGET: dict[str, str] = {
    "nvidia": "nvidia-dgpu",
    "amd-igpu": "amd-igpu",
    "cpu": "x86-avx512",
}


def _prepend_env_path(name: str, candidate: pathlib.Path) -> None:
    if not candidate.is_dir():
        return
    existing = os.environ.get(name, "")
    parts = [str(candidate)]
    if existing:
        parts.append(existing)
    os.environ[name] = ":".join(parts)


def prepare_cuda_library_paths() -> None:
    interpreter = pathlib.Path(sys.executable).resolve()
    for base in interpreter.parents:
        if base.name == ".venv_gladiator":
            site_packages = base / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
            nvidia_root = site_packages / "nvidia"
            _prepend_env_path("LD_LIBRARY_PATH", nvidia_root / "cu13" / "lib")
            for package_name in ("cudnn", "cusparselt", "nccl", "nvshmem"):
                _prepend_env_path("LD_LIBRARY_PATH", nvidia_root / package_name / "lib")
            break
    _prepend_env_path("LD_LIBRARY_PATH", pathlib.Path("/usr/local/cuda/lib64"))
    _prepend_env_path("LD_LIBRARY_PATH", pathlib.Path("/usr/local/cuda/targets/x86_64-linux/lib"))


prepare_cuda_library_paths()


def _optional_import_torch() -> Any | None:
    try:
        import torch

        return torch
    except Exception:
        return None


def _optional_import_cupy() -> Any | None:
    try:
        import cupy as cp

        return cp
    except Exception:
        return None


def _optional_import_matcore() -> Any | None:
    try:
        from matcore import mc

        return mc
    except Exception:
        return None


def host_has_nvidia_gpu() -> bool:
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return False
    try:
        probe = subprocess.run(
            [nvidia_smi, "-L"],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
    except Exception:
        return False
    return probe.returncode == 0 and bool((probe.stdout or "").strip())


@dataclass
class TimingStats:
    cold_ms: float
    warm_samples_ms: list[float]

    @property
    def warm_mean_ms(self) -> float:
        return statistics.fmean(self.warm_samples_ms) if self.warm_samples_ms else float("nan")

    @property
    def warm_min_ms(self) -> float:
        return min(self.warm_samples_ms) if self.warm_samples_ms else float("nan")

    @property
    def warm_p50_ms(self) -> float:
        return statistics.median(self.warm_samples_ms) if self.warm_samples_ms else float("nan")

    @property
    def warm_max_ms(self) -> float:
        return max(self.warm_samples_ms) if self.warm_samples_ms else float("nan")


@dataclass
class WorkerResult:
    framework: str
    dtype: str
    status: str
    message: str
    warmup: int
    runs: int
    cold_ms: float | None = None
    warm_mean_ms: float | None = None
    warm_min_ms: float | None = None
    warm_p50_ms: float | None = None
    warm_max_ms: float | None = None
    checksum: float | None = None
    target: str | None = None
    cache_mode: str | None = None
    compile_note: str | None = None


@dataclass
class MatCorePhases:
    cold_compile: WorkerResult
    warm_cache_boot: WorkerResult
    warm_steady: WorkerResult
    ongoing: WorkerResult


def float32_to_bf16_storage(values: np.ndarray) -> np.ndarray:
    values_f32 = np.ascontiguousarray(values, dtype=np.float32)
    bits_u32 = values_f32.view(np.uint32)
    rounded = bits_u32 + np.uint32(0x7FFF) + ((bits_u32 >> 16) & np.uint32(1))
    return (rounded >> 16).astype(np.uint16)


def has_numpy_bf16() -> bool:
    try:
        np.dtype("bfloat16")
        return True
    except TypeError:
        return False


def benchmark_operation(
    run_once: Callable[[], None],
    sync: Callable[[], None],
    *,
    warmup: int,
    runs: int,
) -> TimingStats:
    start = time.perf_counter()
    run_once()
    sync()
    cold_ms = (time.perf_counter() - start) * 1000.0

    for _ in range(warmup):
        run_once()
    sync()

    warm_samples_ms: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        run_once()
        sync()
        warm_samples_ms.append((time.perf_counter() - t0) * 1000.0)
    return TimingStats(cold_ms=cold_ms, warm_samples_ms=warm_samples_ms)


def stats_to_result(
    *,
    framework: str,
    dtype: str,
    message: str,
    stats: TimingStats,
    warmup: int,
    runs: int,
    checksum: float,
    target: str | None = None,
    cache_mode: str | None = None,
    compile_note: str | None = None,
) -> WorkerResult:
    return WorkerResult(
        framework=framework,
        dtype=dtype,
        status="OK",
        message=message,
        warmup=warmup,
        runs=runs,
        cold_ms=stats.cold_ms,
        warm_mean_ms=stats.warm_mean_ms,
        warm_min_ms=stats.warm_min_ms,
        warm_p50_ms=stats.warm_p50_ms,
        warm_max_ms=stats.warm_max_ms,
        checksum=checksum,
        target=target,
        cache_mode=cache_mode,
        compile_note=compile_note,
    )


def worker_skip(
    framework: str,
    dtype: str,
    message: str,
    *,
    warmup: int,
    runs: int,
    target: str | None = None,
    cache_mode: str | None = None,
    compile_note: str | None = None,
) -> WorkerResult:
    return WorkerResult(
        framework=framework,
        dtype=dtype,
        status="SKIP",
        message=message,
        warmup=warmup,
        runs=runs,
        target=target,
        cache_mode=cache_mode,
        compile_note=compile_note,
    )


def worker_fail(
    framework: str,
    dtype: str,
    exc: Exception,
    *,
    warmup: int,
    runs: int,
    target: str | None = None,
    cache_mode: str | None = None,
    compile_note: str | None = None,
) -> WorkerResult:
    return WorkerResult(
        framework=framework,
        dtype=dtype,
        status="FAIL",
        message=f"{type(exc).__name__}: {exc}",
        warmup=warmup,
        runs=runs,
        target=target,
        cache_mode=cache_mode,
        compile_note=compile_note,
    )


def is_unavailable_target_error(exc: Exception) -> bool:
    lowered = str(exc).lower()
    return any(
        token in lowered
        for token in (
            "execution denied",
            "no amd hip device is available",
            "no cuda device",
            "cuda runtime",
            "rocm runtime is present but no amd hip device is available",
            "hardware limits for target",
        )
    )


def make_fp_inputs(size: int, *, seed: int, dtype: np.dtype) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((size, size), dtype=np.float32).astype(dtype)
    b = rng.standard_normal((size, size), dtype=np.float32).astype(dtype)
    return np.ascontiguousarray(a), np.ascontiguousarray(b)


def make_int8_inputs(size: int, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = rng.integers(-8, 8, size=(size, size), dtype=np.int8)
    b = rng.integers(-8, 8, size=(size, size), dtype=np.int8)
    return np.ascontiguousarray(a), np.ascontiguousarray(b)


def parse_numpy_bf16_dtype() -> np.dtype | None:
    try:
        return np.dtype("bfloat16")
    except TypeError:
        return None


def run_numpy_worker(dtype: str, *, size: int, warmup: int, runs: int, seed: int) -> WorkerResult:
    try:
        if dtype == "float16":
            a, b = make_fp_inputs(size, seed=seed, dtype=np.float16)
        elif dtype == "bfloat16":
            bf16_dtype = parse_numpy_bf16_dtype()
            if bf16_dtype is None:
                return worker_skip("NumPy", dtype, "NumPy build has no bfloat16 dtype", warmup=warmup, runs=runs)
            a, b = make_fp_inputs(size, seed=seed, dtype=bf16_dtype)
        elif dtype == "int8":
            a, b = make_int8_inputs(size, seed=seed)
        else:
            return worker_skip("NumPy", dtype, f"Unsupported dtype '{dtype}'", warmup=warmup, runs=runs)

        out: np.ndarray | None = None

        def run_once() -> None:
            nonlocal out
            if dtype == "int8":
                out = a.astype(np.int32) @ b.astype(np.int32)
            else:
                out = a @ b

        stats = benchmark_operation(run_once, lambda: None, warmup=warmup, runs=runs)
        checksum = float(np.sum(np.asarray(out, dtype=np.float64))) if out is not None else 0.0
        return stats_to_result(
            framework="NumPy",
            dtype=dtype,
            message="np.matmul",
            stats=stats,
            warmup=warmup,
            runs=runs,
            checksum=checksum,
        )
    except MemoryError as exc:
        return worker_skip("NumPy", dtype, f"OOM: {exc}", warmup=warmup, runs=runs)
    except Exception as exc:
        return worker_fail("NumPy", dtype, exc, warmup=warmup, runs=runs)
    finally:
        gc.collect()


def run_torch_worker(
    dtype: str,
    *,
    size: int,
    warmup: int,
    runs: int,
    seed: int,
    require_cuda: bool,
) -> WorkerResult:
    torch = _optional_import_torch()
    if torch is None:
        return worker_skip("PyTorch", dtype, "PyTorch is not importable", warmup=warmup, runs=runs)

    try:
        torch.manual_seed(seed)
        use_cuda = bool(torch.cuda.is_available())
        if require_cuda and not use_cuda:
            return worker_skip(
                "PyTorch",
                dtype,
                "GPU arena requires CUDA, but torch.cuda.is_available() is False",
                warmup=warmup,
                runs=runs,
            )
        device = torch.device("cuda" if use_cuda else "cpu")
        sync = torch.cuda.synchronize if use_cuda else (lambda: None)

        if dtype == "float16":
            torch_dtype = torch.float16
        elif dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif dtype == "int8":
            torch_dtype = torch.int8
        else:
            return worker_skip("PyTorch", dtype, f"Unsupported dtype '{dtype}'", warmup=warmup, runs=runs)

        try:
            if dtype == "int8":
                a_host, b_host = make_int8_inputs(size, seed=seed)
                a = torch.from_numpy(a_host).to(device=device, dtype=torch_dtype)
                b = torch.from_numpy(b_host).to(device=device, dtype=torch_dtype)
            else:
                host_dtype = np.float16 if dtype == "float16" else np.float32
                a_host, b_host = make_fp_inputs(size, seed=seed, dtype=host_dtype)
                a = torch.from_numpy(a_host).to(device=device, dtype=torch_dtype)
                b = torch.from_numpy(b_host).to(device=device, dtype=torch_dtype)
        except Exception as exc:
            return worker_skip(
                "PyTorch",
                dtype,
                f"{device.type} cannot allocate/create dtype={dtype} ({type(exc).__name__}: {exc})",
                warmup=warmup,
                runs=runs,
            )

        compile_note = "eager"
        if dtype == "int8":
            base_fn = lambda x, y: x.to(torch.int32) @ y.to(torch.int32)
        else:
            base_fn = lambda x, y: x @ y
        fn: Callable[[Any, Any], Any] = base_fn
        if use_cuda and hasattr(torch, "compile"):
            try:
                fn = torch.compile(base_fn, dynamic=False, fullgraph=False)
                compile_note = "torch.compile(cuda)"
            except Exception as exc:
                compile_note = f"torch.compile fallback=eager ({type(exc).__name__}: {exc})"
        elif use_cuda:
            compile_note = "torch.compile unavailable, eager"
        else:
            compile_note = "cpu eager"

        out: Any = None

        def run_once() -> None:
            nonlocal out
            out = fn(a, b)

        stats = benchmark_operation(run_once, sync, warmup=warmup, runs=runs)
        checksum = float(out.float().sum().item()) if out is not None else 0.0
        return stats_to_result(
            framework="PyTorch",
            dtype=dtype,
            message=f"device={device.type}",
            stats=stats,
            warmup=warmup,
            runs=runs,
            checksum=checksum,
            compile_note=compile_note,
        )
    except RuntimeError as exc:
        return worker_skip("PyTorch", dtype, f"Runtime unsupported: {exc}", warmup=warmup, runs=runs)
    except Exception as exc:
        return worker_fail("PyTorch", dtype, exc, warmup=warmup, runs=runs)
    finally:
        gc.collect()
        if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_cupy_worker(dtype: str, *, size: int, warmup: int, runs: int, seed: int) -> WorkerResult:
    cp = _optional_import_cupy()
    if cp is None:
        return worker_skip("CuPy", dtype, "CuPy is not importable", warmup=warmup, runs=runs)

    try:
        if int(cp.cuda.runtime.getDeviceCount()) <= 0:
            return worker_skip("CuPy", dtype, "No CUDA device visible", warmup=warmup, runs=runs)
    except Exception as exc:
        return worker_skip("CuPy", dtype, f"CUDA runtime unavailable ({type(exc).__name__}: {exc})", warmup=warmup, runs=runs)

    try:
        if dtype == "float16":
            cp_dtype = cp.float16
        elif dtype == "bfloat16":
            if not hasattr(cp, "bfloat16"):
                return worker_skip("CuPy", dtype, "CuPy build has no bfloat16 dtype", warmup=warmup, runs=runs)
            cp_dtype = cp.bfloat16
        elif dtype == "int8":
            cp_dtype = cp.int8
        else:
            return worker_skip("CuPy", dtype, f"Unsupported dtype '{dtype}'", warmup=warmup, runs=runs)

        if dtype == "int8":
            a_host, b_host = make_int8_inputs(size, seed=seed)
            a = cp.asarray(a_host, dtype=cp_dtype)
            b = cp.asarray(b_host, dtype=cp_dtype)
        else:
            host_dtype = np.float16 if dtype == "float16" else np.float32
            a_host, b_host = make_fp_inputs(size, seed=seed, dtype=host_dtype)
            a = cp.asarray(a_host, dtype=cp_dtype)
            b = cp.asarray(b_host, dtype=cp_dtype)

        out: Any = None
        sync = cp.cuda.Stream.null.synchronize

        def run_once() -> None:
            nonlocal out
            if dtype == "int8":
                out = a.astype(cp.int32) @ b.astype(cp.int32)
            else:
                out = a @ b

        stats = benchmark_operation(run_once, sync, warmup=warmup, runs=runs)
        checksum = float(cp.asnumpy(cp.sum(out, dtype=cp.float64))) if out is not None else 0.0
        return stats_to_result(
            framework="CuPy",
            dtype=dtype,
            message="cuBLAS via cupy.matmul",
            stats=stats,
            warmup=warmup,
            runs=runs,
            checksum=checksum,
        )
    except cp.cuda.memory.OutOfMemoryError as exc:
        return worker_skip("CuPy", dtype, f"OOM: {exc}", warmup=warmup, runs=runs)
    except Exception as exc:
        return worker_fail("CuPy", dtype, exc, warmup=warmup, runs=runs)
    finally:
        gc.collect()


mc_for_decorator = _optional_import_matcore()
if mc_for_decorator is not None:

    @mc_for_decorator.kernel
    def _matcore_kernel(a, b, c):
        lhs = mc_for_decorator.load(a)
        rhs = mc_for_decorator.load(b)
        out = mc_for_decorator.matmul(lhs, rhs)
        mc_for_decorator.store(c, out)

else:
    _matcore_kernel = None


def select_matcore_target(requested_target: str) -> tuple[str | None, str]:
    if requested_target != "auto":
        return requested_target, "user-selected"
    if host_has_nvidia_gpu():
        return "nvidia-dgpu", "auto-selected nvidia-dgpu"
    return "x86-auto", "auto-selected x86-auto"


def prepare_matcore_inputs(mc: Any, dtype: str, *, size: int, seed: int) -> tuple[Any, Any, np.ndarray, dict[str, Any] | None]:
    quant: dict[str, Any] | None = None
    cp = _optional_import_cupy()
    use_device = cp is not None and host_has_nvidia_gpu()
    if dtype == "float16":
        a, b = make_fp_inputs(size, seed=seed, dtype=np.float16)
        if use_device:
            a = cp.asarray(a)
            b = cp.asarray(b)
            out = cp.zeros((size, size), dtype=cp.float16)
        else:
            out = np.zeros((size, size), dtype=np.float16)
        return a, b, out, quant
    if dtype == "bfloat16":
        a_f32, b_f32 = make_fp_inputs(size, seed=seed, dtype=np.float32)
        a_storage = float32_to_bf16_storage(a_f32)
        b_storage = float32_to_bf16_storage(b_f32)
        if use_device:
            a_arg = mc.asdtype(cp.asarray(a_storage), "bfloat16")
            b_arg = mc.asdtype(cp.asarray(b_storage), "bfloat16")
            out = cp.zeros((size, size), dtype=cp.float32)
        else:
            a_arg = mc.asdtype(a_storage, "bfloat16")
            b_arg = mc.asdtype(b_storage, "bfloat16")
            out = np.zeros((size, size), dtype=np.float32)
        return a_arg, b_arg, out, quant
    if dtype == "int8":
        a_i8, b_i8 = make_int8_inputs(size, seed=seed)
        if use_device:
            a_arg = mc.asdtype(cp.asarray(a_i8), "int8", scale=1.0, zero_point=0)
            b_arg = mc.asdtype(cp.asarray(b_i8), "int8", scale=1.0, zero_point=0)
            out = cp.zeros((size, size), dtype=cp.int32)
        else:
            a_arg = mc.asdtype(a_i8, "int8", scale=1.0, zero_point=0)
            b_arg = mc.asdtype(b_i8, "int8", scale=1.0, zero_point=0)
            out = np.zeros((size, size), dtype=np.int32)
        quant = {"scale": 1.0, "zero_point": 0, "enabled": True}
        return a_arg, b_arg, out, quant
    raise ValueError(f"Unsupported dtype '{dtype}'")


def matcore_checksum(out: Any) -> float:
    cp = _optional_import_cupy()
    if cp is not None and hasattr(out, "__cuda_array_interface__"):
        return float(cp.asnumpy(cp.sum(out, dtype=cp.float64)))
    return float(np.sum(np.asarray(out, dtype=np.float64)))


def clear_matcore_cache() -> None:
    shutil.rmtree(CACHE_DIR, ignore_errors=True)


def run_matcore_worker(
    dtype: str,
    *,
    size: int,
    warmup: int,
    runs: int,
    seed: int,
    requested_target: str,
    cache_mode: str,
) -> WorkerResult:
    mc = _optional_import_matcore()
    if mc is None:
        return worker_skip("MatCore", dtype, "MatCore Python module is not importable", warmup=warmup, runs=runs)
    if _matcore_kernel is None:
        return worker_skip("MatCore", dtype, "MatCore kernel decorator is unavailable", warmup=warmup, runs=runs)

    target, target_note = select_matcore_target(requested_target)
    if target is None:
        return worker_skip("MatCore", dtype, "No executable MatCore target detected", warmup=warmup, runs=runs)

    try:
        if cache_mode == "cold":
            clear_matcore_cache()

        a_arg, b_arg, out, quant = prepare_matcore_inputs(mc, dtype, size=size, seed=seed)

        def launch_once() -> None:
            if quant is None:
                mc.launch(_matcore_kernel, a_arg, b_arg, out, target=target)
            else:
                mc.launch(_matcore_kernel, a_arg, b_arg, out, target=target, quant=quant)

        if cache_mode in ("cold", "warm-cache"):
            start = time.perf_counter()
            launch_once()
            cp = _optional_import_cupy()
            if cp is not None and hasattr(out, "__cuda_array_interface__"):
                cp.cuda.Stream.null.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            checksum = matcore_checksum(out)
            return WorkerResult(
                framework="MatCore",
                dtype=dtype,
                status="OK",
                message=f"target={target} ({target_note})",
                warmup=0,
                runs=1,
                cold_ms=elapsed_ms,
                checksum=checksum,
                target=target,
                cache_mode=cache_mode,
                compile_note="disk-cache cold build" if cache_mode == "cold" else "disk-cache warm load",
            )

        if cache_mode == "ongoing":
            launch_once()
            ongoing_runs = 100
            samples_ms: list[float] = []
            for _ in range(ongoing_runs):
                t0 = time.perf_counter()
                launch_once()
                samples_ms.append((time.perf_counter() - t0) * 1000.0)

            checksum = float(np.sum(np.asarray(out, dtype=np.float64)))
            median_ms = statistics.median(samples_ms)
            p50_ms = median_ms
            mean_ms = statistics.fmean(samples_ms)
            min_ms = min(samples_ms)
            max_ms = max(samples_ms)

            return WorkerResult(
                framework="MatCore",
                dtype=dtype,
                status="OK",
                message=f"target={target} ({target_note})",
                warmup=0,
                runs=ongoing_runs,
                cold_ms=None,
                warm_mean_ms=mean_ms,
                warm_min_ms=min_ms,
                warm_p50_ms=p50_ms,
                warm_max_ms=max_ms,
                checksum=checksum,
                target=target,
                cache_mode="ongoing",
                compile_note="ongoing cached execution (100 iters)",
            )

        stats = benchmark_operation(launch_once, lambda: None, warmup=warmup, runs=runs)
        checksum = float(np.sum(np.asarray(out, dtype=np.float64)))
        return stats_to_result(
            framework="MatCore",
            dtype=dtype,
            message=f"target={target} ({target_note})",
            stats=stats,
            warmup=warmup,
            runs=runs,
            checksum=checksum,
            target=target,
            cache_mode=cache_mode,
            compile_note="steady-state cached execution",
        )
    except MemoryError as exc:
        return worker_skip("MatCore", dtype, f"OOM: {exc}", warmup=warmup, runs=runs, target=target, cache_mode=cache_mode)
    except RuntimeError as exc:
        if is_unavailable_target_error(exc):
            return worker_skip("MatCore", dtype, str(exc), warmup=warmup, runs=runs, target=target, cache_mode=cache_mode)
        return worker_fail("MatCore", dtype, exc, warmup=warmup, runs=runs, target=target, cache_mode=cache_mode)
    except Exception as exc:
        return worker_fail("MatCore", dtype, exc, warmup=warmup, runs=runs, target=target, cache_mode=cache_mode)
    finally:
        gc.collect()


def run_worker(args: argparse.Namespace) -> WorkerResult:
    framework = args.worker_framework
    dtype = args.worker_dtype
    arena = args.arena
    matcore_target = args.matcore_target if args.matcore_target != "auto" else ARENA_MATCORE_TARGET[arena]
    if framework == "matcore":
        return run_matcore_worker(
            dtype,
            size=args.size,
            warmup=args.warmup,
            runs=args.runs,
            seed=args.seed,
            requested_target=matcore_target,
            cache_mode=args.matcore_cache_mode,
        )
    if framework == "numpy":
        if arena != "cpu":
            return worker_skip(
                "NumPy",
                dtype,
                f"NumPy is excluded in arena={arena}",
                warmup=args.warmup,
                runs=args.runs,
            )
        return run_numpy_worker(dtype, size=args.size, warmup=args.warmup, runs=args.runs, seed=args.seed)
    if framework == "torch":
        if arena != "nvidia":
            return worker_skip(
                "PyTorch",
                dtype,
                f"PyTorch is excluded in arena={arena}",
                warmup=args.warmup,
                runs=args.runs,
            )
        return run_torch_worker(
            dtype,
            size=args.size,
            warmup=args.warmup,
            runs=args.runs,
            seed=args.seed,
            require_cuda=True,
        )
    if framework == "cupy":
        if arena != "nvidia":
            return worker_skip(
                "CuPy",
                dtype,
                f"CuPy is excluded in arena={arena}",
                warmup=args.warmup,
                runs=args.runs,
            )
        return run_cupy_worker(dtype, size=args.size, warmup=args.warmup, runs=args.runs, seed=args.seed)
    return worker_skip(framework, dtype, f"Unknown framework '{framework}'", warmup=args.warmup, runs=args.runs)


def parse_worker_json(text: str) -> WorkerResult:
    payload = json.loads(text)
    return WorkerResult(**payload)


def run_worker_process(
    framework: str,
    dtype: str,
    *,
    arena: str,
    size: int,
    warmup: int,
    runs: int,
    seed: int,
    matcore_target: str,
    matcore_cache_mode: str,
    timeout_seconds: float,
) -> WorkerResult:
    command = [
        sys.executable,
        __file__,
        "--arena",
        arena,
        "--worker-framework",
        framework,
        "--worker-dtype",
        dtype,
        "--size",
        str(size),
        "--warmup",
        str(warmup),
        "--runs",
        str(runs),
        "--seed",
        str(seed),
        "--matcore-target",
        matcore_target,
        "--matcore-cache-mode",
        matcore_cache_mode,
    ]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    try:
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return WorkerResult(
            framework={
                "matcore": "MatCore",
                "numpy": "NumPy",
                "torch": "PyTorch",
                "cupy": "CuPy",
            }.get(framework, framework),
            dtype=dtype,
            status="FAIL",
            message=f"TimeoutExpired: exceeded {timeout_seconds:.0f}s",
            warmup=warmup,
            runs=runs,
            cache_mode=matcore_cache_mode if framework == "matcore" else None,
        )
    stdout = (result.stdout or "").strip().splitlines()
    stderr = (result.stderr or "").strip()
    framework_label = {
        "matcore": "MatCore",
        "numpy": "NumPy",
        "torch": "PyTorch",
        "cupy": "CuPy",
    }.get(framework, framework)
    if result.returncode != 0:
        detail = stderr.splitlines()[-1] if stderr else "worker exited without stderr"
        return WorkerResult(
            framework=framework_label,
            dtype=dtype,
            status="FAIL",
            message=f"WorkerExit: returncode={result.returncode}, detail={detail}",
            warmup=warmup,
            runs=runs,
            cache_mode=matcore_cache_mode if framework == "matcore" else None,
        )
    if not stdout:
        return WorkerResult(
            framework=framework_label,
            dtype=dtype,
            status="FAIL",
            message="worker exited without JSON output",
            warmup=warmup,
            runs=runs,
            cache_mode=matcore_cache_mode if framework == "matcore" else None,
        )
    return parse_worker_json(stdout[-1])


def format_ms(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.3f} ms"


def format_speedup(reference_ms: float | None, candidate_ms: float | None) -> str:
    if reference_ms is None or candidate_ms is None:
        return "n/a"
    if reference_ms <= 0.0 or candidate_ms <= 0.0:
        return "n/a"
    return f"{reference_ms / candidate_ms:.2f}x"


def throughput_line(dtype: str, size: int, time_ms: float | None) -> str:
    if time_ms is None or time_ms <= 0 or not np.isfinite(time_ms):
        return ""
    flops = 2 * size * size * size
    tput = flops / (time_ms / 1000.0) / 1e12
    unit = "TOPS" if dtype == "int8" else "TFLOP/s"
    return f" [{tput:.4f} {unit}]"


def print_worker_result(result: WorkerResult, *, size: int) -> None:
    prefix = result.framework
    if result.framework == "MatCore" and result.cache_mode:
        prefix = f"{prefix}[{result.cache_mode}]"
    if result.status == "OK":
        tput = ""
        if result.framework != "MatCore" or result.cache_mode in ("steady", "ongoing"):
            tput = throughput_line(result.dtype, size, result.warm_mean_ms)
        print(
            f"[OK]   {prefix:<18} {result.dtype:<8} cold={format_ms(result.cold_ms)} "
            f"warm_mean={format_ms(result.warm_mean_ms)}{tput}"
        )
        note = result.message
        if result.compile_note:
            note += f"; {result.compile_note}"
        if result.checksum is not None:
            note += f"; checksum={result.checksum:.6e}"
        print(f"       note: {note}")
    elif result.status == "SKIP":
        print(f"[SKIP] {prefix:<18} {result.dtype:<8} {result.message}")
    else:
        print(f"[FAIL] {prefix:<18} {result.dtype:<8} {result.message}")


def benchmark_matcore_phases(args: argparse.Namespace, dtype: str) -> MatCorePhases:
    matcore_target = args.matcore_target if args.matcore_target != "auto" else ARENA_MATCORE_TARGET[args.arena]
    cold = run_worker_process(
        "matcore",
        dtype,
        arena=args.arena,
        size=args.size,
        warmup=0,
        runs=1,
        seed=args.seed,
        matcore_target=matcore_target,
        matcore_cache_mode="cold",
        timeout_seconds=args.timeout,
    )
    warm_cache = run_worker_process(
        "matcore",
        dtype,
        arena=args.arena,
        size=args.size,
        warmup=0,
        runs=1,
        seed=args.seed,
        matcore_target=matcore_target,
        matcore_cache_mode="warm-cache",
        timeout_seconds=args.timeout,
    )
    steady = run_worker_process(
        "matcore",
        dtype,
        arena=args.arena,
        size=args.size,
        warmup=args.warmup,
        runs=args.runs,
        seed=args.seed,
        matcore_target=matcore_target,
        matcore_cache_mode="steady",
        timeout_seconds=args.timeout,
    )
    ongoing = run_worker_process(
        "matcore",
        dtype,
        arena=args.arena,
        size=args.size,
        warmup=0,
        runs=1,
        seed=args.seed,
        matcore_target=matcore_target,
        matcore_cache_mode="ongoing",
        timeout_seconds=args.timeout,
    )
    return MatCorePhases(cold_compile=cold, warm_cache_boot=warm_cache, warm_steady=steady, ongoing=ongoing)


def run_framework(args: argparse.Namespace, framework: str, dtype: str) -> WorkerResult:
    matcore_target = args.matcore_target if args.matcore_target != "auto" else ARENA_MATCORE_TARGET[args.arena]
    return run_worker_process(
        framework,
        dtype,
        arena=args.arena,
        size=args.size,
        warmup=args.warmup,
        runs=args.runs,
        seed=args.seed,
        matcore_target=matcore_target,
        matcore_cache_mode="steady",
        timeout_seconds=args.timeout,
    )


def print_speed_summary(
    *,
    arena: str,
    dtype: str,
    numpy_result: WorkerResult | None,
    matcore_phases: MatCorePhases | None,
    torch_result: WorkerResult | None,
    cupy_result: WorkerResult | None,
) -> None:
    numpy_warm = numpy_result.warm_mean_ms if numpy_result and numpy_result.status == "OK" else None
    matcore_warm = (
        matcore_phases.warm_steady.warm_mean_ms
        if matcore_phases and matcore_phases.warm_steady.status == "OK"
        else None
    )
    torch_warm = torch_result.warm_mean_ms if torch_result and torch_result.status == "OK" else None
    cupy_warm = cupy_result.warm_mean_ms if cupy_result and cupy_result.status == "OK" else None

    if arena == "cpu":
        print(f"       speed vs NumPy [{dtype}]:")
        print(f"       MatCore steady: {format_speedup(numpy_warm, matcore_warm)}")
        return

    if arena == "amd-igpu":
        print(f"       speed summary [{dtype}]:")
        print("       MatCore only in this arena")
        return

    if arena != "nvidia":
        return

    print(f"       speed vs PyTorch CUDA [{dtype}]:")
    print(f"       MatCore steady: {format_speedup(torch_warm, matcore_warm)}")
    print(f"       CuPy          : {format_speedup(torch_warm, cupy_warm)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Gladiator benchmark arena with arena-specific framework matchups for "
            "the Phase 4.1 hotfix, using MatCore cold-build, warm-cache, steady-state, "
            "and ongoing timings."
        )
    )
    arena_choices = sorted({*ARENA_FRAMEWORKS.keys(), "gpu"})
    parser.add_argument("--arena", choices=arena_choices, default=DEFAULT_ARENA)
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--seed", type=int, default=20260323)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--dtypes", nargs="*", default=list(DEFAULT_DTYPES))
    parser.add_argument("--frameworks", nargs="*", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--matcore-target", default="auto", help=argparse.SUPPRESS)
    parser.add_argument("--worker-framework", default=None)
    parser.add_argument("--worker-dtype", default=None)
    parser.add_argument("--matcore-cache-mode", choices=["cold", "warm-cache", "steady", "ongoing"], default="steady")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.arena == "gpu":
        if args.worker_framework is None:
            print("warning: --arena gpu is deprecated; use --arena nvidia.", file=sys.stderr)
        args.arena = "nvidia"
    if args.worker_framework is not None:
        worker_result = run_worker(args)
        print(json.dumps(asdict(worker_result), sort_keys=True))
        return

    print(
        "benchmark_gladiator.py "
        f"(python={sys.executable}, size={args.size}, warmup={args.warmup}, runs={args.runs})"
    )
    print(f"cache_dir={CACHE_DIR}")
    resolved_target = args.matcore_target if args.matcore_target != "auto" else ARENA_MATCORE_TARGET[args.arena]
    print(f"arena={args.arena} matcore_target={resolved_target}")
    print("MatCore reports cold-build, warm-cache boot, steady-state, and ongoing separately.")
    print("NumPy, PyTorch, and CuPy run in isolated worker processes with the same warm-up count.")
    if args.frameworks is not None:
        print("note: --frameworks is ignored; framework set is derived from --arena.")
    frameworks = set(ARENA_FRAMEWORKS[args.arena])
    if args.arena in ("nvidia", "amd-igpu"):
        from gpu_preflight import run_full_preflight

        report = run_full_preflight(args.arena)
        if not report.overall_ok:
            print(f"GPU preflight FAILED for {args.arena}: {report.summary}")
            sys.exit(1)
        print(f"GPU preflight PASSED: {report.summary}")

    for dtype in [item.strip().lower() for item in args.dtypes]:
        print(f"\n=== dtype={dtype} ===")
        numpy_result: WorkerResult | None = None
        matcore_phases: MatCorePhases | None = None
        torch_result: WorkerResult | None = None
        cupy_result: WorkerResult | None = None

        if "matcore" in frameworks:
            matcore_phases = benchmark_matcore_phases(args, dtype)
            print_worker_result(matcore_phases.cold_compile, size=args.size)
            print_worker_result(matcore_phases.warm_cache_boot, size=args.size)
            print_worker_result(matcore_phases.warm_steady, size=args.size)
            print_worker_result(matcore_phases.ongoing, size=args.size)

        if "numpy" in frameworks:
            numpy_result = run_framework(args, "numpy", dtype)
            print_worker_result(numpy_result, size=args.size)

        if "torch" in frameworks:
            torch_result = run_framework(args, "torch", dtype)
            print_worker_result(torch_result, size=args.size)

        if "cupy" in frameworks:
            cupy_result = run_framework(args, "cupy", dtype)
            print_worker_result(cupy_result, size=args.size)

        print_speed_summary(
            arena=args.arena,
            dtype=dtype,
            numpy_result=numpy_result,
            matcore_phases=matcore_phases,
            torch_result=torch_result,
            cupy_result=cupy_result,
        )


if __name__ == "__main__":
    main()
