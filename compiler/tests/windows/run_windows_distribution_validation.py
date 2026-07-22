#!/usr/bin/env python3

"""Fail-closed Windows x64 package and artifact validation.

This script intentionally uses LLVM's COFF/PE inspection tools.  It does not
reinterpret ELF evidence as Windows support and it writes only generated CI
evidence beneath the caller-selected output path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
from typing import Any


EXPORT_PATTERN = re.compile(
    r"MATCORE_RUNTIME_API\s+matcore_status_v0\s+"
    r"(matcore_runtime_[A-Za-z0-9_]+)\s*\(",
    re.MULTILINE,
)
AVX2_SYMBOL = "matcore_cpu_packed_avx2_4x16_microkernel_f32_v1"
AVX512_SYMBOL = "matcore_cpu_packed_avx512_4x16_microkernel_f32_v1"


def run(
    command: list[str],
    *,
    environment: dict[str, str] | None = None,
    expect_success: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        check=False,
        text=True,
        capture_output=True,
        env=environment,
    )
    if expect_success and result.returncode != 0:
        print(result.stdout, file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"command failed ({result.returncode}): {command}")
    return result


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise RuntimeError(f"required Windows distribution file is missing: {path}")
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_coff_x64(readobj: Path, path: Path) -> str:
    result = run([str(readobj), "--file-headers", str(path)])
    output = result.stdout + result.stderr
    if "COFF-x86-64" not in output and "IMAGE_FILE_MACHINE_AMD64" not in output:
        raise RuntimeError(f"artifact is not authenticated x64 COFF/PE: {path}\n{output}")
    return output


def inspect_microkernel(objdump: Path, archive: Path, symbol: str, register: str) -> dict[str, int]:
    result = run(
        [str(objdump), f"--disassemble-symbols={symbol}", "--no-show-raw-insn", str(archive)]
    )
    output = result.stdout + result.stderr
    if symbol not in output:
        raise RuntimeError(f"exact microkernel symbol is absent: {symbol}")
    register_count = len(re.findall(rf"\b{register}[0-9]+\b", output, re.IGNORECASE))
    fma_count = len(
        re.findall(r"\bvfmadd(?:132|213|231)ps\b", output, re.IGNORECASE)
    )
    if register_count == 0 or fma_count == 0:
        raise RuntimeError(
            f"{symbol} lacks exact {register.upper()} packed-FMA evidence: "
            f"registers={register_count} fma={fma_count}"
        )
    return {"register_operands": register_count, "packed_fma_sites": fma_count}


def scan_for_path_leaks(paths: list[Path], forbidden: list[Path]) -> None:
    needles: set[bytes] = set()
    for path in forbidden:
        for spelling in (str(path), path.as_posix()):
            needles.add(spelling.encode("utf-8"))
            needles.add(spelling.encode("utf-16-le"))
    for path in paths:
        data = path.read_bytes()
        for needle in needles:
            if needle and needle in data:
                raise RuntimeError(f"absolute checkout/build path leaked into {path}")


def isolated_distribution_environment(prefix_bin: Path, llvm_root: Path) -> dict[str, str]:
    """Return a runtime PATH that cannot borrow DLLs from LLVM_ROOT."""

    environment = os.environ.copy()
    environment.pop("LLVM_ROOT", None)
    retained = [prefix_bin.resolve()]
    for spelling in environment.get("PATH", "").split(os.pathsep):
        if not spelling:
            continue
        candidate = Path(spelling).resolve()
        try:
            candidate.relative_to(llvm_root.resolve())
            continue
        except ValueError:
            retained.append(candidate)

    deduplicated: list[str] = []
    seen: set[str] = set()
    for path in retained:
        identity = os.path.normcase(str(path))
        if identity in seen:
            continue
        seen.add(identity)
        deduplicated.append(str(path))
    environment["PATH"] = os.pathsep.join(deduplicated)
    return environment


def coff_import_names(readobj: Path, binary: Path) -> list[str]:
    imports = run([str(readobj), "--coff-imports", str(binary)]).stdout
    return sorted(
        set(
            re.findall(
                r"\bName:\s+([^\r\n\\/]+\.dll)\b", imports, re.IGNORECASE
            )
        ),
        key=str.casefold,
    )


def is_windows_system_dll(name: str) -> bool:
    lowered = name.casefold()
    if lowered.startswith(("api-ms-win-", "ext-ms-win-")):
        return True
    system_root = Path(os.environ.get("SystemRoot", r"C:\Windows"))
    return (system_root / "System32" / name).is_file()


def validate_recursive_import_closure(
    readobj: Path, prefix_bin: Path, roots: list[Path]
) -> dict[str, list[str]]:
    """Require every non-system PE import to be bundled, recursively."""

    bundled = {
        path.name.casefold(): path
        for path in prefix_bin.iterdir()
        if path.is_file() and path.suffix.casefold() in {".exe", ".dll"}
    }
    pending = list(roots)
    visited: set[str] = set()
    graph: dict[str, list[str]] = {}
    while pending:
        binary = pending.pop()
        identity = os.path.normcase(str(binary.resolve()))
        if identity in visited:
            continue
        visited.add(identity)
        names = coff_import_names(readobj, binary)
        graph[binary.relative_to(prefix_bin.parent).as_posix()] = names
        for name in names:
            bundled_dependency = bundled.get(name.casefold())
            if bundled_dependency is not None:
                pending.append(bundled_dependency)
                continue
            if is_windows_system_dll(name):
                continue
            raise RuntimeError(
                f"{binary} imports non-system DLL {name} that is absent from "
                f"the distribution; inherited PATH lookup is forbidden"
            )
    return dict(sorted(graph.items()))


def main() -> int:
    if os.name != "nt":
        raise RuntimeError("Windows distribution validation requires native Windows")

    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--build-dir", required=True)
    parser.add_argument("--install-prefix", required=True)
    parser.add_argument("--llvm-root", required=True)
    parser.add_argument("--report-out", required=True)
    parser.add_argument("--expected-clang-version", default="21.1.8")
    arguments = parser.parse_args()

    source_root = Path(arguments.source_root).resolve()
    build_dir = Path(arguments.build_dir).resolve()
    prefix = Path(arguments.install_prefix).resolve()
    llvm_root = Path(arguments.llvm_root).resolve()
    report_out = Path(arguments.report_out).resolve()

    clang_cl = require_file(llvm_root / "bin" / "clang-cl.exe")
    llvm_readobj = require_file(llvm_root / "bin" / "llvm-readobj.exe")
    llvm_objdump = require_file(llvm_root / "bin" / "llvm-objdump.exe")
    llvm_nm = require_file(llvm_root / "bin" / "llvm-nm.exe")

    clang_version = run([str(clang_cl), "--version"]).stdout
    if f"clang version {arguments.expected_clang_version}" not in clang_version:
        raise RuntimeError(f"unexpected clang-cl version:\n{clang_version}")

    expected = {
        "driver": prefix / "bin" / "mdslc++.exe",
        "extractor": prefix / "bin" / "matcore-extract.exe",
        "planner": prefix / "bin" / "matcore-plan.exe",
        "benchmark": prefix / "bin" / "matcore-bench.exe",
        "runtime_dll": prefix / "bin" / "matcore_runtime.dll",
        "runtime_import_library": prefix / "lib" / "matcore_runtime.lib",
        "mdsl_header": prefix / "include" / "matcore" / "mdsl.h",
        "runtime_header": prefix / "include" / "matcore" / "runtime_c.h",
        "package_config": prefix / "lib" / "cmake" / "MatcoreDSL" / "MatcoreDSLConfig.cmake",
        "package_targets": prefix / "lib" / "cmake" / "MatcoreDSL" / "MatcoreDSLTargets.cmake",
        "package_compile_helper": prefix / "lib" / "cmake" / "MatcoreDSL" / "MatcoreDSLCompile.cmake",
    }
    for path in expected.values():
        require_file(path)

    # Every bundled third-party DLL must be byte-identical to a file from the
    # authenticated LLVM archive. The Matcore runtime is the sole project DLL.
    authenticated_runtime_dlls: dict[str, str] = {}
    for bundled_dll in sorted((prefix / "bin").glob("*.dll")):
        if bundled_dll.name.casefold() == "matcore_runtime.dll":
            continue
        authenticated = llvm_root / "bin" / bundled_dll.name
        if not authenticated.is_file():
            raise RuntimeError(
                f"bundled third-party DLL has no authenticated archive source: "
                f"{bundled_dll.name}"
            )
        bundled_hash = sha256(bundled_dll)
        authenticated_hash = sha256(authenticated)
        if bundled_hash != authenticated_hash:
            raise RuntimeError(
                f"bundled DLL differs from authenticated archive: {bundled_dll.name}"
            )
        authenticated_runtime_dlls[bundled_dll.name] = bundled_hash

    executable_environment = isolated_distribution_environment(
        prefix / "bin", llvm_root
    )

    coff_evidence: dict[str, str] = {}
    for key in ("driver", "extractor", "planner", "benchmark", "runtime_dll"):
        coff_evidence[key] = assert_coff_x64(llvm_readobj, expected[key])

    header_text = expected["runtime_header"].read_text(encoding="utf-8")
    declared_exports = set(EXPORT_PATTERN.findall(header_text))
    export_output = run(
        [str(llvm_readobj), "--coff-exports", str(expected["runtime_dll"])]
    ).stdout
    missing_exports = sorted(
        symbol for symbol in declared_exports if symbol not in export_output
    )
    if missing_exports:
        raise RuntimeError(f"runtime DLL is missing declared C exports: {missing_exports}")

    imported_dlls = validate_recursive_import_closure(
        llvm_readobj,
        prefix / "bin",
        [
            expected["driver"],
            expected["extractor"],
            expected["planner"],
            expected["benchmark"],
            expected["runtime_dll"],
        ],
    )

    backend_archives = sorted((build_dir / "lib").glob("matcore_cpu_backends_v1*.lib"))
    if len(backend_archives) != 1:
        raise RuntimeError(f"expected one CPU backend archive, found {backend_archives}")
    backend_archive = backend_archives[0]
    archive_symbols = run(
        [str(llvm_nm), "--defined-only", "--format=posix", str(backend_archive)]
    ).stdout
    for symbol in (AVX2_SYMBOL, AVX512_SYMBOL):
        if symbol not in archive_symbols:
            raise RuntimeError(f"CPU backend archive lacks {symbol}")
    isa_evidence = {
        "avx2": inspect_microkernel(llvm_objdump, backend_archive, AVX2_SYMBOL, "ymm"),
        "avx512": inspect_microkernel(llvm_objdump, backend_archive, AVX512_SYMBOL, "zmm"),
    }

    platform_result = run(
        [str(expected["planner"]), "--platform-info"],
        environment=executable_environment,
    )
    platform_output = platform_result.stdout + platform_result.stderr
    for marker in ("platform=windows", "object=coff", "executable=pe", "runtime=windows-dll"):
        if marker not in platform_output:
            raise RuntimeError(f"Windows platform diagnostics lack {marker}:\n{platform_output}")

    frontend_info = run(
        [str(expected["extractor"]), "--frontend-info"],
        environment=executable_environment,
    ).stdout
    if "default: native" not in frontend_info or "native [built]" not in frontend_info:
        raise RuntimeError(
            "installed extractor did not load from the isolated distribution "
            f"or lost native frontend support:\n{frontend_info}"
        )

    runtime_test = require_file(build_dir / "bin" / "matcore_runtime_cpu_test.exe")
    runtime_output = run(
        [str(runtime_test)], environment=executable_environment
    ).stdout

    package_files = list((prefix / "lib" / "cmake" / "MatcoreDSL").glob("*.cmake"))
    binary_files = [expected[key] for key in ("driver", "extractor", "planner", "benchmark", "runtime_dll")]
    # The distribution must not remember either the checkout/build roots or
    # the runner-local authenticated LLVM extraction. Installed tools discover
    # clang-cl and any bundled LLVM runtime from their deployed layout/PATH.
    scan_for_path_leaks(
        package_files + binary_files, [source_root, build_dir, llvm_root]
    )

    inventory: list[dict[str, Any]] = []
    for path in sorted(path for path in prefix.rglob("*") if path.is_file()):
        inventory.append(
            {
                "path": path.relative_to(prefix).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )

    report = {
        "schema": "matcore-windows-artifact-report-v1",
        "windows": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "clang_cl": clang_version.strip(),
        "msvc_tools": os.environ.get("VCToolsInstallDir", "unknown"),
        "windows_sdk": os.environ.get("WindowsSdkDir", "unknown"),
        "windows_sdk_version": os.environ.get("WindowsSDKVersion", "unknown"),
        "linker": shutil.which("lld-link") or shutil.which("link") or "unknown",
        "platform_diagnostics": platform_output.strip(),
        "frontend_diagnostics": frontend_info.strip(),
        "runtime_output": runtime_output.strip(),
        "declared_c_exports": sorted(declared_exports),
        "imported_dlls": imported_dlls,
        "authenticated_llvm_runtime_dlls": authenticated_runtime_dlls,
        "isa_artifact_evidence": isa_evidence,
        "artifacts": inventory,
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "Windows x64 distribution validation PASS: "
        f"{len(inventory)} files, {len(declared_exports)} C exports"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
