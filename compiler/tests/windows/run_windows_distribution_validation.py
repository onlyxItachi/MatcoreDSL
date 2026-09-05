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
import tempfile
from typing import Any
import xml.etree.ElementTree as ET


EXPORT_PATTERN = re.compile(
    r"MATCORE_RUNTIME_API\s+matcore_status_v0\s+"
    r"(matcore_runtime_[A-Za-z0-9_]+)\s*\(",
    re.MULTILINE,
)
AVX2_CHECKED_SYMBOL = "matcore_cpu_packed_avx2_4x16_microkernel_f32_v1"
AVX2_FULL_SYMBOL = "matcore_cpu_packed_avx2_4x16_full_microkernel_f32_v2"
AVX512_CHECKED_SYMBOL = "matcore_cpu_packed_avx512_4x16_microkernel_f32_v1"
AVX512_FULL_SYMBOL = (
    "matcore_internal_cpu_packed_avx512_4x32_full_microkernel_f32_m7"
)
STABLE_VARIANTS = (
    "cpu.reference.f32.v1",
    "cpu.tiled.f32.v1",
    "cpu.compiler-vectorized.avx2-fma.f32.v1",
    "cpu.external.openblas.f32.v1",
    "cpu.native-packed.avx2-fma.f32.v1",
    "cpu.native-packed.avx512-fma.f32.v1",
    "cpu.native-parallel.avx2-fma.f32.v1",
    "cpu.native-parallel.avx512-fma.f32.v1",
)
CPU_FEATURE_BITS_V2 = {
    "portable-scalar-f32": 1 << 0,
    "avx2": 1 << 1,
    "fma": 1 << 2,
    "avx512f": 1 << 3,
    "avx512dq": 1 << 4,
    "avx512bw": 1 << 5,
    "avx512vl": 1 << 6,
    "avx512vnni": 1 << 7,
    "avx512bf16": 1 << 8,
    "amx-tile": 1 << 9,
    "amx-bf16": 1 << 10,
    "amx-int8": 1 << 11,
}
MSVC_RUNTIME_PREREQUISITE_ID = (
    "microsoft-visual-cpp-2015-2022-redistributable-x64"
)
MSVC_RUNTIME_IMPORT_FAMILIES = {
    "vcruntime140*.dll",
    "msvcp140*.dll",
    "concrt140.dll",
    "vcomp140.dll",
}


def run(
    command: list[str],
    *,
    environment: dict[str, str] | None = None,
    expect_success: bool = True,
    working_directory: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        check=False,
        text=True,
        capture_output=True,
        env=environment,
        cwd=working_directory,
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


def inspect_microkernel(
    objdump: Path,
    archive: Path,
    symbol: str,
    register: str,
    *,
    require_no_vector_stack_spill: bool = False,
) -> dict[str, int]:
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
    vector_stack_spills = [
        line
        for line in output.splitlines()
        if re.search(rf"\b{register}[0-9]+\b", line, re.IGNORECASE)
        and (
            re.search(r"\([^)]*%(?:r|e)sp\)", line, re.IGNORECASE)
            or re.search(
                r"\[[^\]]*\b(?:r|e)sp\b[^\]]*\]",
                line,
                re.IGNORECASE,
            )
        )
    ]
    if require_no_vector_stack_spill and vector_stack_spills:
        raise RuntimeError(
            f"{symbol} contains vector stack spill/reload instructions: "
            f"{vector_stack_spills}"
        )
    return {
        "register_operands": register_count,
        "packed_fma_sites": fma_count,
        "vector_stack_spills": len(vector_stack_spills),
    }


def scan_for_path_leaks(paths: list[Path], forbidden: list[Path]) -> int:
    needles: dict[bytes, str] = {}
    for path in forbidden:
        for spelling in (str(path), path.as_posix()):
            for candidate in (spelling, spelling.casefold()):
                needles[candidate.encode("utf-8")] = f"UTF-8 {candidate!r}"
                needles[candidate.encode("utf-16-le")] = (
                    f"UTF-16LE {candidate!r}"
                )
                needles[candidate.encode("utf-16-be")] = (
                    f"UTF-16BE {candidate!r}"
                )
    scanned = 0
    for path in paths:
        data = path.read_bytes()
        folded_data = data.lower()
        for needle, description in needles.items():
            if needle and needle in data:
                raise RuntimeError(
                    f"absolute checkout/build path leaked into {path} "
                    f"as {description}"
                )
            # Windows paths are case-insensitive. ASCII case folding is safe
            # for UTF-8 path bytes and catches the overwhelmingly common drive
            # letter/directory casing changes. Explicit case-folded UTF-16
            # needles above cover wide-string storage without corrupting it.
            if needle and needle == needle.lower() and needle in folded_data:
                raise RuntimeError(
                    f"case-folded absolute checkout/build path leaked into "
                    f"{path} as {description}"
                )
        scanned += 1
    return scanned


def output_line(output: str, prefix: str) -> str:
    matches = [line.strip() for line in output.splitlines() if line.startswith(prefix)]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one {prefix!r} diagnostic, found {len(matches)}:\n"
            f"{output}"
        )
    return matches[0]


def parse_capability_record(line: str) -> dict[str, Any]:
    match = re.fullmatch(
        r"cpu-capabilities-v2\{version=(\d+),architecture=([^,]+),"
        r"usable-vector-bits=(\d+),xstate=([^,]+),amx-permission=([^,]+),"
        r"features=\[(.*)\]\}",
        line,
    )
    if match is None:
        raise RuntimeError(f"malformed capability-v2 diagnostic: {line}")
    feature_records: dict[str, dict[str, str]] = {}
    for encoded in match.group(6).split(","):
        feature = re.fullmatch(
            r"([^:]+):hardware=([^/]+)/os=([^/]+)/compiler=([^/]+)/"
            r"implementation=([^/]+)/runtime=(.+)",
            encoded,
        )
        if feature is None:
            raise RuntimeError(f"malformed capability-v2 feature: {encoded}")
        name = feature.group(1)
        if name not in CPU_FEATURE_BITS_V2:
            raise RuntimeError(f"unknown capability-v2 feature: {name}")
        feature_records[name] = {
            "hardware": feature.group(2),
            "os": feature.group(3),
            "compiler": feature.group(4),
            "implementation": feature.group(5),
            "runtime": feature.group(6),
        }
    if set(feature_records) != set(CPU_FEATURE_BITS_V2):
        raise RuntimeError("capability-v2 diagnostic omitted a versioned feature")
    return {
        "version": int(match.group(1)),
        "architecture": match.group(2),
        "usable_vector_bits": int(match.group(3)),
        "xstate": match.group(4),
        "amx_permission": match.group(5),
        "features": feature_records,
        "raw": line,
    }


def parse_topology_record(line: str) -> dict[str, Any]:
    match = re.fullmatch(
        r"cpu-topology-v1\{version=(\d+),architecture=([^,]+),"
        r"discovery=([^,]+),logical-cpus=(\d+),physical-cores=(\d+),"
        r"sockets=(\d+),numa-nodes=(\d+),cpu-map=\[(.*)\],"
        r"cache-groups=(\d+)\}",
        line,
    )
    if match is None:
        raise RuntimeError(f"malformed topology-v1 diagnostic: {line}")
    return {
        "version": int(match.group(1)),
        "architecture": match.group(2),
        "discovery": match.group(3),
        "logical_cpus": int(match.group(4)),
        "physical_cores": int(match.group(5)),
        "sockets": int(match.group(6)),
        "numa_nodes": int(match.group(7)),
        "cpu_map": match.group(8),
        "cache_groups": int(match.group(9)),
        "raw": line,
    }


def parse_execution_policy(line: str) -> dict[str, Any]:
    requested = re.search(r"\brequested-threads=(\d+)", line)
    actual = re.search(r"\bactual-workers=(\d+)", line)
    openblas = re.search(r"\bopenblas-linked=(true|false)", line)
    openblas_version = re.search(r"\bopenblas-version=(.*)$", line)
    if None in (requested, actual, openblas, openblas_version):
        raise RuntimeError(f"malformed execution-policy diagnostic: {line}")
    assert requested is not None
    assert actual is not None
    assert openblas is not None
    assert openblas_version is not None
    return {
        "requested_threads": int(requested.group(1)),
        "actual_workers": int(actual.group(1)),
        "openblas_linked": openblas.group(1) == "true",
        "openblas_version": openblas_version.group(1),
        "raw": line,
    }


def parse_candidate(encoded: str) -> dict[str, Any]:
    match = re.fullmatch(
        r"(cpu\.[^:]+):(legal|rejected):reason=(.*):cost=(\d+):priority=(\d+):"
        r"workspace=(\d+):shared-workspace=(\d+):per-worker-workspace=(\d+):"
        r"alignment=(\d+):threads=(\d+):thread-ceiling=(\d+):"
        r"row-tasks=(\d+):column-tasks=(\d+):task-count=(\d+):"
        r"thread-capacity-limited=(true|false):runtime-validated=(true|false):"
        r"required-hardware=(\d+):required-os=(\d+):required-compiler=(\d+):"
        r"required-implementation=(\d+):cross-numa=(true|false)",
        encoded,
    )
    if match is None:
        raise RuntimeError(f"malformed planner candidate diagnostic: {encoded}")
    return {
        "stable_id": match.group(1),
        "legality": match.group(2),
        "reason": match.group(3),
        "estimated_cost": int(match.group(4)),
        "priority": int(match.group(5)),
        "workspace_bytes": int(match.group(6)),
        "shared_workspace_bytes": int(match.group(7)),
        "per_worker_workspace_bytes": int(match.group(8)),
        "workspace_alignment": int(match.group(9)),
        "threads": int(match.group(10)),
        "thread_ceiling": int(match.group(11)),
        "row_tasks": int(match.group(12)),
        "column_tasks": int(match.group(13)),
        "task_count": int(match.group(14)),
        "thread_capacity_limited": match.group(15) == "true",
        "runtime_validated": match.group(16) == "true",
        "required_hardware_mask": int(match.group(17)),
        "required_os_mask": int(match.group(18)),
        "required_compiler_mask": int(match.group(19)),
        "required_implementation_mask": int(match.group(20)),
        "cross_numa": match.group(21) == "true",
    }


def parse_plan_record(line: str) -> dict[str, Any]:
    match = re.search(
        r"\bstatus=([^ ]+) selected=([^ ]+) reason=(.*) "
        r"candidates=\[(.*)\]$",
        line,
    )
    if match is None:
        raise RuntimeError(f"malformed planner-v3 diagnostic: {line}")
    encoded_candidates = re.split(r",(?=cpu\.)", match.group(4))
    candidates = [parse_candidate(candidate) for candidate in encoded_candidates]
    ids = [candidate["stable_id"] for candidate in candidates]
    if tuple(ids) != STABLE_VARIANTS:
        raise RuntimeError(f"planner registry changed or is incomplete: {ids}")
    return {
        "status": match.group(1),
        "selected": match.group(2),
        "reason": match.group(3),
        "candidates": candidates,
        "raw": line,
    }


def implementation_available_mask(capabilities: dict[str, Any]) -> int:
    mask = 0
    for name, bit in CPU_FEATURE_BITS_V2.items():
        if capabilities["features"][name]["implementation"] == "yes":
            mask |= bit
    return mask


def collect_plan_evidence(
    planner: Path, environment: dict[str, str]
) -> dict[str, Any]:
    cases = (
        ("reference", "cpu.reference.f32.v1", 1, True),
        ("automatic", "auto", 2, True),
        ("forced-packed-avx2", "cpu.native-packed.avx2-fma.f32.v1", 1, False),
        ("forced-packed-avx512", "cpu.native-packed.avx512-fma.f32.v1", 1, False),
        ("forced-parallel-avx2", "cpu.native-parallel.avx2-fma.f32.v1", 2, False),
        ("forced-parallel-avx512", "cpu.native-parallel.avx512-fma.f32.v1", 2, False),
    )
    records: list[dict[str, Any]] = []
    canonical_capability: dict[str, Any] | None = None
    canonical_topology: dict[str, Any] | None = None
    canonical_policy: dict[str, Any] | None = None
    for name, variant, threads, must_select in cases:
        command = [
            str(planner),
            "--m", "257",
            "--k", "259",
            "--n", "255",
            "--alignment", "64",
            "--threads", str(threads),
            "--variant", variant,
        ]
        result = run(command, environment=environment, expect_success=False)
        if result.returncode not in (0, 1):
            raise RuntimeError(
                f"installed planner case {name} returned unexpected status "
                f"{result.returncode}:\n{result.stdout}\n{result.stderr}"
            )
        output = result.stdout + result.stderr
        capability = parse_capability_record(output_line(output, "cpu-capabilities-v2{"))
        topology = parse_topology_record(output_line(output, "cpu-topology-v1{"))
        policy = parse_execution_policy(
            output_line(output, "cpu-execution-policy-v1 ")
        )
        plan = parse_plan_record(output_line(output, "cpu-planner-v3 "))
        if must_select and (result.returncode != 0 or plan["status"] != "selected"):
            raise RuntimeError(f"required installed planner case {name} was rejected:\n{output}")
        if (result.returncode == 0) != (plan["status"] == "selected"):
            raise RuntimeError(
                f"installed planner case {name} status disagrees with its exit code"
            )
        if canonical_capability is None:
            canonical_capability = capability
            canonical_topology = topology
            canonical_policy = policy
        else:
            if capability != canonical_capability:
                raise RuntimeError("capability-v2 record changed between planner cases")
            if topology != canonical_topology:
                raise RuntimeError("topology-v1 record changed between planner cases")
        records.append(
            {
                "case": name,
                "requested_variant": variant,
                "return_code": result.returncode,
                "explicit_status": (
                    "selected" if result.returncode == 0 else "unavailable"
                ),
                "execution_policy": policy,
                "plan": plan,
            }
        )

    assert canonical_capability is not None
    assert canonical_topology is not None
    assert canonical_policy is not None
    baseline_candidates = records[1]["plan"]["candidates"]
    implementation_mask = implementation_available_mask(canonical_capability)
    registry: list[dict[str, Any]] = []
    for candidate in baseline_candidates:
        required = candidate["required_implementation_mask"]
        compiled = (required & implementation_mask) == required
        if candidate["stable_id"] == "cpu.external.openblas.f32.v1":
            compiled = canonical_policy["openblas_linked"]
        registry.append(
            {
                "stable_id": candidate["stable_id"],
                "implementation_compiled": compiled,
                "compile_evidence": (
                    "openblas-linked execution-policy field"
                    if candidate["stable_id"] == "cpu.external.openblas.f32.v1"
                    else "capability-v2 implementation domain"
                ),
                # This value comes only from planner runtime-validation
                # evidence. ISA disassembly is deliberately not used here.
                "runtime_validated": candidate["runtime_validated"],
                "automatic_legality": candidate["legality"],
                "automatic_rejection_reason": candidate["reason"],
            }
        )
    compiled_variants = [
        entry["stable_id"] for entry in registry if entry["implementation_compiled"]
    ]
    runtime_validated_variants = [
        entry["stable_id"] for entry in registry if entry["runtime_validated"]
    ]
    unavailable_variants = [
        {
            "stable_id": entry["stable_id"],
            "reason": entry["automatic_rejection_reason"],
        }
        for entry in registry
        if entry["automatic_legality"] == "rejected"
    ]
    return {
        "capability_v2": canonical_capability,
        "topology_v1": canonical_topology,
        "variant_registry": registry,
        "compiled_variants": compiled_variants,
        "runtime_validated_variants": runtime_validated_variants,
        "unavailable_variants": unavailable_variants,
        "cases": records,
        "runtime_validation_source": "installed matcore-plan planner evidence",
        "isa_disassembly_role": "compile-time artifact evidence only",
    }


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


def is_debug_msvc_runtime_dll(name: str) -> bool:
    return (
        re.fullmatch(
            r"(?:msvcp|vcruntime|concrt|vcomp)\d+(?:_\d+)?d(?:_[^.]+)?\.dll",
            name,
            re.IGNORECASE,
        )
        is not None
    )


def is_release_msvc_runtime_dll(name: str) -> bool:
    if is_debug_msvc_runtime_dll(name):
        return False
    return (
        re.fullmatch(
            r"(?:vcruntime140[^.]*|msvcp140[^.]*|concrt140|vcomp140)\.dll",
            name,
            re.IGNORECASE,
        )
        is not None
    )


def load_runtime_prerequisite_manifest(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid Windows runtime prerequisite manifest: {path}") from error
    if manifest.get("schema") != "matcore-windows-runtime-prerequisites-v1":
        raise RuntimeError("unexpected Windows runtime prerequisite manifest schema")
    if manifest.get("target") != "x86_64-pc-windows-msvc":
        raise RuntimeError("Windows runtime prerequisite manifest targets the wrong ABI")
    if manifest.get("release_runtime_model") != "dynamic-msvc-runtime":
        raise RuntimeError("Windows distribution runtime model is not declared accurately")
    prerequisites = manifest.get("external_prerequisites")
    if not isinstance(prerequisites, list):
        raise RuntimeError("Windows runtime prerequisite manifest lacks its provider list")
    msvc = next(
        (
            item
            for item in prerequisites
            if isinstance(item, dict)
            and item.get("id") == MSVC_RUNTIME_PREREQUISITE_ID
        ),
        None,
    )
    if msvc is None:
        raise RuntimeError("Windows distribution omits the required VC runtime prerequisite")
    if (
        msvc.get("provider") != "Microsoft"
        or msvc.get("architecture") != "x64"
        or msvc.get("distribution_status") != "not-bundled"
        or set(msvc.get("required_import_families", []))
        != MSVC_RUNTIME_IMPORT_FAMILIES
    ):
        raise RuntimeError("Windows VC runtime prerequisite declaration is incomplete")
    validation = manifest.get("validation")
    if (
        not isinstance(validation, dict)
        or validation.get("release_only") is not True
        or validation.get("debug_crt_allowed_in_distribution") is not False
    ):
        raise RuntimeError("Windows runtime manifest does not fail closed on Debug CRT")
    return manifest


def parse_ctest_junit(path: Path, configuration: str) -> dict[str, Any]:
    require_file(path)
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as error:
        raise RuntimeError(f"invalid {configuration} CTest JUnit report: {path}") from error
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    if not suites:
        raise RuntimeError(f"{configuration} CTest JUnit report has no test suite")

    totals = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0, "disabled": 0}
    testcase_count = 0
    for suite in suites:
        for field in totals:
            encoded = suite.get(field, "0")
            try:
                value = int(encoded)
            except ValueError as error:
                raise RuntimeError(
                    f"{configuration} CTest JUnit has invalid {field}={encoded!r}"
                ) from error
            if value < 0:
                raise RuntimeError(
                    f"{configuration} CTest JUnit has negative {field} count"
                )
            totals[field] += value
        testcase_count += len(suite.findall("testcase"))
    if totals["tests"] == 0 or testcase_count != totals["tests"]:
        raise RuntimeError(
            f"{configuration} CTest JUnit count mismatch: "
            f"declared={totals['tests']} cases={testcase_count}"
        )
    if totals["failures"] != 0 or totals["errors"] != 0:
        raise RuntimeError(
            f"{configuration} hosted CTest did not pass: "
            f"failures={totals['failures']} errors={totals['errors']}"
        )
    passed = (
        totals["tests"]
        - totals["failures"]
        - totals["errors"]
        - totals["skipped"]
        - totals["disabled"]
    )
    if passed < 0:
        raise RuntimeError(f"{configuration} CTest JUnit counts are inconsistent")
    return {
        "configuration": configuration,
        "status": "pass" if totals["skipped"] == 0 else "pass-with-skips",
        "tests": totals["tests"],
        "passed": passed,
        "failures": totals["failures"],
        "errors": totals["errors"],
        "skipped": totals["skipped"],
        "disabled": totals["disabled"],
        "source": "hosted CTest --output-junit",
    }


def is_windows_system_dll(name: str) -> bool:
    lowered = name.casefold()
    # The VC runtime is a separately installed redistributable prerequisite,
    # not an operating-system component. Presence in this runner's System32
    # directory is not distribution evidence.
    if is_release_msvc_runtime_dll(name) or is_debug_msvc_runtime_dll(name):
        return False
    if lowered.startswith(("api-ms-win-", "ext-ms-win-")):
        return True
    system_root = Path(os.environ.get("SystemRoot", r"C:\Windows"))
    return (system_root / "System32" / name).is_file()


def validate_recursive_import_closure(
    readobj: Path, prefix_bin: Path, roots: list[Path]
) -> tuple[dict[str, list[str]], list[str]]:
    """Require every non-system PE import to be bundled, recursively."""

    bundled = {
        path.name.casefold(): path
        for path in prefix_bin.iterdir()
        if path.is_file() and path.suffix.casefold() in {".exe", ".dll"}
    }
    pending = list(roots)
    visited: set[str] = set()
    graph: dict[str, list[str]] = {}
    external_msvc_runtime_imports: set[str] = set()
    while pending:
        binary = pending.pop()
        identity = os.path.normcase(str(binary.resolve()))
        if identity in visited:
            continue
        visited.add(identity)
        names = coff_import_names(readobj, binary)
        try:
            graph_key = binary.relative_to(prefix_bin.parent).as_posix()
        except ValueError:
            graph_key = f"isolated/{binary.name}"
        graph[graph_key] = names
        for name in names:
            if is_debug_msvc_runtime_dll(name):
                raise RuntimeError(
                    f"{binary} imports forbidden Debug CRT dependency {name}"
                )
            if is_release_msvc_runtime_dll(name):
                if name.casefold() in bundled:
                    raise RuntimeError(
                        f"{binary} resolves VC runtime {name} from the bundle, "
                        "but the installed prerequisite manifest declares it "
                        "not-bundled"
                    )
                external_msvc_runtime_imports.add(name)
                continue
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
    return (
        dict(sorted(graph.items())),
        sorted(external_msvc_runtime_imports, key=str.casefold),
    )


def main() -> int:
    if os.name != "nt":
        raise RuntimeError("Windows distribution validation requires native Windows")

    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--build-dir", required=True)
    parser.add_argument("--install-prefix", required=True)
    parser.add_argument("--llvm-root", required=True)
    parser.add_argument("--report-out", required=True)
    parser.add_argument("--release-ctest-junit", required=True)
    parser.add_argument("--debug-ctest-junit", required=True)
    parser.add_argument("--expected-clang-version", default="21.1.8")
    arguments = parser.parse_args()

    source_root = Path(arguments.source_root).resolve()
    build_dir = Path(arguments.build_dir).resolve()
    prefix = Path(arguments.install_prefix).resolve()
    llvm_root = Path(arguments.llvm_root).resolve()
    report_out = Path(arguments.report_out).resolve()
    release_ctest = parse_ctest_junit(
        Path(arguments.release_ctest_junit).resolve(), "Release"
    )
    debug_ctest = parse_ctest_junit(
        Path(arguments.debug_ctest_junit).resolve(), "Debug"
    )

    clang_cl = require_file(llvm_root / "bin" / "clang-cl.exe")
    llvm_readobj = require_file(llvm_root / "bin" / "llvm-readobj.exe")
    llvm_objdump = require_file(llvm_root / "bin" / "llvm-objdump.exe")
    llvm_nm = require_file(llvm_root / "bin" / "llvm-nm.exe")

    clang_version = run([str(clang_cl), "--version"]).stdout
    clang_version_match = re.search(
        r"clang version ([^\s(]+)", clang_version
    )
    if (
        clang_version_match is None
        or clang_version_match.group(1) != arguments.expected_clang_version
    ):
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
        "runtime_prerequisites": prefix / "share" / "MatcoreDSL" / "MatcoreDSLWindowsRuntimePrerequisites.json",
    }
    for path in expected.values():
        require_file(path)
    runtime_prerequisites = load_runtime_prerequisite_manifest(
        expected["runtime_prerequisites"]
    )

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
    for key in (
        "driver",
        "extractor",
        "planner",
        "benchmark",
        "runtime_dll",
        "runtime_import_library",
    ):
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

    imported_dlls, external_msvc_runtime_imports = validate_recursive_import_closure(
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
    for symbol in (
        AVX2_CHECKED_SYMBOL,
        AVX2_FULL_SYMBOL,
        AVX512_CHECKED_SYMBOL,
        AVX512_FULL_SYMBOL,
    ):
        if symbol not in archive_symbols:
            raise RuntimeError(f"CPU backend archive lacks {symbol}")
    isa_evidence = {
        "avx2_checked": inspect_microkernel(
            llvm_objdump, backend_archive, AVX2_CHECKED_SYMBOL, "ymm"
        ),
        "avx2_full": inspect_microkernel(
            llvm_objdump,
            backend_archive,
            AVX2_FULL_SYMBOL,
            "ymm",
            require_no_vector_stack_spill=True,
        ),
        "avx512_checked": inspect_microkernel(
            llvm_objdump, backend_archive, AVX512_CHECKED_SYMBOL, "zmm"
        ),
        "avx512_full": inspect_microkernel(
            llvm_objdump,
            backend_archive,
            AVX512_FULL_SYMBOL,
            "zmm",
            require_no_vector_stack_spill=True,
        ),
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

    # Execute the installed planner rather than equating compile-time ISA
    # evidence with runtime support. Required portable cases must select. Each
    # forced native case is recorded as either selected or explicitly
    # unavailable, with the complete planner reason retained.
    planner_evidence = collect_plan_evidence(
        expected["planner"], executable_environment
    )

    runtime_test = require_file(build_dir / "bin" / "matcore_runtime_cpu_test.exe")
    # Prove the test resolves the installed runtime DLL. Running the build-tree
    # executable in build_dir/bin would let Windows' same-directory DLL search
    # choose the build-tree matcore_runtime.dll before the isolated PATH.
    # Stage only the EXE in a new directory, authenticate its recursive imports
    # against prefix/bin, and execute it there.
    with tempfile.TemporaryDirectory(
        prefix="matcore-installed-runtime-proof-", dir=report_out.parent
    ) as isolated_runtime_directory:
        isolated_runtime_root = Path(isolated_runtime_directory)
        isolated_runtime_test = isolated_runtime_root / runtime_test.name
        shutil.copy2(runtime_test, isolated_runtime_test)
        (
            isolated_runtime_imports,
            isolated_runtime_msvc_imports,
        ) = validate_recursive_import_closure(
            llvm_readobj, prefix / "bin", [isolated_runtime_test]
        )
        runtime_output = run(
            [str(isolated_runtime_test)],
            environment=executable_environment,
            working_directory=isolated_runtime_root,
        ).stdout
    external_msvc_runtime_imports = sorted(
        set(external_msvc_runtime_imports) | set(isolated_runtime_msvc_imports),
        key=str.casefold,
    )
    runtime_test_evidence = {
        "executable_source": "Release build-tree test executable copied alone",
        "launch_directory": "isolated temporary directory without adjacent DLLs",
        "runtime_resolution": "installed prefix/bin through isolated PATH",
        "recursive_imports": isolated_runtime_imports,
        "status": "pass",
        "output": runtime_output.strip(),
    }

    installed_files = sorted(path for path in prefix.rglob("*") if path.is_file())
    # The distribution must not remember either the checkout/build roots or
    # the runner-local authenticated LLVM extraction. Installed tools discover
    # clang-cl and any bundled LLVM runtime from their deployed layout/PATH.
    # Scan every regular payload file, not merely the CMake files and primary
    # executables; these are exactly the bytes archived into the ZIP later in
    # the workflow. UTF-8 and both UTF-16 byte orders are checked.
    leak_scanned_files = scan_for_path_leaks(
        installed_files, [source_root, build_dir, llvm_root]
    )

    inventory: list[dict[str, Any]] = []
    for path in installed_files:
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
        "hosted_ctest": {
            "release": release_ctest,
            "debug": debug_ctest,
        },
        "planner_evidence": planner_evidence,
        "installed_runtime_test": runtime_test_evidence,
        "declared_c_exports": sorted(declared_exports),
        "imported_dlls": imported_dlls,
        "external_runtime_prerequisites": {
            "manifest": runtime_prerequisites,
            "detected_msvc_runtime_imports": external_msvc_runtime_imports,
            "clean_machine_installation": "not validated",
        },
        "authenticated_llvm_runtime_dlls": authenticated_runtime_dlls,
        "isa_artifact_evidence": isa_evidence,
        "path_leak_scan": {
            "scope": "all regular files in installed prefix and ZIP payload",
            "encodings": ["UTF-8", "UTF-16LE", "UTF-16BE"],
            "files_scanned": leak_scanned_files,
            "forbidden_roots": 3,
            "status": "pass",
        },
        "validation_counts": {
            "hosted_release_ctest_total": release_ctest["tests"],
            "hosted_release_ctest_passed": release_ctest["passed"],
            "hosted_debug_ctest_total": debug_ctest["tests"],
            "hosted_debug_ctest_passed": debug_ctest["passed"],
            "installed_planner_cases": len(planner_evidence["cases"]),
            "forced_native_planner_cases": 4,
            "installed_runtime_test_programs": 1,
            "coff_pe_artifacts": len(coff_evidence),
            "declared_c_exports": len(declared_exports),
            "recursive_import_roots": len(imported_dlls),
            "external_msvc_runtime_imports": len(external_msvc_runtime_imports),
            "installed_files": len(inventory),
            "path_leak_scanned_files": leak_scanned_files,
        },
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
