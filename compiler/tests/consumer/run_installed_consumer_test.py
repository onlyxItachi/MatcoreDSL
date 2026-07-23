#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


VC_REDISTRIBUTABLE_PATTERN = re.compile(
    r"^(?:vcruntime140(?:_[0-9]+)?|msvcp140(?:_[0-9]+|_atomic_wait|_codecvt_ids)?|"
    r"concrt140|vcomp140)\.dll$",
    re.IGNORECASE,
)
DEBUG_C_RUNTIME_PATTERN = re.compile(
    r"^(?:vcruntime140d|msvcp140d|concrt140d|vcomp140d|ucrtbased)\.dll$",
    re.IGNORECASE,
)


def run(
    command: list[str],
    *,
    capture: bool = False,
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
        env=environment,
    )
    if result.returncode != 0:
        if result.stdout:
            print(result.stdout, file=sys.stderr)
        raise RuntimeError(f"command failed ({result.returncode}): {command}")
    return result


def make_depfile_paths(path: Path) -> tuple[str, ...]:
    def encode(value: str) -> str:
        value = value.replace("\\", "\\\\").replace("$", "$$")
        return (
            value.replace(" ", "\\ ")
            .replace("#", "\\#")
            .replace(":", "\\:")
        )

    return tuple({encode(str(path)), encode(path.as_posix())})


def is_beneath(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (OSError, ValueError):
        return False


def contains_frontend_toolchain(path: Path) -> bool:
    """Identify PATH entries capable of masking a missing packaged frontend DLL."""

    if not path.is_dir():
        return False
    probes = (
        "clang-cl.exe",
        "clang.exe",
        "clang-cpp.dll",
        "libclang.dll",
        "LLVM.dll",
        "LLVM-C.dll",
    )
    return any((path / probe).is_file() for probe in probes)


def windows_test_environment(prefix_bin: Path) -> dict[str, str]:
    """Build an explicit Windows test PATH without inherited LLVM leakage.

    The installed tools and runtime are tested with only their relocated bin
    directory plus the existing Windows/Visual Studio tool paths.  Compiler
    phases receive a separately staged compiler directory later. LLVM_ROOT and
    every inherited PATH entry containing a Clang/LLVM frontend are removed, so
    a missing packaged frontend DLL cannot be satisfied accidentally.
    """

    environment = os.environ.copy()
    llvm_root_text = environment.pop("LLVM_ROOT", "")
    llvm_root = Path(llvm_root_text).resolve() if llvm_root_text else None

    entries = [prefix_bin.resolve()]
    for spelling in environment.get("PATH", "").split(os.pathsep):
        if not spelling:
            continue
        candidate = Path(spelling).resolve()
        if llvm_root is not None and is_beneath(candidate, llvm_root):
            continue
        if contains_frontend_toolchain(candidate):
            continue
        entries.append(candidate)

    deduplicated: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        identity = os.path.normcase(str(entry))
        if identity in seen:
            continue
        seen.add(identity)
        deduplicated.append(str(entry))
    environment["PATH"] = os.pathsep.join(deduplicated)
    return environment


def coff_import_names(readobj: Path, binary: Path) -> list[str]:
    inspected = run(
        [str(readobj), "--coff-imports", str(binary)],
        capture=True,
    ).stdout
    return sorted(
        set(
            re.findall(
                r"\bName:\s+([^\r\n\\/]+\.dll)\b",
                inspected,
                re.IGNORECASE,
            )
        ),
        key=str.casefold,
    )


def classify_windows_import(name: str) -> str:
    """Classify an unbundled import without treating all System32 DLLs alike."""

    if DEBUG_C_RUNTIME_PATTERN.fullmatch(name):
        return "debug-c-runtime"
    if VC_REDISTRIBUTABLE_PATTERN.fullmatch(name):
        return "vc-redist-prerequisite"
    lowered = name.casefold()
    if lowered.startswith(("api-ms-win-", "ext-ms-win-")):
        return "windows-api-set"
    if lowered == "ucrtbase.dll":
        return "windows-ucrt"
    system_root = Path(os.environ.get("SystemRoot", r"C:\Windows"))
    if (system_root / "System32" / name).is_file():
        return "windows-system"
    return "unresolved"


def validate_recursive_windows_imports(
    readobj: Path,
    prefix_bin: Path,
    roots: list[Path],
) -> dict[str, object]:
    """Prove imports without consulting inherited PATH lookup."""

    bundled = {
        path.name.casefold(): path
        for path in prefix_bin.iterdir()
        if path.is_file() and path.suffix.casefold() in {".exe", ".dll"}
    }
    pending = list(roots)
    visited: set[str] = set()
    graph: dict[str, list[str]] = {}
    system_dependencies: set[str] = set()
    vc_redist_dependencies: set[str] = set()
    while pending:
        binary = pending.pop()
        identity = os.path.normcase(str(binary.resolve()))
        if identity in visited:
            continue
        visited.add(identity)
        imports = coff_import_names(readobj, binary)
        graph[str(binary)] = imports
        for name in imports:
            packaged = bundled.get(name.casefold())
            if packaged is not None:
                pending.append(packaged)
                continue
            classification = classify_windows_import(name)
            if classification == "vc-redist-prerequisite":
                vc_redist_dependencies.add(name)
                continue
            if classification.startswith("windows-"):
                system_dependencies.add(name)
                continue
            if classification == "debug-c-runtime":
                raise RuntimeError(
                    f"release distribution imports non-redistributable debug CRT {name}: "
                    f"{binary}"
                )
            raise RuntimeError(
                f"installed binary imports {name}, which is neither bundled nor an "
                f"explicit Windows/VC runtime prerequisite: {binary}"
            )
    return {
        "schema": "matcore-windows-consumer-import-report-v1",
        "roots": [str(path) for path in roots],
        "graph": dict(sorted(graph.items())),
        "windows_system_dependencies": sorted(system_dependencies, key=str.casefold),
        "vc_redist_prerequisites": sorted(vc_redist_dependencies, key=str.casefold),
    }


def stage_windows_compiler(compiler: Path, readobj: Path, test_root: Path) -> Path:
    """Stage clang-cl, archive helpers and builtin headers without LLVM DLLs."""

    stage_root = test_root / "compiler-only"
    stage_bin = stage_root / "bin"
    stage_bin.mkdir(parents=True)
    staged_compiler = stage_bin / compiler.name
    shutil.copy2(compiler, staged_compiler)

    staged_tools = [staged_compiler]
    for helper_name, required in (("llvm-lib.exe", True), ("llvm-config.exe", False)):
        helper = compiler.parent / helper_name
        if not helper.is_file():
            if required:
                raise RuntimeError(
                    f"compiler-only staging requires adjacent {helper_name}: {helper}"
                )
            continue
        staged_helper = stage_bin / helper_name
        shutil.copy2(helper, staged_helper)
        staged_tools.append(staged_helper)

    for staged_tool in staged_tools:
        tool_imports = coff_import_names(readobj, staged_tool)
        forbidden = [
            name
            for name in tool_imports
            if classify_windows_import(name)
            not in {
                "vc-redist-prerequisite",
                "windows-api-set",
                "windows-ucrt",
                "windows-system",
            }
        ]
        if forbidden:
            raise RuntimeError(
                f"{staged_tool.name} cannot be staged without non-system "
                f"toolchain DLLs: {forbidden}"
            )

    resource = run([str(compiler), "--print-resource-dir"], capture=True).stdout.strip()
    resource_dir = Path(resource).resolve()
    if not resource_dir.is_dir():
        raise RuntimeError(f"clang-cl reported a missing resource directory: {resource_dir}")
    staged_resource = stage_root / "lib" / "clang" / resource_dir.name
    shutil.copytree(resource_dir, staged_resource)
    return staged_compiler


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmake", required=True)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--producer-build-dir", required=True)
    parser.add_argument("--test-root", required=True)
    parser.add_argument("--cxx-compiler", required=True)
    parser.add_argument("--windows-import-report-out")
    args = parser.parse_args()
    is_windows = os.name == "nt"
    executable_suffix = ".exe" if is_windows else ""
    artifact_suffix = ".lib" if is_windows else ".o"

    test_root = Path(args.test_root).resolve()
    if test_root.exists():
        shutil.rmtree(test_root)
    source = test_root / "source"
    build = test_root / "build"
    staging_prefix = test_root / "install-staging"
    # This deliberately exercises both whitespace and Unicode through install,
    # trusted-header discovery, generated objects, linking, and execution.
    # Commas in an ELF runtime directory remain covered at the Linux driver
    # layer because CMake's compiler-driver rpath encoding cannot represent
    # such a path.
    prefix = test_root / "relocated" / "matcoredsl prefix ünicode"
    shutil.copytree(Path(args.source_dir).resolve(), source)

    run([
        args.cmake,
        "--install",
        str(Path(args.producer_build_dir).resolve()),
        "--prefix",
        str(staging_prefix),
    ])
    prefix.parent.mkdir(parents=True)
    shutil.move(staging_prefix, prefix)
    expected_install_files = [
        prefix / "bin" / f"mdslc++{executable_suffix}",
        prefix / "bin" / f"matcore-extract{executable_suffix}",
        prefix / "bin" / f"matcore-plan{executable_suffix}",
        prefix / "bin" / f"matcore-bench{executable_suffix}",
        prefix / "include" / "matcore" / "mdsl.h",
        prefix / "include" / "matcore" / "runtime_c.h",
        prefix / "lib" / "cmake" / "MatcoreDSL" / "MatcoreDSLConfig.cmake",
        prefix / "lib" / "cmake" / "MatcoreDSL" / "MatcoreDSLConfigVersion.cmake",
        prefix / "lib" / "cmake" / "MatcoreDSL" / "MatcoreDSLTargets.cmake",
        prefix / "lib" / "cmake" / "MatcoreDSL" / "MatcoreDSLCompile.cmake",
    ]
    if is_windows:
        expected_install_files.extend(
            [
                prefix / "bin" / "matcore_runtime.dll",
                prefix / "lib" / "matcore_runtime.lib",
                prefix
                / "share"
                / "MatcoreDSL"
                / "MatcoreDSLWindowsRuntimePrerequisites.json",
            ]
        )
    else:
        expected_install_files.append(prefix / "lib" / "libmatcore_runtime.so")
    missing = [path for path in expected_install_files if not path.exists()]
    if missing:
        raise RuntimeError(f"installed package is incomplete: {missing}")

    repository = Path(__file__).resolve().parents[3]
    source_public_header = repository / "compiler" / "include" / "matcore" / "mdsl.h"
    driver = prefix / "bin" / f"mdslc++{executable_suffix}"
    extractor = prefix / "bin" / f"matcore-extract{executable_suffix}"
    planner = prefix / "bin" / f"matcore-plan{executable_suffix}"
    benchmark = prefix / "bin" / f"matcore-bench{executable_suffix}"
    installed_environment = os.environ.copy()
    build_environment = os.environ.copy()
    # Preserve the caller-visible driver basename.  On Unix, clang++ is often
    # a symlink to the clang binary and Clang selects C++ link semantics from
    # argv[0]; resolving that final symlink turns a valid C++ driver into the C
    # driver and drops the C++ standard library from the consumer link.
    test_compiler = Path(os.path.abspath(args.cxx_compiler))
    windows_import_report: dict[str, object] | None = None
    if is_windows:
        llvm_readobj = test_compiler.parent / "llvm-readobj.exe"
        if not llvm_readobj.is_file():
            raise RuntimeError(
                f"Windows installed-consumer validation requires {llvm_readobj}"
            )

        prerequisite_manifest_path = (
            prefix
            / "share"
            / "MatcoreDSL"
            / "MatcoreDSLWindowsRuntimePrerequisites.json"
        )
        prerequisite_manifest = json.loads(
            prerequisite_manifest_path.read_text(encoding="utf-8")
        )
        if (
            prerequisite_manifest.get("schema")
            != "matcore-windows-runtime-prerequisites-v1"
            or prerequisite_manifest.get("release_runtime_model")
            != "dynamic-msvc-runtime"
            or not any(
                prerequisite.get("id")
                == "microsoft-visual-cpp-2015-2022-redistributable-x64"
                for prerequisite in prerequisite_manifest.get(
                    "external_prerequisites", []
                )
            )
        ):
            raise RuntimeError(
                f"installed Windows runtime prerequisite manifest is invalid: "
                f"{prerequisite_manifest_path}"
            )

        # This gate runs before any installed executable. It proves that no
        # frontend DLL can be borrowed from the compiler or inherited PATH.
        windows_import_report = validate_recursive_windows_imports(
            llvm_readobj,
            prefix / "bin",
            [
                driver,
                extractor,
                planner,
                benchmark,
                prefix / "bin" / "matcore_runtime.dll",
            ],
        )
        windows_import_report["declared_runtime_prerequisites"] = (
            prerequisite_manifest
        )
        test_compiler = stage_windows_compiler(
            Path(args.cxx_compiler).resolve(), llvm_readobj, test_root
        )
        installed_environment = windows_test_environment(prefix / "bin")
        build_environment = installed_environment.copy()
        build_environment["PATH"] = (
            str(test_compiler.parent)
            + os.pathsep
            + build_environment.get("PATH", "")
        )
    extractor_bytes = extractor.read_bytes()
    if any(
        spelling.encode() in extractor_bytes
        for spelling in (str(source_public_header), source_public_header.as_posix())
    ):
        raise RuntimeError("installed extractor embeds the source checkout's public-header path")

    planned = run(
        [
            str(planner),
            "--m",
            "2",
            "--k",
            "3",
            "--n",
            "2",
            "--variant",
            "reference",
        ],
        capture=True,
        environment=installed_environment,
    )
    if (
        "status=selected" not in planned.stdout
        or "selected=cpu.reference.f32.v1" not in planned.stdout
        or "candidates=[" not in planned.stdout
    ):
        raise RuntimeError(
            "relocated plan inspector lost selected-plan diagnostics:\n"
            f"{planned.stdout}"
        )

    benchmarked = run(
        [
            str(benchmark),
            "--m",
            "2",
            "--n",
            "3",
            "--k",
            "2",
            "--variant",
            "cpu.reference.f32.v1",
            "--warmup",
            "0",
            "--iterations",
            "2",
            "--timer-floor-us",
            "1",
            "--guard",
        ],
        capture=True,
        environment=installed_environment,
    )
    if (
        "variant=cpu.reference.f32.v1" not in benchmarked.stdout
        or "correctness=pass" not in benchmarked.stdout
        or "timing=valid" not in benchmarked.stdout
    ):
        raise RuntimeError(
            "relocated benchmark tool failed its bounded runtime proof:\n"
            f"{benchmarked.stdout}"
        )

    untrusted_source = test_root / "untrusted-source-header.mdsl"
    untrusted_ir = test_root / "untrusted-source-header.json"
    untrusted_source.write_text(
        f'#include "{source_public_header.as_posix()}"\n\n'
        "namespace md = matcore::mdsl;\n"
        "void capture(md::matrix_view &C, md::matrix_view &A, md::matrix_view &B) {\n"
        "  md::gemm(md::out(C), A, B);\n"
        "}\n",
        encoding="utf-8",
    )
    extraction_compile_options = (
        [str(test_compiler), "/TP", "/std:c++20"]
        if is_windows
        else [str(test_compiler), "-x", "c++", "-std=c++20"]
    )
    untrusted = subprocess.run(
        [
            str(extractor),
            "--frontend=native",
            "--input",
            str(untrusted_source),
            "--ir-out",
            str(untrusted_ir),
            "--",
            *extraction_compile_options,
            str(untrusted_source),
        ],
        check=False,
        text=True,
        capture_output=True,
        env=installed_environment,
    )
    if untrusted.returncode == 0 or "trusted <matcore/mdsl.h> header" not in untrusted.stderr:
        raise RuntimeError(
            "installed extractor trusted a checkout-owned public header:\n"
            f"{untrusted.stderr}"
        )
    if untrusted_ir.exists():
        raise RuntimeError("untrusted checkout header produced Matcore IR")

    override_output = test_root / f"override{artifact_suffix}"
    override_compile_options = (
        ["/c", str(source / "consumer.mdsl"), "-o", str(override_output)]
        if is_windows
        else ["-c", str(source / "consumer.mdsl"), "-o", str(override_output)]
    )
    production_override = subprocess.run(
        [
            str(driver),
            f"--tool-prefix-for-testing={prefix}",
            "--matcore-target=cpu",
            *override_compile_options,
        ],
        check=False,
        text=True,
        capture_output=True,
        env=installed_environment,
    )
    if (
        production_override.returncode == 0
        or "unavailable in this production driver build"
        not in production_override.stderr
    ):
        raise RuntimeError(
            "installed production driver exposed the trusted-prefix test override:\n"
            f"{production_override.stderr}"
        )

    configure_command = [
        args.cmake,
        "-S",
        str(source),
        "-B",
        str(build),
        "-G",
        "Ninja",
        f"-DCMAKE_PREFIX_PATH={prefix}",
        f"-DCMAKE_CXX_COMPILER={test_compiler}",
    ]
    if is_windows:
        # This test validates the redistributable Release package and rejects
        # every Debug CRT import.  Ninja has no portable implicit default
        # configuration, so make the consumer's runtime model explicit.
        configure_command.append("-DCMAKE_BUILD_TYPE=Release")
    run(configure_command, environment=build_environment)
    run(
        [args.cmake, "--build", str(build), "--parallel", "2"],
        environment=build_environment,
    )
    executable = build / f"matcore_consumer{executable_suffix}"
    if is_windows:
        assert windows_import_report is not None
        executable_import_report = validate_recursive_windows_imports(
            llvm_readobj, prefix / "bin", [executable]
        )
        windows_import_report["consumer"] = executable_import_report
        report_path = (
            Path(args.windows_import_report_out).resolve()
            if args.windows_import_report_out
            else test_root / "windows-consumer-import-report.json"
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(windows_import_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    initial_run = run(
        [str(executable)], capture=True, environment=installed_environment
    )
    if "consumer-header=1" not in initial_run.stdout:
        raise RuntimeError(f"initial consumer header value is wrong:\n{initial_run.stdout}")

    artifacts = list(
        (build / "CMakeFiles" / "matcore_consumer.mdsl").glob(
            f"*{artifact_suffix}"
        )
    )
    if len(artifacts) != 1:
        expected_kind = "static archive" if is_windows else "relocatable object"
        raise RuntimeError(
            f"expected one generated MDSLC {expected_kind}, found {artifacts}"
        )
    depfiles = list(
        (build / "CMakeFiles" / "matcore_consumer.mdsl").glob(
            f"*{artifact_suffix}.d"
        )
    )
    if len(depfiles) != 1:
        raise RuntimeError(f"expected one MDSLC depfile, found {depfiles}")
    depfile_text = depfiles[0].read_text(encoding="utf-8")
    for expected_dependency in (
        source / "consumer.mdsl",
        source / "consumer_value.h",
        prefix / "include" / "matcore" / "runtime_c.h",
    ):
        if not any(
            spelling in depfile_text
            for spelling in make_depfile_paths(expected_dependency)
        ):
            raise RuntimeError(
                f"MDSLC depfile does not track {expected_dependency}:\n{depfile_text}"
            )
    for forbidden_dependency in (
        ".host.cpp",
        ".sites.h",
        ".stubs.cpp",
        ".backend.cpp",
    ):
        if forbidden_dependency in depfile_text:
            raise RuntimeError(
                f"temporary generated dependency leaked into depfile: {depfile_text}"
            )
    if re.search(r"[\\/]mdslc-[A-Za-z0-9]{6}[\\/]", depfile_text):
        raise RuntimeError(
            f"temporary MDSLC workspace leaked into depfile: {depfile_text}"
        )

    initial_noop = run(
        [args.cmake, "--build", str(build), "--parallel", "2"],
        capture=True,
        environment=build_environment,
    )
    if "no work to do" not in initial_noop.stdout.lower():
        raise RuntimeError(
            f"depfile caused work immediately after a clean build:\n{initial_noop.stdout}"
        )
    artifact_before = artifacts[0].stat().st_mtime_ns

    mdsl_source = source / "consumer.mdsl"
    mdsl_source.write_text(
        mdsl_source.read_text(encoding="utf-8") + "\n// rebuild probe\n",
        encoding="utf-8",
    )
    rebuild = run(
        [args.cmake, "--build", str(build), "--parallel", "2"],
        capture=True,
        environment=build_environment,
    )
    if "Compiling MDSLC source consumer.mdsl" not in rebuild.stdout:
        raise RuntimeError(f"MDSLC source was not regenerated:\n{rebuild.stdout}")
    rebuild_steps = [
        line for line in rebuild.stdout.splitlines() if line.startswith("[")
    ]
    if len(rebuild_steps) != 2 or "Linking CXX executable" not in rebuild_steps[1]:
        raise RuntimeError(
            "editing one .mdsl should only regenerate its artifact and relink; "
            f"observed:\n{rebuild.stdout}"
        )
    if artifacts[0].stat().st_mtime_ns <= artifact_before:
        raise RuntimeError("generated MDSLC artifact timestamp did not advance")
    source_rebuilt_run = run(
        [str(executable)], capture=True, environment=installed_environment
    )
    if "consumer-header=1" not in source_rebuilt_run.stdout:
        raise RuntimeError(
            f"source rebuild changed the header value unexpectedly:\n{source_rebuilt_run.stdout}"
        )

    header = source / "consumer_value.h"
    artifact_before_header = artifacts[0].stat().st_mtime_ns
    header.write_text(
        "#pragma once\n\ninline constexpr int consumer_header_value = 2;\n",
        encoding="utf-8",
    )
    header_rebuild = run(
        [args.cmake, "--build", str(build), "--parallel", "2"],
        capture=True,
        environment=build_environment,
    )
    if "Compiling MDSLC source consumer.mdsl" not in header_rebuild.stdout:
        raise RuntimeError(
            "editing an included header did not regenerate its MDSLC artifact:\n"
            f"{header_rebuild.stdout}"
        )
    header_rebuild_steps = [
        line for line in header_rebuild.stdout.splitlines() if line.startswith("[")
    ]
    if (
        len(header_rebuild_steps) != 2
        or "Linking CXX executable" not in header_rebuild_steps[1]
    ):
        raise RuntimeError(
            "editing one included header should only regenerate its artifact and relink; "
            f"observed:\n{header_rebuild.stdout}"
        )
    if artifacts[0].stat().st_mtime_ns <= artifact_before_header:
        raise RuntimeError("included-header rebuild did not refresh the MDSLC artifact")
    header_rebuilt_run = run(
        [str(executable)], capture=True, environment=installed_environment
    )
    if "consumer-header=2" not in header_rebuilt_run.stdout:
        raise RuntimeError(
            "included-header rebuild executed stale code:\n"
            f"{header_rebuilt_run.stdout}"
        )

    installed_runtime_header = prefix / "include" / "matcore" / "runtime_c.h"
    artifact_before_runtime_header = artifacts[0].stat().st_mtime_ns
    installed_runtime_header.write_text(
        installed_runtime_header.read_text(encoding="utf-8")
        + "\n/* installed depfile rebuild probe */\n",
        encoding="utf-8",
    )
    runtime_header_rebuild = run(
        [args.cmake, "--build", str(build), "--parallel", "2"],
        capture=True,
        environment=build_environment,
    )
    if "Compiling MDSLC source consumer.mdsl" not in runtime_header_rebuild.stdout:
        raise RuntimeError(
            "editing installed runtime_c.h did not regenerate the MDSLC artifact:\n"
            f"{runtime_header_rebuild.stdout}"
        )
    runtime_header_steps = [
        line
        for line in runtime_header_rebuild.stdout.splitlines()
        if line.startswith("[")
    ]
    if (
        len(runtime_header_steps) != 2
        or "Linking CXX executable" not in runtime_header_steps[1]
    ):
        raise RuntimeError(
            "editing installed runtime_c.h should only regenerate the MDSLC "
            f"artifact and relink; observed:\n{runtime_header_rebuild.stdout}"
        )
    if artifacts[0].stat().st_mtime_ns <= artifact_before_runtime_header:
        raise RuntimeError("runtime_c.h rebuild did not refresh the MDSLC artifact")
    runtime_header_run = run(
        [str(executable)], capture=True, environment=installed_environment
    )
    if "consumer-header=2" not in runtime_header_run.stdout:
        raise RuntimeError(
            "runtime_c.h rebuild executed stale or incorrect code:\n"
            f"{runtime_header_run.stdout}"
        )

    noop = run(
        [args.cmake, "--build", str(build), "--parallel", "2"],
        capture=True,
        environment=build_environment,
    )
    if "no work to do" not in noop.stdout.lower():
        raise RuntimeError(f"second rebuild was not a no-op:\n{noop.stdout}")

    forbidden = {
        str(Path(args.source_dir).resolve().parents[2]),
        str(Path(args.producer_build_dir).resolve()),
        str(staging_prefix),
    }
    package_dir = prefix / "lib" / "cmake" / "MatcoreDSL"
    for package_file in package_dir.glob("*.cmake"):
        contents = package_file.read_text(encoding="utf-8")
        for path in forbidden:
            if path in contents:
                raise RuntimeError(
                    f"absolute local path leaked into {package_file}: {path}"
                )

    print(
        "installed consumer: relocated configure/build/run/source+header "
        "runtime-header rebuild/no-op PASS"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
