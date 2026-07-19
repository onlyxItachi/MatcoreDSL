#!/usr/bin/env python3

import argparse
from pathlib import Path
import shutil
import subprocess
import sys


def run(command: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )
    if result.returncode != 0:
        if result.stdout:
            print(result.stdout, file=sys.stderr)
        raise RuntimeError(f"command failed ({result.returncode}): {command}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmake", required=True)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--producer-build-dir", required=True)
    parser.add_argument("--test-root", required=True)
    parser.add_argument("--cxx-compiler", required=True)
    args = parser.parse_args()

    test_root = Path(args.test_root).resolve()
    if test_root.exists():
        shutil.rmtree(test_root)
    source = test_root / "source"
    build = test_root / "build"
    staging_prefix = test_root / "install-staging"
    # A space-bearing prefix verifies argv-safe package relocation. Commas in
    # an ELF runtime directory are covered at the driver layer, which forwards
    # rpath using separate -Xlinker arguments; CMake's compiler-driver rpath
    # encoding uses comma-separated -Wl syntax and cannot represent that path.
    prefix = test_root / "relocated" / "matcoredsl prefix"
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
        prefix / "bin" / "mdslc++",
        prefix / "bin" / "matcore-extract",
        prefix / "include" / "matcore" / "mdsl.h",
        prefix / "include" / "matcore" / "runtime_c.h",
        prefix / "lib" / "libmatcore_runtime.so",
        prefix / "lib" / "cmake" / "MatcoreDSL" / "MatcoreDSLConfig.cmake",
        prefix / "lib" / "cmake" / "MatcoreDSL" / "MatcoreDSLConfigVersion.cmake",
        prefix / "lib" / "cmake" / "MatcoreDSL" / "MatcoreDSLTargets.cmake",
        prefix / "lib" / "cmake" / "MatcoreDSL" / "MatcoreDSLCompile.cmake",
    ]
    missing = [path for path in expected_install_files if not path.exists()]
    if missing:
        raise RuntimeError(f"installed package is incomplete: {missing}")

    repository = Path(__file__).resolve().parents[3]
    source_public_header = repository / "compiler" / "include" / "matcore" / "mdsl.h"
    driver = prefix / "bin" / "mdslc++"
    extractor = prefix / "bin" / "matcore-extract"
    if str(source_public_header).encode() in extractor.read_bytes():
        raise RuntimeError("installed extractor embeds the source checkout's public-header path")

    untrusted_source = test_root / "untrusted-source-header.mdsl"
    untrusted_ir = test_root / "untrusted-source-header.json"
    untrusted_source.write_text(
        f'#include "{source_public_header}"\n\n'
        "namespace md = matcore::mdsl;\n"
        "void capture(md::matrix_view &C, md::matrix_view &A, md::matrix_view &B) {\n"
        "  md::gemm(md::out(C), A, B);\n"
        "}\n",
        encoding="utf-8",
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
            "clang++",
            "-std=c++20",
            str(untrusted_source),
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    if untrusted.returncode == 0 or "trusted <matcore/mdsl.h> header" not in untrusted.stderr:
        raise RuntimeError(
            "installed extractor trusted a checkout-owned public header:\n"
            f"{untrusted.stderr}"
        )
    if untrusted_ir.exists():
        raise RuntimeError("untrusted checkout header produced Matcore IR")

    production_override = subprocess.run(
        [
            str(driver),
            f"--tool-prefix-for-testing={prefix}",
            "--matcore-target=cpu",
            "-c",
            str(source / "consumer.mdsl"),
            "-o",
            str(test_root / "override.o"),
        ],
        check=False,
        text=True,
        capture_output=True,
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

    run([
        args.cmake,
        "-S",
        str(source),
        "-B",
        str(build),
        "-G",
        "Ninja",
        f"-DCMAKE_PREFIX_PATH={prefix}",
        f"-DCMAKE_CXX_COMPILER={args.cxx_compiler}",
    ])
    run([args.cmake, "--build", str(build), "--", "-j2"])
    executable = build / "matcore_consumer"
    initial_run = run([str(executable)], capture=True)
    if "consumer-header=1" not in initial_run.stdout:
        raise RuntimeError(f"initial consumer header value is wrong:\n{initial_run.stdout}")

    objects = list((build / "CMakeFiles" / "matcore_consumer.mdsl").glob("*.o"))
    if len(objects) != 1:
        raise RuntimeError(f"expected one generated MDSLC object, found {objects}")
    depfiles = list((build / "CMakeFiles" / "matcore_consumer.mdsl").glob("*.o.d"))
    if len(depfiles) != 1:
        raise RuntimeError(f"expected one MDSLC depfile, found {depfiles}")
    depfile_text = depfiles[0].read_text(encoding="utf-8")
    for expected_dependency in (source / "consumer.mdsl", source / "consumer_value.h"):
        if str(expected_dependency) not in depfile_text:
            raise RuntimeError(
                f"MDSLC depfile does not track {expected_dependency}:\n{depfile_text}"
            )
    for forbidden_dependency in (
        ".host.cpp",
        ".sites.h",
        ".stubs.cpp",
        ".backend.cpp",
        "/tmp/mdslc-",
    ):
        if forbidden_dependency in depfile_text:
            raise RuntimeError(
                f"temporary generated dependency leaked into depfile: {depfile_text}"
            )

    initial_noop = run(
        [args.cmake, "--build", str(build), "--", "-j2"], capture=True
    )
    if "no work to do" not in initial_noop.stdout.lower():
        raise RuntimeError(
            f"depfile caused work immediately after a clean build:\n{initial_noop.stdout}"
        )
    object_before = objects[0].stat().st_mtime_ns

    mdsl_source = source / "consumer.mdsl"
    mdsl_source.write_text(
        mdsl_source.read_text(encoding="utf-8") + "\n// rebuild probe\n",
        encoding="utf-8",
    )
    rebuild = run(
        [args.cmake, "--build", str(build), "--", "-j2"], capture=True
    )
    if "Compiling MDSLC source consumer.mdsl" not in rebuild.stdout:
        raise RuntimeError(f"MDSLC source was not regenerated:\n{rebuild.stdout}")
    rebuild_steps = [
        line for line in rebuild.stdout.splitlines() if line.startswith("[")
    ]
    if len(rebuild_steps) != 2 or "Linking CXX executable" not in rebuild_steps[1]:
        raise RuntimeError(
            "editing one .mdsl should only regenerate its object and relink; "
            f"observed:\n{rebuild.stdout}"
        )
    if objects[0].stat().st_mtime_ns <= object_before:
        raise RuntimeError("generated MDSLC object timestamp did not advance")
    source_rebuilt_run = run([str(executable)], capture=True)
    if "consumer-header=1" not in source_rebuilt_run.stdout:
        raise RuntimeError(
            f"source rebuild changed the header value unexpectedly:\n{source_rebuilt_run.stdout}"
        )

    header = source / "consumer_value.h"
    object_before_header = objects[0].stat().st_mtime_ns
    header.write_text(
        "#pragma once\n\ninline constexpr int consumer_header_value = 2;\n",
        encoding="utf-8",
    )
    header_rebuild = run(
        [args.cmake, "--build", str(build), "--", "-j2"], capture=True
    )
    if "Compiling MDSLC source consumer.mdsl" not in header_rebuild.stdout:
        raise RuntimeError(
            "editing an included header did not regenerate its MDSLC object:\n"
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
            "editing one included header should only regenerate its object and relink; "
            f"observed:\n{header_rebuild.stdout}"
        )
    if objects[0].stat().st_mtime_ns <= object_before_header:
        raise RuntimeError("included-header rebuild did not refresh the MDSLC object")
    header_rebuilt_run = run([str(executable)], capture=True)
    if "consumer-header=2" not in header_rebuilt_run.stdout:
        raise RuntimeError(
            "included-header rebuild executed stale code:\n"
            f"{header_rebuilt_run.stdout}"
        )

    noop = run([args.cmake, "--build", str(build), "--", "-j2"], capture=True)
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
        "rebuild/no-op PASS"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
