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
    prefix = test_root / "install"
    shutil.copytree(Path(args.source_dir).resolve(), source)

    run([
        args.cmake,
        "--install",
        str(Path(args.producer_build_dir).resolve()),
        "--prefix",
        str(prefix),
    ])
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
    run([str(executable)])

    objects = list((build / "CMakeFiles" / "matcore_consumer.mdsl").glob("*.o"))
    if len(objects) != 1:
        raise RuntimeError(f"expected one generated MDSLC object, found {objects}")
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
    run([str(executable)])

    noop = run([args.cmake, "--build", str(build), "--", "-j2"], capture=True)
    if "no work to do" not in noop.stdout.lower():
        raise RuntimeError(f"second rebuild was not a no-op:\n{noop.stdout}")

    forbidden = {
        str(Path(args.source_dir).resolve().parents[2]),
        str(Path(args.producer_build_dir).resolve()),
    }
    package_dir = prefix / "lib" / "cmake" / "MatcoreDSL"
    for package_file in package_dir.glob("*.cmake"):
        contents = package_file.read_text(encoding="utf-8")
        for path in forbidden:
            if path in contents:
                raise RuntimeError(
                    f"absolute local path leaked into {package_file}: {path}"
                )

    print("installed consumer: configure/build/run/rebuild/no-op PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
