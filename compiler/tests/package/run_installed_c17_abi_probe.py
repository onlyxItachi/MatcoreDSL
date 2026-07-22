#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


EXPORT_PATTERN = re.compile(
    r"MATCORE_RUNTIME_API\s+matcore_status_v0\s+"
    r"(matcore_runtime_[A-Za-z0-9_]+)\s*\(",
    re.MULTILINE,
)
CALL_PATTERN = re.compile(r"\b(matcore_runtime_[A-Za-z0-9_]+)\s*\(")


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


def exported_symbols(header: Path) -> set[str]:
    symbols = set(EXPORT_PATTERN.findall(header.read_text(encoding="utf-8")))
    if not symbols:
        raise RuntimeError(f"no MATCORE_RUNTIME_API exports found in {header}")
    return symbols


def called_symbols(source: Path) -> set[str]:
    return set(CALL_PATTERN.findall(source.read_text(encoding="utf-8")))


def dynamic_undefined_symbols(executable: Path, nm: str) -> set[str]:
    output = run(
        [nm, "-D", "--undefined-only", "--format=posix", str(executable)],
        capture=True,
    ).stdout
    symbols: set[str] = set()
    for line in output.splitlines():
        fields = line.split()
        if fields:
            symbols.add(fields[0].split("@", 1)[0])
    return symbols


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmake", required=True)
    parser.add_argument("--producer-build-dir", required=True)
    parser.add_argument("--test-root", required=True)
    parser.add_argument("--c-compiler", required=True)
    parser.add_argument("--nm", default="nm")
    args = parser.parse_args()

    source = Path(__file__).resolve().with_name("installed_c17_abi_probe.c")
    producer_build = Path(args.producer_build_dir).resolve()
    test_root = Path(args.test_root).resolve()
    if test_root.exists():
        shutil.rmtree(test_root)
    prefix = test_root / "relocated strict C prefix"
    prefix.mkdir(parents=True)

    run(
        [
            args.cmake,
            "--install",
            str(producer_build),
            "--prefix",
            str(prefix),
        ]
    )

    header = prefix / "include" / "matcore" / "runtime_c.h"
    runtime = prefix / "lib" / "libmatcore_runtime.so"
    if not header.is_file() or not runtime.is_file():
        raise RuntimeError(
            "installed runtime package is incomplete: "
            f"header={header.is_file()} runtime={runtime.is_file()}"
        )

    exports = exported_symbols(header)
    calls = called_symbols(source)
    if calls != exports:
        raise RuntimeError(
            "strict C probe and installed header export sets differ: "
            f"missing calls={sorted(exports - calls)}, "
            f"unknown calls={sorted(calls - exports)}"
        )

    executable = test_root / "installed-c17-abi-probe"
    run(
        [
            str(Path(args.c_compiler).resolve()),
            "-std=c17",
            "-pedantic-errors",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(prefix / "include"),
            str(source),
            "-L",
            str(prefix / "lib"),
            "-lmatcore_runtime",
            "-Xlinker",
            "-rpath",
            "-Xlinker",
            str(prefix / "lib"),
            "-o",
            str(executable),
        ]
    )

    dynamic_symbols = dynamic_undefined_symbols(executable, args.nm)
    missing_dynamic_references = exports - dynamic_symbols
    if missing_dynamic_references:
        raise RuntimeError(
            "linked strict C probe does not retain references to every public "
            f"runtime export: {sorted(missing_dynamic_references)}"
        )

    environment = os.environ.copy()
    prior_library_path = environment.get("LD_LIBRARY_PATH")
    environment["LD_LIBRARY_PATH"] = str(prefix / "lib")
    if prior_library_path:
        environment["LD_LIBRARY_PATH"] += os.pathsep + prior_library_path
    execution = run([str(executable)], capture=True, environment=environment)
    if execution.stdout.strip() != "installed strict C17 ABI probe: PASS":
        raise RuntimeError(
            "installed strict C ABI probe returned unexpected output:\n"
            f"{execution.stdout}"
        )

    print(
        "installed C17 ABI: compile/link/run PASS; "
        f"authenticated {len(exports)} public exports"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
