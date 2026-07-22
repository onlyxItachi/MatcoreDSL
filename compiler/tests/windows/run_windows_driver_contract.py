#!/usr/bin/env python3
"""Focused clang-cl parsing and Windows driver fail-closed contract checks."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path


def run(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command, cwd=cwd, text=True, capture_output=True, check=False
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extractor", type=Path, required=True)
    parser.add_argument("--driver", type=Path)
    args = parser.parse_args()

    repository = Path(__file__).resolve().parents[3]
    source = repository / "compiler/tests/frontend/gemm_capture.mdsl"
    with tempfile.TemporaryDirectory(prefix="matcore-clang-cl-contract-") as encoded:
        temporary = Path(encoded)
        output = temporary / "clang-cl.json"
        tool_arguments = ["--driver-mode=cl"]
        if os.name != "nt":
            # Exercise clang-cl argument parsing with the local Linux headers.
            tool_arguments.append("--target=x86_64-unknown-linux-gnu")
        tool_arguments.extend(
            [
                "/TP",
                "/std:c++20",
                "/EHsc",
                "/I",
                "compiler/include",
                "/I",
                "compiler/tests/frontend",
                source.as_posix(),
            ]
        )
        parsed = run(
            [
                str(args.extractor),
                "--input",
                source.as_posix(),
                "--ir-out",
                str(output),
                "--",
                *tool_arguments,
            ],
            repository,
        )
        require(parsed.returncode == 0, f"clang-cl parse failed:\n{parsed.stderr}")
        document = json.loads(output.read_text(encoding="utf-8"))
        require(
            document.get("producer") == "clang-libtooling-v1"
            and len(document.get("operations", [])) == 1,
            "clang-cl parse did not produce the authenticated GEMM operation",
        )

        rejected = run(
            [
                str(args.extractor),
                "--input",
                source.as_posix(),
                "--ir-out",
                str(temporary / "unsafe.json"),
                "--",
                *tool_arguments[:-1],
                "/Founsafe.obj",
                source.as_posix(),
            ],
            repository,
        )
        require(
            rejected.returncode != 0 and "output-producing clang-cl" in rejected.stderr,
            "native frontend accepted clang-cl output mutation",
        )

        if os.name == "nt":
            require(args.driver is not None, "Windows driver path is required")
            host_source = temporary / "host only ü.mdsl"
            host_source.write_text("int main() { return 0; }\n", encoding="utf-8")

            guard_cases = [
                (["/TC", str(host_source), f"/Fe:{temporary / 'tc.exe'}"], "valid C++"),
                (["--matcore-target=cuda", str(host_source)], "unsupported Matcore target"),
                (
                    [
                        "--matcore-target=cpu",
                        "/LD",
                        str(host_source),
                        f"/Fe:{temporary / 'dll.exe'}",
                    ],
                    "incompatible",
                ),
                (
                    [
                        "--matcore-target=cpu",
                        "@opaque.rsp",
                        str(host_source),
                        f"/Fe:{temporary / 'opaque.exe'}",
                    ],
                    "incompatible",
                ),
                (
                    [
                        "--matcore-target=cpu",
                        "/c",
                        str(host_source),
                        f"/Fo{temporary / 'mixed.lib'}",
                        "/link",
                        "/DEBUG",
                    ],
                    "/link arguments are invalid with /c",
                ),
                (
                    [
                        "--matcore-target=cpu",
                        "/c",
                        str(host_source),
                        f"/Fo{temporary / 'wrong.obj'}",
                    ],
                    "output must use .lib",
                ),
            ]
            for arguments, diagnostic in guard_cases:
                completed = run([str(args.driver), *arguments], repository)
                require(
                    completed.returncode != 0 and diagnostic in completed.stderr,
                    f"driver guard failed for {arguments}:\n{completed.stderr}",
                )

    print("Windows clang-cl frontend/driver contract PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
