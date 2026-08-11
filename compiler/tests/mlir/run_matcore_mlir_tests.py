#!/usr/bin/env python3
"""Black-box tests for the opt-in Matcore MLIR inspection tool."""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import tempfile


def run(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(arguments, check=False, capture_output=True, text=True)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool", required=True, type=pathlib.Path)
    options = parser.parse_args()

    source_dir = pathlib.Path(__file__).resolve().parents[1]
    v1_input = source_dir / "ir" / "gemm_capture.v1.golden.json"
    v0_input = source_dir / "frontend" / "gemm_capture.golden.json"
    malformed_input = source_dir / "fixtures" / "negative" / "malformed_ir.json"
    golden = (source_dir / "mlir" / "gemm_capture.semantic.golden.mlir").read_text()
    profile = "explicit-gemm-f32-v1"

    first = run(
        [
            str(options.tool),
            "--input",
            str(v1_input),
            "--numerical-profile",
            profile,
        ]
    )
    require(first.returncode == 0, f"valid bridge failed: {first.stderr}")
    require(first.stdout == golden, "stdout output differs from reviewed golden")
    require(not first.stderr, "successful stdout bridge emitted diagnostics")

    second = run(
        [
            str(options.tool),
            "--input",
            str(v1_input),
            "--numerical-profile",
            profile,
        ]
    )
    require(second.returncode == 0 and second.stdout == first.stdout,
            "two CLI bridges must be byte deterministic")

    with tempfile.TemporaryDirectory(prefix="matcore mlir ") as temporary:
        output = pathlib.Path(temporary) / "semantic output.mlir"
        written = run(
            [
                str(options.tool),
                "--input",
                str(v1_input),
                "--numerical-profile",
                profile,
                "--output",
                str(output),
            ]
        )
        require(written.returncode == 0, f"file output failed: {written.stderr}")
        require(not written.stdout, "file output unexpectedly wrote to stdout")
        require(output.read_text() == golden, "file output differs from golden")

    negative_cases = [
        (
            [str(options.tool), "--input", str(v1_input)],
            "--numerical-profile is required",
        ),
        (
            [
                str(options.tool),
                "--input",
                str(v1_input),
                "--numerical-profile",
                "implicit-default",
            ],
            "unsupported numerical profile",
        ),
        (
            [
                str(options.tool),
                "--input",
                str(v0_input),
                "--numerical-profile",
                profile,
            ],
            "unsupported Matcore IR version; expected version 1",
        ),
        (
            [
                str(options.tool),
                "--input",
                str(malformed_input),
                "--numerical-profile",
                profile,
            ],
            "verified Matcore IR v1 input required",
        ),
        (
            [
                str(options.tool),
                "--input",
                str(v1_input),
                "--numerical-profile",
                profile,
                "--output",
                str(v1_input),
            ],
            "input and output paths must differ",
        ),
    ]
    for arguments, expected in negative_cases:
        completed = run(arguments)
        require(completed.returncode != 0,
                f"negative case unexpectedly succeeded: {arguments}")
        require(expected in completed.stderr,
                f"negative diagnostic omitted {expected!r}: {completed.stderr}")
        require(not completed.stdout,
                f"negative case emitted semantic output: {arguments}")

    print(f"Matcore MLIR CLI: {2 + len(negative_cases)} checks, 0 failures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
