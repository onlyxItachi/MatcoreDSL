#!/usr/bin/env python3
"""Black-box tests for the opt-in Matcore MLIR inspection tool."""

from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import tempfile


def run(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(arguments, check=False, capture_output=True, text=True)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def contains_mdsl_gemm_operation(module_text: str) -> bool:
    return re.search(
        r'^\s*(?:%[-\w.$]+\s*=\s*)?(?:mdsl\.gemm\b|"mdsl\.gemm"\s*\()',
        module_text,
        re.MULTILINE,
    ) is not None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool", required=True, type=pathlib.Path)
    options = parser.parse_args()

    source_dir = pathlib.Path(__file__).resolve().parents[1]
    v1_input = source_dir / "ir" / "gemm_capture.v1.golden.json"
    v0_input = source_dir / "frontend" / "gemm_capture.golden.json"
    malformed_input = source_dir / "fixtures" / "negative" / "malformed_ir.json"
    golden = (source_dir / "mlir" / "gemm_capture.semantic.golden.mlir").read_text()
    structured_golden = (
        source_dir / "mlir" / "gemm_capture.structured.golden.mlir"
    ).read_text()
    profile = "explicit-gemm-f32-v1"

    first = run(
        [
            str(options.tool),
            "--input",
            str(v1_input),
            "--numerical-profile",
            profile,
            "--execution-intent",
            "generic",
        ]
    )
    require(first.returncode == 0, f"valid bridge failed: {first.stderr}")
    require(first.stdout == golden, "stdout output differs from reviewed golden")
    require(not first.stderr, "successful stdout bridge emitted diagnostics")
    require(contains_mdsl_gemm_operation(first.stdout),
            "default stage no longer emits the semantic mdsl.gemm operation")
    require("linalg.matmul" not in first.stdout,
            "default stage unexpectedly emits the structured handoff")

    second = run(
        [
            str(options.tool),
            "--input",
            str(v1_input),
            "--numerical-profile",
            profile,
            "--execution-intent",
            "generic",
        ]
    )
    require(second.returncode == 0 and second.stdout == first.stdout,
            "two CLI bridges must be byte deterministic")

    structured_arguments = [
        str(options.tool),
        "--input",
        str(v1_input),
        "--numerical-profile",
        profile,
        "--execution-intent",
        "generic",
        "--emit-stage",
        "structured-gemm-v1",
    ]
    structured = run(structured_arguments)
    require(structured.returncode == 0,
            f"structured GEMM handoff failed: {structured.stderr}")
    require(not structured.stderr,
            "successful structured handoff emitted diagnostics")

    structured_text = structured.stdout
    require(structured_text == structured_golden,
            "structured output differs from reviewed golden")
    required_module_contract = (
        'mdsl.analysis_only = true',
        'mdsl.execution_authority = "inspection_only"',
        'mdsl.producer = "matcore-structured-gemm-handoff-v1"',
        'mdsl.source_producer = "clang-libtooling-v1"',
        'mdsl.source_bridge_schema = "matcore-mlir-semantic-v1"',
        'mdsl.structured_handoff_schema = "matcore-structured-gemm-handoff-v1"',
        'mdsl.structured_handoff_version = 1 : i32',
    )
    for signal in required_module_contract:
        require(signal in structured_text,
                f"structured handoff omitted module contract signal {signal!r}")

    required_semantic_contract = (
        "accumulation_type = f32",
        "aliasing =",
        "effects =",
        "lhs_semantics =",
        "numerical =",
        "origin =",
        "output_semantics =",
        "policy =",
        "provenance =",
        "rhs_semantics =",
        "semantic_requirements =",
        "site_id =",
        'synchronization = "synchronous"',
    )
    contract_start = structured_text.find("mdsl.semantic_contract = {")
    require(contract_start >= 0,
            "structured handoff omitted its self-contained semantic contract")
    contract_end = structured_text.find("}, mdsl.site_id =", contract_start)
    require(contract_end > contract_start,
            "structured handoff semantic contract is not independently delimited")
    semantic_contract_text = structured_text[contract_start:contract_end]
    for signal in required_semantic_contract:
        require(signal in semantic_contract_text,
                f"structured handoff dropped semantic contract field {signal!r}")

    retained_facts = (
        'relation = "no_alias"',
        'contract = "required_precondition"',
        'reads = ["lhs", "rhs"]',
        'writes = ["output"]',
        'layout = "row_major_contiguous"',
        'symbol = "m"',
        'symbol = "k"',
        'symbol = "n"',
        'alignment_bytes = 4 : i64',
        'reassociation = "within_k_reduction"',
        'reduction_order = "implementation_defined_within_k"',
        "approximate_math = false",
        "inplace = false",
        'fallback = "error"',
        'target = "cpu"',
    )
    for fact in retained_facts:
        require(fact in semantic_contract_text,
                f"structured handoff dropped retained fact {fact!r}")
    require('destination = "original_output_full_zero_fill"' in structured_text,
            "structured handoff dropped destination overwrite semantics")

    fill_position = structured_text.find("linalg.fill")
    matmul_position = structured_text.find("linalg.matmul")
    return_position = structured_text.find("return %", matmul_position)
    require(fill_position >= 0, "structured handoff omitted linalg.fill")
    require(matmul_position > fill_position,
            "structured handoff must place linalg.fill before linalg.matmul")
    require(return_position > matmul_position,
            "structured handoff must return the linalg.matmul result")
    require(structured_text.count('source_operation = "mdsl.gemm"') == 1,
            "structured handoff source-operation marker is not unique")
    require(not contains_mdsl_gemm_operation(structured_text),
            "structured handoff still contains an mdsl.gemm operation")

    with tempfile.TemporaryDirectory(prefix="matcore mlir ") as temporary:
        output = pathlib.Path(temporary) / "semantic output.mlir"
        written = run(
            [
                str(options.tool),
                "--input",
                str(v1_input),
                "--numerical-profile",
                profile,
                "--execution-intent",
                "generic",
                "--output",
                str(output),
            ]
        )
        require(written.returncode == 0, f"file output failed: {written.stderr}")
        require(not written.stdout, "file output unexpectedly wrote to stdout")
        require(output.read_text() == golden, "file output differs from golden")

        structured_output = pathlib.Path(temporary) / "structured output.mlir"
        structured_written = run(
            structured_arguments + ["--output", str(structured_output)]
        )
        require(structured_written.returncode == 0,
                f"structured file output failed: {structured_written.stderr}")
        require(not structured_written.stdout,
                "structured file output unexpectedly wrote to stdout")
        require(not structured_written.stderr,
                "successful structured file output emitted diagnostics")
        require(structured_output.read_text() == structured_text,
                "structured stdout and file output are not byte deterministic")

    negative_cases = [
        (
            [str(options.tool), "--input", str(v1_input),
             "--execution-intent", "generic"],
            "--numerical-profile is required",
        ),
        (
            [
                str(options.tool),
                "--input",
                str(v1_input),
                "--numerical-profile",
                profile,
                "--execution-intent",
                "generic",
                "--emit-stage",
                "not-a-stage",
            ],
            "unsupported inspection stage: not-a-stage",
        ),
        (
            [str(options.tool), "--input", str(v1_input),
             "--numerical-profile", profile],
            "--execution-intent is required",
        ),
        (
            [
                str(options.tool),
                "--input",
                str(v1_input),
                "--numerical-profile",
                "implicit-default",
                "--execution-intent",
                "generic",
            ],
            "unsupported numerical profile",
        ),
        (
            [
                str(options.tool),
                "--input",
                str(v1_input),
                "--numerical-profile",
                profile,
                "--execution-intent",
                "training",
            ],
            "unsupported execution intent",
        ),
        (
            [
                str(options.tool),
                "--input",
                str(v0_input),
                "--numerical-profile",
                profile,
                "--execution-intent",
                "generic",
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
                "--execution-intent",
                "generic",
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
                "--execution-intent",
                "generic",
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

    print(f"Matcore MLIR CLI: {5 + len(negative_cases)} cases, 0 failures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
