#!/usr/bin/env python3
"""Focused end-to-end proof for the opt-in Matcore MLIR CPU pipeline."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path


class Failure(RuntimeError):
    pass


class Checks:
    def __init__(self) -> None:
        self.count = 0

    def require(self, condition: bool, message: str) -> None:
        self.count += 1
        if not condition:
            raise Failure(message)


def contains_mdsl_gemm_operation(module_text: str) -> bool:
    return re.search(
        r'^\s*(?:%[-\w.$]+\s*=\s*)?(?:mdsl\.gemm\b|"mdsl\.gemm"\s*\()',
        module_text,
        re.MULTILINE,
    ) is not None


def run(argv: list[str], cwd: Path, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            argv,
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        raise Failure(f"command timed out: {argv!r}") from error


def require_ok(checks: Checks, completed: subprocess.CompletedProcess[str], label: str) -> None:
    checks.require(
        completed.returncode == 0,
        f"{label} exited {completed.returncode}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
    )


def extractor_command(
    extractor: Path,
    clangxx: Path,
    repository: Path,
    source: Path,
    outputs: dict[str, Path],
    *,
    emit_structured: bool = True,
) -> list[str]:
    command = [
        str(extractor),
        "--frontend=native",
        "--semantic-pipeline=matcore-mlir",
        "--ir-version=1",
        "--input",
        str(source),
        "--ir-out",
        str(outputs["ir"]),
        "--semantic-ir-out",
        str(outputs["semantic"]),
        "--rewrite-out",
        str(outputs["host"]),
        "--sites-out",
        str(outputs["sites"]),
        "--stubs-out",
        str(outputs["stubs"]),
        "--backend-out",
        str(outputs["backend"]),
    ]
    if emit_structured:
        command.extend(
            ["--structured-ir-out", str(outputs["structured"])]
        )
    command.extend(
        [
            "--",
            str(clangxx),
            "-std=c++20",
            f"-I{repository / 'compiler/include'}",
            str(source),
        ]
    )
    return command


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extractor", type=Path, required=True)
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--clangxx", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument(
        "--expected-default-semantic-pipeline",
        choices=("capture-v0", "matcore-mlir"),
        required=True,
    )
    parser.add_argument("--expect-unavailable", action="store_true")
    arguments = parser.parse_args()

    repository = arguments.repository_root.resolve()
    extractor = arguments.extractor.resolve()
    driver = arguments.driver.resolve()
    # Preserve the compiler driver's argv[0].  On Debian/Ubuntu clang++ is a
    # symlink to the clang binary, and resolving it changes C++ link-driver
    # behavior into C-driver behavior (notably dropping libstdc++).
    clangxx = arguments.clangxx.absolute()
    build_dir = arguments.build_dir.resolve()
    checks = Checks()

    checks.require(extractor.is_file(), f"missing extractor: {extractor}")
    checks.require(driver.is_file(), f"missing driver: {driver}")
    checks.require(clangxx.is_file(), f"missing compiler: {clangxx}")
    source = repository / "compiler/examples/gemm_v0.mdsl"
    two_sites = repository / "compiler/tests/frontend/two_sites.mdsl"
    checks.require(source.is_file(), f"missing execution fixture: {source}")
    checks.require(two_sites.is_file(), f"missing two-site fixture: {two_sites}")

    with tempfile.TemporaryDirectory(prefix="matcore-semantic-cpu-") as encoded:
        temporary = Path(encoded)
        if arguments.expect_unavailable:
            checks.require(
                arguments.expected_default_semantic_pipeline == "capture-v0",
                "MLIR-disabled build advertised a matcore-mlir default",
            )
            unavailable_ir = temporary / "unavailable.json"
            extractor_rejection = run(
                [
                    str(extractor),
                    "--semantic-pipeline=matcore-mlir",
                    "--ir-version=1",
                    "--input",
                    str(source),
                    "--ir-out",
                    str(unavailable_ir),
                ],
                repository,
            )
            checks.require(
                extractor_rejection.returncode != 0,
                "MLIR-disabled extractor accepted semantic pipeline",
            )
            checks.require(
                "Matcore MLIR support was not built" in extractor_rejection.stderr,
                "MLIR-disabled extractor lacked an actionable diagnostic",
            )
            checks.require(
                not unavailable_ir.exists(),
                "MLIR-disabled extractor emitted an artifact",
            )

            unavailable_object = temporary / (
                "unavailable.lib" if os.name == "nt" else "unavailable.o"
            )
            unavailable_compile_arguments = (
                [
                    "/nologo",
                    "/TP",
                    "/std:c++20",
                    "/EHsc",
                    "/MD",
                    "/c",
                    str(source),
                    "-o",
                    str(unavailable_object),
                ]
                if os.name == "nt"
                else [
                    "-std=c++20",
                    "-c",
                    str(source),
                    "-o",
                    str(unavailable_object),
                ]
            )
            driver_rejection = run(
                [
                    str(driver),
                    "--semantic-pipeline=matcore-mlir",
                    "--matcore-target=cpu",
                    *unavailable_compile_arguments,
                ],
                repository,
            )
            checks.require(
                driver_rejection.returncode != 0,
                "MLIR-disabled driver accepted semantic pipeline",
            )
            checks.require(
                "Matcore MLIR support was not built" in driver_rejection.stderr,
                "MLIR-disabled driver lacked the extractor diagnostic",
            )
            checks.require(
                not unavailable_object.exists(),
                "MLIR-disabled driver emitted an object",
            )
            print(
                f"Matcore MLIR unavailable gate: {checks.count} checks, 0 failures"
            )
            return 0

        outputs = {
            "ir": temporary / "direct.matcore.json",
            "semantic": temporary / "direct.semantic.mlir",
            "structured": temporary / "direct.structured.mlir",
            "host": temporary / "direct.host.cpp",
            "sites": temporary / "direct.sites.h",
            "stubs": temporary / "direct.stubs.cpp",
            "backend": temporary / "direct.backend.cpp",
        }
        direct_command = extractor_command(
            extractor, clangxx, repository, source, outputs
        )
        first = run(direct_command, repository)
        require_ok(checks, first, "direct semantic extraction")
        for label, path in outputs.items():
            checks.require(path.is_file() and path.stat().st_size > 0, f"missing {label} output")

        document = json.loads(outputs["ir"].read_text(encoding="utf-8"))
        checks.require(document.get("version") == 1, "semantic extraction did not retain Matcore IR v1")
        checks.require(document.get("producer") == "clang-libtooling-v1", "semantic extraction lost native provenance")
        checks.require(len(document.get("operations", [])) == 1, "semantic extraction operation count changed")
        captured_operation = document["operations"][0]
        captured_site = captured_operation["site_id"]
        captured_source = captured_operation["source"]

        semantic_text = outputs["semantic"].read_text(encoding="utf-8")
        checks.require('mdsl.bridge_schema = "matcore-mlir-semantic-v1"' in semantic_text, "missing semantic bridge schema")
        checks.require('mdsl.execution_intent = "generic"' in semantic_text, "missing generic execution intent")
        checks.require(semantic_text.count("mdsl.gemm") == 1, "semantic module does not contain exactly one GEMM")
        checks.require("mdsl.map" not in semantic_text, "explicit GEMM bridge invented composition")

        structured_text = outputs["structured"].read_text(encoding="utf-8")
        checks.require(
            'mdsl.structured_handoff_schema = "matcore-structured-gemm-handoff-v1"'
            in structured_text,
            "structured handoff lacks its versioned schema",
        )
        checks.require(
            "mdsl.structured_handoff_version = 1 : i32" in structured_text,
            "structured handoff lacks its schema version",
        )
        checks.require(
            "mdsl.analysis_only = true" in structured_text
            and 'mdsl.execution_authority = "inspection_only"'
            in structured_text,
            "structured handoff acquired execution authority",
        )
        checks.require(
            'mdsl.producer = "matcore-structured-gemm-handoff-v1"'
            in structured_text
            and 'mdsl.source_producer = "clang-libtooling-v1"'
            in structured_text
            and 'mdsl.source_bridge_schema = "matcore-mlir-semantic-v1"'
            in structured_text,
            "structured handoff conflates capture, semantic bridge, and stage provenance",
        )
        structured_identity = re.search(
            r"func\.func @__matcore_structured_(mc_[0-9a-f]{32})",
            structured_text,
        )
        checks.require(
            structured_identity is not None
            and structured_identity.group(1) == captured_site
            and f"func.func @__matcore_semantic_{captured_site}" in semantic_text
            and f'mdsl.site_id = "{structured_identity.group(1)}"'
            in structured_text
            and (
                'mdsl.source_semantic_symbol = "__matcore_semantic_'
                f'{structured_identity.group(1)}"'
            )
            in structured_text
            and "mdsl.capture_ordinal = 0 : i64" in structured_text,
            "structured handoff lost or changed site identity and capture order",
        )
        checks.require(
            f'mdsl.source_file = "{captured_source["file"]}"'
            in structured_text
            and (
                f'mdsl.translation_unit = "'
                f'{document["translation_unit"]["identity"]}"'
            )
            in structured_text
            and (
                f'loc("{captured_source["file"]}":'
                f'{captured_source["line"]}:{captured_source["column"]})'
            )
            in structured_text
            and f'line = {captured_source["line"]} : i64' in structured_text
            and f'column = {captured_source["column"]} : i64'
            in structured_text
            and f'offset = {captured_source["byte_offset"]} : i64'
            in structured_text,
            "structured handoff differs from captured source provenance",
        )
        checks.require(
            "mdsl.semantic_contract = {" in structured_text,
            "structured handoff lost the self-contained semantic contract",
        )
        for contract_field in (
            "accumulation_type = f32",
            "aliasing = [",
            'effects = {read_write = [], reads = ["lhs", "rhs"], writes = ["output"]}',
            "lhs_semantics = {",
            "numerical = {",
            "origin = {",
            "output_semantics = {",
            "policy = {",
            "provenance = {",
            "rhs_semantics = {",
            "semantic_requirements = [",
            'site_id = "mc_',
            'synchronization = "synchronous"',
        ):
            checks.require(
                contract_field in structured_text,
                f"structured handoff lost semantic-contract field: {contract_field}",
            )
        for tensor_contract in (
            'lhs_semantics = {alignment_bytes = 4 : i64, alignment_contract = "required_precondition", layout = "row_major_contiguous", memory_space = "host", mutability = "read", role = "lhs", shape = [{kind = "dynamic", symbol = "m"}, {kind = "dynamic", symbol = "k"}]',
            'rhs_semantics = {alignment_bytes = 4 : i64, alignment_contract = "required_precondition", layout = "row_major_contiguous", memory_space = "host", mutability = "read", role = "rhs", shape = [{kind = "dynamic", symbol = "k"}, {kind = "dynamic", symbol = "n"}]',
            'output_semantics = {alignment_bytes = 4 : i64, alignment_contract = "required_precondition", layout = "row_major_contiguous", memory_space = "host", mutability = "write", role = "output", shape = [{kind = "dynamic", symbol = "m"}, {kind = "dynamic", symbol = "n"}]',
        ):
            checks.require(
                tensor_contract in structured_text,
                "structured handoff lost an exact shape/layout/alignment contract",
            )
        for numerical_contract in (
            'profile = "explicit-gemm-f32-v1"',
            'reassociation = "within_k_reduction"',
            'reduction_order = "implementation_defined_within_k"',
            'signed_zero = "relaxed"',
            'subnormals = "ieee_gradual_ftz_daz_forbidden"',
            "approximate_math = false",
        ):
            checks.require(
                numerical_contract in structured_text,
                f"structured handoff lost numerical contract: {numerical_contract}",
            )
        checks.require(
            structured_text.count('contract = "required_precondition"') == 5
            and structured_text.count('relation = "no_alias"') == 2,
            "structured handoff weakened alignment or no-alias preconditions",
        )
        fill_position = structured_text.find("linalg.fill ")
        matmul_position = structured_text.find("linalg.matmul ")
        checks.require(
            fill_position >= 0
            and matmul_position > fill_position
            and structured_text.count("linalg.fill ") == 1
            and structured_text.count("linalg.matmul ") == 1,
            "structured handoff is not the exact fill-then-matmul sequence",
        )
        checks.require(
            'mdsl.structured_role = "destination_overwrite_zero_fill"'
            in structured_text
            and 'destination = "original_output_full_zero_fill"'
            in structured_text
            and "outs(%arg2 : tensor<?x?xf32>)" in structured_text
            and "outs(%0 : tensor<?x?xf32>)" in structured_text
            and "return %1 : tensor<?x?xf32>" in structured_text,
            "structured handoff lost overwrite-output or SSA destination semantics",
        )
        checks.require(
            'source_operation = "mdsl.gemm"' in structured_text
            and not contains_mdsl_gemm_operation(structured_text),
            "structured handoff retained an mdsl.gemm operation instead of a provenance fact",
        )
        checks.require(
            "fastmath" not in structured_text,
            "structured handoff silently introduced fast-math flags",
        )
        backend_text = outputs["backend"].read_text(encoding="utf-8")
        checks.require("Producer: Matcore MLIR CPU runtime-dispatch lowering v1" in backend_text, "backend lacks semantic lowering provenance")
        checks.require("matcore_runtime_gemm_f32_v0" in backend_text, "backend bypassed the stable runtime boundary")

        before = {name: path.read_bytes() for name, path in outputs.items()}
        second = run(direct_command, repository)
        require_ok(checks, second, "repeated semantic extraction")
        for label, path in outputs.items():
            checks.require(path.read_bytes() == before[label], f"{label} output is not byte deterministic")

        outputs["structured"].unlink()
        without_structured = run(
            extractor_command(
                extractor,
                clangxx,
                repository,
                source,
                outputs,
                emit_structured=False,
            ),
            repository,
        )
        require_ok(checks, without_structured, "semantic extraction without structured handoff")
        checks.require(
            not outputs["structured"].exists(),
            "extractor emitted a structured artifact without an explicit request",
        )
        for label, path in outputs.items():
            if label == "structured":
                continue
            checks.require(
                path.read_bytes() == before[label],
                f"requesting the structured handoff changed the {label} artifact",
            )

        two_ir = temporary / "two.matcore.json"
        two_semantic = temporary / "two.semantic.mlir"
        two_structured = temporary / "two.structured.mlir"
        two_command = [
            str(extractor),
            "--frontend=native",
            "--semantic-pipeline=matcore-mlir",
            "--ir-version=1",
            "--input",
            str(two_sites),
            "--ir-out",
            str(two_ir),
            "--semantic-ir-out",
            str(two_semantic),
            "--structured-ir-out",
            str(two_structured),
            "--",
            str(clangxx),
            "-std=c++20",
            f"-I{repository / 'compiler/include'}",
            str(two_sites),
        ]
        two = run(two_command, repository)
        require_ok(checks, two, "two-site semantic extraction")
        two_document = json.loads(two_ir.read_text(encoding="utf-8"))
        captured_site_ids = [
            operation["site_id"] for operation in two_document["operations"]
        ]
        two_text = two_semantic.read_text(encoding="utf-8")
        site_ids = re.findall(
            r"func\.func @__matcore_semantic_(mc_[0-9a-f]{32})", two_text
        )
        checks.require(
            site_ids == captured_site_ids and len(set(site_ids)) == 2,
            "two semantic roots differ from captured site identity/order",
        )
        two_structured_text = two_structured.read_text(encoding="utf-8")
        structured_site_ids = re.findall(
            r"func\.func @__matcore_structured_(mc_[0-9a-f]{32})",
            two_structured_text,
        )
        checks.require(
            structured_site_ids == captured_site_ids,
            "two structured roots differ from captured site identity/order",
        )
        checks.require(
            two_structured_text.count("linalg.fill ") == 2
            and two_structured_text.count("linalg.matmul ") == 2
            and not contains_mdsl_gemm_operation(two_structured_text),
            "two-site structured handoff changed its per-site operation shape",
        )

        negative_cases = [
            (
                "missing-v1",
                [
                    str(extractor),
                    "--semantic-pipeline=matcore-mlir",
                    "--input",
                    str(source),
                    "--ir-out",
                    str(temporary / "missing-v1.json"),
                ],
                "requires --ir-version=1",
                temporary / "missing-v1.json",
            ),
            (
                "bootstrap",
                [
                    str(extractor),
                    "--frontend=ast-json-bootstrap",
                    "--semantic-pipeline=matcore-mlir",
                    "--ir-version=1",
                    "--input",
                    str(source),
                    "--ir-out",
                    str(temporary / "bootstrap.json"),
                ],
                "requires the authenticated native frontend",
                temporary / "bootstrap.json",
            ),
            (
                "analysis-only-recovery",
                [
                    str(extractor),
                    "--semantic-pipeline=matcore-mlir",
                    "--ir-version=1",
                    "--inspect-recovered-gemm",
                    str(temporary / "recovered.json"),
                    "--input",
                    str(source),
                    "--ir-out",
                    str(temporary / "recovery-ir.json"),
                ],
                "analysis-only",
                temporary / "recovery-ir.json",
            ),
            (
                "structured-with-capture-v0",
                [
                    str(extractor),
                    "--frontend=native",
                    "--semantic-pipeline=capture-v0",
                    "--ir-version=1",
                    "--input",
                    str(source),
                    "--ir-out",
                    str(temporary / "capture-structured.json"),
                    "--structured-ir-out",
                    str(temporary / "capture-structured.mlir"),
                ],
                "--structured-ir-out requires --semantic-pipeline=matcore-mlir",
                temporary / "capture-structured.mlir",
            ),
            (
                "structured-output-alias",
                [
                    str(extractor),
                    "--frontend=native",
                    "--semantic-pipeline=matcore-mlir",
                    "--ir-version=1",
                    "--input",
                    str(source),
                    "--ir-out",
                    str(temporary / "shared-structured-output"),
                    "--structured-ir-out",
                    str(temporary / "shared-structured-output"),
                ],
                "--structured-ir-out and --ir-out must refer to distinct output files",
                temporary / "shared-structured-output",
            ),
        ]
        for label, command, diagnostic, forbidden in negative_cases:
            rejected = run(command, repository)
            checks.require(rejected.returncode != 0, f"{label} request was accepted")
            checks.require(diagnostic in rejected.stderr, f"{label} rejection lacked actionable diagnostic: {rejected.stderr}")
            checks.require(not forbidden.exists(), f"{label} rejection emitted an artifact")

        driver_no_target = temporary / "driver-no-target"
        no_target = run(
            [
                str(driver),
                "--semantic-pipeline=matcore-mlir",
                "-std=c++20",
                str(source),
                "-o",
                str(driver_no_target),
            ],
            repository,
        )
        checks.require(no_target.returncode != 0, "driver accepted semantic pipeline without CPU target")
        checks.require("requires --matcore-target=cpu" in no_target.stderr, "driver no-target rejection lacked actionable diagnostic")
        checks.require(not driver_no_target.exists(), "driver no-target rejection emitted an artifact")

        driver_bootstrap = temporary / "driver-bootstrap"
        bootstrap_driver = run(
            [
                str(driver),
                "--frontend=ast-json-bootstrap",
                "--semantic-pipeline=matcore-mlir",
                "--matcore-target=cpu",
                "-std=c++20",
                str(source),
                "-o",
                str(driver_bootstrap),
            ],
            repository,
        )
        checks.require(bootstrap_driver.returncode != 0, "driver accepted bootstrap semantic pipeline")
        checks.require("requires the authenticated native frontend" in bootstrap_driver.stderr, "driver bootstrap rejection lacked actionable diagnostic")
        checks.require(not driver_bootstrap.exists(), "driver bootstrap rejection emitted an artifact")

        executable = temporary / "semantic-program"
        compiled = run(
            [
                str(driver),
                "--semantic-pipeline=matcore-mlir",
                "--save-temps",
                "--matcore-target=cpu",
                "-std=c++20",
                str(source),
                "-o",
                str(executable),
            ],
            repository,
            timeout=180,
        )
        require_ok(checks, compiled, "semantic driver final link")
        executed = run([str(executable)], temporary)
        require_ok(checks, executed, "semantic driver executable")
        checks.require(executed.stdout == "host-before\nMDSLC CPU GEMM PASS\n", "semantic executable produced wrong output")
        saved_semantic = temporary / "semantic-program.semantic.mlir"
        saved_backend = temporary / "semantic-program.backend.cpp"
        saved_ir = temporary / "semantic-program.matcore.json"
        checks.require(saved_semantic.is_file(), "--save-temps omitted semantic MLIR")
        checks.require(saved_backend.is_file(), "--save-temps omitted semantic backend")
        checks.require(saved_ir.is_file(), "--save-temps omitted typed capture IR")
        checks.require("mdsl.gemm" in saved_semantic.read_text(encoding="utf-8"), "saved semantic MLIR lost GEMM")
        checks.require("Producer: Matcore MLIR CPU runtime-dispatch lowering v1" in saved_backend.read_text(encoding="utf-8"), "saved backend used capture-v0 producer")

        default_executable = temporary / "configured-default-program"
        default_compile = run(
            [
                str(driver),
                "--save-temps",
                "--matcore-target=cpu",
                "-std=c++20",
                str(source),
                "-o",
                str(default_executable),
            ],
            repository,
            timeout=180,
        )
        require_ok(checks, default_compile, "configured-default driver link")
        default_run = run([str(default_executable)], temporary)
        require_ok(checks, default_run, "configured-default executable")
        checks.require(
            default_run.stdout == "host-before\nMDSLC CPU GEMM PASS\n",
            "configured-default executable produced wrong output",
        )
        default_semantic = temporary / "configured-default-program.semantic.mlir"
        default_backend = temporary / "configured-default-program.backend.cpp"
        checks.require(
            default_backend.is_file(),
            "configured-default driver omitted its saved backend",
        )
        default_backend_text = default_backend.read_text(encoding="utf-8")
        checks.require(
            "matcore_runtime_gemm_f32_v0" in default_backend_text,
            "configured-default backend omitted the stable runtime dispatch",
        )
        if arguments.expected_default_semantic_pipeline == "matcore-mlir":
            checks.require(
                default_semantic.is_file(),
                "configured matcore-mlir default omitted semantic MLIR",
            )
            checks.require(
                default_semantic.read_bytes() == saved_semantic.read_bytes(),
                "configured default and explicit matcore-mlir differ",
            )
            checks.require(
                "Producer: Matcore MLIR CPU runtime-dispatch lowering v1"
                in default_backend_text,
                "configured matcore-mlir default emitted inspection IR but "
                "executed a non-semantic backend",
            )
        else:
            checks.require(
                not default_semantic.exists(),
                "configured capture-v0 default emitted semantic MLIR",
            )
            checks.require(
                "Producer: Matcore MLIR CPU runtime-dispatch lowering v1"
                not in default_backend_text,
                "configured capture-v0 default used the Matcore MLIR backend",
            )

        capture_executable = temporary / "explicit-capture-program"
        capture_compile = run(
            [
                str(driver),
                "--semantic-pipeline=capture-v0",
                "--save-temps",
                "--matcore-target=cpu",
                "-std=c++20",
                str(source),
                "-o",
                str(capture_executable),
            ],
            repository,
            timeout=180,
        )
        require_ok(checks, capture_compile, "explicit capture-v0 override link")
        capture_run = run([str(capture_executable)], temporary)
        require_ok(checks, capture_run, "explicit capture-v0 override executable")
        checks.require(
            capture_run.stdout == "host-before\nMDSLC CPU GEMM PASS\n",
            "explicit capture-v0 override produced wrong output",
        )
        checks.require(
            not (temporary / "explicit-capture-program.semantic.mlir").exists(),
            "explicit capture-v0 override emitted semantic MLIR",
        )
        capture_backend = temporary / "explicit-capture-program.backend.cpp"
        checks.require(
            capture_backend.is_file(),
            "explicit capture-v0 override omitted its saved backend",
        )
        capture_backend_text = capture_backend.read_text(encoding="utf-8")
        checks.require(
            "matcore_runtime_gemm_f32_v0" in capture_backend_text,
            "explicit capture-v0 backend omitted stable runtime dispatch",
        )
        checks.require(
            "Producer: Matcore MLIR CPU runtime-dispatch lowering v1"
            not in capture_backend_text,
            "explicit capture-v0 override used the Matcore MLIR backend",
        )

        object_path = temporary / "semantic-object.o"
        object_compile = run(
            [
                str(driver),
                "--semantic-pipeline=matcore-mlir",
                "--matcore-target=cpu",
                "-std=c++20",
                "-c",
                str(source),
                "-o",
                str(object_path),
            ],
            repository,
            timeout=180,
        )
        require_ok(checks, object_compile, "semantic driver object production")
        readelf = run(["readelf", "-h", str(object_path)], repository)
        require_ok(checks, readelf, "semantic object inspection")
        checks.require("REL (Relocatable file)" in readelf.stdout, "semantic output is not an ELF relocatable object")
        symbols = run(["nm", "-C", str(object_path)], repository)
        require_ok(checks, symbols, "semantic object symbol inspection")
        checks.require("matcore_generated_backend_" in symbols.stdout, "semantic object lacks generated backend symbol")
        checks.require("matcore_runtime_gemm_f32_v0" in symbols.stdout, "semantic object lacks stable runtime reference")

        externally_linked = temporary / "semantic-external"
        linked = run(
            [
                str(clangxx),
                str(object_path),
                f"-L{build_dir / 'lib'}",
                "-lmatcore_runtime",
                f"-Wl,-rpath,{build_dir / 'lib'}",
                "-o",
                str(externally_linked),
            ],
            repository,
        )
        require_ok(checks, linked, "ordinary external semantic link")
        external_run = run([str(externally_linked)], temporary)
        require_ok(checks, external_run, "externally linked semantic executable")
        checks.require(external_run.stdout == "host-before\nMDSLC CPU GEMM PASS\n", "external semantic executable produced wrong output")

        ordinary_source = temporary / "ordinary-host.mdsl"
        ordinary_source.write_text(
            "#include <matcore/mdsl.h>\n"
            "#include <iostream>\n"
            "int main() { std::cout << \"ordinary-host\\n\"; return 0; }\n",
            encoding="utf-8",
        )
        ordinary_executable = temporary / "ordinary-host"
        ordinary_compile = run(
            [
                str(driver),
                "--semantic-pipeline=matcore-mlir",
                "--matcore-target=cpu",
                "-std=c++20",
                str(ordinary_source),
                "-o",
                str(ordinary_executable),
            ],
            repository,
            timeout=180,
        )
        require_ok(
            checks, ordinary_compile, "ordinary C++ through semantic pipeline"
        )
        ordinary_run = run([str(ordinary_executable)], temporary)
        require_ok(checks, ordinary_run, "ordinary C++ semantic executable")
        checks.require(
            ordinary_run.stdout == "ordinary-host\n",
            "ordinary host behavior changed in semantic pipeline",
        )

    print(f"Matcore MLIR CPU pipeline: {checks.count} checks, 0 failures")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (Failure, OSError, json.JSONDecodeError) as error:
        print(f"Matcore MLIR CPU pipeline: FAIL: {error}", file=__import__("sys").stderr)
        raise SystemExit(1)
