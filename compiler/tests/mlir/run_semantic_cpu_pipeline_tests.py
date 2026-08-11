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
) -> list[str]:
    return [
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
        "--",
        str(clangxx),
        "-std=c++20",
        f"-I{repository / 'compiler/include'}",
        str(source),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extractor", type=Path, required=True)
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--clangxx", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
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

        semantic_text = outputs["semantic"].read_text(encoding="utf-8")
        checks.require('mdsl.bridge_schema = "matcore-mlir-semantic-v1"' in semantic_text, "missing semantic bridge schema")
        checks.require('mdsl.execution_intent = "generic"' in semantic_text, "missing generic execution intent")
        checks.require(semantic_text.count("mdsl.gemm") == 1, "semantic module does not contain exactly one GEMM")
        checks.require("mdsl.map" not in semantic_text, "explicit GEMM bridge invented composition")
        backend_text = outputs["backend"].read_text(encoding="utf-8")
        checks.require("Producer: Matcore MLIR CPU runtime-dispatch lowering v1" in backend_text, "backend lacks semantic lowering provenance")
        checks.require("matcore_runtime_gemm_f32_v0" in backend_text, "backend bypassed the stable runtime boundary")

        before = {name: path.read_bytes() for name, path in outputs.items()}
        second = run(direct_command, repository)
        require_ok(checks, second, "repeated semantic extraction")
        for label, path in outputs.items():
            checks.require(path.read_bytes() == before[label], f"{label} output is not byte deterministic")

        two_ir = temporary / "two.matcore.json"
        two_semantic = temporary / "two.semantic.mlir"
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
            "--",
            str(clangxx),
            "-std=c++20",
            f"-I{repository / 'compiler/include'}",
            str(two_sites),
        ]
        two = run(two_command, repository)
        require_ok(checks, two, "two-site semantic extraction")
        two_text = two_semantic.read_text(encoding="utf-8")
        checks.require(two_text.count("func.func @__matcore_semantic_mc_") == 2, "two capture sites were not independent semantic roots")
        site_ids = re.findall(r'mdsl.site_id = "(mc_[0-9a-f]{32})"', two_text)
        checks.require(len(site_ids) == 2 and len(set(site_ids)) == 2, "two semantic sites collided")

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
