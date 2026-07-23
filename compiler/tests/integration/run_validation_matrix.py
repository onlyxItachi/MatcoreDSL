#!/usr/bin/env python3
"""End-to-end validation for the implemented standalone MDSLC v0 slice."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


SOURCE_DIAGNOSTIC = re.compile(r"(?:^|\n)[^\n:]+:\d+:\d+: (?:fatal )?error:")
GENERATED_SUFFIXES = (
    ".host.cpp",
    ".matcore.json",
    ".sites.h",
    ".stubs.cpp",
    ".backend.cpp",
    ".host.o",
    ".stubs.o",
    ".backend.o",
)


class ValidationFailure(RuntimeError):
    pass


class Context:
    def __init__(self, repository: Path, build_dir: Path) -> None:
        self.repository = repository
        self.build_dir = build_dir
        self.driver = build_dir / "bin" / "mdslc++"
        self.extractor = build_dir / "bin" / "matcore-extract"
        self.runtime_test = build_dir / "bin" / "matcore_runtime_cpu_test"
        self.runtime_directory = build_dir / "lib"
        self.clang = Path("/usr/bin/clang++-21")
        self.clang_c = Path("/usr/bin/clang-21")


def run(
    argv: list[str], cwd: Path, timeout: int = 90
) -> subprocess.CompletedProcess[str]:
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
        raise ValidationFailure(f"command timed out: {argv!r}") from error


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationFailure(message)


def completed_ok(completed: subprocess.CompletedProcess[str], label: str) -> None:
    require(
        completed.returncode == 0,
        f"{label} exited {completed.returncode}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
    )


def source_argument(context: Context, case: dict[str, object]) -> str:
    source = case.get("source")
    require(isinstance(source, str), "case has no source")
    path = context.repository / source
    require(path.is_file(), f"missing source fixture: {source}")
    return source


def extraction_command(
    context: Context,
    source: str,
    output: Path,
    generated: dict[str, Path] | None = None,
) -> list[str]:
    command = [
        str(context.extractor),
        "--input",
        source,
        "--ir-out",
        str(output),
    ]
    if generated is not None:
        command.extend(
            [
                "--rewrite-out",
                str(generated["host"]),
                "--sites-out",
                str(generated["sites"]),
                "--stubs-out",
                str(generated["stubs"]),
                "--backend-out",
                str(generated["backend"]),
            ]
        )
    source_path = Path(source)
    include_parent = (
        source_path.parent
        if not source_path.is_absolute()
        else source_path.resolve().parent
    )
    command.extend(
        [
            "--",
            "clang++",
            "-std=c++20",
            f"-I{context.repository / 'compiler/include'}",
            f"-I{include_parent}",
            source,
        ]
    )
    return command


def load_and_validate_ir(
    path: Path, expected_operations: int, expected_policy: dict[str, str] | None = None
) -> dict[str, object]:
    try:
        encoded = path.read_bytes()
        document = json.loads(encoded)
    except (OSError, json.JSONDecodeError) as error:
        raise ValidationFailure(f"invalid emitted JSON {path}: {error}") from error
    require(document.get("schema") == "matcore.ir", "wrong IR schema")
    require(document.get("version") == 0, "wrong IR version")
    require(
        document.get("producer") == "clang-libtooling-v1",
        "native frontend producer is not explicit",
    )
    operations = document.get("operations")
    require(isinstance(operations, list), "operations is not an array")
    require(
        len(operations) == expected_operations,
        f"expected {expected_operations} operation(s), found {len(operations)}",
    )
    for operation in operations:
        require(operation.get("kind") == "gemm", "unexpected operation kind")
        require(
            operation.get("canonical_callee") == "matcore::mdsl::gemm",
            "operation was not recognized through the canonical declaration",
        )
        location = operation.get("source", {})
        require(location.get("line", 0) > 0, "source line was lost")
        require(location.get("column", 0) > 0, "source column was lost")
        require(str(location.get("file", "")).endswith(".mdsl"), "source file was lost")
        if expected_policy is not None:
            require(operation.get("policy") == expected_policy, "wrong explicit policy")
    return document


def extract(
    context: Context,
    source: str,
    output: Path,
    operations: int,
    expected_policy: dict[str, str] | None = None,
) -> dict[str, object]:
    completed = run(
        extraction_command(context, source, output), context.repository
    )
    completed_ok(completed, "extraction")
    return load_and_validate_ir(output, operations, expected_policy)


def generated_source(scenario: str) -> str:
    if scenario == "textual_spoof":
        return """namespace matcore::mdsl {
inline void gemm(int) {}
}
int main() {
  matcore::mdsl::gemm(1);
  return 0;
}
"""
    if scenario == "policy_variable":
        return """#include <matcore/mdsl.h>
namespace md = matcore::mdsl;
int main() {
  float values[3] = {1, 1, 0};
  md::matrix_view a{values, 1, 1};
  md::matrix_view b{values + 1, 1, 1};
  md::matrix_view c{values + 2, 1, 1};
  md::policy selected{};
  md::gemm(md::out(c), a, b, selected);
  return 0;
}
"""
    raise ValidationFailure(f"unknown generated frontend scenario: {scenario}")


def runtime_error_source(scenario: str, diagnostic: str) -> str:
    if scenario == "shape_mismatch":
        setup = """float a_data[6] = {};
  float b_data[4] = {};
  float c_data[4] = {};
  md::matrix_view a{a_data, 2, 3};
  md::matrix_view b{b_data, 2, 2};
  md::matrix_view c{c_data, 2, 2};"""
    elif scenario == "runtime_alias":
        setup = """float a_data[4] = {};
  float b_data[4] = {};
  md::matrix_view a{a_data, 2, 2};
  md::matrix_view b{b_data, 2, 2};
  md::matrix_view c{a_data, 2, 2};"""
    else:
        raise ValidationFailure(f"unknown generated runtime scenario: {scenario}")
    return f"""#include <matcore/mdsl.h>
#include <iostream>
#include <stdexcept>
#include <string>
namespace md = matcore::mdsl;
int main() {{
  {setup}
  try {{
    md::gemm(md::out(c), a, b,
             md::policy{{.target = md::target::cpu,
                        .fallback = md::fallback::error}});
  }} catch (const std::runtime_error &error) {{
    const std::string message = error.what();
    std::cout << message << '\\n';
    return message.find({json.dumps(diagnostic)}) != std::string::npos ? 0 : 2;
  }}
  return 3;
}}
"""


def inspect_relocatable(context: Context, path: Path) -> None:
    file_result = run(["file", str(path)], context.repository)
    completed_ok(file_result, "file inspection")
    require("ELF 64-bit" in file_result.stdout, "object is not ELF64")
    require("relocatable" in file_result.stdout, "artifact is not relocatable")
    elf = run(["readelf", "-h", str(path)], context.repository)
    completed_ok(elf, "readelf inspection")
    require("REL (Relocatable file)" in elf.stdout, "ELF type is not REL")
    require("Advanced Micro Devices X86-64" in elf.stdout, "wrong ELF machine")


def generated_paths(root: Path, stem: str) -> dict[str, Path]:
    return {
        "ir": root / f"{stem}.matcore.json",
        "host": root / f"{stem}.host.cpp",
        "sites": root / f"{stem}.sites.h",
        "stubs": root / f"{stem}.stubs.cpp",
        "backend": root / f"{stem}.backend.cpp",
    }


def execute_case(context: Context, case: dict[str, object], work: Path) -> None:
    mode = str(case["mode"])

    if mode in {"extract", "extract_policy", "extract_distinct_sites"}:
        source = source_argument(context, case)
        policy = None
        if mode == "extract_policy":
            policy = {"target": str(case["target"]), "fallback": str(case["fallback"])}
        document = extract(
            context, source, work / "output.json", int(case["operations"]), policy
        )
        if mode == "extract_distinct_sites":
            identifiers = [operation["site_id"] for operation in document["operations"]]
            require(len(identifiers) == len(set(identifiers)), "site IDs collided")
            repeated = extract(
                context,
                source,
                work / "repeated.json",
                int(case["operations"]),
            )
            repeated_identifiers = [
                operation["site_id"] for operation in repeated["operations"]
            ]
            require(
                identifiers == repeated_identifiers,
                "site IDs changed across identical extractions",
            )
        return

    if mode in {"generated_extract", "generated_reject"}:
        source_path = work / f"{case['scenario']}.mdsl"
        source_path.write_text(generated_source(str(case["scenario"])))
        output = work / "output.json"
        completed = run(
            extraction_command(context, str(source_path), output), context.repository
        )
        if mode == "generated_extract":
            completed_ok(completed, "generated-source extraction")
            load_and_validate_ir(output, int(case["operations"]))
        else:
            require(completed.returncode != 0, "negative extraction unexpectedly succeeded")
            require(str(case["diagnostic"]) in completed.stderr, "expected diagnostic missing")
            require(SOURCE_DIAGNOSTIC.search(completed.stderr) is not None, "source line/column missing")
            require(not output.exists(), "rejected source emitted IR")
        return

    if mode == "extract_reject":
        source = source_argument(context, case)
        output = work / "rejected.json"
        completed = run(
            extraction_command(context, source, output), context.repository
        )
        require(completed.returncode != 0, "negative extraction unexpectedly succeeded")
        require(str(case["diagnostic"]) in completed.stderr, "expected diagnostic missing")
        require(SOURCE_DIAGNOSTIC.search(completed.stderr) is not None, "source line/column missing")
        require(not output.exists(), "rejected source emitted IR")
        return

    if mode == "deterministic_ir":
        source = source_argument(context, case)
        first = work / "first.json"
        second = work / "second.json"
        first_document = extract(context, source, first, int(case["operations"]))
        second_document = extract(context, source, second, int(case["operations"]))
        require(first.read_bytes() == second.read_bytes(), "IR bytes changed across runs")
        require(first_document == second_document, "IR structure changed across runs")
        verified = run([str(context.extractor), "--verify-ir", str(first)], context.repository)
        completed_ok(verified, "serialized IR verification")
        require("verified Matcore IR v0" in verified.stdout, "verifier success message missing")
        return

    if mode == "verify_ir_reject":
        source = source_argument(context, case)
        completed = run(
            [str(context.extractor), "--verify-ir", source], context.repository
        )
        require(completed.returncode != 0, "invalid IR unexpectedly verified")
        require(str(case["diagnostic"]) in completed.stderr, "IR diagnostic missing")
        require(source in completed.stderr and "error:" in completed.stderr, "IR path diagnostic missing")
        return

    if mode == "deterministic_generation":
        source = source_argument(context, case)
        roots = [work / "first", work / "second"]
        artifact_sets: list[dict[str, Path]] = []
        for root in roots:
            root.mkdir()
            paths = generated_paths(root, "capture")
            completed = run(
                extraction_command(context, source, paths["ir"], paths),
                context.repository,
            )
            completed_ok(completed, "generated artifact extraction")
            artifact_sets.append(paths)
        for kind in artifact_sets[0]:
            require(
                artifact_sets[0][kind].read_bytes() == artifact_sets[1][kind].read_bytes(),
                f"generated {kind} is not byte deterministic",
            )
        require("#line " in artifact_sets[0]["host"].read_text(), "host source lost #line mapping")
        require("__matcore_call_site_" in artifact_sets[0]["host"].read_text(), "host call was not rewritten")
        require("extern \"C\"" in artifact_sets[0]["backend"].read_text(), "backend lacks C ABI")
        return

    if mode in {"driver_host_run", "driver_after_separator"}:
        source = source_argument(context, case)
        output = work / "program"
        argv = [str(context.driver), "--verbose", "-std=c++20", "-o", str(output)]
        if mode == "driver_after_separator":
            argv.extend(["--", source])
        else:
            argv.append(source)
        compiled = run(argv, context.repository)
        completed_ok(compiled, "host driver compile")
        require("'-x' 'c++'" in compiled.stderr, "verbose argv lacks forced -x c++")
        executed = run([str(output)], work)
        completed_ok(executed, "host executable")
        require(executed.stdout == str(case["stdout"]), "host stdout mismatch")
        return

    if mode == "driver_cpu_final":
        source = source_argument(context, case)
        output = work / "program"
        compiled = run(
            [str(context.driver), "--matcore-target=cpu", "-std=c++20", source, "-o", str(output)],
            context.repository,
        )
        completed_ok(compiled, "CPU pipeline final link")
        executed = run([str(output)], work)
        completed_ok(executed, "CPU pipeline executable")
        require(executed.stdout == str(case["stdout"]), "CPU pipeline stdout mismatch")
        return

    if mode == "driver_cpu_saved_object":
        source = source_argument(context, case)
        output = work / "gemm_v0.o"
        compiled = run(
            [str(context.driver), "--matcore-target=cpu", "--save-temps", "-std=c++20", "-c", source, "-o", str(output)],
            context.repository,
        )
        completed_ok(compiled, "saved CPU relocatable pipeline")
        expected = [work / f"gemm_v0{suffix}" for suffix in GENERATED_SUFFIXES]
        expected.append(output)
        require(all(path.is_file() and path.stat().st_size > 0 for path in expected), "saved artifact set is incomplete")
        inspect_relocatable(context, output)
        symbols = run(["nm", "-C", str(output)], context.repository)
        completed_ok(symbols, "symbol inspection")
        for symbol in (" main", "__matcore_call_site_", "matcore_generated_backend_", "matcore_runtime_gemm_f32_v0"):
            require(symbol in symbols.stdout, f"missing object symbol: {symbol}")
        verified = run([str(context.extractor), "--verify-ir", str(work / "gemm_v0.matcore.json")], context.repository)
        completed_ok(verified, "saved IR verification")
        saved_ir = json.loads((work / "gemm_v0.matcore.json").read_text())
        require(saved_ir.get("version") == 1, "mdslc++ did not save typed Matcore IR v1")
        require("verified Matcore IR v1" in verified.stdout, "saved IR used the wrong verifier")
        return

    if mode == "driver_cpu_external_link":
        source = source_argument(context, case)
        object_path = work / "gemm_external.o"
        compiled = run(
            [str(context.driver), "--matcore-target=cpu", "-std=c++20", "-c", source, "-o", str(object_path)],
            context.repository,
        )
        completed_ok(compiled, "CPU relocatable production")
        inspect_relocatable(context, object_path)
        executable = work / "gemm_external"
        linked = run(
            [
                str(context.clang),
                str(object_path),
                f"-L{context.runtime_directory}",
                "-lmatcore_runtime",
                f"-Wl,-rpath,{context.runtime_directory}",
                "-o",
                str(executable),
            ],
            context.repository,
        )
        completed_ok(linked, "ordinary external clang++ link")
        executed = run([str(executable)], work)
        completed_ok(executed, "externally linked executable")
        require(executed.stdout == str(case["stdout"]), "external executable stdout mismatch")
        dependencies = run(["ldd", str(executable)], work)
        completed_ok(dependencies, "ldd inspection")
        require("libmatcore_runtime.so.0" in dependencies.stdout, "runtime SONAME not resolved")
        require("not found" not in dependencies.stdout, "external executable has missing library")
        require(str(context.runtime_directory) in dependencies.stdout, "runtime did not resolve from selected build")
        return

    if mode == "driver_cpu_after_separator":
        source = source_argument(context, case)
        output = work / "after_separator.o"
        compiled = run(
            [str(context.driver), "--matcore-target=cpu", "-std=c++20", "-c", "-o", str(output), "--", source],
            context.repository,
        )
        completed_ok(compiled, "CPU pipeline after --")
        inspect_relocatable(context, output)
        return

    if mode == "driver_metachar":
        fixture = context.repository / source_argument(context, case)
        source = work / "hello;touch injection-marker.mdsl"
        marker = work / "injection-marker.mdsl"
        shutil.copyfile(fixture, source)
        output = work / "program"
        compiled = run(
            [str(context.driver), "-std=c++20", "-o", str(output), source.name], work
        )
        completed_ok(compiled, "metacharacter argv compile")
        executed = run([str(output)], work)
        completed_ok(executed, "metacharacter argv executable")
        require(executed.stdout == str(case["stdout"]), "metacharacter program stdout mismatch")
        require(not marker.exists(), "metacharacter filename triggered shell execution")
        return

    if mode == "driver_cpu_metachar":
        fixture = context.repository / source_argument(context, case)
        source = work / "gemm;touch injection-marker.mdsl"
        marker = work / "injection-marker.mdsl"
        shutil.copyfile(fixture, source)
        output = work / "program;still-one-argv"
        compiled = run(
            [str(context.driver), "--matcore-target=cpu", "-std=c++20", "-o", str(output), source.name],
            work,
        )
        completed_ok(compiled, "metacharacter CPU pipeline")
        executed = run([str(output)], work)
        completed_ok(executed, "metacharacter CPU executable")
        require(executed.stdout == str(case["stdout"]), "metacharacter CPU stdout mismatch")
        require(not marker.exists(), "CPU pipeline evaluated a metacharacter path")
        return

    if mode == "driver_missing_input":
        source = work / "does-not-exist.mdsl"
        output = work / "missing"
        completed = run(
            [str(context.driver), "-std=c++20", str(source), "-o", str(output)], work
        )
        require(completed.returncode == 1, "Clang missing-input status was not propagated")
        require(str(case["diagnostic"]) in completed.stderr, "missing-input diagnostic lost")
        require(str(source) in completed.stderr, "missing source path lost")
        require(not output.exists(), "missing input emitted an executable")
        return

    if mode == "driver_target_reject":
        output = work / "forbidden-target.o"
        completed = run(
            [str(context.driver), f"--matcore-target={case['target']}", "-std=c++20", "-c", "compiler/examples/gemm_v0.mdsl", "-o", str(output)],
            context.repository,
        )
        require(completed.returncode == 2, "unsupported target did not return driver usage failure")
        require(str(case["diagnostic"]) in completed.stderr, "no-fallback diagnostic missing")
        require(not output.exists(), "unsupported target emitted an artifact")
        return

    if mode == "driver_link_mode_reject":
        source = source_argument(context, case)
        output = work / "unsupported-output.so"
        completed = run(
            [
                str(context.driver),
                "--matcore-target=cpu",
                "-std=c++20",
                str(case["link_mode"]),
                source,
                "-o",
                str(output),
            ],
            context.repository,
        )
        require(completed.returncode == 2, "unsupported link mode was not rejected")
        require(str(case["diagnostic"]) in completed.stderr, "link-mode diagnostic missing")
        require(not output.exists(), "unsupported link mode emitted an artifact")
        return

    if mode == "driver_xlinker_mode_reject":
        source = source_argument(context, case)
        output = work / "unsupported-output.so"
        completed = run(
            [
                str(context.driver),
                "--matcore-target=cpu",
                "-std=c++20",
                "-Xlinker",
                str(case["link_mode"]),
                source,
                "-o",
                str(output),
            ],
            context.repository,
        )
        require(completed.returncode == 2, "-Xlinker mode was not rejected")
        require(str(case["diagnostic"]) in completed.stderr, "-Xlinker diagnostic missing")
        require(not output.exists(), "-Xlinker mode emitted an artifact")
        return

    if mode == "driver_link_response_reject":
        source = source_argument(context, case)
        output = work / "response-output.so"
        response = work / "linker.rsp"
        response.write_text("-shared\n", encoding="utf-8")
        forwarding = str(case["forwarding"])
        if forwarding == "wl":
            forwarded = [f"-Wl,@{response}"]
        elif forwarding == "xlinker":
            forwarded = ["-Xlinker", f"@{response}"]
        else:
            raise ValidationFailure(f"unknown linker response forwarding: {forwarding}")
        completed = run(
            [
                str(context.driver),
                "--matcore-target=cpu",
                "-std=c++20",
                *forwarded,
                source,
                "-o",
                str(output),
            ],
            context.repository,
        )
        require(completed.returncode == 2, "linker response file was not rejected")
        require(str(case["diagnostic"]) in completed.stderr, "response-file diagnostic missing")
        require(not output.exists(), "response-file link mode emitted an artifact")
        return

    if mode == "driver_overwrite_guard":
        fixture = context.repository / source_argument(context, case)
        source = work / "guard.mdsl"
        shutil.copyfile(fixture, source)
        before = hashlib.sha256(source.read_bytes()).digest()
        completed = run(
            [str(context.driver), "--matcore-target=cpu", "-std=c++20", "-c", str(source), "-o", str(source)],
            work,
        )
        require(completed.returncode == 2, "input overwrite was not rejected")
        require(str(case["diagnostic"]) in completed.stderr, "overwrite diagnostic missing")
        require(hashlib.sha256(source.read_bytes()).digest() == before, "input source changed")
        return

    if mode == "extractor_overwrite_guard":
        fixture = context.repository / source_argument(context, case)
        source = work / "guard.mdsl"
        shutil.copyfile(fixture, source)
        before = hashlib.sha256(source.read_bytes()).digest()
        completed = run(
            extraction_command(context, str(source), source),
            context.repository,
        )
        require(completed.returncode == 2, "extractor input overwrite was not rejected")
        require(str(case["diagnostic"]) in completed.stderr, "overwrite diagnostic missing")
        require(hashlib.sha256(source.read_bytes()).digest() == before, "input source changed")
        return

    if mode == "extractor_generated_alias_guard":
        fixture = context.repository / source_argument(context, case)
        source = work / "guard.mdsl"
        shutil.copyfile(fixture, source)
        before = hashlib.sha256(source.read_bytes()).digest()
        paths = generated_paths(work, "guard")
        paths["host"].hardlink_to(source)
        completed = run(
            extraction_command(context, str(source), paths["ir"], paths),
            context.repository,
        )
        require(completed.returncode == 2, "generated input alias was not rejected")
        require(str(case["diagnostic"]) in completed.stderr, "alias diagnostic missing")
        require(hashlib.sha256(source.read_bytes()).digest() == before, "input source changed")
        require(not paths["ir"].exists(), "rejected generation emitted IR")
        return

    if mode == "driver_depfile_artifact_guard":
        source = source_argument(context, case)
        output = work / "result.o"
        conflicting_depfile = work / "result.host.cpp"
        completed = run(
            [
                str(context.driver),
                "--matcore-target=cpu",
                "--save-temps",
                "-std=c++20",
                "-MD",
                "-MF",
                str(conflicting_depfile),
                "-c",
                source,
                "-o",
                str(output),
            ],
            context.repository,
        )
        require(completed.returncode == 2, "depfile/save-temps collision was not rejected")
        require(str(case["diagnostic"]) in completed.stderr, "depfile collision diagnostic missing")
        require(not output.exists(), "depfile collision emitted an object")
        require(not conflicting_depfile.exists(), "depfile collision overwrote a saved artifact")
        return

    if mode == "driver_dependency_mutation_guard":
        fixture = context.repository / source_argument(context, case)
        source = work / "guard.mdsl"
        output = work / "result.o"
        depfile = work / "result.d"
        generated = work / "result.host.cpp"
        scenario = str(case["scenario"])
        sentinel_bytes = b"#pragma once\n#define MDSLC_DEPENDENCY_SENTINEL 1\n"

        if scenario == "output":
            dependency = output
        elif scenario == "depfile":
            dependency = depfile
        elif scenario == "generated":
            dependency = generated
        elif scenario == "generated_hardlink":
            dependency = work / "included-dependency.h"
        else:
            raise ValidationFailure(f"unknown dependency mutation scenario: {scenario}")

        dependency.write_bytes(sentinel_bytes)
        if scenario == "generated_hardlink":
            generated.hardlink_to(dependency)
        source.write_text(
            f'#include "{dependency.name}"\n' + fixture.read_text(encoding="utf-8"),
            encoding="utf-8",
        )

        command = [
            str(context.driver),
            "--matcore-target=cpu",
            "-std=c++20",
            "-c",
            str(source),
            "-o",
            str(output),
        ]
        if scenario in {"generated", "generated_hardlink"}:
            command.insert(2, "--save-temps")
        if scenario == "depfile":
            command[2:2] = ["-MD", "-MF", str(depfile)]

        completed = run(command, work)
        require(completed.returncode != 0, "dependency mutation collision was accepted")
        require(
            str(case["diagnostic"]) in completed.stderr,
            f"dependency mutation diagnostic missing: {completed.stderr}",
        )
        require(
            dependency.read_bytes() == sentinel_bytes,
            f"{scenario} dependency sentinel was modified",
        )
        if scenario == "generated_hardlink":
            require(
                generated.read_bytes() == sentinel_bytes,
                "hard-link dependency sentinel was modified",
            )

        mutation_paths = [
            output,
            depfile,
            work / "result.host.cpp",
            work / "result.host-overlay.yaml",
            work / "result.matcore.json",
            work / "result.sites.h",
            work / "result.stubs.cpp",
            work / "result.backend.cpp",
            work / "result.host.o",
            work / "result.stubs.o",
            work / "result.backend.o",
        ]
        preserved = {dependency}
        if scenario == "generated_hardlink":
            preserved.add(generated)
        for path in mutation_paths:
            if path not in preserved:
                require(not path.exists(), f"rejection published mutation path: {path}")
        return

    if mode == "driver_shadow_header":
        source = source_argument(context, case)
        trusted_header = context.repository / "compiler/include/matcore/mdsl.h"
        fake_header = work / "fake-include/matcore/mdsl.h"
        fake_header.parent.mkdir(parents=True)
        header_text = trusted_header.read_text(encoding="utf-8")
        original_return = "  return out_arg{&value};"
        require(original_return in header_text, "trusted out() body changed unexpectedly")
        fake_header.write_text(
            header_text.replace(
                original_return,
                "  value.rows = 0;\n  return out_arg{&value};",
                1,
            ),
            encoding="utf-8",
        )
        output = work / "program"
        compiled = run(
            [
                str(context.driver),
                "--matcore-target=cpu",
                "-std=c++20",
                f"-I{fake_header.parent.parent}",
                source,
                "-o",
                str(output),
            ],
            context.repository,
        )
        completed_ok(compiled, "shadow-header CPU pipeline")
        executed = run([str(output)], work)
        completed_ok(executed, "shadow-header CPU executable")
        require(executed.stdout == str(case["stdout"]), "shadow header changed runtime semantics")
        return

    if mode == "driver_preprocessor_include":
        fixture = context.repository / source_argument(context, case)
        source = work / "preprocessor.mdsl"
        scenario = str(case["scenario"])
        fixture_text = fixture.read_text(encoding="utf-8")
        if scenario == "active":
            prefix = "#if 0\n#include <inactive-bootstrap-probe.h>\n#endif\n"
            source_text = prefix + fixture_text
        elif scenario == "inactive_duplicate":
            prefix = "#if 0\n#include <matcore/mdsl.h>\n#endif\n"
            source_text = prefix + fixture_text
        elif scenario == "macro_raw":
            direct_include = "#include <matcore/mdsl.h>"
            require(direct_include in fixture_text, "fixture lost direct public include")
            source_text = fixture_text.replace(
                direct_include,
                "#define MDSL_HEADER <matcore/mdsl.h>\n"
                "#include MDSL_HEADER\n"
                "[[maybe_unused]] const char *mdsl_include_probe = R\"MDSL(\n"
                "#include <matcore/mdsl.h>\n"
                ")MDSL\";",
                1,
            )
        else:
            raise ValidationFailure(f"unknown preprocessor scenario: {scenario}")
        source.write_text(source_text, encoding="utf-8")
        output = work / "program"
        completed = run(
            [
                str(context.driver),
                "--matcore-target=cpu",
                "-std=c++20",
                str(source),
                "-o",
                str(output),
            ],
            work,
        )
        completed_ok(completed, f"{scenario} preprocessor include pipeline")
        executed = run([str(output)], work)
        completed_ok(executed, f"{scenario} preprocessor include executable")
        require(executed.stdout == str(case["stdout"]), "preprocessor host behavior changed")
        return

    if mode == "driver_default_policy":
        fixture = context.repository / source_argument(context, case)
        source = work / "default-policy.mdsl"
        fixture_text = fixture.read_text(encoding="utf-8")
        explicit_call = """md::gemm(md::out(output), lhs, rhs,
           md::policy{.target = md::target::cpu,
                      .fallback = md::fallback::error});"""
        require(explicit_call in fixture_text, "GEMM fixture policy spelling changed")
        source.write_text(
            fixture_text.replace(
                explicit_call,
                "md::gemm(md::out(output), lhs, rhs);",
                1,
            ),
            encoding="utf-8",
        )
        output = work / "program"
        completed = run(
            [
                str(context.driver),
                "--matcore-target=cpu",
                "-std=c++20",
                str(source),
                "-o",
                str(output),
            ],
            work,
        )
        completed_ok(completed, "default-policy CPU pipeline")
        executed = run([str(output)], work)
        completed_ok(executed, "default-policy CPU executable")
        require(executed.stdout == str(case["stdout"]), "default policy changed execution")
        return

    if mode == "runtime_contract":
        completed = run([str(context.runtime_test)], work)
        completed_ok(completed, "runtime CPU contract")
        require(completed.stdout == str(case["stdout"]), "runtime test stdout mismatch")
        return

    if mode == "runtime_c_header":
        source = work / "runtime_abi_probe.c"
        source.write_text(
            "#include <matcore/runtime_c.h>\n"
            "int main(void) {\n"
            "  matcore_tensor_desc_v0 descriptor = {0};\n"
            "  return (int)descriptor.abi_version;\n"
            "}\n"
        )
        compiled = run(
            [
                str(context.clang_c),
                "-std=c11",
                "-Wall",
                "-Wextra",
                "-Wpedantic",
                "-Werror",
                f"-I{context.build_dir / 'include'}",
                "-fsyntax-only",
                str(source),
            ],
            work,
        )
        completed_ok(compiled, "strict C11 runtime header probe")
        return

    if mode == "runtime_shared_artifact":
        library = context.runtime_directory / "libmatcore_runtime.so.0.0.0"
        require(library.is_file(), "versioned runtime shared object is missing")
        identified = run(["file", str(library)], work)
        completed_ok(identified, "runtime file inspection")
        require("ELF 64-bit LSB shared object" in identified.stdout, "runtime is not an ELF64 shared object")
        dynamic = run(["readelf", "-d", str(library)], work)
        completed_ok(dynamic, "runtime dynamic-section inspection")
        require("Library soname: [libmatcore_runtime.so.0]" in dynamic.stdout, "runtime SONAME is wrong")
        symbols = run(["nm", "-D", "--defined-only", str(library)], work)
        completed_ok(symbols, "runtime dynamic-symbol inspection")
        exported = [line for line in symbols.stdout.splitlines() if line.strip()]
        exported_names = [line.split()[-1] for line in exported]
        require(
            exported_names
            == [
                "matcore_runtime_cpu_execution_context_create_v1",
                "matcore_runtime_cpu_execution_context_destroy_v1",
                "matcore_runtime_cpu_execution_context_query_v1",
                "matcore_runtime_gemm_bf16_f32_reference_v1",
                "matcore_runtime_gemm_f32_context_workspace_size_v2",
                "matcore_runtime_gemm_f32_execute_context_v2",
                "matcore_runtime_gemm_f32_execute_prepacked_b_v1",
                "matcore_runtime_gemm_f32_execute_v1",
                "matcore_runtime_gemm_f32_prepack_b_v1",
                "matcore_runtime_gemm_f32_prepacked_b_size_v1",
                "matcore_runtime_gemm_f32_v0",
                "matcore_runtime_gemm_f32_workspace_size_v1",
                "matcore_runtime_gemm_i8_i32_reference_v1",
                "matcore_runtime_plan_gemm_f32_v1",
                "matcore_runtime_query_cpu_capabilities_v2",
            ],
            f"unexpected runtime exports: {exported!r}",
        )
        return

    if mode == "generated_runtime_error":
        diagnostic = str(case["diagnostic"])
        source = work / f"{case['scenario']}.mdsl"
        source.write_text(runtime_error_source(str(case["scenario"]), diagnostic))
        output = work / "program"
        compiled = run(
            [str(context.driver), "--matcore-target=cpu", "-std=c++20", str(source), "-o", str(output)],
            work,
        )
        completed_ok(compiled, "generated runtime-error program compile")
        executed = run([str(output)], work)
        completed_ok(executed, "generated runtime-error program")
        require(diagnostic in executed.stdout, "runtime status diagnostic was lost")
        require(f"{source.name}:" in executed.stdout, "original .mdsl location was lost")
        return

    raise ValidationFailure(f"unsupported validation mode: {mode}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).with_name("validation_matrix.json"),
    )
    parser.add_argument("--case", action="append", dest="selected")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--require-no-future", action="store_true")
    arguments = parser.parse_args()

    repository = Path(__file__).resolve().parents[3]
    manifest = json.loads(arguments.manifest.read_text())
    if manifest.get("schema") != "matcore.validation-matrix" or manifest.get("version") != 2:
        print("invalid validation manifest schema/version", file=sys.stderr)
        return 2
    cases = manifest.get("cases")
    if not isinstance(cases, list):
        print("validation manifest cases must be an array", file=sys.stderr)
        return 2
    identifiers = [case.get("id") for case in cases]
    if len(identifiers) != len(set(identifiers)) or any(not identifier for identifier in identifiers):
        print("validation case IDs must be nonempty and unique", file=sys.stderr)
        return 2
    for case in cases:
        state = case.get("state")
        if state == "active" and not case.get("mode"):
            print(f"{case['id']}: active case lacks a mode", file=sys.stderr)
            return 2
        if state == "future" and not case.get("reason"):
            print(f"{case['id']}: future case lacks a reason", file=sys.stderr)
            return 2
        if state not in {"active", "future"}:
            print(f"{case['id']}: unknown state {state}", file=sys.stderr)
            return 2
        source = case.get("source")
        if source and not (repository / str(source)).is_file():
            print(f"{case['id']}: missing source {source}", file=sys.stderr)
            return 2

    if arguments.list:
        for case in cases:
            print(f"{case['state']:6} {case['id']}")
        return 0

    selected = set(arguments.selected or [])
    unknown = selected.difference(identifiers)
    if unknown:
        print(f"unknown selected case(s): {', '.join(sorted(unknown))}", file=sys.stderr)
        return 2

    context = Context(repository, arguments.build_dir.resolve())
    for tool in (
        context.driver,
        context.extractor,
        context.runtime_test,
        context.clang,
        context.clang_c,
    ):
        if not tool.is_file():
            print(f"missing validation tool: {tool}", file=sys.stderr)
            return 2

    active = [case for case in cases if case["state"] == "active" and (not selected or case["id"] in selected)]
    future = [case for case in cases if case["state"] == "future" and (not selected or case["id"] in selected)]
    failures = 0
    passes = 0
    with tempfile.TemporaryDirectory(prefix="matcore-integration-v0-") as temporary:
        root = Path(temporary)
        for index, case in enumerate(active):
            work = root / f"{index:02d}-{case['id'].lower()}"
            work.mkdir()
            try:
                execute_case(context, case, work)
            except (ValidationFailure, OSError, json.JSONDecodeError) as error:
                print(f"FAIL {case['id']}: {error}", file=sys.stderr)
                failures += 1
            else:
                print(f"PASS {case['id']}")
                passes += 1

    print(f"summary: pass={passes} fail={failures} future={len(future)}")
    if future:
        print("future capabilities (not counted as passes):")
        for case in future:
            print(f"  {case['id']}: {case['reason']}")
    if failures:
        return 1
    if arguments.require_no_future and future:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
