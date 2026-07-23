#!/usr/bin/env python3
"""Focused clang-cl parsing and Windows driver fail-closed contract checks."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path


def run(
    command: list[str],
    cwd: Path,
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
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
            rejected.returncode != 0 and "output-producing" in rejected.stderr,
            "native frontend accepted clang-cl output mutation",
        )

        source_bytes = source.read_bytes()
        compiler_guard_sentinel = temporary / "compiler-control-sentinel.mdsl"
        compiler_guard_bytes = b"compiler-control-sentinel\n"
        compiler_guard_sentinel.write_bytes(compiler_guard_bytes)
        for unsafe_argument in (
            "/ClAnG:-serialize-diagnostics",
            "-Xclang=-load",
            "--driver-mode=g++",
            f"-ftime-trace={compiler_guard_sentinel}",
            "-gsplit-dwarf",
            "-xc++-header",
            "-openmp:experimental",
            "-fpass-plugin=untrusted-plugin.dll",
            "--config-user-dir=untrusted-config",
            "/Yuattacker.pch",
        ):
            unsafe = run(
                [
                    str(args.extractor),
                    "--input",
                    source.as_posix(),
                    "--ir-out",
                    str(temporary / "unsafe-control.json"),
                    "--",
                    *tool_arguments[:-1],
                    unsafe_argument,
                    source.as_posix(),
                ],
                repository,
            )
            require(
                unsafe.returncode != 0
                and any(
                    marker in unsafe.stderr
                    for marker in ("output-producing", "unsafe", "opaque")
                ),
                f"native frontend accepted hidden compiler control {unsafe_argument}",
            )
            require(
                source.read_bytes() == source_bytes
                and compiler_guard_sentinel.read_bytes() == compiler_guard_bytes,
                f"hidden compiler control modified a sentinel: {unsafe_argument}",
            )
            require(
                not list(temporary.glob("*.dwo"))
                and not list(temporary.glob("*.pch")),
                f"hidden compiler control emitted an undeclared sidecar: {unsafe_argument}",
            )

        nested_response = temporary / "nested-compiler-control.rsp"
        nested_response.write_text(
            f"SAFE=1 -ftime-trace={compiler_guard_sentinel}\n",
            encoding="utf-8",
        )
        nested_extractor = run(
            [
                str(args.extractor),
                "--input",
                source.as_posix(),
                "--ir-out",
                str(temporary / "nested-unsafe.json"),
                "--",
                *tool_arguments[:-1],
                "/D",
                f"@{nested_response}",
                source.as_posix(),
            ],
            repository,
        )
        require(
            nested_extractor.returncode != 0
            and "response-file" in nested_extractor.stderr,
            "native frontend accepted a nested response file as /D value",
        )

        poison_variables = (
            "CL",
            "_CL_",
            "LINK",
            "_LINK_",
            "CCC_OVERRIDE_OPTIONS",
            "CCC_ADD_ARGS",
            "CLANG_CONFIG_PATH",
            "CC_PRINT_OPTIONS",
            "CC_PRINT_OPTIONS_FILE",
            "LINK_REPRO",
            "LINK_REPRO_TARGET",
            "LINK_REPRO_FULLPATHRSP",
            "LLD_REPRODUCE",
        )
        clean_environment = os.environ.copy()
        for variable in poison_variables:
            clean_environment.pop(variable, None)
        for variable in poison_variables:
            poisoned_environment = clean_environment.copy()
            poisoned_environment[variable] = "untrusted-hidden-control"
            poisoned = run(
                [str(args.extractor), "--frontend-info"],
                repository,
                poisoned_environment,
            )
            require(
                poisoned.returncode != 0
                and variable.casefold() in poisoned.stderr.casefold()
                and "inherited compiler control" in poisoned.stderr,
                f"standalone extractor did not fail closed for {variable}",
            )

        if os.name == "nt":
            case_source = temporary / "Prospective-Case.MDSL"
            case_source.write_bytes(source_bytes)
            source_alias = temporary / "prospective-case.mdsl"
            rejected_source_alias = run(
                [
                    str(args.extractor),
                    "--input",
                    str(case_source),
                    "--ir-out",
                    str(source_alias),
                    "--",
                    *tool_arguments[:-1],
                    str(case_source),
                ],
                repository,
            )
            require(
                rejected_source_alias.returncode != 0
                and any(
                    word in rejected_source_alias.stderr.lower()
                    for word in ("overwrite", "alias", "input")
                ),
                "case-insensitive prospective extractor source alias was accepted",
            )
            require(
                case_source.read_bytes() == source_bytes,
                "prospective extractor source alias modified the source",
            )

            first_generated = temporary / "Generated-Case.json"
            second_generated = temporary / "generated-case.JSON"
            rejected_generated_alias = run(
                [
                    str(args.extractor),
                    "--input",
                    str(case_source),
                    "--ir-out",
                    str(first_generated),
                    "--rewrite-out",
                    str(second_generated),
                    "--sites-out",
                    str(temporary / "case.sites.h"),
                    "--stubs-out",
                    str(temporary / "case.stubs.cpp"),
                    "--backend-out",
                    str(temporary / "case.backend.cpp"),
                    "--",
                    *tool_arguments[:-1],
                    str(case_source),
                ],
                repository,
            )
            require(
                rejected_generated_alias.returncode != 0
                and "distinct output" in rejected_generated_alias.stderr,
                "case-insensitive prospective generated-output alias was accepted",
            )

        if os.name == "nt":
            require(args.driver is not None, "Windows driver path is required")
            host_source = temporary / "host only ü.mdsl"
            host_source.write_text("int main() { return 0; }\n", encoding="utf-8")

            # This direct invocation exercises the driver's own response-file
            # serializer with separated-value clang-cl options.  clang-cl
            # treats response-file newlines as option boundaries, so /D and
            # /I must remain paired with their following argv elements.
            response_include = temporary / "response include ü"
            response_include.mkdir()
            response_include.joinpath("response-header.h").write_text(
                "#define RESPONSE_HEADER_OK 1\n", encoding="utf-8"
            )
            response_source = temporary / "response contract ü.mdsl"
            response_source.write_text(
                '#include "response-header.h"\n'
                "#if !defined(RESPONSE_DEFINE_OK) || !RESPONSE_HEADER_OK || "
                "!defined(EMPTY_MACRO)\n"
                '#error "response argv boundary failure"\n'
                "#endif\nconst char* response_text = RESPONSE_QUOTED;\n"
                "int response_contract() { return response_text[0] == 'h' ? 0 : 1; }\n",
                encoding="utf-8",
            )
            response_object = temporary / "response contract ü.obj"
            response_compile = run(
                [
                    str(args.driver),
                    "/D",
                    "RESPONSE_DEFINE_OK=1",
                    "/D",
                    "EMPTY_MACRO=",
                    "/D",
                    'RESPONSE_QUOTED="hello world"',
                    "/I",
                    str(response_include) + "\\",
                    "/c",
                    str(response_source),
                    f"/Fo{response_object}",
                ],
                repository,
            )
            require(
                response_compile.returncode == 0 and response_object.is_file(),
                "driver response file did not preserve separated clang-cl "
                f"option operands:\n{response_compile.stderr}",
            )

            response_link_source = temporary / "response link contract.mdsl"
            response_link_source.write_text(
                "int main() { return 0; }\n", encoding="utf-8"
            )
            response_executable = temporary / "response link contract.exe"
            response_pdb = temporary / "response link symbols with space.pdb"
            response_link = run(
                [
                    str(args.driver),
                    str(response_link_source),
                    f"/Fe{response_executable}",
                    "/link",
                    "/DEBUG",
                    f"/PDB:{response_pdb}",
                ],
                temporary,
            )
            require(
                response_link.returncode == 0
                and response_executable.is_file()
                and response_pdb.is_file(),
                "driver response file did not preserve /link remainder "
                f"scope:\n{response_link.stderr}",
            )

            forced_include = temporary / "--save-temps"
            forced_include.write_text(
                "#define FORCED_INCLUDE_VALUE 7\n", encoding="utf-8"
            )
            forced_source = temporary / "forced include value.mdsl"
            forced_source.write_text(
                "int forced_include_value() { return FORCED_INCLUDE_VALUE; }\n",
                encoding="utf-8",
            )
            forced_object = temporary / "forced include value.obj"
            forced_compile = run(
                [
                    str(args.driver),
                    "/FI",
                    "--save-temps",
                    "/c",
                    str(forced_source),
                    f"/Fo{forced_object}",
                ],
                temporary,
            )
            require(
                forced_compile.returncode == 0 and forced_object.is_file(),
                "wrapper reinterpreted a separated /FI operand as an MDSLC "
                f"control:\n{forced_compile.stderr}",
            )

            nested_driver_output = temporary / "nested-driver.lib"
            nested_driver = run(
                [
                    str(args.driver),
                    "--matcore-target=cpu",
                    "/D",
                    f"@{nested_response}",
                    "/c",
                    str(host_source),
                    "-o",
                    str(nested_driver_output),
                ],
                repository,
            )
            require(
                nested_driver.returncode != 0
                and "response file" in nested_driver.stderr
                and not nested_driver_output.exists(),
                "driver accepted a nested response file as /D value",
            )

            guard_cases = [
                (["/TC", str(host_source), f"/Fe:{temporary / 'tc.exe'}"], "valid C++"),
                (["-TC", str(host_source), f"/Fe:{temporary / 'dash-tc.exe'}"], "valid C++"),
                ([f"/Tp{host_source}", f"/Fe:{temporary / 'tp.exe'}"], "valid C++"),
                (
                    [
                        "--matcore-target=cpu",
                        "/C",
                        str(host_source),
                        f"/Fe:{temporary / 'uppercase-c.exe'}",
                    ],
                    "unsafe compiler mode",
                ),
                (
                    [
                        "--matcore-target=cpu",
                        "/tp",
                        str(host_source),
                        f"/Fe:{temporary / 'lowercase-tp.exe'}",
                    ],
                    "unsafe compiler mode",
                ),
                (
                    [
                        "--matcore-target=cpu",
                        f"/fo{temporary / 'mixed-case-output.obj'}",
                        str(host_source),
                        f"/Fe:{temporary / 'mixed-case-output.exe'}",
                    ],
                    "undeclared output-producing",
                ),
                (["/winsysroot"], "requires a value"),
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
                    "response files are forbidden",
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
                (
                    [
                        "--matcore-target=cpu",
                        str(host_source),
                        f"/Fe:{temporary / 'guard.exe'}",
                        "/link",
                        f"-OuT:{host_source}",
                    ],
                    "/link arguments are not supported",
                ),
                (
                    [
                        "--matcore-target=cpu",
                        str(host_source),
                        f"/Fe:{temporary / 'guard-trace.exe'}",
                        "/link",
                        f"--time-trace={temporary / 'link-trace.json'}",
                    ],
                    "/link arguments are not supported",
                ),
                (
                    [
                        "--matcore-target=cpu",
                        str(host_source),
                        f"/Fe:{temporary / 'guard-lto.exe'}",
                        "/link",
                        f"/lto-obj-path:{temporary / 'link-lto.obj'}",
                    ],
                    "/link arguments are not supported",
                ),
                (
                    [
                        "--matcore-target=cpu",
                        str(host_source),
                        f"/Fe:{temporary / 'guard-dll.exe'}",
                        "/link",
                        "/dLl",
                    ],
                    "/link arguments are not supported",
                ),
                (
                    [
                        "--matcore-target=cpu",
                        str(host_source),
                        f"/Fe:{temporary / 'guard-ltcg.exe'}",
                        "/link",
                        "/LtCg",
                    ],
                    "/link arguments are not supported",
                ),
            ]
            host_bytes = host_source.read_bytes()
            for arguments, diagnostic in guard_cases:
                completed = run([str(args.driver), *arguments], repository)
                require(
                    completed.returncode != 0 and diagnostic in completed.stderr,
                    f"driver guard failed for {arguments}:\n{completed.stderr}",
                )
                require(
                    host_source.read_bytes() == host_bytes,
                    f"driver guard modified its source for {arguments}",
                )

            config_directory = temporary / "hostile clang config"
            config_directory.mkdir()
            config_directory.joinpath("clang-cl.cfg").write_text(
                f"/clang:-serialize-diagnostics\n/clang:{compiler_guard_sentinel}\n",
                encoding="utf-8",
            )
            child_environment_cases = {
                "CL": f"/clang:-serialize-diagnostics /clang:{compiler_guard_sentinel}",
                "_CL_": f"/clang:-serialize-diagnostics /clang:{compiler_guard_sentinel}",
                "LINK": f"/MAP:{compiler_guard_sentinel}",
                "_LINK_": f"/MAP:{compiler_guard_sentinel}",
                "CCC_OVERRIDE_OPTIONS": f"+ -serialize-diagnostics + {compiler_guard_sentinel}",
                "CCC_ADD_ARGS": f"-serialize-diagnostics {compiler_guard_sentinel}",
                "CLANG_CONFIG_PATH": str(config_directory),
                "CC_PRINT_OPTIONS": "1",
                "CC_PRINT_OPTIONS_FILE": str(compiler_guard_sentinel),
                "LINK_REPRO": str(temporary),
                "LINK_REPRO_TARGET": "link.exe",
                "LINK_REPRO_FULLPATHRSP": "1",
                "LLD_REPRODUCE": str(compiler_guard_sentinel),
            }
            for variable, value in child_environment_cases.items():
                environment = clean_environment.copy()
                environment[variable] = value
                executable = temporary / f"sanitized-{variable}.exe"
                completed = run(
                    [str(args.driver), str(host_source), f"/Fe:{executable}"],
                    repository,
                    environment,
                )
                require(
                    completed.returncode == 0 and executable.is_file(),
                    f"driver did not sanitize inherited {variable}:\n{completed.stderr}",
                )
                require(
                    host_source.read_bytes() == host_bytes
                    and compiler_guard_sentinel.read_bytes() == compiler_guard_bytes,
                    f"driver child inherited {variable} and modified a sentinel",
                )

            paired_environment = clean_environment.copy()
            paired_environment["CC_PRINT_OPTIONS"] = "1"
            paired_environment["CC_PRINT_OPTIONS_FILE"] = str(
                compiler_guard_sentinel
            )
            paired_output = temporary / "sanitized-paired-env.exe"
            paired = run(
                [str(args.driver), str(host_source), f"/Fe:{paired_output}"],
                repository,
                paired_environment,
            )
            require(
                paired.returncode == 0
                and paired_output.is_file()
                and compiler_guard_sentinel.read_bytes() == compiler_guard_bytes,
                "driver did not sanitize paired CC_PRINT_OPTIONS controls",
            )

    print("Windows clang-cl frontend/driver contract PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
