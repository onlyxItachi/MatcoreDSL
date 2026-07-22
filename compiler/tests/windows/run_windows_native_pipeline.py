#!/usr/bin/env python3

"""Windows-native frontend authentication and COFF/PE pipeline checks.

The existing integration matrix intentionally authenticates ELF partial links,
SONAME/rpath behavior and POSIX process details.  This test is the explicit
Windows replacement: it exercises native LibTooling without the bootstrap
frontend and, in pipeline mode, requires the driver to archive its normal COFF
objects into a static library that clang-cl links with the runtime import
library.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


def run(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    return completed


def require_success(
    completed: subprocess.CompletedProcess[str], label: str
) -> None:
    if completed.returncode != 0:
        raise RuntimeError(
            f"{label} failed ({completed.returncode})\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )


def extraction_command(
    *,
    extractor: Path,
    clang: Path,
    source: Path,
    ir: Path,
    include_root: Path,
    generated_root: Path | None = None,
) -> list[str]:
    command = [
        str(extractor),
        "--frontend=native",
        f"--clang={clang}",
        "--ir-version=1",
        "--input",
        str(source),
        "--ir-out",
        str(ir),
    ]
    if generated_root is not None:
        command.extend(
            [
                "--rewrite-out",
                str(generated_root / "gemm.host.cpp"),
                "--sites-out",
                str(generated_root / "gemm.sites.h"),
                "--stubs-out",
                str(generated_root / "gemm.stubs.cpp"),
                "--backend-out",
                str(generated_root / "gemm.backend.cpp"),
            ]
        )
    command.extend(
        [
            "--",
            str(clang),
            "/nologo",
            "/TP",
            "/std:c++20",
            "/EHsc",
            f"/I{include_root}",
            f"/I{source.parent}",
            str(source),
        ]
    )
    return command


def validate_native_ir(path: Path, source: Path, expected_operations: int = 1) -> None:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != "matcore.ir" or document.get("version") != 1:
        raise RuntimeError("native extraction did not emit typed Matcore IR v1")
    if document.get("producer") != "clang-libtooling-v1":
        raise RuntimeError("native extraction has the wrong producer")
    operations = document.get("operations")
    if not isinstance(operations, list) or len(operations) != expected_operations:
        raise RuntimeError(
            f"native extraction captured {len(operations) if isinstance(operations, list) else 'invalid'} "
            f"GEMM operations; expected {expected_operations}"
        )
    for operation in operations:
        if operation.get("canonical_callee") != "matcore::mdsl::gemm":
            raise RuntimeError("canonical Matcore declaration identity was lost")
        location = operation.get("source", {})
        if Path(str(location.get("file", ""))).name != source.name:
            raise RuntimeError("native source location does not identify the .mdsl input")
        if int(location.get("line", 0)) <= 0 or int(location.get("column", 0)) <= 0:
            raise RuntimeError("native source location has no line/column")


def frontend_suite(
    *,
    repository: Path,
    build_dir: Path,
    extractor: Path,
    clang: Path,
    temporary: Path,
) -> None:
    frontend_fixtures = repository / "compiler" / "tests" / "frontend"
    positive_cases = {
        "gemm_capture": 1,
        "direct_qualified": 1,
        "two_sites": 2,
    }
    for name, expected_operations in positive_cases.items():
        source = frontend_fixtures / f"{name}.mdsl"
        first = temporary / f"{name}.first.json"
        second = temporary / f"{name}.second.json"
        for iteration, ir in (("first", first), ("second", second)):
            completed = run(
                extraction_command(
                    extractor=extractor,
                    clang=clang,
                    source=source,
                    ir=ir,
                    include_root=build_dir / "include",
                ),
                cwd=repository,
            )
            require_success(completed, f"native clang-cl {name} extraction ({iteration})")
        validate_native_ir(first, source, expected_operations)
        document = json.loads(first.read_text(encoding="utf-8"))
        if len(document.get("operations", [])) != expected_operations:
            raise RuntimeError(
                f"{name} captured {len(document.get('operations', []))} operations; "
                f"expected {expected_operations}"
            )
        if first.read_bytes() != second.read_bytes():
            raise RuntimeError(f"{name} native Matcore IR is not deterministic")

    def reject(
        source: Path,
        expected_words: tuple[str, ...],
        label: str,
        diagnostic_file: str | None = None,
    ) -> None:
        rejected_ir = temporary / f"{label}.must-not-exist.json"
        rejected = run(
            extraction_command(
                extractor=extractor,
                clang=clang,
                source=source,
                ir=rejected_ir,
                include_root=build_dir / "include",
            ),
            cwd=repository,
        )
        if rejected.returncode == 0 or rejected_ir.exists():
            raise RuntimeError(f"Windows native negative case {label} was accepted")
        diagnostic = rejected.stderr.lower()
        if not any(word.lower() in diagnostic for word in expected_words):
            raise RuntimeError(
                f"{label} rejection lacks an actionable diagnostic matching "
                f"{expected_words}:\n{rejected.stderr}"
            )
        expected_file = (diagnostic_file or source.name).lower()
        if expected_file not in diagnostic or not re.search(
            rf"{re.escape(expected_file)}:\d+:\d+", diagnostic
        ):
            raise RuntimeError(
                f"{label} rejection lacks the original .mdsl line and column:\n"
                + rejected.stderr
            )

    frontend_negative_cases = {
        "untrusted_header": ("trusted <matcore/mdsl.h>",),
        "unqualified": ("qualified", "unqualified"),
        "indirect": ("indirect", "function-pointer"),
        "explicit_indirect": ("indirect", "function-pointer"),
        "template_call": ("template",),
        "lambda_call": ("lambda",),
        "macro_call": ("macro",),
        "header_origin": ("header", "input .mdsl"),
        "side_effect": ("stable matrix lvalue", "side effect"),
    }
    for name, words in frontend_negative_cases.items():
        reject(
            frontend_fixtures / f"{name}.mdsl",
            words,
            name,
            "header_call.h" if name == "header_origin" else None,
        )

    adversarial_fixtures = (
        repository / "compiler" / "tests" / "native_validation" / "fixtures" / "negative"
    )
    declaration_negative_cases = {
        "unannotated_overload": ("trusted", "declaration"),
        "untrusted_annotated_overload": ("trusted", "declaration"),
        "mutated_annotation": ("annotation", "payload", "unsupported"),
        "conflicting_annotations": ("conflict", "annotation", "ambiguous"),
        "annotated_wrong_signature_overload": ("trusted", "declaration"),
        "user_overload": ("trusted", "declaration"),
        "dependent_instantiation": ("template", "dependent"),
        "macro_callee": ("macro",),
    }
    for name, words in declaration_negative_cases.items():
        reject(adversarial_fixtures / f"{name}.mdsl", words, name)


def pipeline_suite(
    *,
    repository: Path,
    build_dir: Path,
    driver: Path,
    clang: Path,
    temporary: Path,
) -> None:
    source = temporary / "MDSLC GEMM ünicode.mdsl"
    shutil.copy2(repository / "compiler" / "examples" / "gemm_v0.mdsl", source)
    archive_path = temporary / "MDSLC GEMM ünicode.lib"
    compiled = run(
        [
            str(driver),
            "--frontend=native",
            "--matcore-target=cpu",
            "--save-temps",
            "/nologo",
            "/TP",
            "/std:c++20",
            "/EHsc",
            "/MD",
            "/c",
            str(source),
            "-o",
            str(archive_path),
        ],
        cwd=temporary,
    )
    require_success(compiled, "native .mdsl to COFF static-library pipeline")
    if not archive_path.is_file() or archive_path.stat().st_size == 0:
        raise RuntimeError("mdslc++ emitted no COFF static library")
    if list(temporary.glob("*.o")):
        raise RuntimeError("Windows driver emitted a Unix .o artifact")
    for suffix in (".host.cpp", ".matcore.json", ".sites.h", ".stubs.cpp", ".backend.cpp"):
        if not list(temporary.glob(f"*{suffix}")):
            raise RuntimeError(f"--save-temps omitted {suffix}")
    for suffix in (".host.obj", ".stubs.obj", ".backend.obj"):
        objects = list(temporary.glob(f"*{suffix}"))
        if len(objects) != 1 or objects[0].stat().st_size == 0:
            raise RuntimeError(f"--save-temps omitted constituent COFF {suffix}")

    runtime_import_library = build_dir / "lib" / "matcore_runtime.lib"
    if not runtime_import_library.is_file():
        raise RuntimeError("runtime import library is missing")
    executable = temporary / "MDSLC GEMM ünicode.exe"
    linked = run(
        [
            str(clang),
            "/nologo",
            "/EHsc",
            "/MD",
            str(archive_path),
            str(runtime_import_library),
            f"/Fe{executable}",
            "/link",
            "/subsystem:console",
        ],
        cwd=temporary,
    )
    require_success(linked, "ordinary clang-cl PE link")
    environment = os.environ.copy()
    environment["PATH"] = str(build_dir / "bin") + os.pathsep + environment.get(
        "PATH", ""
    )
    executed = run([str(executable)], cwd=temporary, environment=environment)
    require_success(executed, "generated PE GEMM execution")
    if "MDSLC CPU GEMM PASS" not in executed.stdout:
        raise RuntimeError(f"generated GEMM reported no success:\n{executed.stdout}")


def main() -> int:
    if os.name != "nt":
        raise RuntimeError("Windows native-pipeline validation requires native Windows")
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=("frontend", "pipeline"), required=True)
    parser.add_argument("--extractor", type=Path, required=True)
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--clang", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    arguments = parser.parse_args()

    repository = Path(__file__).resolve().parents[3]
    build_dir = arguments.build_dir.resolve()
    with tempfile.TemporaryDirectory(prefix="matcore windows native ünicode ") as root:
        temporary = Path(root)
        if arguments.suite == "frontend":
            frontend_suite(
                repository=repository,
                build_dir=build_dir,
                extractor=arguments.extractor.resolve(),
                clang=arguments.clang.resolve(),
                temporary=temporary,
            )
        else:
            pipeline_suite(
                repository=repository,
                build_dir=build_dir,
                driver=arguments.driver.resolve(),
                clang=arguments.clang.resolve(),
                temporary=temporary,
            )
    print(f"Windows native {arguments.suite} PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
