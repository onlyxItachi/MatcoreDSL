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
import time


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
    asan: bool,
) -> None:
    sanitizer_compile_options = ["/fsanitize=address", "/Oy-"] if asan else []
    sanitizer_link_options = ["/fsanitize=address"] if asan else []

    def reject_driver(
        label: str,
        arguments: list[str],
        output: Path,
        expected_words: tuple[str, ...],
        *,
        clear_output: bool = True,
    ) -> None:
        if clear_output:
            output.unlink(missing_ok=True)
        rejected = run([str(driver), *arguments], cwd=temporary)
        if rejected.returncode == 0:
            raise RuntimeError(f"Windows driver guard {label} was accepted")
        diagnostic = (rejected.stdout + rejected.stderr).lower()
        if not any(word.casefold() in diagnostic for word in expected_words):
            raise RuntimeError(
                f"Windows driver guard {label} lacks an actionable diagnostic "
                f"matching {expected_words}:\n{rejected.stderr}"
            )
        if clear_output and output.exists():
            raise RuntimeError(f"Windows driver guard {label} emitted {output}")

    source = temporary / "MDSLC GEMM ünicode.mdsl"
    shutil.copy2(repository / "compiler" / "examples" / "gemm_v0.mdsl", source)

    # Omitting --frontend must remain exactly the supported native path. The
    # compatibility frontend is never an implicit fallback.
    default_archive = temporary / "MDSLC default native.lib"
    default_compiled = run(
        [
            str(driver),
            "--matcore-target=cpu",
            "/nologo",
            "/TP",
            "/std:c++20",
            "/EHsc",
            "/MD",
            *sanitizer_compile_options,
            "/c",
            str(source),
            "-o",
            str(default_archive),
        ],
        cwd=temporary,
    )
    require_success(default_compiled, "default native Windows driver pipeline")
    if not default_archive.is_file() or default_archive.stat().st_size == 0:
        raise RuntimeError("default native Windows driver emitted no static library")

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
            *sanitizer_compile_options,
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
    generated_irs = list(temporary.glob("*.matcore.json"))
    if not generated_irs or not any(
        json.loads(path.read_text(encoding="utf-8")).get("producer")
        == "clang-libtooling-v1"
        for path in generated_irs
    ):
        raise RuntimeError("explicit native driver path emitted no native Matcore IR")

    # Metacharacters stay one argv element; no command shell is involved.
    metachar_source = temporary / "gemm&echo mdslc-injection.mdsl"
    shutil.copy2(source, metachar_source)
    metachar_output = temporary / "gemm&still-one-argv.lib"
    metachar_marker = temporary / "mdslc-injection.mdsl"
    metachar_compiled = run(
        [
            str(driver),
            "--matcore-target=cpu",
            "/TP",
            "/std:c++20",
            *sanitizer_compile_options,
            "/c",
            str(metachar_source),
            "-o",
            str(metachar_output),
        ],
        cwd=temporary,
    )
    require_success(metachar_compiled, "metacharacter single-argv compile")
    if not metachar_output.is_file() or metachar_marker.exists():
        raise RuntimeError("Windows process launch interpreted a metacharacter path")

    common = ["--matcore-target=cpu", "/std:c++20"]
    guard_cases = [
        (
            "shared-library-mode",
            [*common, "/LD", "/c", str(source), "-o", str(temporary / "ld.lib")],
            temporary / "ld.lib",
            ("/ld", "shared", "link mode"),
        ),
        (
            "link-tail-in-compile-mode",
            [
                *common,
                "/c",
                str(source),
                "-o",
                str(temporary / "link-tail.lib"),
                "/link",
                "/DEBUG",
            ],
            temporary / "link-tail.lib",
            ("/link", "link-only", "compile"),
        ),
        (
            "wrong-static-library-extension",
            [*common, "/c", str(source), "-o", str(temporary / "wrong.obj")],
            temporary / "wrong.obj",
            (".lib", "static library", "archive"),
        ),
        (
            "cuda-no-fallback",
            [
                "--matcore-target=cuda",
                "/c",
                str(source),
                "-o",
                str(temporary / "cuda.lib"),
            ],
            temporary / "cuda.lib",
            ("only cpu", "unsupported", "never falls back"),
        ),
        (
            "c-language-mode",
            [*common, "/TC", "/c", str(source), "-o", str(temporary / "c.lib")],
            temporary / "c.lib",
            ("/tc", "c++", "language"),
        ),
        (
            "unknown-frontend",
            [
                "--frontend=unknown",
                *common,
                "/c",
                str(source),
                "-o",
                str(temporary / "unknown.lib"),
            ],
            temporary / "unknown.lib",
            ("frontend", "native"),
        ),
        (
            "bootstrap-unavailable",
            [
                "--frontend=ast-json-bootstrap",
                *common,
                "/c",
                str(source),
                "-o",
                str(temporary / "bootstrap.lib"),
            ],
            temporary / "bootstrap.lib",
            ("not built", "unavailable", "compatibility"),
        ),
        (
            "production-test-prefix",
            [
                f"--tool-prefix-for-testing={temporary}",
                *common,
                "/c",
                str(source),
                "-o",
                str(temporary / "test-prefix.lib"),
            ],
            temporary / "test-prefix.lib",
            ("production", "unavailable"),
        ),
    ]
    response_file = temporary / "driver-bypass.rsp"
    response_output = temporary / "response.lib"
    response_file.write_text(
        f'/c "{source}" -o "{response_output}"\n', encoding="utf-8"
    )
    guard_cases.append(
        (
            "response-file-bypass",
            [*common, f"@{response_file}"],
            response_output,
            ("response", "incompatible", "@"),
        )
    )
    for label, arguments, output, expected_words in guard_cases:
        reject_driver(label, arguments, output, expected_words)

    reserved_device = run(
        [
            str(driver),
            *common,
            "/c",
            str(source),
            "-o",
            str(temporary / "NUL.lib"),
        ],
        cwd=temporary,
    )
    if reserved_device.returncode == 0 or not any(
        word in (reserved_device.stdout + reserved_device.stderr).lower()
        for word in ("reserved", "device", "output path")
    ):
        raise RuntimeError(
            "Windows reserved device output was not rejected actionably:\n"
            f"{reserved_device.stderr}"
        )

    overwrite_bytes = source.read_bytes()
    reject_driver(
        "input-output-overwrite",
        [*common, "/c", str(source), "-o", str(source)],
        source,
        ("overwrite", "input", "alias"),
        clear_output=False,
    )
    if source.read_bytes() != overwrite_bytes:
        raise RuntimeError("input/output overwrite guard modified its .mdsl source")

    # LINK-compatible option names are case-insensitive and accept both '/'
    # and '-' prefixes. Prove that mixed-case output and artifact-mode options
    # cannot bypass the driver's validated output path or mutate a sentinel.
    link_guard_sentinel = temporary / "link option sentinel.bin"
    sentinel_bytes = b"mdslc-link-option-sentinel\n"
    link_guard_sentinel.write_bytes(sentinel_bytes)
    link_guard_cases = [
        ("lowercase-link-out-source", f"/out:{source}"),
        ("dash-mixed-case-link-out", f"-OuT:{link_guard_sentinel}"),
        ("mixed-case-link-dll", "/dLl"),
        ("mixed-case-link-ltcg", "/LtCg"),
    ]
    for label, linker_argument in link_guard_cases:
        link_guard_output = temporary / f"{label}.exe"
        reject_driver(
            label,
            [
                *common,
                str(source),
                f"/Fe:{link_guard_output}",
                "/link",
                linker_argument,
            ],
            link_guard_output,
            ("output-changing", "unsupported", "link argument"),
        )
        if source.read_bytes() != overwrite_bytes:
            raise RuntimeError(f"Windows driver guard {label} modified its source")
        if link_guard_sentinel.read_bytes() != sentinel_bytes:
            raise RuntimeError(f"Windows driver guard {label} modified its sentinel")

    compiler_control_cases: list[tuple[str, list[str]]] = [
        (
            "opaque-clang-forwarding",
            [
                "/ClAnG:-serialize-diagnostics",
                f"/ClAnG:{link_guard_sentinel}",
            ],
        ),
        ("joined-xclang", ["-Xclang=-load"]),
        ("driver-mode-override", ["--driver-mode=g++"]),
        ("time-trace-output", [f"-ftime-trace={link_guard_sentinel}"]),
        ("split-dwarf-output", ["-gsplit-dwarf"]),
        ("joined-header-language", ["-xc++-header"]),
        ("unsupported-openmp-runtime", ["-openmp:experimental"]),
        ("pass-plugin", ["-fpass-plugin=untrusted-plugin.dll"]),
        ("module-map", ["-fmodule-map-file=untrusted.modulemap"]),
        ("config-directory", ["--config-user-dir=untrusted-config"]),
        ("precompiled-header", ["/Yuattacker.pch"]),
    ]
    for label, controls in compiler_control_cases:
        control_output = temporary / f"{label}.lib"
        reject_driver(
            label,
            [*common, "/c", str(source), "-o", str(control_output), *controls],
            control_output,
            ("output-producing", "unsafe", "opaque", "forbidden"),
        )
        if source.read_bytes() != overwrite_bytes:
            raise RuntimeError(f"Windows driver guard {label} modified its source")
        if link_guard_sentinel.read_bytes() != sentinel_bytes:
            raise RuntimeError(f"Windows driver guard {label} modified its sentinel")
        if list(temporary.glob("*.dwo")) or list(temporary.glob("*.pch")):
            raise RuntimeError(f"Windows driver guard {label} emitted a sidecar")

    # Prospective Windows destinations are case-insensitive even before an
    # output exists. A differently cased depfile must not alias the archive.
    dep_output = temporary / "Prospective-Dep.lib"
    for label, alias in (
        ("case-insensitive", temporary / "prospective-dep.LIB"),
        ("trailing-dot", temporary / "Prospective-Dep.lib."),
        ("trailing-space", temporary / "Prospective-Dep.lib "),
    ):
        rejected_dep_alias = run(
            [
                str(driver),
                *common,
                "/c",
                f"--matcore-depfile={alias}",
                str(source),
                "-o",
                str(dep_output),
            ],
            cwd=temporary,
        )
        if rejected_dep_alias.returncode == 0 or not any(
            word in (rejected_dep_alias.stdout + rejected_dep_alias.stderr).lower()
            for word in ("dependency", "alias", "distinct")
        ):
            raise RuntimeError(
                f"{label} prospective depfile/output alias was accepted:\n"
                f"{rejected_dep_alias.stderr}"
            )

    # Exercise the driver's authenticated snapshot checks using an observable
    # phase boundary, not a private test hook. A large comment makes the
    # generated compile window long enough for deterministic CI polling.
    mutation_dir = temporary / "mutation"
    mutation_dir.mkdir()
    mutation_source = mutation_dir / "mutation.mdsl"
    mutation_source.write_text(
        source.read_text(encoding="utf-8") + "\n/*" + ("x" * 1024 * 1024) + "*/\n",
        encoding="utf-8",
    )
    mutation_output = mutation_dir / "mutation.lib"
    process = subprocess.Popen(
        [
            str(driver),
            "--save-temps",
            *common,
            "/c",
            str(mutation_source),
            "-o",
            str(mutation_output),
        ],
        cwd=mutation_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    deadline = time.monotonic() + 30.0
    mutated = False
    while process.poll() is None and time.monotonic() < deadline:
        if list(mutation_dir.glob("*.host.cpp")):
            mutation_source.write_text(
                mutation_source.read_text(encoding="utf-8") + "// changed during build\n",
                encoding="utf-8",
            )
            mutated = True
            break
        time.sleep(0.001)
    stdout, stderr = process.communicate(timeout=60)
    if not mutated:
        raise RuntimeError("source-mutation test could not observe the rewrite phase")
    if process.returncode == 0 or mutation_output.exists():
        raise RuntimeError(
            "source mutation between extraction and compilation was not rejected:\n"
            f"stdout:\n{stdout}\nstderr:\n{stderr}"
        )
    if not any(word in stderr.lower() for word in ("changed", "snapshot", "dependency")):
        raise RuntimeError(f"source mutation rejection was not actionable:\n{stderr}")

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
            *sanitizer_link_options,
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

    if asan:
        driver_linked = temporary / "MDSLC driver linked ASan.exe"
        linked_by_driver = run(
            [
                str(driver),
                "--verbose",
                "--matcore-target=cpu",
                "/std:c++20",
                "/EHsc",
                "/MD",
                "/fsanitize=address",
                "/Oy-",
                str(source),
                "-o",
                str(driver_linked),
            ],
            cwd=temporary,
        )
        require_success(linked_by_driver, "driver-owned ASan PE final link")
        final_link_lines = [
            line
            for line in linked_by_driver.stderr.splitlines()
            if str(driver_linked) in line
        ]
        if not final_link_lines:
            raise RuntimeError("verbose driver output omitted the ASan final link")
        final_link = final_link_lines[-1]
        if "/fsanitize=address" not in final_link or " /link " in final_link:
            raise RuntimeError(
                "ASan was not kept in clang-cl driver scope during final link:\n"
                + final_link
            )
        driver_executed = run(
            [str(driver_linked)], cwd=temporary, environment=environment
        )
        require_success(driver_executed, "driver-owned ASan GEMM execution")
        if "MDSLC CPU GEMM PASS" not in driver_executed.stdout:
            raise RuntimeError("driver-owned ASan GEMM reported no success")


def main() -> int:
    if os.name != "nt":
        raise RuntimeError("Windows native-pipeline validation requires native Windows")
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=("frontend", "pipeline"), required=True)
    parser.add_argument("--extractor", type=Path, required=True)
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--clang", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument(
        "--asan",
        action="store_true",
        help="instrument generated objects and the final PE link with clang-cl ASan",
    )
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
                asan=arguments.asan,
            )
    print(f"Windows native {arguments.suite} PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
