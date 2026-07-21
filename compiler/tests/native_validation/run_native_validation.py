#!/usr/bin/env python3
"""Differential and adversarial validation for the native Clang frontend."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


NATIVE_PRODUCER = "clang-libtooling-v1"
BOOTSTRAP_PRODUCER = "clang-ast-json-bootstrap-v0"
REPOSITORY = Path(__file__).resolve().parents[3]
FIXTURES = Path(__file__).resolve().parent / "fixtures"


@dataclass(frozen=True)
class Extraction:
    completed: subprocess.CompletedProcess[str]
    ir: Path
    host: Path | None = None
    sites: Path | None = None
    stubs: Path | None = None
    backend: Path | None = None


class Checks:
    def __init__(self) -> None:
        self.count = 0
        self.failures: list[str] = []

    def require(self, condition: bool, message: str) -> None:
        self.count += 1
        if not condition:
            self.failures.append(message)

    def case(self, name: str, operation) -> None:  # type: ignore[no-untyped-def]
        try:
            operation()
        except Exception as error:  # keep independent adversarial cases running
            self.failures.append(f"{name}: unexpected exception: {error}")


def run(
    command: Sequence[str], *, cwd: Path = REPOSITORY, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )


def source_argument(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPOSITORY).as_posix()
    except ValueError:
        return str(path.resolve())


def extraction(
    extractor: Path,
    source: Path,
    output_root: Path,
    label: str,
    *,
    frontend: str | None,
    clang: Path,
    ir_version: int | None = None,
    generate: bool = False,
    extra_compile_arguments: Iterable[str] = (),
) -> Extraction:
    output_root.mkdir(parents=True, exist_ok=True)
    ir = output_root / f"{label}.matcore.json"
    command = [str(extractor)]
    if frontend is not None:
        command.append(f"--frontend={frontend}")
    if ir_version is not None:
        command.append(f"--ir-version={ir_version}")
    command.extend(["--input", source_argument(source), "--ir-out", str(ir)])
    host = sites = stubs = backend = None
    if generate:
        host = output_root / f"{label}.host.cpp"
        sites = output_root / f"{label}.sites.h"
        stubs = output_root / f"{label}.stubs.cpp"
        backend = output_root / f"{label}.backend.cpp"
        command.extend(
            [
                "--rewrite-out",
                str(host),
                "--sites-out",
                str(sites),
                "--stubs-out",
                str(stubs),
                "--backend-out",
                str(backend),
            ]
        )
    encoded_source = source_argument(source)
    command.extend(
        [
            "--",
            str(clang),
            "-x",
            "c++",
            "-std=c++20",
            *extra_compile_arguments,
            encoded_source,
        ]
    )
    return Extraction(
        run(command), ir, host=host, sites=sites, stubs=stubs, backend=backend
    )


def load_ir(result: Extraction) -> dict:  # type: ignore[type-arg]
    return json.loads(result.ir.read_text(encoding="utf-8"))


def semantic_ir(document: dict) -> dict:  # type: ignore[type-arg]
    # Producer is the sole intentional frontend-specific field in IR v0/v1.
    normalized = json.loads(json.dumps(document))
    normalized.pop("producer", None)
    return normalized


def validate_ir_ranges(
    checks: Checks,
    label: str,
    document: dict,
    source: Path,
    expected_operations: int,
    expected_version: int = 0,
) -> None:  # type: ignore[type-arg]
    source_bytes = source.read_bytes()
    checks.require(document.get("schema") == "matcore.ir", f"{label}: wrong schema")
    checks.require(
        document.get("version") == expected_version,
        f"{label}: wrong IR version",
    )
    operations = document.get("operations", [])
    checks.require(
        len(operations) == expected_operations,
        f"{label}: expected {expected_operations} operations, got {len(operations)}",
    )
    prior_offset = -1
    site_ids: set[str] = set()
    for index, operation in enumerate(operations):
        prefix = f"{label} operation {index}"
        checks.require(operation.get("kind") == "gemm", f"{prefix}: wrong kind")
        checks.require(
            operation.get("canonical_callee") == "matcore::mdsl::gemm",
            f"{prefix}: canonical declaration identity was lost",
        )
        site_id = operation.get("site_id", "")
        checks.require(site_id not in site_ids, f"{prefix}: duplicate site ID")
        site_ids.add(site_id)
        source_record = operation.get("source", {})
        byte_range = source_record.get("byte_range", {})
        begin = byte_range.get("begin", -1)
        end = byte_range.get("end", -1)
        checks.require(
            isinstance(begin, int)
            and isinstance(end, int)
            and 0 <= begin < end <= len(source_bytes),
            f"{prefix}: invalid SourceManager byte range {begin}:{end}",
        )
        if not isinstance(begin, int) or not isinstance(end, int) or begin < 0:
            continue
        call_bytes = source_bytes[begin:end]
        checks.require(b"gemm" in call_bytes, f"{prefix}: call range misses callee")
        checks.require(begin > prior_offset, f"{prefix}: operations not source ordered")
        prior_offset = begin
        checks.require(
            source_record.get("byte_offset") == begin,
            f"{prefix}: location offset differs from range begin",
        )
        expected_line = source_bytes[:begin].count(b"\n") + 1
        last_newline = source_bytes.rfind(b"\n", 0, begin)
        expected_column = begin - last_newline
        checks.require(
            source_record.get("line") == expected_line,
            f"{prefix}: line is not backed by source bytes",
        )
        checks.require(
            source_record.get("column") == expected_column,
            f"{prefix}: column is not backed by source bytes",
        )
        argument_ranges = operation.get("source_argument_ranges", [])
        checks.require(
            len(argument_ranges) in (3, 4),
            f"{prefix}: expected three source arguments plus optional explicit policy",
        )
        prior_argument_end = begin
        for argument_range in argument_ranges:
            argument_begin = argument_range.get("begin", -1)
            argument_end = argument_range.get("end", -1)
            checks.require(
                begin <= argument_begin < argument_end <= end,
                f"{prefix}: argument range escapes validated CallExpr",
            )
            checks.require(
                argument_begin >= prior_argument_end,
                f"{prefix}: argument ranges overlap or are unordered",
            )
            prior_argument_end = argument_end


def compile_generated(
    checks: Checks,
    label: str,
    result: Extraction,
    clang: Path,
    extractor: Path,
    extra_compile_arguments: Iterable[str] = (),
) -> None:
    include = extractor.resolve().parent.parent / "include"
    assert result.host and result.sites and result.stubs and result.backend
    for kind, path in (
        ("host", result.host),
        ("stubs", result.stubs),
        ("backend", result.backend),
    ):
        completed = run(
            [
                str(clang),
                "-std=c++20",
                *extra_compile_arguments,
                "-Wall",
                "-Wextra",
                "-Wpedantic",
                "-Werror",
                f"-I{include}",
                f"-I{path.parent}",
                "-x",
                "c++",
                "-fsyntax-only",
                str(path),
            ]
        )
        checks.require(
            completed.returncode == 0,
            f"{label}: generated {kind} failed strict syntax check:\n{completed.stderr}",
        )


def positive_parity_case(
    checks: Checks,
    extractor: Path,
    clang: Path,
    temporary: Path,
    name: str,
    source: Path,
    expected_operations: int,
    extra_compile_arguments: Iterable[str] = (),
) -> None:
    native = extraction(
        extractor,
        source,
        temporary / name / "native",
        name,
        frontend="native",
        clang=clang,
        generate=True,
        extra_compile_arguments=extra_compile_arguments,
    )
    bootstrap = extraction(
        extractor,
        source,
        temporary / name / "bootstrap",
        name,
        frontend="ast-json-bootstrap",
        clang=clang,
        generate=True,
        extra_compile_arguments=extra_compile_arguments,
    )
    checks.require(
        native.completed.returncode == 0,
        f"{name}: native extraction failed:\n{native.completed.stderr}",
    )
    checks.require(
        bootstrap.completed.returncode == 0,
        f"{name}: bootstrap extraction failed:\n{bootstrap.completed.stderr}",
    )
    if native.completed.returncode or bootstrap.completed.returncode:
        return
    native_document = load_ir(native)
    bootstrap_document = load_ir(bootstrap)
    checks.require(
        native_document.get("producer") == NATIVE_PRODUCER,
        f"{name}: native producer is not explicit",
    )
    checks.require(
        bootstrap_document.get("producer") == BOOTSTRAP_PRODUCER,
        f"{name}: bootstrap producer is not explicit",
    )
    checks.require(
        semantic_ir(native_document) == semantic_ir(bootstrap_document),
        f"{name}: native/bootstrap semantic IR mismatch",
    )
    validate_ir_ranges(
        checks, f"{name}/native", native_document, source, expected_operations
    )
    for kind in ("host", "sites", "stubs", "backend"):
        native_path = getattr(native, kind)
        bootstrap_path = getattr(bootstrap, kind)
        assert native_path and bootstrap_path
        checks.require(
            native_path.read_bytes() == bootstrap_path.read_bytes(),
            f"{name}: generated {kind} differs between frontends",
        )
    compile_generated(checks, name, native, clang, extractor, extra_compile_arguments)


def typed_ir_parity_case(
    checks: Checks,
    extractor: Path,
    clang: Path,
    temporary: Path,
    name: str,
    source: Path,
    expected_operations: int,
    extra_compile_arguments: Iterable[str] = (),
) -> None:
    native = extraction(
        extractor,
        source,
        temporary / f"{name}-v1" / "native",
        name,
        frontend="native",
        clang=clang,
        ir_version=1,
        extra_compile_arguments=extra_compile_arguments,
    )
    bootstrap = extraction(
        extractor,
        source,
        temporary / f"{name}-v1" / "bootstrap",
        name,
        frontend="ast-json-bootstrap",
        clang=clang,
        ir_version=1,
        extra_compile_arguments=extra_compile_arguments,
    )
    checks.require(
        native.completed.returncode == 0,
        f"{name}/v1: native extraction failed:\n{native.completed.stderr}",
    )
    checks.require(
        bootstrap.completed.returncode == 0,
        f"{name}/v1: bootstrap extraction failed:\n{bootstrap.completed.stderr}",
    )
    if native.completed.returncode or bootstrap.completed.returncode:
        return
    native_document = load_ir(native)
    bootstrap_document = load_ir(bootstrap)
    checks.require(
        semantic_ir(native_document) == semantic_ir(bootstrap_document),
        f"{name}/v1: native/bootstrap typed semantic IR mismatch",
    )
    validate_ir_ranges(
        checks,
        f"{name}/native-v1",
        native_document,
        source,
        expected_operations,
        expected_version=1,
    )


def native_range_case(
    checks: Checks,
    extractor: Path,
    clang: Path,
    temporary: Path,
    name: str,
    source: Path,
    expected_operations: int,
) -> None:
    first = extraction(
        extractor,
        source,
        temporary / name / "first",
        name,
        frontend="native",
        clang=clang,
        generate=True,
    )
    second = extraction(
        extractor,
        source,
        temporary / name / "second",
        name,
        frontend="native",
        clang=clang,
        generate=True,
    )
    checks.require(
        first.completed.returncode == 0,
        f"{name}: native extraction failed:\n{first.completed.stderr}",
    )
    checks.require(
        second.completed.returncode == 0,
        f"{name}: repeated native extraction failed:\n{second.completed.stderr}",
    )
    if first.completed.returncode or second.completed.returncode:
        return
    first_document = load_ir(first)
    validate_ir_ranges(checks, name, first_document, source, expected_operations)
    checks.require(
        first.ir.read_bytes() == second.ir.read_bytes(),
        f"{name}: native IR is not byte deterministic",
    )
    for kind in ("host", "sites", "stubs", "backend"):
        first_path = getattr(first, kind)
        second_path = getattr(second, kind)
        assert first_path and second_path
        checks.require(
            first_path.read_bytes() == second_path.read_bytes(),
            f"{name}: native generated {kind} is not deterministic",
        )
    compile_generated(checks, name, first, clang, extractor)


def negative_case(
    checks: Checks,
    extractor: Path,
    clang: Path,
    temporary: Path,
    name: str,
    source: Path,
    expected_words: Sequence[str],
    location_extension: str = "mdsl",
    frontend: str = "native",
) -> None:
    result = extraction(
        extractor,
        source,
        temporary / "negative" / name,
        name,
        frontend=frontend,
        clang=clang,
    )
    diagnostic = result.completed.stderr.lower()
    diagnostic_messages = "\n".join(
        line.split(": error:", 1)[1]
        for line in diagnostic.splitlines()
        if ": error:" in line
    )
    checks.require(result.completed.returncode != 0, f"{name}: was accepted")
    checks.require(not result.ir.exists(), f"{name}: emitted IR after rejection")
    checks.require("error" in diagnostic, f"{name}: diagnostic was not actionable")
    checks.require(
        any(word.lower() in diagnostic_messages for word in expected_words),
        f"{name}: diagnostic lacked one of {expected_words!r}:\n{result.completed.stderr}",
    )
    checks.require(
        re.search(
            rf"\.{re.escape(location_extension)}:\d+:\d+", result.completed.stderr
        )
        is not None,
        f"{name}: diagnostic lacks an actionable source line and column:\n"
        f"{result.completed.stderr}",
    )


def compiler_argument_rejection_case(
    checks: Checks,
    extractor: Path,
    clang: Path,
    temporary: Path,
    name: str,
    arguments: Sequence[str],
) -> None:
    result = extraction(
        extractor,
        FIXTURES / "positive/namespace_alias.mdsl",
        temporary / "compiler-arguments" / name,
        name,
        frontend="native",
        clang=clang,
        extra_compile_arguments=arguments,
    )
    checks.require(
        result.completed.returncode != 0,
        f"{name}: invalid compiler arguments were accepted",
    )
    checks.require(
        not result.ir.exists(), f"{name}: invalid compiler arguments emitted IR"
    )
    checks.require(
        "error" in result.completed.stderr.lower(),
        f"{name}: invalid compiler argument diagnostic was not actionable:\n"
        f"{result.completed.stderr}",
    )


def trusted_header_mutation_case(
    checks: Checks,
    extractor: Path,
    clang: Path,
    temporary: Path,
    name: str,
    old: str,
    new: str,
    expected_words: Sequence[str],
) -> None:
    prefix = temporary / "trusted-header-mutations" / name
    copied_extractor = prefix / "bin/matcore-extract"
    copied_header = prefix / "include/matcore/mdsl.h"
    copied_extractor.parent.mkdir(parents=True)
    copied_header.parent.mkdir(parents=True)
    shutil.copy2(extractor, copied_extractor)
    original_header_path = extractor.resolve().parent.parent / "include/matcore/mdsl.h"
    original_header = original_header_path.read_text(encoding="utf-8")
    checks.require(old in original_header, f"{name}: header mutation anchor is absent")
    mutated_header = original_header.replace(old, new, 1)
    checks.require(
        mutated_header != original_header,
        f"{name}: trusted-header mutation did not alter the header",
    )
    copied_header.write_text(mutated_header, encoding="utf-8")
    source = FIXTURES / "positive/namespace_alias.mdsl"
    result = extraction(
        copied_extractor,
        source,
        prefix / "output",
        name,
        frontend="native",
        clang=clang,
    )
    diagnostic = result.completed.stderr.lower()
    diagnostic_messages = "\n".join(
        line.split(": error:", 1)[1]
        for line in diagnostic.splitlines()
        if ": error:" in line
    )
    checks.require(result.completed.returncode != 0, f"{name}: was accepted")
    checks.require(not result.ir.exists(), f"{name}: emitted IR after rejection")
    checks.require(
        any(word.lower() in diagnostic_messages for word in expected_words),
        f"{name}: wrong trusted-declaration diagnostic:\n{result.completed.stderr}",
    )
    checks.require(
        re.search(r"\.mdsl:\d+:\d+", result.completed.stderr) is not None,
        f"{name}: diagnostic lacks original .mdsl line and column:\n"
        f"{result.completed.stderr}",
    )


def core_suite(checks: Checks, extractor: Path, clang: Path) -> None:
    positives = FIXTURES / "positive"
    old = REPOSITORY / "compiler/tests/frontend"
    with tempfile.TemporaryDirectory(prefix="matcore-native-validation-") as encoded:
        temporary = Path(encoded)
        parity_cases = (
            ("old_host_only", old / "host_only.mdsl", 0, ()),
            ("old_direct", old / "direct_qualified.mdsl", 1, ()),
            ("old_alias", old / "gemm_capture.mdsl", 1, ()),
            ("old_two_sites", old / "two_sites.mdsl", 2, ()),
            ("namespace_alias", positives / "namespace_alias.mdsl", 1, ()),
            ("class_two_calls", positives / "class_two_calls_one_line.mdsl", 2, ()),
            (
                "flag_forwarding",
                positives / "flag_forwarding.mdsl",
                1,
                ("-DMDSLC_NATIVE_VALIDATION_FLAG=17",),
            ),
        )

        producer_only_left = {
            "producer": NATIVE_PRODUCER,
            "operations": [{"target": "cpu"}],
        }
        producer_only_right = {
            "producer": BOOTSTRAP_PRODUCER,
            "operations": [{"target": "cpu"}],
        }
        semantic_mismatch = {
            "producer": BOOTSTRAP_PRODUCER,
            "operations": [{"target": "cuda"}],
        }
        checks.require(
            semantic_ir(producer_only_left) == semantic_ir(producer_only_right),
            "parity normalizer did not ignore the intentional producer difference",
        )
        checks.require(
            semantic_ir(producer_only_left) != semantic_ir(semantic_mismatch),
            "parity harness failed to detect a semantic field mismatch",
        )
        for name, source, operation_count, extra in parity_cases:
            checks.case(
                name,
                lambda name=name, source=source, operation_count=operation_count, extra=extra: (
                    positive_parity_case(
                        checks,
                        extractor,
                        clang,
                        temporary,
                        name,
                        source,
                        operation_count,
                        extra,
                    )
                ),
            )
            checks.case(
                f"{name}_typed_v1",
                lambda name=name, source=source, operation_count=operation_count, extra=extra: (
                    typed_ir_parity_case(
                        checks,
                        extractor,
                        clang,
                        temporary,
                        name,
                        source,
                        operation_count,
                        extra,
                    )
                ),
            )

        range_source = positives / "range_edges.mdsl"
        checks.case(
            "range_edges",
            lambda: native_range_case(
                checks, extractor, clang, temporary, "range_edges", range_source, 1
            ),
        )
        crlf = temporary / "materialized" / "range_edges_crlf.mdsl"
        crlf.parent.mkdir(parents=True)
        crlf.write_bytes(range_source.read_bytes().replace(b"\n", b"\r\n"))
        checks.case(
            "range_edges_crlf",
            lambda: native_range_case(
                checks, extractor, clang, temporary, "range_edges_crlf", crlf, 1
            ),
        )
        no_final_newline = temporary / "materialized" / "range_edges_no_final.mdsl"
        no_final_newline.write_bytes(range_source.read_bytes().rstrip(b"\n"))
        checks.case(
            "range_edges_no_final",
            lambda: native_range_case(
                checks,
                extractor,
                clang,
                temporary,
                "range_edges_no_final",
                no_final_newline,
                1,
            ),
        )

        default_result = extraction(
            extractor,
            positives / "namespace_alias.mdsl",
            temporary / "default",
            "default",
            frontend=None,
            clang=clang,
        )
        checks.require(
            default_result.completed.returncode == 0,
            f"default native frontend failed:\n{default_result.completed.stderr}",
        )
        if default_result.completed.returncode == 0:
            checks.require(
                load_ir(default_result).get("producer") == NATIVE_PRODUCER,
                "default extractor invocation did not select native frontend",
            )

        shadow_result = extraction(
            extractor,
            positives / "namespace_alias.mdsl",
            temporary / "shadow",
            "shadow",
            frontend="native",
            clang=clang,
            extra_compile_arguments=(f"-I{FIXTURES / 'shadow'}",),
        )
        checks.require(
            shadow_result.completed.returncode == 0,
            "user include directory shadowed the tool-owned header:\n"
            f"{shadow_result.completed.stderr}",
        )
        if shadow_result.completed.returncode == 0:
            checks.require(
                load_ir(shadow_result).get("producer") == NATIVE_PRODUCER,
                "shadow attempt changed frontend producer",
            )

        negative = FIXTURES / "negative"
        negative_cases = (
            ("no_direct_include", ("direct", "include", "trusted"), "mdsl"),
            ("copied_header", ("trusted", "header"), "mdsl"),
            ("quoted_shadow", ("trusted", "header"), "mdsl"),
            ("unannotated_overload", ("trusted", "declaration"), "mdsl"),
            ("untrusted_annotated_overload", ("trusted", "declaration"), "mdsl"),
            ("mutated_annotation", ("annotation", "payload", "unsupported"), "mdsl"),
            (
                "conflicting_annotations",
                ("conflict", "annotation", "ambiguous"),
                "mdsl",
            ),
            (
                "annotated_wrong_signature_overload",
                ("trusted", "declaration"),
                "mdsl",
            ),
            ("user_overload", ("trusted", "declaration"), "mdsl"),
            ("unqualified", ("qualified", "unqualified"), "mdsl"),
            ("indirect", ("indirect", "function-pointer"), "mdsl"),
            ("template_body", ("template",), "mdsl"),
            ("dependent_instantiation", ("template", "dependent"), "mdsl"),
            ("lambda", ("lambda",), "mdsl"),
            ("macro", ("macro",), "mdsl"),
            ("macro_callee", ("macro",), "mdsl"),
            ("abi_macro", ("abi", "macro"), "mdsl"),
            ("unevaluated_noexcept", ("unevaluated",), "mdsl"),
            ("unevaluated_builtin", ("unevaluated",), "mdsl"),
            ("unevaluated_requires", ("unevaluated",), "mdsl"),
            ("unevaluated_typeid", ("unevaluated",), "mdsl"),
            ("unevaluated_decltype", ("unevaluated",), "mdsl"),
            ("policy_side_effect", ("policy", "side effect"), "mdsl"),
            ("policy_field_confusion", ("policy",), "mdsl"),
            ("qualified_reexport", ("qualified", "re-export"), "mdsl"),
            (
                "qualified_using_directive",
                ("qualified", "namespace alias"),
                "mdsl",
            ),
            ("header_origin", ("header", "main source", "input .mdsl"), "h"),
        )
        for name, words, location_extension in negative_cases:
            checks.case(
                f"negative/{name}",
                lambda name=name, words=words, location_extension=location_extension: (
                    negative_case(
                        checks,
                        extractor,
                        clang,
                        temporary,
                        name,
                        negative / f"{name}.mdsl",
                        words,
                        location_extension,
                    )
                ),
            )

        checks.case(
            "negative/policy_side_effect_bootstrap",
            lambda: negative_case(
                checks,
                extractor,
                clang,
                temporary,
                "policy_side_effect_bootstrap",
                negative / "policy_side_effect.mdsl",
                ("policy", "side effect"),
                frontend="ast-json-bootstrap",
            ),
        )
        for name, words in (
            ("policy_field_confusion", ("policy",)),
            ("qualified_reexport", ("qualified", "re-export")),
        ):
            checks.case(
                f"negative/{name}_bootstrap",
                lambda name=name, words=words: negative_case(
                    checks,
                    extractor,
                    clang,
                    temporary,
                    f"{name}_bootstrap",
                    negative / f"{name}.mdsl",
                    words,
                    frontend="ast-json-bootstrap",
                ),
            )

        extra_source = temporary / "compiler-arguments/extra-source.cpp"
        extra_source.parent.mkdir(parents=True, exist_ok=True)
        extra_source.write_text("int unrelated_translation_unit;\n", encoding="utf-8")
        trusted_header = extractor.resolve().parent.parent / "include/matcore/mdsl.h"
        replacement_header = negative / "copied/matcore/mdsl.h"
        overlay = temporary / "compiler-arguments/header-replacement.yaml"
        overlay.write_text(
            json.dumps(
                {
                    "version": 0,
                    "case-sensitive": True,
                    "use-external-names": False,
                    "roots": [
                        {
                            "type": "file",
                            "name": str(trusted_header),
                            "use-external-name": False,
                            "external-contents": str(replacement_header),
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        invalid_compiler_arguments = (
            ("unknown_clang_option", ("-fdefinitely-not-a-clang-option",)),
            ("invalid_language_standard", ("-std=c++99",)),
            ("missing_clang_config", ("--config=/definitely/missing.cfg",)),
            ("bare_clang_plugin", ("-fplugin", "/definitely/missing.so")),
            ("response_file", ("@/definitely/missing.rsp",)),
            ("header_vfs_overlay", ("-ivfsoverlay", str(overlay))),
            ("joined_header_vfs_overlay", (f"-ivfsoverlay{overlay}",)),
            ("cc1_header_vfs_overlay", (f"-vfsoverlay{overlay}",)),
            ("precompiled_header", ("-include-pch", "/definitely/missing.pch")),
            ("module_file", ("-fmodule-file=/definitely/missing.pcm",)),
            (
                "prebuilt_module_path",
                ("-fprebuilt-module-path=/definitely/missing-modules",),
            ),
            ("extra_source", (str(extra_source),)),
        )
        for name, arguments in invalid_compiler_arguments:
            checks.case(
                f"compiler-arguments/{name}",
                lambda name=name, arguments=arguments: (
                    compiler_argument_rejection_case(
                        checks, extractor, clang, temporary, name, arguments
                    )
                ),
            )

        trusted_annotation = 'MATCORE_MDSL_ANNOTATE("matcore.op.gemm")\nvoid gemm'
        trusted_signature = (
            "void gemm(out_arg output, const matrix_view &lhs, "
            "const matrix_view &rhs,\n          policy execution_policy = {});"
        )
        trusted_mutations = (
            (
                "trusted_missing_annotation",
                trusted_annotation,
                "void gemm",
                ("annotation", "annotate"),
            ),
            (
                "trusted_wrong_signature",
                trusted_signature,
                (
                    "void gemm(out_arg output, const matrix_view &lhs, "
                    "matrix_view rhs,\n          policy execution_policy = {});"
                ),
                ("signature", "parameter", "const matrix_view"),
            ),
        )
        for name, old, new, words in trusted_mutations:
            checks.case(
                name,
                lambda name=name, old=old, new=new, words=words: (
                    trusted_header_mutation_case(
                        checks, extractor, clang, temporary, name, old, new, words
                    )
                ),
            )

        without_flag = extraction(
            extractor,
            positives / "flag_forwarding.mdsl",
            temporary / "missing-flag",
            "missing-flag",
            frontend="native",
            clang=clang,
        )
        checks.require(
            without_flag.completed.returncode != 0,
            "native extraction ignored a missing semantic -D flag",
        )
        checks.require(
            not without_flag.ir.exists(), "missing compile flag still produced IR"
        )


def installed_suite(checks: Checks, extractor: Path, clang: Path) -> None:
    prefix = extractor.resolve().parent.parent
    trusted_header = prefix / "include/matcore/mdsl.h"
    checks.require(trusted_header.is_file(), "installed trusted header is missing")
    with tempfile.TemporaryDirectory(prefix="matcore-native-installed-") as encoded:
        temporary = Path(encoded)
        source = FIXTURES / "positive/namespace_alias.mdsl"
        installed = extraction(
            extractor,
            source,
            temporary / "installed",
            "installed",
            frontend=None,
            clang=clang,
        )
        checks.require(
            installed.completed.returncode == 0,
            f"installed native extraction failed:\n{installed.completed.stderr}",
        )
        if installed.completed.returncode == 0:
            checks.require(
                load_ir(installed).get("producer") == NATIVE_PRODUCER,
                "installed extractor defaulted away from native",
            )
        shadow = extraction(
            extractor,
            source,
            temporary / "shadow",
            "shadow",
            frontend="native",
            clang=clang,
            extra_compile_arguments=(f"-I{FIXTURES / 'shadow'}",),
        )
        checks.require(
            shadow.completed.returncode == 0,
            f"installed header was shadowed by user include path:\n{shadow.completed.stderr}",
        )
        copied = extraction(
            extractor,
            FIXTURES / "negative/copied_header.mdsl",
            temporary / "copied",
            "copied",
            frontend="native",
            clang=clang,
        )
        checks.require(
            copied.completed.returncode != 0,
            "installed extractor trusted a copied public header",
        )
        checks.require(
            not copied.ir.exists(), "installed copied-header attack emitted IR"
        )
    binary = extractor.read_bytes()
    checks.require(
        str(REPOSITORY).encode() not in binary,
        "installed extractor embeds the source checkout path",
    )


def unavailable_suite(
    checks: Checks, extractor: Path, driver: Path, clang: Path
) -> None:
    with tempfile.TemporaryDirectory(prefix="matcore-native-unavailable-") as encoded:
        temporary = Path(encoded)
        source = FIXTURES / "positive/namespace_alias.mdsl"
        default = extraction(
            extractor,
            source,
            temporary / "default",
            "default",
            frontend=None,
            clang=clang,
        )
        checks.require(
            default.completed.returncode != 0,
            "native-unavailable default silently succeeded",
        )
        checks.require(not default.ir.exists(), "native-unavailable default emitted IR")
        diagnostic = default.completed.stderr.lower()
        checks.require(
            "native" in diagnostic
            and ("unavailable" in diagnostic or "not built" in diagnostic),
            f"native-unavailable diagnostic was unclear:\n{default.completed.stderr}",
        )
        bootstrap = extraction(
            extractor,
            source,
            temporary / "bootstrap",
            "bootstrap",
            frontend="ast-json-bootstrap",
            clang=clang,
        )
        checks.require(
            bootstrap.completed.returncode == 0,
            f"explicit bootstrap mode failed:\n{bootstrap.completed.stderr}",
        )
        if bootstrap.completed.returncode == 0:
            checks.require(
                load_ir(bootstrap).get("producer") == BOOTSTRAP_PRODUCER,
                "explicit compatibility mode did not label bootstrap producer",
            )

        driver_default_output = temporary / "driver-default.o"
        driver_default = run(
            [
                str(driver),
                "--matcore-target=cpu",
                "-std=c++20",
                "-c",
                source_argument(source),
                "-o",
                str(driver_default_output),
            ]
        )
        checks.require(
            driver_default.returncode != 0,
            "bootstrap-only mdslc++ silently used bootstrap by default",
        )
        checks.require(
            not driver_default_output.exists(),
            "bootstrap-only default mdslc++ emitted an object",
        )
        driver_default_diagnostic = driver_default.stderr.lower()
        checks.require(
            "native" in driver_default_diagnostic
            and (
                "unavailable" in driver_default_diagnostic
                or "not built" in driver_default_diagnostic
            ),
            "bootstrap-only default mdslc++ lacked a native-unavailable diagnostic:\n"
            f"{driver_default.stderr}",
        )

        bootstrap_root = temporary / "driver-bootstrap"
        bootstrap_root.mkdir()
        driver_bootstrap_output = bootstrap_root / "driver-bootstrap.o"
        driver_bootstrap = run(
            [
                str(driver),
                "--frontend=ast-json-bootstrap",
                "--save-temps",
                "--matcore-target=cpu",
                "-std=c++20",
                "-c",
                source_argument(source),
                "-o",
                str(driver_bootstrap_output),
            ]
        )
        checks.require(
            driver_bootstrap.returncode == 0,
            "bootstrap-only mdslc++ explicit compatibility mode failed:\n"
            f"{driver_bootstrap.stderr}",
        )
        checks.require(
            driver_bootstrap_output.is_file(),
            "explicit bootstrap mdslc++ emitted no object",
        )
        driver_bootstrap_ir = bootstrap_root / "driver-bootstrap.matcore.json"
        checks.require(
            driver_bootstrap_ir.is_file()
            and json.loads(driver_bootstrap_ir.read_text()).get("version") == 1
            and json.loads(driver_bootstrap_ir.read_text()).get("producer")
            == BOOTSTRAP_PRODUCER,
            "explicit bootstrap mdslc++ did not preserve bootstrap provenance",
        )


def copy_driver_layout(driver: Path, extractor: Path, root: Path) -> tuple[Path, Path]:
    original_root = driver.resolve().parent.parent
    (root / "bin").mkdir(parents=True)
    (root / "include").mkdir()
    (root / "lib").mkdir()
    copied_driver = root / "bin/mdslc++"
    shutil.copy2(driver, copied_driver)
    shutil.copytree(original_root / "include", root / "include", dirs_exist_ok=True)
    runtime = original_root / "lib/libmatcore_runtime.so"
    if runtime.exists():
        os.symlink(runtime.resolve(), root / "lib/libmatcore_runtime.so")
    real_extractor = root / "bin/matcore-extract-real"
    os.symlink(extractor.resolve(), real_extractor)
    wrapper = root / "bin/matcore-extract"
    wrapper.write_text(
        "#!/usr/bin/env python3\n"
        "import pathlib, subprocess, sys\n"
        f"command = [{str(extractor.resolve())!r}, *sys.argv[1:]]\n"
        "completed = subprocess.run(command, check=False)\n"
        "if completed.returncode == 0 and '--input' in sys.argv:\n"
        "    source = pathlib.Path(sys.argv[sys.argv.index('--input') + 1])\n"
        "    with source.open('ab') as stream:\n"
        "        stream.write(b'\\n// deterministic post-extraction edit\\n')\n"
        "raise SystemExit(completed.returncode)\n",
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    return copied_driver, wrapper


def driver_suite(checks: Checks, extractor: Path, driver: Path, clang: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="matcore-native-driver-") as encoded:
        temporary = Path(encoded)
        source = FIXTURES / "positive/flag_forwarding.mdsl"
        for frontend in (None, "native", "ast-json-bootstrap"):
            label = frontend or "default"
            output_root = temporary / label
            output_root.mkdir()
            output = output_root / f"{label}.o"
            command = [str(driver), "--verbose", "--save-temps"]
            if frontend is not None:
                command.append(f"--frontend={frontend}")
            command.extend(
                [
                    "--matcore-target=cpu",
                    "-std=c++20",
                    "-DMDSLC_NATIVE_VALIDATION_FLAG=17",
                    "-c",
                    source_argument(source),
                    "-o",
                    str(output),
                ]
            )
            completed = run(command)
            checks.require(
                completed.returncode == 0,
                f"driver {label} mode failed:\n{completed.stderr}",
            )
            checks.require(output.is_file(), f"driver {label} emitted no object")
            ir = output_root / f"{label}.matcore.json"
            if ir.exists():
                expected = (
                    BOOTSTRAP_PRODUCER
                    if frontend == "ast-json-bootstrap"
                    else NATIVE_PRODUCER
                )
                checks.require(
                    json.loads(ir.read_text()).get("producer") == expected,
                    f"driver {label} selected the wrong frontend",
                )
            else:
                checks.require(
                    False, f"driver {label} did not preserve IR with --save-temps"
                )
            checks.require(
                completed.stderr.count("MDSLC_NATIVE_VALIDATION_FLAG=17") >= 2,
                f"driver {label} did not forward semantic flags to extraction and compile",
            )

        missing_output = temporary / "missing-flag.o"
        missing = run(
            [
                str(driver),
                "--frontend=native",
                "--matcore-target=cpu",
                "-std=c++20",
                "-c",
                source_argument(source),
                "-o",
                str(missing_output),
            ]
        )
        checks.require(
            missing.returncode != 0,
            "driver accepted source when extraction flag was missing",
        )
        checks.require(
            not missing_output.exists(),
            "driver emitted object after semantic flag failure",
        )

        driver_overlay = temporary / "driver-header-replacement.yaml"
        driver_overlay.write_text(
            json.dumps(
                {
                    "version": 0,
                    "roots": [
                        {
                            "type": "file",
                            "name": str(
                                extractor.resolve().parent.parent
                                / "include/matcore/mdsl.h"
                            ),
                            "external-contents": str(
                                FIXTURES / "negative/copied/matcore/mdsl.h"
                            ),
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        overlay_output = temporary / "driver-overlay.o"
        overlay_attempt = run(
            [
                str(driver),
                "--frontend=native",
                "--matcore-target=cpu",
                f"-ivfsoverlay{driver_overlay}",
                "-c",
                source_argument(FIXTURES / "positive/namespace_alias.mdsl"),
                "-o",
                str(overlay_output),
            ]
        )
        checks.require(
            overlay_attempt.returncode != 0,
            "driver accepted a user-controlled Clang VFS overlay",
        )
        checks.require(
            not overlay_output.exists(),
            "driver VFS overlay rejection still emitted an object",
        )
        checks.require(
            "incompatible" in overlay_attempt.stderr.lower(),
            "driver VFS overlay rejection was not actionable:\n"
            f"{overlay_attempt.stderr}",
        )

        race_root = temporary / "race-prefix"
        race_driver, _ = copy_driver_layout(driver, extractor, race_root)
        race_source = temporary / "race.mdsl"
        race_source.write_bytes(
            (FIXTURES / "positive/namespace_alias.mdsl").read_bytes()
        )
        race_output = temporary / "race.o"
        raced = run(
            [
                str(race_driver),
                "--frontend=native",
                "--matcore-target=cpu",
                "-std=c++20",
                "-c",
                str(race_source),
                "-o",
                str(race_output),
            ]
        )
        checks.require(
            raced.returncode != 0, "driver accepted a source changed after extraction"
        )
        checks.require(
            not race_output.exists(),
            "source-change race emitted an inconsistent object",
        )
        checks.require(
            any(
                word in raced.stderr.lower()
                for word in ("changed", "snapshot", "modified")
            ),
            f"source-change race diagnostic was unclear:\n{raced.stderr}",
        )

        header_race_root = temporary / "header-race-prefix"
        header_race_driver, header_race_extractor = copy_driver_layout(
            driver, extractor, header_race_root
        )
        header_race_directory = temporary / "header-race-source"
        header_race_directory.mkdir()
        header_race_header = header_race_directory / "mode.h"
        header_race_header.write_text(
            "#pragma once\n#define MDSLC_HEADER_MODE 0\n", encoding="utf-8"
        )
        header_race_source = header_race_directory / "header_race.mdsl"
        header_race_source.write_text(
            '#include "mode.h"\n'
            "#include <matcore/mdsl.h>\n\n"
            "namespace md = matcore::mdsl;\n\n"
            "int header_race_entry() {\n"
            "  float lhs_data[1] = {2.0F};\n"
            "  float rhs_data[1] = {3.0F};\n"
            "  float output_data[1] = {};\n"
            "  md::matrix_view lhs{lhs_data, 1, 1};\n"
            "  md::matrix_view rhs{rhs_data, 1, 1};\n"
            "  md::matrix_view output{output_data, 1, 1};\n"
            "#if MDSLC_HEADER_MODE\n"
            "  md::gemm(md::out(output), lhs, rhs);\n"
            "#endif\n"
            "  return 0;\n"
            "}\n",
            encoding="utf-8",
        )
        header_race_extractor.write_text(
            "#!/usr/bin/env python3\n"
            "import pathlib, subprocess, sys\n"
            f"command = [{str(extractor.resolve())!r}, *sys.argv[1:]]\n"
            "completed = subprocess.run(command, check=False)\n"
            "if completed.returncode == 0:\n"
            f"    pathlib.Path({str(header_race_header)!r}).write_text("
            "'#pragma once\\n#define MDSLC_HEADER_MODE 1\\n', encoding='utf-8')\n"
            "raise SystemExit(completed.returncode)\n",
            encoding="utf-8",
        )
        header_race_extractor.chmod(0o755)
        header_race_output = temporary / "header-race.o"
        header_race = run(
            [
                str(header_race_driver),
                "--frontend=native",
                "--matcore-target=cpu",
                "-std=c++20",
                "-c",
                str(header_race_source),
                "-o",
                str(header_race_output),
            ]
        )
        checks.require(
            header_race.returncode != 0,
            "driver accepted an included header changed after extraction",
        )
        checks.require(
            not header_race_output.exists(),
            "included-header race emitted an inconsistent object",
        )
        checks.require(
            "dependency changed" in header_race.stderr.lower()
            and str(header_race_header) in header_race.stderr,
            f"included-header race diagnostic was unclear:\n{header_race.stderr}",
        )

        isystem_race_root = temporary / "isystem-race-prefix"
        isystem_race_driver, isystem_race_extractor = copy_driver_layout(
            driver, extractor, isystem_race_root
        )
        isystem_directory = temporary / "isystem-headers"
        isystem_directory.mkdir()
        isystem_header = isystem_directory / "system_mode.h"
        isystem_header.write_text(
            "#pragma once\n#define MDSLC_SYSTEM_MODE 17\n", encoding="utf-8"
        )
        isystem_source = temporary / "isystem_race.mdsl"
        isystem_source.write_text(
            "#include <system_mode.h>\n"
            + (FIXTURES / "positive/namespace_alias.mdsl").read_text(
                encoding="utf-8"
            ),
            encoding="utf-8",
        )
        isystem_race_extractor.write_text(
            "#!/usr/bin/env python3\n"
            "import pathlib, subprocess, sys\n"
            f"command = [{str(extractor.resolve())!r}, *sys.argv[1:]]\n"
            "completed = subprocess.run(command, check=False)\n"
            "if completed.returncode == 0:\n"
            f"    pathlib.Path({str(isystem_header)!r}).write_text("
            "'#pragma once\\n#define MDSLC_SYSTEM_MODE 91\\n', encoding='utf-8')\n"
            "raise SystemExit(completed.returncode)\n",
            encoding="utf-8",
        )
        isystem_race_extractor.chmod(0o755)
        isystem_output = temporary / "isystem-race.o"
        isystem_race = run(
            [
                str(isystem_race_driver),
                "--frontend=native",
                "--matcore-target=cpu",
                "-std=c++20",
                "-isystem",
                str(isystem_directory),
                "-c",
                str(isystem_source),
                "-o",
                str(isystem_output),
            ]
        )
        checks.require(
            isystem_race.returncode != 0,
            "private consistency scan omitted a changed -isystem header",
        )
        checks.require(
            not isystem_output.exists(),
            "changed -isystem header still produced an object",
        )
        checks.require(
            "dependency changed" in isystem_race.stderr.lower()
            and str(isystem_header) in isystem_race.stderr,
            f"-isystem race diagnostic was unclear:\n{isystem_race.stderr}",
        )

        symlink_race_root = temporary / "symlink-race-prefix"
        symlink_race_driver, symlink_race_extractor = copy_driver_layout(
            driver, extractor, symlink_race_root
        )
        symlink_directory = temporary / "symlink-source"
        symlink_directory.mkdir()
        symlink_target_a = symlink_directory / "target-a.h"
        symlink_target_b = symlink_directory / "target-b.h"
        symlink_target_a.write_text("#pragma once\n", encoding="utf-8")
        os.link(symlink_target_a, symlink_target_b)
        symlink_header = symlink_directory / "mode.h"
        symlink_header.symlink_to(symlink_target_a.name)
        symlink_source = symlink_directory / "symlink_race.mdsl"
        symlink_source.write_text(
            '#include "mode.h"\n'
            + (FIXTURES / "positive/namespace_alias.mdsl").read_text(
                encoding="utf-8"
            ),
            encoding="utf-8",
        )
        symlink_race_extractor.write_text(
            "#!/usr/bin/env python3\n"
            "import os, pathlib, subprocess, sys\n"
            f"command = [{str(extractor.resolve())!r}, *sys.argv[1:]]\n"
            "completed = subprocess.run(command, check=False)\n"
            "if completed.returncode == 0:\n"
            f"    link = pathlib.Path({str(symlink_header)!r})\n"
            "    link.unlink()\n"
            f"    link.symlink_to({symlink_target_b.name!r})\n"
            "raise SystemExit(completed.returncode)\n",
            encoding="utf-8",
        )
        symlink_race_extractor.chmod(0o755)
        symlink_output = temporary / "symlink-race.o"
        symlink_race = run(
            [
                str(symlink_race_driver),
                "--frontend=native",
                "--matcore-target=cpu",
                "-std=c++20",
                "-c",
                str(symlink_source),
                "-o",
                str(symlink_output),
            ]
        )
        checks.require(
            symlink_race.returncode != 0,
            "driver accepted a dependency symlink retargeted to a hardlink",
        )
        checks.require(
            not symlink_output.exists(),
            "dependency symlink retarget emitted an inconsistent object",
        )
        checks.require(
            "dependency changed" in symlink_race.stderr.lower()
            and "identity" in symlink_race.stderr.lower(),
            f"symlink retarget diagnostic was unclear:\n{symlink_race.stderr}",
        )

        dotdot_race_root = temporary / "symlink-dotdot-race-prefix"
        dotdot_race_driver, dotdot_race_extractor = copy_driver_layout(
            driver, extractor, dotdot_race_root
        )
        dotdot_directory = temporary / "symlink-dotdot-source"
        actual_directory = dotdot_directory / "actual"
        (actual_directory / "sub").mkdir(parents=True)
        actual_header = actual_directory / "mode.h"
        actual_header.write_text(
            "#pragma once\n#define MDSLC_DOTDOT_MODE 17\n", encoding="utf-8"
        )
        decoy_header = dotdot_directory / "mode.h"
        decoy_header.write_text(
            "#pragma once\n#define MDSLC_DOTDOT_MODE 17\n", encoding="utf-8"
        )
        (dotdot_directory / "link").symlink_to("actual/sub", target_is_directory=True)
        dotdot_source = dotdot_directory / "symlink_dotdot_race.mdsl"
        dotdot_source.write_text(
            '#include "link/../mode.h"\n'
            + (FIXTURES / "positive/namespace_alias.mdsl").read_text(
                encoding="utf-8"
            ),
            encoding="utf-8",
        )
        dotdot_race_extractor.write_text(
            "#!/usr/bin/env python3\n"
            "import pathlib, subprocess, sys\n"
            f"command = [{str(extractor.resolve())!r}, *sys.argv[1:]]\n"
            "completed = subprocess.run(command, check=False)\n"
            "if completed.returncode == 0:\n"
            f"    pathlib.Path({str(actual_header)!r}).write_text("
            "'#pragma once\\n#define MDSLC_DOTDOT_MODE 91\\n', encoding='utf-8')\n"
            "raise SystemExit(completed.returncode)\n",
            encoding="utf-8",
        )
        dotdot_race_extractor.chmod(0o755)
        dotdot_output = temporary / "symlink-dotdot-race.o"
        dotdot_race = run(
            [
                str(dotdot_race_driver),
                "--frontend=native",
                "--matcore-target=cpu",
                "-std=c++20",
                "-c",
                str(dotdot_source),
                "-o",
                str(dotdot_output),
            ]
        )
        checks.require(
            dotdot_race.returncode != 0,
            "driver normalized link/../mode.h to a decoy dependency path",
        )
        checks.require(
            not dotdot_output.exists(),
            "changed link/../mode.h dependency still produced an object",
        )
        checks.require(
            "dependency changed" in dotdot_race.stderr.lower()
            and "link/../mode.h" in dotdot_race.stderr,
            f"symlink/.. race diagnostic was unclear:\n{dotdot_race.stderr}",
        )

        shadow_race_root = temporary / "include-shadow-race-prefix"
        shadow_race_driver, shadow_race_extractor = copy_driver_layout(
            driver, extractor, shadow_race_root
        )
        shadow_directory = temporary / "include-shadow-source"
        early_include = shadow_directory / "early"
        late_include = shadow_directory / "late"
        early_include.mkdir(parents=True)
        late_include.mkdir()
        late_header = late_include / "mode.h"
        late_header.write_text(
            "#pragma once\n#define MDSLC_SHADOW_MODE 17\n", encoding="utf-8"
        )
        early_header = early_include / "mode.h"
        shadow_source = shadow_directory / "include_shadow_race.mdsl"
        shadow_source.write_text(
            "#include <mode.h>\n"
            + (FIXTURES / "positive/namespace_alias.mdsl").read_text(
                encoding="utf-8"
            ),
            encoding="utf-8",
        )
        shadow_race_extractor.write_text(
            "#!/usr/bin/env python3\n"
            "import pathlib, subprocess, sys\n"
            f"command = [{str(extractor.resolve())!r}, *sys.argv[1:]]\n"
            "completed = subprocess.run(command, check=False)\n"
            "if completed.returncode == 0:\n"
            f"    pathlib.Path({str(early_header)!r}).write_text("
            "'#pragma once\\n#define MDSLC_SHADOW_MODE 91\\n', encoding='utf-8')\n"
            "raise SystemExit(completed.returncode)\n",
            encoding="utf-8",
        )
        shadow_race_extractor.chmod(0o755)
        shadow_output = temporary / "include-shadow-race.o"
        shadow_race = run(
            [
                str(shadow_race_driver),
                "--frontend=native",
                "--matcore-target=cpu",
                "-std=c++20",
                "-I",
                str(early_include),
                "-I",
                str(late_include),
                "-c",
                str(shadow_source),
                "-o",
                str(shadow_output),
            ]
        )
        checks.require(
            shadow_race.returncode != 0,
            "driver accepted a higher-priority header created after extraction",
        )
        checks.require(
            not shadow_output.exists(),
            "changed include resolution still produced an object",
        )
        checks.require(
            "dependency resolution changed" in shadow_race.stderr.lower()
            and str(early_header) in shadow_race.stderr
            and str(late_header) in shadow_race.stderr,
            f"include-shadow race diagnostic was unclear:\n{shadow_race.stderr}",
        )

        postscan_race_root = temporary / "postscan-shadow-race-prefix"
        postscan_race_driver, postscan_race_extractor = copy_driver_layout(
            driver, extractor, postscan_race_root
        )
        postscan_directory = temporary / "postscan-shadow-source"
        postscan_early_include = postscan_directory / "early"
        postscan_late_include = postscan_directory / "late"
        postscan_early_include.mkdir(parents=True)
        postscan_late_include.mkdir()
        postscan_late_header = postscan_late_include / "mode.h"
        postscan_late_header.write_text(
            "#pragma once\n#define MDSLC_POSTSCAN_MODE 17\n", encoding="utf-8"
        )
        postscan_early_header = postscan_early_include / "mode.h"
        postscan_source = postscan_directory / "postscan_shadow_race.mdsl"
        postscan_source.write_text(
            "#include <mode.h>\n"
            + (FIXTURES / "positive/namespace_alias.mdsl").read_text(
                encoding="utf-8"
            ),
            encoding="utf-8",
        )
        postscan_watcher = postscan_directory / "watch_private_stubs.py"
        postscan_watcher.write_text(
            "import pathlib, sys, time\n"
            "trigger = pathlib.Path(sys.argv[1])\n"
            "header = pathlib.Path(sys.argv[2])\n"
            "deadline = time.monotonic() + 10.0\n"
            "while time.monotonic() < deadline:\n"
            "    if trigger.exists():\n"
            "        header.write_text("
            "'#pragma once\\n#define MDSLC_POSTSCAN_MODE 91\\n', encoding='utf-8')\n"
            "        raise SystemExit(0)\n"
            "    time.sleep(0.001)\n"
            "raise SystemExit(2)\n",
            encoding="utf-8",
        )
        postscan_race_extractor.write_text(
            "#!/usr/bin/env python3\n"
            "import pathlib, subprocess, sys, tempfile\n"
            f"command = [{str(extractor.resolve())!r}, *sys.argv[1:]]\n"
            "completed = subprocess.run(command, check=False)\n"
            "if completed.returncode == 0 and '--input' in sys.argv:\n"
            "    source = pathlib.Path(sys.argv[sys.argv.index('--input') + 1]).resolve()\n"
            "    candidates = []\n"
            "    for depfile in pathlib.Path(tempfile.gettempdir()).glob("
            "'mdslc-*/private-source-closure.d'):\n"
            "        try:\n"
            "            if str(source) in depfile.read_text(encoding='utf-8'):\n"
            "                candidates.append(depfile.parent)\n"
            "        except OSError:\n"
            "            pass\n"
            "    if not candidates:\n"
            "        raise SystemExit(92)\n"
            "    dependency_root = max(candidates, key=lambda path: path.stat().st_mtime_ns)\n"
            f"    subprocess.Popen([sys.executable, {str(postscan_watcher)!r}, "
            "str(dependency_root / 'private-stubs-closure.d'), "
            f"{str(postscan_early_header)!r}], stdout=subprocess.DEVNULL, "
            "stderr=subprocess.DEVNULL, start_new_session=True)\n"
            "raise SystemExit(completed.returncode)\n",
            encoding="utf-8",
        )
        postscan_race_extractor.chmod(0o755)
        postscan_output = temporary / "postscan-shadow-race.o"
        postscan_race = run(
            [
                str(postscan_race_driver),
                "--frontend=native",
                "--matcore-target=cpu",
                "-std=c++20",
                "-I",
                str(postscan_early_include),
                "-I",
                str(postscan_late_include),
                "-c",
                str(postscan_source),
                "-o",
                str(postscan_output),
            ]
        )
        checks.require(
            postscan_race.returncode != 0,
            "driver accepted include shadowing introduced after its pre-host scan",
        )
        checks.require(
            not postscan_output.exists(),
            "postscan include shadowing still produced a final object",
        )
        checks.require(
            "dependency resolution changed" in postscan_race.stderr.lower()
            and str(postscan_early_header) in postscan_race.stderr
            and str(postscan_late_header) in postscan_race.stderr,
            f"postscan include-shadow diagnostic was unclear:\n{postscan_race.stderr}",
        )

        host_artifact_root = temporary / "host-artifact-race-prefix"
        host_artifact_driver, host_artifact_extractor = copy_driver_layout(
            driver, extractor, host_artifact_root
        )
        host_artifact_source = temporary / "host_artifact_race.mdsl"
        host_artifact_source.write_text(
            "int main() { return 0; }\n", encoding="utf-8"
        )
        host_artifact_watcher = temporary / "watch_private_host.py"
        host_artifact_watcher.write_text(
            "import pathlib, sys, time\n"
            "trigger = pathlib.Path(sys.argv[1])\n"
            "host = pathlib.Path(sys.argv[2])\n"
            "deadline = time.monotonic() + 10.0\n"
            "while time.monotonic() < deadline:\n"
            "    if trigger.exists():\n"
            "        contents = host.read_text(encoding='utf-8')\n"
            "        host.write_text(contents.replace('return 0;', 'return 73;'), "
            "encoding='utf-8')\n"
            "        raise SystemExit(0)\n"
            "    time.sleep(0.001)\n"
            "raise SystemExit(2)\n",
            encoding="utf-8",
        )
        host_artifact_extractor.write_text(
            "#!/usr/bin/env python3\n"
            "import pathlib, subprocess, sys, tempfile\n"
            f"command = [{str(extractor.resolve())!r}, *sys.argv[1:]]\n"
            "completed = subprocess.run(command, check=False)\n"
            "if completed.returncode == 0 and '--input' in sys.argv:\n"
            "    source = pathlib.Path(sys.argv[sys.argv.index('--input') + 1]).resolve()\n"
            "    host = pathlib.Path(sys.argv[sys.argv.index('--rewrite-out') + 1])\n"
            "    candidates = []\n"
            "    for depfile in pathlib.Path(tempfile.gettempdir()).glob("
            "'mdslc-*/private-source-closure.d'):\n"
            "        try:\n"
            "            if str(source) in depfile.read_text(encoding='utf-8'):\n"
            "                candidates.append(depfile.parent)\n"
            "        except OSError:\n"
            "            pass\n"
            "    if not candidates:\n"
            "        raise SystemExit(93)\n"
            "    dependency_root = max(candidates, key=lambda path: path.stat().st_mtime_ns)\n"
            f"    subprocess.Popen([sys.executable, {str(host_artifact_watcher)!r}, "
            "str(dependency_root / 'private-host-closure.d'), str(host)], "
            "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, "
            "start_new_session=True)\n"
            "raise SystemExit(completed.returncode)\n",
            encoding="utf-8",
        )
        host_artifact_extractor.chmod(0o755)
        host_artifact_output = temporary / "host-artifact-race"
        host_artifact_race = run(
            [
                str(host_artifact_driver),
                "--frontend=native",
                "--save-temps",
                "--matcore-target=cpu",
                "-std=c++20",
                str(host_artifact_source),
                "-o",
                str(host_artifact_output),
            ]
        )
        checks.require(
            host_artifact_race.returncode != 0,
            "driver accepted a generated host source changed after its scan",
        )
        checks.require(
            not host_artifact_output.exists(),
            "generated host source mutation still produced a final executable",
        )
        expected_host_artifact = temporary / "host-artifact-race.host.cpp"
        checks.require(
            "dependency changed" in host_artifact_race.stderr.lower()
            and str(expected_host_artifact) in host_artifact_race.stderr,
            f"generated host mutation diagnostic was unclear:\n{host_artifact_race.stderr}",
        )

        stub_shadow_root = temporary / "stub-shadow-race-prefix"
        stub_shadow_driver, stub_shadow_extractor = copy_driver_layout(
            driver, extractor, stub_shadow_root
        )
        stub_shadow_directory = temporary / "stub-shadow-source"
        stub_early_include = stub_shadow_directory / "early"
        stub_late_include = stub_shadow_directory / "late"
        stub_early_include.mkdir(parents=True)
        stub_late_include.mkdir()
        stub_late_header = stub_late_include / "stdexcept"
        stub_late_header.write_text(
            "#pragma once\n#include_next <stdexcept>\n", encoding="utf-8"
        )
        stub_early_header = stub_early_include / "stdexcept"
        stub_shadow_source = stub_shadow_directory / "stub_shadow_race.mdsl"
        stub_shadow_source.write_bytes(
            (FIXTURES / "positive/namespace_alias.mdsl").read_bytes()
        )
        stub_shadow_watcher = stub_shadow_directory / "watch_public_stubs.py"
        stub_shadow_watcher.write_text(
            "import pathlib, sys, time\n"
            "trigger = pathlib.Path(sys.argv[1])\n"
            "header = pathlib.Path(sys.argv[2])\n"
            "deadline = time.monotonic() + 10.0\n"
            "while time.monotonic() < deadline:\n"
            "    if trigger.exists():\n"
            "        header.write_text("
            "'#pragma once\\n#include_next <stdexcept>\\n', encoding='utf-8')\n"
            "        raise SystemExit(0)\n"
            "    time.sleep(0.001)\n"
            "raise SystemExit(2)\n",
            encoding="utf-8",
        )
        stub_shadow_extractor.write_text(
            "#!/usr/bin/env python3\n"
            "import pathlib, subprocess, sys, tempfile\n"
            f"command = [{str(extractor.resolve())!r}, *sys.argv[1:]]\n"
            "completed = subprocess.run(command, check=False)\n"
            "if completed.returncode == 0 and '--input' in sys.argv:\n"
            "    source = pathlib.Path(sys.argv[sys.argv.index('--input') + 1]).resolve()\n"
            "    candidates = []\n"
            "    for depfile in pathlib.Path(tempfile.gettempdir()).glob("
            "'mdslc-*/private-source-closure.d'):\n"
            "        try:\n"
            "            if str(source) in depfile.read_text(encoding='utf-8'):\n"
            "                candidates.append(depfile.parent)\n"
            "        except OSError:\n"
            "            pass\n"
            "    if not candidates:\n"
            "        raise SystemExit(94)\n"
            "    dependency_root = max(candidates, key=lambda path: path.stat().st_mtime_ns)\n"
            f"    subprocess.Popen([sys.executable, {str(stub_shadow_watcher)!r}, "
            "str(dependency_root / 'public-stubs.d'), "
            f"{str(stub_early_header)!r}], stdout=subprocess.DEVNULL, "
            "stderr=subprocess.DEVNULL, start_new_session=True)\n"
            "raise SystemExit(completed.returncode)\n",
            encoding="utf-8",
        )
        stub_shadow_extractor.chmod(0o755)
        stub_shadow_output = temporary / "stub-shadow-race.o"
        stub_shadow_depfile = temporary / "stub-shadow-race.d"
        stub_shadow_race = run(
            [
                str(stub_shadow_driver),
                "--frontend=native",
                "--matcore-target=cpu",
                "-std=c++20",
                "-I",
                str(stub_early_include),
                "-I",
                str(stub_late_include),
                "-MMD",
                "-MF",
                str(stub_shadow_depfile),
                "-c",
                str(stub_shadow_source),
                "-o",
                str(stub_shadow_output),
            ]
        )
        checks.require(
            stub_shadow_race.returncode != 0,
            "driver accepted generated-stub header shadowing after public scan",
        )
        checks.require(
            not stub_shadow_output.exists() and not stub_shadow_depfile.exists(),
            "generated-stub shadowing left a final object or published depfile",
        )
        checks.require(
            "dependency resolution changed" in stub_shadow_race.stderr.lower()
            and "generated stubs" in stub_shadow_race.stderr.lower()
            and str(stub_early_header) in stub_shadow_race.stderr,
            f"generated-stub shadow diagnostic was unclear:\n{stub_shadow_race.stderr}",
        )

        include_parity = temporary / "include-parity"
        include_source = include_parity / "source"
        include_output = include_parity / "output"
        include_source.mkdir(parents=True)
        include_output.mkdir()
        (include_source / "config.h").write_text(
            "#pragma once\ninline constexpr int selected_header_value = 17;\n",
            encoding="utf-8",
        )
        (include_output / "config.h").write_text(
            "#pragma once\ninline constexpr int selected_header_value = 91;\n",
            encoding="utf-8",
        )
        parity_source = include_source / "parity.mdsl"
        parity_source.write_text(
            '#include "config.h"\n'
            "#include <matcore/mdsl.h>\n\n"
            "namespace md = matcore::mdsl;\n\n"
            "int main() {\n"
            "  float lhs_data[1] = {2.0F};\n"
            "  float rhs_data[1] = {3.0F};\n"
            "  float output_data[1] = {};\n"
            "  md::matrix_view lhs{lhs_data, 1, 1};\n"
            "  md::matrix_view rhs{rhs_data, 1, 1};\n"
            "  md::matrix_view output{output_data, 1, 1};\n"
            "  md::gemm(md::out(output), lhs, rhs);\n"
            "  return selected_header_value == 17 && output_data[0] == 6.0F"
            " ? 0 : 7;\n"
            "}\n",
            encoding="utf-8",
        )
        parity_executable = include_output / "parity"
        parity_compile = run(
            [
                str(driver),
                "--verbose",
                "--save-temps",
                "--frontend=native",
                "--matcore-target=cpu",
                "-std=c++20",
                str(parity_source),
                "-o",
                str(parity_executable),
            ]
        )
        checks.require(
            parity_compile.returncode == 0,
            "driver quote-include parity build failed:\n"
            f"{parity_compile.stderr}",
        )
        checks.require(
            (include_output / "parity.host-overlay.yaml").is_file(),
            "driver did not preserve the host VFS overlay with --save-temps",
        )
        checks.require(
            "-ivfsoverlay" in parity_compile.stderr,
            "driver host compilation did not use the original source virtual path",
        )
        if parity_compile.returncode == 0:
            parity_run = run([str(parity_executable)])
            checks.require(
                parity_run.returncode == 0,
                "rewritten host resolved a quote-include from the generated "
                "artifact directory instead of the original source directory",
            )

        escaped_dependency_directory = temporary / "escaped dependency paths"
        escaped_dependency_directory.mkdir()
        escaped_header = escaped_dependency_directory / "value $ hash#.h"
        escaped_header.write_text("#pragma once\n", encoding="utf-8")
        escaped_source = escaped_dependency_directory / "escaped.mdsl"
        escaped_source.write_text(
            '#include "value $ hash#.h"\n'
            + (FIXTURES / "positive/namespace_alias.mdsl").read_text(
                encoding="utf-8"
            ),
            encoding="utf-8",
        )
        escaped_output = escaped_dependency_directory / "escaped:result.o"
        escaped_depfile = escaped_dependency_directory / "escaped $ dep#.d"
        escaped_compile = run(
            [
                str(driver),
                "--frontend=native",
                "--matcore-target=cpu",
                "-std=c++20",
                "-MMD",
                "-MF",
                str(escaped_depfile),
                "-c",
                str(escaped_source),
                "-o",
                str(escaped_output),
            ]
        )
        checks.require(
            escaped_compile.returncode == 0 and escaped_output.is_file(),
            "driver could not decode escaped Make dependency paths or a "
            f"colon-bearing target:\n{escaped_compile.stderr}",
        )
        if escaped_depfile.is_file():
            escaped_depfile_text = escaped_depfile.read_text(encoding="utf-8")
            checks.require(
                "escaped\\:result.o:" in escaped_depfile_text
                and "value\\ $$\\ hash\\#.h" in escaped_depfile_text,
                "merged public depfile did not preserve Make escaping for "
                f"colon, space, dollar, and hash paths:\n{escaped_depfile_text}",
            )
            make = shutil.which("make")
            if make is not None:
                make_parse = run([make, "-f", str(escaped_depfile), "-n"])
                checks.require(
                    make_parse.returncode == 0,
                    f"GNU make rejected the merged depfile:\n{make_parse.stderr}",
                )
        else:
            checks.require(False, "escaped dependency build emitted no depfile")

        pic_parity = temporary / "pic-parity"
        pic_source_directory = pic_parity / "source"
        pic_output_directory = pic_parity / "output"
        pic_source_directory.mkdir(parents=True)
        pic_output_directory.mkdir()
        (pic_source_directory / "pie_value.h").write_text(
            "#pragma once\ninline constexpr int selected_pic_value = 17;\n",
            encoding="utf-8",
        )
        (pic_source_directory / "pic_value.h").write_text(
            "#pragma once\ninline constexpr int selected_pic_value = 91;\n",
            encoding="utf-8",
        )
        pic_source = pic_source_directory / "pic_parity.mdsl"
        pic_source.write_text(
            "#if defined(__PIE__)\n"
            '#include "pie_value.h"\n'
            "#else\n"
            '#include "pic_value.h"\n'
            "#endif\n"
            "#include <matcore/mdsl.h>\n\n"
            "namespace md = matcore::mdsl;\n\n"
            "int main() {\n"
            "  float lhs_data[1] = {2.0F};\n"
            "  float rhs_data[1] = {3.0F};\n"
            "  float output_data[1] = {};\n"
            "  md::matrix_view lhs{lhs_data, 1, 1};\n"
            "  md::matrix_view rhs{rhs_data, 1, 1};\n"
            "  md::matrix_view output{output_data, 1, 1};\n"
            "  md::gemm(md::out(output), lhs, rhs);\n"
            "  return selected_pic_value == 17 && output_data[0] == 6.0F"
            " ? 0 : 9;\n"
            "}\n",
            encoding="utf-8",
        )
        pic_executable = pic_output_directory / "pic_parity"
        pic_dependency = pic_output_directory / "pic_parity.d"
        pic_compile = run(
            [
                str(driver),
                "--verbose",
                "--frontend=native",
                "--matcore-target=cpu",
                "-std=c++20",
                "-MMD",
                "-MF",
                str(pic_dependency),
                str(pic_source),
                "-o",
                str(pic_executable),
            ]
        )
        checks.require(
            pic_compile.returncode == 0,
            "driver changed predefined position-independent-code macros "
            f"between phases:\n{pic_compile.stderr}",
        )
        checks.require(
            "'-fPIC'" not in pic_compile.stderr,
            "driver forced -fPIC instead of preserving the user's compiler flags",
        )
        if pic_compile.returncode == 0:
            checks.require(
                run([str(pic_executable)]).returncode == 0,
                "generated host observed different predefined macros than extraction",
            )
        if pic_dependency.is_file():
            dependency_text = pic_dependency.read_text(encoding="utf-8")
            runtime_header = (
                driver.resolve().parent.parent / "include/matcore/runtime_c.h"
            )
            checks.require(
                "pie_value.h" in dependency_text and "pic_value.h" not in dependency_text,
                "dependency scan and host compilation selected different macro branches",
            )
            checks.require(
                str(runtime_header) in dependency_text,
                "public -MMD depfile omitted runtime_c.h used by generated sources",
            )
            checks.require(
                not any(
                    generated in dependency_text
                    for generated in (".host.cpp", ".sites.h", ".stubs.cpp", ".backend.cpp")
                ),
                "public depfile leaked temporary generated artifacts",
            )
            checks.require(
                "/usr/include/" not in dependency_text
                and "/lib/clang/21/include/" not in dependency_text,
                "public -MMD depfile unexpectedly included system headers",
            )
        else:
            checks.require(False, "PIC parity build emitted no dependency file")

        md_output = pic_output_directory / "public-md.o"
        md_dependency = pic_output_directory / "public-md.d"
        md_compile = run(
            [
                str(driver),
                "--frontend=native",
                "--matcore-target=cpu",
                "-std=c++20",
                "-MD",
                "-MF",
                str(md_dependency),
                "-c",
                str(pic_source),
                "-o",
                str(md_output),
            ]
        )
        checks.require(
            md_compile.returncode == 0 and md_output.is_file(),
            f"public -MD build failed:\n{md_compile.stderr}",
        )
        if md_dependency.is_file():
            md_dependency_text = md_dependency.read_text(encoding="utf-8")
            checks.require(
                "/usr/include/" in md_dependency_text
                or "/lib/clang/21/include/" in md_dependency_text,
                "public -MD depfile omitted system headers",
            )
        else:
            checks.require(False, "public -MD build emitted no dependency file")

        mp_output = temporary / "unsupported-mp.o"
        mp_compile = run(
            [
                str(driver),
                "--frontend=native",
                "--matcore-target=cpu",
                "-MMD",
                "-MP",
                "-c",
                source_argument(FIXTURES / "positive/namespace_alias.mdsl"),
                "-o",
                str(mp_output),
            ]
        )
        checks.require(
            mp_compile.returncode != 0 and "incompatible" in mp_compile.stderr.lower(),
            f"driver did not reject unsupported multi-rule -MP depfiles:\n{mp_compile.stderr}",
        )
        checks.require(
            not mp_output.exists(), "unsupported -MP mode emitted an object"
        )

        m32_output = temporary / "target-context-32.o"
        m32_compile = run(
            [
                str(driver),
                "--verbose",
                "--frontend=native",
                "--matcore-target=cpu",
                "-std=c++20",
                "-m32",
                "-c",
                source_argument(FIXTURES / "positive/namespace_alias.mdsl"),
                "-o",
                str(m32_output),
            ]
        )
        checks.require(
            m32_compile.returncode == 0,
            "driver did not preserve the 32-bit target context through the "
            f"relocatable partial link:\n{m32_compile.stderr}",
        )
        checks.require(
            m32_output.is_file()
            and m32_output.read_bytes()[:5] == b"\x7fELF\x01",
            "driver -m32 pipeline did not emit an ELF32 relocatable object",
        )
        checks.require(
            m32_compile.stderr.count("'-m32'") >= 5,
            "driver did not forward -m32 to extraction, generated "
            "compilation, and partial linking",
        )

        lto_output = temporary / "unsupported-lto.o"
        lto_compile = run(
            [
                str(driver),
                "--frontend=native",
                "--matcore-target=cpu",
                "-flto=thin",
                "-c",
                source_argument(FIXTURES / "positive/namespace_alias.mdsl"),
                "-o",
                str(lto_output),
            ]
        )
        checks.require(
            lto_compile.returncode != 0 and "lto" in lto_compile.stderr.lower(),
            f"driver did not reject unsupported LTO structurally:\n{lto_compile.stderr}",
        )
        checks.require(
            not lto_output.exists(),
            "unsupported LTO mode emitted an output object",
        )

        multi_tu = temporary / "multi-tu"
        multi_tu.mkdir()
        multi_source = multi_tu / "variant.mdsl"
        multi_source.write_text(
            "#include <matcore/mdsl.h>\n\n"
            "#ifndef REVIEW_ENTRY\n"
            '#error "REVIEW_ENTRY is required"\n'
            "#endif\n\n"
            "namespace md = matcore::mdsl;\n\n"
            "int REVIEW_ENTRY() {\n"
            "  float lhs_data[1] = {2.0F};\n"
            "  float rhs_data[1] = {3.0F};\n"
            "  float output_data[1] = {};\n"
            "  md::matrix_view lhs{lhs_data, 1, 1};\n"
            "  md::matrix_view rhs{rhs_data, 1, 1};\n"
            "  md::matrix_view output{output_data, 1, 1};\n"
            "  md::gemm(md::out(output), lhs, rhs);\n"
            "  return output_data[0] == 6.0F ? 0 : 1;\n"
            "}\n",
            encoding="utf-8",
        )
        variant_objects: list[Path] = []
        variant_site_ids: list[str] = []
        for entry in ("first_entry", "second_entry"):
            variant_directory = multi_tu / entry
            variant_directory.mkdir()
            variant_object = variant_directory / f"{entry}.o"
            variant_compile = run(
                [
                    str(driver),
                    "--save-temps",
                    "--frontend=native",
                    "--matcore-target=cpu",
                    "-std=c++20",
                    f"-DREVIEW_ENTRY={entry}",
                    "-c",
                    str(multi_source),
                    "-o",
                    str(variant_object),
                ]
            )
            checks.require(
                variant_compile.returncode == 0,
                f"multi-TU variant {entry} failed:\n{variant_compile.stderr}",
            )
            variant_objects.append(variant_object)
            variant_ir = variant_directory / f"{entry}.matcore.json"
            if variant_ir.is_file():
                variant_site_ids.append(
                    json.loads(variant_ir.read_text(encoding="utf-8"))["operations"][
                        0
                    ]["site_id"]
                )
            else:
                checks.require(False, f"multi-TU variant {entry} emitted no IR")
        checks.require(
            len(variant_site_ids) == 2 and len(set(variant_site_ids)) == 2,
            "compile-context variants reused a generated site ID",
        )
        multi_main = multi_tu / "main.cpp"
        multi_main.write_text(
            "int first_entry();\n"
            "int second_entry();\n"
            "int main() { return first_entry() || second_entry(); }\n",
            encoding="utf-8",
        )
        runtime_directory = driver.resolve().parent.parent / "lib"
        multi_executable = multi_tu / "multi-tu"
        multi_link = run(
            [
                str(clang),
                "-std=c++20",
                str(multi_main),
                *(str(path) for path in variant_objects),
                f"-L{runtime_directory}",
                "-lmatcore_runtime",
                "-Xlinker",
                "-rpath",
                "-Xlinker",
                str(runtime_directory),
                "-o",
                str(multi_executable),
            ]
        )
        checks.require(
            multi_link.returncode == 0,
            "compile-context variants had colliding generated symbols:\n"
            f"{multi_link.stderr}",
        )
        if multi_link.returncode == 0:
            checks.require(
                run([str(multi_executable)]).returncode == 0,
                "linked compile-context variants did not both execute correctly",
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        choices=("core", "installed", "unavailable", "driver"),
        default="core",
    )
    parser.add_argument("--extractor", type=Path, required=True)
    parser.add_argument("--driver", type=Path)
    parser.add_argument("--clang", type=Path, default=Path("/usr/bin/clang++-21"))
    arguments = parser.parse_args()

    extractor = arguments.extractor.resolve()
    # Preserve the clang++ argv[0] spelling: resolving its symlink to `clang`
    # changes final-link driver behavior and omits the C++ standard library.
    clang = arguments.clang.absolute()
    if not extractor.is_file():
        parser.error(f"extractor does not exist: {extractor}")
    if not clang.is_file():
        parser.error(f"Clang does not exist: {clang}")
    if arguments.suite in ("driver", "unavailable") and arguments.driver is None:
        parser.error(f"--suite {arguments.suite} requires --driver")

    checks = Checks()
    if arguments.suite == "core":
        core_suite(checks, extractor, clang)
    elif arguments.suite == "installed":
        installed_suite(checks, extractor, clang)
    elif arguments.suite == "unavailable":
        assert arguments.driver is not None
        unavailable_suite(checks, extractor, arguments.driver.resolve(), clang)
    else:
        assert arguments.driver is not None
        driver_suite(checks, extractor, arguments.driver.resolve(), clang)

    if checks.failures:
        print(
            f"native frontend {arguments.suite}: {len(checks.failures)} failure(s) "
            f"across {checks.count} checks",
            file=sys.stderr,
        )
        for failure in checks.failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    print(f"native frontend {arguments.suite}: {checks.count} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
