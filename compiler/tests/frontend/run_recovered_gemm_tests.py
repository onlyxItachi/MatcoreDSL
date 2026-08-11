#!/usr/bin/env python3
"""Diagnostic-only ordinary-C++ GEMM recovery tests.

The recovery path intentionally emits no optimizer IR and performs no rewrite.
These tests therefore authenticate both the inspection record and preservation
of the ordinary C++ object/execution path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path


RELAXED_FP_FLAGS = [
    "-O2",
    "-ffp-contract=fast",
    "-fassociative-math",
    "-fno-signed-zeros",
    "-fno-trapping-math",
    "-fhonor-nans",
    "-fhonor-infinities",
    "-fno-reciprocal-math",
    "-fno-approx-func",
    "-fno-rounding-math",
]


def report_values(report: str, candidate: int, key: str) -> list[str]:
    prefix = re.escape(f"candidate[{candidate}].{key}")
    values: list[str] = []
    for line in report.splitlines():
        if re.fullmatch(prefix + r"=.*", line):
            value = line.split("=", 1)[1]
            if value.startswith('"'):
                decoded, end = json.JSONDecoder().raw_decode(value)
                value = str(decoded) + value[end:]
            values.append(value)
    return values


def numbered_values(report: str, candidate: int, key: str) -> list[str]:
    prefix = re.escape(f"candidate[{candidate}].{key}")
    values: list[str] = []
    for line in report.splitlines():
        if re.fullmatch(prefix + r"\[[0-9]+\]=.*", line):
            value = line.split("=", 1)[1]
            if value.startswith('"'):
                decoded, end = json.JSONDecoder().raw_decode(value)
                value = str(decoded) + value[end:]
            values.append(value)
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extractor", type=Path, required=True)
    parser.add_argument("--clang", type=Path, required=True)
    args = parser.parse_args()

    repository = Path(__file__).resolve().parents[3]
    fixtures = repository / "compiler/tests/frontend/recovered"
    failures: list[str] = []
    checks = 0

    def check(condition: bool, message: str) -> None:
        nonlocal checks
        checks += 1
        if not condition:
            failures.append(message)

    with tempfile.TemporaryDirectory(prefix="matcore-recovered-gemm-") as temp:
        root = Path(temp)

        def inspect(
            source: Path,
            name: str,
            flags: list[str],
            *,
            frontend: str = "native",
        ) -> tuple[subprocess.CompletedProcess[str], str, Path]:
            ir = root / f"{name}.json"
            report = root / f"{name}.report"
            command = [
                str(args.extractor),
                f"--frontend={frontend}",
                "--input",
                str(source),
                "--ir-out",
                str(ir),
                "--inspect-recovered-gemm",
                str(report),
                "--",
                "clang++",
                "-std=c++20",
                *flags,
                f"-I{fixtures}",
                str(source),
            ]
            completed = subprocess.run(
                command,
                cwd=repository,
                text=True,
                capture_output=True,
                check=False,
            )
            return completed, report.read_text() if report.exists() else "", ir

        positive_reports: dict[str, str] = {}
        for name, count in (("canonical", 1), ("renamed", 1), ("two_functions", 2)):
            source = fixtures / f"{name}.mdsl"
            before = source.read_bytes()
            completed, report, ir = inspect(source, name, RELAXED_FP_FLAGS)
            check(completed.returncode == 0, f"positive {name} failed: {completed.stderr}")
            check(source.read_bytes() == before, f"inspection modified {name} source")
            check(report.startswith("matcore.recovered-gemm-inspection-v1\n"),
                  f"positive {name} emitted the wrong inspection format")
            check(f"candidate_count={count}" in report,
                  f"positive {name} emitted the wrong candidate count")
            for index in range(count):
                check(report_values(report, index, "state") == ["recognized_guard_required"],
                      f"positive {name}[{index}] did not require guards")
                check(report_values(report, index, "rewrite") == ["preserve_original_cpp"],
                      f"positive {name}[{index}] did not preserve C++")
                check(report_values(report, index, "fp.rounding_mode") == ["tonearest"],
                      f"positive {name}[{index}] saw an unexpected rounding spelling")
                check(report_values(report, index, "fp.denormal_mode") == ["ieee,ieee"],
                      f"positive {name}[{index}] saw an unexpected denormal spelling")
                check(report_values(report, index, "fp.fp32_denormal_mode") == ["ieee,ieee"],
                      f"positive {name}[{index}] saw an unexpected f32 denormal spelling")
                check(report_values(report, index, "rejection_reason_count") == ["0"],
                      f"positive {name}[{index}] unexpectedly had rejection reasons")
                check(report_values(report, index, "semantic_contract") ==
                      ["f32_row_major_overwrite_m_k__k_n__m_n"],
                      f"positive {name}[{index}] lost its typed GEMM contract")
                for binding in ("output", "lhs", "rhs", "m", "n", "k"):
                    check(bool(report_values(report, index, f"binding.{binding}") and
                               report_values(report, index, f"binding.{binding}")[0]),
                          f"positive {name}[{index}] omitted {binding} binding")
                proof_ranges = numbered_values(report, index, "proof_range")
                for role in ("outer_loop", "accumulator_update", "output_store"):
                    check(any(value.startswith(f"{role}:") for value in proof_ranges),
                          f"positive {name}[{index}] omitted {role} proof range")
                check("matcore::mdsl::gemm" not in report,
                      f"positive {name}[{index}] forged explicit-call provenance")
            document = json.loads(ir.read_text()) if ir.exists() else {}
            check(document.get("operations") == [],
                  f"positive {name} fabricated a Matcore IR capture operation")
            positive_reports[name] = report

        canonical_source = fixtures / "canonical.mdsl"
        expected_digest = "sha256:" + hashlib.sha256(canonical_source.read_bytes()).hexdigest()
        check(report_values(positive_reports["canonical"], 0, "source_snapshot_sha256") ==
              [expected_digest], "canonical source snapshot digest is not exact")
        check(report_values(positive_reports["canonical"], 0, "fp.allow_reassociation") == ["1"],
              "canonical relaxed profile did not prove reassociation")
        check(report_values(positive_reports["canonical"], 0,
                            "fp.contract_across_statement") == ["1"],
              "canonical relaxed profile did not prove cross-statement contraction")
        canonical_range = report_values(
            positive_reports["canonical"], 0, "outer_loop_range"
        )
        if canonical_range and ":" in canonical_range[0]:
            begin_text, end_text = canonical_range[0].split(":", 1)
            outer_bytes = canonical_source.read_bytes()[int(begin_text):int(end_text)]
            check(outer_bytes.startswith(b"for (") and outer_bytes.rstrip().endswith(b"}"),
                  "canonical outer-loop token range does not select the exact loop")
        else:
            check(False, "canonical outer-loop range is malformed")
        guards = numbered_values(positive_reports["canonical"], 0, "required_guard")
        check(guards == [
            "positive_m_n_k",
            "overflow_safe_shapes_elements_and_bytes",
            "nonnull_observable_pointers",
            "natural_f32_alignment",
            "output_disjoint_from_lhs_and_rhs",
            "compatible_runtime_floating_point_environment",
            "legal_cpu_implementation_available",
        ], "canonical guard contract changed or became unordered")

        first_id = report_values(positive_reports["two_functions"], 0, "site_id")
        second_id = report_values(positive_reports["two_functions"], 1, "site_id")
        check(bool(first_id and second_id and first_id != second_id),
              "two recovered functions did not receive distinct stable IDs")

        again, deterministic_report, _ = inspect(
            canonical_source, "canonical-repeat", RELAXED_FP_FLAGS
        )
        check(again.returncode == 0, f"repeat inspection failed: {again.stderr}")
        check(deterministic_report == positive_reports["canonical"],
              "recovered inspection report is not deterministic")

        rejected_cases = {
            "macro": "macro_origin",
            "template": "template_context",
            "lambda": "lambda_context",
            "optnone": "optimization_barrier",
            "volatile": "volatile_access",
            "atomic": "atomic_access",
            "observable_call": "observable_call",
            "early_return": "unsupported_control_flow",
            "header_origin": "non_main_file",
            "loop_hint": "optimization_barrier",
        }
        for name, reason in rejected_cases.items():
            source = fixtures / f"{name}.mdsl"
            before = source.read_bytes()
            completed, report, ir = inspect(source, name, RELAXED_FP_FLAGS)
            check(completed.returncode == 0,
                  f"recognized rejection {name} failed compilation: {completed.stderr}")
            check(source.read_bytes() == before,
                  f"recognized rejection {name} modified ordinary source")
            check(report_values(report, 0, "state") == ["recognized_rejected"],
                  f"{name} was not classified as recognized_rejected")
            check(reason in numbered_values(report, 0, "rejection_reason"),
                  f"{name} omitted rejection reason {reason}")
            check(report_values(report, 0, "rewrite") == ["preserve_original_cpp"],
                  f"{name} did not preserve ordinary C++")
            check(json.loads(ir.read_text()).get("operations") == [],
                  f"{name} emitted an incorrect semantic artifact")
            ordinary_object = root / f"{name}.o"
            ordinary_compile = subprocess.run(
                [str(args.clang), "-x", "c++", "-std=c++20",
                 *RELAXED_FP_FLAGS, f"-I{fixtures}", "-c", str(source),
                 "-o", str(ordinary_object)],
                cwd=repository,
                text=True,
                capture_output=True,
                check=False,
            )
            check(ordinary_compile.returncode == 0,
                  f"recognized rejection {name} did not compile to an ordinary object: "
                  f"{ordinary_compile.stderr}")
            if ordinary_compile.returncode == 0:
                ordinary_symbols = subprocess.run(
                    ["nm", "-C", str(ordinary_object)], text=True,
                    capture_output=True, check=False,
                )
                check(ordinary_symbols.returncode == 0 and
                      "matcore_runtime" not in ordinary_symbols.stdout,
                      f"recognized rejection {name} gained a recovered runtime symbol")
            else:
                check(False, f"recognized rejection {name} object was unavailable")

        for name in ("not_output_accumulate", "not_transposed_b"):
            completed, report, ir = inspect(
                fixtures / f"{name}.mdsl", name, RELAXED_FP_FLAGS
            )
            check(completed.returncode == 0,
                  f"not-recognized {name} failed compilation: {completed.stderr}")
            check(report_values(report, 0, "state") == ["not_recognized"],
                  f"{name} was recognized too broadly")
            check(report_values(report, 0, "rewrite") == ["preserve_original_cpp"],
                  f"{name} did not preserve ordinary C++")
            check(json.loads(ir.read_text()).get("operations") == [],
                  f"{name} emitted an incorrect semantic artifact")

        fp_rejections = [
            ("strict", ["-O2"], "fp_reassociation_forbidden"),
            ("optimization-zero", [*RELAXED_FP_FLAGS, "-O0"], "optimization_barrier"),
            ("contract-off", [*RELAXED_FP_FLAGS, "-ffp-contract=off"],
             "fp_contraction_forbidden"),
            ("fast-math", ["-O2", "-ffast-math"], "fp_approximation_mismatch"),
            ("denormal-flush", [*RELAXED_FP_FLAGS,
                                "-fdenormal-fp-math=preserve-sign,preserve-sign"],
             "fp_denormal_mode_mismatch"),
            ("dynamic-rounding", [*RELAXED_FP_FLAGS, "-frounding-math"],
             "fp_rounding_dynamic"),
        ]
        for name, flags, reason in fp_rejections:
            completed, report, ir = inspect(canonical_source, name, flags)
            check(completed.returncode == 0,
                  f"FP rejection {name} failed ordinary parsing: {completed.stderr}")
            check(report_values(report, 0, "state") == ["recognized_rejected"],
                  f"FP rejection {name} was not rejected")
            check(reason in numbered_values(report, 0, "rejection_reason"),
                  f"FP rejection {name} omitted {reason}")
            check(json.loads(ir.read_text()).get("operations") == [],
                  f"FP rejection {name} emitted an operation")

        for name, transform in (
            ("crlf", lambda data: data.replace(b"\n", b"\r\n")),
            ("no-final-newline", lambda data: data.rstrip(b"\n")),
        ):
            source = root / f"{name}.mdsl"
            source.write_bytes(transform(canonical_source.read_bytes()))
            completed, report, ir = inspect(source, name, RELAXED_FP_FLAGS)
            check(completed.returncode == 0,
                  f"{name} source failed inspection: {completed.stderr}")
            check(report_values(report, 0, "state") == ["recognized_guard_required"],
                  f"{name} source did not retain exact recognition")
            expected = "sha256:" + hashlib.sha256(source.read_bytes()).hexdigest()
            check(report_values(report, 0, "source_snapshot_sha256") == [expected],
                  f"{name} source snapshot digest was not exact")
            check(json.loads(ir.read_text()).get("operations") == [],
                  f"{name} source emitted an operation")

        default_ir = root / "default.json"
        default = subprocess.run(
            [
                str(args.extractor), "--input", str(canonical_source),
                "--ir-out", str(default_ir), "--", "clang++", "-std=c++20",
                *RELAXED_FP_FLAGS, str(canonical_source),
            ],
            cwd=repository,
            text=True,
            capture_output=True,
            check=False,
        )
        check(default.returncode == 0, f"default native extraction regressed: {default.stderr}")
        check(json.loads(default_ir.read_text()).get("operations") == [],
              "default extraction recognized an ordinary loop implicitly")

        bootstrap, _, _ = inspect(
            canonical_source, "bootstrap-forbidden", RELAXED_FP_FLAGS,
            frontend="ast-json-bootstrap"
        )
        check(bootstrap.returncode != 0 and "available only" in bootstrap.stderr,
              "bootstrap frontend accepted native-only recovery inspection")

        rewrite = subprocess.run(
            [
                str(args.extractor), "--input", str(canonical_source),
                "--ir-out", str(root / "rewrite.json"),
                "--inspect-recovered-gemm", str(root / "rewrite.report"),
                "--rewrite-out", str(root / "rewrite.host.cpp"),
                "--sites-out", str(root / "rewrite.sites.h"),
                "--stubs-out", str(root / "rewrite.stubs.cpp"),
                "--backend-out", str(root / "rewrite.backend.cpp"),
                "--", "clang++", "-std=c++20", *RELAXED_FP_FLAGS,
                str(canonical_source),
            ],
            cwd=repository,
            text=True,
            capture_output=True,
            check=False,
        )
        check(rewrite.returncode != 0 and "never authorizes" in rewrite.stderr,
              "recovery inspection accepted a host-rewrite request")
        check(not (root / "rewrite.host.cpp").exists(),
              "recovery inspection produced rewritten host source")

        duplicate_stdout = subprocess.run(
            [
                str(args.extractor), "--input", str(canonical_source),
                "--ir-out", "-", "--inspect-recovered-gemm", "-", "--",
                "clang++", "-std=c++20", *RELAXED_FP_FLAGS,
                str(canonical_source),
            ],
            cwd=repository,
            text=True,
            capture_output=True,
            check=False,
        )
        check(duplicate_stdout.returncode != 0 and
              "cannot both write to standard output" in duplicate_stdout.stderr,
              "IR and inspection report were allowed to corrupt shared stdout")

        object_path = root / "canonical.o"
        compile_object = subprocess.run(
            [str(args.clang), "-x", "c++", "-std=c++20", *RELAXED_FP_FLAGS,
             "-c", str(canonical_source), "-o", str(object_path)],
            cwd=repository,
            text=True,
            capture_output=True,
            check=False,
        )
        check(compile_object.returncode == 0,
              f"ordinary recovered source did not compile: {compile_object.stderr}")
        harness = root / "harness.cpp"
        harness.write_text(
            "#include <cmath>\n#include <cstddef>\n"
            "extern void ordinary_gemm(float*, const float*, const float*, "
            "std::size_t, std::size_t, std::size_t);\n"
            "int main() {\n"
            "  const float a[6] = {1, 2, 3, 4, 5, 6};\n"
            "  const float b[6] = {7, 8, 9, 10, 11, 12};\n"
            "  float c[4] = {};\n"
            "  ordinary_gemm(c, a, b, 2, 2, 3);\n"
            "  const double expected[4] = {58, 64, 139, 154};\n"
            "  for (int i = 0; i != 4; ++i)\n"
            "    if (std::fabs(static_cast<double>(c[i]) - expected[i]) > 1e-6) "
            "return 1;\n"
            "  return 0;\n}\n",
            encoding="utf-8",
        )
        executable = root / "ordinary-gemm"
        link = subprocess.run(
            [str(args.clang), "-std=c++20", str(harness), str(object_path),
             "-o", str(executable)],
            cwd=repository,
            text=True,
            capture_output=True,
            check=False,
        )
        check(link.returncode == 0, f"ordinary object did not link: {link.stderr}")
        if link.returncode == 0:
            execution = subprocess.run([str(executable)], check=False)
            check(execution.returncode == 0,
                  "ordinary recovered GEMM execution changed behavior")
        else:
            check(False, "ordinary recovered GEMM executable was unavailable")
        symbols = subprocess.run(
            ["nm", "-C", str(object_path)], text=True, capture_output=True,
            check=False,
        )
        check(symbols.returncode == 0 and "matcore_runtime" not in symbols.stdout,
              "inspection introduced a recovered runtime symbol into the object")

    if failures:
        print(f"Recovered GEMM inspection: {checks} checks, {len(failures)} failure(s)")
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    print(f"Recovered GEMM inspection: {checks} checks, 0 failures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
