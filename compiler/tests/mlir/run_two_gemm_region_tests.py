#!/usr/bin/env python3
"""Source-connected region inspection and unchanged per-call execution oracles.

These tests never execute derived region IR. The executable oracles use the
existing mdslc++ runtime-dispatch routes, including their exception ordering.
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path


def run(argv: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, cwd=cwd, text=True, capture_output=True,
                          timeout=120, check=False)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def ok(result: subprocess.CompletedProcess[str], label: str) -> None:
    require(result.returncode == 0,
            f"{label}: exit {result.returncode}\n{result.stdout}\n{result.stderr}")


PREFIX = """#include <matcore/mdsl.h>
#include <cstdio>
#include <exception>
namespace md = matcore::mdsl;
"""


def program(*, between: str = "", rhs: str = "B", d_pointer: str = "d",
            d_rows: int = 1, first_output: str = "C",
            extra_setup: str = "", expected_c: int = 6,
            expected_e: int = 24, expect_failure: bool = False) -> str:
    failure = "true" if expect_failure else "false"
    return PREFIX + f"""
int main() {{
  float a[1] = {{2}}, b[1] = {{3}}, c[1] = {{-7}};
  float d[1] = {{4}}, e[1] = {{-9}}, replacement[1] = {{5}};
  md::matrix_view A{{a, 1, 1}}, B{{b, 1, 1}}, C{{c, 1, 1}};
  md::matrix_view D{{{d_pointer}, {d_rows}, 1}}, E{{e, 1, 1}};
  {extra_setup}
  bool failed = false;
  try {{
    md::gemm(md::out({first_output}), A, {rhs});
    {between}
    md::gemm(md::out(E), C, D);
  }} catch (const std::exception &) {{ failed = true; }}
  std::printf("C=%g E=%g failed=%d\\n", c[0], e[0], failed);
  return c[0] == {expected_c}.0f && e[0] == {expected_e}.0f &&
         failed == {failure} ? 0 : 1;
}}
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extractor", type=Path, required=True)
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--clangxx", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--region-available", choices=("0", "1"), required=True)
    args = parser.parse_args()
    extractor = str(args.extractor.resolve())
    driver = str(args.driver.resolve())
    clangxx = str(args.clangxx.absolute())  # Preserve the clang++ argv[0].
    repository = args.repository_root.resolve()
    include = repository / "compiler/include"
    checks = 0  # Grouped test steps, not a count of individual assertions.

    with tempfile.TemporaryDirectory(prefix="mdslc-two-gemm-region-") as directory:
        temporary = Path(directory)

        def source(name: str, text: str) -> Path:
            path = temporary / f"{name}.mdsl"
            path.write_text(text, encoding="utf-8")
            return path

        def command(path: Path, output: Path | str,
                    extra: tuple[str, ...] = ()) -> list[str]:
            return [extractor, "--input", str(path), "--two-gemm-region-out",
                    str(output), *extra, "--", clangxx, "-std=c++20",
                    f"-I{include}", str(path)]

        positive = source("dependent", program())
        output = temporary / "region.mlir"
        if args.region_available == "0":
            result = run(command(positive, output), temporary)
            require(result.returncode != 0 and not output.exists(),
                    "unavailable region inspection must reject without artifact")
            require("requires the Linux native frontend and Matcore MLIR" in result.stderr,
                    "unavailable mode needs an actionable diagnostic")
            print("PASS: two-GEMM region unavailable boundary (2 checks)")
            return 0

        # Native extraction and derivation are repeatable without a v0 codegen
        # projection or any source rewrite. In-process tests verify the actual
        # dataflow, mathematical body and sealed source pairing.
        ok(run(command(positive, output), temporary), "region admission")
        first = output.read_text(encoding="utf-8")
        require("mdsl.region_guard" in first and "mdsl.region_commit" in first,
                "connected region must expose validation and output frontiers")
        require("analysis_only" in first and "inspection_only" in first,
                "region must retain its non-executable authority boundary")
        require("linalg.matmul" in first, "region must contain real structured computation")
        require("bufferization.to_tensor" not in first,
                "descriptor bindings may not become unjustified restricted imports")
        ok(run(command(positive, output), temporary), "repeat region admission")
        require(output.read_text(encoding="utf-8") == first,
                "same sealed source must derive deterministic region text")
        checks += 7

        cases = {
            "dependent": (positive, True),
            "input_same_descriptor": (source("input_same_descriptor", program(
                rhs="A", expected_c=4, expected_e=16)), True),
            "input_alias_bytes": (source("input_alias_bytes", program(
                extra_setup="B.data = a;", expected_c=4, expected_e=16)), True),
            "late_input_alias_first_output": (source("late_input_alias_first_output", program(
                d_pointer="c", expected_e=36)), True),
            "reference_binding": (source("reference_binding", program(
                extra_setup="md::matrix_view &Cref = C;", first_output="Cref")), True),
            "second_failure": (source("second_failure", program(
                d_rows=2, expected_e=-9, expect_failure=True)), True),
            "observer": (source("observer", program(
                between='std::printf("observed:%g\\n", c[0]);')), False),
            "descriptor_mutation": (source("descriptor_mutation", program(
                between="C.data = replacement;", expected_e=20)), False),
        }
        for name, (path, admitted) in cases.items():
            artifact = temporary / f"{name}.mlir"
            result = run(command(path, artifact), temporary)
            if admitted:
                ok(result, name + " inspection")
                require(artifact.is_file(), name + " must produce paired region")
            else:
                require(result.returncode != 0 and not artifact.exists(),
                        name + " must not admit across a host boundary")
            checks += 1
            # Both established routes must retain the original call sequence.
            for pipeline in ("capture-v0", "matcore-mlir"):
                executable = temporary / f"{name}-{pipeline}"
                ok(run([driver, "--matcore-target=cpu",
                        f"--semantic-pipeline={pipeline}", "-std=c++20",
                        str(path), "-o", str(executable)], temporary),
                   name + " compile " + pipeline)
                result = run([str(executable)], temporary)
                ok(result, name + " execution " + pipeline)
                if name == "second_failure":
                    require("C=6 E=-9 failed=1" in result.stdout,
                            "second failure must preserve the first successful write")
                if name == "observer":
                    require("observed:6" in result.stdout,
                            "host observer must see the first GEMM output")
                checks += 2

        # Competing definitions remain valid C++, but cannot authenticate a
        # mathematical region. In particular a definition after the call must
        # not escape the final redeclaration-chain audit.
        definition = """
namespace matcore::mdsl {
void gemm(out_arg out, const matrix_view &, const matrix_view &, policy) {
  out.value->data[0] = 123.0f;
}
}
"""
        for label, text in (("definition_before", PREFIX + definition + program()[len(PREFIX):]),
                            ("definition_after", program() + definition)):
            path = source(label, text)
            ok(run([clangxx, "-x", "c++", "-std=c++20", "-fsyntax-only",
                    f"-I{include}", str(path)], temporary), label + " valid C++")
            artifact = temporary / f"{label}.mlir"
            result = run(command(path, artifact), temporary)
            require(result.returncode != 0 and not artifact.exists(),
                    "competing intrinsic definition must reject region admission")
            checks += 2

        # No region invocation is allowed to manufacture source or execution
        # artifacts. Reject before publishing even a valid requested region.
        for index, extra in enumerate((
            ("--frontend=ast-json-bootstrap",),
            ("--semantic-pipeline=matcore-mlir",),
            ("--ir-out", str(temporary / "forbidden.json")),
            ("--rewrite-out", str(temporary / "forbidden.cpp")),
            ("--verify-ir", str(temporary / "anything.json")),
        )):
            artifact = temporary / f"forbidden-{index}.mlir"
            result = run(command(positive, artifact, extra), temporary)
            require(result.returncode != 0 and not artifact.exists(),
                    f"incompatible region options accepted: {extra}")
            checks += 1
        require(not (temporary / "forbidden.json").exists() and
                not (temporary / "forbidden.cpp").exists(),
                "inspection must not publish forbidden execution artifacts")
        checks += 1

        dependency = temporary / "context.h"
        dependency.write_text("inline constexpr int region_context = 1;\n", encoding="utf-8")
        with_dependency = source("with_dependency", '#include "context.h"\n' + program())
        dependency_bytes = dependency.read_bytes()
        for alias in (dependency, temporary / "context-hardlink.h",
                      temporary / "context-symlink.h"):
            if alias.name == "context-hardlink.h":
                alias.hardlink_to(dependency)
            elif alias.name == "context-symlink.h":
                alias.symlink_to(dependency)
            result = run(command(with_dependency, alias), temporary)
            require(result.returncode != 0 and dependency.read_bytes() == dependency_bytes,
                    "region output must not overwrite any captured source dependency")
            checks += 1

        before = positive.read_bytes()
        result = run(command(positive, positive), temporary)
        require(result.returncode != 0 and positive.read_bytes() == before,
                "region output must not overwrite source")
        checks += 1
        sentinel = temporary / "unchanged.mlir"
        sentinel.write_text("keep user state\n", encoding="utf-8")
        result = run(command(cases["observer"][0], sentinel), temporary)
        require(result.returncode != 0 and sentinel.read_text() == "keep user state\n",
                "failed admission must not replace an existing output")
        checks += 1

    print(f"PASS: two-GEMM inspection and unchanged execution ({checks} test steps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
