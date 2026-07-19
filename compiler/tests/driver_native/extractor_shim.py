#!/usr/bin/env python3

import json
import os
from pathlib import Path
import sys


def option_value(arguments: list[str], option: str) -> str | None:
    for index, argument in enumerate(arguments):
        if argument == option and index + 1 < len(arguments):
            return arguments[index + 1]
    return None


def main() -> int:
    arguments = sys.argv[1:]
    log_path = os.environ.get("MDSLC_DRIVER_TEST_LOG")
    if not log_path:
        print("extractor shim: MDSLC_DRIVER_TEST_LOG is required", file=sys.stderr)
        return 90
    with Path(log_path).open("a", encoding="utf-8") as log:
        log.write(json.dumps(arguments) + "\n")

    frontend = next(
        (argument.removeprefix("--frontend=") for argument in arguments
         if argument.startswith("--frontend=")),
        "missing",
    )
    if os.environ.get("MDSLC_DRIVER_TEST_REJECT_NATIVE") == "1":
        return 86 if frontend == "native" else 0

    if (
        os.environ.get("MDSLC_DRIVER_TEST_MUTATE") == "1"
        or os.environ.get("MDSLC_DRIVER_TEST_SUCCESS") == "1"
    ):
        generated_contents = {
            "--ir-out": "{}\n",
            "--rewrite-out": (
                'extern "C" int mdslc_driver_fixture_status();\n'
                "int main() {\n"
                "  return mdslc_driver_fixture_status();\n"
                "}\n"
            ),
            "--sites-out": "#pragma once\n",
            "--stubs-out": (
                "#include <matcore/runtime_c.h>\n"
                'extern "C" int mdslc_driver_fixture_status() {\n'
                "  const auto status = matcore_runtime_gemm_f32_v0(\n"
                "      nullptr, nullptr, nullptr, nullptr);\n"
                "  return status.code == MATCORE_STATUS_INVALID_ARGUMENT_V0 ? 0 : 1;\n"
                "}\n"
            ),
            "--backend-out": "// generated backend fixture\n",
        }
        for option, contents in generated_contents.items():
            output = option_value(arguments, option)
            if output:
                Path(output).write_text(contents, encoding="utf-8")

    if os.environ.get("MDSLC_DRIVER_TEST_MUTATE") == "1":
        source = option_value(arguments, "--input")
        if not source:
            print("extractor shim: --input is required", file=sys.stderr)
            return 91
        with Path(source).open("a", encoding="utf-8") as stream:
            stream.write("\n// extractor mutation probe\n")
        return 0

    if os.environ.get("MDSLC_DRIVER_TEST_SUCCESS") == "1":
        return 0

    return int(os.environ.get("MDSLC_DRIVER_TEST_EXIT", "73"))


if __name__ == "__main__":
    raise SystemExit(main())
