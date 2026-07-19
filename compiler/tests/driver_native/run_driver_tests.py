#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def run(
    command: list[str], environment: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=False,
        text=True,
        capture_output=True,
        env=environment,
    )


def read_log(log: Path) -> list[list[str]]:
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]


def frontend_from(arguments: list[str]) -> str:
    frontends = [
        argument.removeprefix("--frontend=")
        for argument in arguments
        if argument.startswith("--frontend=")
    ]
    if len(frontends) != 1:
        raise RuntimeError(f"expected exactly one extractor frontend flag: {arguments}")
    return frontends[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", required=True)
    parser.add_argument("--bindir", default="bin")
    parser.add_argument("--includedir", default="include")
    parser.add_argument("--libdir", default="lib")
    parser.add_argument(
        "--test-prefix-override",
        choices=("disabled", "enabled"),
        default="disabled",
    )
    args = parser.parse_args()

    driver_source = Path(args.driver).resolve()
    if not driver_source.is_file():
        raise RuntimeError(f"driver does not exist: {driver_source}")
    shim_source = Path(__file__).with_name("extractor_shim.py").resolve()

    with tempfile.TemporaryDirectory(prefix="matcore-driver-native-") as temporary:
        root = Path(temporary)
        prefix = root / "relocated prefix,with comma"
        binary_directory = prefix / args.bindir
        include_root = prefix / args.includedir
        include_directory = include_root / "matcore"
        runtime_directory = prefix / args.libdir
        binary_directory.mkdir(parents=True)
        include_directory.mkdir(parents=True)
        runtime_directory.mkdir(parents=True)

        driver = binary_directory / "mdslc++"
        extractor = binary_directory / "matcore-extract"
        shutil.copy2(driver_source, driver)
        shutil.copy2(shim_source, extractor)
        extractor.chmod(0o755)
        (include_directory / "mdsl.h").write_text("#pragma once\n", encoding="utf-8")
        (include_directory / "runtime_c.h").write_text(
            "#pragma once\n", encoding="utf-8"
        )
        (runtime_directory / "libmatcore_runtime.so").write_bytes(b"fixture\n")

        source_directory = root / "source with spaces"
        output_directory = root / "output with spaces"
        source_directory.mkdir()
        output_directory.mkdir()
        source = source_directory / "driver test.mdsl"
        pristine_source = "int main() { return 0; }\n"
        source.write_text(pristine_source, encoding="utf-8")
        output = output_directory / "driver test.o"
        log = root / "extractor-arguments.jsonl"
        shell_marker = root / "must-not-exist"

        base_environment = os.environ.copy()
        base_environment["MDSLC_DRIVER_TEST_LOG"] = str(log)

        def invoke(
            *,
            frontend: str | None = None,
            environment_updates: dict[str, str] | None = None,
            additional_wrapper_arguments: list[str] | None = None,
        ) -> subprocess.CompletedProcess[str]:
            source.write_text(pristine_source, encoding="utf-8")
            output.unlink(missing_ok=True)
            log.unlink(missing_ok=True)
            command = [str(driver), "--matcore-target=cpu"]
            if frontend is not None:
                command.append(f"--frontend={frontend}")
            if additional_wrapper_arguments:
                command.extend(additional_wrapper_arguments)
            command.extend(
                [
                    "-std=c++20",
                    "-DDRIVER_VALUE=alpha beta",
                    f"-DNO_SHELL=$(touch {shell_marker})",
                    "-c",
                    str(source),
                    "-o",
                    str(output),
                ]
            )
            environment = base_environment.copy()
            environment["MDSLC_DRIVER_TEST_EXIT"] = "73"
            environment.pop("MDSLC_DRIVER_TEST_REJECT_NATIVE", None)
            environment.pop("MDSLC_DRIVER_TEST_MUTATE", None)
            if environment_updates:
                environment.update(environment_updates)
            return run(command, environment)

        default_result = invoke()
        default_log = read_log(log)
        if default_result.returncode != 73 or len(default_log) != 1:
            raise RuntimeError(
                "default frontend invocation did not propagate one extractor failure:\n"
                f"{default_result.stderr}\n{default_log}"
            )
        if frontend_from(default_log[0]) != "native":
            raise RuntimeError(f"driver default was not native: {default_log[0]}")

        explicit_native = invoke(frontend="native")
        explicit_native_log = read_log(log)
        if explicit_native.returncode != 73 or len(explicit_native_log) != 1:
            raise RuntimeError("explicit native frontend was not invoked exactly once")
        if frontend_from(explicit_native_log[0]) != "native":
            raise RuntimeError("explicit native frontend was not forwarded")

        bootstrap = invoke(frontend="ast-json-bootstrap")
        bootstrap_log = read_log(log)
        if bootstrap.returncode != 73 or len(bootstrap_log) != 1:
            raise RuntimeError("bootstrap frontend was not invoked exactly once")
        if frontend_from(bootstrap_log[0]) != "ast-json-bootstrap":
            raise RuntimeError("explicit bootstrap frontend was not forwarded")

        rejected_native = invoke(
            environment_updates={"MDSLC_DRIVER_TEST_REJECT_NATIVE": "1"}
        )
        rejected_log = read_log(log)
        if rejected_native.returncode != 86 or len(rejected_log) != 1:
            raise RuntimeError(
                "native failure was retried or its status was hidden:\n"
                f"{rejected_native.stderr}\n{rejected_log}"
            )
        if frontend_from(rejected_log[0]) != "native":
            raise RuntimeError("native failure silently selected another frontend")

        unknown = invoke(frontend="unknown")
        if unknown.returncode != 2 or read_log(log):
            raise RuntimeError("unknown frontend reached the extractor")
        if "no fallback" not in unknown.stderr:
            raise RuntimeError(
                f"unknown frontend diagnostic is unclear: {unknown.stderr}"
            )

        forwarded = default_log[0]
        separator = forwarded.index("--")
        compiler_arguments = forwarded[separator + 1 :]
        for expected in (
            "-std=c++20",
            "-DDRIVER_VALUE=alpha beta",
            f"-DNO_SHELL=$(touch {shell_marker})",
            str(source),
        ):
            if compiler_arguments.count(expected) != 1:
                raise RuntimeError(
                    f"compiler argument was not forwarded exactly once: {expected}: "
                    f"{compiler_arguments}"
                )
        if shell_marker.exists():
            raise RuntimeError("driver executed shell syntax from a compiler argument")
        if f"-I{include_root}" not in compiler_arguments:
            raise RuntimeError(
                f"relocated trusted include was not forwarded: {compiler_arguments}"
            )

        mutation = invoke(
            environment_updates={
                "MDSLC_DRIVER_TEST_EXIT": "0",
                "MDSLC_DRIVER_TEST_MUTATE": "1",
            }
        )
        if mutation.returncode == 0 or output.exists():
            raise RuntimeError("source mutation produced an output object")
        if (
            "source changed during compilation after frontend extraction"
            not in mutation.stderr
        ):
            raise RuntimeError(
                f"source mutation diagnostic is not actionable:\n{mutation.stderr}"
            )

        override = invoke(
            additional_wrapper_arguments=[
                f"--tool-prefix-for-testing={prefix}"
            ]
        )
        if args.test_prefix_override == "disabled":
            if (
                override.returncode != 2
                or "production driver build" not in override.stderr
            ):
                raise RuntimeError(
                    "production driver exposed the test-only prefix override:\n"
                    f"{override.stderr}"
                )
        else:
            override_log = read_log(log)
            if override.returncode != 73 or len(override_log) != 1:
                raise RuntimeError(
                    "test-enabled prefix override did not select the fixture prefix:\n"
                    f"{override.stderr}\n{override_log}"
                )

    print(
        "native driver: default/explicit/bootstrap/no-fallback/argv/relocation/"
        "source-snapshot PASS"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
