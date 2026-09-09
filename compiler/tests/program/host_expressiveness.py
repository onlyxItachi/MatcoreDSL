#!/usr/bin/env python3
"""Actual header-free utilities and TU-local ordinary helper execution."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
from run import digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--expected-driver-sha256")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    driver = args.driver.resolve(strict=True)
    if args.expected_driver_sha256 and digest(driver) != args.expected_driver_sha256:
        parser.error("driver identity differs")
    output = args.output.absolute() if args.output else Path(tempfile.mkdtemp(prefix="mdslc-program-hosts-")) / "run"
    if output.exists():
        parser.error("output must be new")
    output.mkdir(parents=True)
    source = Path(__file__).resolve().parent
    fixtures = source / "host_expressiveness"
    hashes = {str(path): digest(path) for folder in (fixtures, source / "fixtures")
              for path in folder.iterdir() if path.is_file()}
    for path in (driver, Path(__file__).resolve(), source / "run.py"):
        hashes[str(path)] = digest(path)
    report = {"schema": "authenticated-multi-tu-host-expressiveness-v1", "status": "running",
              "source_and_driver_sha256": hashes, "commands": [], "cases": [],
              "driver_source_change": False, "timing": "not_requested",
              "sanitizer_environment": {name: os.environ.get(name) for name in
                  ("ASAN_OPTIONS", "UBSAN_OPTIONS", "DEBUGINFOD_URLS")}}
    def save():
        (output / "evidence.json").write_text(json.dumps(report, indent=2) + "\n")
    def unchanged():
        if any(digest(path) != sha for path, sha in hashes.items()):
            raise RuntimeError("frozen experiment source or binary changed")
    def run(command, cwd):
        unchanged()
        result = subprocess.run([str(item) for item in command], cwd=cwd, text=True,
                                capture_output=True, timeout=180)
        report["commands"].append({"argv": [str(item) for item in command], "cwd": str(cwd),
                                  "exit": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
        save(); unchanged()
        return result
    cases = [
        ("header_present_control", ["utility_main.cpp", "helper_with_header.cpp"],
         "PASS ordinary_host_utility 11 checks"),
        ("header_free_utility", ["utility_main.cpp", "helper_header_free.cpp"], "PASS ordinary_host_utility 11 checks"),
        ("repeated_internal_linkage", ["static_main.cpp", "static_a.cpp", "static_b.cpp"],
         "PASS ordinary_internal_linkage 12 checks"),
        ("ordinary_overload", ["overload_main.cpp", "overload.cpp"],
         "PASS ordinary_overload 11 checks"),
    ]
    try:
        for name, hosts, expected in cases:
            work = output / name; work.mkdir()
            executable = work / "program"
            command = [driver, "--program", "--candidate", "generated-strict"]
            for host in hosts: command += ["--host", fixtures / host]
            command += ["--region", source / "fixtures/first.mdsl", "first", "--region",
                        source / "fixtures/second.mdsl", "second", "-o", executable]
            compiled = run(command, work)
            if compiled.returncode != 0 or not executable.is_file():
                raise RuntimeError("source execution control failed: " + compiled.stderr)
            hashes[str(executable)] = digest(executable)
            actual = run([executable], work)
            if actual.returncode != 0 or actual.stdout.strip() != expected or actual.stderr:
                raise RuntimeError("wrong ordinary-host execution: " + actual.stdout + actual.stderr)
            report["cases"].append({"name": name, "observed": "EXECUTION_PASS",
                                    "expected_output": expected, "executable_sha256": digest(executable)})
            save(); print(name, report["cases"][-1]["observed"])
        report["status"] = "PASS"
    except Exception as error:
        report["status"] = "FAIL"; report["failure"] = str(error); save(); raise
    save()


if __name__ == "__main__":
    main()
