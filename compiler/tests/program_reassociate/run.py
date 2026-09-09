#!/usr/bin/env python3
"""Three actual source TUs; no missing-policy, crash, or artifact-as-authority fallback."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def output(policy, checks, failures=0):
    return (f"Program reassociate composition: {policy}; {checks} checks; "
            f"{failures} failures; host trace 123\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--expected-driver-sha256")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    driver = args.driver.absolute()
    fixture = Path(__file__).resolve().parent
    directory = (args.output.absolute() if args.output else
                 Path(tempfile.mkdtemp(prefix="mdslc-program-reassociate-")) / "matrix")
    directory.mkdir(parents=True, exist_ok=False)
    paths = [fixture / name for name in ("api.h", "first.mdsl", "second.mdsl", "host.cpp", "run.py")]
    paths += [driver, driver.parent.parent / "lib/libmatcore_closed_candidates_isolated_v1.so",
              driver.parent.parent / "lib/libmatcore_runtime.so"]
    pins = {str(path): digest(path) for path in paths}
    if args.expected_driver_sha256 and pins[str(driver)] != args.expected_driver_sha256:
        parser.error("supplied driver SHA256 differs from actual artifact")
    report = {"schema": "program-reassociate-composition-v1", "status": "running",
              "inputs": dict(pins), "commands": [], "executables": [], "timing": "not requested"}
    environment = os.environ.copy()
    for name in ("LD_LIBRARY_PATH", "LD_PRELOAD", "LD_AUDIT"):
        environment.pop(name, None)
    environment["DEBUGINFOD_URLS"] = ""
    environment["ASAN_OPTIONS"] = "detect_leaks=1:halt_on_error=1:strict_string_checks=1:check_initialization_order=1"
    environment["UBSAN_OPTIONS"] = "halt_on_error=1:print_stacktrace=1"
    def save():
        (directory / "evidence.json").write_text(json.dumps(report, indent=2) + "\n")
    def unchanged():
        for path, expected in pins.items():
            if digest(path) != expected:
                raise RuntimeError("frozen input or produced executable changed: " + path)
    def command(argv, *, expected=0, produces=None, skip=False):
        unchanged()
        if produces is not None and produces.exists():
            raise RuntimeError("refusing existing output")
        result = subprocess.run([str(arg) for arg in argv], cwd=directory,
                                env=environment, text=True, capture_output=True, timeout=240)
        report["commands"].append({"argv": [str(arg) for arg in argv], "exit": result.returncode,
                                   "stdout": result.stdout, "stderr": result.stderr})
        save()
        unchanged()
        if skip and result.returncode == 77:
            if result.stdout != "SKIP program_reassociate: AVX2/FMA unavailable\n" or result.stderr:
                raise RuntimeError("wrong or diagnostic-bearing hardware skip")
            report["status"] = "SKIP"
            save()
            print("SKIP program_reassociate: AVX2/FMA unavailable")
            raise SystemExit(77)
        if result.returncode != expected:
            raise RuntimeError(f"wrong exit {result.returncode}; expected {expected}: {result.stderr}")
        if expected == 0 and result.stderr:
            raise RuntimeError("positive command emitted diagnostics: " + result.stderr)
        if produces is not None:
            pins[str(produces)] = digest(produces)
            report["executables"].append({"path": str(produces), "sha256": digest(produces)})
            save()
        return result
    try:
        for policy, count in (("generated-strict", 164), ("generated-reassociate", 152)):
            executable = directory / policy
            compiled = command([
                driver, "--program", "--host", fixture / "host.cpp",
                "--region", fixture / "first.mdsl", "permissive_first",
                "--region", fixture / "second.mdsl", "strict_second",
                "--candidate", policy, "-o", executable], produces=executable)
            if "VALIDATED program ownership and ABI before cross-TU link/optimization\n" not in compiled.stdout:
                raise RuntimeError("missing real multi-source ownership gate")
            executed = command([executable, policy], skip=(policy == "generated-reassociate"))
            if executed.stdout != output(policy, count):
                raise RuntimeError("source numerical/effect oracle failed: " + executed.stdout)
            elf = command(["readelf", "-dW", executable])
            needed = re.findall(r"\(NEEDED\).*\[([^]]+)\]", elf.stdout)
            if needed.count("libmatcore_closed_candidates_isolated_v1.so") != 1 or needed.count("libmatcore_runtime.so.0") != 1:
                raise RuntimeError("source executable omitted direct private/canonical runtime ownership")
        # Same frozen binaries with deliberately wrong numerical/refusal oracles:
        # accepting a strict fallback under the new name cannot satisfy the test.
        for actual, claimed, count, failures in (
                ("generated-strict", "generated-reassociate", 152, 143),
                ("generated-reassociate", "generated-strict", 164, 155)):
            result = command([directory / actual, claimed], expected=1)
            if result.stdout != output(claimed, count, failures):
                raise RuntimeError("wrong arithmetic/refusal counteroracle identity")
            lines = result.stderr.splitlines()
            allowed = {
                "FAIL: actual first full-tile arithmetic", "FAIL: first observation exact math",
                "FAIL: first observation survives first Result and host mutation",
                "FAIL: first owning observation survives both Result lifetimes and final mutations",
                "FAIL: later strict GEMM refuses forced reassociate without fallback",
                "FAIL: second region retains exact pre-GEMM prefix",
                "FAIL: later failure names physical second source GEMM",
                "FAIL: strict refusal never publishes second output",
                "FAIL: generated strict accepts both source numerical profiles",
                "FAIL: strict control second region completes all effects",
                "FAIL: second strict observation has rectangular dimensions",
                "FAIL: second GEMM reads ordinary host mutation",
                "FAIL: second strict owning observation exact math"}
            if len(lines) != failures or not set(lines) <= allowed:
                raise RuntimeError("unexpected crash/sanitizer/non-oracle negative diagnostics")
        unchanged()
        report["status"] = "PASS"
        report["positive_checks"] = 316
        report["negative_controls"] = 2
        save()
    except Exception as error:
        report["status"] = "FAIL"
        report["failure"] = str(error)
        save()
        raise
    print("PASS program_reassociate: 2 policies; 316 checks; 2 negative controls")


if __name__ == "__main__":
    main()
