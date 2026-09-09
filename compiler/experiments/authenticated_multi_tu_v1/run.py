#!/usr/bin/env python3
"""Source-only multi-TU execution and pre-link rejection research evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rejected_before_link(result, diagnostic, artifact_exists):
    # The driver catches checked refusals and returns exactly 1. A diagnostic
    # followed by an abnormal cleanup exit must never satisfy a negative case.
    crash_markers = ("addresssanitizer", "undefinedbehaviorsanitizer", "runtime error:",
                     "segmentation fault", "deadlysignal", "core dumped", "aborted")
    text = (result.stdout + result.stderr).lower()
    return (result.returncode == 1 and diagnostic in result.stderr and not artifact_exists and
            not any(marker in result.stdout for marker in
                ("VALIDATED program", "CROSS_TU_LINK", "NATIVE_LINK", "PUBLISHED")) and
            not any(marker in text for marker in crash_markers))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--expected-driver-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    driver = args.driver.resolve(strict=True)
    if digest(driver) != args.expected_driver_sha256:
        parser.error("research driver differs from supplied identity")
    output = args.output.absolute()
    if output.exists():
        parser.error("output must be new")
    output.mkdir(parents=True)
    fixtures = Path(__file__).resolve().parent / "fixtures"
    hashes = {str(path): digest(path) for path in fixtures.iterdir() if path.is_file()}
    hashes[str(driver)] = digest(driver)
    hashes[str(Path(__file__).resolve())] = digest(__file__)
    report = {"schema": "authenticated-multi-tu-research-v1", "status": "running",
              "source_and_driver_sha256": hashes, "commands": [], "cases": [],
              "sanitizer_environment": {name: os.environ.get(name) for name in
                  ("ASAN_OPTIONS", "UBSAN_OPTIONS", "DEBUGINFOD_URLS")},
              "authority": "research_source_only_not_installed_product_support",
              "timing": "not_requested"}
    def save():
        (output / "evidence.json").write_text(json.dumps(report, indent=2) + "\n")
    def unchanged():
        for path, sha in hashes.items():
            if digest(path) != sha:
                raise RuntimeError("frozen source or driver changed: " + path)
    def run(command, cwd):
        unchanged()
        result = subprocess.run([str(x) for x in command], cwd=cwd, text=True,
                                capture_output=True, timeout=180)
        report["commands"].append({"argv": [str(x) for x in command], "cwd": str(cwd),
                                  "exit": result.returncode, "stdout": result.stdout,
                                  "stderr": result.stderr})
        save()
        unchanged()
        return result
    cases = [
      ("positive", "main.cpp", None),
      ("packed", "main_packed.cpp", "ABI: foreign region declaration differs from sealed producer"),
      ("pure", "main_pure.cpp", "foreign region declaration carries unadmitted function attributes"),
      ("const", "main_const.cpp", "foreign region declaration carries unadmitted function attributes"),
      ("asm_pure", "main_asm_pure.cpp", "foreign region source spelling differs from its issued symbol"),
      ("c_pure", "main_c_pure.cpp", "foreign region source spelling differs from its issued symbol"),
      ("weak", "main_weak.cpp", "OWNERSHIP: competing region definition before linker resolution"),
      ("comdat", "main_comdat.cpp", "OWNERSHIP: competing region definition before linker resolution"),
      ("inline_unused", "main_inline_unused.cpp", "OWNERSHIP: foreign source defines an issued region"),
      ("retired", "main_retired.cpp", "RETIREMENT: foreign TU uses a retired source-only Value helper"),
      ("duplicate_main", "main.cpp", "ENTRY: program requires exactly one ordinary C++ main definition"),
      ("entry_alias", "main.cpp", "ENTRY: foreign source symbol competes with main"),
      ("missing_main", None, "ENTRY: program requires exactly one ordinary C++ main definition"),
      ("opaque_value", "main.cpp", "unknown host call or unadmitted helper"),
    ]
    try:
        for name, host, rejection in cases:
            work = output / name
            work.mkdir()
            executable = work / "program"
            command = [driver]
            if host:
                command += ["--host", fixtures / host]
            if name == "duplicate_main":
                command += ["--host", fixtures / "extra_main.cpp"]
            if name == "entry_alias":
                command += ["--host", fixtures / "extra_main_asm.cpp"]
            command += ["--region", fixtures / "first.mdsl", "first", "--region",
                        fixtures / ("second_opaque.mdsl" if name == "opaque_value" else "second.mdsl"),
                        "second", "-o", executable]
            result = run(command, work)
            if rejection:
                if not rejected_before_link(result, rejection, executable.exists()):
                    raise RuntimeError(f"{name}: wrong rejection phase or diagnostic: {result.stderr}")
                record = {"name": name, "pass": True, "rejected_before_cross_tu_link": True,
                          "expected_diagnostic": rejection}
                if name in ("weak", "comdat"):
                    linkage = re.search(r"linkage=(\d+) comdat=([01])", result.stderr)
                    if not linkage or (name == "weak" and linkage.groups() != ("4", "0")) or (
                            name == "comdat" and linkage.groups() != ("3", "1")):
                        raise RuntimeError(name + ": negative fixture did not emit the intended LLVM linkage")
                    record["actual_llvm_linkage"] = int(linkage[1])
                    record["actual_comdat"] = bool(int(linkage[2]))
            else:
                if result.returncode or not executable.is_file() or "VALIDATED program" not in result.stdout:
                    raise RuntimeError("actual multi-TU source compilation failed: " + result.stderr)
                hashes[str(executable)] = digest(executable)
                execution = run([executable], work)
                if execution.returncode or execution.stdout.strip() != "PASS authenticated_multi_tu 37 checks" or execution.stderr:
                    raise RuntimeError("multi-TU executable oracle failed: " + execution.stdout + execution.stderr)
                dynamic = run(["readelf", "-d", executable], work)
                needed = re.findall(r"\(NEEDED\).*Shared library: \[([^\]]+)\]", dynamic.stdout)
                if dynamic.returncode or needed.count("libmatcore_closed_candidates_isolated_v1.so") != 1 or needed.count("libmatcore_runtime.so.0") != 1:
                    raise RuntimeError("missing direct canonical/private runtime dependencies")
                record = {"name": name, "pass": True, "actual_checks": 37,
                          "dt_needed": needed,
                          "executable_sha256": digest(executable)}
            report["cases"].append(record)
            save()
            print("PASS", name)
        report["status"] = "PASS"
    except Exception as error:
        report["status"] = "FAIL"
        report["failure"] = str(error)
        save()
        raise
    save()
    print("PASS authenticated multi-TU research", len(report["cases"]), "cases")


if __name__ == "__main__":
    main()
