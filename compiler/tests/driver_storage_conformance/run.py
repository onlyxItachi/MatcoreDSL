#!/usr/bin/env python3
"""Public-driver semantic checks; no build, timing, or implementation-identity claim."""
import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import re
import subprocess
import tempfile


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dynamic_dependencies(text):
    needed = re.findall(r"\(NEEDED\).*?Shared library:\s*\[([^\]]+)\]", text)
    required = {"libmatcore_closed_candidates_isolated_v1.so", "libmatcore_runtime.so.0"}
    missing = required.difference(needed)
    if missing:
        raise ValueError("missing required ELF NEEDED entries: " + ", ".join(sorted(missing)))
    return {"needed": needed, "required": sorted(required)}


def relocatable_header(text, machine="Advanced Micro Devices X86-64"):
    fields = {}
    for line in text.splitlines():
        key, separator, value = line.partition(":")
        if separator:
            fields[key.strip()] = value.strip()
    required = {"Class": "ELF64", "Type": "REL (Relocatable file)",
                "Machine": machine}
    for key, expected in required.items():
        if fields.get(key) != expected:
            raise ValueError(f"expected ELF {key}={expected!r}, got {fields.get(key)!r}")
    return required


def reassociate_sources(source, target):
    """Copy the same source/host oracle, changing only both per-GEMM profiles."""
    target.mkdir()
    for carry in ("lhs", "rhs"):
        original = (source / f"{carry}.mdsl").read_text()
        spelling = "mdsl::Numerics::strict_f32"
        if original.count(spelling) != 2:
            raise ValueError(f"{carry}.mdsl must contain exactly two per-operation strict profiles")
        changed = original.replace(spelling, "mdsl::Numerics::reassociate_f32")
        if len(changed.splitlines()) != len(original.splitlines()):
            raise ValueError("reassociation fixture must preserve original operation line locations")
        (target / f"{carry}.mdsl").write_text(changed)
    header = "storage_conformance_host.h"
    (target / header).write_bytes((source / header).read_bytes())
    return {str(path): digest(path) for path in sorted(target.iterdir())}


def ordinary_link_command(clangxx, obj, libraries, library_dir, output, sanitized):
    sanitizer_flags = ["-fsanitize=address,undefined"] if sanitized else []
    return [clangxx, obj, *libraries, *sanitizer_flags, "-pthread",
            "-Xlinker", "-rpath", "-Xlinker", library_dir, "-o", output]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("driver", type=Path)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--clangxx", type=Path, default=Path("/usr/bin/clang++-21"))
    parser.add_argument("--library-directory", type=Path)
    parser.add_argument("--sanitized", action="store_true",
                        help="require an ASan+UBSan driver and instrument the ordinary final link")
    parser.add_argument("--has-openblas", action="store_true",
                        help="also execute reassociate_f32 cases with requested openblas policy")
    args = parser.parse_args()
    architecture = platform.machine()
    if architecture not in ("x86_64", "aarch64"):
        parser.error("storage conformance requires native Linux x86_64 or aarch64")
    driver = args.driver.resolve(strict=True)
    initial_hash = digest(driver)
    if args.expected_sha256 and initial_hash != args.expected_sha256:
        raise SystemExit("frozen driver SHA256 mismatch before testing")
    output = args.output or Path(tempfile.mkdtemp(prefix="mdslc-storage-conformance-"))
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise SystemExit("output directory must be empty (no overwrite)")
    source = Path(__file__).resolve().parent
    library_dir = (args.library_directory or driver.parent.parent / "lib").resolve()
    libraries = [library_dir / "libmatcore_closed_candidates_isolated_v1.so",
                 library_dir / "libmatcore_runtime.so"]
    library_hashes = {str(path): digest(path) for path in libraries}
    report = {"driver": str(driver), "driver_sha256": initial_hash, "commands": [],
              "status": "running", "validation_checks": [], "lanes": [],
              "sanitized_requested": args.sanitized,
              "openblas_lane": "requested" if args.has_openblas else "not_requested_not_tested"}
    report["library_sha256"] = library_hashes
    report["fixture_sha256"] = {
        str(path): digest(path) for path in sorted(source.iterdir())
        if path.suffix in (".mdsl", ".h")}
    report_path = output / "evidence.json"

    def save():
        report_path.write_text(json.dumps(report, indent=2) + "\n")

    def fail(message):
        report["status"] = "fail"
        report["failure"] = message
        save()
        raise SystemExit(message)

    def validate(name, function):
        try:
            details = function()
        except ValueError as error:
            report["validation_checks"].append({"name": name, "status": "fail", "detail": str(error)})
            fail(f"{name}: {error}")
        report["validation_checks"].append({"name": name, "status": "pass", "details": details})
        save()

    def unchanged():
        if digest(driver) != initial_hash:
            fail("frozen driver changed during testing")
        if any(digest(Path(path)) != sha for path, sha in library_hashes.items()):
            fail("frozen runtime/candidate library changed during testing")

    def run(command, expected_success=True):
        unchanged()
        result = subprocess.run([str(item) for item in command], text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                timeout=180, check=False, env={**os.environ, "LC_ALL": "C"})
        record = {"argv": [str(item) for item in command], "returncode": result.returncode,
                  "stdout": result.stdout, "stderr": result.stderr}
        report["commands"].append(record)
        save()
        if (result.returncode == 0) != expected_success:
            fail(json.dumps(record, indent=2))
        return result

    version = run([driver, "--version"])
    report["driver_version"] = version.stdout.strip()
    if ("ASan+UBSan" in version.stdout) != args.sanitized:
        fail("driver sanitizer profile does not match --sanitized selection")
    reassociated = output / "reassociate-source"
    report["reassociate_fixture_sha256"] = reassociate_sources(source, reassociated)
    save()

    def inspect_dynamic(binary):
        dynamic = run(["readelf", "-d", binary])
        validate(f"{binary.name}: shared dependencies", lambda: dynamic_dependencies(dynamic.stdout))

    def execute_lane(carry, candidate, numerics, fixture_source):
        binary = output / f"{carry}-{numerics}-{candidate}"
        run([driver, fixture_source / f"{carry}.mdsl", "--region", "pipeline",
             "--candidate", candidate, "-o", binary])
        reject_policy = candidate == "existing-native" and numerics == "strict_f32"
        unavailable_policy = (architecture == "aarch64" and candidate == "existing-native"
                              and numerics == "reassociate_f32")
        flags = (["--expect-policy-rejection"] if reject_policy else
                 ["--expect-policy-unavailable"] if unavailable_policy else [])
        execution = run([binary, *flags])
        print(f"{carry}/{numerics}/{candidate}: {execution.stdout.splitlines()[-1]}", flush=True)
        inspect_dynamic(binary)
        report["lanes"].append({"carry": carry, "numerics": numerics,
                                "requested_candidate": candidate, "status": "pass",
                                "expected_policy_rejection": reject_policy,
                                "expected_policy_unavailable": unavailable_policy,
                                "summary": execution.stdout.splitlines()[-1]})
        save()

    for carry in ("lhs", "rhs"):
        for candidate in ("native-strict", "generated-strict", "automatic", "existing-native"):
            execute_lane(carry, candidate, "strict_f32", source)
        execute_lane(carry, "existing-native", "reassociate_f32", reassociated)
        if args.has_openblas:
            execute_lane(carry, "openblas", "reassociate_f32", reassociated)
    # Also exercise the ordinary relocatable-object/final-link route.
    obj = output / "lhs-generated-strict.o"
    run([driver, source / "lhs.mdsl", "--region", "pipeline", "--candidate",
         "generated-strict", "-c", "-o", obj])
    header = run(["readelf", "-h", obj])
    machine = "AArch64" if architecture == "aarch64" else "Advanced Micro Devices X86-64"
    validate("generated object: native ELF64 REL", lambda: relocatable_header(header.stdout, machine))
    linked = output / "lhs-generated-strict-ordinary-link"
    run(ordinary_link_command(args.clangxx, obj, libraries, library_dir, linked, args.sanitized))
    linked_execution = run([linked])
    print("ordinary object link: " + linked_execution.stdout.splitlines()[-1], flush=True)
    inspect_dynamic(linked)
    for name in ("nested_read", "host_effect"):
        binary = output / f"rejected-{name}"
        rejection = run([driver, source / f"{name}.mdsl", "--region", "pipeline",
                         "--candidate", "automatic", "-c", "-o", binary], False)
        if binary.exists() or name + ".mdsl:" not in rejection.stderr:
            fail("unsupported source must reject with original source location and no output")
        print(f"{name}: rejected without output", flush=True)
    report["final_driver_sha256"] = digest(driver)
    unchanged()
    if args.has_openblas:
        report["openblas_lane"] = "requested_and_passed"
    report["status"] = "pass"
    save()
    print(report_path)


if __name__ == "__main__":
    main()
