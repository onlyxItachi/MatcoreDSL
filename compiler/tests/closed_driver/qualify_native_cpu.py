#!/usr/bin/env python3
"""Fail-closed native closed-source qualification, not legacy CPU support."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import xml.etree.ElementTree as ET


SCOPE = (r"^(platform\.closed_fp_environment\.v1|generated_cpu\..*|closed_candidates\..*|"
         r"runtime\.(closed_host|experimental_region)\..*|private_runtime\..*|"
         r"frontend\.(closed_.*|experimental_.*|frozen_host_.*|program_input_mutation)|"
         r"driver\.(program_.*|source_library_.*|generated_reassociate\..*|"
         r"experimental_region_contract|implementation_ownership|storage_conformance(_runner)?)|"
         r"codegen\.(artifact_symbol_ownership_v1|authenticated_host_thunk_v1|program_call_interface_v1)|"
         r"mlir\.closed_region_semantics|package\.experimental_regions\..*)$")
REQUIRED = {
    "platform.closed_fp_environment.v1", "generated_cpu.target_contract",
    "generated_cpu.target_cli_contract", "generated_cpu.execution.normal",
    "generated_cpu.independent_execution.normal", "generated_cpu.input_alias.normal",
    "generated_cpu.execution.asan", "generated_cpu.input_alias.asan",
    "generated_cpu.input_alias_corruption.normal", "generated_cpu.sanitizer_negative_control",
    "generated_cpu.object_contract", "generated_cpu.source_schedule.boundary_matrix",
    "closed_candidates.production", "closed_candidates.provider_negative.partial_failure",
    "closed_candidates.provider_negative.control_corruption", "runtime.closed_host.contract_v1",
    "runtime.closed_host.independent_v1", "runtime.closed_host.authority_v1",
    "runtime.closed_host.private_value_abi_v2", "frontend.closed_source_generated_execution_v1",
    "frontend.experimental_region_admission", "codegen.authenticated_host_thunk_v1",
    "codegen.program_call_interface_v1",
    "driver.program_core", "driver.program_reassociate", "driver.program_installed",
    "driver.storage_conformance", "driver.implementation_ownership",
    "driver.generated_reassociate.generated-reassociate",
    "codegen.artifact_symbol_ownership_v1", "private_runtime.symbol_contract",
    "package.experimental_regions.install_contract",
    "package.experimental_regions.closed_source_inaccessible",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", required=True, type=Path)
    parser.add_argument("--require-arm", action="store_true")
    args = parser.parse_args()
    build = args.build.resolve(strict=True)
    machine = platform.machine()
    if platform.system() != "Linux" or machine not in ("x86_64", "aarch64"):
        parser.error("unsupported native qualification machine")
    if args.require_arm and machine != "aarch64":
        parser.error("ARM qualification cannot be satisfied by cross-compilation or x86 execution")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    listing = subprocess.check_output(["ctest", "--test-dir", str(build), "--show-only=json-v1"], env=env)
    selected = {test["name"] for test in json.loads(listing)["tests"] if re.fullmatch(SCOPE, test["name"])}
    missing = REQUIRED - selected
    if missing:
        parser.error("qualification lost mandatory tests: " + ", ".join(sorted(missing)))
    junit = build / "native-cpu-junit.xml"
    report_path = build / "native-cpu-qualification.json"
    driver = build / "bin/mdslc-region"
    report = {"machine": machine, "status": "running", "tests": sorted(selected),
              "driver_sha256": hashlib.sha256(driver.read_bytes()).hexdigest(),
              "scope": "closed-source strict/native; legacy CPU planner/provider performance unqualified"}
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    result = subprocess.run(["ctest", "--test-dir", str(build), "--output-on-failure",
                             "--no-tests=error", "-j1", "-R", SCOPE,
                             "--output-junit", str(junit)], env=env)
    cases = ET.parse(junit).getroot().findall("testcase")
    actual = {case.attrib["name"] for case in cases}
    failed = [case.attrib["name"] for case in cases
              if case.find("failure") is not None or case.find("skipped") is not None]
    report.update(status="PASS" if result.returncode == 0 and actual == selected and not failed else "FAIL",
                  failed_or_skipped=failed, executed_count=len(cases))
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    if report["status"] != "PASS":
        raise SystemExit("native qualification failed or omitted/skipped selected tests")
    print(f"PASS native {machine}: {len(cases)} closed-source tests; no skips; no performance claim")


if __name__ == "__main__":
    main()
