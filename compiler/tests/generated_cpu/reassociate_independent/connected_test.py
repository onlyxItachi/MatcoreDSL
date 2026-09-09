#!/usr/bin/env python3
"""Independent source/installed-DSO oracles; no timings or public identity API."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_files(pins):
    for path, expected in pins.items():
        if digest(Path(path)) != expected:
            raise RuntimeError(f"Frozen artifact changed: {path}")


def classify_result(result, expected=0, allowed_skip=None):
    if result.returncode == 77 and allowed_skip is not None:
        if result.stdout == allowed_skip and not result.stderr:
            return "skip"
        raise RuntimeError("Unrecognized or diagnostic-bearing skip")
    if result.returncode != expected:
        raise RuntimeError(f"exit {result.returncode}, expected {expected}")
    if expected == 0 and result.stderr:
        raise RuntimeError("Positive command emitted diagnostics")
    return "ok"


def source_negative(stdout, stderr):
    # Exact expected mathematical failures, never an unrelated sanitizer/crash
    # exit with a convenient earlier assertion in its diagnostic stream.
    if stdout != "Independent connected source: 2980 checks; 21 failures\n":
        raise RuntimeError("Incorrect negative source oracle identity")
    allowed = {"FAIL: first observation preserves exact math",
               "FAIL: second owning observation exact math",
               "FAIL: entire overlapping arena and sentinels exact",
               "FAIL: owning first observation survives Result and all resource mutation"}
    lines = stderr.splitlines()
    if len(lines) != 21 or not set(lines) <= allowed or not any(
            line == "FAIL: first observation preserves exact math" for line in lines):
        raise RuntimeError("Unexpected source negative diagnostics")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sdk", type=Path)
    parser.add_argument("--provider", type=Path)
    parser.add_argument("--cxx", type=Path, default=Path("/usr/bin/clang++-21"))
    parser.add_argument("--sanitized", action="store_true")
    args = parser.parse_args()
    args.driver = args.driver.absolute()
    args.output.mkdir(parents=True, exist_ok=False)
    fixture = Path(__file__).resolve().parent
    clean = os.environ.copy()
    for name in ("LD_LIBRARY_PATH", "LD_PRELOAD", "LD_AUDIT"):
        clean.pop(name, None)
    clean["DEBUGINFOD_URLS"] = ""
    clean["ASAN_OPTIONS"] = "detect_leaks=1:halt_on_error=1:strict_string_checks=1:check_initialization_order=1"
    clean["UBSAN_OPTIONS"] = "halt_on_error=1:print_stacktrace=1"
    inputs = [args.driver, args.cxx, fixture / "connected.mdsl",
              fixture / "installed_identity.cpp", Path(__file__).resolve()]
    # Record the actual driver-owned artifacts, not only a CLI spelling.
    driver_lib = args.driver.parent.parent / "lib"
    inputs += [driver_lib / "libmatcore_closed_candidates_isolated_v1.so",
               driver_lib / "libmatcore_runtime.so"]
    drivers = [("build", args.driver)]
    if args.sdk:
        args.sdk = args.sdk.absolute()
        private = args.sdk / "lib/mdslc/experimental-regions"
        dso = private / "libmatcore_closed_candidates_isolated_v1.so"
        runtime = args.sdk / "lib/libmatcore_runtime.so"
        installed = args.sdk / "bin/mdslc-region"
        inputs += [installed, dso, runtime, private / "include/closed_host_v1.h",
                   args.sdk / "include/matcore/region.h",
                   args.sdk / "include/matcore/detail/region_storage.h"]
        drivers.append(("installed", installed))
    if args.provider:
        args.provider = args.provider.absolute()  # retain configured symlink spelling
        inputs.append(args.provider)
    identities = {str(path): digest(path) for path in inputs}
    produced = {}
    records = []
    def checkpoint(**extra):
        (args.output / "evidence.json").write_text(json.dumps(
            {"inputs": identities, "produced": produced, "commands": records,
             **extra}, indent=2) + "\n")
    def command(label, argv, expected=0, produces=None, allowed_skip=None):
        verify_files(identities)
        verify_files(produced)
        if produces is not None and produces.exists():
            raise RuntimeError(f"Refusing an existing output: {produces}")
        result = subprocess.run([str(arg) for arg in argv], env=clean,
                                text=True, capture_output=True, timeout=180)
        records.append({"label": label, "argv": [str(arg) for arg in argv],
                        "returncode": result.returncode,
                        "stdout": result.stdout, "stderr": result.stderr})
        checkpoint()
        verify_files(identities)
        verify_files(produced)
        try:
            disposition = classify_result(result, expected, allowed_skip)
        except RuntimeError as error:
            raise RuntimeError(f"{label}: {error}\n{result.stdout}\n{result.stderr}") from error
        if disposition == "skip":
            checkpoint(status="skipped")
            print("SKIP independent connected matrix: AVX2/FMA unavailable")
            raise SystemExit(77)
        if produces is not None:
            produced[str(produces)] = digest(produces)
            checkpoint()
        return result.stdout
    def dynamic(path):
        text = command("ELF dynamic " + path.name, ["readelf", "-dW", path])
        return set(re.findall(r"\(NEEDED\).*\[([^]]+)\]", text)), text
    policies = ["generated-reassociate", "generated-strict", "automatic",
                "native-strict", "existing-native"]
    if args.provider:
        policies.append("openblas")
    outputs = []
    for origin, driver in drivers:
        for policy in policies:
            executable = args.output / f"{origin}-{policy}"
            argv = [driver, fixture / "connected.mdsl", "--region", "connected",
                    "--candidate", policy, "-o", executable]
            # The actual source driver's installed profile owns sanitization;
            # user -fsanitize options are correctly rejected. --sanitized only
            # selects ordinary private-consumer final-link flags below.
            command(f"compile {origin} {policy}", argv, produces=executable)
            before = digest(executable)
            stdout = command(f"execute {origin} {policy}", [executable, policy],
                             allowed_skip="SKIP independent connected source: AVX2/FMA unavailable\n")
            count = 1495 if policy in ("existing-native", "openblas") else 2980
            assert stdout == f"Independent connected source: {count} checks; 0 failures\n", stdout
            needed, _ = dynamic(executable)
            assert "libmatcore_closed_candidates_isolated_v1.so" in needed, needed
            assert "libmatcore_runtime.so.0" in needed, needed
            symbols = command("source instrumentation " + executable.name,
                              ["readelf", "-sW", executable])
            assert ("__asan_init" in symbols) == args.sanitized, "wrong source installation profile"
            assert before == digest(executable), "owned executable changed"
            outputs.append({"path": str(executable), "sha256": before, "checks": count})
        for actual, claimed in (("generated-reassociate", "generated-strict"),
                                ("generated-strict", "generated-reassociate")):
            stdout = command(f"rounding identity negative {origin} {actual}",
                             [args.output / f"{origin}-{actual}", claimed], expected=1)
            source_negative(stdout, records[-1]["stderr"])
    if args.sdk:
        executable = args.output / "installed-private-identity"
        argv = [args.cxx, "-std=c++20", "-O2", "-ffp-contract=off", "-fno-fast-math",
                "-I" + str(args.sdk / "include"), "-I" + str(private / "include"),
                fixture / "installed_identity.cpp", dso, runtime]
        if args.provider:
            argv += ["-Xlinker", "--push-state", "-Xlinker", "--no-as-needed",
                     args.provider, "-Xlinker", "--pop-state",
                     "-Xlinker", "-rpath", "-Xlinker", args.provider.parent]
        for directory in (private, args.sdk / "lib"):
            argv += ["-Xlinker", "-rpath", "-Xlinker", directory]
        argv += ["-pthread", "-ldl", "-o", executable]
        if args.sanitized:
            argv += ["-fsanitize=address,undefined"]
        command("compile installed private identity", argv, produces=executable)
        before = digest(executable)
        stdout = command("execute installed private identity", [executable, dso, runtime],
                         allowed_skip="SKIP independent installed private adapter: AVX2/FMA unavailable\n")
        assert stdout == (
            "Private installed implementation: closed.generated.reassociate_f32.avx2_fma.mlir21.v1\n"
            "Private installed implementation: closed.generated.strict_f32.mlir21.v1\n"
            "Private installed implementation: closed.generated.strict_f32.mlir21.v1\n"
            "Private installed implementation: closed.native.strict_f32.v1\n"
            "Independent installed private adapter: 327 checks; 0 failures\n"), stdout
        needed, _ = dynamic(executable)
        assert {"libmatcore_closed_candidates_isolated_v1.so", "libmatcore_runtime.so.0"} <= needed
        if args.provider:
            provider_elf = command("provider SONAME", ["readelf", "-dW", args.provider])
            soname = re.search(r"\(SONAME\).*\[([^]]+)\]", provider_elf).group(1)
            assert soname in needed, "private manual consumer omitted direct provider ownership"
        # An installed package must keep both issued definitions local and real.
        symbols = command("installed private symbol table", ["readelf", "-sW", dso])
        for symbol in ("__matcore_reassociate_gemm_f32_avx2_v1",
                       "_mlir_ciface___matcore_reassociate_gemm_f32_avx2_v1"):
            entries = [line for line in symbols.splitlines() if line.split()[-1:] == [symbol]]
            assert len(entries) == 1 and " FUNC " in entries[0] and " LOCAL " in entries[0], entries
        assert before == digest(executable), "owned private executable changed"
        outputs.append({"path": str(executable), "sha256": before, "checks": 327})
        stdout = command("wrong installed owner negative", [executable, runtime, dso], expected=1)
        assert stdout.endswith("Independent installed private adapter: 327 checks; 2 failures\n"), stdout
        assert records[-1]["stderr"] == "FAIL: dynamic symbol resolves to exact installed owner\n" * 2
    verify_files(identities)
    verify_files(produced)
    assert all(output["sha256"] == digest(Path(output["path"])) for output in outputs), "owned output changed"
    checkpoint(outputs=outputs, status="passed")
    negatives = 2 * len(drivers) + bool(args.sdk)
    print(f"Independent connected matrix: {len(outputs)} positive executions; "
          f"{sum(output['checks'] for output in outputs)} checks; {negatives} negative controls; 0 failures")


if __name__ == "__main__":
    main()
