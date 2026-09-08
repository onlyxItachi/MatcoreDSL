#!/usr/bin/env python3
"""Public-driver semantic checks; no private APIs, build, provider, or timing claim."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("driver", type=Path)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--clangxx", type=Path, default=Path("/usr/bin/clang++-21"))
    parser.add_argument("--library-directory", type=Path)
    args = parser.parse_args()
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
    report = {"driver": str(driver), "driver_sha256": initial_hash, "commands": []}
    report["library_sha256"] = library_hashes
    report["fixture_sha256"] = {
        str(path): digest(path) for path in sorted(source.iterdir())
        if path.suffix in (".mdsl", ".h")}
    report_path = output / "evidence.json"

    def run(command, expected_success=True):
        if digest(driver) != initial_hash:
            raise SystemExit("frozen driver changed during testing")
        if any(digest(Path(path)) != sha for path, sha in library_hashes.items()):
            raise SystemExit("frozen runtime/candidate library changed during testing")
        result = subprocess.run([str(item) for item in command], text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                timeout=180, check=False)
        record = {"argv": [str(item) for item in command], "returncode": result.returncode,
                  "stdout": result.stdout, "stderr": result.stderr}
        report["commands"].append(record)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        if (result.returncode == 0) != expected_success:
            raise SystemExit(json.dumps(record, indent=2))
        return result

    run([driver, "--version"])
    for carry in ("lhs", "rhs"):
        for candidate in ("native-strict", "generated-strict", "automatic", "existing-native"):
            binary = output / f"{carry}-{candidate}"
            run([driver, source / f"{carry}.mdsl", "--region", "pipeline",
                 "--candidate", candidate, "-o", binary])
            execution = run([binary, "--expect-policy-rejection"]
                            if candidate == "existing-native" else [binary])
            print(f"{carry}/{candidate}: {execution.stdout.splitlines()[-1]}", flush=True)
            run(["readelf", "-d", binary])
    # Also exercise the ordinary relocatable-object/final-link route.
    obj = output / "lhs-generated-strict.o"
    run([driver, source / "lhs.mdsl", "--region", "pipeline", "--candidate",
         "generated-strict", "-c", "-o", obj])
    run(["readelf", "-h", obj])
    linked = output / "lhs-generated-strict-ordinary-link"
    run([args.clangxx, obj, *libraries, "-pthread", f"-Wl,-rpath,{library_dir}",
         "-o", linked])
    linked_execution = run([linked])
    print("ordinary object link: " + linked_execution.stdout.splitlines()[-1], flush=True)
    for name in ("nested_read", "host_effect"):
        binary = output / f"rejected-{name}"
        rejection = run([driver, source / f"{name}.mdsl", "--region", "pipeline",
                         "--candidate", "automatic", "-c", "-o", binary], False)
        if binary.exists() or name + ".mdsl:" not in rejection.stderr:
            raise SystemExit("unsupported source must reject with original source location and no output")
        print(f"{name}: rejected without output", flush=True)
    report["final_driver_sha256"] = digest(driver)
    if report["final_driver_sha256"] != initial_hash:
        raise SystemExit("frozen driver changed at final verification")
    report["status"] = "pass"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(report_path)


if __name__ == "__main__":
    main()
