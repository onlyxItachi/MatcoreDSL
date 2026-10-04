#!/usr/bin/env python3
"""Record a tiny host build and exact-artifact research execution separately.

Coordinate compiler/device windows with the integration owner before invocation.
No benchmark, runtime authority, installer, target fallback, or arbitrary shader
input exists in this harness. Artifact checks bind observations, not source seals.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    commands = ap.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--clangxx", type=Path, required=True)
    build.add_argument("--link-lock", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    execute = commands.add_parser("run")
    execute.add_argument("--output", type=Path, required=True)
    execute.add_argument("--artifacts", type=Path, required=True)
    execute.add_argument("--device", required=True, help="inventory or an explicitly selected decimal device index")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).with_name("probe.cpp")
    binary = args.output / "probe"
    build_manifest = args.output / "probe-build.json"
    if args.command == "build":
        version = subprocess.run([str(args.clangxx), "--version"], capture_output=True, text=True, timeout=30)
        if version.returncode or "21.1.8" not in version.stdout:
            raise RuntimeError("exact Clang21.1.8 required: " + version.stdout + version.stderr)
        source_hash = digest(source)
        argv = ["flock", str(args.link_lock), str(args.clangxx), "-std=c++20", "-O1", "-g",
                "-Wall", "-Wextra", "-Wno-missing-field-initializers", "-fno-fast-math",
                "-ffp-contract=off", "-frounding-math", str(source), "-lvulkan", "-o", str(binary)]
        result = subprocess.run(argv, capture_output=True, text=True, timeout=60)
        evidence = dict(classification="research-only host build", compiler_version=version.stdout,
                        compiler_sha256=digest(args.clangxx), source_sha256=source_hash, argv=argv,
                        returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
        if not result.returncode and digest(source) == source_hash:
            evidence["binary_sha256"] = digest(binary)
        build_manifest.write_text(json.dumps(evidence, indent=2) + "\n")
        if result.returncode or "binary_sha256" not in evidence:
            raise RuntimeError("host build failed or source changed: " + result.stderr)
        print("PASS research host build;", evidence["binary_sha256"])
        return

    build_evidence = json.loads(build_manifest.read_text())
    if build_evidence.get("returncode") != 0 or build_evidence.get("source_sha256") != digest(source) or build_evidence.get("binary_sha256") != digest(binary):
        raise RuntimeError("host source/binary differs from recorded successful build")
    generated = json.loads((args.artifacts / "manifest.json").read_text())
    if generated.get("status") != "COMPILED_VALIDATED_ONLY" or generated.get("strict_contract_qualified") is not False:
        raise RuntimeError("missing bounded derivation manifest")
    specimen = Path(__file__).with_name("static_specimen.mlir")
    if generated.get("specimen_sha256") != digest(specimen):
        raise RuntimeError("repository static specimen changed")
    inputs = [args.artifacts / (kind + "-strict.spv") for kind in ("fill", "gemm")]
    hashes = {str(p): digest(p) for p in inputs}
    if any(generated["artifact_sha256"].get(p.name) != hashes[str(p)] for p in inputs):
        raise RuntimeError("derived shader changed before execution")
    if args.device != "inventory" and (not args.device.isdecimal() or int(args.device) >= 2**32):
        raise RuntimeError("explicit nonnegative device index required")
    observation = args.output / ("result-" + args.device + ".json")
    argv = [str(binary), *(str(p) for p in inputs), args.device, str(observation)]
    result = subprocess.run(argv, capture_output=True, text=True, timeout=20)
    evidence = dict(classification="exact-artifact research observation; no MDSLC execution authority",
                    argv=argv, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr,
                    probe_sha256=digest(binary), source_sha256=digest(source), shader_sha256=hashes)
    evidence_path = args.output / ("execution-" + args.device + ".json")
    evidence_path.write_text(json.dumps(evidence, indent=2) + "\n")
    if digest(binary) != build_evidence["binary_sha256"] or any(digest(p) != hashes[str(p)] for p in inputs):
        raise RuntimeError("artifact changed during probe")
    data = json.loads(observation.read_text())
    if data.get("strict_contract_qualified") is not False:
        raise RuntimeError("research observation incorrectly claims product qualification")
    if result.returncode == 78 and data.get("status", "").startswith("REFUSED_") and data.get("dispatches", 0) == 0:
        print("REFUSED", args.device, data["status"], "without fallback or dispatch")
    elif result.returncode == 0 and data.get("status") in (
            "INVENTORIED_NO_DISPATCH", "EXECUTED_BOUNDED_STRICT_SAMPLE_MATCH", "EXECUTED_STRICT_COUNTEREXAMPLE"):
        print(data["status"], "device", args.device, "comparisons", data.get("comparisons", 0),
              "mismatches", data.get("mismatches", 0), "; product qualification remains false")
    else:
        raise RuntimeError("probe/API/control error: " + repr(evidence))


if __name__ == "__main__":
    main()
