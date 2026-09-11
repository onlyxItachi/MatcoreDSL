#!/usr/bin/env python3
"""Real driver -> authenticated source -> private candidate -> observable trace.

GPU-enabled mode requires real hardware; it never substitutes CPU/emulation or
silently skips. GPU-disabled mode proves explicit refusal, not target support.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--target", choices=("nvvm", "rocdl"), required=True)
    parser.add_argument("--enabled", action="store_true")
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    work = Path(tempfile.mkdtemp(prefix=f"gpu-{args.target}-source-", dir=args.build))
    commands = []
    driver_hash = hashlib.sha256(args.driver.read_bytes()).hexdigest()

    def run(command, success=True):
        command = list(map(str, command))
        result = subprocess.run(command, capture_output=True, text=True, timeout=180,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "LC_ALL": "C"})
        commands.append({"argv": command, "exit": result.returncode,
                         "stdout": result.stdout, "stderr": result.stderr})
        (work / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        assert (result.returncode == 0) == success, commands[-1]
        return result

    def compile(fixture, name, success=True):
        output = work / name
        result = run([args.driver, fixture, "--region", "pipeline", "--candidate",
                      "generated-" + args.target, "-o", output], success)
        if not success:
            assert not output.exists(), "rejected source published an executable"
        return output, result

    binary, _ = compile(source / "failure_frontier.mdsl", "frontier")
    result = run([binary] if args.enabled else [binary, "--unavailable"])
    assert "0 failures" in result.stdout, result.stdout
    if args.enabled:
        # Reuse the existing independent all-MAY-alias/late-read/observation/
        # failed-second-operation oracle, with no changes to its mathematics.
        for carry in ("lhs", "rhs"):
            binary, _ = compile(source.parent / "driver_storage_conformance" /
                                f"{carry}.mdsl", carry)
            result = run([binary])
            assert "0 failures" in result.stdout and "22 cases" in result.stdout
        binary, _ = compile(source / "zero_guard.mdsl", "zero-guard")
        result = run([binary])
        assert "0 failures" in result.stdout
        # Whole original host IR must not interpose the actual trusted GPU
        # implementation, even when a definition has a misleading C++ name.
        api = "cuLaunchKernel" if args.target == "nvvm" else "hipModuleLaunchKernel"
        for index, symbol in enumerate((api, "pthread_create", "pthread_join", "_ZNSt6thread4joinEv")):
            forged = work / f"interposed-{index}.mdsl"
            forged.write_text((source / "failure_frontier.mdsl").read_text() +
                f'\nextern "C" int innocent() asm("{symbol}");\n'
                'extern "C" int innocent() { return 0; }\n')
            _, rejected = compile(forged, f"must-not-exist-{index}", False)
            assert "original host defines symbol owned by trusted artifact" in rejected.stderr
            assert symbol in rejected.stderr
    assert hashlib.sha256(args.driver.read_bytes()).hexdigest() == driver_hash
    (work / "result.json").write_text(json.dumps({
        "driver_sha256": driver_hash, "target": args.target,
        "evidence": "physical authenticated source execution" if args.enabled
                    else "unavailable candidate refusal only",
        "commands": len(commands), "status": "PASS"}, indent=2) + "\n")
    print(f"{args.target} connected source contract PASS: {work}")


if __name__ == "__main__":
    main()
