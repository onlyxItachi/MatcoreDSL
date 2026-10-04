#!/usr/bin/env python3
"""Authenticated same-source none/combined GPU execution, never CPU fallback.

Enabled mode requires the qualified physical target. Disabled mode checks an
explicit unavailable result; it is not execution support. No performance claim.
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
    parser.add_argument("--nm", type=Path, required=True)
    parser.add_argument("--enabled", action="store_true")
    parser.add_argument("--program", action="store_true")
    parser.add_argument("--install-build", type=Path)
    parser.add_argument("--install-bindir", default="bin")
    args = parser.parse_args()
    fixtures = Path(__file__).resolve().parent
    work = Path(tempfile.mkdtemp(prefix=f"gpu-pair-{args.target}-", dir=args.build))
    commands = []

    def run(command, success=True, extra_env=None):
        command = list(map(str, command))
        result = subprocess.run(command, capture_output=True, text=True, timeout=180,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "LC_ALL": "C",
                 **(extra_env or {})})
        commands.append({"argv": command, "exit": result.returncode,
                         "stdout": result.stdout, "stderr": result.stderr})
        (work / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        assert (result.returncode == 0) == success, commands[-1]
        return result

    if args.install_build:
        prefix = work / "installed"
        run(["cmake", "--install", args.install_build, "--prefix", prefix])
        args.driver = prefix / args.install_bindir / "mdslc-region"
    driver_hash = hashlib.sha256(args.driver.read_bytes()).hexdigest()

    def compile(text, name, optimization, success=True):
        source = work / f"{name}.mdsl"
        if args.program:
            assert "int main()" in text
            text = text.replace("int main()", "int math_main()")
            host = work / f"{name}-main.cpp"
            host.write_text("extern int math_main(); int main(){return math_main();}\n")
            invocation = ["--program", "--host", host, "--region", source,
                          "strict_fused_pipeline"]
        else:
            invocation = [source, "--region", "strict_fused_pipeline"]
        source.write_text(text)
        output = work / name
        result = run([args.driver, *invocation, "--candidate", "generated-" + args.target,
                      "--optimization", optimization, "-o", output], success)
        if not success:
            assert not output.exists(), "rejected host published an executable"
            return output, result
        symbols = run([args.nm, "--undefined-only", "--demangle", output]).stdout
        connected = "closed_host_v1::SessionAbiV2::gemmStrictFusedPair(" in symbols
        assert connected == (optimization == "strict-fused-pair"), symbols
        return output, result

    if args.enabled:
        text = "#define MDSLC_TEST_FUSED_GPU 1\n" + (
            fixtures.parent / "closed_driver/strict_fused_pair.mdsl").read_text()
        outcomes = []
        for optimization in ("none", "strict-fused-pair"):
            binary, _ = compile(text, "math-" + optimization, optimization)
            result = run([binary])
            assert "0 failures; 22 executed cases" in result.stdout, result.stdout
            outcomes.append(result.stdout)
        assert outcomes[0] == outcomes[1], outcomes
        api = "cuLaunchKernel" if args.target == "nvvm" else "hipModuleLaunchKernel"
        for index, symbol in enumerate((api, "pthread_create", "pthread_join")):
            forged = text + f'\nextern "C" int innocent() asm("{symbol}");\n' + \
                'extern "C" int innocent() { return 0; }\n'
            _, rejected = compile(forged, f"interposed-{index}", "strict-fused-pair", False)
            assert "original host defines symbol owned by trusted artifact" in rejected.stderr
            assert symbol in rejected.stderr

    # The same eligible source must refuse missing hardware/image before any
    # result publication, preserving its earlier ordinary publication/observation.
    unavailable = (fixtures / "unavailable.mdsl").read_text()
    binary, _ = compile(unavailable, "unavailable", "strict-fused-pair")
    visibility = "CUDA_VISIBLE_DEVICES" if args.target == "nvvm" else "HIP_VISIBLE_DEVICES"
    result = run([binary], extra_env={visibility: "-1"} if args.enabled else None)
    assert "GPU fused pair unavailable: PASS" in result.stdout, result.stdout
    assert hashlib.sha256(args.driver.read_bytes()).hexdigest() == driver_hash
    (work / "result.json").write_text(json.dumps({
        "driver_sha256": driver_hash, "target": args.target,
        "installed": bool(args.install_build), "program": args.program,
        "evidence": "physical authenticated source execution" if args.enabled
                    else "unavailable candidate refusal only",
        "commands": len(commands), "status": "PASS"}, indent=2) + "\n")
    print(f"{args.target} fused source contract PASS: {work}")


if __name__ == "__main__":
    main()
