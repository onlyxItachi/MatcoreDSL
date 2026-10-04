#!/usr/bin/env python3
"""Installed closed-source execution after deleting disposable producer trees."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "package"))
from run_source_inaccessible_consumer import authenticate_repository, clone_clean_commit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repository", "expected-commit", "configured-clean", "cmake", "git",
                 "clang", "llvm-dir", "clang-dir", "mlir-dir", "schedule"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--linker-launcher", default="")
    parser.add_argument("--nm", default="nm")
    parser.add_argument("--has-fused-pair", choices=("ON", "OFF"), default="OFF")
    parser.add_argument("--has-nvvm", choices=("ON", "OFF"), default="OFF")
    parser.add_argument("--has-rocdl", choices=("ON", "OFF"), default="OFF")
    args = parser.parse_args()
    repository = Path(args.repository).resolve(strict=True)
    commit = authenticate_repository(args.git, repository, args.expected_commit,
                                     args.configured_clean == "ON")
    if not shutil.rmtree.avoids_symlink_attacks:
        raise RuntimeError("symlink-safe disposable producer removal is required")
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    for name in ("LD_PRELOAD", "LD_AUDIT", "LD_LIBRARY_PATH"):
        environment.pop(name, None)
    with tempfile.TemporaryDirectory(prefix="mdslc-closed-installed-") as temporary:
        root = Path(temporary).resolve()
        if root == repository or repository in root.parents:
            raise RuntimeError("disposable package root must be outside its source repository")
        source, build, stage = root / "producer-source", root / "producer-build", root / "stage"
        installed = root / "relocated-install"
        log = []

        def run(argv, expected=0, stdout=None):
            result = subprocess.run([str(arg) for arg in argv], cwd=root, env=environment,
                                    text=True, capture_output=True, timeout=900)
            log.append({"argv": [str(arg) for arg in argv], "exit": result.returncode,
                        "stdout": result.stdout, "stderr": result.stderr})
            if result.returncode != expected or (stdout is not None and result.stdout != stdout):
                print(json.dumps(log, indent=2))
                raise RuntimeError("closed installed-source command or oracle failed")
            return result

        clone_clean_commit(args.git, repository, source, commit)
        consumer = root / "consumer.mdsl"
        shutil.copyfile(source / "compiler/examples/experimental/two_gemm.mdsl", consumer)
        forwarding_consumer = root / "forwarding-consumer.mdsl"
        shutil.copyfile(source / "compiler/tests/closed_driver/publication_read_forwarding.mdsl",
                        forwarding_consumer)
        fused_consumer = root / "fused-consumer.mdsl"
        if args.has_fused_pair == "ON":
            shutil.copyfile(source / "compiler/tests/closed_driver/strict_fused_pair.mdsl",
                            fused_consumer)
        gpu_kinds = [kind for kind in ("nvvm", "rocdl") if getattr(args, "has_" + kind) == "ON"]
        gpu_consumer = root / "gpu-fused-consumer.mdsl"
        gpu_program = root / "gpu-fused-program.mdsl"
        gpu_host = root / "gpu-fused-main.cpp"
        gpu_cap_consumers = {}
        gpu_cap_programs = {}
        if gpu_kinds:
            gpu_text = "#define MDSLC_TEST_FUSED_GPU 1\n" + fused_consumer.read_text()
            gpu_consumer.write_text(gpu_text)
            gpu_program.write_text(gpu_text.replace("int main()", "int math_main()"))
            gpu_host.write_text("extern int math_main(); int main(){return math_main();}\n")
            cap_text = (source / "compiler/tests/generated_gpu_fused_pair/work_cap.mdsl").read_text()
            for optimization, expect_refusal in (("none", 0), ("strict-fused-pair", 1)):
                # The private macro changes only the ordinary host expectation;
                # the admitted region body is identical in both copies.
                text = f"#define MDSLC_EXPECT_COMBINED_WORK_REFUSAL {expect_refusal}\n" + cap_text
                gpu_cap_consumers[optimization] = root / f"gpu-cap-{optimization}.mdsl"
                gpu_cap_programs[optimization] = root / f"gpu-cap-{optimization}-program.mdsl"
                gpu_cap_consumers[optimization].write_text(text)
                gpu_cap_programs[optimization].write_text(text.replace("int main()", "int math_main()"))
        configure = [args.cmake, "-S", source / "compiler", "-B", build, "-G", "Ninja",
                     "-DBUILD_TESTING=OFF", "-DCMAKE_BUILD_TYPE=Release",
                     "-DCMAKE_C_COMPILER=" + args.clang.replace("clang++", "clang"),
                     "-DCMAKE_CXX_COMPILER=" + args.clang,
                     "-DMDSLC_CLANGXX_EXECUTABLE=" + args.clang,
                     "-DMDSLC_ENABLE_NATIVE_FRONTEND=ON", "-DMDSLC_ENABLE_BOOTSTRAP_FRONTEND=ON",
                     "-DMDSLC_ENABLE_MATCORE_MLIR=ON", "-DMDSLC_ENABLE_EXPERIMENTAL_REGIONS=ON",
                     "-DMDSLC_ENABLE_OPENBLAS=OFF", "-DMDSLC_REQUIRE_OPENBLAS=OFF",
                     "-DMDSLC_ENABLE_EXPERIMENTAL_NVVM=" + args.has_nvvm,
                     "-DMDSLC_ENABLE_EXPERIMENTAL_ROCDL=" + args.has_rocdl,
                     "-DMDSLC_EXPERIMENTAL_CPU_GEMM_SCHEDULE=" + args.schedule,
                     "-DLLVM_DIR=" + args.llvm_dir, "-DClang_DIR=" + args.clang_dir,
                     "-DMLIR_DIR=" + args.mlir_dir]
        if args.linker_launcher:
            configure.append("-DCMAKE_CXX_LINKER_LAUNCHER=" + args.linker_launcher)
        run(configure)
        run([args.cmake, "--build", build, "--parallel", "2"])
        run([args.cmake, "--install", build, "--prefix", stage])
        stage.rename(installed)
        driver = installed / "bin/mdslc-region"
        digest = hashlib.sha256(driver.read_bytes()).hexdigest()
        # Only these exact task-created producer directories are removed.
        shutil.rmtree(source)
        shutil.rmtree(build)
        if source.exists() or build.exists() or stage.exists():
            raise RuntimeError("producer source/build/staging remains accessible")
        for candidate in ("native-strict", "generated-strict", "automatic"):
            executable = root / candidate
            run([driver, consumer, "--region", "pipeline", "--candidate", candidate,
                 "-o", executable])
            run([executable], stdout="22 28 49 64 -> 120 156 49 64\n")
            run([executable, "late-failure"],
                stdout="checked failure; first publication and observation retained\n")
        forwarding_outputs = []
        for optimization in ("none", "publication-read-forwarding"):
            executable = root / ("forwarding-" + optimization)
            # Both consumer copies outlive producer removal. Never locate the
            # oracle/runner or a private helper in the deleted source/build.
            run([driver, forwarding_consumer, "--region", "forwarding_pipeline",
                 "--candidate", "generated-strict", "--optimization", optimization,
                 "-o", executable])
            symbols = run([args.nm, "--undefined-only", "--demangle", executable]).stdout
            connected = "closed_host_v1::SessionAbiV2::readForwarded(" in symbols
            if connected != (optimization == "publication-read-forwarding"):
                print(json.dumps(log, indent=2))
                raise RuntimeError("source-inaccessible optimization did not connect its checked read")
            output = run([executable, "--require-execution"]).stdout
            if not re.fullmatch(
                    r"Publication forwarding source: [1-9][0-9]* checks; 0 failures; "
                    r"6 executed cases; 0 explicit refusals\n", output):
                print(json.dumps(log, indent=2))
                raise RuntimeError("source-inaccessible forwarding did not execute its exact oracle")
            forwarding_outputs.append(output)
        if forwarding_outputs[0] != forwarding_outputs[1]:
            print(json.dumps(log, indent=2))
            raise RuntimeError("source-inaccessible same-source optimization outcomes differ")
        if args.has_fused_pair == "ON":
            fused_outputs = []
            for optimization in ("none", "strict-fused-pair"):
                executable = root / ("fused-" + optimization)
                run([driver, fused_consumer, "--region", "strict_fused_pipeline",
                     "--candidate", "generated-strict", "--optimization", optimization,
                     "-o", executable])
                symbols = run([args.nm, "--undefined-only", "--demangle", executable]).stdout
                connected = "closed_host_v1::SessionAbiV2::gemmStrictFusedPair(" in symbols
                if connected != (optimization == "strict-fused-pair"):
                    print(json.dumps(log, indent=2))
                    raise RuntimeError("source-inaccessible strict pair option did not connect checked runtime")
                # The source includes an INT64_MAX legal empty shape: prevent
                # an erroneous generated empty row loop from hanging the gate.
                result = subprocess.run([str(executable)], cwd=root, env=environment,
                                        text=True, capture_output=True, timeout=20)
                log.append({"argv": [str(executable)], "exit": result.returncode,
                            "stdout": result.stdout, "stderr": result.stderr})
                if result.returncode != 0 or not re.fullmatch(
                        r"Strict fused pair source: [1-9][0-9]* checks; 0 failures; "
                        r"22 executed cases\n", result.stdout):
                    print(json.dumps(log, indent=2))
                    raise RuntimeError("source-inaccessible strict pair exact source oracle failed")
                fused_outputs.append(result.stdout)
            if fused_outputs[0] != fused_outputs[1]:
                print(json.dumps(log, indent=2))
                raise RuntimeError("source-inaccessible fused/unfused outcomes differ")
        for kind in gpu_kinds:
            for route in ("source", "program"):
                outputs = []
                invocation = [gpu_consumer, "--region", "strict_fused_pipeline"]
                if route == "program":
                    invocation = ["--program", "--host", gpu_host,
                                  "--region", gpu_program, "strict_fused_pipeline"]
                for optimization in ("none", "strict-fused-pair"):
                    executable = root / f"gpu-{kind}-{route}-{optimization}"
                    run([driver, *invocation, "--candidate", "generated-" + kind,
                         "--optimization", optimization, "-o", executable])
                    symbols = run([args.nm, "--undefined-only", "--demangle", executable]).stdout
                    connected = "closed_host_v1::SessionAbiV2::gemmStrictFusedPair(" in symbols
                    if connected != (optimization == "strict-fused-pair"):
                        raise RuntimeError("source-inaccessible GPU pair call discriminator failed")
                    output = run([executable]).stdout
                    if not re.fullmatch(
                            r"Strict fused pair source: [1-9][0-9]* checks; 0 failures; "
                            r"22 executed cases\n", output):
                        print(json.dumps(log, indent=2))
                        raise RuntimeError("source-inaccessible GPU pair physical oracle failed")
                    outputs.append(output)
                if outputs[0] != outputs[1]:
                    raise RuntimeError("source-inaccessible GPU fused/unfused trace differs")
                # Keep the 22-case parity oracle above independent. This
                # additional source case intentionally differs: original
                # per-GEMM envelopes allow M65, but the combined private work
                # cap must refuse at f2 after retaining the earlier effects.
                for optimization in ("none", "strict-fused-pair"):
                    invocation = [gpu_cap_consumers[optimization], "--region", "strict_fused_pipeline"]
                    if route == "program":
                        invocation = ["--program", "--host", gpu_host,
                                      "--region", gpu_cap_programs[optimization], "strict_fused_pipeline"]
                    executable = root / f"gpu-cap-{kind}-{route}-{optimization}"
                    run([driver, *invocation, "--candidate", "generated-" + kind,
                         "--optimization", optimization, "-o", executable])
                    symbols = run([args.nm, "--undefined-only", "--demangle", executable]).stdout
                    connected = "closed_host_v1::SessionAbiV2::gemmStrictFusedPair(" in symbols
                    if connected != (optimization == "strict-fused-pair"):
                        print(json.dumps(log, indent=2))
                        raise RuntimeError("source-inaccessible GPU work cap call discriminator failed")
                    output = run([executable]).stdout
                    outcome = ("over-cap refused at f2" if optimization == "strict-fused-pair"
                               else "over-cap executed")
                    if not re.fullmatch(
                            r"GPU fused pair work cap: [1-9][0-9]* checks; 0 failures; "
                            r"boundary executed; " + re.escape(outcome) + r"\n", output):
                        print(json.dumps(log, indent=2))
                        raise RuntimeError("source-inaccessible GPU work cap exact source oracle failed")
        if hashlib.sha256(driver.read_bytes()).hexdigest() != digest:
            raise RuntimeError("installed driver changed during package acceptance")
        if source.exists() or build.exists() or stage.exists():
            raise RuntimeError("installed driver recreated producer trees")
        print(f"PASS closed source/build-inaccessible package at {commit}: "
              "3 policies, exact math, retained observations and ordered late failure; "
              "same-source none/forwarding executed after producer removal with checked call reference; "
              + ("same-source none/strict-pair executed after producer removal with checked call reference; "
                 if args.has_fused_pair == "ON" else "") +
              (f"physical GPU none/strict-pair single/multi-source after producer removal: {gpu_kinds}, "
               "22-case parity plus shared-work boundary execution and f2 refusal; "
                 if gpu_kinds else "") +
              f"installed driver sha256={digest}")


if __name__ == "__main__":
    main()
