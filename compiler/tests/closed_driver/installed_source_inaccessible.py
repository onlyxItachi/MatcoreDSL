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
        configure = [args.cmake, "-S", source / "compiler", "-B", build, "-G", "Ninja",
                     "-DBUILD_TESTING=OFF", "-DCMAKE_BUILD_TYPE=Release",
                     "-DCMAKE_C_COMPILER=" + args.clang.replace("clang++", "clang"),
                     "-DCMAKE_CXX_COMPILER=" + args.clang,
                     "-DMDSLC_CLANGXX_EXECUTABLE=" + args.clang,
                     "-DMDSLC_ENABLE_NATIVE_FRONTEND=ON", "-DMDSLC_ENABLE_BOOTSTRAP_FRONTEND=ON",
                     "-DMDSLC_ENABLE_MATCORE_MLIR=ON", "-DMDSLC_ENABLE_EXPERIMENTAL_REGIONS=ON",
                     "-DMDSLC_ENABLE_OPENBLAS=OFF", "-DMDSLC_REQUIRE_OPENBLAS=OFF",
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
        if hashlib.sha256(driver.read_bytes()).hexdigest() != digest:
            raise RuntimeError("installed driver changed during package acceptance")
        if source.exists() or build.exists() or stage.exists():
            raise RuntimeError("installed driver recreated producer trees")
        print(f"PASS closed source/build-inaccessible package at {commit}: "
              "3 policies, exact math, retained observations and ordered late failure; "
              "same-source none/forwarding executed after producer removal with checked call reference; "
              f"installed driver sha256={digest}")


if __name__ == "__main__":
    main()
