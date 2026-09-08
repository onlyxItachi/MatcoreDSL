#!/usr/bin/env python3
"""Bounded source-library feasibility experiment; no CMake/Ninja build runs.

Reads one existing admission-test link recipe using `ninja -t commands`, then
compiles/links only the new probe into a new owned output directory. No source
text recognition or execution interpreter: Clang and existing MDSLC admit C++;
upstream mlir-opt verifies/inlines/round-trips the independent MLIR specimens.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--mlir-prefix", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--has-openblas", action="store_true")
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    repo = source.parents[2]
    build, mlir, output = (p.resolve() for p in (args.build, args.mlir_prefix, args.out))
    if output.exists():
        raise SystemExit("--out must not exist: never overwrite a prior evidence run")
    output.mkdir(parents=True)
    fixtures = output / "fixtures"
    binaries = output / "executables"
    binaries.mkdir()
    commands = []
    record = {"source_head": subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
        "provider_lane_requested": args.has_openblas, "commands": commands}
    env = dict(os.environ)
    env.pop("TMPDIR", None)
    env["DEBUGINFOD_URLS"] = ""

    def run(argv, cwd=output, expected=0):
        start = time.monotonic()
        result = subprocess.run([str(x) for x in argv], cwd=cwd, env=env,
                                text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        commands.append({"argv": [str(x) for x in argv], "cwd": str(cwd),
                         "returncode": result.returncode, "expected": expected,
                         "stdout": result.stdout, "stderr": result.stderr,
                         "elapsed_seconds_not_a_benchmark": time.monotonic()-start})
        argument = str(argv[1])[:100] if len(argv) > 1 else ""
        print(f"[{result.returncode}] {Path(str(argv[0])).name} {argument}", flush=True)
        if result.returncode != expected:
            raise RuntimeError(result.stdout + result.stderr)
        return result.stdout

    watched = [build / "bin/mdslc-region", mlir / "bin/mlir-opt"]
    watched += sorted((build / "lib").glob("libmatcore_*.a"))
    watched += sorted(source.glob("*.*"))
    before = {str(p): sha(p) for p in watched}
    record["input_sha256"] = before
    try:
        clang = Path("/usr/bin/clang++-21")
        resource = run([clang, "-print-resource-dir"]).strip()
        run([clang, "--version"])
        run([mlir / "bin/mlir-opt", "--version"])
        run([build / "bin/mdslc-region", "--version"])
        # Ask Ninja for its command inventory, not build execution. Its final
        # link recipe supplies the coherent configured MLIR static-library set.
        recipe = run(["ninja", "-C", build, "-t", "commands",
                      "bin/matcore_experimental_region_admission_tests"])
        lines = [line for line in recipe.splitlines()
                 if " -o bin/matcore_experimental_region_admission_tests " in line]
        assert len(lines) == 1, "expected exactly one existing test link recipe"
        tokens = shlex.split(lines[0])
        assert tokens[:2] == [":", "&&"] and tokens[-2:] == ["&&", ":"]
        tokens = tokens[2:-2]
        assert not any(t in ("&&", ";", "|") for t in tokens)
        obj, executable = output / "probe.o", output / "probe"
        rewritten, objects, destinations = [], 0, 0
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token == "-Xlinker" and tokens[i+1].startswith("--dependency-file="):
                i += 2  # Never write the old build's link.d.
                continue
            if token == "-o":
                rewritten += [token, str(executable)]
                destinations += 1
                i += 2
                continue
            if token.endswith("admission_test.cpp.o"):
                rewritten.append(str(obj))
                objects += 1
            else:
                rewritten.append(token)
            i += 1
        assert objects == 1 and destinations == 1
        assert not any(t.endswith(".o") and t != str(obj) for t in rewritten)
        run([clang, "-std=c++20", "-O1", "-I" + str(repo / "compiler/lib/frontend"),
             "-I" + str(mlir / "include"), "-I/usr/lib/llvm-21/include",
             "-c", source / "probe.cpp", "-o", obj])
        run(rewritten, cwd=build)
        probe = run([executable, clang, resource, repo / "compiler/include", fixtures])
        assert "semantic module checks, 0 failures" in probe
        driver = build / "bin/mdslc-region"
        lanes = [("main_strict", "generated-strict"), ("main_strict", "native-strict"),
                 ("main_reassociate", "existing-native"), ("main_wrong_math", "native-strict")]
        if args.has_openblas:
            lanes += [("main_strict", "openblas"), ("main_reassociate", "openblas")]
        for case, candidate in lanes:
            exe = binaries / (case + "-" + candidate)
            run([driver, fixtures / case / "source.mdsl", "--region", "region",
                 "--candidate", candidate, "-o", exe])
            record.setdefault("executable_sha256", {})[str(exe)] = sha(exe)
            if case == "main_wrong_math":
                assert "MATH FAIL" in run([exe, "square"], expected=1)
            elif case == "main_strict" and candidate == "openblas":
                assert "failed=4 completed=3 effects=0 publications=0" in run([exe, "refusal"])
            else:
                assert "MATH PASS" in run([exe])
                assert "MATH PASS" in run([exe, "square"])
                assert "failed=3 completed=2 effects=0 publications=0" in run([exe, "shape"])
        # Exercise the actual driver refusal too, not just its admission library.
        refused = binaries / "header-rejected"
        run([driver, fixtures / "header_strict/source.mdsl", "--region", "region",
             "-o", refused], expected=1)
        assert not refused.exists() and "source-owned free function" in commands[-1]["stderr"]
        opt = mlir / "bin/mlir-opt"
        for case, op in (("visible", "arith.addf"), ("wrong_body", "arith.subf")):
            text = run([opt, source / (case + ".mlir"), "--inline"])
            assert "call @library" not in text and op in text
            bytecode = output / (case + ".mlirbc")
            run([opt, source / (case + ".mlir"), "--emit-bytecode", "-o", bytecode])
            assert bytecode.read_bytes()[:4] == b"ML\xefR"
            assert op in run([opt, bytecode, "--inline"])
        assert "call @library" in run([opt, source / "opaque.mlir", "--inline"])
        run([opt, source / "wrong_type.mlir"], expected=1)
        assert "result type mismatch" in commands[-1]["stderr"]
        assert all(sha(Path(path)) == digest for path, digest in before.items()), "input changed"
        assert all(sha(Path(path)) == digest for path, digest in record["executable_sha256"].items())
        record["all_assertions_passed"] = True
    finally:
        (output / "evidence.json").write_text(json.dumps(record, indent=2) + "\n")
    print("Semantic module feasibility controls passed; no module support added.")


if __name__ == "__main__":
    main()
