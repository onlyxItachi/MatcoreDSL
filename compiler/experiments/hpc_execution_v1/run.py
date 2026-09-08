#!/usr/bin/env python3
"""Compile/check real region and standalone primitive lanes; timing is opt-in."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import re
import statistics
import subprocess
import sys


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def elf_header(text, relocatable=False):
    fields = dict(line.strip().split(":", 1) for line in text.splitlines() if ":" in line)
    fields = {key: value.strip() for key, value in fields.items()}
    if fields.get("Class") != "ELF64" or fields.get("Machine") != "Advanced Micro Devices X86-64":
        raise ValueError("expected ELF64 x86-64")
    kind = fields.get("Type", "")
    if (relocatable and not kind.startswith("REL ")) or (
            not relocatable and not kind.startswith(("DYN ", "EXEC "))):
        raise ValueError("unexpected ELF kind: " + kind)


def compiler_processes():
    found = []
    for item in Path("/proc").iterdir():
        if not item.name.isdigit():
            continue
        try:
            name = (item / "comm").read_text().strip()
        except (OSError, ProcessLookupError):
            continue
        if re.fullmatch(r"(clang.*|cc1.*|ninja|cmake|ld|ld\.bfd|ld\.lld|lld|make|g\+\+.*|gcc.*)", name):
            found.append({"pid": int(item.name), "name": name})
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driver", required=True, type=Path)
    parser.add_argument("--driver-build-commit", required=True)
    parser.add_argument("--expected-driver-sha256", required=True)
    parser.add_argument("--primitive-object", required=True, type=Path)
    parser.add_argument("--library-directory", type=Path)
    parser.add_argument("--provider", type=Path, help="configured exact provider; omission means OFF lane not exercised")
    parser.add_argument("--clangxx", type=Path, default=Path("/usr/bin/clang++-21"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--extra-primitive", nargs=3, action="append", default=[], metavar=("LABEL", "OBJECT", "ENTRY"))
    parser.add_argument("--extra-driver", nargs=4, action="append", default=[],
                        metavar=("LABEL", "DRIVER", "SHA256", "SOURCE_ROOT"),
                        help="compare additional generated/native source routes; capture source diff provenance")
    parser.add_argument("--timing", action="store_true")
    parser.add_argument("--quiet-window-note", help="required timing coordination note; not a machine-isolation proof")
    parser.add_argument("--cpu", type=int, help="bind each execution to this permitted CPU")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--target-ms", type=float, default=30)
    parser.add_argument("--timing-cases", nargs="+", default=["square16", "square128", "square512",
                        "rect128x256x64", "skinny16x512x256", "tail65x67x63"])
    args = parser.parse_args()
    if args.timing and not args.quiet_window_note:
        parser.error("--timing requires --quiet-window-note; coordinate other compilation first")
    if args.rounds < 1 or args.rounds > 30:
        parser.error("rounds must be 1..30")
    if args.cpu is not None and args.cpu not in os.sched_getaffinity(0):
        parser.error("requested CPU is outside the permitted affinity")
    source = Path(__file__).resolve().parent
    repo = source.parents[2]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error("output must be empty; no existing evidence/artifact is overwritten")
    driver = args.driver.resolve(strict=True)
    library = (args.library_directory or driver.parent.parent / "lib").resolve(strict=True)
    artifacts = [driver, args.clangxx.resolve(strict=True), args.primitive_object.resolve(strict=True),
                 library / "libmatcore_closed_candidates_isolated_v1.so",
                 library / "libmatcore_runtime.so"]
    if args.provider:
        artifacts.append(args.provider.absolute())
    for label, obj, entry in args.extra_primitive:
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", label) or not re.fullmatch(r"[a-zA-Z_][a-zA-Z_0-9]*", entry):
            parser.error("extra primitive label/entry must be simple identifiers")
        artifacts.append(Path(obj).resolve(strict=True))
    for label, path, expected, source_root in args.extra_driver:
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", label):
            parser.error("extra driver label must be a simple identifier")
        other = Path(path).resolve(strict=True)
        if digest(other) != expected:
            parser.error("extra driver differs from supplied SHA256")
        artifacts.extend([other, other.parent.parent / "lib/libmatcore_closed_candidates_isolated_v1.so",
                          other.parent.parent / "lib/libmatcore_runtime.so"])
    for obj in [args.primitive_object, *(Path(item[1]) for item in args.extra_primitive)]:
        llvm_ir = obj.with_suffix(".ll")
        for path in [llvm_ir, Path(str(llvm_ir) + ".manifest")]:
            if path.is_file():
                artifacts.append(path.resolve(strict=True))
    hashes = {str(path): digest(path) for path in artifacts}
    if hashes[str(driver)] != args.expected_driver_sha256:
        parser.error("driver hash differs from frozen baseline")
    source_hashes = {str(path): digest(path) for path in source.iterdir() if path.is_file()}
    report = {"schema": "mdslc-hpc-execution-experiment-v1", "status": "running",
              "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "driver_build_commit": args.driver_build_commit,
              "artifact_sha256": hashes, "source_sha256": source_hashes,
              "provider_lane": "requested_ON" if args.provider else "not_requested_not_tested",
              "commands": [], "lanes": [], "timing": [], "timing_requested": args.timing,
              "quiet_window_note": args.quiet_window_note,
              "affinity": sorted(os.sched_getaffinity(0)), "selected_cpu": args.cpu,
              "selected_environment": {key: os.environ.get(key) for key in (
                  "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "GOMP_CPU_AFFINITY", "OMP_PROC_BIND")}}
    report_path = output / "evidence.json"

    def save():
        report_path.write_text(json.dumps(report, indent=2) + "\n")

    def unchanged():
        for path, value in {**hashes, **source_hashes}.items():
            if digest(path) != value:
                raise RuntimeError("frozen input changed during experiment: " + path)

    def run(command, expected=0, execution=False):
        unchanged()
        argv = [str(value) for value in command]
        if execution and args.cpu is not None:
            argv = ["taskset", "-c", str(args.cpu), *argv]
        result = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, env={**os.environ, "LC_ALL": "C"}, timeout=300)
        report["commands"].append({"argv": argv, "exit": result.returncode,
                                   "stdout": result.stdout, "stderr": result.stderr})
        save()
        unchanged()
        if result.returncode != expected:
            raise RuntimeError(f"exit {result.returncode}, expected {expected}: {argv}\n{result.stderr}")
        return result

    def inspect(binary, region=False, obj=False):
        elf_header(run(["readelf", "-h", binary]).stdout, obj)
        if region:
            dynamic = run(["readelf", "-d", binary]).stdout
            needed = set(re.findall(r"Shared library: \[([^\]]+)\]", dynamic))
            required = {"libmatcore_closed_candidates_isolated_v1.so", "libmatcore_runtime.so.0"}
            if args.provider:
                provider_dynamic = run(["readelf", "-d", args.provider]).stdout
                soname = re.search(r"Library soname: \[([^\]]+)\]", provider_dynamic)
                if not soname:
                    raise RuntimeError("configured provider has no ELF SONAME")
                required.add(soname.group(1))
            if not required <= needed:
                raise RuntimeError("missing direct dependencies: " + str(required - needed))

    def json_lines(execution):
        return [json.loads(line) for line in execution.stdout.splitlines()]

    try:
        report["source_commit"] = run(["git", "-C", repo, "rev-parse", "HEAD"]).stdout.strip()
        report["source_status"] = run(["git", "-C", repo, "status", "--short"]).stdout
        diff = run(["git", "-C", repo, "diff", "--name-only", args.driver_build_commit,
                    report["source_commit"], "--", "compiler"])
        report["compiler_paths_different_from_build_commit"] = diff.stdout.splitlines()
        report["production_paths_different_from_build_commit"] = run([
            "git", "-C", repo, "diff", "--name-only", args.driver_build_commit, report["source_commit"],
            "--", "compiler/lib", "compiler/tools", "compiler/include", "compiler/cmake",
            "compiler/CMakeLists.txt"]).stdout.splitlines()
        report["driver_version"] = run([driver, "--version"]).stdout.strip()
        report["clang_version"] = run([args.clangxx, "--version"]).stdout.strip()
        report["lscpu"] = json.loads(run(["lscpu", "-J"]).stdout)
        report["cpu_policy"] = {}
        if args.cpu is not None:
            policy = Path(f"/sys/devices/system/cpu/cpu{args.cpu}/cpufreq")
            for key in ("scaling_driver", "scaling_governor", "scaling_min_freq", "scaling_max_freq"):
                try:
                    report["cpu_policy"][key] = (policy / key).read_text().strip()
                except OSError:
                    report["cpu_policy"][key] = "unavailable"
        inspect(args.primitive_object, obj=True)
        lanes = []
        selections = [("strict", "generated-strict"), ("strict", "native-strict"),
                      ("strict", "automatic"), ("reassociate", "existing-native")]
        if args.provider:
            selections.append(("reassociate", "openblas"))
        drivers = [("", driver, selections)]
        report["extra_drivers"] = []
        for label, path, expected, source_root in args.extra_driver:
            other_head = run(["git", "-C", source_root, "rev-parse", "HEAD"]).stdout.strip()
            other_diff = run(["git", "-C", source_root, "diff", "--", "compiler"]).stdout
            report["extra_drivers"].append({"label": label, "path": path, "sha256": expected,
                "source_root": source_root, "source_head": other_head,
                "uncommitted_compiler_diff_sha256": hashlib.sha256(other_diff.encode()).hexdigest(),
                "source_status": run(["git", "-C", source_root, "status", "--short"]).stdout})
            drivers.append((label + "-", Path(path), [("strict", "generated-strict"), ("strict", "native-strict")]))
        for prefix, selected_driver, policy_selections in drivers:
            for profile, candidate in policy_selections:
                name = f"region-{prefix}{profile}-{candidate}"
                binary = output / name
                run([selected_driver, source / f"{profile}.mdsl", "--region", "hpc_region",
                     "--candidate", candidate, "-o", binary])
                inspect(binary, region=True)
                lanes.append({"name": name, "binary": str(binary), "driver": str(selected_driver),
                              "layer": "region_end_to_end", "requested_candidate": candidate,
                              "actual_identity": "not_exposed_by_public_Result", "profile": profile})
        primitives = [("baseline", args.primitive_object, "_mlir_ciface___matcore_strict_gemm_f32_v1"),
                      *args.extra_primitive]
        for label, obj, entry in primitives:
            name = "primitive-" + label
            binary = output / name
            inspect(obj, obj=True)
            symbols = run(["nm", "--defined-only", obj]).stdout
            if not re.search(r"\bT " + re.escape(entry) + r"$", symbols, re.MULTILINE):
                raise RuntimeError("primitive C wrapper is missing: " + entry)
            run([args.clangxx, "-std=c++20", "-O2", "-ffp-contract=off", "-frounding-math",
                 "-DMDSLC_EXPERIMENT_ENTRY=" + entry, source / "primitive.cpp", obj, "-o", binary])
            inspect(binary)
            lanes.append({"name": name, "binary": str(binary), "layer": "standalone_primitive",
                          "profile": "strict", "entry": entry,
                          "product_authority": "none_standalone_experiment"})
        for lane in lanes:
            binary = Path(lane["binary"])
            opts = ["--relaxed"] if lane["profile"] == "reassociate" else []
            records = json_lines(run([binary, *opts], execution=True))
            expected_count = 16 if opts else 24
            if len(records) != expected_count or not all(r.get("pass") for r in records):
                raise RuntimeError("wrong oracle count/output identity")
            lane["correctness_cases"] = len(records)
            negative = run([binary, "--case", "square16", "--corrupt-output", *opts],
                           expected=4, execution=True)
            if "oracle/guard/input mismatch" not in negative.stderr:
                raise RuntimeError("corrupt-output negative control missed oracle")
            if lane["layer"] == "region_end_to_end":
                for fault in ("--bad-input-shape", "--readonly-output"):
                    for case in ("square16", "zero_k", "zero_m"):
                        run([binary, "--case", case, fault, *opts], execution=True)
            lane["executable_sha256"] = digest(binary)
            print(f"{lane['name']}: {len(records)} oracle cases and negative control passed", flush=True)
            report["lanes"].append(lane)
            save()
        for candidate in ["existing-native", *(["openblas"] if args.provider else [])]:
            binary = output / ("strict-refusal-" + candidate)
            run([driver, source / "strict.mdsl", "--region", "hpc_region", "--candidate", candidate,
                 "-o", binary])
            records = json_lines(run([binary, "--expect-incompatible"], execution=True))
            if len(records) != 24:
                raise RuntimeError("strict policy refusal matrix was incomplete")
        if args.timing:
            busy = compiler_processes()
            report["timing_preflight"] = {"compiler_processes": busy, "loadavg": os.getloadavg()}
            if busy:
                raise RuntimeError("timing refused while compiler/build processes are present")
            generator = random.Random(92831)
            for round_id in range(args.rounds):
                ordered = list(lanes)
                generator.shuffle(ordered)
                for lane in ordered:
                    for case in args.timing_cases:
                        busy = compiler_processes()
                        if busy:
                            raise RuntimeError("timing interrupted by compiler/build activity: " + str(busy))
                        opts = ["--relaxed"] if lane["profile"] == "reassociate" else []
                        result = run([lane["binary"], "--case", case, "--timing", "--samples", args.samples,
                                      "--target-ms", args.target_ms, *opts], execution=True)
                        rows = [r for r in json_lines(result) if r["kind"] == "timing"]
                        if len(rows) != args.samples:
                            raise RuntimeError("missing timing samples")
                        report["timing"].append({"lane": lane["name"], "case": case,
                                                 "round": round_id, "samples": rows,
                                                 "loadavg_after": os.getloadavg()})
                        save()
            summary = []
            for lane in lanes:
                for case in args.timing_cases:
                    values = [s["ns_per_call"] for batch in report["timing"]
                              if batch["lane"] == lane["name"] and batch["case"] == case
                              for s in batch["samples"]]
                    summary.append({"lane": lane["name"], "case": case, "samples": len(values),
                                    "median_ns": statistics.median(values), "min_ns": min(values), "max_ns": max(values)})
            report["timing_summary"] = summary
        report["status"] = "pass"
    except Exception as error:
        report["status"] = "fail"
        report["failure"] = str(error)
        save()
        raise
    finally:
        save()
    print(report_path)


if __name__ == "__main__":
    main()
