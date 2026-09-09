#!/usr/bin/env python3
"""Independent source-alias/effect and publication falsifiers; no IR import."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_refusal(result, diagnostic):
    combined = (result.stdout + result.stderr).lower()
    return (result.returncode == 1 and diagnostic in result.stderr and
            not any(marker in result.stdout for marker in
                    ("VALIDATED program", "CROSS_TU_LINK", "PUBLISHED")) and
            not any(marker in combined for marker in
                    ("addresssanitizer", "undefinedbehaviorsanitizer", "runtime error:",
                     "segmentation fault", "deadlysignal", "core dumped", "aborted")))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--clang", type=Path, required=True)
    parser.add_argument("--compiler-source", type=Path, required=True)
    parser.add_argument("--expected-driver-sha256")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    driver = args.driver.resolve(strict=True)
    compiler = args.compiler_source.resolve(strict=True)
    fixtures = compiler / "tests/program/fixtures"
    own = Path(__file__).resolve().parent
    work = args.output.absolute() if args.output else Path(tempfile.mkdtemp(prefix="mdslc-program-independent-")) / "evidence"
    work.mkdir(parents=True, exist_ok=False)
    inputs = [driver, args.clang.resolve(strict=True), Path(__file__).resolve()]
    inputs += sorted(own.glob("main_*.cpp")) + sorted(own.glob("utility*.cpp"))
    inputs += sorted(own.glob("*.mdsl")) + sorted(fixtures.iterdir())
    inputs += [compiler / "include/matcore/region.h", compiler / "include/matcore/detail/region_storage.h"]
    frozen = {str(path): digest(path) for path in inputs if path.is_file()}
    if args.expected_driver_sha256 and frozen[str(driver)] != args.expected_driver_sha256:
        parser.error("driver differs from the independent frozen identity")
    report = {"schema": "program-independent-source-authority-v1", "status": "running",
              "inputs_sha256": frozen, "commands": [], "checks": 0,
              "scope": "source-only cross-TU identity/effects and no-clobber; no performance claim"}

    def unchanged():
        for name, sha in frozen.items():
            if digest(name) != sha:
                raise RuntimeError("review input changed: " + name)

    def save():
        (work / "evidence.json").write_text(json.dumps(report, indent=2) + "\n")

    def invoke(command, cwd):
        unchanged()
        result = subprocess.run([str(x) for x in command], cwd=cwd, capture_output=True,
                                text=True, timeout=180)
        report["commands"].append({"argv": [str(x) for x in command], "cwd": str(cwd),
                                    "exit": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
        save()
        unchanged()
        return result

    def demand(condition, explanation):
        report["checks"] += 1
        if not condition:
            raise RuntimeError(explanation)

    def command(host, executable, utility=False, first=None):
        result = [driver, "--program", "--candidate", "generated-strict", "--host", host]
        if utility:
            result += ["--host", own / ("utility.cpp" if utility is True else utility)]
        return result + ["--region", first or fixtures / "first.mdsl", "first", "--region",
                         fixtures / "second.mdsl", "second", "-o", executable,
                         "--", "-I" + str(fixtures)]

    def clean_refusal(result, executable, diagnostic):
        return not os.path.lexists(executable) and checked_refusal(result, diagnostic)

    cases = [
        ("baseline", fixtures / "main.cpp", False, None),
        ("ordinary-weakref", own / "main_ordinary_weakref.cpp", True, None),
        ("ordinary-overload", own / "main_ordinary_overload.cpp", True, None),
        ("ordinary-function-pointers", own / "main_function_pointer.cpp", False, None),
        ("ordinary-host-namespace", own / "main_ordinary_weakref.cpp", "utility_namespace.cpp", None),
        ("region-host-namespace", fixtures / "main.cpp", False, None),
        ("protected-weakref-pure", own / "main_weakref_pure.cpp", False, "foreign region source alias"),
        ("protected-weakref-const", own / "main_weakref_const.cpp", False, "foreign region source alias"),
        ("protected-data-definition", own / "main_region_data_alias.cpp", False, "OWNERSHIP: competing region definition"),
    ]
    try:
        for status, stdout, stderr, expected in (
            (1, "", "output already exists", True),
            (139, "", "output already exists", False),
            (1, "", "output already exists\nAddressSanitizer", False),
            (1, "", "output already exists\nruntime error: cleanup", False),
            (1, "VALIDATED program", "output already exists", False),
            (1, "CROSS_TU_LINK", "output already exists", False),
            (1, "PUBLISHED", "output already exists", False),
            (1, "", "", False),
        ):
            control = subprocess.CompletedProcess([], status, stdout, stderr)
            demand(checked_refusal(control, "output already exists") == expected,
                   "refusal classifier accepted false evidence")
        for name, host, utility, diagnostic in cases:
            case = work / name
            case.mkdir()
            syntax = invoke([args.clang, "-x", "c++", "-std=c++20", "-fsyntax-only",
                             "-I" + str(compiler / "include"), "-I" + str(fixtures), host], case)
            demand(syntax.returncode == 0, name + " is not a valid C++ falsifier")
            executable = case / "program"
            first = own / "first_namespace.mdsl" if name == "region-host-namespace" else None
            result = invoke(command(host, executable, utility, first), case)
            if diagnostic:
                demand(clean_refusal(result, executable, diagnostic), name + " did not fail closed before link")
            else:
                demand(result.returncode == 0 and executable.is_file(), name + " did not compile")
                frozen[str(executable)] = digest(executable)
                execution = invoke([executable], case)
                demand(execution.returncode == 0 and not execution.stderr and
                       execution.stdout.strip() == "PASS authenticated_multi_tu 37 checks",
                       name + " did not execute the independent expected fixture")

        case = work / "single-region-host-namespace"
        case.mkdir()
        host = own / "single_namespace.mdsl"
        syntax = invoke([args.clang, "-x", "c++", "-std=c++20", "-fsyntax-only",
                         "-I" + str(compiler / "include"), host], case)
        demand(syntax.returncode == 0, "single-region namespace control is not valid C++")
        executable = case / "program"
        result = invoke([driver, host, "--region", "namespace_region", "--candidate",
                         "generated-strict", "-o", executable], case)
        demand(result.returncode == 0 and executable.is_file(), "single-region host namespace did not compile")
        frozen[str(executable)] = digest(executable)
        execution = invoke([executable], case)
        demand(execution.returncode == 0 and not execution.stderr and
               execution.stdout.strip() == "PASS independent original host namespace",
               "single-region host namespace did not execute")

        for kind in ("existing", "dangling-link"):
            case = work / kind
            case.mkdir()
            target = case / "program"
            if kind == "existing":
                target.write_bytes(b"owner sentinel\n")
                original = digest(target)
            else:
                target.symlink_to("missing-owner-target")
                original = os.readlink(target)
            result = invoke(command(fixtures / "main.cpp", target), case)
            demand(checked_refusal(result, "output already exists"), kind + " publication was not refused")
            demand((digest(target) if kind == "existing" else os.readlink(target)) == original,
                   kind + " owner output was altered")
        report["status"] = "pass"
        save()
        print("PASS independent multi-source", report["checks"], "checks")
    except Exception as error:
        report["status"] = "fail"
        report["error"] = str(error)
        save()
        raise


if __name__ == "__main__":
    main()
