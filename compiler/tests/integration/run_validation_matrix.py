#!/usr/bin/env python3
"""Run only validation cases marked active for the current milestone."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def run(argv: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, cwd=cwd, text=True, capture_output=True, check=False)


def execute(case: dict[str, object], driver: Path, fixture_root: Path) -> tuple[bool, str]:
    fixture_value = case.get("fixture")
    fixture = fixture_root / str(fixture_value) if fixture_value else None
    mode = str(case["mode"])

    with tempfile.TemporaryDirectory(prefix=f"mdslc-{case['id'].lower()}-") as raw:
        work = Path(raw)

        if mode == "host_run":
            output = work / "program"
            compiled = run(
                [str(driver), "--verbose", "-std=c++20", str(fixture), "-o", str(output)],
                work,
            )
            if compiled.returncode != 0:
                return False, compiled.stderr
            if "'-x' 'c++'" not in compiled.stderr:
                return False, "verbose command did not show forced -x c++"
            executed = run([str(output)], work)
            expected = str(case.get("stdout", ""))
            return (
                executed.returncode == 0 and executed.stdout == expected,
                f"exit={executed.returncode} stdout={executed.stdout!r}",
            )

        if mode == "host_object":
            output = work / "host.o"
            compiled = run(
                [str(driver), "--save-temps", "-std=c++20", "-c", str(fixture), "-o", str(output)],
                work,
            )
            if compiled.returncode != 0:
                return False, compiled.stderr
            header = output.read_bytes()[:4] if output.exists() else b""
            elf = run(["readelf", "-h", str(output)], work)
            symbols = run(["nm", "-C", str(output)], work)
            temps = all(any(work.glob(f"*.{suffix}")) for suffix in ("ii", "bc", "s"))
            ok = (
                header == b"\x7fELF"
                and elf.returncode == 0
                and "REL (Relocatable file)" in elf.stdout
                and symbols.returncode == 0
                and " main" in symbols.stdout
                and temps
            )
            return ok, "ELF REL, main symbol, and .ii/.bc/.s/.o required"

        if mode == "missing_input":
            missing = work / "does-not-exist.mdsl"
            compiled = run([str(driver), "-std=c++20", str(missing), "-o", str(work / "missing")], work)
            ok = compiled.returncode == 1 and str(missing) in compiled.stderr and "no such file" in compiled.stderr
            return ok, f"exit={compiled.returncode} stderr={compiled.stderr!r}"

        if mode == "metachar_argv":
            source = work / "hello;touch injection-marker.mdsl"
            marker = work / "injection-marker.mdsl"
            shutil.copyfile(fixture, source)
            output = work / "program"
            compiled = run(
                [str(driver), "-std=c++20", "-o", str(output), source.name], work
            )
            executed = run([str(output)], work) if compiled.returncode == 0 else None
            ok = (
                compiled.returncode == 0
                and executed is not None
                and executed.returncode == 0
                and executed.stdout == "5\n"
                and not marker.exists()
            )
            return ok, f"compile_exit={compiled.returncode} marker_exists={marker.exists()}"

        if mode == "double_dash":
            output = work / "program"
            compiled = run(
                [str(driver), "-std=c++20", "-o", str(output), "--", str(fixture)], work
            )
            ok = compiled.returncode == 0 and output.exists()
            return ok, f"exit={compiled.returncode} stderr={compiled.stderr!r}"

    return False, f"unsupported active mode: {mode}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("validation_matrix.json"))
    parser.add_argument("--include-known-failures", action="store_true")
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    if manifest.get("schema") != "matcore.validation-matrix" or manifest.get("version") != 1:
        print("invalid validation manifest schema/version", file=sys.stderr)
        return 2

    cases = manifest["cases"]
    ids = [case["id"] for case in cases]
    if len(ids) != len(set(ids)):
        print("duplicate validation case ID", file=sys.stderr)
        return 2

    fixture_root = args.manifest.resolve().parent.parent / "fixtures"
    for case in cases:
        fixture = case.get("fixture")
        if fixture and not (fixture_root / fixture).is_file():
            print(f"{case['id']}: missing fixture {fixture}", file=sys.stderr)
            return 2

    driver = args.build_dir.resolve() / "bin" / "mdslc++"
    if not driver.is_file():
        print(f"missing built driver: {driver}", file=sys.stderr)
        return 2

    failed = 0
    passed = 0
    xfailed = 0
    for case in cases:
        state = case["state"]
        if state == "active" or (state == "known_failure" and args.include_known_failures):
            ok, detail = execute(case, driver, fixture_root)
            if state == "known_failure":
                if ok:
                    print(f"XPASS {case['id']}: manifest must be updated")
                    failed += 1
                else:
                    print(f"XFAIL {case['id']}: {detail}")
                    xfailed += 1
            elif ok:
                print(f"PASS  {case['id']}")
                passed += 1
            else:
                print(f"FAIL  {case['id']}: {detail}")
                failed += 1

    pending = sum(case["state"] == "pending" for case in cases)
    known = sum(case["state"] == "known_failure" for case in cases)
    print(f"summary: pass={passed} fail={failed} pending={pending} known_failure={known} xfail_run={xfailed}")
    if failed:
        return 1
    if args.require_all and (pending or known):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
