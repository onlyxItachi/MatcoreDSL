#!/usr/bin/env python3
"""Synthetic runner classifiers/pins, never evidence of generated execution."""
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("connected_runner", Path(__file__).with_name("connected_test.py"))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
checks = 0


def case(name, action, expected_error=False):
    global checks
    checks += 1
    try:
        action()
    except (RuntimeError, FileNotFoundError):
        if expected_error:
            return
        raise
    if expected_error:
        raise AssertionError("Accepted negative control: " + name)


def result(code, stdout="", stderr=""):
    return SimpleNamespace(returncode=code, stdout=stdout, stderr=stderr)


skip = "SKIP independent connected source: AVX2/FMA unavailable\n"
case("clean positive", lambda: runner.classify_result(result(0)))
case("positive diagnostic", lambda: runner.classify_result(result(0, stderr="warning\n")), True)
case("positive hidden ASan", lambda: runner.classify_result(result(0, stderr="AddressSanitizer: error\n")), True)
case("exact skip", lambda: runner.classify_result(result(77, skip), allowed_skip=skip))
case("unapproved skip", lambda: runner.classify_result(result(77, skip)), True)
case("wrong skip text", lambda: runner.classify_result(result(77, "SKIP\n"), allowed_skip=skip), True)
case("diagnostic skip", lambda: runner.classify_result(result(77, skip, "runtime error: bad\n"), allowed_skip=skip), True)
case("wrong skip exit", lambda: runner.classify_result(result(1, skip), allowed_skip=skip), True)
case("crashed negative", lambda: runner.classify_result(result(-11), expected=1), True)
case("wrong negative exit", lambda: runner.classify_result(result(2), expected=1), True)
stdout = "Independent connected source: 2980 checks; 21 failures\n"
stderr = "FAIL: first observation preserves exact math\n" * 21
case("exact math negative", lambda: runner.source_negative(stdout, stderr))
case("appended sanitizer", lambda: runner.source_negative(stdout, stderr + "AddressSanitizer: error\n"), True)
case("substituted sanitizer", lambda: runner.source_negative(stdout, stderr.splitlines(True)[0] * 20 + "runtime error: overflow\n"), True)
case("wrong negative identity", lambda: runner.source_negative(stdout.replace("2980", "2979"), stderr), True)
case("missing negative line", lambda: runner.source_negative(stdout, stderr.splitlines(True)[0] * 20), True)
with tempfile.TemporaryDirectory(prefix="mdslc-connected-pin-controls-") as directory:
    source = Path(directory) / "input"
    executable = Path(directory) / "produced"
    source.write_bytes(b"frozen source")
    executable.write_bytes(b"frozen executable")
    inputs = {str(source): runner.digest(source)}
    outputs = {str(executable): runner.digest(executable)}
    case("frozen inputs", lambda: runner.verify_files(inputs))
    case("frozen outputs", lambda: runner.verify_files(outputs))
    source.write_bytes(b"changed source")
    case("changed input", lambda: runner.verify_files(inputs), True)
    executable.write_bytes(b"changed executable")
    case("changed prior executable", lambda: runner.verify_files(outputs), True)
    executable.unlink()
    case("missing prior executable", lambda: runner.verify_files(outputs), True)
print(f"Independent connected runner controls: {checks} checks; 0 failures")
