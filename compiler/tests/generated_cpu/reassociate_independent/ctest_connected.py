#!/usr/bin/env python3
"""Run the independent matrix against a fresh SDK without hiding build sources."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmake", required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--driver", required=True)
    parser.add_argument("--cxx", required=True)
    parser.add_argument("--provider")
    parser.add_argument("--sanitized", action="store_true")
    args = parser.parse_args()
    # Retain the unique evidence directory on success, failure and true skip.
    # connected_test refuses an existing output; no previous run is overwritten.
    root = Path(tempfile.mkdtemp(prefix="reassociate-connected-", dir=args.build))
    sdk = root / "sdk"
    environment = os.environ.copy()
    for name in ("LD_LIBRARY_PATH", "LD_PRELOAD", "LD_AUDIT"):
        environment.pop(name, None)
    print(f"Independent connected evidence: {root}", flush=True)
    installation = subprocess.run(
        [args.cmake, "--install", str(args.build), "--prefix", str(sdk)],
        env=environment, text=True, capture_output=True, timeout=180)
    (root / "install.stdout").write_text(installation.stdout)
    (root / "install.stderr").write_text(installation.stderr)
    if installation.returncode != 0 or installation.stderr:
        print(installation.stdout, end="")
        print(installation.stderr, end="", file=sys.stderr)
        raise RuntimeError(f"Fresh SDK installation failed: {installation.returncode}")
    command = [sys.executable, str(Path(__file__).with_name("connected_test.py")),
               "--driver", args.driver, "--cxx", args.cxx,
               "--sdk", str(sdk), "--output", str(root / "matrix")]
    if args.provider:
        command += ["--provider", args.provider]
    if args.sanitized:
        command += ["--sanitized"]
    result = subprocess.run(command, env=environment, timeout=400)
    # Only the matrix's exact recognized hardware-unavailable result becomes a
    # CTest skip. Other nonzero statuses, including signals, are failures.
    return 77 if result.returncode == 77 else (0 if result.returncode == 0 else 1)


if __name__ == "__main__":
    sys.exit(main())
