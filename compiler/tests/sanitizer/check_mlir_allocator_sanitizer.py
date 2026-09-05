"""Check the pinned prebuilt-MLIR allocator boundary without disabling ASan."""

import argparse
import os
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mixed", required=True)
    parser.add_argument("--compatible", required=True)
    args = parser.parse_args()
    environment = os.environ.copy()
    # Symbolization can try network debuginfo; diagnostics and poisoning remain
    # enabled. Explicitly require manual poisoning for the negative control.
    environment["ASAN_OPTIONS"] = (
        environment.get("ASAN_OPTIONS", "")
        + ":symbolize=0:allow_user_poisoning=1"
    )

    def run(binary, *mode):
        result = subprocess.run(
            [binary, *mode], env=environment, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30,
            check=False,
        )
        return result.returncode, result.stdout

    status, output = run(args.mixed)
    assert status != 0 and "AddressSanitizer: use-after-poison" in output, output
    assert "builtin context constructed" not in output, output
    status, output = run(args.compatible)
    assert status == 0 and "custom singleton registered" in output, output
    for mode, diagnostic in (
        ("heap_oob", "AddressSanitizer: heap-buffer-overflow"),
        ("manual_poison", "AddressSanitizer: use-after-poison"),
    ):
        status, output = run(args.compatible, mode)
        assert status != 0 and diagnostic in output, output
        assert "custom singleton registered" in output, output
    print("MLIR allocator sanitizer controls: 4/4 passed")


if __name__ == "__main__":
    main()
