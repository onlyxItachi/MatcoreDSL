#!/usr/bin/env python3
"""Synthetic classifier controls, never evidence that generated code executed."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def fixture():
    variant = os.environ["MDSLC_REASSOCIATE_CLASSIFIER_FIXTURE"]
    text = (
        "==123==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x1234\n"
        "READ of size 32 at 0x1234 thread T0\n"
        "    #0 0xabcd in __matcore_reassociate_gemm_f32_avx2_v1 /tmp/leaf.ll:1\n"
    )
    status = 1
    if variant.startswith("exit-"):
        status = int(variant.removeprefix("exit-"))
    elif variant == "scalar":
        text = text.replace("size 32", "size 4")
    elif variant == "write":
        text = text.replace("READ of", "WRITE of")
    elif variant == "non-top":
        text = text.replace("#0", "#1")
    elif variant == "wrong-owner":
        text = text.replace("in __matcore_reassociate_gemm_f32_avx2_v1", "in unrelated_owner")
    elif variant == "allocation-owner":
        text = text.replace("in __matcore_reassociate_gemm_f32_avx2_v1", "in unrelated_owner")
        text += ("\nallocated by thread T0 here:\n"
                 "    #0 0xcdef in __matcore_reassociate_gemm_f32_avx2_v1 /tmp/leaf.ll:1\n")
    elif variant == "wrong-kind":
        text = text.replace("heap-buffer-overflow", "stack-buffer-overflow")
    elif variant == "no-error":
        text = text.split("\n", 1)[1]
    elif variant == "valid-a":
        text = text.replace("size 32", "size 4")
    elif variant != "valid-b":
        raise AssertionError(variant)
    sys.stderr.write(text)
    return status


def main():
    if "MDSLC_REASSOCIATE_CLASSIFIER_FIXTURE" in os.environ:
        return fixture()
    assert len(sys.argv) == 2, "usage: classifier_test.py expect_reassociate.sh"
    wrapper = Path(sys.argv[1]).resolve(strict=True)
    expected = {"valid-a": 0, "valid-b": 0, "exit-77": 77}
    expected.update({name: 1 for name in (
        "exit-0", "exit-2", "exit-99", "exit-134", "scalar", "write",
        "non-top", "wrong-owner", "allocation-owner", "wrong-kind", "no-error")})
    with tempfile.TemporaryDirectory(prefix="mdslc-reassociate-classifier-") as root:
        for variant, status in expected.items():
            environment = dict(os.environ, MDSLC_REASSOCIATE_CLASSIFIER_FIXTURE=variant)
            result = subprocess.run(
                ["bash", str(wrapper), str(Path(__file__).resolve()),
                 "read-a" if variant == "valid-a" else "read-b", str(Path(root) / variant)],
                env=environment, capture_output=True, text=True, check=False)
            assert result.returncode == status, (
                variant, result.returncode, status, result.stdout, result.stderr)
    print(f"Independent reassociate read classifier: {len(expected)} controls passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
