#!/usr/bin/env python3
"""Process/diagnostic transport regression; runs on Linux and native Windows."""
from pathlib import Path
import sys
import tempfile
import unittest

from run_windows_native_pipeline import run, start


class DiagnosticCapture(unittest.TestCase):
    def test_mixed_diagnostic_bytes_preserve_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run([
                sys.executable, "-c",
                "import os; os.write(1, b'valid UTF-8: \\xcf\\x80\\n'); "
                "os.write(2, b'reserved device output path: \\xfc\\xff\\n'); "
                "raise SystemExit(7)",
            ], cwd=Path(directory))
        self.assertEqual(result.returncode, 7)
        self.assertEqual(result.stdout, "valid UTF-8: π\n")
        self.assertEqual(result.stderr, "reserved device output path: \\xfc\\xff\n")

    def test_success_is_not_changed(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run([sys.executable, "-c", "print('PASS')"], cwd=Path(directory))
        self.assertEqual((result.returncode, result.stdout, result.stderr), (0, "PASS\n", ""))

    def test_async_mixed_bytes_preserve_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            with start([
                sys.executable, "-c",
                "import os; os.write(1, b'valid UTF-8: \\xcf\\x80\\n'); "
                "os.write(2, b'dependency changed: \\xfc\\xff\\n'); "
                "raise SystemExit(7)",
            ], cwd=Path(directory)) as process:
                stdout, stderr = process.communicate(timeout=10)
            self.assertEqual(process.returncode, 7)
        self.assertEqual(stdout, "valid UTF-8: π\n")
        self.assertEqual(stderr, "dependency changed: \\xfc\\xff\n")

    def test_async_success_is_not_changed(self):
        with tempfile.TemporaryDirectory() as directory:
            with start([sys.executable, "-c", "print('PASS')"], cwd=Path(directory)) as process:
                stdout, stderr = process.communicate(timeout=10)
            self.assertEqual(process.returncode, 0)
        self.assertEqual((stdout, stderr), ("PASS\n", ""))


if __name__ == "__main__":
    unittest.main()
