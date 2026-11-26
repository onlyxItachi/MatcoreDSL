#!/usr/bin/env python3

"""Disposable-sentinel tests of the guard AND both real runner entry points.

Positive CLI checks stop at a synthetic CMake install boundary; they are not
package build/execution qualification. Only this suite's temporary fixtures
are cleaned up. Never point a destructive-regression witness at real data.
"""

import argparse
import contextlib
import io
import json
import os
from pathlib import Path, PureWindowsPath
import shutil
import stat
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


sys.dont_write_bytecode = True
TESTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TESTS))
import installed_test_scratch as scratch


RUNNERS = {
    "installed-consumer": TESTS / "consumer" / "run_installed_consumer_test.py",
    "installed-c17-abi": TESTS / "package" / "run_installed_c17_abi_probe.py",
}
INSTALL_BOUNDARY = "MDSLC_SAFETY_TEST_INSTALL_BOUNDARY"
CMAKE = "cmake"


def snapshot(root: Path) -> dict[str, object]:
    """Record content and mutation metadata without following fixture links."""
    result: dict[str, object] = {}

    def visit(path: Path) -> None:
        status = path.lstat()
        identity = (status.st_mode, status.st_size, status.st_mtime_ns, status.st_ctime_ns)
        key = str(path.relative_to(root))
        if scratch.is_link(status):
            result[key] = (identity, os.readlink(path))
        elif stat.S_ISDIR(status.st_mode):
            result[key] = identity
            for child in sorted(path.iterdir()):
                visit(child)
        elif stat.S_ISREG(status.st_mode):
            result[key] = (identity, path.read_bytes())
        else:
            result[key] = identity

    visit(root)
    return result


class InstalledTestRootSafety(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="mdslc-installed-root-safety-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.repository = self.root / "protected-repository"
        self.repository.mkdir()
        self.source = self.root / "consumer-source"
        self.source.mkdir()
        (self.source / "must-survive").write_bytes(b"consumer source sentinel\n")
        (self.repository / "must-survive").write_bytes(b"repository sentinel\n")
        self.build = self.root / "producer-build"
        self.build.mkdir()
        (self.build / "CMakeCache.txt").write_text(
            "# Synthetic build identity; no compiler or package is executed.\n",
            encoding="utf-8",
        )
        (self.build / "cmake_install.cmake").write_text(
            f'message(FATAL_ERROR "{INSTALL_BOUNDARY}")\n', encoding="utf-8"
        )

    def registered(self, owner: str) -> Path:
        return self.build / "tests" / owner

    def prepare(self, owner: str, root: Path | None = None) -> Path:
        with contextlib.redirect_stdout(io.StringIO()):
            run, build = scratch.prepare_test_root(
                str(root if root is not None else self.registered(owner)),
                str(self.build),
                owner=owner,
                repository=self.repository,
                source_directories=(self.source,),
            )
        self.assertEqual(build, self.build)
        return run

    def copied_runner(self, owner: str) -> Path:
        # The real entry point derives its repository from __file__. Exercise
        # that boundary in a disposable checkout, never in the real one.
        tests = self.repository / "compiler" / "tests"
        original = RUNNERS[owner]
        runner = tests / original.relative_to(TESTS)
        runner.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(TESTS / "installed_test_scratch.py", tests / "installed_test_scratch.py")
        shutil.copyfile(original, runner)
        return runner

    def cli(
        self, owner: str, root: Path, *, build: Path | None = None,
        runner: Path | None = None, source: Path | None = None,
    ):
        command = [
            sys.executable,
            "-B",
            str(runner if runner is not None else RUNNERS[owner]),
            "--cmake", CMAKE,
            "--producer-build-dir", str(build if build is not None else self.build),
            "--test-root", str(root),
        ]
        if owner == "installed-consumer":
            command += [
                "--source-dir", str(self.source if source is None else source),
                "--cxx-compiler", "unused-compiler",
                "--matcore-mlir-available", "OFF",
                "--default-semantic-pipeline", "capture-v0",
                "--windows-import-report-out", str(self.root / "external-report.json"),
            ]
        else:
            command += ["--c-compiler", "unused-compiler"]
        return subprocess.run(
            command, cwd=self.root, check=False, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )

    def rejected(self, owner: str, root: Path, fragment: str, **kwargs) -> None:
        before = snapshot(self.root)
        completed = self.cli(owner, root, **kwargs)
        self.assertNotEqual(completed.returncode, 0, completed.stdout)
        self.assertIn(fragment, completed.stdout)
        self.assertNotIn(INSTALL_BOUNDARY, completed.stdout)
        self.assertEqual(snapshot(self.root), before, completed.stdout)

    def symlink(self, link: Path, target: Path, *, directory: bool = True) -> None:
        try:
            link.symlink_to(target, target_is_directory=directory)
        except OSError as error:
            if os.name == "nt" and getattr(error, "winerror", None) == 1314:
                self.skipTest("Windows token cannot create symlinks; junctions tested separately")
            raise

    def test_cli_first_and_repeated_runs_registered_and_external(self) -> None:
        for owner in RUNNERS:
            for root in (self.registered(owner), self.root / f"external {owner} ünicode"):
                with self.subTest(owner=owner, root=root):
                    first = self.cli(owner, root)
                    self.assertNotEqual(first.returncode, 0)
                    self.assertIn(INSTALL_BOUNDARY, first.stdout)
                    self.assertIn("retained (no automatic cleanup)", first.stdout)
                    first_run, = root.glob("run-*")
                    self.assertIn(str(first_run), first.stdout)
                    before = snapshot(first_run)
                    marker = (root / scratch.SENTINEL).read_bytes()
                    second = self.cli(owner, root)
                    self.assertNotEqual(second.returncode, 0)
                    self.assertIn(INSTALL_BOUNDARY, second.stdout)
                    self.assertEqual(len(list(root.glob("run-*"))), 2)
                    self.assertEqual(snapshot(first_run), before)
                    self.assertEqual((root / scratch.SENTINEL).read_bytes(), marker)

    def test_repeat_does_not_require_symlink_safe_recursive_removal(self) -> None:
        for owner in RUNNERS:
            with self.subTest(owner=owner):
                first = self.prepare(owner)
                (first / "must-survive").write_bytes(b"prior run evidence\n")
                before = snapshot(first)
                with mock.patch.object(shutil.rmtree, "avoids_symlink_attacks", False, create=True):
                    second = self.prepare(owner)
                self.assertNotEqual(first, second)
                self.assertEqual(snapshot(first), before)

    def test_prior_run_symlinks_are_not_traversed_or_removed(self) -> None:
        first = self.prepare("installed-consumer")
        self.symlink(first / "unrelated-source", self.source)
        before = snapshot(first), snapshot(self.source)
        self.prepare("installed-consumer")
        self.assertEqual((snapshot(first), snapshot(self.source)), before)

    def test_cli_unowned_empty_and_nonempty_roots(self) -> None:
        for owner in RUNNERS:
            for populated in (False, True):
                with self.subTest(owner=owner, populated=populated):
                    root = self.root / f"unowned-{owner}-{populated}"
                    root.mkdir()
                    if populated:
                        (root / "must-survive").write_bytes(b"unrelated data\n")
                    self.rejected(owner, root, "lacks a valid ownership sentinel")

    def test_cli_unowned_registered_root(self) -> None:
        for owner in RUNNERS:
            with self.subTest(owner=owner):
                root = self.registered(owner)
                root.mkdir(parents=True)
                (root / "must-survive").write_bytes(b"old unowned test artifacts\n")
                self.rejected(owner, root, "lacks a valid ownership sentinel")

    def test_cli_root_symlink(self) -> None:
        for owner in RUNNERS:
            with self.subTest(owner=owner):
                root = self.registered(owner)
                root.parent.mkdir(exist_ok=True)
                self.symlink(root, self.source)
                self.rejected(owner, root, "symlink or reparse point")

    def test_cli_dangling_root_symlink(self) -> None:
        for owner in RUNNERS:
            with self.subTest(owner=owner):
                root = self.root / f"dangling-{owner}"
                self.symlink(root, self.root / "nonexistent-victim")
                self.rejected(owner, root, "symlink or reparse point")

    def test_cli_intermediate_tests_symlink(self) -> None:
        self.symlink(self.build / "tests", self.source)
        for owner in RUNNERS:
            with self.subTest(owner=owner):
                self.rejected(owner, self.registered(owner), "symlink or reparse point")

    def test_cli_external_intermediate_symlink(self) -> None:
        parent = self.root / "linked-external-parent"
        self.symlink(parent, self.source)
        for owner in RUNNERS:
            with self.subTest(owner=owner):
                self.rejected(owner, parent / owner, "symlink or reparse point")

    def test_cli_build_symlink(self) -> None:
        build = self.root / "linked-build"
        self.symlink(build, self.build)
        for owner in RUNNERS:
            with self.subTest(owner=owner):
                self.rejected(owner, self.root / owner, "symlink or reparse point", build=build)

    def test_cli_file_root(self) -> None:
        for owner in RUNNERS:
            with self.subTest(owner=owner):
                root = self.root / f"file-{owner}"
                root.write_bytes(b"unrelated file\n")
                self.rejected(owner, root, "is not a directory")

    def test_cli_lexical_dotdot(self) -> None:
        for owner in RUNNERS:
            with self.subTest(owner=owner):
                root = self.build / "absent" / ".." / "tests" / owner
                self.rejected(owner, root, "unsafe lexical spelling")

    def test_cli_protected_build_and_ancestor(self) -> None:
        for owner in RUNNERS:
            for root in (self.build, self.root):
                with self.subTest(owner=owner, root=root):
                    self.rejected(owner, root, "equals or contains a protected path")

    def test_cli_unrelated_build_child(self) -> None:
        for owner in RUNNERS:
            with self.subTest(owner=owner):
                self.rejected(owner, self.build / "unrelated" / owner, "inside producer build must be")

    def test_cli_protected_consumer_source_and_child(self) -> None:
        self.rejected("installed-consumer", self.source, "equals or contains a protected path")
        self.rejected("installed-consumer", self.source / "absent", "inside a protected source")

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux // path aliases")
    def test_cli_double_slash_protected_paths(self) -> None:
        def alias(path: Path) -> Path:
            return Path("/" + str(path))

        for owner in RUNNERS:
            runner = self.copied_runner(owner)
            with self.subTest(owner=owner, case="repo child"):
                self.rejected(
                    owner, alias(self.repository / "absent"), "inside a protected source",
                    runner=runner,
                )
            child = self.build / "unrelated" / owner
            for root, build in ((alias(child), self.build), (child, alias(self.build))):
                with self.subTest(owner=owner, root=root, build=build):
                    self.rejected(owner, root, "inside producer build must be", build=build)
        self.rejected("installed-consumer", alias(self.source), "equals or contains a protected path")

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux // path aliases")
    def test_cli_double_slash_registered_root_reuses_canonical_marker(self) -> None:
        # Exercise the registered-child exception for a build inside the repo.
        self.build = self.build.rename(self.repository / self.build.name)
        for owner in RUNNERS:
            runner = self.copied_runner(owner)
            root = self.registered(owner)
            with self.subTest(owner=owner):
                first = self.cli(owner, Path("/" + str(root)), runner=runner)
                self.assertNotEqual(first.returncode, 0, first.stdout)
                self.assertIn(INSTALL_BOUNDARY, first.stdout)
                first_run, = root.glob("run-*")
                before = snapshot(first_run)
                marker = (root / scratch.SENTINEL).read_bytes()
                self.assertEqual(json.loads(marker)["test_root"], str(root))
                second = self.cli(owner, root, build=Path("/" + str(self.build)), runner=runner)
                self.assertNotEqual(second.returncode, 0, second.stdout)
                self.assertIn(INSTALL_BOUNDARY, second.stdout)
                self.assertEqual(len(list(root.glob("run-*"))), 2)
                self.assertEqual(snapshot(first_run), before)
                self.assertEqual((root / scratch.SENTINEL).read_bytes(), marker)

    def test_windows_namespace_classification(self) -> None:
        for spelling in ("//?/C:/temp", "//./C:/temp", "//?/UNC/server/share/temp"):
            with self.subTest(spelling=spelling), self.assertRaisesRegex(
                scratch.TestRootError, "device namespace"
            ):
                scratch.check_namespace(PureWindowsPath(spelling))
        for spelling in ("C:/runner temp/MDSLC external consumer ünicode", "//server/share/temp"):
            scratch.check_namespace(PureWindowsPath(spelling))

    @unittest.skipUnless(os.name == "nt", "native Windows device namespace test")
    def test_cli_windows_device_namespace(self) -> None:
        def device(path: Path) -> Path:
            spelling = path.as_posix()
            if spelling.startswith("//"):
                return Path("//?/UNC/" + spelling[2:])
            return Path("//?/" + spelling)

        for owner in RUNNERS:
            with self.subTest(owner=owner):
                self.rejected(owner, device(self.build / "absent"), "device namespace")
                self.rejected(owner, self.root / owner, "device namespace", build=device(self.build))
        self.rejected("installed-consumer", self.root / "external", "device namespace",
                      source=device(self.source))

    def test_protected_repository_and_home_are_unchanged(self) -> None:
        # Substitute only disposable paths: never pass a real checkout/home to
        # a deletion-regression witness, even when the guard should reject it.
        home = self.root / "synthetic-home"
        home.mkdir()
        (home / "must-survive").write_bytes(b"home sentinel\n")
        for root in (self.repository, self.repository / "absent", self.root, home):
            with self.subTest(root=root), mock.patch.object(Path, "home", return_value=home):
                before = snapshot(self.root)
                with self.assertRaises(scratch.TestRootError):
                    self.prepare("installed-consumer", root)
                self.assertEqual(snapshot(self.root), before)

    def test_cli_marker_identity_mismatches(self) -> None:
        for owner in RUNNERS:
            root = self.registered(owner)
            self.prepare(owner)
            marker = root / scratch.SENTINEL
            expected = json.loads(marker.read_bytes())
            for key, value in (
                ("schema", "foreign-schema"), ("version", 2), ("version", True),
                ("version", 1.0), ("unexpected-key", "extra"),
                ("owner", next(other for other in RUNNERS if other != owner)),
                ("producer_build", str(self.root / "other-build")),
                ("test_root", str(self.root / "other-root")),
            ):
                with self.subTest(owner=owner, key=key):
                    marker.write_text(json.dumps({**expected, key: value}), encoding="utf-8")
                    self.rejected(owner, root, "sentinel does not match")

    def test_cli_malformed_and_oversized_markers(self) -> None:
        for owner in RUNNERS:
            root = self.registered(owner)
            self.prepare(owner)
            for payload in (b"{not json", b"\xff", b"x" * (scratch.MAX_SENTINEL_BYTES + 1)):
                with self.subTest(owner=owner, size=len(payload)):
                    (root / scratch.SENTINEL).write_bytes(payload)
                    self.rejected(owner, root, "lacks a valid ownership sentinel")

    def test_cli_marker_symlink(self) -> None:
        for owner in RUNNERS:
            with self.subTest(owner=owner):
                root = self.registered(owner)
                self.prepare(owner)
                marker = root / scratch.SENTINEL
                victim = self.root / f"marker-victim-{owner}"
                marker.rename(victim)
                self.symlink(marker, victim, directory=False)
                self.rejected(owner, root, "regular non-link file")

    def test_cli_marker_directory(self) -> None:
        for owner in RUNNERS:
            with self.subTest(owner=owner):
                root = self.registered(owner)
                root.mkdir(parents=True)
                (root / scratch.SENTINEL).mkdir()
                self.rejected(owner, root, "regular non-link file")

    def test_cli_missing_or_linked_build_cache(self) -> None:
        missing = self.root / "unconfigured-build"
        missing.mkdir()
        for owner in RUNNERS:
            with self.subTest(owner=owner, case="missing"):
                self.rejected(owner, self.root / owner, "no regular CMakeCache.txt", build=missing)
        cache = self.build / "CMakeCache.txt"
        victim = self.root / "cache-victim"
        cache.rename(victim)
        self.symlink(cache, victim, directory=False)
        for owner in RUNNERS:
            with self.subTest(owner=owner, case="linked"):
                self.rejected(owner, self.root / owner, "no regular CMakeCache.txt")

    def test_reparse_point_is_a_link_even_without_symlink_mode(self) -> None:
        self.assertTrue(scratch.is_link(SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=0x400)))
        self.assertTrue(scratch.is_link(SimpleNamespace(st_mode=stat.S_IFREG, st_file_attributes=0x400)))
        self.assertFalse(scratch.is_link(SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=0)))

    @unittest.skipUnless(os.name == "nt", "native Windows junction test")
    def test_cli_windows_junction_root_and_intermediate(self) -> None:
        for owner in RUNNERS:
            for intermediate in (False, True):
                with self.subTest(owner=owner, intermediate=intermediate):
                    link = self.root / f"junction-{owner}-{intermediate}"
                    made = subprocess.run(
                        ["cmd", "/d", "/c", "mklink", "/J", str(link), str(self.source)],
                        check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    )
                    self.assertEqual(made.returncode, 0, made.stdout)
                    root = link / "absent-child" if intermediate else link
                    self.rejected(owner, root, "symlink or reparse point")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cmake", required=True)
    args = parser.parse_args()
    CMAKE = args.cmake
    unittest.main(argv=[sys.argv[0]], verbosity=2)
