#!/usr/bin/env python3

"""Adversarial deletion-safety tests for the source-inaccessible package gate."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import shutil
import subprocess
import tempfile


def load_test_module(path: Path):
    specification = importlib.util.spec_from_file_location(
        "source_inaccessible_consumer", path
    )
    if specification is None or specification.loader is None:
        raise AssertionError(f"cannot load source-inaccessible test module: {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def expect_rejection(module, callback, expected_fragment: str) -> None:
    try:
        callback()
    except module.TestFailure as error:
        if expected_fragment not in str(error):
            raise AssertionError(
                f"rejection omitted {expected_fragment!r}: {error}"
            ) from error
        return
    raise AssertionError("unsafe test-root request was accepted")


def make_build_root(root: Path, name: str = "build") -> Path:
    build = root / name
    build.mkdir()
    (build / "CMakeCache.txt").write_text(
        "# synthetic configured build identity\n", encoding="utf-8"
    )
    return build


def run_checked(command: list[str], cwd: Path | None = None) -> None:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"setup command failed ({completed.returncode}): {command}\n"
            f"{completed.stdout}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", required=True)
    parser.add_argument("--git", required=True)
    arguments = parser.parse_args()
    module = load_test_module(Path(arguments.script).resolve())

    with tempfile.TemporaryDirectory(
        prefix="matcoredsl-source-inaccessible-safety-"
    ) as temporary:
        root = Path(temporary)
        repository = root / "protected-repository"
        repository.mkdir()
        build = make_build_root(root)
        expected = build / "tests" / module.TEST_ROOT_BASENAME

        prepared, prepared_build = module.prepare_test_root(
            str(expected), str(build), repository
        )
        assert prepared == expected
        assert prepared_build == build
        assert (prepared / module.TEST_ROOT_SENTINEL).is_file()
        generated = prepared / "generated-output"
        generated.write_text("generated\n", encoding="utf-8")
        module.prepare_test_root(str(expected), str(build), repository)
        assert not generated.exists()

        mismatch = root / module.TEST_ROOT_BASENAME
        mismatch.mkdir()
        mismatch_marker = mismatch / "must-survive"
        mismatch_marker.write_text("user data\n", encoding="utf-8")
        expect_rejection(
            module,
            lambda: module.prepare_test_root(
                str(mismatch), str(build), repository
            ),
            "test root mismatch",
        )
        assert mismatch_marker.read_text(encoding="utf-8") == "user data\n"

        shutil.rmtree(expected)
        victim = root / "symlink-victim"
        victim.mkdir()
        victim_marker = victim / "must-survive"
        victim_marker.write_text("user data\n", encoding="utf-8")
        expected.symlink_to(victim, target_is_directory=True)
        expect_rejection(
            module,
            lambda: module.prepare_test_root(
                str(expected), str(build), repository
            ),
            "test root must not be a symlink",
        )
        assert victim_marker.read_text(encoding="utf-8") == "user data\n"
        expected.unlink()

        expected.mkdir()
        unsentinelled_marker = expected / "must-survive"
        unsentinelled_marker.write_text("user data\n", encoding="utf-8")
        expect_rejection(
            module,
            lambda: module.prepare_test_root(
                str(expected), str(build), repository
            ),
            "without its valid safety sentinel",
        )
        assert unsentinelled_marker.read_text(encoding="utf-8") == "user data\n"
        shutil.rmtree(expected)

        dotdot = build / "tests" / "spare" / ".." / module.TEST_ROOT_BASENAME
        expect_rejection(
            module,
            lambda: module.prepare_test_root(
                str(dotdot), str(build), repository
            ),
            "must not contain '..'",
        )

        real_build = make_build_root(root, "real-build")
        linked_build = root / "linked-build"
        linked_build.symlink_to(real_build, target_is_directory=True)
        linked_expected = (
            linked_build / "tests" / module.TEST_ROOT_BASENAME
        )
        expect_rejection(
            module,
            lambda: module.prepare_test_root(
                str(linked_expected), str(linked_build), repository
            ),
            "contains a symlink",
        )

        containing_build = make_build_root(root, "containing-build")
        containing_root = (
            containing_build / "tests" / module.TEST_ROOT_BASENAME
        )
        protected_inside = containing_root / "real-checkout"
        protected_inside.mkdir(parents=True)
        checkout_marker = protected_inside / "must-survive"
        checkout_marker.write_text("source\n", encoding="utf-8")
        expect_rejection(
            module,
            lambda: module.prepare_test_root(
                str(containing_root), str(containing_build), protected_inside
            ),
            "must not equal or contain the real source checkout",
        )
        assert checkout_marker.read_text(encoding="utf-8") == "source\n"

        authenticated = root / "authenticated-repository"
        authenticated.mkdir()
        run_checked([arguments.git, "init", "--quiet"], cwd=authenticated)
        run_checked(
            [arguments.git, "config", "user.name", "MatcoreDSL test"],
            cwd=authenticated,
        )
        run_checked(
            [arguments.git, "config", "user.email", "test@matcoredsl.invalid"],
            cwd=authenticated,
        )
        tracked = authenticated / "tracked.txt"
        tracked.write_text("first\n", encoding="utf-8")
        run_checked([arguments.git, "add", "tracked.txt"], cwd=authenticated)
        run_checked(
            [arguments.git, "commit", "--quiet", "-m", "first"],
            cwd=authenticated,
        )
        first_commit = module.exact_commit(arguments.git, authenticated)
        assert (
            module.authenticate_repository(
                arguments.git, authenticated, first_commit, True
            )
            == first_commit
        )
        expect_rejection(
            module,
            lambda: module.authenticate_repository(
                arguments.git, authenticated, first_commit, False
            ),
            "dirty when the package test was configured",
        )
        tracked.write_text("dirty\n", encoding="utf-8")
        expect_rejection(
            module,
            lambda: module.authenticate_repository(
                arguments.git, authenticated, first_commit, True
            ),
            "became dirty",
        )
        tracked.write_text("second\n", encoding="utf-8")
        run_checked([arguments.git, "add", "tracked.txt"], cwd=authenticated)
        run_checked(
            [arguments.git, "commit", "--quiet", "-m", "second"],
            cwd=authenticated,
        )
        expect_rejection(
            module,
            lambda: module.authenticate_repository(
                arguments.git, authenticated, first_commit, True
            ),
            "HEAD changed",
        )
        expect_rejection(
            module,
            lambda: module.authenticate_repository(
                arguments.git, authenticated, "not-a-commit", True
            ),
            "canonical commit ID",
        )

    print("source-inaccessible package deletion safety: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
