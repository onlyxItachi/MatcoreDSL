#!/usr/bin/env python3

"""Prove that the installed package does not need its producer source/build tree."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


class TestFailure(RuntimeError):
    """The installed package retained an undeclared producer dependency."""


TEST_ROOT_BASENAME = "installed-source-inaccessible"
TEST_ROOT_SENTINEL = ".matcoredsl-source-inaccessible-test-root-v1.json"
TEST_ROOT_SENTINEL_SCHEMA = "matcoredsl.source-inaccessible-test-root"


def run(
    command: list[str],
    *,
    capture: bool = False,
    cwd: Path | None = None,
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        check=False,
        cwd=cwd,
        env=environment,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )
    if completed.returncode != 0:
        if completed.stdout:
            print(completed.stdout, file=sys.stderr)
        raise TestFailure(
            f"command returned {completed.returncode}: {command}"
        )
    return completed


def exact_commit(git: str, repository: Path) -> str:
    commit = run(
        [git, "-C", str(repository), "rev-parse", "--verify", "HEAD"],
        capture=True,
    ).stdout.strip()
    if len(commit) not in {40, 64} or any(
        character not in "0123456789abcdefABCDEF" for character in commit
    ):
        raise TestFailure(f"Git returned an invalid HEAD object ID: {commit!r}")
    return commit.lower()


def absolute_tool(spelling: str, description: str) -> Path:
    # Preserve the caller-visible basename. Clang selects C versus C++ driver
    # semantics from argv[0], and resolving the clang++ symlink changes it to
    # the C driver on the audited Linux toolchain.
    path = Path(os.path.abspath(spelling))
    if not path.is_file():
        raise TestFailure(f"{description} is not an existing file: {path}")
    return path


def lexical_absolute(spelling: str, description: str) -> Path:
    if ".." in Path(spelling).parts:
        raise TestFailure(f"{description} must not contain '..': {spelling}")
    return Path(os.path.abspath(spelling))


def validate_no_symlink_resolution(path: Path, description: str) -> None:
    if path.resolve() != path:
        raise TestFailure(
            f"{description} contains a symlink or non-canonical component: {path}"
        )


def sentinel_payload(build_root: Path, test_root: Path) -> dict[str, object]:
    return {
        "schema": TEST_ROOT_SENTINEL_SCHEMA,
        "version": 1,
        "expected_build_root": str(build_root),
        "test_root": str(test_root),
    }


def prepare_test_root(
    test_root_spelling: str,
    expected_build_root_spelling: str,
    protected_repository: Path,
) -> tuple[Path, Path]:
    build_root = lexical_absolute(
        expected_build_root_spelling, "expected CMake build root"
    )
    test_root = lexical_absolute(test_root_spelling, "test root")
    validate_no_symlink_resolution(build_root, "expected CMake build root")
    cmake_cache = build_root / "CMakeCache.txt"
    if (
        not build_root.is_dir()
        or cmake_cache.is_symlink()
        or not cmake_cache.is_file()
    ):
        raise TestFailure(
            "expected build root is not a configured CMake build tree: "
            f"{build_root}"
        )

    expected_test_root = build_root / "tests" / TEST_ROOT_BASENAME
    if test_root != expected_test_root:
        raise TestFailure(
            f"test root mismatch: expected {expected_test_root}, got {test_root}"
        )
    if (
        test_root == protected_repository
        or test_root in protected_repository.parents
    ):
        raise TestFailure(
            "test root must not equal or contain the real source checkout"
        )

    tests_directory = expected_test_root.parent
    if tests_directory.is_symlink():
        raise TestFailure(
            f"CMake tests directory must not be a symlink: {tests_directory}"
        )
    if tests_directory.exists():
        validate_no_symlink_resolution(tests_directory, "CMake tests directory")
        if not tests_directory.is_dir():
            raise TestFailure(
                f"CMake tests path is not a directory: {tests_directory}"
            )
    else:
        tests_directory.mkdir(parents=True)
        validate_no_symlink_resolution(tests_directory, "CMake tests directory")

    expected_sentinel = sentinel_payload(build_root, test_root)
    sentinel = test_root / TEST_ROOT_SENTINEL
    if test_root.is_symlink():
        raise TestFailure(f"test root must not be a symlink: {test_root}")
    if test_root.exists():
        validate_no_symlink_resolution(test_root, "test root")
        if not test_root.is_dir():
            raise TestFailure(f"test root is not a directory: {test_root}")
        try:
            actual_sentinel = json.loads(sentinel.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise TestFailure(
                "refusing to remove an existing test root without its valid "
                f"safety sentinel: {test_root}: {error}"
            ) from error
        if actual_sentinel != expected_sentinel:
            raise TestFailure(
                "refusing to remove an existing test root whose safety "
                f"sentinel does not match this configured build: {test_root}"
            )
        if not shutil.rmtree.avoids_symlink_attacks:
            raise TestFailure(
                "this platform lacks symlink-safe recursive removal; refusing "
                f"to reset the existing test root: {test_root}"
            )
        shutil.rmtree(test_root)

    test_root.mkdir()
    sentinel.write_text(
        json.dumps(expected_sentinel, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return test_root, build_root


def clone_clean_commit(
    git: str, repository: Path, destination: Path, commit: str
) -> None:
    run(
        [
            git,
            "clone",
            "--no-hardlinks",
            "--no-checkout",
            "--quiet",
            str(repository),
            str(destination),
        ]
    )
    run([git, "-C", str(destination), "checkout", "--detach", "--quiet", commit])
    cloned_commit = exact_commit(git, destination)
    if cloned_commit != commit:
        raise TestFailure(
            f"disposable clone resolved {cloned_commit}, expected {commit}"
        )
    status = run(
        [
            git,
            "-C",
            str(destination),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ],
        capture=True,
    ).stdout
    if status:
        raise TestFailure(f"disposable producer source is not clean:\n{status}")


def forbidden_spellings(paths: list[Path]) -> tuple[bytes, ...]:
    spellings: set[bytes] = set()
    for path in paths:
        absolute = str(path.resolve())
        spellings.add(absolute.encode("utf-8"))
        spellings.add(Path(absolute).as_posix().encode("utf-8"))
    return tuple(sorted(spellings))


def scan_forbidden_paths(
    root: Path, spellings: tuple[bytes, ...], description: str
) -> None:
    for candidate in root.rglob("*"):
        if candidate.is_symlink():
            link = os.readlink(candidate).encode("utf-8")
            matched = next(
                (spelling for spelling in spellings if spelling in link), None
            )
            if matched is not None:
                raise TestFailure(
                    f"{description} symlink {candidate} leaks producer path "
                    f"{matched.decode('utf-8')}"
                )
            continue
        if not candidate.is_file():
            continue
        contents = candidate.read_bytes()
        matched = next(
            (spelling for spelling in spellings if spelling in contents), None
        )
        if matched is not None:
            raise TestFailure(
                f"{description} file {candidate} leaks producer path "
                f"{matched.decode('utf-8')}"
            )


def require_install(prefix: Path) -> None:
    required = (
        prefix / "bin" / "mdslc++",
        prefix / "bin" / "matcore-extract",
        prefix / "bin" / "matcore-plan",
        prefix / "bin" / "matcore-bench",
        prefix / "include" / "matcore" / "mdsl.h",
        prefix / "include" / "matcore" / "runtime_c.h",
        prefix / "lib" / "libmatcore_runtime.so",
        prefix / "lib" / "cmake" / "MatcoreDSL" / "MatcoreDSLConfig.cmake",
        prefix
        / "lib"
        / "cmake"
        / "MatcoreDSL"
        / "MatcoreDSLCompile.cmake",
        prefix
        / "lib"
        / "cmake"
        / "MatcoreDSL"
        / "MatcoreDSLTargets.cmake",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise TestFailure(f"relocated installed package is incomplete: {missing}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmake", required=True)
    parser.add_argument("--git", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--expected-build-root", required=True)
    parser.add_argument("--test-root", required=True)
    parser.add_argument("--c-compiler", required=True)
    parser.add_argument("--cxx-compiler", required=True)
    parser.add_argument("--clangxx", required=True)
    parser.add_argument("--llvm-dir", required=True)
    parser.add_argument("--clang-dir", required=True)
    arguments = parser.parse_args()

    repository = Path(arguments.repository_root).resolve()
    c_compiler = absolute_tool(arguments.c_compiler, "C compiler")
    cxx_compiler = absolute_tool(arguments.cxx_compiler, "C++ compiler")
    clangxx = absolute_tool(arguments.clangxx, "Clang C++ driver")
    test_root, _ = prepare_test_root(
        arguments.test_root,
        arguments.expected_build_root,
        repository,
    )

    producer_source = test_root / "producer-source"
    producer_build = test_root / "producer-build"
    staging_prefix = test_root / "install-staging"
    relocated_prefix = (
        test_root / "relocated package" / "MatcoreDSL prefix ünicode"
    )
    consumer_source = test_root / "consumer-only"
    consumer_build = test_root / "consumer-build"

    commit = exact_commit(arguments.git, repository)
    clone_clean_commit(
        arguments.git, repository, producer_source, commit
    )
    shutil.copytree(
        producer_source / "compiler" / "tests" / "consumer",
        consumer_source,
    )

    configure = [
        arguments.cmake,
        "-S",
        str(producer_source / "compiler"),
        "-B",
        str(producer_build),
        "-G",
        "Ninja",
        f"-DCMAKE_C_COMPILER={c_compiler}",
        f"-DCMAKE_CXX_COMPILER={cxx_compiler}",
        f"-DMDSLC_CLANGXX_EXECUTABLE={clangxx}",
        "-DMDSLC_ENABLE_NATIVE_FRONTEND=ON",
        "-DMDSLC_ENABLE_BOOTSTRAP_FRONTEND=ON",
        "-DMDSLC_ENABLE_OPENBLAS=OFF",
        "-DMDSLC_REQUIRE_OPENBLAS=OFF",
        f"-DLLVM_DIR={Path(arguments.llvm_dir).resolve()}",
        f"-DClang_DIR={Path(arguments.clang_dir).resolve()}",
        "-DBUILD_TESTING=OFF",
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    run(configure)
    run([arguments.cmake, "--build", str(producer_build), "--parallel", "2"])
    run(
        [
            arguments.cmake,
            "--install",
            str(producer_build),
            "--prefix",
            str(staging_prefix),
        ]
    )
    relocated_prefix.parent.mkdir(parents=True)
    shutil.move(staging_prefix, relocated_prefix)
    require_install(relocated_prefix)

    forbidden_paths = [
        repository,
        producer_source,
        producer_build,
        staging_prefix,
    ]
    spellings = forbidden_spellings(forbidden_paths)
    scan_forbidden_paths(relocated_prefix, spellings, "installed package")
    scan_forbidden_paths(consumer_source, spellings, "copied consumer source")

    # These are disposable directories created by this test. Removing them
    # before consumer configuration makes stale source/build dependencies fail
    # rather than merely proving that the package prefers its installed files.
    shutil.rmtree(producer_source)
    shutil.rmtree(producer_build)
    if producer_source.exists() or producer_build.exists():
        raise TestFailure("producer source or build tree remained accessible")

    run(
        [
            arguments.cmake,
            "-S",
            str(consumer_source),
            "-B",
            str(consumer_build),
            "-G",
            "Ninja",
            f"-DCMAKE_PREFIX_PATH={relocated_prefix}",
            f"-DCMAKE_CXX_COMPILER={cxx_compiler}",
            "-DCMAKE_BUILD_TYPE=Release",
        ]
    )
    run(
        [
            arguments.cmake,
            "--build",
            str(consumer_build),
            "--parallel",
            "2",
        ]
    )

    executable = consumer_build / "matcore_consumer"
    if not executable.is_file():
        raise TestFailure(f"consumer executable is missing: {executable}")
    environment = os.environ.copy()
    prior_library_path = environment.get("LD_LIBRARY_PATH")
    environment["LD_LIBRARY_PATH"] = str(relocated_prefix / "lib")
    if prior_library_path:
        environment["LD_LIBRARY_PATH"] += os.pathsep + prior_library_path
    executed = run([str(executable)], capture=True, environment=environment)
    for expected in ("consumer-before", "consumer-header=1", "consumer-pass"):
        if expected not in executed.stdout:
            raise TestFailure(
                f"source-inaccessible consumer omitted {expected!r}:\n"
                f"{executed.stdout}"
            )

    scan_forbidden_paths(relocated_prefix, spellings, "installed package")
    scan_forbidden_paths(consumer_build, spellings, "consumer build")
    if producer_source.exists() or producer_build.exists():
        raise TestFailure(
            "consumer execution recreated or accessed the producer tree"
        )

    print(
        "installed source/build-inaccessible consumer: "
        f"commit={commit} configure/build/run PASS"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except TestFailure as error:
        print(f"source-inaccessible package test failed: {error}", file=sys.stderr)
        raise SystemExit(1)
