"""Owned, non-destructive scratch containers for installed-package tests.

Paths must remain under the caller's control during a run. These checks guard
against mistaken/reused paths, not a concurrently hostile filesystem. No old
run is removed: each invocation gets a new exclusive child on every platform.
"""

import json
import os
from pathlib import Path, PurePath
import stat
import tempfile


SENTINEL = ".matcoredsl-installed-test-root-v1.json"
MAX_SENTINEL_BYTES = 4096
OWNERS = {"installed-consumer", "installed-c17-abi"}


class TestRootError(RuntimeError):
    """A requested scratch container has no safe ownership contract."""


def check_namespace(path: PurePath) -> None:
    # Windows realpath can retain a device prefix for an explicitly prefixed
    # input but drop it for the same ordinary drive/UNC path. Do not mix those
    # namespaces in either side of a containment check or in a marker.
    if path.drive.startswith(("\\\\?\\", "\\\\.\\")):
        raise TestRootError(f"Windows device namespace is not supported: {path}")


def resolve_identity(path: Path, *, strict: bool = False) -> Path:
    check_namespace(path)
    resolved = path.resolve(strict=strict)
    check_namespace(resolved)
    return resolved


def lexical_absolute(spelling: str, description: str) -> Path:
    path = Path(spelling)
    check_namespace(path)
    if not spelling or ".." in path.parts or (path.drive and not path.is_absolute()):
        raise TestRootError(
            f"{description} has an unsafe lexical spelling: {spelling!r}"
        )
    absolute = Path(os.path.abspath(spelling))
    check_namespace(absolute)
    return absolute


def is_link(status: os.stat_result) -> bool:
    # On Windows junctions and other reparse points need not be symlinks.
    return stat.S_ISLNK(status.st_mode) or bool(
        getattr(status, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )


def check_directory_path(path: Path, description: str) -> bool:
    """Inspect lexical components, including dangling links, without resolving."""
    for component in (*reversed(path.parents), path):
        try:
            status = component.lstat()
        except FileNotFoundError:
            continue
        if is_link(status):
            raise TestRootError(
                f"{description} contains a symlink or reparse point: {component}"
            )
        if not stat.S_ISDIR(status.st_mode):
            raise TestRootError(f"{description} is not a directory: {component}")
    return path.exists()


def read_sentinel(path: Path) -> object:
    try:
        status = path.lstat()
        if is_link(status) or not stat.S_ISREG(status.st_mode):
            raise ValueError("sentinel must be a regular non-link file")
        if status.st_size > MAX_SENTINEL_BYTES:
            raise ValueError("sentinel is too large")
        with path.open("rb") as stream:
            payload = stream.read(MAX_SENTINEL_BYTES + 1)
        if len(payload) > MAX_SENTINEL_BYTES:
            raise ValueError("sentinel is too large")
        return json.loads(payload)
    except (OSError, ValueError) as error:
        raise TestRootError(
            f"existing test root lacks a valid ownership sentinel: {path}: {error}; "
            "choose a new scratch container (existing data was not removed)"
        ) from error


def prepare_test_root(
    test_root_spelling: str,
    producer_build_spelling: str,
    *,
    owner: str,
    repository: Path,
    source_directories: tuple[Path, ...] = (),
) -> tuple[Path, Path]:
    """Return a fresh run directory and checked build; never reset old outputs.

    The registered build/tests child and separately located external containers
    are allowed. Existing containers must have this runner/build/path's marker;
    even an empty pre-existing directory is not adopted automatically.
    """
    if owner not in OWNERS:
        raise TestRootError(f"unknown installed-test scratch owner: {owner}")
    build = lexical_absolute(producer_build_spelling, "producer build")
    root = lexical_absolute(test_root_spelling, "test root")
    if not check_directory_path(build, "producer build"):
        raise TestRootError(f"producer build does not exist: {build}")
    cache = build / "CMakeCache.txt"
    try:
        cache_status = cache.lstat()
    except FileNotFoundError as error:
        raise TestRootError(
            f"producer build has no regular CMakeCache.txt: {build}"
        ) from error
    if is_link(cache_status) or not stat.S_ISREG(cache_status.st_mode):
        raise TestRootError(f"producer build has no regular CMakeCache.txt: {build}")
    exists = check_directory_path(root, "test root")

    # Only now resolve build/root: resolving first would hide link components;
    # comparing abspath with realpath would split aliases (e.g. Linux //tmp).
    build = resolve_identity(build, strict=True)
    root = resolve_identity(root, strict=exists)
    repository = resolve_identity(repository)
    home = resolve_identity(Path.home())
    sources = tuple(resolve_identity(path) for path in source_directories)
    protected = (build, repository, home, *sources)
    if root == Path(root.anchor) or any(
        root == path or root in path.parents for path in protected
    ):
        raise TestRootError(f"test root equals or contains a protected path: {root}")
    registered_root = build / "tests" / owner
    if build in root.parents and root != registered_root:
        raise TestRootError(f"test root inside producer build must be {registered_root}")
    if (
        repository in root.parents and root != registered_root
    ) or any(path in root.parents for path in sources):
        raise TestRootError(f"test root is inside a protected source directory: {root}")

    expected = {
        "schema": "matcoredsl.installed-test-root",
        "version": 1,
        "owner": owner,
        "producer_build": str(build),
        "test_root": str(root),
    }
    marker = root / SENTINEL
    payload = (json.dumps(expected, sort_keys=True, indent=2) + "\n").encode("utf-8")
    if len(payload) > MAX_SENTINEL_BYTES:
        raise TestRootError("test-root ownership identity exceeds the sentinel size limit")
    if exists:
        actual = read_sentinel(marker)
        if actual != expected or type(actual.get("version")) is not int:
            raise TestRootError(
                "test-root ownership sentinel does not match this runner/build/path: "
                f"{root}; choose a new scratch container (existing data was not removed)"
            )
    else:
        # All path, build and protection checks precede the first mutation.
        # mkdir/exclusive marker creation never adopt a concurrently created root.
        root.mkdir(parents=True)
        with marker.open("xb") as stream:
            stream.write(payload)

    scratch = Path(tempfile.mkdtemp(prefix="run-", dir=root))
    print(f"installed test scratch retained (no automatic cleanup): {scratch}", flush=True)
    return scratch, build
