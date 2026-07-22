#!/usr/bin/env python3

import argparse
import pathlib
import re
import shutil
import subprocess
import tempfile
import time


def run(command: list[str], *, cwd: pathlib.Path | None = None) -> str:
    completed = subprocess.run(
        command, cwd=cwd, text=True, capture_output=True, check=False
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"command failed ({completed.returncode}): {command}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed.stdout.strip()


def generate(
    cmake: str,
    script: pathlib.Path,
    source: pathlib.Path,
    output: pathlib.Path,
    *,
    git: str,
    commit_override: str = "",
    state_override: str = "",
) -> str:
    run(
        [
            cmake,
            f"-DMATCORE_BENCH_PROVENANCE_OUTPUT={output}",
            f"-DMATCORE_BENCH_SOURCE_ROOT={source}",
            f"-DMATCORE_BENCH_GIT_EXECUTABLE={git}",
            f"-DMATCORE_BENCH_SOURCE_COMMIT_OVERRIDE={commit_override}",
            f"-DMATCORE_BENCH_SOURCE_STATE_OVERRIDE={state_override}",
            "-P",
            str(script),
        ]
    )
    return output.read_text(encoding="utf-8")


def macro(header: str, name: str) -> str:
    match = re.search(rf'^#define {re.escape(name)} (.+)$', header, re.MULTILINE)
    assert match, name
    return match.group(1).strip().strip('"')


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmake", required=True)
    parser.add_argument("--generator-script", required=True)
    args = parser.parse_args()
    script = pathlib.Path(args.generator_script).resolve()
    git = shutil.which("git")
    assert git is not None

    with tempfile.TemporaryDirectory(prefix="matcore provenance ") as temporary:
        root = pathlib.Path(temporary)
        repository = root / "repository with spaces"
        repository.mkdir()
        run(["git", "init", "--quiet"], cwd=repository)
        run(["git", "config", "user.name", "Matcore Test"], cwd=repository)
        run(["git", "config", "user.email", "matcore@example.invalid"], cwd=repository)
        tracked = repository / "tracked.txt"
        tracked.write_text("first\n", encoding="utf-8")
        run(["git", "add", "tracked.txt"], cwd=repository)
        run(["git", "commit", "--quiet", "-m", "initial"], cwd=repository)
        first_commit = run(["git", "rev-parse", "HEAD"], cwd=repository)
        output = root / "generated" / "provenance.h"

        clean = generate(args.cmake, script, repository, output, git=git)
        assert macro(clean, "MATCORE_BENCH_SOURCE_COMMIT") == first_commit
        assert macro(clean, "MATCORE_BENCH_SOURCE_WORKTREE_DIRTY") == "0"
        assert macro(clean, "MATCORE_BENCH_SOURCE_PROVENANCE_STATE") == "clean"
        assert macro(clean, "MATCORE_BENCH_SOURCE_PROVENANCE_ORIGIN") == "git-worktree"
        unchanged_mtime = output.stat().st_mtime_ns
        time.sleep(0.01)
        assert generate(args.cmake, script, repository, output, git=git) == clean
        assert output.stat().st_mtime_ns == unchanged_mtime

        tracked.write_text("modified\n", encoding="utf-8")
        dirty = generate(args.cmake, script, repository, output, git=git)
        assert macro(dirty, "MATCORE_BENCH_SOURCE_COMMIT") == first_commit
        assert macro(dirty, "MATCORE_BENCH_SOURCE_WORKTREE_DIRTY") == "1"
        assert macro(dirty, "MATCORE_BENCH_SOURCE_PROVENANCE_STATE") == "dirty"
        assert output.stat().st_mtime_ns > unchanged_mtime

        run(["git", "add", "tracked.txt"], cwd=repository)
        run(["git", "commit", "--quiet", "-m", "second"], cwd=repository)
        second_commit = run(["git", "rev-parse", "HEAD"], cwd=repository)
        committed = generate(args.cmake, script, repository, output, git=git)
        assert second_commit != first_commit
        assert macro(committed, "MATCORE_BENCH_SOURCE_COMMIT") == second_commit
        assert macro(committed, "MATCORE_BENCH_SOURCE_PROVENANCE_STATE") == "clean"

        archive = root / "source archive"
        archive.mkdir()
        unavailable = generate(args.cmake, script, archive, output, git=git)
        assert macro(unavailable, "MATCORE_BENCH_SOURCE_COMMIT") == "unknown"
        assert macro(unavailable, "MATCORE_BENCH_SOURCE_PROVENANCE_STATE") == "unknown"
        assert macro(unavailable, "MATCORE_BENCH_SOURCE_PROVENANCE_ORIGIN") == "unavailable"

        overridden = generate(
            args.cmake,
            script,
            archive,
            output,
            git=git,
            commit_override=second_commit,
            state_override="clean",
        )
        assert macro(overridden, "MATCORE_BENCH_SOURCE_COMMIT") == second_commit
        assert macro(overridden, "MATCORE_BENCH_SOURCE_PROVENANCE_STATE") == "clean"
        assert macro(overridden, "MATCORE_BENCH_SOURCE_PROVENANCE_ORIGIN") == "explicit-override"

    print("matcore benchmark provenance incremental test PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
