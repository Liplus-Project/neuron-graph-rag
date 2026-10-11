"""Run explicit historical test modules against current assets and archived README."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

ROOT = Path(__file__).resolve().parents[1]
README_COMMIT = "b3cc03a15b81f0e395ae564387a46fe57d320f31"
README_SHA256 = "5a1438ea5c8eecbed015661ad1decd7451e0c07aa848efc110557334f7de3638"
README_ARCHIVE = "tools/historical_readme/README.b3cc03a.md"
REGISTRY = "tools/historical_readme/registered-assets.json"
REGISTRY_SHA256 = "ec23e934f0dc31ca106615f5d2ff1bb3ff7bcaa5cc97eaf03bdba43a8762e0bc"
MODULE_COUNTS = {
    "test_cross_encoder_precision": 11,
    "test_cross_encoder_precision_observation": 7,
    "test_cross_encoder_precision_v2": 15,
    "test_cross_encoder_precision_v2_observation": 8,
    "test_cross_encoder_precision_v3": 15,
    "test_cross_encoder_precision_v3_observation": 9,
    "test_cross_encoder_precision_v4": 17,
    "test_cross_encoder_precision_v5": 22,
    "test_cross_encoder_precision_v6": 28,
    "test_cross_encoder_precision_v7": 9,
    "test_cross_encoder_precision_v8": 16,
    "test_cross_encoder_precision_v8_observation": 9,
    "test_cross_encoder_precision_v9_performance_observation": 11,
    "test_cross_encoder_precision_v10_performance_observation": 13,
    "test_cross_encoder_precision_v12_performance_observation": 16,
    "test_cross_encoder_precision_v13_observation": 11,
    "test_cross_encoder_precision_v14_performance_observation": 16,
    "test_cross_encoder_precision_v15_observation": 13,
    "test_cross_encoder_precision_v16_observation": 14,
    "test_github_retrieval_parity": 11,
    "test_github_retrieval_parity_observation": 5,
}
RESULT_PREFIX = "HISTORICAL_README_RESULT="


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def asset_path(root: Path, relative: str) -> Path:
    path = PurePosixPath(relative)
    if (not relative or path.is_absolute() or ".." in path.parts
            or path.as_posix() != relative or "\\" in relative or ":" in relative):
        raise ValueError(f"invalid registered path: {relative}")
    target = root / relative
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"registered path escapes root: {relative}")
    if any(parent.is_symlink() for parent in [target, *target.parents] if parent != root):
        raise ValueError(f"registered symlink: {relative}")
    return target


def readme_bytes(root: Path) -> bytes:
    raw = asset_path(root, README_ARCHIVE).read_bytes()
    if len(raw) != 31856 or sha256(raw) != README_SHA256:
        raise ValueError("historical README archive identity mismatch")
    return raw


def load_registry(root: Path) -> dict[str, str]:
    raw = asset_path(root, REGISTRY).read_bytes()
    if sha256(raw) != REGISTRY_SHA256:
        raise ValueError("historical registered-assets registry identity mismatch")
    value = json.loads(raw)
    if (set(value) != {"readme_commit", "assets"}
            or value["readme_commit"] != README_COMMIT
            or not isinstance(value["assets"], dict)):
        raise ValueError("historical registry shape/identity mismatch")
    registry = value["assets"]
    if registry.get("README.md") != README_SHA256:
        raise ValueError("unknown historical README mapping")
    for relative, expected in registry.items():
        asset_path(root, relative)
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError(f"invalid registered hash: {relative}")
    return registry


def verify_current_assets(root: Path = ROOT) -> dict[str, str]:
    archive = readme_bytes(root)
    registry = load_registry(root)
    # The live display slot must still exist; only its exact registered identity
    # resolves to the independently pinned archive.
    asset_path(root, "README.md").read_bytes()
    for relative, expected in registry.items():
        raw = archive if relative == "README.md" else asset_path(root, relative).read_bytes()
        if sha256(raw) != expected:
            raise ValueError(f"current registered asset mismatch: {relative}")
    return registry


def git(root: Path, *arguments: str) -> bytes:
    environment = child_environment(root)
    return subprocess.run(["git", *arguments], cwd=root, env=environment,
                          check=True, capture_output=True).stdout


def child_environment(root: Path) -> dict[str, str]:
    environment = os.environ.copy()
    # Synthetic repositories inside frozen tests retain their own Git context.
    for name in tuple(environment):
        if name.startswith("GIT_"):
            del environment[name]
    environment["PYTHONPATH"] = os.pathsep.join((str(root / "src"), str(root)))
    environment["PYTHONNOUSERSITE"] = "1"
    return environment


def materialize_root(destination: Path, root: Path = ROOT) -> None:
    verify_current_assets(root)
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise ValueError("historical destination must be empty")
    paths = git(root, "ls-files", "-z").decode("utf-8").split("\0")
    for relative in filter(None, paths):
        source = asset_path(root, relative)
        target = asset_path(destination, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    (destination / "README.md").write_bytes(readme_bytes(root))
    verify_current_assets(destination)
    # Independent metadata and refs: child writes cannot reach the source index
    # or refs. Alternates only supply immutable object reads, without a fetch.
    git(destination, "init", "--quiet")
    git(destination, "config", "core.autocrlf", "false")
    common = Path(git(root, "rev-parse", "--git-common-dir").decode().strip())
    if not common.is_absolute():
        common = root / common
    (destination / ".git/objects/info/alternates").write_text(
        (common.resolve() / "objects").as_posix() + "\n", encoding="utf-8", newline="\n")
    for row in git(root, "for-each-ref", "--format=%(refname) %(objectname)").decode().splitlines():
        reference, object_id = row.split(" ", 1)
        git(destination, "update-ref", reference, object_id)
    git(destination, "update-ref", "--no-deref", "HEAD", git(root, "rev-parse", "HEAD").decode().strip())
    git(destination, "read-tree", "HEAD")


def canonical_module(name: str) -> str:
    bare = name.removeprefix("tests.")
    if name not in (bare, "tests." + bare) or bare not in MODULE_COUNTS:
        raise ValueError(f"unknown historical test module: {name}")
    return bare


def run_module(name: str, root: Path = ROOT) -> str:
    bare = canonical_module(name)
    with TemporaryDirectory(prefix="ngr-historical-readme-") as directory:
        projection = Path(directory)
        materialize_root(projection, root)
        completed = subprocess.run(
            [sys.executable, str(projection / "tools/historical_readme_bridge.py"), "--child", bare],
            cwd=projection, env=child_environment(projection), check=False,
            capture_output=True, text=True, encoding="utf-8", errors="strict")
    output = completed.stdout + completed.stderr
    rows = [line[len(RESULT_PREFIX):] for line in completed.stdout.splitlines()
            if line.startswith(RESULT_PREFIX)]
    if completed.returncode != 0 or len(rows) != 1:
        raise AssertionError(f"historical {bare} subprocess failed\n{output}")
    report = json.loads(rows[0])
    if report != {"module": bare, "loaded": MODULE_COUNTS[bare],
                  "executed": MODULE_COUNTS[bare], "successful": True}:
        raise AssertionError(f"historical {bare} original test execution mismatch\n{output}")
    return output


def run_child(name: str) -> int:
    bare = canonical_module(name)
    verify_current_assets(ROOT)
    if (ROOT / "README.md").read_bytes() != readme_bytes(ROOT):
        raise ValueError("historical child README slot mismatch")
    sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
    module = importlib.import_module("tests." + bare)
    if Path(module.__file__).resolve() != ROOT / "tests" / (bare + ".py"):
        raise ValueError("historical child test module root mismatch")
    suite = unittest.TestLoader().loadTestsFromModule(module)
    count = suite.countTestCases()
    if count != MODULE_COUNTS[bare]:
        raise ValueError(f"historical {bare} expected {MODULE_COUNTS[bare]} original tests, found {count}")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(RESULT_PREFIX + json.dumps({"module": bare, "loaded": count,
        "executed": result.testsRun, "successful": result.wasSuccessful()}))
    return 0 if result.wasSuccessful() and result.testsRun == count else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--module")
    group.add_argument("--child", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        return run_child(args.child)
    print(run_module(args.module), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
