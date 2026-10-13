"""Run explicit historical test modules against current assets and archived slot bytes.

A slot is an exact (path, SHA-256) pair registered by frozen evidence whose live file
has since changed. Only those pairs resolve to archived original bytes; every other
registered asset is checked against the current tree.
"""

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
RUNTIME_COMMIT = "b78dbcaf0167cc6975428ccce662b8228a88cfe2"
RUNTIME_REGISTRY = "tools/historical_readme/judgment-runtime-assets.json"
RUNTIME_REGISTRY_SHA256 = "8d2f37d5311c11fd2c4d178c824ae150a1f6ed0417b0ac6ad758a7337f32bc87"
# Exact registered (path, sha256) pairs and the archived original bytes they resolve to.
ARCHIVED_SLOTS = {
    "README.md": {
        "commit": README_COMMIT, "sha256": README_SHA256, "size": 31856,
        "archive": README_ARCHIVE,
    },
    "src/neuron_graph_rag/storage.py": {
        "commit": RUNTIME_COMMIT, "size": 106345,
        "sha256": "3aaf1fb2283ac97486fc9cab34a44676650aec5b6e1ccb8e3100132efc8459d0",
        "archive": "tools/historical_readme/storage.b78dbca.py.orig",
    },
    "src/neuron_graph_rag/judgments.py": {
        "commit": RUNTIME_COMMIT, "size": 40847,
        "sha256": "cfc21058f63fcd665e8fa233cdcd01a8dfe44f4241e867886155eddcc4fac3f7",
        "archive": "tools/historical_readme/judgments.b78dbca.py.orig",
    },
}
# Each group pins one registry of current assets (by the named module constant)
# and the slots its modules resolve.
GROUPS = {
    "readme": {
        "registry": REGISTRY, "registry_pin": "REGISTRY_SHA256",
        "commit_key": "readme_commit", "commit": README_COMMIT,
        "slots": ("README.md",),
    },
    "judgment-runtime": {
        "registry": RUNTIME_REGISTRY, "registry_pin": "RUNTIME_REGISTRY_SHA256",
        "commit_key": "archive_commit", "commit": RUNTIME_COMMIT,
        "slots": ("src/neuron_graph_rag/judgments.py", "src/neuron_graph_rag/storage.py"),
    },
}
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
RUNTIME_MODULE_COUNTS = {
    "test_github_retrieval_parity_v2": 8,
    "test_github_retrieval_parity_v3": 9,
    "test_github_retrieval_parity_v4": 9,
    "test_github_retrieval_parity_v5_observation": 6,
}
# Modules the unittest loader router hands to the bridge.
ROUTED_MODULE_COUNTS = {**MODULE_COUNTS, **RUNTIME_MODULE_COUNTS}
# The frozen v5 suite keeps its own loader route; that route's wrapper runs in the projection.
V5_WRAPPED_MODULE = "test_github_retrieval_parity_v5"
V5_WRAPPED_COUNT = 11
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


def _slot_name(slot: str) -> str:
    return "README" if slot == "README.md" else slot


def archived_bytes(root: Path, slot: str) -> bytes:
    if slot not in ARCHIVED_SLOTS:
        raise ValueError(f"unknown historical archived slot: {slot}")
    identity = ARCHIVED_SLOTS[slot]
    raw = asset_path(root, identity["archive"]).read_bytes()
    if len(raw) != identity["size"] or sha256(raw) != identity["sha256"]:
        raise ValueError(f"historical {_slot_name(slot)} archive identity mismatch")
    return raw


def readme_bytes(root: Path) -> bytes:
    return archived_bytes(root, "README.md")


def group_of(bare: str) -> str:
    if bare in MODULE_COUNTS:
        return "readme"
    if bare in RUNTIME_MODULE_COUNTS or bare == V5_WRAPPED_MODULE:
        return "judgment-runtime"
    raise ValueError(f"unknown historical test module: {bare}")


def _group(group: str) -> dict:
    if group not in GROUPS:
        raise ValueError(f"unknown historical group: {group}")
    return GROUPS[group]


def load_registry(root: Path, group: str = "readme") -> dict[str, str]:
    spec = _group(group)
    raw = asset_path(root, spec["registry"]).read_bytes()
    if sha256(raw) != globals()[spec["registry_pin"]]:
        raise ValueError("historical registered-assets registry identity mismatch")
    value = json.loads(raw)
    if (set(value) != {spec["commit_key"], "assets"}
            or value[spec["commit_key"]] != spec["commit"]
            or not isinstance(value["assets"], dict)):
        raise ValueError("historical registry shape/identity mismatch")
    registry = value["assets"]
    for slot in spec["slots"]:
        identity = ARCHIVED_SLOTS[slot]
        if identity["commit"] != spec["commit"] or registry.get(slot) != identity["sha256"]:
            raise ValueError(f"unknown historical {_slot_name(slot)} mapping")
    for relative, expected in registry.items():
        asset_path(root, relative)
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError(f"invalid registered hash: {relative}")
    return registry


def verify_current_assets(root: Path = ROOT, group: str = "readme") -> dict[str, str]:
    archives = {slot: archived_bytes(root, slot) for slot in _group(group)["slots"]}
    registry = load_registry(root, group)
    # Each live slot must still exist; only its exact registered identity
    # resolves to the independently pinned archive.
    for slot in archives:
        asset_path(root, slot).read_bytes()
    for relative, expected in registry.items():
        raw = archives[relative] if relative in archives else asset_path(root, relative).read_bytes()
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


def materialize_root(destination: Path, root: Path = ROOT, group: str = "readme") -> None:
    verify_current_assets(root, group)
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise ValueError("historical destination must be empty")
    paths = git(root, "ls-files", "-z").decode("utf-8").split("\0")
    for relative in filter(None, paths):
        source = asset_path(root, relative)
        target = asset_path(destination, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    for slot in GROUPS[group]["slots"]:
        asset_path(destination, slot).write_bytes(archived_bytes(root, slot))
    verify_current_assets(destination, group)
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


def canonical_module(name: str, *, wrapped: bool = False) -> str:
    bare = name.removeprefix("tests.")
    known = bare in ROUTED_MODULE_COUNTS or (wrapped and bare == V5_WRAPPED_MODULE)
    if name not in (bare, "tests." + bare) or not known:
        raise ValueError(f"unknown historical test module: {name}")
    return bare


def expected_count(bare: str) -> int:
    return V5_WRAPPED_COUNT if bare == V5_WRAPPED_MODULE else ROUTED_MODULE_COUNTS[bare]


def run_module(name: str, root: Path = ROOT) -> str:
    bare = canonical_module(name, wrapped=True)
    with TemporaryDirectory(prefix="ngr-historical-readme-") as directory:
        projection = Path(directory)
        materialize_root(projection, root, group_of(bare))
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
    if report != {"module": bare, "loaded": expected_count(bare),
                  "executed": expected_count(bare), "successful": True}:
        raise AssertionError(f"historical {bare} original test execution mismatch\n{output}")
    return output


def run_child(name: str) -> int:
    bare = canonical_module(name, wrapped=True)
    group = group_of(bare)
    verify_current_assets(ROOT, group)
    for slot in GROUPS[group]["slots"]:
        if (ROOT / slot).read_bytes() != archived_bytes(ROOT, slot):
            raise ValueError(f"historical child {_slot_name(slot)} slot mismatch")
    sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
    module = importlib.import_module("tests." + bare)
    if Path(module.__file__).resolve() != ROOT / "tests" / (bare + ".py"):
        raise ValueError("historical child test module root mismatch")
    suite = unittest.TestLoader().loadTestsFromModule(module)
    count = suite.countTestCases()
    if count != expected_count(bare):
        raise ValueError(f"historical {bare} expected {expected_count(bare)} original tests, found {count}")
    if bare == V5_WRAPPED_MODULE:
        return run_v5_wrapper(count)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(RESULT_PREFIX + json.dumps({"module": bare, "loaded": count,
        "executed": result.testsRun, "successful": result.wasSuccessful()}))
    return 0 if result.wasSuccessful() and result.testsRun == count else 1


def run_v5_wrapper(count: int) -> int:
    """Run the v5 route's wrapper here: verify the observation, then the frozen suite."""
    from neuron_graph_rag import github_retrieval_parity_v5_observation as observation
    expected = ROOT / "src/neuron_graph_rag/github_retrieval_parity_v5_observation.py"
    if Path(observation.__file__).resolve() != expected or observation.ROOT.resolve() != ROOT:
        raise ValueError("historical child v5 observation root mismatch")
    observation.verify_observation()
    output = observation.run_frozen_v5_tests()
    print(output, end="")
    ran = re.findall(r"^Ran (\d+) tests? in ", output, re.MULTILINE)
    executed = int(ran[0]) if len(ran) == 1 else -1
    successful = executed == count and re.search(r"^OK\b", output, re.MULTILINE) is not None
    print(RESULT_PREFIX + json.dumps({"module": V5_WRAPPED_MODULE, "loaded": count,
        "executed": executed, "successful": successful}))
    return 0 if successful else 1


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
