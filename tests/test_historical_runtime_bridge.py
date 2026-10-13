from __future__ import annotations

import importlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from tools import historical_readme_bridge as bridge
from tests import test_000_historical_readme_routing as routing

GROUP = "judgment-runtime"
SLOTS = ("src/neuron_graph_rag/judgments.py", "src/neuron_graph_rag/storage.py")
RUNTIME_MODULES = {
    "test_github_retrieval_parity_v2": 8,
    "test_github_retrieval_parity_v3": 9,
    "test_github_retrieval_parity_v4": 9,
    "test_github_retrieval_parity_v5_observation": 6,
}


class HistoricalRuntimeBridgeTests(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory)
        paths = set(bridge.load_registry(bridge.ROOT, GROUP)) | {bridge.RUNTIME_REGISTRY}
        paths |= {bridge.ARCHIVED_SLOTS[slot]["archive"] for slot in SLOTS}
        for name in paths:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(bridge.ROOT / name, path)
        return root

    def rewrite_registry(self, root, mutate):
        path = root / bridge.RUNTIME_REGISTRY
        value = json.loads(path.read_bytes())
        mutate(value)
        raw = json.dumps(value).encode()
        path.write_bytes(raw)
        return mock.patch.object(bridge, "RUNTIME_REGISTRY_SHA256", bridge.sha256(raw))

    def test_group_slots_are_exactly_the_changed_runtime_files(self):
        self.assertEqual(bridge.GROUPS[GROUP]["slots"], SLOTS)
        self.assertEqual(bridge.GROUPS["readme"]["slots"], ("README.md",))
        self.assertEqual(set(bridge.ARCHIVED_SLOTS), {"README.md", *SLOTS})

    def test_archives_are_exact_git_bytes_at_the_registered_commit(self):
        for slot in SLOTS:
            with self.subTest(slot=slot):
                raw = bridge.archived_bytes(bridge.ROOT, slot)
                self.assertEqual(raw, bridge.git(bridge.ROOT, "show", f"{bridge.RUNTIME_COMMIT}:{slot}"))
                self.assertNotIn(b"\r", raw)
                self.assertEqual(len(raw), bridge.ARCHIVED_SLOTS[slot]["size"])
                for version in (2, 3, 4, 5):
                    manifest = json.loads((bridge.ROOT / "tests/fixtures"
                                           / f"github_retrieval_parity_v{version}.manifest.json").read_bytes())
                    self.assertEqual(manifest["runtime_sha256"][slot], bridge.sha256(raw))

    def test_registry_covers_v2_to_v5_manifest_union_and_related_identities(self):
        registry = bridge.load_registry(bridge.ROOT, GROUP)
        union = {}
        for version in (2, 3, 4, 5):
            relative = f"tests/fixtures/github_retrieval_parity_v{version}.manifest.json"
            self.assertEqual(registry[relative], bridge.sha256((bridge.ROOT / relative).read_bytes()))
            for key, mapping in json.loads((bridge.ROOT / relative).read_bytes()).items():
                if not key.endswith("sha256") or not isinstance(mapping, dict):
                    continue
                for path, expected in mapping.items():
                    if isinstance(expected, str) and re.fullmatch("[0-9a-f]{64}", expected):
                        self.assertEqual(union.setdefault(path, expected), expected)
        self.assertEqual(len(union), 155)
        for path, expected in union.items():
            self.assertEqual(registry.get(path), expected, path)
        tracked = bridge.git(bridge.ROOT, "ls-files", "-z", "tests").decode().split("\0")
        for relative in filter(None, tracked):
            if re.search(r"github_retrieval_parity_v[2-5](?:[/._-]|$)", relative):
                self.assertEqual(registry.get(relative), bridge.sha256((bridge.ROOT / relative).read_bytes()), relative)
        observation = "src/neuron_graph_rag/github_retrieval_parity_v5_observation.py"
        self.assertEqual(registry[observation], bridge.sha256((bridge.ROOT / observation).read_bytes()))
        self.assertEqual(len(registry), 169)

    def test_current_assets_accept_only_live_runtime_slot_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.fixture(directory)
            for slot in SLOTS:
                (root / slot).write_bytes(b"# current runtime\n")
            registry = bridge.verify_current_assets(root, GROUP)
            for slot in SLOTS:
                self.assertEqual(registry[slot], bridge.ARCHIVED_SLOTS[slot]["sha256"])

    def test_archive_corruption_and_missing_fail_closed(self):
        for slot in SLOTS:
            for missing in (False, True):
                with self.subTest(slot=slot, missing=missing), tempfile.TemporaryDirectory() as directory:
                    root = self.fixture(directory)
                    archive = root / bridge.ARCHIVED_SLOTS[slot]["archive"]
                    if missing:
                        archive.unlink()
                    else:
                        archive.write_bytes(archive.read_bytes().replace(b"\n", b"\r\n"))
                    with self.assertRaises((ValueError, FileNotFoundError)):
                        bridge.verify_current_assets(root, GROUP)

    def test_non_slot_registered_asset_tamper_fails_closed(self):
        names = (
            "src/neuron_graph_rag/github_retrieval_parity_v2.py",
            "src/neuron_graph_rag/engine.py",
            "tests/test_github_retrieval_parity_v4.py",
            "tests/evidence/github_retrieval_parity_v5/development.observed.json",
            "tests/fixtures/github_retrieval_parity_v3.manifest.json",
            "tests/test_000_github_retrieval_parity_v5_routing.py",
        )
        for name in names:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = self.fixture(directory)
                path = root / name
                path.write_bytes(path.read_bytes() + b"\n")
                with self.assertRaisesRegex(ValueError, "current registered asset mismatch"):
                    bridge.verify_current_assets(root, GROUP)

    def test_live_slot_and_registered_asset_missing_fail_closed(self):
        for name in (*SLOTS, "src/neuron_graph_rag/github_retrieval_parity_v3.py"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = self.fixture(directory)
                (root / name).unlink()
                with self.assertRaises(FileNotFoundError):
                    bridge.verify_current_assets(root, GROUP)

    def test_registry_rewrite_and_missing_fail_closed(self):
        for missing in (False, True):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as directory:
                root = self.fixture(directory)
                registry = root / bridge.RUNTIME_REGISTRY
                if missing:
                    registry.unlink()
                else:
                    value = json.loads(registry.read_bytes())
                    value["assets"].pop("src/neuron_graph_rag/storage.py")
                    registry.write_text(json.dumps(value), encoding="utf-8")
                with self.assertRaises((ValueError, FileNotFoundError)):
                    bridge.verify_current_assets(root, GROUP)

    def test_unknown_slot_hash_path_and_archive_commit_fail_closed(self):
        storage = "src/neuron_graph_rag/storage.py"
        mutations = {
            "hash": lambda value: value["assets"].__setitem__(storage, "0" * 64),
            "current-hash": lambda value: value["assets"].__setitem__(
                storage, bridge.sha256((bridge.ROOT / storage).read_bytes())),
            "path": lambda value: value["assets"].__setitem__(
                "src/neuron_graph_rag/storage_v0.py", value["assets"].pop(storage)),
            "commit": lambda value: value.__setitem__("archive_commit", "0" * 40),
            "readme-key": lambda value: value.__setitem__("readme_commit", value.pop("archive_commit")),
        }
        for name, mutate in mutations.items():
            with self.subTest(mutation=name), tempfile.TemporaryDirectory() as directory:
                root = self.fixture(directory)
                with self.rewrite_registry(root, mutate):
                    with self.assertRaisesRegex(ValueError, "mapping|identity"):
                        bridge.verify_current_assets(root, GROUP)

    def test_slot_of_another_group_is_checked_against_current_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.fixture(directory)
            shutil.copyfile(bridge.ROOT / "README.md", root / "README.md")
            add_readme = lambda value: value["assets"].__setitem__("README.md", bridge.README_SHA256)
            with self.rewrite_registry(root, add_readme):
                with self.assertRaisesRegex(ValueError, "current registered asset mismatch: README.md"):
                    bridge.verify_current_assets(root, GROUP)

    def test_registry_paths_and_invalid_hash_fail_closed(self):
        for name, digest in (("../escape", "a" * 64), ("/absolute", "a" * 64),
                             ("src\\alias.py", "a" * 64), ("src/new.py", "unknown")):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = self.fixture(directory)
                with self.rewrite_registry(root, lambda value: value["assets"].__setitem__(name, digest)):
                    with self.assertRaisesRegex(ValueError, "invalid registered"):
                        bridge.load_registry(root, GROUP)

    def test_unknown_group_and_slot_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "unknown historical group"):
            bridge.verify_current_assets(bridge.ROOT, "storage")
        with self.assertRaisesRegex(ValueError, "unknown historical archived slot"):
            bridge.archived_bytes(bridge.ROOT, "src/neuron_graph_rag/engine.py")
        with self.assertRaisesRegex(ValueError, "unknown historical test module"):
            bridge.group_of("test_judgments")

    def test_projection_resolves_only_runtime_slots_and_keeps_source_untouched(self):
        source_head = bridge.git(bridge.ROOT, "rev-parse", "HEAD")
        live = {name: (bridge.ROOT / name).read_bytes() for name in (*SLOTS, "README.md")}
        with tempfile.TemporaryDirectory() as directory:
            projection = Path(directory)
            bridge.materialize_root(projection, group=GROUP)
            for slot in SLOTS:
                self.assertEqual((projection / slot).read_bytes(), bridge.archived_bytes(bridge.ROOT, slot))
            for name in ("README.md", "src/neuron_graph_rag/engine.py",
                         "src/neuron_graph_rag/decision_wiki_incremental.py",
                         "tests/test_github_retrieval_parity_v2.py"):
                self.assertEqual((projection / name).read_bytes(), (bridge.ROOT / name).read_bytes())
            self.assertEqual(bridge.git(projection, "rev-parse", "HEAD"), source_head)
        self.assertEqual(bridge.git(bridge.ROOT, "rev-parse", "HEAD"), source_head)
        for name, raw in live.items():
            self.assertEqual((bridge.ROOT / name).read_bytes(), raw)

    def test_runtime_modules_route_with_original_counts_under_both_names(self):
        self.assertEqual(bridge.RUNTIME_MODULE_COUNTS, RUNTIME_MODULES)
        self.assertFalse(set(bridge.RUNTIME_MODULE_COUNTS) & set(bridge.MODULE_COUNTS))
        loader = unittest.TestLoader()
        original = getattr(unittest.TestLoader, routing.ORIGINAL)
        with mock.patch.object(sys, "path", [str(bridge.ROOT / "tests"), *sys.path]):
            for prefix in ("", "tests."):
                for name, count in RUNTIME_MODULES.items():
                    module = importlib.import_module(prefix + name)
                    self.assertEqual(original(loader, module).countTestCases(), count)
                    self.assertEqual(loader.loadTestsFromModule(module).countTestCases(), 1)
                    self.assertEqual(bridge.group_of(name), GROUP)

    def test_additional_original_test_is_rejected(self):
        module = importlib.import_module("tests.test_github_retrieval_parity_v3")
        class ExtraTest(unittest.TestCase):
            def test_added(self):
                pass
        with mock.patch.object(module, "ExtraTest", ExtraTest, create=True):
            with self.assertRaisesRegex(ValueError, "original tests, found 10"):
                unittest.TestLoader().loadTestsFromModule(module)

    def test_v5_keeps_its_route_and_its_wrapper_runs_in_the_projection(self):
        v5_routing = importlib.import_module("tests.test_000_github_retrieval_parity_v5_routing")
        v5 = importlib.import_module("tests.test_github_retrieval_parity_v5")
        self.assertEqual(unittest.TestLoader().loadTestsFromModule(v5).countTestCases(), 1)
        with self.assertRaisesRegex(ValueError, "unknown historical"):
            bridge.canonical_module(bridge.V5_WRAPPED_MODULE)
        self.assertEqual(bridge.canonical_module("tests." + bridge.V5_WRAPPED_MODULE, wrapped=True),
                         bridge.V5_WRAPPED_MODULE)
        self.assertEqual(bridge.group_of(bridge.V5_WRAPPED_MODULE), GROUP)
        with mock.patch.object(bridge, "run_module", return_value="") as run:
            v5_routing._run_frozen_suite()
        run.assert_called_once_with(bridge.V5_WRAPPED_MODULE)

    def test_v5_wrapper_report_requires_all_frozen_tests(self):
        from neuron_graph_rag import github_retrieval_parity_v5_observation as observation
        cases = (("Ran 11 tests in 1.0s\n\nOK\n", 0), ("Ran 10 tests in 1.0s\n\nOK\n", 1),
                 ("Ran 11 tests in 1.0s\n\nFAILED (failures=1)\n", 1), ("no summary\n", 1))
        for output, status in cases:
            with self.subTest(output=output):
                with mock.patch.object(observation, "verify_observation") as verify, mock.patch.object(
                        observation, "run_frozen_v5_tests", return_value=output), mock.patch(
                        "builtins.print") as printed:
                    self.assertEqual(bridge.run_v5_wrapper(11), status)
                verify.assert_called_once_with()
                report = json.loads(printed.call_args_list[-1].args[0][len(bridge.RESULT_PREFIX):])
                self.assertEqual(report["successful"], status == 0)

    def test_subprocess_failure_and_wrong_counts_fail_closed(self):
        for name, count in (("test_github_retrieval_parity_v4", 9), (bridge.V5_WRAPPED_MODULE, 11)):
            valid = {"module": name, "loaded": count, "executed": count, "successful": True}
            cases = ((1, bridge.RESULT_PREFIX + json.dumps(valid)),
                     (0, f"Ran {count} tests\nOK"),
                     (0, bridge.RESULT_PREFIX + json.dumps(dict(valid, executed=count - 1))),
                     (0, bridge.RESULT_PREFIX + json.dumps(dict(valid, loaded=count + 1))))
            for status, output in cases:
                with self.subTest(name=name, status=status, output=output):
                    completed = subprocess.CompletedProcess([], status, output, "failure details")
                    with mock.patch.object(bridge, "materialize_root") as materialize, mock.patch.object(
                            bridge.subprocess, "run", return_value=completed):
                        with self.assertRaises(AssertionError):
                            bridge.run_module(name)
                    self.assertEqual(materialize.call_args.args[2], GROUP)


if __name__ == "__main__":
    unittest.main()
