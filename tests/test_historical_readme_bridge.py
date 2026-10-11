from __future__ import annotations

import importlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

from tools import historical_readme_bridge as bridge
from tests import test_000_historical_readme_routing as routing


class HistoricalReadmeBridgeTests(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory)
        paths = set(bridge.load_registry(bridge.ROOT)) | {bridge.REGISTRY, bridge.README_ARCHIVE}
        for name in paths:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(bridge.ROOT / name, path)
        return root

    def test_current_assets_accept_only_live_readme_display_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.fixture(directory)
            (root / "README.md").write_bytes(b"# Current logo and underline\n")
            registry = bridge.verify_current_assets(root)
            self.assertEqual(len(registry), 404)
            self.assertEqual(registry["README.md"], bridge.README_SHA256)

    def test_archive_exact_git_bytes_and_line_endings(self):
        raw = bridge.readme_bytes(bridge.ROOT)
        self.assertEqual(raw, bridge.git(bridge.ROOT, "show", f"{bridge.README_COMMIT}:README.md"))
        self.assertNotIn(b"\r", raw)
        self.assertEqual(len(raw), 31856)

    def test_registry_covers_original_manifest_union_and_additional_identities(self):
        registry = bridge.load_registry(bridge.ROOT)
        fixtures = bridge.ROOT / "tests/fixtures"
        names = ["github_retrieval_parity_v1.manifest.json"]
        names += [f"github_cross_encoder_precision_v{version}.manifest.json"
                  for version in (*range(1, 12), 13, 15, 16)]
        names += [f"github_cross_encoder_precision_v{version}_observation.manifest.json"
                  for version in (9, 10, 12, 14)]
        self.assertEqual(len(names), 19)
        original = {}
        for name in names:
            path = fixtures / name
            self.assertEqual(registry[path.relative_to(bridge.ROOT).as_posix()], bridge.sha256(path.read_bytes()))
            manifest = json.loads(path.read_bytes())
            for key, mapping in manifest.items():
                if not key.endswith("sha256") or not isinstance(mapping, dict):
                    continue
                for relative, expected in mapping.items():
                    if not isinstance(expected, str) or not re.fullmatch("[0-9a-f]{64}", expected):
                        continue
                    self.assertEqual(registry.get(relative), expected, (name, key, relative))
                    if relative in original:
                        self.assertEqual(original[relative], expected)
                    original[relative] = expected
        self.assertEqual(len(original), 367)
        for name in bridge.MODULE_COUNTS:
            relative = "tests/" + name + ".py"
            self.assertEqual(registry.get(relative), bridge.sha256((bridge.ROOT / relative).read_bytes()))
        corpus = json.loads((fixtures / "github_cross_encoder_precision_v8.corpus.json").read_bytes())
        self.assertEqual(len(corpus["documents"]), 24)
        for row in corpus["documents"]:
            self.assertEqual(registry.get(row["path"]), row["content_sha256"])
        for path in (bridge.ROOT / "tests").rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            relative = path.relative_to(bridge.ROOT).as_posix()
            match = re.search(r"github_cross_encoder_precision_v(\d+)(?:[/._-]|$)", relative)
            if (match and int(match[1]) <= 16) or "github_retrieval_parity_v1" in relative:
                self.assertEqual(registry.get(relative), bridge.sha256(path.read_bytes()), relative)

    def test_archive_corruption_and_missing_fail_closed(self):
        for missing in (False, True):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as directory:
                root = self.fixture(directory)
                archive = root / bridge.README_ARCHIVE
                if missing:
                    archive.unlink()
                else:
                    archive.write_bytes(archive.read_bytes().replace(b"\n", b"\r\n"))
                with self.assertRaises((ValueError, FileNotFoundError)):
                    bridge.verify_current_assets(root)

    def test_source_tests_evidence_corpus_and_identity_tamper_fail_closed(self):
        names = (
            "src/neuron_graph_rag/cross_encoder_precision_v3_evaluation.py",
            "tests/test_cross_encoder_precision_v13_observation.py",
            "tests/evidence/github_cross_encoder_precision_v1/development.error.json",
            "docs/anchored-bm25-graph-hybrid-experiment.md",
            "tests/fixtures/github_cross_encoder_precision_v15.source-identity.json",
        )
        for name in names:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = self.fixture(directory)
                path = root / name
                path.write_bytes(path.read_bytes() + b"\n")
                with self.assertRaisesRegex(ValueError, "current registered asset mismatch"):
                    bridge.verify_current_assets(root)

    def test_registered_asset_and_live_readme_missing_fail_closed(self):
        for name in ("README.md", "src/neuron_graph_rag/cross_encoder_precision_v3_evaluation.py"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = self.fixture(directory)
                (root / name).unlink()
                with self.assertRaises(FileNotFoundError):
                    bridge.verify_current_assets(root)

    def test_self_consistent_manifest_rewrite_still_fails_pinned_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.fixture(directory)
            source = "src/neuron_graph_rag/cross_encoder_precision_v3_evaluation.py"
            (root / source).write_bytes(b"# malicious replacement\n")
            manifest_path = root / "tests/fixtures/github_cross_encoder_precision_v3.manifest.json"
            manifest = json.loads(manifest_path.read_bytes())
            manifest["artifact_sha256"][source] = bridge.sha256((root / source).read_bytes())
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "current registered asset mismatch"):
                bridge.verify_current_assets(root)

    def test_registry_rewrite_and_missing_fail_closed(self):
        for missing in (False, True):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as directory:
                root = self.fixture(directory)
                registry = root / bridge.REGISTRY
                if missing:
                    registry.unlink()
                else:
                    value = json.loads(registry.read_bytes())
                    value["assets"].pop("README.md")
                    registry.write_text(json.dumps(value), encoding="utf-8")
                with self.assertRaises((ValueError, FileNotFoundError)):
                    bridge.verify_current_assets(root)

    def test_unknown_readme_hash_path_and_archive_commit_fail_closed(self):
        for mutation in ("hash", "path", "commit"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = self.fixture(directory)
                path = root / bridge.REGISTRY
                value = json.loads(path.read_bytes())
                if mutation == "hash":
                    value["assets"]["README.md"] = "0" * 64
                elif mutation == "path":
                    value["assets"]["other/README.md"] = value["assets"].pop("README.md")
                else:
                    value["readme_commit"] = "0" * 40
                raw = json.dumps(value).encode()
                path.write_bytes(raw)
                with mock.patch.object(bridge, "REGISTRY_SHA256", bridge.sha256(raw)):
                    with self.assertRaisesRegex(ValueError, "mapping|identity"):
                        bridge.verify_current_assets(root)

    def test_registry_paths_and_invalid_hash_fail_closed(self):
        for name, digest in (("../escape", "a" * 64), ("/absolute", "a" * 64),
                             ("docs\\alias.md", "a" * 64), ("docs/new.md", "unknown")):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = self.fixture(directory)
                path = root / bridge.REGISTRY
                value = json.loads(path.read_bytes())
                value["assets"][name] = digest
                raw = json.dumps(value).encode()
                path.write_bytes(raw)
                with mock.patch.object(bridge, "REGISTRY_SHA256", bridge.sha256(raw)):
                    with self.assertRaisesRegex(ValueError, "invalid registered"):
                        bridge.load_registry(root)

    def test_symlink_registered_asset_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.fixture(directory)
            name = "docs/anchored-bm25-graph-hybrid-experiment.md"
            path = root / name
            path.unlink()
            try:
                path.symlink_to(bridge.ROOT / name)
            except OSError:
                self.skipTest("symlink permission unavailable")
            with self.assertRaisesRegex(ValueError, "escapes root|symlink"):
                bridge.verify_current_assets(root)

    def test_projection_uses_current_bytes_and_independent_git_metadata(self):
        source_head = bridge.git(bridge.ROOT, "rev-parse", "HEAD")
        source_refs = bridge.git(bridge.ROOT, "for-each-ref")
        source_index = Path(bridge.git(bridge.ROOT, "rev-parse", "--git-path", "index").decode().strip())
        if not source_index.is_absolute():
            source_index = bridge.ROOT / source_index
        index_bytes = source_index.read_bytes()
        live_readme = (bridge.ROOT / "README.md").read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            projection = Path(directory)
            bridge.materialize_root(projection)
            self.assertEqual((projection / "README.md").read_bytes(), bridge.readme_bytes(bridge.ROOT))
            self.assertTrue((projection / ".git").is_dir())
            self.assertEqual(bridge.git(projection, "rev-parse", "HEAD"), source_head)
            self.assertEqual(bridge.git(projection, "show", f"{bridge.README_COMMIT}:README.md"),
                             bridge.readme_bytes(bridge.ROOT))
            for name in ("src/neuron_graph_rag/engine.py", "tests/test_cross_encoder_precision_v13_observation.py",
                         "docs/anchored-bm25-graph-hybrid-experiment.md"):
                self.assertEqual((projection / name).read_bytes(), (bridge.ROOT / name).read_bytes())
            bridge.git(projection, "update-ref", "refs/heads/bridge-isolation-test", source_head.decode().strip())
            bridge.git(projection, "read-tree", "--empty")
            self.assertEqual(bridge.git(bridge.ROOT, "rev-parse", "HEAD"), source_head)
            self.assertEqual(bridge.git(bridge.ROOT, "for-each-ref"), source_refs)
            self.assertEqual(source_index.read_bytes(), index_bytes)
        self.assertEqual((bridge.ROOT / "README.md").read_bytes(), live_readme)

    def test_preflight_rejects_before_materialization_or_subprocess(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.fixture(directory)
            (root / bridge.README_ARCHIVE).write_bytes(b"corrupt")
            destination = root / "projection"
            with mock.patch.object(bridge, "git", side_effect=AssertionError("Git must not run")):
                with self.assertRaisesRegex(ValueError, "archive identity"):
                    bridge.materialize_root(destination, root)
            self.assertFalse(destination.exists())

    def test_environment_preserves_synthetic_repo_context(self):
        with mock.patch.dict(bridge.os.environ, {"GIT_DIR": "wrong", "GIT_WORK_TREE": "wrong"}):
            env = bridge.child_environment(bridge.ROOT)
        self.assertNotIn("GIT_DIR", env)
        self.assertNotIn("GIT_WORK_TREE", env)
        with tempfile.TemporaryDirectory() as directory:
            subprocess.run(["git", "init", "--quiet"], cwd=directory, env=env, check=True)
            result = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=directory,
                                    env=env, check=True, capture_output=True, text=True)
            self.assertEqual(Path(result.stdout.strip()).resolve(), Path(directory).resolve())

    def test_exact_21_module_closure_and_276_original_tests_both_names(self):
        expected = {
            "test_cross_encoder_precision", "test_cross_encoder_precision_observation",
            "test_cross_encoder_precision_v2", "test_cross_encoder_precision_v2_observation",
            "test_cross_encoder_precision_v3", "test_cross_encoder_precision_v3_observation",
            "test_cross_encoder_precision_v4", "test_cross_encoder_precision_v5",
            "test_cross_encoder_precision_v6", "test_cross_encoder_precision_v7",
            "test_cross_encoder_precision_v8", "test_cross_encoder_precision_v8_observation",
            "test_cross_encoder_precision_v9_performance_observation",
            "test_cross_encoder_precision_v10_performance_observation",
            "test_cross_encoder_precision_v12_performance_observation",
            "test_cross_encoder_precision_v13_observation",
            "test_cross_encoder_precision_v14_performance_observation",
            "test_cross_encoder_precision_v15_observation", "test_cross_encoder_precision_v16_observation",
            "test_github_retrieval_parity", "test_github_retrieval_parity_observation",
        }
        self.assertEqual(set(bridge.MODULE_COUNTS), expected)
        self.assertEqual(sum(bridge.MODULE_COUNTS.values()), 276)
        loader = unittest.TestLoader()
        original = getattr(unittest.TestLoader, routing.ORIGINAL)
        with mock.patch.object(sys, "path", [str(bridge.ROOT / "tests"), *sys.path]):
            for prefix in ("", "tests."):
                total = 0
                for name in sorted(expected):
                    module = importlib.import_module(prefix + name)
                    total += original(loader, module).countTestCases()
                    self.assertEqual(loader.loadTestsFromModule(module).countTestCases(), 1)
                self.assertEqual(total, 276)

    def test_additional_original_test_and_spoofed_root_are_rejected(self):
        name = "test_github_retrieval_parity_observation"
        module = importlib.import_module("tests." + name)
        class ExtraTest(unittest.TestCase):
            def test_added(self):
                pass
        with mock.patch.object(module, "ExtraTest", ExtraTest, create=True):
            with self.assertRaisesRegex(ValueError, "original tests, found 6"):
                unittest.TestLoader().loadTestsFromModule(module)
        with mock.patch.object(module, "__file__", "/wrong/root/" + name + ".py"):
            with self.assertRaisesRegex(ValueError, "module root mismatch"):
                unittest.TestLoader().loadTestsFromModule(module)

    def test_unrelated_loader_and_existing_v5_router_are_preserved(self):
        module = types.ModuleType("unrelated")
        class Test(unittest.TestCase):
            def test_one(self):
                pass
        module.Test = Test
        loader = unittest.TestLoader()
        self.assertEqual(loader.loadTestsFromModule(module).countTestCases(), 1)
        importlib.import_module("tests.test_000_github_retrieval_parity_v5_routing")
        v5 = importlib.import_module("tests.test_github_retrieval_parity_v5")
        self.assertEqual(loader.loadTestsFromModule(v5).countTestCases(), 1)
        for name in ("test_github_retrieval_parity_v5", "test_cross_encoder_precision_v4_observation",
                     "test_cross_encoder_precision_v9", "tests.tests.test_cross_encoder_precision"):
            with self.assertRaisesRegex(ValueError, "unknown historical"):
                bridge.canonical_module(name)

    def test_subprocess_failure_missing_report_and_wrong_execution_count(self):
        name = "test_github_retrieval_parity_observation"
        valid = {"module": name, "loaded": 5, "executed": 5, "successful": True}
        cases = ((1, bridge.RESULT_PREFIX + json.dumps(valid)), (0, "Ran 5 tests\nOK"),
                 (0, bridge.RESULT_PREFIX + json.dumps(dict(valid, executed=4))),
                 (0, bridge.RESULT_PREFIX + json.dumps(valid) + "\n" + bridge.RESULT_PREFIX + json.dumps(valid)))
        for status, output in cases:
            with self.subTest(status=status, output=output):
                completed = subprocess.CompletedProcess([], status, output, "failure details")
                with mock.patch.object(bridge, "materialize_root"), mock.patch.object(
                        bridge.subprocess, "run", return_value=completed):
                    with self.assertRaises(AssertionError):
                        bridge.run_module(name)

    def test_router_runs_target_through_bridge(self):
        module = importlib.import_module("tests.test_github_retrieval_parity_observation")
        suite = unittest.TestLoader().loadTestsFromModule(module)
        with mock.patch.object(bridge, "run_module", return_value="") as run:
            result = unittest.TestResult()
            suite.run(result)
        run.assert_called_once_with("test_github_retrieval_parity_observation")
        self.assertEqual(result.testsRun, 1)
        self.assertTrue(result.wasSuccessful())


if __name__ == "__main__":
    unittest.main()
