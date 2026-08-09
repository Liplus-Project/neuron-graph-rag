from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from neuron_graph_rag import EngineConfig, NeuronGraphRAG
from neuron_graph_rag.github_source import GitHubSnapshot, changed_paths, index_github_snapshot
from tools.acquire_github_snapshot import acquire_snapshot
from tools.run_github_rag_compatibility import run_compatibility


FIXTURES = Path(__file__).parent / "fixtures"
SNAPSHOT = FIXTURES / "github_rag_compatibility.snapshot.json"
UPDATED_SNAPSHOT = FIXTURES / "github_rag_compatibility.updated.snapshot.json"
CASES = FIXTURES / "github_rag_compatibility.cases.json"
RESULT = FIXTURES / "github_rag_compatibility.result.json"


class GitHubRagCompatibilityTest(unittest.TestCase):
    def test_snapshot_indexes_deterministically_with_provenance(self) -> None:
        snapshot = GitHubSnapshot.read(SNAPSHOT)
        config = EngineConfig(
            sparse_weight=1.0,
            dense_weight=0.0,
            entry_weight=1.0,
            graph_weight=0.0,
            use_dense_retrieval=False,
            use_graph_propagation=False,
        )
        with NeuronGraphRAG(config=config) as engine:
            receipt = index_github_snapshot(engine, snapshot)
            trace = engine.search("dense sparse rerank", now=0.0)

        self.assertEqual(receipt.node_ids, ("github:Liplus-Project/github-rag-mcp:docs/Home.md",))
        self.assertTrue(receipt.fingerprint.startswith("sha256:"))
        self.assertEqual(trace.hits[0].node.metadata["source_adapter"], "github_read_only_snapshot")
        self.assertEqual(trace.hits[0].node.metadata["source_url"], SNAPSHOT_URL)

    def test_runner_compares_expected_sources_and_follows_one_update(self) -> None:
        result = run_compatibility(SNAPSHOT, CASES, updated_snapshot_path=UPDATED_SNAPSHOT)

        self.assertEqual(result["verdict"], "continue_candidate")
        self.assertTrue(all(case["expected_source_found"] for case in result["comparisons"]))
        self.assertEqual(result["source_update"]["changed_paths"], ["docs/Home.md"])
        self.assertTrue(result["source_update"]["followed"])

    def test_committed_result_is_the_canonical_observation(self) -> None:
        result = run_compatibility(SNAPSHOT, CASES, updated_snapshot_path=UPDATED_SNAPSHOT)
        canonical = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"

        self.assertEqual(RESULT.read_text(encoding="utf-8"), canonical)

    def test_changed_paths_rejects_cross_repository_comparison(self) -> None:
        before = GitHubSnapshot.read(SNAPSHOT)
        updated_value = json.loads(UPDATED_SNAPSHOT.read_text(encoding="utf-8"))
        updated_value["repository"] = "other/repository"
        updated_value["documents"][0]["source_url"] = (
            "https://github.com/other/repository/blob/"
            + updated_value["commit"]
            + "/docs/Home.md"
        )
        after = GitHubSnapshot.from_mapping(updated_value)

        with self.assertRaisesRegex(ValueError, "same GitHub repository"):
            changed_paths(before, after)

    def test_acquisition_uses_only_pinned_get_requests(self) -> None:
        responses = [
            {"sha": "c" * 40},
            {"type": "file", "encoding": "base64", "sha": "d" * 40, "content": "SGVsbG8="},
        ]
        with patch("tools.acquire_github_snapshot.github_get", side_effect=responses) as get:
            snapshot = acquire_snapshot("octo/repo", "main", ["README.md"])

        self.assertEqual(snapshot["commit"], "c" * 40)
        self.assertEqual(snapshot["documents"][0]["content"], "Hello")
        self.assertEqual(get.call_args_list[0].args[0], "/repos/octo/repo/commits/main")
        self.assertIn("ref=" + "c" * 40, get.call_args_list[1].args[0])


SNAPSHOT_URL = "https://github.com/Liplus-Project/github-rag-mcp/blob/53d8feec58a23e098f41e370910e824187adc84e/docs/Home.md"


if __name__ == "__main__":
    unittest.main()
