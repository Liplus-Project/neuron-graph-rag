from __future__ import annotations

import json
import re
import unittest
from collections import defaultdict
from pathlib import Path

from neuron_graph_rag.corpus_integrity import verify_source_sha256


ROOT = Path(__file__).resolve().parents[1]
CORPUS_ROOT = ROOT / "corpora" / "feedback-policy-comparison-v1"
MANIFEST_PATH = CORPUS_ROOT / "manifest.json"
RELATIVE_LINK = re.compile(r"\[[^]]+\]\(([^)]+\.md)\)")
EXPLICIT_REFERENCES = "## 文書間の明示的な参照\n"


def _manifest() -> dict[str, object]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _documents(manifest: dict[str, object]):
    for split_name, split in manifest["splits"].items():
        for document in split["documents"]:
            yield split_name, document


def _edge_identity(edge: dict[str, str]) -> tuple[str, str, str]:
    return edge["source_id"], edge["target_id"], edge["edge_type"]


class FeedbackPolicyComparisonCorpusTest(unittest.TestCase):
    def test_manifest_inventory_hashes_and_newlines_are_deterministic(self) -> None:
        raw_manifest = MANIFEST_PATH.read_bytes()
        manifest = json.loads(raw_manifest.decode("utf-8", errors="strict"))

        self.assertNotIn(b"\r", raw_manifest)
        self.assertEqual(
            raw_manifest.decode("utf-8"),
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        )
        self.assertEqual(manifest["corpus_id"], "feedback-policy-comparison-v1")
        self.assertEqual(manifest["phase"], "corpus-only")
        self.assertEqual(manifest["newline_policy"]["canonical"], "LF")

        listed_paths = set()
        for split_name, document in _documents(manifest):
            path = ROOT / document["path"]
            listed_paths.add(path)
            raw = path.read_bytes()
            text = raw.decode("utf-8", errors="strict")

            self.assertNotIn(b"\r", raw, document["path"])
            self.assertTrue(raw.endswith(b"\n"), document["path"])
            self.assertEqual(
                verify_source_sha256(path, document["raw_sha256"]).decision,
                "raw_match",
                document["path"],
            )
            self.assertIn(f"- Corpus split: `{split_name}`", text)
            self.assertIn(f"- Corpus cluster: `{document['cluster_id']}`", text)
            self.assertIn(f"- Corpus node ID: `{document['node_id']}`", text)
            self.assertIn(f"- Source URL: {document['source_url']}", text)
            self.assertIn("- Provenance: repository-authored public source", text)

        actual_paths = set(CORPUS_ROOT.glob("*.md")) - {CORPUS_ROOT / "README.md"}
        self.assertEqual(listed_paths, actual_paths)
        self.assertEqual(len(listed_paths), 16)

    def test_manifest_edges_are_recomputed_from_explicit_links(self) -> None:
        manifest = _manifest()

        for split_name, split in manifest["splits"].items():
            documents = split["documents"]
            by_filename = {Path(item["path"]).name: item for item in documents}
            by_node_id = {item["node_id"]: item for item in documents}
            derived_edges = []

            for document in documents:
                text = (ROOT / document["path"]).read_text(encoding="utf-8")
                all_links = RELATIVE_LINK.findall(text)
                if EXPLICIT_REFERENCES in text:
                    reference_body = text.split(EXPLICIT_REFERENCES, 1)[1]
                    self.assertEqual(all_links, RELATIVE_LINK.findall(reference_body))
                else:
                    self.assertEqual(all_links, [])

                for target_filename in all_links:
                    self.assertIn(target_filename, by_filename)
                    target = by_filename[target_filename]
                    self.assertEqual(target["cluster_id"], document["cluster_id"])
                    derived_edges.append(
                        (document["node_id"], target["node_id"], "mention")
                    )

            registered_edges = [_edge_identity(edge) for edge in split["edges"]]
            self.assertEqual(derived_edges, registered_edges)

            cluster_nodes: dict[str, set[str]] = defaultdict(set)
            for document in documents:
                cluster_nodes[document["cluster_id"]].add(document["node_id"])
            self.assertEqual(len(cluster_nodes), 2)

            for cluster_id, node_ids in cluster_nodes.items():
                self.assertEqual(len(node_ids), 4, cluster_id)
                cluster_edges = [
                    edge
                    for edge in registered_edges
                    if edge[0] in node_ids or edge[1] in node_ids
                ]
                self.assertTrue(
                    all(source in node_ids and target in node_ids for source, target, _ in cluster_edges)
                )
                self.assertEqual(len(cluster_edges), 3)

                outdegree = {node_id: 0 for node_id in node_ids}
                indegree = {node_id: 0 for node_id in node_ids}
                for source_id, target_id, edge_type in cluster_edges:
                    self.assertEqual(edge_type, "mention")
                    outdegree[source_id] += 1
                    indegree[target_id] += 1
                self.assertEqual(sorted(outdegree.values()), [0, 0, 1, 2])
                self.assertEqual(sorted(indegree.values()), [0, 1, 1, 1])

                branch_source = next(node for node, degree in outdegree.items() if degree == 2)
                branch_targets = {
                    target for source, target, _ in cluster_edges if source == branch_source
                }
                self.assertEqual(
                    sorted(outdegree[target] for target in branch_targets),
                    [0, 1],
                )
                self.assertEqual(by_node_id[branch_source]["cluster_id"], cluster_id)

    def test_split_and_prior_fixture_identities_are_disjoint(self) -> None:
        manifest = _manifest()
        split_identities: dict[str, set[str]] = {}

        for split_name, split in manifest["splits"].items():
            identities = {split["split_id"]}
            for document in split["documents"]:
                identities.update(
                    {
                        document["node_id"],
                        document["path"],
                        document["source_url"],
                    }
                )
            for edge in split["edges"]:
                identities.add("|".join(_edge_identity(edge)))
            split_identities[split_name] = identities

        self.assertTrue(
            split_identities["development"].isdisjoint(split_identities["holdout"])
        )

        prior_files = [
            path
            for path in (ROOT / "corpora").rglob("*")
            if path.is_file() and CORPUS_ROOT not in path.parents
        ]
        fixture_files = [path for path in (ROOT / "tests" / "fixtures").rglob("*") if path.is_file()]
        fixture_names = {path.name.lower() for path in fixture_files}
        self.assertTrue(any("feedback" in name for name in fixture_names))
        self.assertTrue(any("rank_elasticity" in name for name in fixture_names))
        self.assertTrue(any("channels_blind" in name for name in fixture_names))
        self.assertTrue(any("channels_node_first" in name for name in fixture_names))

        prior_bytes = b"\0".join(path.read_bytes() for path in prior_files + fixture_files)
        for split_name, identities in split_identities.items():
            for identity in identities:
                self.assertNotIn(identity.encode("utf-8"), prior_bytes, (split_name, identity))

    def test_corpus_has_no_evaluation_payload(self) -> None:
        manifest = _manifest()
        prohibited_keys = {
            "query",
            "gold",
            "feedback_schedule",
            "expected_node",
            "policy_arm",
            "outcome_role",
            "acceptance_condition",
            "rank",
            "mrr",
            "gate",
            "result",
            "engine_config",
        }

        def visit(value: object) -> None:
            if isinstance(value, dict):
                self.assertTrue(prohibited_keys.isdisjoint(value))
                for child in value.values():
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)

        visit(manifest)

        prohibited_markers = re.compile(
            r"\b(query|gold|feedback schedule|expected node|policy arm|outcome role|"
            r"acceptance condition|rank|mrr|gate|observed result|engineconfig)\b",
            re.IGNORECASE,
        )
        for path in CORPUS_ROOT.glob("*.md"):
            self.assertIsNone(prohibited_markers.search(path.read_text(encoding="utf-8")), path.name)


if __name__ == "__main__":
    unittest.main()
