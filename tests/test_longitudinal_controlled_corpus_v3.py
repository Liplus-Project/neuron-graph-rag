from __future__ import annotations

import re
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CORPUS_ROOT = REPOSITORY_ROOT / "corpora" / "repository-native-controlled-v3"
CLUSTERS = (
    "signal-stability",
    "boundary-recovery",
    "evidence-continuity",
)
CEILINGS = (0, 1, 3, 10)
RELATIVE_LINK = re.compile(r"\[[^]]+\]\(([^)]+\.md)\)")


class LongitudinalControlledCorpusV3Test(unittest.TestCase):
    def test_cluster_topology_and_feedback_headroom_are_auditable(self) -> None:
        expected_documents = {
            f"{cluster}-overview.md" for cluster in CLUSTERS
        } | {
            f"{cluster}-credit-{ceiling}.md"
            for cluster in CLUSTERS
            for ceiling in CEILINGS
        }
        actual_documents = {path.name for path in CORPUS_ROOT.glob("*.md")} - {"README.md"}
        self.assertEqual(actual_documents, expected_documents)

        node_ids: set[str] = set()
        source_urls: set[str] = set()
        credited_edges: set[tuple[str, str]] = set()

        for cluster in CLUSTERS:
            overview = CORPUS_ROOT / f"{cluster}-overview.md"
            overview_text = overview.read_text(encoding="utf-8")
            self.assertIn("## 文書間の明示的な参照", overview_text)
            self.assertIn("- Feedback credit ceiling: `10`", overview_text)
            self.assertEqual(
                RELATIVE_LINK.findall(overview_text),
                [f"{cluster}-credit-{ceiling}.md" for ceiling in CEILINGS],
            )

            for ceiling in CEILINGS:
                document = CORPUS_ROOT / f"{cluster}-credit-{ceiling}.md"
                text = document.read_text(encoding="utf-8")
                self.assertNotRegex(text, RELATIVE_LINK)
                self.assertIn(f"- Corpus cluster: `{cluster}`", text)
                self.assertIn(f"- Feedback credit ceiling: `{ceiling}`", text)

            for document in (overview, *(CORPUS_ROOT / f"{cluster}-credit-{ceiling}.md" for ceiling in CEILINGS)):
                text = document.read_text(encoding="utf-8")
                node_id = re.search(r"^- Corpus node ID: `([^`]+)`$", text, re.MULTILINE)
                source_url = re.search(r"^- Source URL: (https://\S+)$", text, re.MULTILINE)
                self.assertIsNotNone(node_id)
                self.assertIsNotNone(source_url)
                self.assertTrue(node_id.group(1).startswith("v3-"))
                self.assertIn("repository-native-controlled-v3", source_url.group(1))
                self.assertTrue(source_url.group(1).endswith(f"/{document.name}"))
                self.assertNotIn(node_id.group(1), node_ids)
                self.assertNotIn(source_url.group(1), source_urls)
                node_ids.add(node_id.group(1))
                source_urls.add(source_url.group(1))

            for target in RELATIVE_LINK.findall(overview_text):
                edge = (overview.name, target)
                self.assertNotIn(edge, credited_edges)
                credited_edges.add(edge)

        self.assertEqual(len(node_ids), len(expected_documents))
        self.assertEqual(len(source_urls), len(expected_documents))
        self.assertEqual(len(credited_edges), len(CLUSTERS) * len(CEILINGS))

    def test_no_v3_document_uses_an_evaluation_artifact_or_prior_corpus_identity(self) -> None:
        prohibited_terms = (
            "tests/fixtures",
            ".gold.",
            ".schedule.",
            ".manifest.",
            ".result.",
            "repository-native-controlled-v1",
            "repository-native-controlled-v2",
        )
        for document in CORPUS_ROOT.glob("*.md"):
            text = document.read_text(encoding="utf-8")
            for term in prohibited_terms:
                self.assertNotIn(term, text, document.name)

        v3_node_ids = set()
        v3_source_urls = set()
        for document in CORPUS_ROOT.glob("*.md"):
            text = document.read_text(encoding="utf-8")
            v3_node_ids.update(re.findall(r"^- Corpus node ID: `([^`]+)`$", text, re.MULTILINE))
            v3_source_urls.update(re.findall(r"^- Source URL: (https://\S+)$", text, re.MULTILINE))

        prior_node_ids = set()
        prior_source_urls = set()
        for prior_corpus in REPOSITORY_ROOT.glob("corpora/repository-native-controlled-v[12]"):
            for document in prior_corpus.glob("*.md"):
                text = document.read_text(encoding="utf-8")
                prior_node_ids.update(re.findall(r"^- Corpus node ID: `([^`]+)`$", text, re.MULTILINE))
                prior_source_urls.update(re.findall(r"^- Source URL: (https://\S+)$", text, re.MULTILINE))

        self.assertTrue(v3_node_ids.isdisjoint(prior_node_ids))
        self.assertTrue(v3_source_urls.isdisjoint(prior_source_urls))


if __name__ == "__main__":
    unittest.main()
