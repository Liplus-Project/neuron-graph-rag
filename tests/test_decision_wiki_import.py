from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from neuron_graph_rag import NeuronGraphRAG
from neuron_graph_rag.decision_wiki_import import (
    DecisionWikiImportError,
    WikiSource,
    build_payload,
    deterministic_json,
    import_atomically,
)


def _source(root: Path, repository: str, rows: str, pages: dict[str, str]) -> WikiSource:
    clone = root / repository.rsplit("/", 1)[-1]
    clone.mkdir()
    index = clone / "Decision-Structure.md"
    index.write_text(
        "# Decision Structure\n\n| Node | State | Current resolution |\n"
        "| --- | --- | --- |\n" + rows,
        encoding="utf-8",
    )
    for slug, page in pages.items():
        (clone / f"{slug}.md").write_text(page, encoding="utf-8")
    return WikiSource(repository, clone, index, "a" * 40)


class DecisionWikiImportTests(unittest.TestCase):
    def test_namespace_relations_lifecycle_and_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = _source(
                root, "Example/one",
                "| [same](https://github.com/Example/one/wiki/same) | archived | Old |\n",
                {"same": "## Current resolution\n\nOld\n\n## Edges\n\n- **depends on** none\n"},
            )
            second = _source(
                root, "Example/two",
                "| [same](https://github.com/Example/two/wiki/same) | active | New |\n"
                "| [next](https://github.com/Example/two/wiki/next) | active | Next |\n",
                {
                    "same": "## Current resolution\n\nNew\n\n## Edges\n\n- **informs** [next](next)\n",
                    "next": "## Current resolution\n\nNext\n\n## Edges\n\n- **supersedes** [same](same)\n",
                },
            )
            payload, manifest = build_payload((first, second))
            by_id = {item["judgment_id"]: item for item in payload["judgments"]}
            self.assertEqual(set(by_id), {"one:same", "two:same", "two:next"})
            self.assertEqual(by_id["two:same"]["superseded_by"], "two:next")
            self.assertEqual(by_id["two:same"]["lifecycle"], "archived")
            self.assertEqual(by_id["two:next"]["relations"][0]["relation_type"], "supersedes")
            self.assertEqual(by_id["one:same"]["provenance"]["wiki_commit"], "a" * 40)
            self.assertEqual(manifest["judgment_count"], 3)
            self.assertEqual(manifest["relation_count"], 2)

    def test_unknown_relation_target_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = _source(
                Path(directory), "Example/one",
                "| [source](https://github.com/Example/one/wiki/source) | active | Source |\n",
                {"source": "## Edges\n\n- **depends on** [missing](missing)\n"},
            )
            with self.assertRaisesRegex(DecisionWikiImportError, "unknown relation target"):
                build_payload((source,))

    def test_duplicate_index_identity_and_ambiguous_edge_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _source(
                root, "Example/one",
                "| [same](https://github.com/Example/one/wiki/same) | active | Same |\n"
                "| [same](https://github.com/Example/one/wiki/same) | active | Same again |\n",
                {"same": "## Edges\n\n- **depends on** none\n"},
            )
            with self.assertRaisesRegex(DecisionWikiImportError, "duplicate"):
                build_payload((source,))
            source.index.write_text(
                "| [same](https://github.com/Example/one/wiki/same) | active | Same |\n"
                "| [other](https://github.com/Example/one/wiki/other) | active | Other |\n",
                encoding="utf-8",
            )
            (source.clone / "same.md").write_text(
                "## Edges\n\n- **depends on** [other](other) and [same](same)\n", encoding="utf-8"
            )
            (source.clone / "other.md").write_text("## Edges\n\n- **informs** none\n", encoding="utf-8")
            with self.assertRaisesRegex(DecisionWikiImportError, "multiple targets"):
                build_payload((source,))

    def test_atomic_import_refuses_overwrite_and_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = {
                "format": "ngr-judgment-graph/v1",
                "judgments": [{
                    "judgment_id": "one:a", "revision": 1, "statement": "A",
                    "rationale": "Reason", "provenance": {"source": "wiki"},
                    "lifecycle": "active", "superseded_by": None, "relations": [],
                }],
            }
            destination = root / "pilot.sqlite"
            import_atomically(destination, payload)
            with NeuronGraphRAG(destination) as engine:
                first = deterministic_json(engine.judgments.export())
                second = deterministic_json(engine.judgments.export())
            self.assertEqual(first, second)
            before = destination.read_bytes()
            with self.assertRaises(FileExistsError):
                import_atomically(destination, payload)
            self.assertEqual(destination.read_bytes(), before)
            broken = root / "broken.sqlite"
            invalid = {"format": "ngr-judgment-graph/v1", "judgments": [{"judgment_id": "bad"}]}
            with self.assertRaises(Exception):
                import_atomically(broken, invalid)
            self.assertFalse(broken.exists())


if __name__ == "__main__":
    unittest.main()
