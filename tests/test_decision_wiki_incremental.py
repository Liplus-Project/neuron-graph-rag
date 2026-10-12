from __future__ import annotations

import hashlib
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from neuron_graph_rag import NeuronGraphRAG
from neuron_graph_rag.decision_wiki_import import WikiSource, build_payload, import_atomically
from neuron_graph_rag.decision_wiki_incremental import parse_edges
from tools.import_decision_wikis_incremental import dry_run, write


REGISTERED = {"supersedes", "depends_on", "conflicts_with", "refines", "informs"}


def _page(resolution: str, *edges: str) -> str:
    return f"## Current resolution\n\n{resolution}\n\n## Edges\n\n" + "".join(f"{edge}\n" for edge in edges)


class _Wiki:
    """A one-repository Decision Structure Wiki written to disk."""

    def __init__(self, root: Path) -> None:
        self.clone = root / "one.wiki"
        self.clone.mkdir()
        self.rows: dict[str, tuple[str, str]] = {}

    def put(self, slug: str, resolution: str, *edges: str, state: str = "active") -> None:
        self.rows[slug] = (state, resolution)
        (self.clone / f"{slug}.md").write_text(_page(resolution, *edges), encoding="utf-8")

    def source(self, commit: str = "a" * 40) -> WikiSource:
        index = self.clone / "Decision-Structure.md"
        index.write_text(
            "| Node | State | Current resolution |\n| --- | --- | --- |\n"
            + "".join(
                f"| [{slug}](https://github.com/Example/one/wiki/{slug}) | {state} | {resolution} |\n"
                for slug, (state, resolution) in self.rows.items()
            ),
            encoding="utf-8",
        )
        return WikiSource("Example/one", self.clone, index, commit)


def _link(slug: str) -> str:
    return f"[{slug}]({slug})"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class IncrementalImportTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.root = Path(self._directory.name)
        self.wiki = _Wiki(self.root)
        self.wiki.put("base", "Base resolution", "- **depends on** none")
        self.wiki.put("old", "Old resolution", "- **depends on** " + _link("base"))
        self.database = self.root / "db" / "knowledge.db"
        payload, _ = build_payload((self.wiki.source(),))
        import_atomically(self.database, payload)

    def tearDown(self) -> None:
        self._directory.cleanup()

    def _judgment(self, identity: str) -> dict:
        with NeuronGraphRAG(self.database) as engine:
            return engine.judgments.get(identity)

    def _count(self) -> int:
        with closing(sqlite3.connect(self.database)) as connection:
            return connection.execute("SELECT COUNT(*) FROM judgments").fetchone()[0]

    def test_dry_run_writes_nothing(self) -> None:
        self.wiki.put("fresh", "Fresh resolution", "- **depends on** " + _link("base"))
        before = _sha(self.database)
        plan = dry_run((self.wiki.source("b" * 40),), self.database)
        self.assertEqual(plan["counts"]["add"], 1)
        self.assertEqual(_sha(self.database), before)
        # A read-only WAL reader may create an empty -wal; it must carry no frames.
        wal = self.database.with_name(self.database.name + "-wal")
        self.assertEqual(wal.stat().st_size if wal.exists() else 0, 0)
        self.assertEqual(sorted(path.name for path in self.database.parent.iterdir() if "pre-wiki-increment" in path.name), [])
        self.assertEqual(self._count(), 2)

    def test_add_writes_targets_first_and_backs_up_next_to_database(self) -> None:
        # "a-new" depends on "z-new": alphabetical order would write the dependent first.
        self.wiki.put("a-new", "A new", "- **depends on** " + _link("z-new"))
        self.wiki.put("z-new", "Z new", "- **informs** " + _link("base"))
        result = write((self.wiki.source("b" * 40),), self.database)
        self.assertEqual([item["judgment_id"] for item in result["written"]], ["one:z-new", "one:a-new"])
        backup = Path(result["backup"])
        self.assertEqual(backup.parent, self.database.parent)
        with closing(sqlite3.connect(backup)) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM judgments").fetchone()[0], 2)
        added = self._judgment("one:a-new")
        self.assertEqual(added["revision"], 1)
        self.assertEqual(added["provenance"]["wiki_commit"], "b" * 40)
        self.assertEqual(added["provenance"]["import_mode"], "incremental")
        self.assertEqual([(r["relation_type"], r["target_id"]) for r in added["relations"]], [("depends_on", "one:z-new")])
        with NeuronGraphRAG(self.database) as engine:
            self.assertEqual(engine.store.get_node("one:a-new").text.split("\n")[0], "A new")
        again = write((self.wiki.source("b" * 40),), self.database)
        self.assertEqual(again["written"], [])
        self.assertIsNone(again["backup"])

    def test_supersede_archives_active_predecessor(self) -> None:
        self.wiki.put("new", "New resolution", "- **supersedes** " + _link("old"))
        self.wiki.rows["old"] = ("superseded", "Old resolution")
        result = write((self.wiki.source("b" * 40),), self.database)
        self.assertEqual([(item["action"], item["judgment_id"]) for item in result["written"]], [("supersede", "one:new")])
        old = self._judgment("one:old")
        self.assertEqual((old["lifecycle"], old["superseded_by"]), ("archived", "one:new"))

    def test_supersedes_of_inactive_or_several_predecessors_is_not_written(self) -> None:
        self.wiki.put("first", "First", "- **supersedes** " + _link("old"))
        self.wiki.put("second", "Second", "- **supersedes** " + _link("old"))
        self.wiki.put("third", "Third", "- **supersedes** " + _link("base"), "- **supersedes** " + _link("first"))
        plan = dry_run((self.wiki.source("b" * 40),), self.database)
        actions = {item["judgment_id"]: item["action"] for item in plan["operations"]}
        self.assertEqual(actions["one:first"], "supersede")
        self.assertEqual(actions["one:second"], "add")
        self.assertNotIn("one:third", actions)
        self.assertIn(
            {"judgment_id": "one:second", "line": "- **supersedes** [old](old)",
             "reason": "supersedes target is not active (superseded by one:first)"},
            plan["skipped_edges"],
        )
        self.assertTrue(any(item["judgment_id"] == "one:third" for item in plan["blocked"]))

    def test_revise_only_while_store_still_holds_the_imported_revision(self) -> None:
        self.wiki.put("old", "Old resolution, revised", "- **depends on** " + _link("base"))
        result = write((self.wiki.source("b" * 40),), self.database)
        self.assertEqual([(item["action"], item["revision"]) for item in result["written"]], [("revise", 2)])
        revised = self._judgment("one:old")
        self.assertEqual(revised["statement"], "Old resolution, revised")
        self.assertEqual(revised["provenance"]["wiki_commit"], "b" * 40)
        self.assertEqual(self._judgment("one:base")["revision"], 1)

    def test_changed_on_both_sides_is_listed_and_not_written(self) -> None:
        with NeuronGraphRAG(self.database) as engine:
            engine.judgments.update(
                "one:old", "Store-side rewrite", "Edited through write_judgment",
                {"source": "dialogue"}, expected_revision=1,
                relations=[{"target_id": "one:base", "relation_type": "depends_on"}],
            )
        self.wiki.put("old", "Wiki-side rewrite", "- **depends on** " + _link("base"))
        before = _sha(self.database)
        result = write((self.wiki.source("b" * 40),), self.database)
        self.assertEqual(result["written"], [])
        self.assertEqual(_sha(self.database), before)
        self.assertEqual(
            result["changed_on_both_sides"],
            [{"judgment_id": "one:old", "reason": "changed on both sides", "current_revision": 2, "imported_revision": 1}],
        )
        self.assertEqual(self._judgment("one:old")["statement"], "Store-side rewrite")

    def test_unregistered_relation_is_counted_with_its_line(self) -> None:
        self.wiki.put(
            "fresh", "Fresh",
            "- **depends on** " + _link("base"),
            "- **relates to** " + _link("old"),
            "- `old-entry` に relates to —— prose form",
        )
        plan = dry_run((self.wiki.source("b" * 40),), self.database)
        self.assertEqual(plan["skipped_edge_reasons"], {"unparsed edge line": 1, "unregistered relation type: relates to": 1})
        self.assertIn(
            {"judgment_id": "one:fresh", "line": "- **relates to** [old](old)", "reason": "unregistered relation type: relates to"},
            plan["skipped_edges"],
        )
        result = write((self.wiki.source("b" * 40),), self.database)
        relations = self._judgment("one:fresh")["relations"]
        self.assertEqual([(r["relation_type"], r["target_id"]) for r in relations], [("depends_on", "one:base")])
        self.assertEqual(result["counts"]["skipped_edges"], 2)

    def test_slug_change_of_a_stored_judgment_is_blocked(self) -> None:
        del self.wiki.rows["old"]
        self.wiki.put("decision-old", "Old resolution", "- **depends on** " + _link("base"))
        plan = dry_run((self.wiki.source("b" * 40),), self.database)
        self.assertEqual(plan["operations"], [])
        self.assertEqual(plan["absent_from_index"], ["one:old"])
        self.assertEqual(
            plan["blocked"],
            [{"judgment_id": "one:decision-old", "reason": "possible identity change of stored judgment: one:old"}],
        )

    def test_missing_database_is_never_created(self) -> None:
        missing = self.root / "absent" / "knowledge.db"
        with self.assertRaises(FileNotFoundError):
            write((self.wiki.source(),), missing)
        self.assertFalse(missing.parent.exists())


class EdgeParsingTests(unittest.TestCase):
    def test_target_is_taken_from_the_head_before_the_dash(self) -> None:
        page = _page("R", "- **supersedes** [`a`](https://github.com/Example/one/wiki/a) — prose naming [`b`](https://github.com/Example/one/wiki/b)")
        edges, skipped = parse_edges(page, "Example/one", REGISTERED)
        self.assertEqual([(kind, slug) for kind, _, slug, _ in edges], [("supersedes", "a")])
        self.assertEqual(skipped, [])

    def test_unparsed_lines_are_reported_not_raised(self) -> None:
        page = _page(
            "R",
            "- `a-entry` に **depends on** — target in backticks",
            "- **depends on** [a](a), [b](b)",
            "- なし",
        )
        edges, skipped = parse_edges(page, "Example/one", REGISTERED)
        self.assertEqual(edges, [])
        self.assertEqual([reason for _, reason in skipped], ["unparsed edge line", "edge line has multiple targets"])


if __name__ == "__main__":
    unittest.main()
