"""Source-side checks that keep the docs -> GitHub Wiki sync passable (#283).

The post-release sync copies every top-level ``docs/*.md`` over the Wiki page of the same
name, rebuilds the Wiki ``_Sidebar.md`` head from ``docs/wiki/_Sidebar-head.md``, and keeps
the Wiki-only Decision Structure entries and the sidebar part from the ``**判断構造**`` line
onward untouched. Its pre-push assertions stop the release when the sidebar misses a page, an
index slug has no page, or a Wiki-internal link does not resolve. These tests move the
source-side causes of those stops to the pull request that introduces them.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
HEAD = DOCS / "wiki" / "_Sidebar-head.md"
INDEX = DOCS / "Decision-Structure.md"
BOUNDARY = "**判断構造**"
# _Footer is navigation infrastructure; the Decision-Structure row heads the Wiki-only tail.
NOT_IN_HEAD = {"_Footer", "Decision-Structure"}
WIKI_URL = "https://github.com/Liplus-Project/neuron-graph-rag/wiki/"

MARKDOWN_LINK = re.compile(r"\]\(([^)\s]+)\)")
NATIVE_LINK = re.compile(r"\[\[([^\]]+)\]\]")
INDEX_SLUG = re.compile(re.escape(WIKI_URL) + r"([A-Za-z0-9._-]+)")
KEBAB = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def strip_code(text: str) -> str:
    """Drop notation that is not a link: comments, code elements, fences, indented and inline code."""
    text = text.replace("\r", "")
    text = re.sub(r"<!--.*?(?:-->|\Z)", "", text, flags=re.S)
    text = re.sub(r"<(pre|code)\b[^>]*>.*?(?:</\1>|\Z)", "", text, flags=re.S | re.I)
    kept: list[str] = []
    fence: str | None = None
    for line in text.split("\n"):
        if fence is not None:
            closing = re.fullmatch(r"\s*(`{3,}|~{3,})\s*", line)
            if closing and closing.group(1)[0] == fence[0] and len(closing.group(1)) >= len(fence):
                fence = None
            continue
        opening = re.fullmatch(r"\s*(`{3,}|~{3,}).*", line)
        if opening:
            fence = opening.group(1)
            continue
        if line.startswith("    ") or line.startswith("\t"):
            continue
        kept.append(line)
    text = "\n".join(kept)
    out: list[str] = []
    position = 0
    while position < len(text):
        run = re.match(r"`+", text[position:])
        if run is None:
            out.append(text[position])
            position += 1
            continue
        width = len(run.group(0))
        closing = re.search(r"(?<!`)`{%d}(?!`)" % width, text[position + width :])
        if closing is None:
            out.append(run.group(0))
            position += width
        else:
            position += width + closing.end()
    return "".join(out)


def link_targets(text: str) -> set[str]:
    body = strip_code(text)
    targets = set(MARKDOWN_LINK.findall(body))
    for inner in NATIVE_LINK.findall(body):
        targets.add(inner.split("|", 1)[-1])
    return {target.split("#", 1)[0] for target in targets}


def top_level_pages() -> dict[str, Path]:
    return {path.stem: path for path in sorted(DOCS.glob("*.md"))}


def index_slugs() -> set[str]:
    return set(INDEX_SLUG.findall(strip_code(INDEX.read_text(encoding="utf-8"))))


class SidebarHeadSourceTest(unittest.TestCase):
    """The head links every top-level page and only top-level pages."""

    def setUp(self) -> None:
        self.assertTrue(HEAD.exists(), "docs/wiki/_Sidebar-head.md is missing")
        self.text = HEAD.read_text(encoding="utf-8")
        self.pages = set(top_level_pages())

    def test_every_top_level_page_is_in_the_head(self) -> None:
        missing = sorted(self.pages - NOT_IN_HEAD - link_targets(self.text))
        self.assertEqual(missing, [], "add these top-level docs pages to docs/wiki/_Sidebar-head.md")

    def test_every_head_target_is_a_top_level_page(self) -> None:
        stray = sorted(link_targets(self.text) - self.pages)
        self.assertEqual(stray, [], "docs/wiki/_Sidebar-head.md links pages that are not top-level docs")

    def test_head_does_not_carry_the_boundary_line(self) -> None:
        lines = [line.rstrip("\r") for line in self.text.splitlines()]
        self.assertNotIn(BOUNDARY, lines, "the sync would locate the Wiki-only tail inside the head")

    def test_head_ends_with_one_newline(self) -> None:
        self.assertTrue(self.text.endswith("\n"))
        self.assertFalse(self.text.endswith("\n\n"))


class DecisionIndexOwnershipTest(unittest.TestCase):
    """Index slugs are Wiki-only pages and never share a name with a docs-owned page."""

    def test_index_names_entries(self) -> None:
        self.assertTrue(index_slugs(), "docs/Decision-Structure.md names no Wiki entry")

    def test_index_slugs_do_not_collide_with_docs_pages(self) -> None:
        collisions = sorted(index_slugs() & set(top_level_pages()))
        self.assertEqual(
            collisions,
            [],
            "the sync would overwrite these Wiki-only entries with the docs page of the same name",
        )

    def test_index_slugs_are_lowercase_kebab_case(self) -> None:
        malformed = sorted(slug for slug in index_slugs() if not KEBAB.fullmatch(slug))
        self.assertEqual(malformed, [], "Decision Structure entries are lowercase kebab-case")


class MirroredLinkTest(unittest.TestCase):
    """Relative links in mirrored pages resolve both in docs/ and on the flat Wiki."""

    def test_relative_links_name_top_level_markdown_pages(self) -> None:
        names = {path.name for path in top_level_pages().values()}
        broken: list[str] = []
        for path in top_level_pages().values():
            for target in sorted(set(MARKDOWN_LINK.findall(strip_code(path.read_text(encoding="utf-8"))))):
                if "://" in target or target.startswith("#") or target.startswith("mailto:"):
                    continue
                if target.split("#", 1)[0] not in names:
                    broken.append(f"{path.name}: {target}")
        self.assertEqual(
            broken,
            [],
            "use a top-level docs/*.md target or a blob/main URL; the Wiki has no subfolders",
        )


class StripCodeTest(unittest.TestCase):
    def test_notation_inside_code_is_not_a_link(self) -> None:
        text = (
            "[a](a.md) `[b](b.md)` <!-- [c](c.md) -->\n"
            "```\n[d](d.md)\n```\n"
            "    [e](e.md)\n"
            "<code>[f](f.md)</code> [[Label|g]]\n"
        )
        self.assertEqual(link_targets(text), {"a.md", "g"})


if __name__ == "__main__":
    unittest.main()
