from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import tempfile
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

from .judgments import JudgmentContractError
from .storage import SQLiteStore


_SLUG = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_TABLE_LINK = re.compile(r"^\s*\[\s*`?([^]`]+)`?\s*\]\((https://github\.com/([^/]+/[^/]+)/wiki/([^)#]+))\)\s*$")
_HEADING = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
_EDGE_LINE = re.compile(
    r"^\s*[-*]\s+(?:\*\*(supersedes|depends on|conflicts with|refines|informs)(?:[^*]*)\*\*|(supersedes|depends on|conflicts with|refines|informs))\s*:?[ \t]*(.*?)\s*$",
    re.IGNORECASE,
)
_WIKILINK = re.compile(r"\[\[([a-z0-9]+(?:-[a-z0-9]+)*)\]\]")
_MARKDOWN_LINK = re.compile(r"\[[^]]+\]\(([^)]+)\)")
_RELATION_TYPES = {"supersedes", "depends_on", "conflicts_with", "refines", "informs"}


class DecisionWikiImportError(JudgmentContractError):
    """A fail-closed Decision Structure migration error."""


@dataclass(frozen=True)
class WikiSource:
    repository: str
    clone: Path
    index: Path
    commit: str

    @property
    def namespace(self) -> str:
        return self.repository.rsplit("/", 1)[-1]


def _identity(repository: str, slug: str) -> str:
    return f"{repository.rsplit('/', 1)[-1]}:{slug}"


def _section(text: str, names: Iterable[str]) -> str | None:
    matches = list(_HEADING.finditer(text))
    wanted = {name.casefold() for name in names}
    for position, match in enumerate(matches):
        if match.group(1).strip().casefold() in wanted:
            end = matches[position + 1].start() if position + 1 < len(matches) else len(text)
            return text[match.end() : end].strip()
    return None


def parse_index(source: WikiSource) -> list[dict[str, str]]:
    text = source.index.read_text(encoding="utf-8")
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for line in text.splitlines():
        if not line.lstrip().startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if not cells:
            continue
        link = _TABLE_LINK.fullmatch(cells[0])
        if link is None:
            continue
        label, url, repository, slug = link.groups()
        if repository.casefold() != source.repository.casefold() or label != slug:
            raise DecisionWikiImportError(f"ambiguous index identity: {line}")
        if _SLUG.fullmatch(slug) is None or slug in seen:
            raise DecisionWikiImportError(f"invalid or duplicate index slug: {slug}")
        seen.add(slug)
        if len(cells) >= 3:
            state, resolution = cells[1], cells[2]
        elif len(cells) == 2:
            state, resolution = "active", cells[1]
        else:
            raise DecisionWikiImportError(f"index row has no resolution: {slug}")
        if not resolution or state not in {"active", "archived", "superseded", "evaluating"}:
            raise DecisionWikiImportError(f"unsupported index state or empty resolution: {slug}")
        rows.append({"slug": slug, "state": state, "resolution": resolution, "wiki_url": url})
    if not rows:
        raise DecisionWikiImportError(f"no Decision Structure entries in {source.index}")
    return rows


def _target_from_url(url: str, current_repository: str) -> tuple[str, str] | None:
    parsed = urlparse(url)
    if not parsed.scheme and not parsed.netloc:
        slug = parsed.path.strip("/")
        return (current_repository, slug) if _SLUG.fullmatch(slug) else None
    parts = parsed.path.strip("/").split("/")
    if parsed.netloc.casefold() == "github.com" and len(parts) == 4 and parts[2] == "wiki":
        return f"{parts[0]}/{parts[1]}", parts[3]
    return None


def parse_relations(page: str, repository: str) -> list[tuple[str, str, str]]:
    edges = _section(page, ("Edges",))
    if edges is None:
        return []
    relations: list[tuple[str, str, str]] = []
    for line in edges.splitlines():
        match = _EDGE_LINE.fullmatch(line)
        if match is None:
            if line.strip() and re.search(r"\b(supersedes|depends on|conflicts with|refines|informs)\b", line, re.I):
                raise DecisionWikiImportError(f"ambiguous edge declaration: {line}")
            continue
        kind = (match.group(1) or match.group(2)).casefold().replace(" ", "_")
        if kind not in _RELATION_TYPES:
            raise DecisionWikiImportError(f"unsupported relation: {kind}")
        body = match.group(3)
        if not body or re.match(r"^(none|（.*なし.*）|\(.*none.*\))", body, re.I):
            continue
        targets = [(repository, slug) for slug in _WIKILINK.findall(body)]
        for url in _MARKDOWN_LINK.findall(body):
            target = _target_from_url(url, repository)
            if target is not None:
                targets.append(target)
        targets = list(dict.fromkeys(targets))
        if len(targets) > 1:
            raise DecisionWikiImportError(f"edge line has multiple targets: {line}")
        if targets:
            relations.append((kind, targets[0][0], targets[0][1]))
    if len(relations) != len(set(relations)):
        raise DecisionWikiImportError("duplicate relation")
    return sorted(relations)


def build_payload(sources: Iterable[WikiSource]) -> tuple[dict[str, Any], dict[str, Any]]:
    sources = tuple(sources)
    records: list[dict[str, Any]] = []
    indexed: set[str] = set()
    source_rows: list[tuple[WikiSource, dict[str, str]]] = []
    for source in sources:
        for row in parse_index(source):
            identity = _identity(source.repository, row["slug"])
            if identity in indexed:
                raise DecisionWikiImportError(f"duplicate identity: {identity}")
            indexed.add(identity)
            source_rows.append((source, row))

    incoming_supersession: dict[str, str] = {}
    pending: list[tuple[str, str, str]] = []
    for source, row in source_rows:
        page_path = source.clone / f'{row["slug"]}.md'
        if not page_path.is_file():
            raise DecisionWikiImportError(f"indexed page is missing: {page_path}")
        page_bytes = page_path.read_bytes()
        try:
            page = page_bytes.decode("utf-8")
        except UnicodeDecodeError as error:
            raise DecisionWikiImportError(f"page is not UTF-8: {page_path}") from error
        source_id = _identity(source.repository, row["slug"])
        parsed = parse_relations(page, source.repository)
        relations: list[dict[str, str]] = []
        for kind, target_repository, target_slug in parsed:
            target_id = _identity(target_repository, target_slug)
            if target_id not in indexed:
                raise DecisionWikiImportError(f"unknown relation target: {source_id} -> {target_id}")
            relations.append({"target_id": target_id, "relation_type": kind})
            pending.append((source_id, target_id, kind))
            if kind == "supersedes":
                if target_id in incoming_supersession:
                    raise DecisionWikiImportError(f"multiple successors for {target_id}")
                incoming_supersession[target_id] = source_id
        statement = row["resolution"] or _section(page, ("Current resolution", "判断"))
        if not statement:
            raise DecisionWikiImportError(f"no resolution for {source_id}")
        records.append(
            {
                "judgment_id": source_id,
                "revision": 1,
                "statement": statement,
                "rationale": page,
                "provenance": {
                    "page_body": page,
                    "page_sha256": hashlib.sha256(page_bytes).hexdigest(),
                    "repository": source.repository,
                    "source_state": row["state"],
                    "wiki_commit": source.commit,
                    "wiki_url": row["wiki_url"],
                },
                "lifecycle": "archived" if row["state"] in {"archived", "superseded"} else "active",
                "superseded_by": None,
                "relations": relations,
            }
        )
    by_id = {record["judgment_id"]: record for record in records}
    for record in records:
        if record["provenance"]["source_state"] == "superseded" and record["judgment_id"] not in incoming_supersession:
            raise DecisionWikiImportError(f"superseded entry has no successor: {record['judgment_id']}")
    for predecessor, successor in incoming_supersession.items():
        record = by_id[predecessor]
        if record["lifecycle"] == "active" and record["provenance"]["source_state"] != "active":
            raise DecisionWikiImportError(f"supersession state conflict: {predecessor}")
        record["lifecycle"] = "archived"
        record["superseded_by"] = successor
    records.sort(key=lambda item: item["judgment_id"])
    manifest = {
        "format": "ngr-decision-wiki-pilot/v1",
        "judgment_count": len(records),
        "relation_count": len(pending),
        "lifecycle_counts": {
            state: sum(record["lifecycle"] == state for record in records)
            for state in ("active", "archived")
        },
        "repositories": {
            source.repository: {
                "commit": source.commit,
                "judgment_count": sum(
                    record["provenance"]["repository"] == source.repository for record in records
                ),
            }
            for source in sources
        },
    }
    return {"format": "ngr-judgment-graph/v1", "judgments": records}, manifest


def import_atomically(destination: Path, payload: dict[str, Any]) -> None:
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite database: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    try:
        store = SQLiteStore(temporary)
        try:
            from .judgments import JudgmentGraph

            JudgmentGraph(store).import_graph(payload)
        finally:
            store.close()
        with closing(sqlite3.connect(temporary)) as connection:
            if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise DecisionWikiImportError("SQLite integrity check failed")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def deterministic_json(payload: object) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
