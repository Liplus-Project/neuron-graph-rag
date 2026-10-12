"""Incremental Decision Structure Wiki import into an existing judgment store.

The plan is computed from read-only SQL. Writes go through ``JudgmentGraph``
(add / supersede / update / archive), the same domain API ``write_judgment`` uses.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable

from .decision_wiki_import import (
    _MARKDOWN_LINK,
    _WIKILINK,
    DecisionWikiImportError,
    WikiSource,
    _identity,
    _section,
    _target_from_url,
    parse_index,
)
from .judgments import JudgmentGraph


PLAN_FORMAT = "ngr-decision-wiki-incremental-plan/v1"
IMPORT_MODE = "incremental"

_BULLET = re.compile(r"^\s*[-*]\s+(.*?)\s*$")
_BOLD_LABEL = re.compile(r"^\*\*([^*]+?)\*\*\s*[:：]?[ \t]*(.*)$")
_NONE_BODY = re.compile(r"^(none|（.*なし.*）|\(.*none.*\))", re.IGNORECASE)
_HEAD_SEPARATOR = re.compile(r"\s+[—–]+\s*|——")
_BACKTICK_SLUG = re.compile(r"`[a-z0-9]+(?:-[a-z0-9]+)+`")


@dataclass
class WikiEntry:
    judgment_id: str
    statement: str
    page: str
    page_sha256: str
    source_state: str
    repository: str
    commit: str
    wiki_url: str
    edges: list[tuple[str, str, str]] = field(default_factory=list)  # (kind, target_id, line)

    @property
    def lifecycle(self) -> str:
        return "archived" if self.source_state in {"archived", "superseded"} else "active"

    def provenance(self) -> dict[str, Any]:
        return {
            "import_mode": IMPORT_MODE,
            "page_body": self.page,
            "page_sha256": self.page_sha256,
            "repository": self.repository,
            "source_state": self.source_state,
            "wiki_commit": self.commit,
            "wiki_url": self.wiki_url,
        }


def _label_kind(label: str, relation_types: set[str]) -> str | None:
    """Return the registered relation type a label starts with, if any."""
    folded = label.strip().casefold()
    for kind in sorted(relation_types, key=len, reverse=True):
        spoken = kind.replace("_", " ")
        if folded == spoken or (
            folded.startswith(spoken) and not folded[len(spoken)].isalpha()
        ):
            return kind
    return None


def _targets(text: str, repository: str) -> list[tuple[str, str]]:
    targets = [(repository, slug) for slug in _WIKILINK.findall(text)]
    for url in _MARKDOWN_LINK.findall(text):
        target = _target_from_url(url, repository)
        if target is not None:
            targets.append(target)
    return list(dict.fromkeys(targets))


def parse_edges(
    page: str, repository: str, relation_types: set[str]
) -> tuple[list[tuple[str, str, str, str]], list[tuple[str, str]]]:
    """Parse ``## Edges`` leniently.

    Returns ``(edges, skipped)`` where each edge is
    ``(kind, target_repository, target_slug, line)`` and each skipped item is
    ``(line, reason)``. Nothing raises: a line that cannot be read as an edge of a
    registered type is reported instead of failing the whole import.
    """
    section = _section(page, ("Edges",))
    if section is None:
        return [], []
    edges: list[tuple[str, str, str, str]] = []
    skipped: list[tuple[str, str]] = []
    for line in section.splitlines():
        bullet = _BULLET.fullmatch(line)
        if bullet is None:
            continue
        body = bullet.group(1)
        kind: str | None = None
        rest = ""
        label: str | None = None
        bold = _BOLD_LABEL.match(body)
        if bold is not None:
            label = bold.group(1).strip()
            kind = _label_kind(label, relation_types)
            rest = bold.group(2)
        else:
            for candidate in sorted(relation_types, key=len, reverse=True):
                spoken = candidate.replace("_", " ")
                match = re.match(rf"^{re.escape(spoken)}(?![A-Za-z])\s*[:：]?[ \t]*(.*)$", body, re.IGNORECASE)
                if match is not None:
                    kind, rest = candidate, match.group(1)
                    break
        if kind is None:
            if label is not None and _targets(body, repository):
                skipped.append((line.strip(), f"unregistered relation type: {label.casefold()}"))
            elif _targets(body, repository) or _BACKTICK_SLUG.search(body):
                skipped.append((line.strip(), "unparsed edge line"))
            continue
        if not rest or _NONE_BODY.match(rest):
            continue
        head = _HEAD_SEPARATOR.split(rest, maxsplit=1)[0]
        targets = _targets(head, repository) or _targets(rest, repository)
        if not targets:
            if _BACKTICK_SLUG.search(rest):
                skipped.append((line.strip(), "unparsed edge line"))
            continue
        if len(targets) > 1:
            skipped.append((line.strip(), "edge line has multiple targets"))
            continue
        edges.append((kind, targets[0][0], targets[0][1], line.strip()))
    return edges, skipped


def read_wiki_entries(
    sources: Iterable[WikiSource], relation_types: set[str]
) -> tuple[dict[str, WikiEntry], list[dict[str, str]], list[dict[str, str]]]:
    """Read every indexed page. Returns entries, skipped edge lines, and blocked entries."""
    entries: dict[str, WikiEntry] = {}
    skipped: list[dict[str, str]] = []
    blocked: list[dict[str, str]] = []
    for source in sources:
        for row in parse_index(source):
            identity = _identity(source.repository, row["slug"])
            if identity in entries:
                raise DecisionWikiImportError(f"duplicate identity: {identity}")
            page_path = source.clone / f'{row["slug"]}.md'
            if not page_path.is_file():
                blocked.append({"judgment_id": identity, "reason": "indexed page is missing"})
                continue
            page_bytes = page_path.read_bytes()
            try:
                page = page_bytes.decode("utf-8")
            except UnicodeDecodeError:
                blocked.append({"judgment_id": identity, "reason": "page is not UTF-8"})
                continue
            parsed, page_skipped = parse_edges(page, source.repository, relation_types)
            entry = WikiEntry(
                judgment_id=identity,
                statement=row["resolution"],
                page=page,
                page_sha256=hashlib.sha256(page_bytes).hexdigest(),
                source_state=row["state"],
                repository=source.repository,
                commit=source.commit,
                wiki_url=row["wiki_url"],
            )
            seen: set[tuple[str, str]] = set()
            for kind, target_repository, target_slug, line in parsed:
                target = _identity(target_repository, target_slug)
                if (kind, target) in seen:
                    continue
                seen.add((kind, target))
                entry.edges.append((kind, target, line))
            entries[identity] = entry
            skipped.extend(
                {"judgment_id": identity, "line": line, "reason": reason}
                for line, reason in page_skipped
            )
    return entries, skipped, blocked


@dataclass
class _StoreJudgment:
    judgment_id: str
    current_revision: int
    lifecycle: str
    superseded_by: str | None
    supersedes: set[str]


def _load_store(connection: sqlite3.Connection) -> tuple[dict[str, _StoreJudgment], set[str]]:
    store: dict[str, _StoreJudgment] = {}
    for judgment_id, revision, lifecycle, successor in connection.execute(
        "SELECT judgment_id, current_revision, lifecycle, superseded_by FROM judgments"
    ):
        store[judgment_id] = _StoreJudgment(judgment_id, revision, lifecycle, successor, set())
    for source_id, target_id in connection.execute(
        "SELECT source_id, target_id FROM judgment_relations WHERE relation_type = 'supersedes'"
    ):
        if source_id in store:
            store[source_id].supersedes.add(target_id)
    registered = {
        row[0]
        for row in connection.execute(
            "SELECT relation_type FROM judgment_relation_types WHERE lifecycle = 'active'"
        )
    }
    return store, registered


def _import_baseline(
    connection: sqlite3.Connection, judgment_id: str
) -> tuple[int, str, dict[str, Any]] | None:
    """Latest revision written by a Wiki import (pilot or incremental)."""
    rows = connection.execute(
        """
        SELECT revision, statement, provenance_json FROM judgment_revisions
        WHERE judgment_id = ? ORDER BY revision DESC
        """,
        (judgment_id,),
    ).fetchall()
    for revision, statement, provenance_json in rows:
        provenance = json.loads(provenance_json)
        if isinstance(provenance, dict) and "page_sha256" in provenance and "wiki_url" in provenance:
            return revision, statement, provenance
    return None


def build_plan(
    connection: sqlite3.Connection, sources: Iterable[WikiSource]
) -> dict[str, Any]:
    """Compute the incremental plan using SELECT statements only."""
    sources = tuple(sources)
    store, registered = _load_store(connection)
    entries, skipped_edges, blocked = read_wiki_entries(sources, registered)

    baselines = {identity: _import_baseline(connection, identity) for identity in sorted(store)}
    repositories = {source.repository for source in sources}
    absent_from_index = [
        identity
        for identity, baseline in baselines.items()
        if baseline is not None and baseline[2].get("repository") in repositories and identity not in entries
    ]

    unchanged: list[str] = []
    store_only_changed: list[str] = []
    both_changed: list[dict[str, Any]] = []
    new_ids: set[str] = set()
    revise: dict[str, int] = {}
    for identity, entry in sorted(entries.items()):
        current = store.get(identity)
        if current is None:
            # An index slug change would otherwise add a second copy of a stored judgment.
            renamed = [
                previous
                for previous in absent_from_index
                if baselines[previous][2].get("repository") == entry.repository
                and (
                    baselines[previous][2]["page_sha256"] == entry.page_sha256
                    or baselines[previous][1] == entry.statement
                )
            ]
            if renamed:
                blocked.append({
                    "judgment_id": identity,
                    "reason": "possible identity change of stored judgment: " + ", ".join(renamed),
                })
            else:
                new_ids.add(identity)
            continue
        baseline = baselines[identity]
        if baseline is None:
            both_changed.append({
                "judgment_id": identity, "reason": "store has no imported revision",
                "current_revision": current.current_revision, "imported_revision": None,
            })
            continue
        revision, statement, provenance = baseline
        wiki_changed = provenance["page_sha256"] != entry.page_sha256 or statement != entry.statement
        if not wiki_changed:
            (unchanged if revision == current.current_revision else store_only_changed).append(identity)
            continue
        if revision != current.current_revision:
            both_changed.append({
                "judgment_id": identity, "reason": "changed on both sides",
                "current_revision": current.current_revision, "imported_revision": revision,
            })
            continue
        if current.lifecycle != "active" or current.superseded_by is not None:
            blocked.append({"judgment_id": identity, "reason": "wiki changed but store judgment is inactive"})
            continue
        revise[identity] = current.current_revision

    def resolve(
        identity: str, edges: Iterable[tuple[str, str, str]], written: set[str]
    ) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]]]:
        """Split edges into those whose target exists in the store or is written earlier
        in the plan, and the unresolved ones as ``(target, line, reason)``."""
        resolved: list[tuple[str, str, str]] = []
        unresolved: list[tuple[str, str, str]] = []
        for kind, target, line in edges:
            if target == identity:
                reason = "self relation"
            elif target in store or target in written:
                resolved.append((kind, target, line))
                continue
            elif target in entries:
                reason = "relation target not written"
            else:
                reason = "unknown relation target"
            unresolved.append((target, line, reason))
        return resolved, unresolved

    def skip(identity: str, unresolved: Iterable[tuple[str, str, str]]) -> None:
        skipped_edges.extend(
            {"judgment_id": identity, "line": line, "reason": reason} for _, line, reason in unresolved
        )

    # New judgments are written predecessor-first along supersedes, the only edge kind whose target
    # must exist when its source is written; a supersedes cycle among them cannot be, so it is blocked.
    remaining = {
        identity: {
            target for kind, target, _ in entries[identity].edges
            if kind == "supersedes" and target in new_ids and target != identity
        }
        for identity in new_ids
    }
    # Among the ready ones, prefer a judgment whose other targets are already placed, so a link
    # is needed only where those edges form a cycle.
    targets_of = {
        identity: {target for _, target, _ in entries[identity].edges if target in new_ids and target != identity}
        for identity in new_ids
    }
    order: list[str] = []
    ready = sorted(identity for identity, targets in remaining.items() if not targets)
    while ready:
        placed = set(order)
        identity = next((item for item in ready if targets_of[item] <= placed), ready[0])
        ready.remove(identity)
        order.append(identity)
        del remaining[identity]
        for other, targets in remaining.items():
            if identity in targets:
                targets.discard(identity)
                if not targets:
                    ready.append(other)
        ready.sort()
    for identity in sorted(remaining):
        blocked.append({"judgment_id": identity, "reason": "supersedes cycle among new judgments"})

    # Simulate lifecycle in write order so each supersession is checked against the state it will meet.
    state = {
        identity: [item.lifecycle, item.superseded_by, item.current_revision]
        for identity, item in store.items()
    }
    written: set[str] = set()
    pending = set(order)
    operations: list[dict[str, Any]] = []
    # Other edges to a new judgment not yet written are deferred, then written by a ``link``
    # (an update carrying the full relation set) once every deferred target has been decided.
    deferred: dict[str, list[tuple[str, str, str]]] = {}
    full_relations: dict[str, list[dict[str, str]]] = {}

    def link(source: str, late_reason: str = "relation target not written") -> None:
        added: list[dict[str, str]] = []
        for kind, target, line in deferred.pop(source):
            if target in written:
                added.append({"target_id": target, "relation_type": kind})
            else:
                reason = late_reason if target in pending else "relation target not written"
                skipped_edges.append({"judgment_id": source, "line": line, "reason": reason})
        if added:
            full_relations[source].extend(added)
            operations.append({
                "action": "link", "judgment_id": source,
                "expected_revision": state[source][2], "relations": list(full_relations[source]),
            })
            state[source][2] += 1

    for identity in order:
        pending.discard(identity)
        now: list[tuple[str, str, str]] = []
        later: list[tuple[str, str, str]] = []
        for edge in entries[identity].edges:
            (later if edge[0] != "supersedes" and edge[1] in pending else now).append(edge)
        resolved, unresolved = resolve(identity, now, written)
        predecessors: list[str] = []
        relations: list[dict[str, str]] = []
        for kind, target, line in resolved:
            if kind != "supersedes":
                relations.append({"target_id": target, "relation_type": kind})
                continue
            lifecycle, successor, _ = state[target]
            if lifecycle != "active" or successor is not None:
                detail = f"superseded by {successor}" if successor else lifecycle
                unresolved.append((target, line, f"supersedes target is not active ({detail})"))
                continue
            predecessors.append(target)
        if len(predecessors) > 1:
            blocked.append({
                "judgment_id": identity,
                "reason": "supersedes more than one active judgment; the store allows one predecessor per successor: "
                + ", ".join(predecessors),
            })
        else:
            skip(identity, unresolved)
            operation: dict[str, Any] = {"action": "add", "judgment_id": identity, "relations": relations}
            full_relations[identity] = list(relations)
            if predecessors:
                target = predecessors[0]
                if target in deferred:
                    # The predecessor is archived by this supersede and cannot be linked afterwards.
                    link(target, "relation target written after its source was superseded")
                revision = state[target][2]
                if target in revise:
                    del revise[target]
                    blocked.append({
                        "judgment_id": target,
                        "reason": f"wiki changed but superseded in this plan by {identity}",
                    })
                state[target] = ["archived", identity, revision]
                operation = {
                    "action": "supersede", "judgment_id": identity,
                    "predecessor": {"judgment_id": target, "expected_revision": revision},
                    "relations": relations,
                }
                full_relations[identity].append({"target_id": target, "relation_type": "supersedes"})
            operations.append(operation)
            state[identity] = ["active", None, 1]
            written.add(identity)
            if later:
                deferred[identity] = later
        for source in sorted(deferred):
            if not any(target in pending for _, target, _ in deferred[source]):
                link(source)
    for identity in sorted(revise):
        edges, unresolved = resolve(identity, entries[identity].edges, written)
        # update replaces the stored relation set; an edge it cannot write would drop a live relation.
        unwritable = sorted({target for target, _, reason in unresolved if reason == "relation target not written"})
        if unwritable:
            blocked.append({
                "judgment_id": identity,
                "reason": "relation target cannot be written in this plan, so a revise would drop its edge: "
                + ", ".join(unwritable),
            })
            continue
        if {target for kind, target, _ in edges if kind == "supersedes"} != store[identity].supersedes:
            blocked.append({"judgment_id": identity, "reason": "supersedes edges differ from store"})
            continue
        skip(identity, unresolved)
        operations.append({
            "action": "revise",
            "judgment_id": identity,
            "expected_revision": revise[identity],
            "relations": [{"target_id": target, "relation_type": kind} for kind, target, _ in edges],
        })
    for identity in order:
        if identity in written and entries[identity].lifecycle == "archived" and state[identity][0] == "active":
            operations.append({"action": "archive", "judgment_id": identity, "expected_revision": state[identity][2]})
            state[identity][0] = "archived"

    lifecycle_mismatch = [
        {"judgment_id": identity, "wiki_state": entry.source_state, "store_lifecycle": state[identity][0]}
        for identity, entry in sorted(entries.items())
        if identity in state and state[identity][0] != entry.lifecycle
    ]
    skipped_reasons = Counter(item["reason"] for item in skipped_edges)
    actions = Counter(operation["action"] for operation in operations)
    return {
        "format": PLAN_FORMAT,
        "sources": {source.repository: source.commit for source in sources},
        "registered_relation_types": sorted(registered),
        "counts": {
            "wiki_entries": len(entries),
            "add": actions["add"],
            "supersede": actions["supersede"],
            "revise": actions["revise"],
            "link": actions["link"],
            "archive": actions["archive"],
            "unchanged": len(unchanged),
            "store_only_changed": len(store_only_changed),
            "changed_on_both_sides": len(both_changed),
            "blocked": len(blocked),
            "skipped_edges": len(skipped_edges),
            "lifecycle_mismatch": len(lifecycle_mismatch),
            "absent_from_index": len(absent_from_index),
        },
        "skipped_edge_reasons": dict(sorted(skipped_reasons.items())),
        "operations": operations,
        "changed_on_both_sides": both_changed,
        "store_only_changed": store_only_changed,
        "blocked": sorted(blocked, key=lambda item: item["judgment_id"]),
        "skipped_edges": skipped_edges,
        "lifecycle_mismatch": lifecycle_mismatch,
        "absent_from_index": absent_from_index,
        "_entries": entries,
    }


def public_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """The plan without page bodies, for printing."""
    return {key: value for key, value in plan.items() if not key.startswith("_")}


def apply_plan(graph: JudgmentGraph, plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Execute the planned operations through the judgment domain API.

    The caller owns the surrounding transaction; every call here nests in it.
    """
    entries: dict[str, WikiEntry] = plan["_entries"]
    results: list[dict[str, Any]] = []
    for operation in plan["operations"]:
        identity = operation["judgment_id"]
        action = operation["action"]
        if action == "archive":
            judgment = graph.archive(identity, expected_revision=operation["expected_revision"])
        else:
            entry = entries[identity]
            content = (entry.statement, entry.page, entry.provenance())
            if action == "add":
                judgment = graph.add(identity, *content, relations=operation["relations"])
            elif action == "supersede":
                predecessor = operation["predecessor"]
                judgment = graph.supersede(
                    predecessor["judgment_id"], identity, *content,
                    expected_revision=predecessor["expected_revision"],
                    relations=operation["relations"],
                )
            elif action in {"revise", "link"}:
                judgment = graph.update(
                    identity, *content,
                    expected_revision=operation["expected_revision"],
                    relations=operation["relations"],
                )
            else:
                raise DecisionWikiImportError(f"unknown planned action: {action}")
        results.append({
            "action": action, "judgment_id": identity,
            "revision": judgment["revision"], "lifecycle": judgment["lifecycle"],
        })
    return results
