"""Result-free protocol for the repository-native feedback evaluation.

The evaluator intentionally owns no benchmark result.  It derives its graph only
from same-directory Markdown links in the frozen v2 corpus and makes an output
file only after all preflight gates, including the identity-only prior-fixture
audit, have passed.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROTOCOL_VERSION = "repository-native-feedback-v2"
EDGE_TYPE = "mention"
_METADATA = re.compile(r"^- (?P<name>Corpus split|Path ordinal|Corpus node ID|Source URL): `?(?P<value>.+?)`?\s*$", re.MULTILINE)
_LINK = re.compile(r"\[[^\]]+\]\((?P<target>[^)]+)\)")
_IDENTITY_KEYS = frozenset(
    {
        "node_id",
        "node_ids",
        "document_path",
        "document_paths",
        "source_url",
        "source_urls",
        "credited_edge",
        "credited_edges",
        "edge",
        "edges",
    }
)


class ProtocolStop(RuntimeError):
    """A preflight condition failed; no result output may be created."""


@dataclass(frozen=True)
class CorpusDocument:
    split: str
    ordinal: int
    node_id: str
    path: Path
    source_url: str
    text: str


@dataclass(frozen=True)
class SavedRelationTrace:
    query: str
    target_node_id: str
    credited_path: tuple[tuple[str, str, str], ...]


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_json_utf8(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as error:
        raise ProtocolStop(f"UTF-8 decode failed: {path}") from error
    except json.JSONDecodeError as error:
        raise ProtocolStop(f"JSON decode failed: {path}") from error


def load_corpus(corpus_directory: Path) -> tuple[dict[str, CorpusDocument], tuple[tuple[str, str, str], ...], str]:
    if not corpus_directory.is_dir():
        raise ProtocolStop(f"corpus directory is missing: {corpus_directory}")

    documents: dict[str, CorpusDocument] = {}
    by_path: dict[Path, CorpusDocument] = {}
    source_bytes: list[bytes] = []
    for path in sorted(corpus_directory.glob("*.md")):
        if path.name == "README.md":
            continue
        raw = path.read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ProtocolStop(f"UTF-8 decode failed: {path}") from error
        fields = {match.group("name"): match.group("value") for match in _METADATA.finditer(text)}
        expected = {"Corpus split", "Path ordinal", "Corpus node ID", "Source URL"}
        if set(fields) != expected:
            raise ProtocolStop(f"corpus metadata is incomplete: {path}")
        try:
            document = CorpusDocument(
                split=fields["Corpus split"],
                ordinal=int(fields["Path ordinal"]),
                node_id=fields["Corpus node ID"],
                path=path,
                source_url=fields["Source URL"],
                text=text,
            )
        except ValueError as error:
            raise ProtocolStop(f"invalid path ordinal: {path}") from error
        if document.node_id in documents:
            raise ProtocolStop(f"duplicate corpus node: {document.node_id}")
        documents[document.node_id] = document
        by_path[path.resolve()] = document
        source_bytes.append(path.name.encode("utf-8") + b"\0" + raw)

    edges: list[tuple[str, str, str]] = []
    for document in documents.values():
        for match in _LINK.finditer(document.text):
            target_text = match.group("target")
            if "://" in target_text or "#" in target_text:
                continue
            target = (document.path.parent / target_text).resolve()
            if target.parent != document.path.parent.resolve() or target not in by_path:
                raise ProtocolStop(f"non-local corpus link: {document.path} -> {target_text}")
            edges.append((document.node_id, by_path[target].node_id, EDGE_TYPE))

    return documents, tuple(sorted(edges)), sha256_bytes(b"".join(source_bytes))


def assert_topology(documents: dict[str, CorpusDocument], edges: tuple[tuple[str, str, str], ...]) -> None:
    if len(documents) != 8 or len(edges) != 6 or len(set(edges)) != len(edges):
        raise ProtocolStop("corpus topology is not the frozen two-chain shape")
    for split in ("development", "holdout"):
        members = sorted(
            (document for document in documents.values() if document.split == split),
            key=lambda document: document.ordinal,
        )
        if [document.ordinal for document in members] != [0, 1, 2, 3]:
            raise ProtocolStop(f"invalid {split} ordinals")
        expected = tuple(
            (members[index].node_id, members[index + 1].node_id, EDGE_TYPE)
            for index in range(3)
        )
        member_ids = {document.node_id for document in members}
        actual = tuple(edge for edge in edges if edge[0] in member_ids)
        if actual != expected:
            raise ProtocolStop(f"{split} topology was not derived from local links")


def assert_split_disjoint(documents: dict[str, CorpusDocument], edges: tuple[tuple[str, str, str], ...]) -> None:
    development = [document for document in documents.values() if document.split == "development"]
    holdout = [document for document in documents.values() if document.split == "holdout"]
    for attribute in ("node_id", "source_url"):
        if {getattr(document, attribute) for document in development} & {
            getattr(document, attribute) for document in holdout
        }:
            raise ProtocolStop(f"split shares {attribute}")
    if {document.path.name for document in development} & {document.path.name for document in holdout}:
        raise ProtocolStop("split shares corpus document path")
    development_edges = {edge for edge in edges if edge[0] in {item.node_id for item in development}}
    holdout_edges = {edge for edge in edges if edge[0] in {item.node_id for item in holdout}}
    if development_edges & holdout_edges:
        raise ProtocolStop("split shares credited edge identity")


def _identity_values(value: Any, *, name: str | None = None) -> set[str]:
    if isinstance(value, dict):
        collected: set[str] = set()
        for key, item in value.items():
            collected.update(_identity_values(item, name=str(key)))
        return collected
    if isinstance(value, list):
        return set().union(*(_identity_values(item, name=name) for item in value)) if value else set()
    if name in _IDENTITY_KEYS and isinstance(value, (str, int, float)):
        return {str(value)}
    return set()


def _edge_identities(value: Any, *, name: str | None = None) -> set[tuple[str, str, str]]:
    if isinstance(value, dict):
        if name in {"credited_edge", "credited_edges", "edge", "edges"} and {
            "source_id",
            "target_id",
            "edge_type",
        } <= set(value):
            return {(str(value["source_id"]), str(value["target_id"]), str(value["edge_type"]))}
        collected: set[tuple[str, str, str]] = set()
        for key, item in value.items():
            collected.update(_edge_identities(item, name=str(key)))
        return collected
    if isinstance(value, list):
        return set().union(*(_edge_identities(item, name=name) for item in value)) if value else set()
    return set()


def audit_prior_identities(
    fixtures_directory: Path,
    documents: dict[str, CorpusDocument],
    edges: tuple[tuple[str, str, str], ...],
) -> dict[str, int]:
    """Read only identity-bearing fields from earlier fixture JSON files."""
    current_values = {
        document.node_id for document in documents.values()
    } | {document.path.name for document in documents.values()} | {
        document.source_url for document in documents.values()
    }
    current_edges = set(edges)
    scanned = 0
    for path in sorted(fixtures_directory.glob("*.json")):
        if path.name.startswith("repository_native_feedback_v2"):
            continue
        raw = load_json_utf8(path)
        scanned += 1
        overlap = _identity_values(raw) & current_values
        edge_overlap = _edge_identities(raw) & current_edges
        if overlap or edge_overlap:
            raise ProtocolStop(f"prior identity overlap detected in {path.name}")
    return {"scanned_fixture_count": scanned, "overlap_count": 0}


def lexical_score(query: str, text: str) -> float:
    terms = set(re.findall(r"[a-z0-9]+", query.casefold()))
    return float(sum(text.casefold().count(term) for term in terms))


def relation_ranks(
    documents: dict[str, CorpusDocument],
    edges: tuple[tuple[str, str, str], ...],
    weights: dict[tuple[str, str, str], float],
    query: str,
    split: str,
) -> tuple[tuple[str, ...], dict[str, tuple[tuple[str, str, str], ...]]]:
    members = [document for document in documents.values() if document.split == split]
    scores = {document.node_id: lexical_score(query, document.text) for document in members}
    paths: dict[str, tuple[tuple[str, str, str], ...]] = {document.node_id: () for document in members}
    for edge in edges:
        source, target, _ = edge
        if source not in scores or target not in scores:
            continue
        candidate = (scores[source] + 1.0) * weights[edge] * 0.6
        if candidate > scores[target]:
            scores[target] = candidate
            paths[target] = paths[source] + (edge,)
    ranked = tuple(sorted(scores, key=lambda node_id: (-scores[node_id], node_id)))
    return ranked, paths


def feedback_event(
    documents: dict[str, CorpusDocument],
    edges: tuple[tuple[str, str, str], ...],
    split: str,
    feedback_query: str,
    target_node_id: str,
) -> SavedRelationTrace:
    weights = {edge: 1.0 for edge in edges}
    _, paths = relation_ranks(documents, edges, weights, feedback_query, split)
    path = paths[target_node_id]
    if not path:
        raise ProtocolStop("feedback target has no saved credited relation path")
    return SavedRelationTrace(feedback_query, target_node_id, path)


def apply_feedback(
    weights: dict[tuple[str, str, str], float],
    trace: SavedRelationTrace,
    *,
    mutate: bool,
) -> tuple[dict[tuple[str, str, str], float], tuple[tuple[str, str, str], ...]]:
    updated = dict(weights)
    projected = tuple((source, target, edge_type) for source, target, edge_type in trace.credited_path)
    if not mutate:
        return updated, ()
    if not projected or any(edge not in updated for edge in projected):
        raise ProtocolStop("credited path cannot be projected to stored edges")
    for edge in projected:
        updated[edge] = min(2.0, updated[edge] + 0.5)
    return updated, projected


def reciprocal_rank(ranking: tuple[str, ...], target_node_id: str) -> float:
    return 1.0 / (ranking.index(target_node_id) + 1)


def evaluate_stage(
    stage: str,
    specification: dict[str, Any],
    documents: dict[str, CorpusDocument],
    edges: tuple[tuple[str, str, str], ...],
) -> dict[str, Any]:
    split = str(specification["split"])
    trace = feedback_event(
        documents,
        edges,
        split,
        str(specification["feedback_query"]),
        str(specification["feedback_target_node_id"]),
    )
    initial = {edge: 1.0 for edge in edges}
    control_weights, control_mutations = apply_feedback(initial, trace, mutate=False)
    treatment_weights, treatment_mutations = apply_feedback(initial, trace, mutate=True)
    if control_weights != initial or control_mutations:
        raise ProtocolStop("control feedback mutated a relation edge")
    if set(treatment_mutations) != set(trace.credited_path) or len(treatment_mutations) != len(trace.credited_path):
        raise ProtocolStop("treatment mutated an uncredited relation edge")

    target = str(specification["score_target_node_id"])
    control_rank, _ = relation_ranks(documents, edges, control_weights, str(specification["score_query"]), split)
    treatment_rank, _ = relation_ranks(documents, edges, treatment_weights, str(specification["score_query"]), split)
    baseline_mrr = reciprocal_rank(control_rank, target)
    treatment_mrr = reciprocal_rank(treatment_rank, target)
    if baseline_mrr < 1.0 and not treatment_mrr > baseline_mrr:
        raise ProtocolStop("strict relation-MRR improvement gate failed")
    if baseline_mrr == 1.0 and treatment_mrr < baseline_mrr:
        raise ProtocolStop("baseline-perfect non-regression gate failed")
    return {
        "stage": stage,
        "split": split,
        "control_and_treatment_feedback_query_equal": True,
        "control_mutated_edges": list(control_mutations),
        "treatment_mutated_edges": list(treatment_mutations),
        "credited_path": list(trace.credited_path),
        "score_query": specification["score_query"],
        "score_target_node_id": target,
        "baseline_relation_mrr": baseline_mrr,
        "treatment_relation_mrr": treatment_mrr,
        "improvement_claim_allowed": baseline_mrr < 1.0,
    }


def build_stage_result(
    stage: str,
    repository_root: Path,
    *,
    allow_holdout: bool,
) -> dict[str, Any]:
    fixtures = repository_root / "tests" / "fixtures"
    manifest = load_json_utf8(fixtures / "repository_native_feedback_v2.manifest.json")
    gold = load_json_utf8(fixtures / "repository_native_feedback_v2.gold.json")
    corpus = repository_root / str(manifest["corpus_directory"])
    documents, edges, source_hash = load_corpus(corpus)
    if source_hash != manifest["source_sha256"]:
        raise ProtocolStop("fixed source hash mismatch")
    assert_topology(documents, edges)
    assert_split_disjoint(documents, edges)
    audit = audit_prior_identities(fixtures, documents, edges)
    if stage == "holdout" and not allow_holdout:
        raise ProtocolStop("holdout is unavailable until development gates pass")
    if stage not in gold["stages"]:
        raise ProtocolStop(f"unknown stage: {stage}")
    evaluation = evaluate_stage(stage, gold["stages"][stage], documents, edges)
    replay = evaluate_stage(stage, gold["stages"][stage], documents, edges)
    if canonical_json(evaluation) != canonical_json(replay):
        raise ProtocolStop("deterministic replay gate failed")
    result = {
        "protocol_version": PROTOCOL_VERSION,
        "stage": stage,
        "source_sha256": source_hash,
        "audit": audit,
        "evaluation": evaluation,
    }
    result["output_sha256"] = sha256_bytes(canonical_json(result))
    return result


def assert_valid_development_result(path: Path) -> None:
    development = load_json_utf8(path)
    recorded_hash = development.pop("output_sha256", None)
    if development.get("protocol_version") != PROTOCOL_VERSION:
        raise ProtocolStop("development result has another protocol version")
    if development.get("stage") != "development":
        raise ProtocolStop("development result has another stage")
    if recorded_hash != sha256_bytes(canonical_json(development)):
        raise ProtocolStop("development result output hash mismatch")
    evaluation = development.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ProtocolStop("development result has no evaluation")
    baseline = evaluation.get("baseline_relation_mrr")
    treatment = evaluation.get("treatment_relation_mrr")
    if not isinstance(baseline, (int, float)) or not isinstance(treatment, (int, float)):
        raise ProtocolStop("development result has invalid relation MRR")
    if baseline < 1.0 and not treatment > baseline:
        raise ProtocolStop("development result did not pass the improvement gate")
    if baseline == 1.0 and treatment < baseline:
        raise ProtocolStop("development result did not pass the non-regression gate")


def write_exclusive(path: Path, result: dict[str, Any]) -> None:
    if path.exists():
        raise ProtocolStop(f"exclusive output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_json(result) + b"\n"
    try:
        with path.open("xb") as handle:
            handle.write(encoded)
    except FileExistsError as error:
        raise ProtocolStop(f"exclusive output already exists: {path}") from error
