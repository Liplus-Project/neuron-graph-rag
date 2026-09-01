"""Result-free source-grounded relation-seed retrieval protocol."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import Any

from .engine import NeuronGraphRAG
from .github_source import GitHubSnapshot, index_github_snapshot

ROOT = Path(__file__).resolve().parents[2]
STEM = "github_source_grounded_relation_v1"
PROTOCOL_ID = "github-ngr-source-grounded-relation-seed-v1"
STAGES = ("development", "holdout")
ARMS = ("original-full-query-ngr-default", "source-grounded-relation-seed")
RUNS = ("primary", "replay")
COHORTS = (
    "direct_lexical",
    "semantic_paraphrase",
    "relation_linked",
    "negative_control",
)
GATE_IDS = (
    "protocol-validity",
    "deterministic-replay",
    "relation-path-completeness-strict-improvement",
    "relation-mrr-strict-improvement",
    "relation-hit-at-5-strict-improvement",
    "direct-per-case-non-regression",
    "semantic-per-case-non-regression",
    "direct-cohort-non-regression",
    "semantic-cohort-non-regression",
    "negative-forbidden-count-non-regression",
)
MANIFEST_PATH = ROOT / "tests" / "fixtures" / f"{STEM}.manifest.json"
_MARKDOWN_LINK = re.compile(r"(?<!!)\[[^\]]+\]\(([^)\s]+)(?:\s+['\"][^'\"]*['\"])?\)")
_REFERENCE_MARKERS = (
    "related",
    "relation",
    "linked",
    "links to",
    "referenced by",
    "参照",
    "関連",
    "リンク",
    "接続",
)


def _encoded(payload: Any) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _exclusive_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_encoded(payload))
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _read_json(path: Path, *, require_canonical: bool = True) -> dict[str, Any]:
    raw = path.read_bytes()
    text = raw.decode("utf-8", errors="strict")
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise TypeError(f"JSON object required: {path}")
    if require_canonical and raw != _encoded(payload):
        raise ValueError(f"non-canonical JSON: {path}")
    return payload


def _git_bytes(root: Path, object_name: str) -> bytes:
    result = subprocess.run(
        ["git", "show", object_name],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise ValueError(f"git object is unavailable: {object_name}")
    return result.stdout


def _mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    item = value.get(key)
    if not isinstance(item, Mapping):
        raise TypeError(f"object required: {key}")
    return item


def _rows(value: Mapping[str, Any], key: str) -> list[Mapping[str, Any]]:
    item = value.get(key)
    if not isinstance(item, list) or any(not isinstance(row, Mapping) for row in item):
        raise ValueError(f"object array required: {key}")
    return list(item)


def _string(value: Mapping[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.strip():
        raise ValueError(f"non-empty string required: {key}")
    return item


def _normalize_query(value: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9\u3040-\u30ff\u3400-\u9fff]+", value.casefold()))


def _query_similarity(left: str, right: str) -> float:
    a, b = set(_normalize_query(left)), set(_normalize_query(right))
    return len(a & b) / len(a | b) if a or b else 1.0


def _resolve_markdown_target(source_path: str, raw_target: str) -> str | None:
    target = raw_target.split("#", 1)[0].split("?", 1)[0]
    if not target or "://" in target or target.startswith(("mailto:", "/")):
        return None
    source = PurePosixPath(source_path)
    resolved = source.parent.joinpath(target)
    parts: list[str] = []
    for part in resolved.parts:
        if part in ("", "."):
            continue
        if part == "..":
            if not parts:
                return None
            parts.pop()
        else:
            parts.append(part)
    return "/".join(parts)


def extract_source_grounded_relations(
    snapshot: GitHubSnapshot,
) -> tuple[dict[str, str], ...]:
    documents = {document.path: document for document in snapshot.documents}
    relations: set[tuple[str, str, str, str, str]] = set()
    for document in snapshot.documents:
        for match in _MARKDOWN_LINK.finditer(document.content):
            target = _resolve_markdown_target(document.path, match.group(1))
            if target not in documents:
                continue
            target_document = documents[target]
            relations.add(
                (
                    document.path,
                    target,
                    "markdown_link",
                    document.content_sha256,
                    target_document.content_sha256,
                )
            )
    return tuple(
        {
            "source_path": source,
            "target_path": target,
            "edge_type": edge_type,
            "acquisition_method": "markdown-relative-link-regex-v1",
            "source_content_sha256": source_hash,
            "target_content_sha256": target_hash,
        }
        for source, target, edge_type, source_hash, target_hash in sorted(relations)
    )


def _load_protocol_artifacts(root: Path, *, include_finalizer: bool) -> dict[str, Any]:
    manifest_path = root / MANIFEST_PATH.relative_to(ROOT)
    manifest = _read_json(manifest_path)
    artifacts = {
        name: root / str(path)
        for name, path in _mapping(manifest, "protocol_artifacts").items()
    }
    protocol = {
        "root": root,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "corpus": GitHubSnapshot.read(artifacts["corpus"]),
        "queries": _read_json(artifacts["queries"]),
    }
    if include_finalizer:
        protocol.update(
            {
                "gold": _read_json(artifacts["gold"]),
                "gate": _read_json(artifacts["gate"]),
                "audit": _read_json(artifacts["audit"]),
            }
        )
    return protocol


def load_worker_protocol(root: Path = ROOT) -> dict[str, Any]:
    """Load only source and queries; gold is unreachable on the worker surface."""
    protocol = _load_protocol_artifacts(root, include_finalizer=False)
    validate_worker_protocol(protocol)
    return protocol


def load_protocol(
    root: Path = ROOT, *, require_result_free: bool = True
) -> dict[str, Any]:
    """Load finalizer fixtures after workers have completed."""
    protocol = _load_protocol_artifacts(root, include_finalizer=True)
    validate_protocol(protocol, require_result_free=require_result_free)
    return protocol


def _validate_source_contract(protocol: Mapping[str, Any]) -> tuple[str, ...]:
    manifest = _mapping(protocol, "manifest")
    corpus = protocol.get("corpus")
    queries = _mapping(protocol, "queries")
    if not isinstance(corpus, GitHubSnapshot):
        raise TypeError("corpus must be a GitHub snapshot")
    for artifact in (manifest, queries):
        if artifact.get("protocol_id") != PROTOCOL_ID:
            raise ValueError("protocol_id mismatch")
    if manifest.get("issue") != 200 or manifest.get("phase") != "result-free-freeze":
        raise ValueError("manifest issue or phase mismatch")
    source = _mapping(manifest, "source")
    if (
        corpus.repository != source.get("repository")
        or corpus.commit != source.get("commit")
        or source.get("access") != "git-show-read-only-fixed-commit"
        or source.get("generated_by")
        != "tools/acquire_source_grounded_relation_corpus.py"
    ):
        raise ValueError("source provenance mismatch")
    paths = [document.path for document in corpus.documents]
    if paths != sorted(source.get("paths", [])):
        raise ValueError("source path registry mismatch")
    for document in corpus.documents:
        if document.content_sha256 != source["content_sha256"].get(document.path):
            raise ValueError(f"source content hash mismatch: {document.path}")
    extracted = extract_source_grounded_relations(corpus)
    if list(extracted) != manifest.get("relationships"):
        raise ValueError("source-grounded relationship registry mismatch")
    if any(
        row["acquisition_method"] != "markdown-relative-link-regex-v1"
        for row in extracted
    ):
        raise ValueError("unsupported relationship acquisition method")
    if tuple(manifest.get("arms", ())) != ARMS:
        raise ValueError("retrieval arm identity mismatch")
    return tuple(paths)


def _stage_case_rows(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for stage_value in _mapping(payload, "stages").values():
        raw = (
            stage_value.get("cases")
            if isinstance(stage_value, Mapping)
            else stage_value
        )
        if not isinstance(raw, list) or any(
            not isinstance(row, Mapping) for row in raw
        ):
            raise ValueError("predecessor stage cases must be an object array")
        rows.extend(raw)
    return rows


def _validate_query_cases(
    protocol: Mapping[str, Any],
) -> dict[str, list[Mapping[str, Any]]]:
    manifest = _mapping(protocol, "manifest")
    queries = _mapping(protocol, "queries")
    query_stages = _mapping(queries, "stages")
    stage_cases: dict[str, list[Mapping[str, Any]]] = {}
    all_case_ids: set[str] = set()
    current_queries: list[str] = []
    for stage in STAGES:
        cases = _rows(query_stages, stage)
        if len(cases) != 8:
            raise ValueError("each stage must freeze exactly eight queries")
        cohorts: list[str] = []
        for case in cases:
            case_id = _string(case, "case_id")
            cohort = _string(case, "cohort")
            query = _string(case, "query")
            if case_id in all_case_ids:
                raise ValueError("query case ids must be globally unique")
            all_case_ids.add(case_id)
            current_queries.append(query)
            cohorts.append(cohort)
        expected_cohorts = tuple(cohort for cohort in COHORTS for _ in range(2))
        if tuple(cohorts) != expected_cohorts:
            raise ValueError("cohort order or cardinality mismatch")
        stage_cases[stage] = cases
    prior_queries: list[str] = []
    for relative in manifest.get("predecessor_query_gold", []):
        if not isinstance(relative, Mapping):
            raise TypeError("predecessor query/gold pair must be an object")
        item = relative
        query_payload = _read_json(
            Path(protocol["root"]) / _string(item, "queries"),
            require_canonical=False,
        )
        prior_queries.extend(
            _string(row, "query") for row in _stage_case_rows(query_payload)
        )
    limit = float(manifest.get("maximum_predecessor_query_similarity", 0.72))
    for query in current_queries:
        if (
            max(
                (_query_similarity(query, prior) for prior in prior_queries),
                default=0.0,
            )
            >= limit
        ):
            raise ValueError("query is too similar to a predecessor")
    return stage_cases


def _validate_artifact_hashes(protocol: Mapping[str, Any]) -> None:
    root = Path(protocol["root"])
    manifest = _mapping(protocol, "manifest")
    for relative, expected in _mapping(manifest, "artifact_sha256").items():
        if _sha256(root / str(relative)) != expected:
            raise ValueError(f"frozen artifact hash mismatch: {relative}")


def _raw_packet_paths(
    protocol: Mapping[str, Any], stage: str | None = None
) -> dict[tuple[str, str, str], Path]:
    root = Path(protocol["root"])
    raw_packets = _mapping(_mapping(protocol, "manifest"), "raw_packets")
    if set(raw_packets) != set(STAGES):
        raise ValueError("raw packet stages must match the frozen stages")
    paths: dict[tuple[str, str, str], Path] = {}
    for registered_stage in STAGES:
        stage_packets = _mapping(raw_packets, registered_stage)
        if set(stage_packets) != set(ARMS):
            raise ValueError("raw packet arms must match the frozen arms")
        for arm in ARMS:
            arm_packets = _mapping(stage_packets, arm)
            if set(arm_packets) != set(RUNS):
                raise ValueError("raw packet runs must be primary and replay")
            for run in RUNS:
                relative = _string(arm_packets, run)
                relative_path = Path(relative)
                if relative_path.is_absolute() or ".." in relative_path.parts:
                    raise ValueError("raw packet path must stay inside the repository")
                paths[(registered_stage, arm, run)] = root / relative_path
    if len({path.resolve() for path in paths.values()}) != len(paths):
        raise ValueError("raw packet paths must be unique")
    if stage is not None:
        if stage not in STAGES:
            raise ValueError("unknown stage")
        return {key: path for key, path in paths.items() if key[0] == stage}
    return paths


def validate_worker_protocol(protocol: Mapping[str, Any]) -> None:
    _validate_source_contract(protocol)
    _validate_query_cases(protocol)


def validate_protocol(
    protocol: Mapping[str, Any], *, require_result_free: bool = True
) -> None:
    root = Path(protocol["root"])
    manifest = _mapping(protocol, "manifest")
    gold = _mapping(protocol, "gold")
    gate = _mapping(protocol, "gate")
    audit = _mapping(protocol, "audit")
    paths = _validate_source_contract(protocol)
    for artifact in (gold, gate, audit):
        if artifact.get("protocol_id") != PROTOCOL_ID:
            raise ValueError("protocol_id mismatch")
    _validate_cases(protocol, paths, _mapping(protocol, "manifest")["relationships"])
    configured_gates = tuple(row.get("gate_id") for row in _rows(gate, "gates"))
    if configured_gates != GATE_IDS or not all(
        row.get("hard") is True for row in _rows(gate, "gates")
    ):
        raise ValueError("gate order or hardness mismatch")
    expected_audit = {
        "phase": "result-free-freeze",
        "development_run_count": 0,
        "holdout_run_count": 0,
        "retry_count": 0,
        "observed_result_count": 0,
        "shared_database_open_count": 0,
        "worker_gold_open_count": 0,
        "predecessor_copy_paraphrase_near_transform_count": 0,
        "fresh_query_gold_split_frozen_before_observation": True,
        "performance": "not assessed",
    }
    if any(audit.get(key) != value for key, value in expected_audit.items()):
        raise ValueError("result-free audit mismatch")
    if require_result_free:
        for registry in ("claims", "outputs"):
            for artifact in _mapping(manifest, registry).values():
                if (root / str(artifact)).exists():
                    raise ValueError(
                        f"registered {registry[:-1]} must be absent at freeze"
                    )
        for artifact in _raw_packet_paths(protocol).values():
            if artifact.exists():
                raise ValueError("registered raw packet must be absent at freeze")
    _validate_artifact_hashes(protocol)


def _validate_cases(
    protocol: Mapping[str, Any],
    corpus_paths: Sequence[str],
    relations: Sequence[Mapping[str, str]],
) -> None:
    manifest = _mapping(protocol, "manifest")
    gold = _mapping(protocol, "gold")
    gold_stages = _mapping(gold, "stages")
    relation_keys = {
        (row["source_path"], row["target_path"], row["edge_type"]) for row in relations
    }
    all_case_ids: set[str] = set()
    stage_gold: dict[str, set[str]] = {}
    current_gold_signatures: set[tuple[Any, ...]] = set()
    query_cases = _validate_query_cases(protocol)
    for stage in STAGES:
        cases = query_cases[stage]
        rows = _rows(gold_stages, stage)
        if len(cases) != 8 or len(rows) != 8:
            raise ValueError("each stage must freeze exactly eight cases")
        by_id = {_string(row, "case_id"): row for row in rows}
        if len(by_id) != 8:
            raise ValueError("gold case ids must be unique")
        cohorts: list[str] = []
        identities: set[str] = set()
        for case in cases:
            case_id = _string(case, "case_id")
            cohort = _string(case, "cohort")
            _string(case, "query")
            if case_id in all_case_ids or case_id not in by_id:
                raise ValueError("query and gold ids must align and be globally unique")
            all_case_ids.add(case_id)
            cohorts.append(cohort)
            row = by_id[case_id]
            if row.get("cohort") != cohort:
                raise ValueError("query and gold cohort mismatch")
            expected = row.get("expected_path")
            forbidden = row.get("forbidden_path")
            if expected is not None and expected not in corpus_paths:
                raise ValueError("expected path outside corpus")
            if forbidden is not None and forbidden not in corpus_paths:
                raise ValueError("forbidden path outside corpus")
            identities.update(
                value for value in (expected, forbidden) if isinstance(value, str)
            )
            seed = row.get("relation_seed_path")
            edge_type = row.get("relation_edge_type")
            if cohort == "relation_linked":
                if not isinstance(seed, str) or not isinstance(edge_type, str):
                    raise ValueError("relation gold requires seed and edge type")
                if (seed, expected, edge_type) not in relation_keys:
                    raise ValueError("relation gold lacks source-grounded edge")
                identities.add(seed)
            signature = (cohort, expected, forbidden, seed, edge_type)
            if signature in current_gold_signatures:
                raise ValueError("gold signatures must be fresh")
            current_gold_signatures.add(signature)
        expected_cohorts = tuple(cohort for cohort in COHORTS for _ in range(2))
        if tuple(cohorts) != expected_cohorts:
            raise ValueError("cohort order or cardinality mismatch")
        stage_gold[stage] = identities
    if not stage_gold["development"].isdisjoint(stage_gold["holdout"]):
        raise ValueError("development and holdout gold identities overlap")
    prior_signatures: set[tuple[Any, ...]] = set()
    for relative in manifest.get("predecessor_query_gold", []):
        if not isinstance(relative, Mapping):
            raise TypeError("predecessor query/gold pair must be an object")
        item = relative
        gold_payload = _read_json(
            Path(protocol["root"]) / _string(item, "gold"),
            require_canonical=False,
        )
        for row in _stage_case_rows(gold_payload):
            prior_signatures.add(
                (
                    row.get("cohort"),
                    row.get("expected_path", row.get("expected_source_id")),
                    row.get("forbidden_path"),
                    row.get("relation_seed_path", row.get("relation_seed_source_id")),
                    row.get("relation_edge_type"),
                )
            )
    if current_gold_signatures & prior_signatures:
        raise ValueError("gold signature reuses a predecessor case")


def _has_relation_intent(query: str) -> bool:
    normalized = query.casefold()
    return any(marker in normalized for marker in _REFERENCE_MARKERS)


def _aliases(document: Any) -> tuple[str, ...]:
    title = ""
    for line in document.content.splitlines():
        if line.startswith("# "):
            title = line[2:].strip()
            break
    path = PurePosixPath(document.path)
    values = {document.path, path.name, path.stem.replace("-", " ")}
    if title:
        values.add(title)
    return tuple(
        sorted(
            (value.casefold() for value in values),
            key=lambda value: (-len(value), value),
        )
    )


def _referenced_seeds(query: str, snapshot: GitHubSnapshot) -> tuple[str, ...]:
    if not _has_relation_intent(query):
        return ()
    normalized = query.casefold()
    return tuple(
        document.path
        for document in snapshot.documents
        if any(len(alias) >= 8 and alias in normalized for alias in _aliases(document))
    )


def _hit_payload(hit: Any, snapshot: GitHubSnapshot) -> dict[str, Any]:
    path = str(hit.node.metadata["path"])
    relation_paths = []
    for raw_path in hit.explain()["paths"]:
        steps = raw_path.get("steps", [])
        if not steps:
            continue
        relation_paths.append(
            {
                "seed_path": str(raw_path["seed_id"]).split(":", 2)[-1],
                "target_path": path,
                "steps": [
                    {
                        "source_path": str(step["source_id"]).split(":", 2)[-1],
                        "target_path": str(step["target_id"]).split(":", 2)[-1],
                        "edge_type": step["edge_type"],
                    }
                    for step in steps
                ],
            }
        )
    return {
        "path": path,
        "source_url": hit.node.metadata["source_url"],
        "content_sha256": hit.node.metadata["content_sha256"],
        "sparse_score": hit.sparse_score,
        "dense_score": hit.dense_score,
        "entry_score": hit.entry_score,
        "graph_activation": hit.graph_activation,
        "final_score": hit.final_score,
        "relation_paths": relation_paths,
    }


def run_worker(
    protocol: Mapping[str, Any],
    stage: str,
    arm: str,
    database: Path,
    *,
    protocol_commit: str,
    run: str,
) -> dict[str, Any]:
    """Run a gold-blind arm against one fresh SQLite database."""
    if "gold" in protocol:
        raise ValueError("worker protocol must not expose gold")
    if stage not in STAGES or arm not in ARMS or run not in RUNS:
        raise ValueError("unknown stage, arm, or run")
    if not re.fullmatch(r"[0-9a-f]{40}", protocol_commit):
        raise ValueError("protocol commit must be lowercase full SHA")
    if database.exists():
        raise FileExistsError("worker database must be fresh")
    snapshot = protocol["corpus"]
    if not isinstance(snapshot, GitHubSnapshot):
        raise TypeError("invalid corpus")
    cases = _rows(_mapping(_mapping(protocol, "queries"), "stages"), stage)
    relations = list(_mapping(protocol, "manifest")["relationships"])
    outgoing: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for relation in relations:
        outgoing[relation["source_path"]].append(relation)
    with NeuronGraphRAG(database) as engine:
        index_github_snapshot(engine, snapshot)
        for relation in relations:
            engine.add_edge(
                f"github:{snapshot.repository}:{relation['source_path']}",
                f"github:{snapshot.repository}:{relation['target_path']}",
                relation["edge_type"],
            )
        observed = []
        for case in cases:
            search_limit = len(snapshot.documents) if arm == ARMS[1] else 5
            trace = engine.search(
                _string(case, "query"), limit=search_limit, now=2_000.0
            )
            all_hits = [_hit_payload(hit, snapshot) for hit in trace.hits]
            hits = all_hits[:5]
            anchors: tuple[str, ...] = ()
            if arm == "source-grounded-relation-seed":
                anchors = _referenced_seeds(_string(case, "query"), snapshot)
                promoted_by_path: dict[str, dict[str, Any]] = {}
                for seed in anchors:
                    for relation in outgoing.get(seed, []):
                        target = relation["target_path"]
                        base = next(
                            (hit for hit in all_hits if hit["path"] == target), None
                        )
                        if base is None:
                            document = next(
                                doc for doc in snapshot.documents if doc.path == target
                            )
                            base = {
                                "path": target,
                                "source_url": document.source_url,
                                "content_sha256": document.content_sha256,
                                "sparse_score": 0.0,
                                "dense_score": 0.0,
                                "entry_score": 0.0,
                                "graph_activation": 0.0,
                                "final_score": 0.0,
                                "relation_paths": [],
                            }
                        promoted_hit = promoted_by_path.setdefault(target, dict(base))
                        relation_path = {
                            "seed_path": seed,
                            "target_path": target,
                            "steps": [
                                {
                                    "source_path": seed,
                                    "target_path": target,
                                    "edge_type": relation["edge_type"],
                                }
                            ],
                        }
                        paths = list(promoted_hit["relation_paths"])
                        if relation_path not in paths:
                            paths.append(relation_path)
                        promoted_hit["relation_paths"] = paths
                full_query_rank = {
                    hit["path"]: index for index, hit in enumerate(all_hits)
                }
                promoted = sorted(
                    promoted_by_path.values(),
                    key=lambda hit: (
                        full_query_rank.get(hit["path"], len(all_hits)),
                        hit["path"],
                    ),
                )
                promoted_paths = set(promoted_by_path)
                hits = (
                    promoted
                    + [hit for hit in hits if hit["path"] not in promoted_paths]
                )[:5]
            observed.append(
                {
                    "case_id": _string(case, "case_id"),
                    "query": _string(case, "query"),
                    "referenced_seed_paths": list(anchors),
                    "hits": hits,
                }
            )
    return {
        "protocol_id": PROTOCOL_ID,
        "protocol_commit": protocol_commit,
        "stage": stage,
        "arm": arm,
        "run": run,
        "attempt": 1,
        "retry_count": 0,
        "cases": observed,
    }


def _validate_worker_packet(
    packet: Mapping[str, Any],
    *,
    stage: str,
    arm: str,
    run: str,
    protocol_commit: str,
) -> None:
    expected_keys = {
        "protocol_id",
        "protocol_commit",
        "stage",
        "arm",
        "run",
        "attempt",
        "retry_count",
        "cases",
    }
    if set(packet) != expected_keys:
        raise ValueError("raw worker packet fields do not match the frozen schema")
    expected_identity = {
        "protocol_id": PROTOCOL_ID,
        "protocol_commit": protocol_commit,
        "stage": stage,
        "arm": arm,
        "run": run,
        "attempt": 1,
        "retry_count": 0,
    }
    if any(packet.get(key) != value for key, value in expected_identity.items()):
        raise ValueError("raw worker packet identity mismatch")
    _rows(packet, "cases")


def _persist_worker_packet(
    protocol: Mapping[str, Any],
    packet: Mapping[str, Any],
    *,
    stage: str,
    arm: str,
    run: str,
    protocol_commit: str,
) -> Path:
    _validate_worker_packet(
        packet,
        stage=stage,
        arm=arm,
        run=run,
        protocol_commit=protocol_commit,
    )
    path = _raw_packet_paths(protocol, stage)[(stage, arm, run)]
    _exclusive_write(path, packet)
    return path


def _load_registered_worker_packets(
    protocol: Mapping[str, Any], stage: str, protocol_commit: str
) -> dict[str, dict[str, Mapping[str, Any]]]:
    paths = _raw_packet_paths(protocol, stage)
    workers: dict[str, dict[str, Mapping[str, Any]]] = {
        run: {} for run in RUNS
    }
    for run in RUNS:
        for arm in ARMS:
            packet = _read_json(paths[(stage, arm, run)])
            _validate_worker_packet(
                packet,
                stage=stage,
                arm=arm,
                run=run,
                protocol_commit=protocol_commit,
            )
            workers[run][arm] = packet
    return workers


def _rank(hits: Sequence[Mapping[str, Any]], expected: str | None) -> int:
    if expected is None:
        return 6
    return next(
        (index for index, hit in enumerate(hits, 1) if hit["path"] == expected), 6
    )


def _arm_metrics(
    worker: Mapping[str, Any], gold_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    gold = {_string(row, "case_id"): row for row in gold_rows}
    cases = []
    for observed in _rows(worker, "cases"):
        row = gold[_string(observed, "case_id")]
        hits = _rows(observed, "hits")
        rank = _rank(hits, row.get("expected_path"))
        forbidden = row.get("forbidden_path")
        forbidden_count = (
            sum(hit["path"] == forbidden for hit in hits) if forbidden else 0
        )
        expected_path = row.get("expected_path")
        expected_relation = {
            "seed_path": row.get("relation_seed_path"),
            "target_path": expected_path,
            "steps": [
                {
                    "source_path": row.get("relation_seed_path"),
                    "target_path": expected_path,
                    "edge_type": row.get("relation_edge_type"),
                }
            ],
        }
        relation_complete = False
        if row["cohort"] == "relation_linked":
            target = next((hit for hit in hits if hit["path"] == expected_path), None)
            relation_complete = bool(
                target and expected_relation in target["relation_paths"]
            )
        cases.append(
            {
                "case_id": row["case_id"],
                "cohort": row["cohort"],
                "rank": rank,
                "hit_at_5": rank <= 5,
                "forbidden_count": forbidden_count,
                "relation_path_complete": relation_complete,
            }
        )
    cohorts = {}
    for cohort in COHORTS:
        rows = [case for case in cases if case["cohort"] == cohort]
        cohorts[cohort] = {
            "mrr": sum(1.0 / case["rank"] for case in rows) / len(rows),
            "hit_at_5": sum(case["hit_at_5"] for case in rows) / len(rows),
            "forbidden_count": sum(case["forbidden_count"] for case in rows),
            "relation_path_complete_count": sum(
                case["relation_path_complete"] for case in rows
            ),
        }
    return {"cases": cases, "cohorts": cohorts}


def finalize_stage(
    protocol: Mapping[str, Any],
    stage: str,
    protocol_commit: str,
    shared_database_sha256: str,
) -> dict[str, Any]:
    """Read the exact registered raw packet set, then open gold."""
    workers = _load_registered_worker_packets(protocol, stage, protocol_commit)
    primary_workers = workers["primary"]
    replay_workers = workers["replay"]
    gold_rows = _rows(_mapping(_mapping(protocol, "gold"), "stages"), stage)
    metrics = {arm: _arm_metrics(primary_workers[arm], gold_rows) for arm in ARMS}
    deterministic = all(
        primary_workers[arm]["cases"] == replay_workers[arm]["cases"] for arm in ARMS
    )
    baseline = metrics[ARMS[0]]
    candidate = metrics[ARMS[1]]
    base_cases = {row["case_id"]: row for row in baseline["cases"]}
    candidate_cases = {row["case_id"]: row for row in candidate["cases"]}
    relation_base = baseline["cohorts"]["relation_linked"]
    relation_candidate = candidate["cohorts"]["relation_linked"]
    checks = {
        "protocol-validity": True,
        "deterministic-replay": deterministic,
        "relation-path-completeness-strict-improvement": relation_candidate[
            "relation_path_complete_count"
        ]
        > relation_base["relation_path_complete_count"],
        "relation-mrr-strict-improvement": relation_candidate["mrr"]
        > relation_base["mrr"],
        "relation-hit-at-5-strict-improvement": relation_candidate["hit_at_5"]
        > relation_base["hit_at_5"],
        "direct-per-case-non-regression": all(
            candidate_cases[key]["rank"] <= row["rank"]
            for key, row in base_cases.items()
            if row["cohort"] == "direct_lexical"
        ),
        "semantic-per-case-non-regression": all(
            candidate_cases[key]["rank"] <= row["rank"]
            for key, row in base_cases.items()
            if row["cohort"] == "semantic_paraphrase"
        ),
        "direct-cohort-non-regression": candidate["cohorts"]["direct_lexical"]["mrr"]
        >= baseline["cohorts"]["direct_lexical"]["mrr"]
        and candidate["cohorts"]["direct_lexical"]["hit_at_5"]
        >= baseline["cohorts"]["direct_lexical"]["hit_at_5"],
        "semantic-cohort-non-regression": candidate["cohorts"]["semantic_paraphrase"][
            "mrr"
        ]
        >= baseline["cohorts"]["semantic_paraphrase"]["mrr"]
        and candidate["cohorts"]["semantic_paraphrase"]["hit_at_5"]
        >= baseline["cohorts"]["semantic_paraphrase"]["hit_at_5"],
        "negative-forbidden-count-non-regression": candidate["cohorts"][
            "negative_control"
        ]["forbidden_count"]
        <= baseline["cohorts"]["negative_control"]["forbidden_count"],
    }
    gates = [
        {"gate_id": gate_id, "hard": True, "passed": checks[gate_id]}
        for gate_id in GATE_IDS
    ]
    passed = all(checks.values())
    return {
        "protocol_id": PROTOCOL_ID,
        "protocol_commit": protocol_commit,
        "stage": stage,
        "status": "passed" if passed else "failed",
        "selected_arm": ARMS[1] if passed else ARMS[0],
        "selection_reason": "all-candidate-gates-passed"
        if passed
        else "candidate-gate-failed",
        "shared_database_sha256_before": shared_database_sha256,
        "shared_database_sha256_after": shared_database_sha256,
        "actual_stage_run_count": 1,
        "actual_worker_run_count": 4,
        "retry_count": 0,
        "protocol_validity": {
            "source_acquisition_and_identity": True,
            "development_holdout_identity_separation": True,
            "predecessor_leakage_rejected": True,
            "worker_gold_blind": True,
            "baseline_relation_failure_is_performance_result": True,
        },
        "workers": {
            "primary": {arm: primary_workers[arm] for arm in ARMS},
            "replay": {arm: replay_workers[arm] for arm in ARMS},
        },
        "arms": metrics,
        "gates": gates,
        "all_hard_gates_pass": passed,
    }


def verify_protocol_commit(protocol: Mapping[str, Any], commit: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("protocol commit must be lowercase full SHA")
    root = Path(protocol["root"])
    manifest = _mapping(protocol, "manifest")
    _git_bytes(root, f"{commit}^{{commit}}")
    branch_check = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "origin/main"],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if branch_check.returncode != 0:
        raise ValueError("protocol commit must be merged into origin/main")
    manifest_relative = MANIFEST_PATH.relative_to(ROOT).as_posix()
    first_parent = f"{commit}^1"
    try:
        _git_bytes(root, f"{first_parent}^{{commit}}")
    except ValueError as error:
        raise ValueError("protocol commit must have a valid first parent") from error
    parent_manifest = subprocess.run(
        ["git", "cat-file", "-e", f"{first_parent}:{manifest_relative}"],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if parent_manifest.returncode == 0:
        raise ValueError(
            "protocol commit must be the commit that first introduces the manifest"
        )
    introductions = subprocess.run(
        [
            "git",
            "log",
            "--first-parent",
            "--diff-filter=A",
            "--format=%H",
            commit,
            "--",
            manifest_relative,
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if introductions.returncode != 0 or introductions.stdout.splitlines() != [commit]:
        raise ValueError(
            "protocol commit must be the manifest's unique first-parent introduction"
        )
    if (
        _git_bytes(root, f"{commit}:{manifest_relative}")
        != (root / manifest_relative).read_bytes()
    ):
        raise ValueError("running manifest drifted from the frozen merge commit")
    for relative, expected in _mapping(manifest, "artifact_sha256").items():
        registered = _git_bytes(root, f"{commit}:{relative}")
        if hashlib.sha256(registered).hexdigest() != expected:
            raise ValueError(f"protocol commit artifact mismatch: {relative}")
        if _sha256(root / str(relative)) != expected:
            raise ValueError(f"running artifact drifted from freeze: {relative}")
    frozen_absent = [
        *(str(path) for path in _mapping(manifest, "claims").values()),
        *(str(path) for path in _mapping(manifest, "outputs").values()),
        *(
            path.relative_to(root).as_posix()
            for path in _raw_packet_paths(protocol).values()
        ),
    ]
    for relative in frozen_absent:
        exists = subprocess.run(
            ["git", "cat-file", "-e", f"{commit}:{relative}"],
            cwd=root,
            check=False,
            capture_output=True,
        )
        if exists.returncode == 0:
            raise ValueError("frozen merge commit must not contain observation artifacts")


def run_stage(
    stage: str,
    protocol_commit: str,
    shared_database: Path,
    output: Path,
    root: Path = ROOT,
) -> dict[str, Any]:
    worker_protocol = load_worker_protocol(root)
    verify_protocol_commit(worker_protocol, protocol_commit)
    if stage not in STAGES:
        raise ValueError("unknown stage")
    manifest = _mapping(worker_protocol, "manifest")
    expected_output = root / str(_mapping(manifest, "outputs")[stage])
    expected_claim = root / str(_mapping(manifest, "claims")[stage])
    if output.resolve() != expected_output.resolve():
        raise ValueError("output must be the registered stage path")
    if output.exists():
        raise FileExistsError("refusing to overwrite observed output")
    if expected_claim.exists():
        raise FileExistsError("stage attempt is already claimed; retry is forbidden")
    if any(path.exists() for path in _raw_packet_paths(worker_protocol, stage).values()):
        raise FileExistsError("stage raw evidence already exists; retry is forbidden")
    if stage == "holdout":
        development = root / str(_mapping(manifest, "outputs")["development"])
        if not development.exists():
            raise ValueError("holdout is closed until development exists")
        prior = _read_json(development)
        if not prior.get("all_hard_gates_pass") or prior.get("selected_arm") != ARMS[1]:
            raise ValueError(
                "holdout is closed because development did not select candidate"
            )
    _exclusive_write(
        expected_claim,
        {
            "protocol_id": PROTOCOL_ID,
            "protocol_commit": protocol_commit,
            "stage": stage,
            "attempt": 1,
            "retry_count": 0,
        },
    )
    shared_before = _sha256(shared_database)
    with TemporaryDirectory(prefix=f"ngr-{STEM}-{stage}-") as directory:
        temporary = Path(directory)
        for arm in ARMS:
            primary = run_worker(
                worker_protocol,
                stage,
                arm,
                temporary / f"{arm}-primary.sqlite",
                protocol_commit=protocol_commit,
                run="primary",
            )
            _persist_worker_packet(
                worker_protocol,
                primary,
                stage=stage,
                arm=arm,
                run="primary",
                protocol_commit=protocol_commit,
            )
            replay = run_worker(
                worker_protocol,
                stage,
                arm,
                temporary / f"{arm}-replay.sqlite",
                protocol_commit=protocol_commit,
                run="replay",
            )
            _persist_worker_packet(
                worker_protocol,
                replay,
                stage=stage,
                arm=arm,
                run="replay",
                protocol_commit=protocol_commit,
            )
    if _sha256(shared_database) != shared_before:
        raise RuntimeError("shared database changed")
    protocol = load_protocol(root, require_result_free=False)
    verify_protocol_commit(protocol, protocol_commit)
    result = finalize_stage(protocol, stage, protocol_commit, shared_before)
    _exclusive_write(output, result)
    return result


def audit_result_free(root: Path = ROOT) -> dict[str, Any]:
    protocol = load_protocol(root)
    return {
        "status": "result-free-protocol-valid",
        "protocol_id": PROTOCOL_ID,
        "source_document_count": len(protocol["corpus"].documents),
        "source_grounded_relation_count": len(protocol["manifest"]["relationships"]),
        "development_case_count": len(protocol["queries"]["stages"]["development"]),
        "holdout_case_count": len(protocol["queries"]["stages"]["holdout"]),
        "observed_result_count": 0,
        "performance": "not assessed",
    }
