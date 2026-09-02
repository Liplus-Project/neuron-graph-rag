"""Superseding source-grounded relation protocol with append-only lifecycle audit."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from . import source_grounded_relation_observation as v1

ROOT = Path(__file__).resolve().parents[2]
STEM = "github_source_grounded_relation_v2"
PROTOCOL_ID = "github-ngr-source-grounded-relation-seed-v2"
MANIFEST_PATH = ROOT / "tests" / "fixtures" / f"{STEM}.manifest.json"
STAGES = v1.STAGES
ARMS = v1.ARMS
RUNS = v1.RUNS
COHORTS = v1.COHORTS
GATE_IDS = v1.GATE_IDS
LIFECYCLE_CONTRACT = {
    "registry_policy": "append-only-exclusive-canonical-json",
    "stage_order": ["development", "conditional-holdout"],
    "partial_evidence_policy": "preserve-and-refuse-retry",
    "holdout_open_condition": "development-passed-and-candidate-selected",
    "freeze_identity_scope": "protocol-artifacts-only",
    "observation_identity_scope": "registered-claim-raw-output",
}

_V1_VALIDATE_SOURCE_CONTRACT = v1._validate_source_contract
_PATCH_NAMES = ("STEM", "PROTOCOL_ID", "MANIFEST_PATH", "_validate_source_contract")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_predecessor_freeze(root: Path, manifest: Mapping[str, Any]) -> None:
    predecessor = v1._read_json(
        root / "tests" / "fixtures" / "github_source_grounded_relation_v1.manifest.json"
    )
    if (
        predecessor.get("protocol_id")
        != "github-ngr-source-grounded-relation-seed-v1"
        or predecessor.get("issue") != 200
        or predecessor.get("phase") != "result-free-freeze"
    ):
        raise ValueError("v1 predecessor freeze identity mismatch")
    for relative, expected in v1._mapping(predecessor, "artifact_sha256").items():
        if _sha256(root / str(relative)) != expected:
            raise ValueError(f"v1 frozen artifact drifted: {relative}")
    for relative, expected in v1._mapping(
        manifest, "predecessor_identity_sha256"
    ).items():
        if _sha256(root / str(relative)) != expected:
            raise ValueError(f"predecessor identity drifted: {relative}")


def _validate_source_contract(protocol: Mapping[str, Any]) -> tuple[str, ...]:
    manifest = v1._mapping(protocol, "manifest")
    if (
        manifest.get("issue") != 203
        or manifest.get("phase") != "result-free-freeze-v2"
    ):
        raise ValueError("v2 manifest issue or phase mismatch")
    if manifest.get("lifecycle_contract") != LIFECYCLE_CONTRACT:
        raise ValueError("v2 lifecycle contract mismatch")
    root = Path(protocol["root"])
    _validate_predecessor_freeze(root, manifest)
    predecessor = v1._read_json(
        root / "tests" / "fixtures" / "github_source_grounded_relation_v1.manifest.json"
    )
    source = v1._mapping(manifest, "source")
    predecessor_source = v1._mapping(predecessor, "source")
    if source.get("generated_by") != "tools/acquire_source_grounded_relation_corpus_v2.py":
        raise ValueError("v2 source generator identity mismatch")
    if source.get("commit") == predecessor_source.get("commit"):
        raise ValueError("v2 source commit must be fresh")
    if set(source.get("paths", ())) & set(predecessor_source.get("paths", ())):
        raise ValueError("v2 source corpus must be path-disjoint from v1")
    current_source_identities = {
        (source.get("repository"), str(path)) for path in source.get("paths", ())
    }
    predecessor_source_identities: set[tuple[object, str]] = set()
    for relative in manifest.get("predecessor_source_corpora", ()):
        payload = v1._read_json(root / str(relative), require_canonical=False)
        repository = payload.get("repository")
        for document in v1._rows(payload, "documents"):
            predecessor_source_identities.add(
                (repository, v1._string(document, "path"))
            )
    if current_source_identities & predecessor_source_identities:
        raise ValueError("v2 source identity overlaps a named predecessor corpus")
    for registry in ("claims", "outputs"):
        if set(v1._mapping(manifest, registry).values()) & set(
            v1._mapping(predecessor, registry).values()
        ):
            raise ValueError(f"v2 {registry} paths overlap v1")
    current_raw = {
        str(path)
        for stage in v1._mapping(manifest, "raw_packets").values()
        for arm in stage.values()
        for path in arm.values()
    }
    predecessor_raw = {
        str(path)
        for stage in v1._mapping(predecessor, "raw_packets").values()
        for arm in stage.values()
        for path in arm.values()
    }
    if current_raw & predecessor_raw:
        raise ValueError("v2 raw packet paths overlap v1")
    proxy_manifest = dict(manifest)
    proxy_manifest["issue"] = 200
    proxy_manifest["phase"] = "result-free-freeze"
    proxy_source = dict(source)
    proxy_source["generated_by"] = "tools/acquire_source_grounded_relation_corpus.py"
    proxy_manifest["source"] = proxy_source
    proxy_protocol = dict(protocol)
    proxy_protocol["manifest"] = proxy_manifest
    return _V1_VALIDATE_SOURCE_CONTRACT(proxy_protocol)


@contextmanager
def _v2_scope() -> Iterator[None]:
    replacements = {
        "STEM": STEM,
        "PROTOCOL_ID": PROTOCOL_ID,
        "MANIFEST_PATH": MANIFEST_PATH,
        "_validate_source_contract": _validate_source_contract,
    }
    original = {name: getattr(v1, name) for name in _PATCH_NAMES}
    try:
        for name, value in replacements.items():
            setattr(v1, name, value)
        yield
    finally:
        for name, value in original.items():
            setattr(v1, name, value)


def extract_source_grounded_relations(snapshot: Any) -> tuple[dict[str, str], ...]:
    return v1.extract_source_grounded_relations(snapshot)


def load_worker_protocol(root: Path = ROOT) -> dict[str, Any]:
    with _v2_scope():
        return v1.load_worker_protocol(root)


def load_protocol(
    root: Path = ROOT, *, require_result_free: bool = True
) -> dict[str, Any]:
    with _v2_scope():
        return v1.load_protocol(root, require_result_free=require_result_free)


def verify_protocol_commit(protocol: Mapping[str, Any], commit: str) -> None:
    """Verify immutable freeze identity, separately from current registry state."""
    with _v2_scope():
        v1.verify_protocol_commit(protocol, commit)


def run_stage(
    stage: str,
    protocol_commit: str,
    shared_database: Path,
    output: Path,
    root: Path = ROOT,
) -> dict[str, Any]:
    with _v2_scope():
        return v1.run_stage(stage, protocol_commit, shared_database, output, root)


def _claim(
    protocol: Mapping[str, Any], stage: str, protocol_commit: str | None
) -> tuple[Path, Mapping[str, Any] | None, str | None]:
    root = Path(protocol["root"])
    manifest = v1._mapping(protocol, "manifest")
    path = root / str(v1._mapping(manifest, "claims")[stage])
    if not path.exists():
        return path, None, protocol_commit
    payload = v1._read_json(path)
    expected_keys = {
        "protocol_id",
        "protocol_commit",
        "stage",
        "attempt",
        "retry_count",
    }
    if set(payload) != expected_keys:
        raise ValueError("claim fields do not match the frozen schema")
    observed_commit = payload.get("protocol_commit")
    expected = {
        "protocol_id": PROTOCOL_ID,
        "protocol_commit": observed_commit,
        "stage": stage,
        "attempt": 1,
        "retry_count": 0,
    }
    if (
        not isinstance(observed_commit, str)
        or not re.fullmatch(r"[0-9a-f]{40}", observed_commit)
        or payload != expected
    ):
        raise ValueError("claim identity mismatch")
    if protocol_commit is not None and observed_commit != protocol_commit:
        raise ValueError("observation registry mixes protocol commits")
    return path, payload, observed_commit


def _stage_registry(
    protocol: Mapping[str, Any], stage: str, protocol_commit: str | None
) -> dict[str, Any]:
    root = Path(protocol["root"])
    manifest = v1._mapping(protocol, "manifest")
    _, claim, protocol_commit = _claim(protocol, stage, protocol_commit)
    with _v2_scope():
        registered = v1._raw_packet_paths(protocol, stage)
    ordered_keys = [
        (stage, arm, run)
        for arm in ARMS
        for run in RUNS
    ]
    existing_keys = [key for key in ordered_keys if registered[key].exists()]
    if existing_keys != ordered_keys[: len(existing_keys)]:
        raise ValueError("raw packets must form the registered append-only prefix")
    if claim is None and existing_keys:
        raise ValueError("raw packet exists without a stage claim")
    for _, arm, run in existing_keys:
        packet = v1._read_json(registered[(stage, arm, run)])
        if protocol_commit is None:
            raise ValueError("raw packet has no protocol commit claim")
        with _v2_scope():
            v1._validate_worker_packet(
                packet,
                stage=stage,
                arm=arm,
                run=run,
                protocol_commit=protocol_commit,
            )
        _validate_packet_shape(protocol, packet, stage)
    output_path = root / str(v1._mapping(manifest, "outputs")[stage])
    output: Mapping[str, Any] | None = None
    if output_path.exists():
        if claim is None or len(existing_keys) != len(ordered_keys):
            raise ValueError("observed output requires claim and complete raw packet set")
        output = v1._read_json(output_path)
        shared_before = output.get("shared_database_sha256_before")
        shared_after = output.get("shared_database_sha256_after")
        if (
            not isinstance(shared_before, str)
            or not re.fullmatch(r"[0-9a-f]{64}", shared_before)
            or shared_after != shared_before
        ):
            raise ValueError("shared database hash identity mismatch")
        with _v2_scope():
            expected = v1.finalize_stage(
                protocol, stage, str(protocol_commit), shared_before
            )
        if v1._encoded(output) != v1._encoded(expected):
            raise ValueError("observed output does not match registered raw packets")
    return {
        "claim": claim is not None,
        "raw_packet_count": len(existing_keys),
        "output": output,
        "protocol_commit": protocol_commit,
    }


def _validate_packet_shape(
    protocol: Mapping[str, Any], packet: Mapping[str, Any], stage: str
) -> None:
    query_rows = v1._rows(
        v1._mapping(v1._mapping(protocol, "queries"), "stages"), stage
    )
    cases = v1._rows(packet, "cases")
    if len(cases) != len(query_rows):
        raise ValueError("raw packet case count mismatch")
    documents = {document.path: document for document in protocol["corpus"].documents}
    relation_keys = {
        (row["source_path"], row["target_path"], row["edge_type"])
        for row in v1._mapping(protocol, "manifest")["relationships"]
    }
    for case, query in zip(cases, query_rows, strict=True):
        if set(case) != {"case_id", "query", "referenced_seed_paths", "hits"}:
            raise ValueError("raw packet case fields do not match the frozen schema")
        if case["case_id"] != query["case_id"] or case["query"] != query["query"]:
            raise ValueError("raw packet query identity mismatch")
        seeds = case["referenced_seed_paths"]
        hits = case["hits"]
        if (
            not isinstance(seeds, list)
            or any(seed not in documents for seed in seeds)
            or not isinstance(hits, list)
        ):
            raise ValueError("raw packet seed or hit registry mismatch")
        for hit in hits:
            if not isinstance(hit, Mapping) or set(hit) != {
                "path",
                "source_url",
                "content_sha256",
                "sparse_score",
                "dense_score",
                "entry_score",
                "graph_activation",
                "final_score",
                "relation_paths",
            }:
                raise ValueError("raw packet hit fields do not match the frozen schema")
            document = documents.get(hit["path"])
            if (
                document is None
                or hit["source_url"] != document.source_url
                or hit["content_sha256"] != document.content_sha256
                or any(
                    not isinstance(hit[field], (int, float))
                    for field in (
                        "sparse_score",
                        "dense_score",
                        "entry_score",
                        "graph_activation",
                        "final_score",
                    )
                )
                or not isinstance(hit["relation_paths"], list)
            ):
                raise ValueError("raw packet hit identity mismatch")
            for relation_path in hit["relation_paths"]:
                if not isinstance(relation_path, Mapping) or set(relation_path) != {
                    "seed_path",
                    "target_path",
                    "steps",
                }:
                    raise ValueError("relation path fields do not match the frozen schema")
                if relation_path["target_path"] != hit["path"]:
                    raise ValueError("relation path target mismatch")
                steps = relation_path["steps"]
                if not isinstance(steps, list) or not steps:
                    raise ValueError("relation path must contain steps")
                for step in steps:
                    if not isinstance(step, Mapping) or set(step) != {
                        "source_path",
                        "target_path",
                        "edge_type",
                    }:
                        raise ValueError("relation step fields do not match the frozen schema")
                    if (
                        step["source_path"],
                        step["target_path"],
                        step["edge_type"],
                    ) not in relation_keys:
                        raise ValueError("relation step is outside the frozen registry")


def audit_repository_lifecycle(root: Path = ROOT) -> dict[str, Any]:
    """Audit the mutable append-only registry without requiring it to stay empty."""
    protocol = load_protocol(root, require_result_free=False)
    development = _stage_registry(protocol, "development", None)
    holdout = _stage_registry(
        protocol, "holdout", development["protocol_commit"]
    )
    development_output = development["output"]
    eligible = bool(
        development_output
        and development_output.get("all_hard_gates_pass") is True
        and development_output.get("selected_arm") == ARMS[1]
    )
    if development_output is None and (
        holdout["claim"] or holdout["raw_packet_count"] or holdout["output"]
    ):
        raise ValueError("holdout opened before development completed")
    if development_output is not None and not eligible and (
        holdout["claim"] or holdout["raw_packet_count"] or holdout["output"]
    ):
        raise ValueError("holdout opened without candidate eligibility")
    if development_output is None:
        if development["raw_packet_count"]:
            phase = "development-partial"
        elif development["claim"]:
            phase = "development-claimed"
        else:
            phase = "result-free"
    elif not eligible:
        phase = "development-closed"
    elif holdout["output"] is not None:
        phase = "holdout-completed"
    elif holdout["raw_packet_count"]:
        phase = "holdout-partial"
    elif holdout["claim"]:
        phase = "holdout-claimed"
    else:
        phase = "holdout-eligible"
    return {
        "status": "observation-registry-valid",
        "protocol_id": PROTOCOL_ID,
        "phase": phase,
        "protocol_commit": development["protocol_commit"],
        "development": {
            "claim": development["claim"],
            "raw_packet_count": development["raw_packet_count"],
            "output": development_output is not None,
            "candidate_selected": eligible,
        },
        "holdout": {
            "eligible": eligible,
            "claim": holdout["claim"],
            "raw_packet_count": holdout["raw_packet_count"],
            "output": holdout["output"] is not None,
        },
    }


def audit_result_free(root: Path = ROOT) -> dict[str, Any]:
    state = audit_repository_lifecycle(root)
    if state["phase"] != "result-free":
        raise ValueError("repository observation registry is not result-free")
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
