"""Source-grounded relation v3 with phase-aware frozen-test support."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from . import source_grounded_relation_observation as v1
from . import source_grounded_relation_observation_v2 as v2

ROOT = Path(__file__).resolve().parents[2]
STEM = "github_source_grounded_relation_v3"
PROTOCOL_ID = "github-ngr-source-grounded-relation-seed-v3"
MANIFEST_PATH = ROOT / "tests" / "fixtures" / f"{STEM}.manifest.json"
V2_MANIFEST_RELATIVE = (
    "tests/fixtures/github_source_grounded_relation_v2.manifest.json"
)
STAGES = v1.STAGES
ARMS = v1.ARMS
RUNS = v1.RUNS
COHORTS = v1.COHORTS
GATE_IDS = v1.GATE_IDS
LIFECYCLE_CONTRACT = dict(v2.LIFECYCLE_CONTRACT)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(root: Path) -> dict[str, Any]:
    return v1._read_json(root / MANIFEST_PATH.relative_to(ROOT))


def _registry_paths(manifest: Mapping[str, Any]) -> set[str]:
    paths = {
        str(path)
        for registry in ("claims", "outputs")
        for path in v1._mapping(manifest, registry).values()
    }
    paths.update(
        str(path)
        for stage in v1._mapping(manifest, "raw_packets").values()
        for arm in stage.values()
        for path in arm.values()
    )
    return paths


def _effective_manifest(
    frozen: Mapping[str, Any], predecessor: Mapping[str, Any]
) -> dict[str, Any]:
    effective = dict(predecessor)
    for key in (
        "protocol_id",
        "issue",
        "phase",
        "lifecycle_contract",
        "claims",
        "outputs",
        "raw_packets",
        "artifact_sha256",
    ):
        effective[key] = frozen[key]
    effective["inherited_protocol_manifest"] = frozen[
        "inherited_protocol_manifest"
    ]
    effective["inherited_protocol_artifacts"] = frozen[
        "inherited_protocol_artifacts"
    ]
    return effective


def _validate_frozen_manifest(
    root: Path,
    frozen: Mapping[str, Any],
    predecessor: Mapping[str, Any],
    *,
    include_finalizer: bool,
) -> None:
    if (
        frozen.get("protocol_id") != PROTOCOL_ID
        or frozen.get("issue") != 206
        or frozen.get("phase") != "result-free-freeze-v3"
    ):
        raise ValueError("v3 manifest identity mismatch")
    if frozen.get("lifecycle_contract") != LIFECYCLE_CONTRACT:
        raise ValueError("v3 lifecycle contract mismatch")
    inherited_manifest = v1._mapping(frozen, "inherited_protocol_manifest")
    if set(inherited_manifest) != {"path", "sha256"}:
        raise ValueError("v2 inherited manifest registry mismatch")
    if inherited_manifest.get("path") != V2_MANIFEST_RELATIVE:
        raise ValueError("v3 must inherit the frozen v2 manifest")
    if _sha256(root / V2_MANIFEST_RELATIVE) != inherited_manifest.get("sha256"):
        raise ValueError("v2 inherited manifest bytes drifted")

    inherited = v1._mapping(frozen, "inherited_protocol_artifacts")
    expected_names = {"corpus", "queries", "gold", "gate"}
    if set(inherited) != expected_names:
        raise ValueError("v2 inherited artifact registry mismatch")
    predecessor_artifacts = v1._mapping(predecessor, "protocol_artifacts")
    predecessor_hashes = v1._mapping(predecessor, "artifact_sha256")
    names = expected_names if include_finalizer else {"corpus", "queries"}
    for name in names:
        item = v1._mapping(inherited, name)
        if set(item) != {"path", "sha256"}:
            raise ValueError(f"inherited {name} registry mismatch")
        relative = str(predecessor_artifacts[name])
        if item.get("path") != relative:
            raise ValueError(f"v3 inherited {name} path mismatch")
        expected = predecessor_hashes.get(relative)
        if item.get("sha256") != expected or _sha256(root / relative) != expected:
            raise ValueError(f"v3 inherited {name} bytes drifted")

    predecessor_v1 = v1._read_json(
        root / "tests/fixtures/github_source_grounded_relation_v1.manifest.json"
    )
    frozen_predecessors = v1._mapping(frozen, "predecessor_manifest_sha256")
    expected_predecessors = {
        "tests/fixtures/github_source_grounded_relation_v1.manifest.json": _sha256(
            root / "tests/fixtures/github_source_grounded_relation_v1.manifest.json"
        ),
        V2_MANIFEST_RELATIVE: _sha256(root / V2_MANIFEST_RELATIVE),
    }
    if frozen_predecessors != expected_predecessors:
        raise ValueError("v1/v2 predecessor manifest identity mismatch")
    if predecessor_v1.get("protocol_id") != "github-ngr-source-grounded-relation-seed-v1":
        raise ValueError("v1 predecessor protocol identity mismatch")

    current_paths = _registry_paths(frozen)
    if len(current_paths) != 12:
        raise ValueError("v3 observation registry must contain twelve unique paths")
    predecessor_paths = _registry_paths(predecessor) | _registry_paths(predecessor_v1)
    if current_paths & predecessor_paths:
        raise ValueError("v3 observation registry overlaps a predecessor")
    if any(Path(path).is_absolute() or ".." in Path(path).parts for path in current_paths):
        raise ValueError("v3 observation registry must stay inside the repository")


def _validate_v3_artifacts(root: Path, frozen: Mapping[str, Any]) -> None:
    for relative, expected in v1._mapping(frozen, "artifact_sha256").items():
        if _sha256(root / str(relative)) != expected:
            raise ValueError(f"v3 frozen artifact hash mismatch: {relative}")


def _validate_v3_audit(audit: Mapping[str, Any]) -> None:
    expected = {
        "protocol_id": PROTOCOL_ID,
        "phase": "result-free-freeze-v3",
        "development_run_count": 0,
        "holdout_run_count": 0,
        "retry_count": 0,
        "observed_result_count": 0,
        "shared_database_open_count": 0,
        "worker_gold_open_count": 0,
        "fresh_sqlite_per_worker": True,
        "fresh_query_gold_split_frozen_before_observation": True,
        "predecessor_copy_paraphrase_near_transform_count": 0,
        "same_protocol_retry_allowed": False,
        "performance": "not assessed",
    }
    if dict(audit) != expected:
        raise ValueError("v3 result-free audit mismatch")


def _require_result_free(protocol: Mapping[str, Any]) -> None:
    root = Path(protocol["root"])
    manifest = v1._mapping(protocol, "manifest")
    for relative in _registry_paths(manifest):
        if (root / relative).exists():
            raise ValueError("registered observation artifact must be absent at freeze")


def load_worker_protocol(root: Path = ROOT) -> dict[str, Any]:
    """Load inherited corpus/query bytes without exposing gold to a worker."""
    base = v2.load_worker_protocol(root)
    frozen = _manifest(root)
    predecessor = v1._read_json(root / V2_MANIFEST_RELATIVE)
    _validate_frozen_manifest(
        root, frozen, predecessor, include_finalizer=False
    )
    protocol = dict(base)
    protocol["manifest"] = _effective_manifest(frozen, predecessor)
    protocol["manifest_path"] = root / MANIFEST_PATH.relative_to(ROOT)
    return protocol


def load_protocol(
    root: Path = ROOT, *, require_result_free: bool = True
) -> dict[str, Any]:
    """Load v2's exact frozen evaluation identity under the v3 lifecycle."""
    base = v2.load_protocol(root, require_result_free=False)
    frozen = _manifest(root)
    predecessor = v1._mapping(base, "manifest")
    _validate_frozen_manifest(
        root, frozen, predecessor, include_finalizer=True
    )
    audit_relative = v1._string(frozen, "audit")
    audit = v1._read_json(root / audit_relative)
    _validate_v3_audit(audit)
    _validate_v3_artifacts(root, frozen)
    protocol = dict(base)
    protocol["manifest"] = _effective_manifest(frozen, predecessor)
    protocol["manifest_path"] = root / MANIFEST_PATH.relative_to(ROOT)
    protocol["audit"] = audit
    if require_result_free:
        _require_result_free(protocol)
    return protocol


@contextmanager
def _runtime_scope() -> Iterator[None]:
    replacements = {
        "STEM": STEM,
        "PROTOCOL_ID": PROTOCOL_ID,
        "MANIFEST_PATH": MANIFEST_PATH,
    }
    original_v1 = {name: getattr(v1, name) for name in replacements}
    original_v2 = {name: getattr(v2, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(v1, name, value)
            setattr(v2, name, value)
        yield
    finally:
        for name, value in original_v2.items():
            setattr(v2, name, value)
        for name, value in original_v1.items():
            setattr(v1, name, value)


def extract_source_grounded_relations(snapshot: Any) -> tuple[dict[str, str], ...]:
    return v1.extract_source_grounded_relations(snapshot)


def verify_protocol_commit(protocol: Mapping[str, Any], commit: str) -> None:
    """Verify the v3 freeze plus inherited v2 bytes at the same commit."""
    with _runtime_scope():
        v1.verify_protocol_commit(protocol, commit)
    root = Path(protocol["root"])
    frozen = _manifest(root)
    inherited = [
        v1._mapping(frozen, "inherited_protocol_manifest"),
        *(
            v1._mapping(v1._mapping(frozen, "inherited_protocol_artifacts"), name)
            for name in ("corpus", "queries", "gold", "gate")
        ),
    ]
    for item in inherited:
        relative = v1._string(item, "path")
        expected = v1._string(item, "sha256")
        committed = v1._git_bytes(root, f"{commit}:{relative}")
        if hashlib.sha256(committed).hexdigest() != expected:
            raise ValueError(f"protocol commit inherited artifact mismatch: {relative}")


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
    manifest = v1._mapping(worker_protocol, "manifest")
    expected_output = root / str(v1._mapping(manifest, "outputs")[stage])
    expected_claim = root / str(v1._mapping(manifest, "claims")[stage])
    if output.resolve() != expected_output.resolve():
        raise ValueError("output must be the registered stage path")
    if output.exists():
        raise FileExistsError("refusing to overwrite observed output")
    if expected_claim.exists():
        raise FileExistsError("stage attempt is already claimed; retry is forbidden")
    with _runtime_scope():
        raw_paths = v1._raw_packet_paths(worker_protocol, stage)
    if any(path.exists() for path in raw_paths.values()):
        raise FileExistsError("stage raw evidence already exists; retry is forbidden")
    if stage == "holdout":
        development = root / str(v1._mapping(manifest, "outputs")["development"])
        if not development.exists():
            raise ValueError("holdout is closed until development exists")
        prior = v1._read_json(development)
        if not prior.get("all_hard_gates_pass") or prior.get("selected_arm") != ARMS[1]:
            raise ValueError(
                "holdout is closed because development did not select candidate"
            )
    v1._exclusive_write(
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
            for run in RUNS:
                with _runtime_scope():
                    packet = v1.run_worker(
                        worker_protocol,
                        stage,
                        arm,
                        temporary / f"{arm}-{run}.sqlite",
                        protocol_commit=protocol_commit,
                        run=run,
                    )
                    v1._persist_worker_packet(
                        worker_protocol,
                        packet,
                        stage=stage,
                        arm=arm,
                        run=run,
                        protocol_commit=protocol_commit,
                    )
    if _sha256(shared_database) != shared_before:
        raise RuntimeError("shared database changed")
    protocol = load_protocol(root, require_result_free=False)
    verify_protocol_commit(protocol, protocol_commit)
    with _runtime_scope():
        result = v1.finalize_stage(protocol, stage, protocol_commit, shared_before)
    v1._exclusive_write(output, result)
    return result


def _stage_registry(
    protocol: Mapping[str, Any], stage: str, protocol_commit: str | None
) -> dict[str, Any]:
    with _runtime_scope():
        return v2._stage_registry(protocol, stage, protocol_commit)


def audit_repository_lifecycle(root: Path = ROOT) -> dict[str, Any]:
    """Audit every valid append-only phase without requiring result-free state."""
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


def protocol_file_inventory(root: Path = ROOT) -> tuple[str, ...]:
    """Return the exact protocol closure needed by isolated repository probes."""
    frozen = _manifest(root)
    predecessor = v1._read_json(root / V2_MANIFEST_RELATIVE)
    predecessor_v1_path = (
        "tests/fixtures/github_source_grounded_relation_v1.manifest.json"
    )
    predecessor_v1 = v1._read_json(root / predecessor_v1_path)
    paths = {
        MANIFEST_PATH.relative_to(ROOT).as_posix(),
        V2_MANIFEST_RELATIVE,
        predecessor_v1_path,
        v1._string(frozen, "audit"),
        *v1._mapping(frozen, "artifact_sha256"),
        *v1._mapping(frozen, "predecessor_manifest_sha256"),
        *v1._mapping(predecessor, "artifact_sha256"),
        *v1._mapping(predecessor, "predecessor_identity_sha256"),
        *v1._mapping(predecessor_v1, "artifact_sha256"),
    }
    for item in v1._mapping(frozen, "inherited_protocol_artifacts").values():
        paths.add(v1._string(item, "path"))
    for pair in predecessor.get("predecessor_query_gold", ()):
        paths.update((str(pair["queries"]), str(pair["gold"])))
    return tuple(sorted(paths))
