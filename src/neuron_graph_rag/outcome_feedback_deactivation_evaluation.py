from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from .corpus_integrity import verify_manifest_source_hashes
from .evidence_feedback import EngineConfig, NeuronGraphRAG
from .feedback import FeedbackLedger
from .models import SourceUseEvent


ROOT = Path(__file__).resolve().parents[2]
STEM = "outcome_feedback_deactivation_v1"
PROTOCOL_ID = "outcome-driven-feedback-deactivation-v1"
MANIFEST_PATH = ROOT / "tests" / "fixtures" / f"{STEM}.manifest.json"
STAGES = ("development", "holdout")
ARMS = ("control", "deactivation_candidate")
EDGE_FIELDS = ("source_id", "target_id", "edge_type")
_ABSOLUTE_PRIVATE_PATH = re.compile(
    r"(?:[A-Za-z]:[\\/]|/Users/|/home/|\\\\[^\\]+\\[^\\]+)"
)
_CREDENTIAL = re.compile(
    r"(?i)(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,}|"
    r"(?:token|password|secret|api[_-]?key)\s*[:=]\s*[^\s,}]+)"
)


def _encoded(payload: Any) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def read_json(path: Path) -> Any:
    raw = path.read_bytes()
    text = raw.decode("utf-8", errors="strict")
    payload = json.loads(text)
    if text != json.dumps(payload, ensure_ascii=False, indent=2) + "\n":
        raise ValueError(f"non-canonical JSON artifact: {path}")
    return payload


def write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
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


def assert_public_payload(payload: Any) -> None:
    def walk(value: Any, path: tuple[str, ...]) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                name = str(key)
                if name.lower() in {
                    "node_text",
                    "document_text",
                    "raw_text",
                    "private_path",
                    "source_path",
                    "snapshot_path",
                }:
                    raise ValueError(f"private field is forbidden: {'.'.join((*path, name))}")
                walk(item, (*path, name))
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                walk(item, (*path, str(index)))
        elif isinstance(value, str):
            if value.startswith(("/", "\\")) or _ABSOLUTE_PRIVATE_PATH.search(value):
                raise ValueError(f"absolute private path is forbidden: {'.'.join(path)}")
            if _CREDENTIAL.search(value):
                raise ValueError(f"credential-shaped value is forbidden: {'.'.join(path)}")

    walk(payload, ())


def _logical_hash(connection: sqlite3.Connection) -> str:
    return _sha256_bytes("\n".join(connection.iterdump()).encode("utf-8"))


def _schema_identity(connection: sqlite3.Connection) -> dict[str, Any]:
    rows = [
        {"name": str(row[0]), "sql": str(row[1])}
        for row in connection.execute(
            "SELECT name, sql FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
    ]
    return {
        "schema_sha256": _sha256_bytes(_encoded(rows)),
        "table_names": [row["name"] for row in rows],
    }


def acquire_transactional_snapshot(source: Path, destination: Path) -> dict[str, Any]:
    """Create one exclusive read-only SQLite backup and prove logical stability."""
    source = source.resolve(strict=True)
    destination = destination.resolve(strict=False)
    if source == destination:
        raise ValueError("source and snapshot must differ")
    destination.parent.mkdir(parents=True, exist_ok=True)
    reservation = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(reservation)
    captured_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    try:
        source_connection = sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)
        snapshot_connection = sqlite3.connect(destination)
        post_connection = sqlite3.connect(":memory:")
        try:
            source_connection.execute("PRAGMA query_only = ON")
            source_connection.backup(snapshot_connection)
            snapshot_connection.commit()
            source_before = _logical_hash(snapshot_connection)
            source_connection.backup(post_connection)
            source_after = _logical_hash(post_connection)
            if source_before != source_after:
                raise RuntimeError("source database changed during snapshot acquisition")
            if snapshot_connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise RuntimeError("snapshot integrity check failed")
            schema = _schema_identity(snapshot_connection)
            counts = {
                name: int(
                    snapshot_connection.execute(f"SELECT count(*) FROM {name}").fetchone()[0]
                )
                for name in (
                    "nodes",
                    "edges",
                    "retrievals",
                    "success_feedback",
                    "delayed_outcomes",
                )
            }
        finally:
            post_connection.close()
            snapshot_connection.close()
            source_connection.close()
        provenance = {
            "source_locator": "local_codex_ngr_database",
            "source_access": "sqlite-uri-mode-ro-query-only",
            "capture_method": "sqlite-backup-api",
            "captured_at": captured_at,
            "source_logical_sha256_before": source_before,
            "source_logical_sha256_after": source_after,
            "snapshot_logical_sha256": source_before,
            "snapshot_sha256": _sha256(destination),
            "snapshot_size": destination.stat().st_size,
            **schema,
            "row_counts": counts,
        }
        assert_public_payload(provenance)
        return provenance
    except BaseException:
        destination.unlink(missing_ok=True)
        raise


def prove_writer_verifier_round_trip(path: Path) -> None:
    payload = {
        "protocol_id": PROTOCOL_ID,
        "stage": "placeholder",
        "gate_order": ["zulu-placeholder", "alpha-placeholder", "mike-placeholder"],
    }
    write_json_exclusive(path, payload)
    try:
        if read_json(path) != payload:
            raise ValueError("placeholder semantics changed")
    finally:
        path.unlink(missing_ok=True)


def _load_current_protocol() -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = read_json(MANIFEST_PATH)
    for relative, expected in manifest["artifact_sha256"].items():
        if _sha256(ROOT / relative) != expected:
            raise ValueError(f"current artifact hash mismatch: {relative}")
    artifacts = {
        name: read_json(ROOT / relative)
        for name, relative in manifest["protocol_artifacts"].items()
    }
    assert_public_payload(manifest)
    assert_public_payload(artifacts)
    return manifest, artifacts


def _load_registered_protocol() -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = read_json(MANIFEST_PATH)
    registered = verify_manifest_source_hashes(
        ROOT, MANIFEST_PATH, manifest["artifact_sha256"]
    )
    artifacts = {
        name: json.loads(registered.artifact_bytes[relative].decode("utf-8", errors="strict"))
        for name, relative in manifest["protocol_artifacts"].items()
    }
    assert_public_payload(manifest)
    assert_public_payload(artifacts)
    return manifest, artifacts


def _edge_key(value: Mapping[str, Any] | Sequence[str]) -> str:
    if isinstance(value, Mapping):
        return "|".join(str(value[field]) for field in EDGE_FIELDS)
    return "|".join(str(item) for item in value)


def _config(candidate: bool) -> EngineConfig:
    return EngineConfig(
        soft_start_feedback_reinforcement=True,
        soft_start_feedback_ratio=0.25,
        confirmation_decay_ratio=0.5,
        sibling_feedback_normalization=1.0,
        outcome_driven_feedback_deactivation=candidate,
    )


def _edge_state(engine: NeuronGraphRAG) -> dict[str, dict[str, Any]]:
    return {
        _edge_key((edge.source_id, edge.target_id, edge.edge_type)): {
            "weight": edge.weight,
            "reinforced_count": edge.reinforced_count,
            "dormant": engine.store.edge_is_dormant(
                edge.source_id, edge.target_id, edge.edge_type
            ),
        }
        for edge in engine.store.list_edges()
    }


def _select_trace(engine: NeuronGraphRAG, case: Mapping[str, Any], now: float) -> str:
    channels = engine.search_channels(str(case["query"]), limit=21, now=now)
    surface = str(case["search_surface"])
    trace = channels.relation if surface == "relation" else channels.lexical
    target = str(case["used_node_id"])
    if target not in {hit.node.node_id for hit in trace.hits}:
        raise RuntimeError("registered node is absent from its search trace")
    if surface == "relation":
        selected = max(
            engine.store.retrieval_paths(trace.trace_id, target),
            key=lambda path: (float(path["contribution"]), str(path["seed_id"])),
            default=None,
        )
        expected = tuple(str(case["credited_edge"][field]) for field in EDGE_FIELDS)
        actual = () if selected is None else tuple(
            (str(step["source_id"]), str(step["target_id"]), str(step["edge_type"]))
            for step in selected["steps"]
        )
        if actual != (expected,):
            raise RuntimeError("registered credited edge is not the selected path")
    return trace.trace_id


def _used(ledger: FeedbackLedger, trace_id: str, node_id: str, key: str, now: float) -> Any:
    return ledger.record_source_use(
        trace_id,
        tuple(SourceUseEvent(node_id, stage) for stage in ("selected", "validated", "used")),
        idempotency_key=key,
        now=now,
    )


def _receipt(value: Any) -> dict[str, Any]:
    return asdict(value)


def _observe_case(
    snapshot: Path,
    arm_id: str,
    case: Mapping[str, Any],
    case_index: int,
) -> dict[str, Any]:
    candidate = arm_id == "deactivation_candidate"
    with TemporaryDirectory() as directory:
        clone = Path(directory) / "case.sqlite"
        shutil.copyfile(snapshot, clone)
        clock = 100_000.0 + case_index * 1_000.0
        reopened: NeuronGraphRAG | None = None
        with NeuronGraphRAG(clone, config=_config(candidate)) as engine:
            baseline = _edge_state(engine)
            ledger = FeedbackLedger(engine)
            trace_id = _select_trace(engine, case, clock)
            used = _used(
                ledger,
                trace_id,
                str(case["used_node_id"]),
                f"{arm_id}-{case['case_id']}-used",
                clock + 1.0,
            )
            after_used = _edge_state(engine)
            confirmed: dict[str, Any] | None = None
            atomicity = True
            if case["case_role"] in {"corrected", "rolled_back"}:
                confirmed_receipt = ledger.record_outcome(
                    trace_id,
                    [str(case["used_node_id"])],
                    "confirmed",
                    "registered positive outcome",
                    idempotency_key=f"{arm_id}-{case['case_id']}-confirmed",
                    now=clock + 2.0,
                )
                confirmed = _receipt(confirmed_receipt)
                if case["case_role"] == "rolled_back":
                    engine.close()
                    reopened = NeuronGraphRAG(clone, config=_config(candidate))
                    engine = reopened
                    ledger = FeedbackLedger(engine)
                before_negative = _edge_state(engine)
                if candidate:
                    engine.store.connection.execute(
                        "CREATE TEMP TRIGGER fail_deactivation BEFORE UPDATE OF active "
                        "ON feedback_contributions WHEN NEW.active = 0 "
                        "BEGIN SELECT RAISE(ABORT, 'atomicity probe'); END"
                    )
                    try:
                        ledger.record_outcome(
                            trace_id,
                            [str(case["used_node_id"])],
                            str(case["negative_outcome"]),
                            "atomicity probe",
                            idempotency_key=f"{arm_id}-{case['case_id']}-atomicity",
                            now=clock + 2.5,
                        )
                    except sqlite3.IntegrityError:
                        pass
                    else:
                        atomicity = False
                    finally:
                        engine.store.connection.execute("DROP TRIGGER fail_deactivation")
                    atomicity = atomicity and _edge_state(engine) == before_negative
            else:
                before_negative = _edge_state(engine)

            negative = ledger.record_outcome(
                trace_id,
                [str(case["used_node_id"])],
                str(case["negative_outcome"]),
                "registered negative outcome",
                idempotency_key=f"{arm_id}-{case['case_id']}-negative",
                now=clock + 3.0,
            )
            replay = ledger.record_outcome(
                trace_id,
                [str(case["used_node_id"])],
                str(case["negative_outcome"]),
                "registered negative outcome",
                idempotency_key=f"{arm_id}-{case['case_id']}-negative",
                now=clock + 9.0,
            )
            after_negative = _edge_state(engine)
            dormant_hidden = None
            reactivated: dict[str, Any] | None = None
            if case["case_role"] == "superseded":
                edge = case["credited_edge"]
                dormant_hidden = _edge_key(edge) not in {
                    _edge_key((item.source_id, item.target_id, item.edge_type))
                    for item in engine.store.outgoing_edges(str(edge["source_id"]))
                }
                reactivated_receipt = ledger.record_outcome(
                    trace_id,
                    [str(case["used_node_id"])],
                    "confirmed",
                    "registered reactivation outcome",
                    idempotency_key=f"{arm_id}-{case['case_id']}-reactivated",
                    now=clock + 4.0,
                )
                reactivated = _receipt(reactivated_receipt)
            final = _edge_state(engine)
            if reopened is not None:
                reopened.close()
        return {
            "case_id": case["case_id"],
            "case_role": case["case_role"],
            "baseline": baseline,
            "after_used": after_used,
            "before_negative": before_negative,
            "after_negative": after_negative,
            "final": final,
            "used": _receipt(used),
            "confirmed": confirmed,
            "negative": _receipt(negative),
            "idempotency_replay_equal": replay == negative,
            "atomicity_probe_passed": atomicity,
            "dormant_hidden": dormant_hidden,
            "reactivated": reactivated,
        }


def evaluate_gates(
    preflight: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
    arms: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, bool]:
    registered = {str(case["case_id"]): case for case in cases}
    candidate = {str(item["case_id"]): item for item in arms["deactivation_candidate"]}
    control = {str(item["case_id"]): item for item in arms["control"]}

    def exact(role: str) -> bool:
        case = next(item for item in cases if item["case_role"] == role)
        key = _edge_key(case["credited_edge"])
        observed = candidate[str(case["case_id"])]
        receipt = observed["negative"]
        mutation_roles = {
            mutation["mutation_role"]
            for contribution in receipt["reversed_contributions"]
            for mutation in contribution["mutations"]
        }
        return (
            receipt["deactivation_applied"]
            and mutation_roles == {"credited", "sibling"}
            and observed["after_negative"][key] == observed["baseline"][key]
            and observed["after_negative"] == observed["baseline"]
            and control[str(case["case_id"])]["after_negative"]
            == control[str(case["case_id"])]["before_negative"]
        )

    superseded_case = next(item for item in cases if item["case_role"] == "superseded")
    superseded = candidate[str(superseded_case["case_id"])]
    edge_key = _edge_key(superseded_case["credited_edge"])
    dormancy = (
        superseded["negative"]["deactivation_applied"]
        and superseded["after_negative"][edge_key]["dormant"]
        and superseded["dormant_hidden"] is True
        and superseded["reactivated"] is not None
        and len(superseded["reactivated"]["reactivated_edges"]) == 1
        and not superseded["final"][edge_key]["dormant"]
    )
    all_cases = [item for values in arms.values() for item in values]
    unattributed = [item for item in all_cases if item["case_role"] == "unattributed"]
    locality = all(
        item["after_negative"] == item["before_negative"]
        for item in unattributed
    ) and all(
        item["idempotency_replay_equal"] for item in all_cases
    )
    baseline_floor = all(
        all(
            state["weight"] >= item["baseline"][key]["weight"]
            for key, state in item["after_negative"].items()
            if key in item["baseline"]
        )
        for item in candidate.values()
    )
    control_audit_only = all(
        not item["negative"]["deactivation_applied"]
        for item in control.values()
    )
    return {
        "protocol-and-preflight-integrity": bool(preflight["passed"]),
        "corrected-exact-contribution-reversal": exact("corrected"),
        "rolled-back-exact-contribution-reversal": exact("rolled_back"),
        "superseded-dormancy-and-reactivation": dormancy,
        "baseline-floor-without-punitive-update": baseline_floor,
        "idempotency-restart-and-transaction-atomicity": all(
            item["atomicity_probe_passed"] and item["idempotency_replay_equal"]
            for item in all_cases
        ),
        "credited-sibling-locality-and-controls": locality and control_audit_only,
        "source-snapshot-privacy-and-exclusive-output": bool(
            preflight["snapshot_unchanged"]
        ),
    }


def _preflight(
    snapshot: Path,
    manifest: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    stage: str,
) -> dict[str, Any]:
    snapshot_before = _sha256(snapshot)
    cases = artifacts["fixture"]["stages"][stage]
    with sqlite3.connect(snapshot.as_uri() + "?mode=ro", uri=True) as connection:
        connection.execute("PRAGMA query_only = ON")
        schema = _schema_identity(connection)
        logical = _logical_hash(connection)
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        node_ids = {str(row[0]) for row in connection.execute("SELECT node_id FROM nodes")}
        edge_rows = {
            _edge_key(row[:3]): (float(row[3]), int(row[4]))
            for row in connection.execute(
                "SELECT source_id, target_id, edge_type, weight, reinforced_count FROM edges"
            )
        }
        baseline_match = all(
            edge_rows.get(_edge_key(case["credited_edge"]))
            == (
                float(case["registered_initial_state"]["weight"]),
                int(case["registered_initial_state"]["reinforced_count"]),
            )
            for case in cases
            if case["search_surface"] == "relation"
        )
        sibling_capacity = all(
            int(
                connection.execute(
                    "SELECT count(*) FROM edges WHERE source_id = ? AND NOT "
                    "(target_id = ? AND edge_type = ?)",
                    (
                        case["credited_edge"]["source_id"],
                        case["credited_edge"]["target_id"],
                        case["credited_edge"]["edge_type"],
                    ),
                ).fetchone()[0]
            )
            > 0
            for case in cases
            if case["case_role"] in {"corrected", "rolled_back"}
        )
    trace_eligibility = True
    try:
        with TemporaryDirectory() as directory:
            clone = Path(directory) / "preflight.sqlite"
            shutil.copyfile(snapshot, clone)
            with NeuronGraphRAG(clone, config=_config(True)) as engine:
                for index, case in enumerate(cases):
                    _select_trace(engine, case, 80_000.0 + index)
    except Exception:
        trace_eligibility = False
    registered_nodes = {
        str(case["used_node_id"])
        for case in cases
    } | {
        str(case["credited_edge"][field])
        for case in cases
        if case["search_surface"] == "relation"
        for field in ("source_id", "target_id")
    }
    output_absent = not (ROOT / manifest["outputs"][stage]).exists()
    holdout_absent = not (ROOT / manifest["outputs"]["holdout"]).exists()
    checks = {
        "snapshot_container_hash": snapshot_before == manifest["snapshot"]["snapshot_sha256"],
        "snapshot_logical_hash": logical == manifest["snapshot"]["snapshot_logical_sha256"],
        "snapshot_integrity": integrity == "ok",
        "snapshot_schema": schema == {
            "schema_sha256": manifest["snapshot"]["schema_sha256"],
            "table_names": manifest["snapshot"]["table_names"],
        },
        "registered_nodes": registered_nodes <= node_ids,
        "registered_baseline": baseline_match,
        "same_source_sibling_capacity": sibling_capacity,
        "trace_and_path_identity": trace_eligibility,
        "arm_order": [arm["arm_id"] for arm in artifacts["schedule"]["arms"]] == list(ARMS),
        "case_identity": len({case["case_id"] for case in cases}) == len(cases),
        "registered_output_absent": output_absent,
        "holdout_absent_before_development": stage != "development" or holdout_absent,
        "privacy": True,
        "placeholder_round_trip": artifacts["audit"]["placeholder_round_trip_passed"] is True,
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "snapshot_sha256_before": snapshot_before,
        "snapshot_sha256_after": _sha256(snapshot),
        "snapshot_unchanged": _sha256(snapshot) == snapshot_before,
    }


def preflight_snapshot(snapshot: Path) -> dict[str, Any]:
    manifest, artifacts = _load_current_protocol()
    reports = {
        stage: _preflight(snapshot, manifest, artifacts, stage) for stage in STAGES
    }
    return {
        "protocol_id": PROTOCOL_ID,
        "stages": reports,
        "passed": all(report["passed"] for report in reports.values()),
    }


def run_registered_stage(stage: str, snapshot: Path) -> Path:
    if stage not in STAGES:
        raise ValueError("unknown stage")
    manifest, artifacts = _load_registered_protocol()
    preflight = _preflight(snapshot, manifest, artifacts, stage)
    if not preflight["passed"]:
        raise RuntimeError("protocol preflight failed; no registered result was written")
    if stage == "holdout":
        development = ROOT / manifest["outputs"]["development"]
        if not development.exists() or not verify_registered_result("development"):
            raise RuntimeError("development must pass before holdout")
    cases = artifacts["fixture"]["stages"][stage]
    observed = {
        arm: [
            _observe_case(snapshot, arm, case, index)
            for index, case in enumerate(cases)
        ]
        for arm in ARMS
    }
    gates = evaluate_gates(preflight, cases, observed)
    gate_order = [item["gate_id"] for item in artifacts["gate"]["gates"]]
    if list(gates) != gate_order:
        raise RuntimeError("gate order differs from the frozen contract")
    payload = {
        "protocol_id": PROTOCOL_ID,
        "stage": stage,
        "snapshot_sha256": _sha256(snapshot),
        "arms": observed,
        "hard_gates": gates,
        "all_hard_gates_pass": all(gates.values()),
    }
    assert_public_payload(payload)
    output = ROOT / manifest["outputs"][stage]
    write_json_exclusive(output, payload)
    verify_registered_result(stage)
    return output


def verify_registered_result(stage: str) -> bool:
    if stage not in STAGES:
        raise ValueError("unknown stage")
    manifest, artifacts = _load_registered_protocol()
    payload = read_json(ROOT / manifest["outputs"][stage])
    assert_public_payload(payload)
    gate_order = [item["gate_id"] for item in artifacts["gate"]["gates"]]
    if payload["protocol_id"] != PROTOCOL_ID or payload["stage"] != stage:
        raise ValueError("registered result identity mismatch")
    if list(payload["hard_gates"]) != gate_order:
        raise ValueError("registered gate order mismatch")
    if payload["all_hard_gates_pass"] != all(payload["hard_gates"].values()):
        raise ValueError("registered aggregate gate mismatch")
    return bool(payload["all_hard_gates_pass"])
