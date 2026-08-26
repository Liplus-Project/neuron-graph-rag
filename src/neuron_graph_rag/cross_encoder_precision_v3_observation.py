from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import subprocess
import sys
import time
import unittest
from collections.abc import Iterator, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any

from .cross_encoder_precision_v3_evaluation import (
    PROTOCOL_ID,
    ROOT,
    archive_stage,
    evaluate_result_payload,
    load_protocol,
    project_passages,
    read_json,
    register_stage_claim,
    sha256_bytes,
    verify_phase_state,
    verify_protocol_commit,
    verify_result_payload,
    write_json_exclusive,
    write_stage_error,
    write_stage_result,
)
from .engine import EngineConfig, NeuronGraphRAG
from .models import SearchHit

PROTOCOL_COMMIT = "b762645d2521a3e23ac201b662ea1cbf25e2a260"
SOURCE_COMMIT = "c32b3049fd3daaa2190faf5e3e85955a195ee88c"
OBSERVATION_NOW = 1_700_000_000.0
BATCH_SIZE = 8
EVIDENCE = Path("tests/evidence/github_cross_encoder_precision_v3")
PREFLIGHT_FILES = (
    "preflight.json",
    "model-verification.json",
    "dependency-report.json",
    "preflight-commands.json",
)


def canonical_sha256(value: object) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def default_external_root() -> Path:
    return ROOT.parent.parent / "experiments" / "github_cross_encoder_precision_v3"


def default_model_cache() -> Path:
    return (
        ROOT.parent.parent
        / "experiments"
        / "github_cross_encoder_precision_v1"
        / "model-cache"
    )


def shared_database_path() -> Path:
    return Path.home() / ".ngrdb" / "knowledge.db"


def hash_file_shared(path: Path) -> str:
    digest = hashlib.sha256()
    with _open_shared_read(path) as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@contextlib.contextmanager
def _open_shared_read(path: Path) -> Iterator[IO[bytes]]:
    if os.name != "nt":
        with path.open("rb") as handle:
            yield handle
        return
    import ctypes
    import msvcrt
    from ctypes import wintypes

    create_file = ctypes.windll.kernel32.CreateFileW
    create_file.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    create_file.restype = wintypes.HANDLE
    handle = create_file(
        str(path),
        0x80000000,
        0x00000001 | 0x00000002 | 0x00000004,
        None,
        3,
        0x00000080,
        None,
    )
    invalid = ctypes.c_void_p(-1).value
    if handle == invalid:
        raise ctypes.WinError()
    try:
        descriptor = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    except BaseException:
        ctypes.windll.kernel32.CloseHandle(handle)
        raise
    with os.fdopen(descriptor, "rb") as stream:
        yield stream


def experiment_python(external_root: Path) -> Path:
    return external_root / "venv" / "Scripts" / "python.exe"


def preflight(
    root: Path = ROOT,
    external_root: Path | None = None,
    model_cache: Path | None = None,
) -> dict[str, Any]:
    external = (external_root or default_external_root()).resolve()
    cache = (model_cache or default_model_cache()).resolve()
    protocol = load_protocol(root)
    verify_protocol_commit(PROTOCOL_COMMIT, protocol)
    if (
        root.resolve() != ROOT.resolve()
        or subprocess.run(
            ["git", "merge-base", "--is-ancestor", PROTOCOL_COMMIT, "HEAD"],
            cwd=root,
            check=False,
            capture_output=True,
        ).returncode
        != 0
    ):
        raise ValueError("preflight HEAD must contain the exact freeze merge commit")
    if verify_phase_state(protocol) != {
        "development": "unobserved",
        "holdout": "unobserved",
    }:
        raise ValueError("preflight requires an unobserved protocol")

    evidence_root = root / EVIDENCE
    if any((evidence_root / name).exists() for name in PREFLIGHT_FILES):
        raise FileExistsError("preflight evidence already exists")
    shared = shared_database_path()
    if not shared.is_file():
        raise FileNotFoundError("shared database path is unavailable")
    shared_before = hash_file_shared(shared)

    external.mkdir(parents=True, exist_ok=True)
    venv = external / "venv"
    command_rows: list[dict[str, Any]] = []
    if not experiment_python(external).is_file():
        _run_logged(
            [sys.executable, "-m", "venv", str(venv)],
            root,
            os.environ.copy(),
            command_rows,
        )
    python = experiment_python(external)
    lock = root / "tests/fixtures/github_cross_encoder_precision_v3.requirements.lock"
    _run_logged(
        [str(python), "-m", "pip", "install", "--require-hashes", "-r", str(lock)],
        root,
        os.environ.copy(),
        command_rows,
    )
    package_output = _run_logged(
        [str(python), "-m", "pip", "freeze", "--all"],
        root,
        os.environ.copy(),
        command_rows,
    )

    verification_report_path = external / "model-verification.json"
    if verification_report_path.exists():
        raise FileExistsError("external model verification report already exists")
    _run_logged(
        [
            str(python),
            "-m",
            "neuron_graph_rag.cross_encoder_precision_v3_observation",
            "model-verify",
            "--cache",
            str(cache),
            "--output",
            str(verification_report_path),
        ],
        root,
        _worker_environment(root, offline=True),
        command_rows,
    )
    model_report = read_json(verification_report_path)
    _verify_model_report(protocol, model_report, cache)

    offline = _worker_environment(root, offline=True)
    probe_output = _run_logged(
        [
            str(python),
            "-m",
            "neuron_graph_rag.cross_encoder_precision_v3_observation",
            "model-probe",
            "--cache",
            str(cache),
        ],
        root,
        offline,
        command_rows,
    )
    probe = json.loads(probe_output)
    if (
        probe.get("forward_inference_count") != 2
        or probe.get("batch_size") != BATCH_SIZE
    ):
        raise ValueError("offline model probe did not cover both frozen models")

    for arguments in (
        [str(python), "-m", "unittest", "tests.test_cross_encoder_precision_v3"],
        [
            str(python),
            "-m",
            "unittest",
            "tests.test_cross_encoder_precision_v3_observation",
        ],
        [
            str(python),
            "-m",
            "neuron_graph_rag.cross_encoder_precision_v3_evaluation",
            "audit",
        ],
        [
            str(python),
            "-m",
            "neuron_graph_rag.cross_encoder_precision_v3_evaluation",
            "probe",
        ],
    ):
        _run_logged(arguments, root, offline, command_rows)
    project_python = root / ".venv" / "Scripts" / "python.exe"
    if not project_python.is_file():
        raise FileNotFoundError("project verification environment is unavailable")
    full_test_commands = _full_test_commands(project_python, root)
    for arguments in full_test_commands:
        _run_logged(arguments, root, offline, command_rows)
    shared_after = hash_file_shared(shared)
    if shared_after != shared_before:
        raise ValueError("shared database changed during preflight")
    dependency_report = {
        "protocol_id": PROTOCOL_ID,
        "protocol_commit": PROTOCOL_COMMIT,
        "lock_path": str(lock.relative_to(root)).replace("\\", "/"),
        "lock_sha256": sha256_bytes(lock.read_bytes()),
        "venv_path": str(venv),
        "packages": sorted(line for line in package_output.splitlines() if line),
    }
    preflight_report = {
        "protocol_id": PROTOCOL_ID,
        "protocol_commit": PROTOCOL_COMMIT,
        "source_commit": SOURCE_COMMIT,
        "completed_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "external_root": str(external),
        "cache_path": str(cache),
        "cache_bytes": _tree_bytes(cache),
        "cache_reused_as_verified_model_bytes_only": True,
        "v1_evidence_semantic_content_read": False,
        "v2_evidence_semantic_content_read": False,
        "v1_worker_packet_reused": False,
        "v2_worker_packet_reused": False,
        "shared_database_path": str(shared),
        "shared_database_sha256_before": shared_before,
        "shared_database_sha256_after": shared_after,
        "offline": True,
        "trust_remote_code": False,
        "batch_size": BATCH_SIZE,
        "preflight_forward_inference_count": 2,
        "full_test_process_isolation": (
            "per-test" if os.name == "nt" else "single-discover"
        ),
        "registered_query_execution_count": 0,
        "observed_stage_inference_count": 0,
        "claim_count": 0,
        "phase": {"development": "unobserved", "holdout": "unobserved"},
        "model_report_sha256": canonical_sha256(model_report),
        "dependency_report_sha256": canonical_sha256(dependency_report),
    }
    write_json_exclusive(evidence_root / "model-verification.json", model_report)
    write_json_exclusive(evidence_root / "dependency-report.json", dependency_report)
    write_json_exclusive(
        evidence_root / "preflight-commands.json", {"commands": command_rows}
    )
    write_json_exclusive(evidence_root / "preflight.json", preflight_report)
    verify_preflight(root, external, cache)
    return preflight_report


def verify_preflight(
    root: Path = ROOT,
    external_root: Path | None = None,
    model_cache: Path | None = None,
) -> dict[str, Any]:
    external = (external_root or default_external_root()).resolve()
    cache = (model_cache or default_model_cache()).resolve()
    protocol = load_protocol(root)
    verify_protocol_commit(PROTOCOL_COMMIT, protocol)
    evidence_root = root / EVIDENCE
    report = read_json(evidence_root / "preflight.json")
    model_report = read_json(evidence_root / "model-verification.json")
    dependency_report = read_json(evidence_root / "dependency-report.json")
    commands = read_json(evidence_root / "preflight-commands.json")
    if (
        report.get("protocol_id") != PROTOCOL_ID
        or report.get("protocol_commit") != PROTOCOL_COMMIT
        or report.get("source_commit") != SOURCE_COMMIT
        or report.get("external_root") != str(external)
        or report.get("cache_path") != str(cache)
        or report.get("cache_reused_as_verified_model_bytes_only") is not True
        or report.get("v1_evidence_semantic_content_read") is not False
        or report.get("v2_evidence_semantic_content_read") is not False
        or report.get("v1_worker_packet_reused") is not False
        or report.get("v2_worker_packet_reused") is not False
        or report.get("offline") is not True
        or report.get("trust_remote_code") is not False
        or report.get("batch_size") != BATCH_SIZE
        or report.get("full_test_process_isolation")
        != ("per-test" if os.name == "nt" else "single-discover")
        or report.get("claim_count") != 0
        or report.get("registered_query_execution_count") != 0
        or report.get("observed_stage_inference_count") != 0
        or report.get("phase") != {"development": "unobserved", "holdout": "unobserved"}
    ):
        raise ValueError("preflight report identity mismatch")
    if report.get("model_report_sha256") != canonical_sha256(model_report):
        raise ValueError("preflight model report binding mismatch")
    if report.get("dependency_report_sha256") != canonical_sha256(dependency_report):
        raise ValueError("preflight dependency report binding mismatch")
    if not isinstance(commands.get("commands"), list) or not commands["commands"]:
        raise ValueError("preflight command evidence is missing")
    lock = root / "tests/fixtures/github_cross_encoder_precision_v3.requirements.lock"
    if dependency_report.get("lock_sha256") != sha256_bytes(lock.read_bytes()):
        raise ValueError("dependency lock binding mismatch")
    _verify_model_report(protocol, model_report, cache)
    shared = shared_database_path()
    shared_hash = hash_file_shared(shared)
    if (
        report.get("shared_database_sha256_before") != shared_hash
        or report.get("shared_database_sha256_after") != shared_hash
    ):
        raise ValueError("shared database hash no longer matches preflight")
    return report


def run_conditional(
    root: Path = ROOT,
    external_root: Path | None = None,
    model_cache: Path | None = None,
) -> dict[str, Any]:
    external = (external_root or default_external_root()).resolve()
    cache = (model_cache or default_model_cache()).resolve()
    preflight_report = verify_preflight(root, external, cache)
    before = preflight_report["shared_database_sha256_before"]
    execution_rows: list[dict[str, Any]] = []
    development: dict[str, Any] | None = None
    holdout: dict[str, Any] | None = None
    try:
        development = _run_stage_once(
            "development", root, external, cache, execution_rows
        )
        if development.get("all_hard_gates_pass") is True:
            holdout = _run_stage_once("holdout", root, external, cache, execution_rows)
    except BaseException as error:
        after = hash_file_shared(shared_database_path())
        phases = verify_phase_state(load_protocol(root))
        failure = {
            "protocol_id": PROTOCOL_ID,
            "protocol_commit": PROTOCOL_COMMIT,
            "shared_database_sha256_before": before,
            "shared_database_sha256_after": after,
            "shared_database_unchanged": after == before,
            "phase": phases,
            "error": f"{type(error).__name__}: {error}",
            "commands": execution_rows,
        }
        write_json_exclusive(root / EVIDENCE / "execution-error.json", failure)
        raise
    after = hash_file_shared(shared_database_path())
    if after != before:
        raise ValueError("shared database changed during observation")
    phases = verify_phase_state(load_protocol(root))
    execution_report = {
        "protocol_id": PROTOCOL_ID,
        "protocol_commit": PROTOCOL_COMMIT,
        "shared_database_sha256_before": before,
        "shared_database_sha256_after": after,
        "claim_count": 1 + int(holdout is not None),
        "stage_process_count": 6 * (1 + int(holdout is not None)),
        "baseline_stage_process_count": 2 * (1 + int(holdout is not None)),
        "model_stage_process_count": 4 * (1 + int(holdout is not None)),
        "phase": phases,
        "selected_candidate": {
            "development": development.get("selected_candidate_id"),
            "holdout": None
            if holdout is None
            else holdout.get("selected_candidate_id"),
        },
        "commands": execution_rows,
    }
    write_json_exclusive(root / EVIDENCE / "execution.json", execution_report)
    return {
        "development": development,
        "holdout": holdout,
        "execution": execution_report,
    }


def _run_stage_once(
    stage: str,
    root: Path,
    external: Path,
    cache: Path,
    command_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    protocol = load_protocol(root)
    claim_path = register_stage_claim(stage, PROTOCOL_COMMIT, root)
    claim_raw = claim_path.read_bytes()
    stage_root = external / "runs" / stage
    stage_root.mkdir(parents=True, exist_ok=True)
    try:
        baseline_primary = _run_worker(
            root,
            external,
            cache,
            stage,
            "baseline",
            "primary",
            stage_root,
            command_rows,
        )
        baseline_replay = _run_worker(
            root, external, cache, stage, "baseline", "replay", stage_root, command_rows
        )
        baseline = {
            "baseline_id": "current-ngr",
            "cases": baseline_primary["cases"],
            "state": _combine_state(baseline_primary, baseline_replay),
        }
        model_raws = []
        for model_key in ("base", "v2-m3"):
            primary = _run_worker(
                root,
                external,
                cache,
                stage,
                model_key,
                "primary",
                stage_root,
                command_rows,
            )
            replay = _run_worker(
                root,
                external,
                cache,
                stage,
                model_key,
                "replay",
                stage_root,
                command_rows,
            )
            if primary["cases"] != replay["cases"]:
                raise ValueError(f"{model_key} replay raw cases differ")
            model_raws.append(
                {
                    "model_id": primary["model_id"],
                    "revision": primary["revision"],
                    "cases": primary["cases"],
                    "state": _combine_state(primary, replay),
                    "metrics": primary["metrics"],
                }
            )
        _archive_raw_workers(stage, stage_root, root)
        result = evaluate_result_payload(
            protocol, stage, claim_raw, baseline, model_raws
        )
        verify_result_payload(protocol, stage, result, claim_raw)
        write_stage_result(stage, result, root)
        archive_stage(stage, root)
        verify_phase_state(load_protocol(root))
        return result
    except BaseException as error:
        raw_manifest = root / EVIDENCE / f"{stage}.raw-archive.json"
        if not raw_manifest.exists() and stage_root.exists():
            _archive_raw_workers(stage, stage_root, root)
        if claim_path.exists():
            write_stage_error(stage, f"{type(error).__name__}: {error}", root)
            archive_stage(stage, root)
        raise


def _archive_raw_workers(stage: str, stage_root: Path, root: Path) -> Path:
    names = tuple(
        f"{kind}-{replay}.json"
        for kind in ("baseline", "base", "v2-m3")
        for replay in ("primary", "replay")
    )
    archive_root = root / EVIDENCE / "raw" / stage
    rows = []
    process_ids: list[str] = []
    database_ids: list[str] = []
    for name in names:
        source = stage_root / name
        if not source.is_file():
            continue
        raw = source.read_bytes()
        packet = json.loads(raw.decode("utf-8", errors="strict"))
        if not isinstance(packet, dict):
            raise TypeError("raw worker packet must be an object")
        process_id = packet.get("process_id")
        database_id = packet.get("database_id")
        if not isinstance(process_id, str) or not process_id:
            raise ValueError("raw worker process identity is missing")
        if not isinstance(database_id, str) or not database_id:
            raise ValueError("raw worker database identity is missing")
        process_ids.append(process_id)
        database_ids.append(database_id)
        destination = archive_root / name
        _write_bytes_exclusive(destination, raw)
        rows.append(
            {
                "runtime_path": str(source.resolve()).replace("\\", "/"),
                "archive_path": str(destination.relative_to(root)).replace("\\", "/"),
                "size": len(raw),
                "sha256": sha256_bytes(raw),
                "byte_identity": destination.read_bytes() == raw,
            }
        )
    complete = len(rows) == len(names)
    if complete and (
        len(set(process_ids)) != len(names) or len(set(database_ids)) != len(names)
    ):
        raise ValueError("worker process/database identities are not unique")
    manifest = root / EVIDENCE / f"{stage}.raw-archive.json"
    write_json_exclusive(
        manifest,
        {
            "protocol_id": PROTOCOL_ID,
            "protocol_commit": PROTOCOL_COMMIT,
            "stage": stage,
            "stage_execution_count": 1,
            "expected_worker_packet_count": len(names),
            "archived_worker_packet_count": len(rows),
            "complete": complete,
            "fresh_process_identity_count": len(set(process_ids)),
            "fresh_database_identity_count": len(set(database_ids)),
            "files": rows,
        },
    )
    return manifest


def _write_bytes_exclusive(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _run_worker(
    root: Path,
    external: Path,
    cache: Path,
    stage: str,
    worker_kind: str,
    replay_kind: str,
    stage_root: Path,
    command_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    python = experiment_python(external)
    output = stage_root / f"{worker_kind}-{replay_kind}.json"
    database = stage_root / f"{worker_kind}-{replay_kind}.sqlite3"
    command = [
        str(python),
        "-m",
        "neuron_graph_rag.cross_encoder_precision_v3_observation",
        "worker",
        "--stage",
        stage,
        "--kind",
        worker_kind,
        "--cache",
        str(cache),
        "--database",
        str(database),
        "--output",
        str(output),
    ]
    _run_logged(command, root, _worker_environment(root, offline=True), command_rows)
    return read_json(output)


def worker(stage: str, kind: str, cache: Path, database: Path, output: Path) -> None:
    if stage not in ("development", "holdout") or kind not in (
        "baseline",
        "base",
        "v2-m3",
    ):
        raise ValueError("unknown worker stage/kind")
    if database.exists() or output.exists():
        raise FileExistsError("worker DB/output must be fresh")
    protocol = load_protocol()
    documents = _load_documents(protocol)
    model_spec = None
    if kind != "baseline":
        model_index = 0 if kind == "base" else 1
        model_spec = _rows(_mapping(protocol, "models"), "models")[model_index]
    started = time.perf_counter()
    with NeuronGraphRAG(database, config=EngineConfig()) as engine:
        _index_corpus(engine, protocol, documents)
        edge_before = canonical_sha256(_edge_state(engine))
        feedback_before = engine.store.count_feedback()
        sqlite_before = canonical_sha256(_static_sqlite_state(engine))
        cases = []
        model_runtime = _load_model(model_spec, cache) if model_spec else None
        pair_count = 0
        for query in _rows(_mapping(_mapping(protocol, "queries"), "stages"), stage):
            trace = engine.search(
                _string(query, "query"), limit=24, now=OBSERVATION_NOW
            )
            baseline_hits = [
                _baseline_hit_row(hit, rank)
                for rank, hit in enumerate(trace.hits, start=1)
            ]
            if len(baseline_hits) != 24:
                raise ValueError("NGR baseline did not rank all frozen sources")
            if model_runtime is None:
                ranked_hits = baseline_hits
            else:
                ranked_hits, case_pairs = _score_case(
                    _string(query, "query"),
                    baseline_hits[:20],
                    documents,
                    model_runtime,
                )
                pair_count += case_pairs
            cases.append(
                {
                    "case_id": _string(query, "case_id"),
                    "cohort": _string(query, "cohort"),
                    "ranked_hits": ranked_hits,
                }
            )
        edge_after = canonical_sha256(_edge_state(engine))
        feedback_after = engine.store.count_feedback()
        sqlite_after = canonical_sha256(_static_sqlite_state(engine))
        activation = canonical_sha256(_activation_state(engine))
    ranking = canonical_sha256(cases)
    process_id = sha256_bytes(
        f"{os.getpid()}:{time.time_ns()}:{database.resolve()}".encode()
    )
    payload: dict[str, Any] = {
        "process_id": process_id,
        "database_id": sha256_bytes(str(database.resolve()).encode("utf-8")),
        "cases": cases,
        "ranking_sha256": ranking,
        "activation_sha256": activation,
        "edge_sha256_before": edge_before,
        "edge_sha256_after": edge_after,
        "feedback_count_before": feedback_before,
        "feedback_count_after": feedback_after,
        "sqlite_sha256_before": sqlite_before,
        "sqlite_sha256_after": sqlite_after,
    }
    if model_spec:
        import psutil

        payload.update(
            {
                "model_id": model_spec["model_id"],
                "revision": model_spec["revision"],
                "metrics": {
                    "latency_ms": (time.perf_counter() - started) * 1000.0,
                    "peak_rss_bytes": psutil.Process().memory_info().rss,
                    "cache_bytes": _tree_bytes(cache),
                    "pair_count": pair_count,
                    "window_count": pair_count,
                },
            }
        )
    write_json_exclusive(output, payload)


def _load_model(spec: Mapping[str, Any], cache: Path) -> tuple[Any, Any, Any]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    snapshot = _snapshot_path(
        cache, _string(spec, "model_id"), _string(spec, "revision")
    )
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.float32,
    )
    model.to("cpu")
    model.eval()
    if model.training or next(model.parameters()).device.type != "cpu":
        raise ValueError("model is not in frozen CPU eval mode")
    return tokenizer, model, torch


def _score_case(
    query: str,
    baseline_hits: list[dict[str, Any]],
    documents: list[dict[str, Any]],
    runtime: tuple[Any, Any, Any],
) -> tuple[list[dict[str, Any]], int]:
    tokenizer, model, torch = runtime
    text_by_path = {row["path"]: row["text"] for row in documents}
    result = []
    pair_count = 0
    for hit in baseline_hits:
        chunks = project_passages(text_by_path[hit["source_path"]])
        scores: list[float] = []
        for offset in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[offset : offset + BATCH_SIZE]
            encoded = tokenizer(
                [query] * len(batch),
                [chunk["text"] for chunk in batch],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {key: value.to("cpu") for key, value in encoded.items()}
            with torch.inference_mode():
                logits = model(**encoded).logits.reshape(-1).to(dtype=torch.float32)
            scores.extend(float(value) for value in logits.tolist())
        pair_count += len(chunks)
        winner = min(
            index for index, value in enumerate(scores) if value == max(scores)
        )
        result.append(
            {
                **hit,
                "chunks": [
                    {
                        "chunk_index": chunk["chunk_index"],
                        "start_codepoint": chunk["start_codepoint"],
                        "end_codepoint": chunk["end_codepoint"],
                        "text_sha256": sha256_bytes(chunk["text"].encode("utf-8")),
                        "raw_logit": scores[index],
                    }
                    for index, chunk in enumerate(chunks)
                ],
                "raw_logit": scores[winner],
                "winning_chunk_index": winner,
            }
        )
    return result, pair_count


def _combine_state(
    primary: Mapping[str, Any], replay: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "fresh_database_id": primary["database_id"],
        "replay_database_id": replay["database_id"],
        "ranking_sha256": primary["ranking_sha256"],
        "replay_ranking_sha256": replay["ranking_sha256"],
        "activation_sha256": primary["activation_sha256"],
        "replay_activation_sha256": replay["activation_sha256"],
        "edge_sha256_before": primary["edge_sha256_before"],
        "edge_sha256_after": primary["edge_sha256_after"],
        "feedback_count_before": primary["feedback_count_before"],
        "feedback_count_after": primary["feedback_count_after"],
        "sqlite_sha256_before": primary["sqlite_sha256_before"],
        "sqlite_sha256_after": primary["sqlite_sha256_after"],
        "cpu_only": True,
        "offline": True,
        "fresh_process": True,
    }


def _load_documents(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    root = _path(protocol, "root")
    corpus = _mapping(protocol, "corpus")
    if _string(corpus, "commit") != SOURCE_COMMIT:
        raise ValueError("unexpected source commit")
    result = []
    for row in _rows(corpus, "documents"):
        path = _string(row, "path")
        raw = _git_bytes(root, SOURCE_COMMIT, path)
        if sha256_bytes(raw) != _string(row, "content_sha256"):
            raise ValueError("source bytes mismatch")
        result.append(
            {
                "path": path,
                "text": raw.decode("utf-8", errors="strict"),
                "sha256": row["content_sha256"],
            }
        )
    return result


def _index_corpus(
    engine: NeuronGraphRAG,
    protocol: Mapping[str, Any],
    documents: list[dict[str, Any]],
) -> None:
    repository = _string(_mapping(protocol, "corpus"), "repository")
    for row in documents:
        engine.add_document(
            f"github:{repository}:doc:{row['path']}",
            row["text"],
            metadata={
                "repository": repository,
                "commit": SOURCE_COMMIT,
                "path": row["path"],
                "content_sha256": row["sha256"],
            },
        )
    for relation in _rows(_mapping(protocol, "corpus"), "relationships"):
        engine.add_edge(
            f"github:{repository}:doc:{_string(relation, 'source_path')}",
            f"github:{repository}:doc:{_string(relation, 'target_path')}",
            _string(relation, "edge_type"),
        )


def _baseline_hit_row(hit: SearchHit, rank: int) -> dict[str, Any]:
    metadata = hit.node.metadata
    relations = []
    for path in hit.paths:
        if len(path.steps) != 1:
            continue
        step = path.steps[0]
        seed_path = _node_path_from_id(path.seed_id)
        relations.append(
            {
                "seed_path": seed_path,
                "target_path": metadata["path"],
                "edge_type": step.edge_type,
                "step_count": 1,
            }
        )
    return {
        "source_path": metadata["path"],
        "rank": rank,
        "ngr_score": hit.final_score,
        "source_sha256": metadata["content_sha256"],
        "relation_paths": relations,
    }


def _node_path_from_id(node_id: str) -> str:
    marker = ":doc:"
    if marker not in node_id:
        raise ValueError("unexpected node identity")
    return node_id.split(marker, 1)[1]


def _edge_state(engine: NeuronGraphRAG) -> list[dict[str, Any]]:
    return [
        {
            "source_id": row.source_id,
            "target_id": row.target_id,
            "edge_type": row.edge_type,
            "weight": row.weight,
            "factuality": row.factuality,
            "reinforced_count": row.reinforced_count,
        }
        for row in engine.store.list_edges()
    ]


def _activation_state(engine: NeuronGraphRAG) -> list[dict[str, Any]]:
    return [
        {"node_id": node.node_id, "activation": engine.store.activation(node.node_id)}
        for node in engine.store.list_nodes()
    ]


def _static_sqlite_state(engine: NeuronGraphRAG) -> dict[str, Any]:
    return {
        "nodes": [
            {
                "node_id": node.node_id,
                "text_sha256": sha256_bytes(node.text.encode("utf-8")),
                "metadata": node.metadata,
                "confidence": node.confidence,
            }
            for node in engine.store.list_nodes()
        ],
        "edges": _edge_state(engine),
        "feedback_count": engine.store.count_feedback(),
    }


def model_verify(cache: Path, output: Path) -> None:
    protocol = load_protocol()
    rows = []
    for spec in _rows(_mapping(protocol, "models"), "models"):
        snapshot = _snapshot_path(
            cache, _string(spec, "model_id"), _string(spec, "revision")
        )
        rows.append(_verify_snapshot(spec, snapshot))
    write_json_exclusive(
        output,
        {
            "protocol_id": PROTOCOL_ID,
            "protocol_commit": PROTOCOL_COMMIT,
            "cache_path": str(cache.resolve()),
            "models": rows,
        },
    )


def model_probe(cache: Path) -> dict[str, Any]:
    protocol = load_protocol()
    reports = []
    for spec in _rows(_mapping(protocol, "models"), "models"):
        tokenizer, model, torch = _load_model(spec, cache)
        encoded = tokenizer(
            ["probe query"] * BATCH_SIZE,
            ["probe passage"] * BATCH_SIZE,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        with torch.inference_mode():
            logits = model(**encoded).logits.reshape(-1)
        reports.append(
            {
                "model_id": spec["model_id"],
                "revision": spec["revision"],
                "device": next(model.parameters()).device.type,
                "dtype": str(next(model.parameters()).dtype),
                "training": model.training,
                "logit_count": int(logits.numel()),
            }
        )
        del model
    return {
        "offline": os.environ.get("HF_HUB_OFFLINE") == "1"
        and os.environ.get("TRANSFORMERS_OFFLINE") == "1",
        "trust_remote_code": False,
        "batch_size": BATCH_SIZE,
        "forward_inference_count": len(reports),
        "models": reports,
    }


def _verify_model_report(
    protocol: Mapping[str, Any], report: Mapping[str, Any], cache: Path
) -> None:
    if (
        report.get("protocol_id") != PROTOCOL_ID
        or report.get("protocol_commit") != PROTOCOL_COMMIT
        or report.get("cache_path") != str(cache.resolve())
    ):
        raise ValueError("model verification report identity mismatch")
    frozen = _rows(_mapping(protocol, "models"), "models")
    rows = _rows(report, "models")
    if len(rows) != len(frozen):
        raise ValueError("model verification report cardinality mismatch")
    for spec, row in zip(frozen, rows, strict=True):
        snapshot = Path(_string(row, "snapshot_path"))
        if row != _verify_snapshot(spec, snapshot):
            raise ValueError("model verification report is not reproducible")


def _verify_snapshot(spec: Mapping[str, Any], snapshot: Path) -> dict[str, Any]:
    files = []
    for frozen in _rows(spec, "required_files"):
        path = snapshot / _string(frozen, "path")
        raw = path.read_bytes()
        if len(raw) != frozen.get("size"):
            raise ValueError(f"model artifact size mismatch: {path.name}")
        lfs = frozen.get("lfs_sha256")
        if lfs is not None:
            actual = sha256_bytes(raw)
            if actual != lfs:
                raise ValueError(f"model LFS hash mismatch: {path.name}")
            hash_kind = "lfs_sha256"
        else:
            actual = hashlib.sha1(
                f"blob {len(raw)}\0".encode("ascii") + raw, usedforsecurity=False
            ).hexdigest()
            if actual != frozen.get("git_blob_id"):
                raise ValueError(f"model git blob mismatch: {path.name}")
            hash_kind = "git_blob_id"
        files.append(
            {
                "path": frozen["path"],
                "size": len(raw),
                "hash_kind": hash_kind,
                "verified_hash": actual,
            }
        )
    return {
        "model_id": spec["model_id"],
        "revision": spec["revision"],
        "license": spec["license"],
        "snapshot_path": str(snapshot.resolve()),
        "files": files,
    }


def _snapshot_path(cache: Path, model_id: str, revision: str) -> Path:
    repository = "models--" + model_id.replace("/", "--")
    path = cache / repository / "snapshots" / revision
    if not path.is_dir():
        raise FileNotFoundError(f"model snapshot unavailable: {model_id}@{revision}")
    return path


def _run_logged(
    command: Sequence[str],
    cwd: Path,
    environment: Mapping[str, str],
    rows: list[dict[str, Any]],
) -> str:
    completed = subprocess.run(
        list(command),
        cwd=cwd,
        env=dict(environment),
        check=False,
        capture_output=True,
    )
    command_list = list(command)
    row = {
        "command": command_list,
        "command_sha256": canonical_sha256(command_list),
        "returncode": completed.returncode,
        "stdout_sha256": sha256_bytes(completed.stdout),
        "stderr_sha256": sha256_bytes(completed.stderr),
    }
    rows.append(row)
    if completed.returncode != 0:
        message = completed.stderr.decode("utf-8", errors="replace")[-4000:]
        raise RuntimeError(f"command failed ({completed.returncode}): {message}")
    return completed.stdout.decode("utf-8", errors="strict")


def _full_test_commands(python: Path, root: Path) -> list[list[str]]:
    if os.name != "nt":
        return [
            [
                str(python),
                "-m",
                "unittest",
                "discover",
                "-s",
                "tests",
                "-p",
                "test_*.py",
            ]
        ]
    suite = unittest.defaultTestLoader.discover(
        str(root / "tests"), pattern="test_*.py"
    )
    test_ids = [
        test_id if test_id.startswith("tests.") else f"tests.{test_id}"
        for test_id in _test_ids(suite)
    ]
    return [[str(python), "-m", "unittest", test_id] for test_id in test_ids]


def _test_ids(suite: unittest.TestSuite) -> list[str]:
    result = []
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            result.extend(_test_ids(test))
        else:
            result.append(test.id())
    return result


def _worker_environment(root: Path, *, offline: bool) -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src")
    environment["PYTHONUTF8"] = "1"
    if offline:
        environment["HF_HUB_OFFLINE"] = "1"
        environment["TRANSFORMERS_OFFLINE"] = "1"
    else:
        environment.pop("HF_HUB_OFFLINE", None)
        environment.pop("TRANSFORMERS_OFFLINE", None)
    return environment


def _tree_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _git_bytes(root: Path, commit: str, path: str) -> bytes:
    completed = subprocess.run(
        ["git", "show", f"{commit}:{path}"], cwd=root, check=False, capture_output=True
    )
    if completed.returncode != 0:
        raise ValueError(completed.stderr.decode("utf-8", errors="replace").strip())
    return completed.stdout


def _git_output(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments], cwd=root, check=False, capture_output=True
    )
    if completed.returncode != 0:
        raise ValueError(completed.stderr.decode("utf-8", errors="replace").strip())
    return completed.stdout.decode("ascii", errors="strict").strip()


def _mapping(value: Mapping[str, Any], key: str) -> dict[str, Any]:
    row = value.get(key)
    if not isinstance(row, dict):
        raise TypeError(f"{key} must be an object")
    return row


def _rows(value: Mapping[str, Any], key: str) -> list[dict[str, Any]]:
    rows = value.get(key)
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise TypeError(f"{key} must be an array of objects")
    return rows


def _string(value: Mapping[str, Any], key: str) -> str:
    row = value.get(key)
    if not isinstance(row, str) or not row:
        raise TypeError(f"{key} must be a non-empty string")
    return row


def _path(value: Mapping[str, Any], key: str) -> Path:
    row = value.get(key)
    if not isinstance(row, Path):
        raise TypeError(f"{key} must be a path")
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run frozen cross-encoder observation once"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "verify-preflight", "run"):
        command = commands.add_parser(name)
        command.add_argument(
            "--external-root", type=Path, default=default_external_root()
        )
        command.add_argument("--model-cache", type=Path, default=default_model_cache())
    verify = commands.add_parser("model-verify")
    verify.add_argument("--cache", type=Path, required=True)
    verify.add_argument("--output", type=Path, required=True)
    probe = commands.add_parser("model-probe")
    probe.add_argument("--cache", type=Path, required=True)
    worker_command = commands.add_parser("worker")
    worker_command.add_argument("--stage", required=True)
    worker_command.add_argument("--kind", required=True)
    worker_command.add_argument("--cache", type=Path, required=True)
    worker_command.add_argument("--database", type=Path, required=True)
    worker_command.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    if arguments.command == "preflight":
        result = preflight(ROOT, arguments.external_root, arguments.model_cache)
    elif arguments.command == "verify-preflight":
        result = verify_preflight(ROOT, arguments.external_root, arguments.model_cache)
    elif arguments.command == "run":
        result = run_conditional(ROOT, arguments.external_root, arguments.model_cache)
    elif arguments.command == "model-verify":
        model_verify(arguments.cache, arguments.output)
        result = {"status": "verified"}
    elif arguments.command == "model-probe":
        result = model_probe(arguments.cache)
    else:
        worker(
            arguments.stage,
            arguments.kind,
            arguments.cache,
            arguments.database,
            arguments.output,
        )
        result = {"status": "completed"}
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
