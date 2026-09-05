from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from neuron_graph_rag import source_grounded_relation_observation as v1
from neuron_graph_rag import source_grounded_relation_observation_v3 as observation

COMMIT = "a" * 40
SHARED_HASH = "b" * 64
TEST_MODULE = "tests.test_source_grounded_relation_observation_v3"


def _copy_probe_root(source: Path, target: Path) -> None:
    shutil.copytree(
        source / "src",
        target / "src",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    for relative in observation.protocol_file_inventory(source):
        source_path = source / relative
        target_path = target / relative
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)
    init = source / "tests" / "__init__.py"
    if init.exists():
        target_init = target / "tests" / "__init__.py"
        target_init.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(init, target_init)


def _claim(root: Path, protocol: dict[str, object]) -> None:
    path = root / protocol["manifest"]["claims"]["development"]
    v1._exclusive_write(
        path,
        {
            "protocol_id": observation.PROTOCOL_ID,
            "protocol_commit": COMMIT,
            "stage": "development",
            "attempt": 1,
            "retry_count": 0,
        },
    )


def _packet(
    protocol: dict[str, object], arm: str, run: str, *, candidate_passes: bool
) -> dict[str, object]:
    stage = "development"
    queries = protocol["queries"]["stages"][stage]
    gold = {row["case_id"]: row for row in protocol["gold"]["stages"][stage]}
    documents = {document.path: document for document in protocol["corpus"].documents}
    cases = []
    for query in queries:
        row = gold[query["case_id"]]
        seeds = []
        hits = []
        if (
            candidate_passes
            and arm == observation.ARMS[1]
            and row["cohort"] == "relation_linked"
        ):
            seed = row["relation_seed_path"]
            expected = row["expected_path"]
            document = documents[expected]
            seeds = [seed]
            hits = [
                {
                    "path": expected,
                    "source_url": document.source_url,
                    "content_sha256": document.content_sha256,
                    "sparse_score": 0.0,
                    "dense_score": 0.0,
                    "entry_score": 0.0,
                    "graph_activation": 0.0,
                    "final_score": 0.0,
                    "relation_paths": [
                        {
                            "seed_path": seed,
                            "target_path": expected,
                            "steps": [
                                {
                                    "source_path": seed,
                                    "target_path": expected,
                                    "edge_type": row["relation_edge_type"],
                                }
                            ],
                        }
                    ],
                }
            ]
        cases.append(
            {
                "case_id": query["case_id"],
                "query": query["query"],
                "referenced_seed_paths": seeds,
                "hits": hits,
            }
        )
    return {
        "protocol_id": observation.PROTOCOL_ID,
        "protocol_commit": COMMIT,
        "stage": stage,
        "arm": arm,
        "run": run,
        "attempt": 1,
        "retry_count": 0,
        "cases": cases,
    }


def _seed_development(root: Path, *, candidate_passes: bool) -> str:
    protocol = observation.load_protocol(root, require_result_free=False)
    _claim(root, protocol)
    for arm in observation.ARMS:
        for run in observation.RUNS:
            packet = _packet(
                protocol, arm, run, candidate_passes=candidate_passes
            )
            with observation._runtime_scope():
                v1._persist_worker_packet(
                    protocol,
                    packet,
                    stage="development",
                    arm=arm,
                    run=run,
                    protocol_commit=COMMIT,
                )
    with observation._runtime_scope():
        result = v1.finalize_stage(
            protocol, "development", COMMIT, SHARED_HASH
        )
    output = root / protocol["manifest"]["outputs"]["development"]
    v1._exclusive_write(output, result)
    return str(observation.audit_repository_lifecycle(root)["phase"])


def _run_frozen_module(root: Path) -> dict[str, object]:
    environment = dict(os.environ)
    python_path = str(root / "src")
    if environment.get("PYTHONPATH"):
        python_path += os.pathsep + environment["PYTHONPATH"]
    environment["PYTHONPATH"] = python_path
    completed = subprocess.run(
        [sys.executable, "-m", "unittest", TEST_MODULE, "-v"],
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "frozen v3 test module failed\n"
            + completed.stdout
            + completed.stderr
        )
    return {
        "returncode": completed.returncode,
        "test_module": TEST_MODULE,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def probe(source_root: Path = observation.ROOT) -> dict[str, object]:
    outcomes: dict[str, object] = {}
    with tempfile.TemporaryDirectory(prefix="ngr-v3-result-free-") as directory:
        root = Path(directory)
        _copy_probe_root(source_root, root)
        phase = observation.audit_repository_lifecycle(root)["phase"]
        if phase != "result-free":
            raise RuntimeError(f"result-free probe root entered phase {phase}")
        outcomes["result_free"] = {
            "phase": phase,
            **_run_frozen_module(root),
        }
    observed: dict[str, object] = {}
    for name, candidate_passes, expected_phase in (
        ("development_closed", False, "development-closed"),
        ("holdout_eligible", True, "holdout-eligible"),
    ):
        with tempfile.TemporaryDirectory(prefix=f"ngr-v3-{name}-") as directory:
            root = Path(directory)
            _copy_probe_root(source_root, root)
            phase = _seed_development(root, candidate_passes=candidate_passes)
            if phase != expected_phase:
                raise RuntimeError(
                    f"synthetic {name} root entered phase {phase}"
                )
            observed[name] = {
                "phase": phase,
                **_run_frozen_module(root),
            }
    outcomes["synthetic_post_observation"] = observed
    return {
        "status": "whole-module-two-state-probe-valid",
        "protocol_id": observation.PROTOCOL_ID,
        "outcomes": outcomes,
        "real_queries_executed": 0,
        "shared_database_opened": False,
        "persistent_artifacts_created": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the frozen v3 test module in isolated lifecycle states."
    )
    parser.add_argument("--root", type=Path, default=observation.ROOT)
    args = parser.parse_args()
    print(json.dumps(probe(args.root.resolve()), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
