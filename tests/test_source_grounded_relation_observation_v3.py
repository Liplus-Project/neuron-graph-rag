from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from neuron_graph_rag import source_grounded_relation_observation as v1
from neuron_graph_rag import source_grounded_relation_observation_v3 as observation

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "a" * 40
SHARED_HASH = "b" * 64
VALID_PHASES = {
    "result-free",
    "development-claimed",
    "development-partial",
    "development-closed",
    "holdout-eligible",
    "holdout-claimed",
    "holdout-partial",
    "holdout-completed",
}


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _copy(root: Path, relative: str) -> None:
    source = ROOT / relative
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


class SourceGroundedRelationObservationV3Tests(unittest.TestCase):
    def _protocol_root(self) -> tempfile.TemporaryDirectory[str]:
        directory = tempfile.TemporaryDirectory()
        root = Path(directory.name)
        for relative in observation.protocol_file_inventory():
            _copy(root, relative)
        return directory

    def _protocol(self, root: Path) -> dict[str, object]:
        return observation.load_protocol(root, require_result_free=False)

    def _claim(self, root: Path, stage: str) -> None:
        protocol = self._protocol(root)
        path = root / protocol["manifest"]["claims"][stage]
        v1._exclusive_write(
            path,
            {
                "protocol_id": observation.PROTOCOL_ID,
                "protocol_commit": COMMIT,
                "stage": stage,
                "attempt": 1,
                "retry_count": 0,
            },
        )

    def _packet(
        self,
        protocol: dict[str, object],
        stage: str,
        arm: str,
        run: str,
        *,
        candidate_passes: bool,
    ) -> dict[str, object]:
        queries = protocol["queries"]["stages"][stage]
        gold = {
            row["case_id"]: row for row in protocol["gold"]["stages"][stage]
        }
        documents = {document.path: document for document in protocol["corpus"].documents}
        cases = []
        for query in queries:
            row = gold[query["case_id"]]
            hits = []
            seeds = []
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

    def _persist_packets(
        self,
        root: Path,
        stage: str,
        count: int,
        *,
        candidate_passes: bool,
    ) -> None:
        protocol = self._protocol(root)
        identities = [
            (arm, run) for arm in observation.ARMS for run in observation.RUNS
        ]
        for arm, run in identities[:count]:
            with observation._runtime_scope():
                registered = v1._raw_packet_paths(protocol, stage)[
                    (stage, arm, run)
                ]
            if registered.exists():
                continue
            packet = self._packet(
                protocol,
                stage,
                arm,
                run,
                candidate_passes=candidate_passes,
            )
            with observation._runtime_scope():
                v1._persist_worker_packet(
                    protocol,
                    packet,
                    stage=stage,
                    arm=arm,
                    run=run,
                    protocol_commit=COMMIT,
                )

    def _finalize(self, root: Path, stage: str) -> dict[str, object]:
        protocol = self._protocol(root)
        with observation._runtime_scope():
            result = v1.finalize_stage(protocol, stage, COMMIT, SHARED_HASH)
        output = root / protocol["manifest"]["outputs"][stage]
        v1._exclusive_write(output, result)
        return result

    def _git(self, root: Path, *args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        ).stdout.strip()

    def _freeze_repository(
        self,
    ) -> tuple[tempfile.TemporaryDirectory[str], Path, str]:
        directory = tempfile.TemporaryDirectory()
        root = Path(directory.name)
        manifest = _json(observation.MANIFEST_PATH)
        v3_paths = {
            observation.MANIFEST_PATH.relative_to(ROOT).as_posix(),
            manifest["audit"],
            *manifest["artifact_sha256"],
        }
        inventory = observation.protocol_file_inventory()
        for relative in inventory:
            if relative not in v3_paths:
                _copy(root, relative)
        self._git(root, "init", "-b", "main")
        self._git(root, "config", "user.name", "Synthetic Probe")
        self._git(root, "config", "user.email", "probe@example.invalid")
        self._git(root, "add", ".")
        self._git(root, "commit", "-m", "baseline")
        for relative in inventory:
            if relative in v3_paths:
                _copy(root, relative)
        self._git(root, "add", ".")
        self._git(root, "commit", "-m", "freeze v3")
        commit = self._git(root, "rev-parse", "HEAD")
        self._git(root, "update-ref", "refs/remotes/origin/main", commit)
        return directory, root, commit

    def test_repository_root_accepts_current_lifecycle_phase(self) -> None:
        result = observation.audit_repository_lifecycle()
        self.assertEqual(result["status"], "observation-registry-valid")
        self.assertIn(result["phase"], VALID_PHASES)
        if result["phase"] == "result-free":
            frozen = observation.audit_result_free()
            self.assertEqual(frozen["status"], "result-free-protocol-valid")
            self.assertEqual(frozen["observed_result_count"], 0)

    def test_v1_v2_freeze_artifacts_remain_byte_identical(self) -> None:
        for version in ("v1", "v2"):
            manifest_path = (
                ROOT
                / "tests"
                / "fixtures"
                / f"github_source_grounded_relation_{version}.manifest.json"
            )
            manifest = _json(manifest_path)
            for relative, expected in manifest["artifact_sha256"].items():
                actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
                self.assertEqual(actual, expected, relative)

    def test_v2_corpus_query_gold_gate_are_inherited_as_exact_bytes(self) -> None:
        manifest = _json(observation.MANIFEST_PATH)
        inherited = manifest["inherited_protocol_artifacts"]
        for name in ("corpus", "queries", "gold", "gate"):
            item = inherited[name]
            self.assertRegex(item["path"], r"_v2\.(?:corpus|queries|gold|gate)\.json$")
            self.assertEqual(
                hashlib.sha256((ROOT / item["path"]).read_bytes()).hexdigest(),
                item["sha256"],
            )

    def test_runner_verifies_effective_manifest_and_runtime_scope(self) -> None:
        directory, root, commit = self._freeze_repository()
        with directory:
            protocol = observation.load_protocol(root)
            shared = root / "synthetic-shared-database.bin"
            shared.write_bytes(b"hash-only; never opened as sqlite")
            output = root / protocol["manifest"]["outputs"]["development"]

            def synthetic_worker(
                worker_protocol: dict[str, object],
                stage: str,
                arm: str,
                database: Path,
                *,
                protocol_commit: str,
                run: str,
            ) -> dict[str, object]:
                self.assertNotIn("gold", worker_protocol)
                self.assertFalse(database.exists())
                self.assertEqual(stage, "development")
                self.assertEqual(protocol_commit, commit)
                packet = self._packet(
                    protocol,
                    stage,
                    arm,
                    run,
                    candidate_passes=True,
                )
                packet["protocol_commit"] = protocol_commit
                return packet

            with patch.object(v1, "run_worker", side_effect=synthetic_worker):
                result = observation.run_stage(
                    "development", commit, shared, output, root
                )
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["selected_arm"], observation.ARMS[1])
            self.assertEqual(
                observation.audit_repository_lifecycle(root)["phase"],
                "holdout-eligible",
            )
            self.assertEqual(shared.read_bytes(), b"hash-only; never opened as sqlite")

    def test_placeholder_lifecycle_accepts_every_registered_phase(self) -> None:
        with self._protocol_root() as directory:
            root = Path(directory)
            self.assertEqual(
                observation.audit_repository_lifecycle(root)["phase"], "result-free"
            )
            self._claim(root, "development")
            self.assertEqual(
                observation.audit_repository_lifecycle(root)["phase"],
                "development-claimed",
            )
            self._persist_packets(
                root, "development", 2, candidate_passes=True
            )
            self.assertEqual(
                observation.audit_repository_lifecycle(root)["phase"],
                "development-partial",
            )
            self._persist_packets(
                root, "development", 4, candidate_passes=True
            )
            self._finalize(root, "development")
            self.assertEqual(
                observation.audit_repository_lifecycle(root)["phase"],
                "holdout-eligible",
            )
            self._claim(root, "holdout")
            self.assertEqual(
                observation.audit_repository_lifecycle(root)["phase"],
                "holdout-claimed",
            )
            self._persist_packets(root, "holdout", 3, candidate_passes=False)
            self.assertEqual(
                observation.audit_repository_lifecycle(root)["phase"],
                "holdout-partial",
            )
            self._persist_packets(root, "holdout", 4, candidate_passes=False)
            self._finalize(root, "holdout")
            self.assertEqual(
                observation.audit_repository_lifecycle(root)["phase"],
                "holdout-completed",
            )

    def test_failed_development_keeps_holdout_closed(self) -> None:
        with self._protocol_root() as directory:
            root = Path(directory)
            self._claim(root, "development")
            self._persist_packets(
                root, "development", 4, candidate_passes=False
            )
            result = self._finalize(root, "development")
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["selected_arm"], observation.ARMS[0])
            self.assertEqual(
                observation.audit_repository_lifecycle(root)["phase"],
                "development-closed",
            )
            self._claim(root, "holdout")
            with self.assertRaisesRegex(ValueError, "without candidate eligibility"):
                observation.audit_repository_lifecycle(root)

    def test_registry_rejects_gap_tamper_and_overwrite(self) -> None:
        with self._protocol_root() as directory:
            root = Path(directory)
            self._claim(root, "development")
            with self.assertRaises(FileExistsError):
                self._claim(root, "development")
            protocol = self._protocol(root)
            packet = self._packet(
                protocol,
                "development",
                observation.ARMS[1],
                "primary",
                candidate_passes=True,
            )
            with observation._runtime_scope():
                v1._persist_worker_packet(
                    protocol,
                    packet,
                    stage="development",
                    arm=observation.ARMS[1],
                    run="primary",
                    protocol_commit=COMMIT,
                )
            with self.assertRaisesRegex(ValueError, "append-only prefix"):
                observation.audit_repository_lifecycle(root)

        with self._protocol_root() as directory:
            root = Path(directory)
            self._claim(root, "development")
            self._persist_packets(
                root, "development", 1, candidate_passes=True
            )
            protocol = self._protocol(root)
            raw = next(
                path
                for path in (
                    root / relative
                    for relative in protocol["manifest"]["raw_packets"][
                        "development"
                    ][observation.ARMS[0]].values()
                )
                if path.exists()
            )
            payload = _json(raw)
            payload["cases"][0]["unexpected"] = True
            raw.write_bytes(v1._encoded(payload))
            with self.assertRaisesRegex(ValueError, "case fields"):
                observation.audit_repository_lifecycle(root)


if __name__ == "__main__":
    unittest.main()
