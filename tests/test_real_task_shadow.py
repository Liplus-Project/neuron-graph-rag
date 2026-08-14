from __future__ import annotations

import copy
import hashlib
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from neuron_graph_rag import EngineConfig
from neuron_graph_rag.real_task_shadow import (
    ARM_IDS,
    PROTOCOL_ID,
    build_placeholder_packet,
    capture_packet,
    create_placeholder_snapshot,
    load_effective_registry,
    probe_placeholder,
    read_canonical_json,
    replay_packet,
    replay_packets,
    replay_registry,
    sha256_file,
    validate_packet,
    validate_result,
    verify_packet_against_snapshot,
    verify_result_against_packets,
    verify_result_against_registry,
    write_json_exclusive,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
PLACEHOLDER = FIXTURES / "real_task_shadow_v1.placeholder.json"
MANIFEST = FIXTURES / "real_task_shadow_v1.manifest.json"


class RealTaskShadowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = read_canonical_json(PLACEHOLDER)

    def _packet(self, root: Path) -> tuple[Path, dict[str, object]]:
        snapshot = root / "placeholder.db"
        create_placeholder_snapshot(self.fixture, snapshot)
        return snapshot, build_placeholder_packet(self.fixture, snapshot)

    def test_placeholder_runs_actual_writer_replay_and_verifier_round_trip(self) -> None:
        self.assertEqual(
            probe_placeholder(PLACEHOLDER),
            {
                "protocol_id": PROTOCOL_ID,
                "placeholder_only": True,
                "packet_round_trip": True,
                "replay_round_trip": True,
                "exclusive_writer_verified": True,
            },
        )

    def test_pending_packet_replays_both_frozen_arms_without_observed_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            snapshot, packet = self._packet(Path(directory))
            source_hash = sha256_file(snapshot)
            result = replay_packet(packet, snapshot)
            self.assertEqual(sha256_file(snapshot), source_hash)
            self.assertEqual(result["snapshot_sha256"], source_hash)
        self.assertEqual(tuple(result["arms"]), ARM_IDS)
        self.assertTrue(result["replay"]["deterministic"])
        self.assertIsNone(result["arms"]["used_q3_s1"]["packets"][0]["outcome"])
        self.assertIsNone(result["arms"]["confirmed_r05_s1"]["packets"][0]["outcome"])
        self.assertTrue(result["arms"]["used_q3_s1"]["packets"][0]["idempotency_replay"])

    def test_confirmed_requires_objective_evidence_and_only_confirmed_arm_reinforces(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            snapshot, packet = self._packet(Path(directory))
            packet["outcome"] = {
                "status": "confirmed",
                "summary": "placeholder test passed",
                "external_ref": "https://example.invalid/placeholder/check/1",
                "evidence": [
                    {
                        "kind": "test_passed",
                        "node_id": "placeholder-target",
                        "external_ref": "https://example.invalid/placeholder/check/1",
                        "target_commit": "0000000000000000000000000000000000000000",
                        "details": {"command": "placeholder-test", "exit_code": 0},
                    }
                ],
            }
            result = replay_packet(packet, snapshot)
        used_changed = result["comparison"]["final_changed_edge_count"]["used_q3_s1"]
        confirmed_changed = result["comparison"]["final_changed_edge_count"]["confirmed_r05_s1"]
        self.assertEqual(used_changed, 0)
        self.assertGreater(confirmed_changed, 0)

        invalid = copy.deepcopy(packet)
        invalid["outcome"]["evidence"] = []
        with self.assertRaisesRegex(ValueError, "positive objective evidence"):
            validate_packet(invalid)

    def test_snapshot_hash_node_source_content_and_path_mismatches_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot, packet = self._packet(root)

            bad_hash = copy.deepcopy(packet)
            bad_hash["database_snapshot"]["sha256"] = "sha256:" + "0" * 64
            with self.assertRaisesRegex(ValueError, "snapshot hash mismatch"):
                verify_packet_against_snapshot(bad_hash, snapshot)

            absent = copy.deepcopy(packet)
            absent["retrieval"]["candidates"][1]["node_id"] = "placeholder-absent"
            with self.assertRaisesRegex(ValueError, "captured node is absent"):
                verify_packet_against_snapshot(absent, snapshot)

            bad_source = copy.deepcopy(packet)
            bad_source["retrieval"]["candidates"][0]["source_url"] = "https://example.invalid/wrong"
            with self.assertRaisesRegex(ValueError, "source identity mismatch"):
                verify_packet_against_snapshot(bad_source, snapshot)

            bad_content = copy.deepcopy(packet)
            bad_content["retrieval"]["candidates"][0]["content_sha256"] = "sha256:" + "f" * 64
            with self.assertRaisesRegex(ValueError, "content hash mismatch"):
                verify_packet_against_snapshot(bad_content, snapshot)

            bad_path = copy.deepcopy(packet)
            bad_path["retrieval"]["credited_path"]["steps"][0]["edge_type"] = "wrong"
            with self.assertRaisesRegex(ValueError, "credited path mismatch"):
                verify_packet_against_snapshot(bad_path, snapshot)

    def test_registry_is_append_only_sequential_and_corrections_supersede(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, packet = self._packet(root)
            registry = root / "registry"
            first = capture_packet(packet, registry)
            original = first.read_bytes()
            with self.assertRaises(FileExistsError):
                capture_packet(packet, registry)
            self.assertEqual(first.read_bytes(), original)

            skipped = copy.deepcopy(packet)
            skipped["packet_id"] = "placeholder-packet-0003"
            skipped["slot"] = 3
            with self.assertRaisesRegex(ValueError, "sequential slot 2"):
                capture_packet(skipped, registry)

            correction = copy.deepcopy(packet)
            correction["packet_id"] = "placeholder-packet-0001-correction"
            correction["supersedes_packet_id"] = packet["packet_id"]
            correction["outcome"] = {
                "status": "corrected",
                "summary": "placeholder correction",
                "external_ref": "https://example.invalid/placeholder/correction/1",
                "evidence": [
                    {
                        "kind": "rollback_or_correction",
                        "node_id": "placeholder-target",
                        "external_ref": "https://example.invalid/placeholder/correction/1",
                        "target_commit": "0000000000000000000000000000000000000000",
                        "details": {"reason": "placeholder correction"},
                    }
                ],
            }
            correction_path = capture_packet(correction, registry)
            self.assertNotEqual(first, correction_path)
            self.assertEqual(read_canonical_json(correction_path)["slot"], 1)
            self.assertEqual(
                [item["packet_id"] for item in load_effective_registry(registry)],
                [correction["packet_id"]],
            )

    def test_registry_replay_accumulates_q3_evidence_in_one_clone_per_arm(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot, first = self._packet(root)
            packets = []
            registry = root / "registry"
            for slot in range(1, 4):
                packet = copy.deepcopy(first)
                packet["packet_id"] = f"placeholder-packet-{slot:04d}"
                packet["slot"] = slot
                packet["task"]["task_url"] = f"https://example.invalid/placeholder/task/{slot}"
                packet["captured_at"] = f"2000-01-01T00:00:0{slot}Z"
                capture_packet(packet, registry)
                packets.append(packet)

            direct = replay_packets(packets, snapshot)
            registered = replay_registry(registry, snapshot)
            self.assertEqual(registered, direct)
            used_packets = direct["arms"]["used_q3_s1"]["packets"]
            self.assertEqual(
                [
                    row["source_use"]["feedback"]["evidence"][0]["count"]
                    for row in used_packets
                ],
                [1, 2, 3],
            )
            self.assertEqual(
                direct["comparison"]["final_changed_edge_count"]["used_q3_s1"],
                2,
            )
            self.assertEqual(
                direct["comparison"]["final_changed_edge_count"]["confirmed_r05_s1"],
                0,
            )

            reversed_packets = list(reversed(packets))
            with self.assertRaisesRegex(ValueError, "sequential slot order"):
                replay_packets(reversed_packets, snapshot)
            wrong_snapshot = copy.deepcopy(packets)
            wrong_snapshot[1]["database_snapshot"]["sha256"] = "sha256:" + "0" * 64
            with self.assertRaisesRegex(ValueError, "snapshot hash mismatch"):
                replay_packets(wrong_snapshot, snapshot)

    def test_registry_lock_prevents_concurrent_different_packets_in_one_slot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, first = self._packet(root)
            second = copy.deepcopy(first)
            second["packet_id"] = "placeholder-competing-packet"
            registry = root / "registry"
            barrier = threading.Barrier(2)

            def write(packet: dict[str, object]) -> object:
                barrier.wait()
                try:
                    return capture_packet(packet, registry)
                except (FileExistsError, ValueError) as error:
                    return error

            with ThreadPoolExecutor(max_workers=2) as pool:
                outcomes = list(pool.map(write, (first, second)))
            self.assertEqual(sum(isinstance(item, Path) for item in outcomes), 1)
            self.assertEqual(sum(isinstance(item, Exception) for item in outcomes), 1)
            self.assertEqual(len(list(registry.glob("*.json"))), 1)
            self.assertFalse((registry / ".registry.lock").exists())

    def test_cli_replays_registry_and_exactly_verifies_stored_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot, packet = self._packet(root)
            registry = root / "registry"
            capture_packet(packet, registry)
            output = root / "batch.result.json"
            environment = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
            tool = ROOT / "tools" / "run_real_task_shadow.py"
            replay = subprocess.run(
                [
                    sys.executable,
                    str(tool),
                    "replay",
                    "--registry-dir",
                    str(registry),
                    "--snapshot",
                    str(snapshot),
                    "--output",
                    str(output),
                ],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )
            self.assertEqual(replay.returncode, 0, replay.stderr)
            verify = subprocess.run(
                [
                    sys.executable,
                    str(tool),
                    "verify-result",
                    "--result",
                    str(output),
                    "--registry-dir",
                    str(registry),
                    "--snapshot",
                    str(snapshot),
                ],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )
            self.assertEqual(verify.returncode, 0, verify.stderr)
            self.assertEqual(verify.stdout.strip(), "result verified")

    def test_exclusive_result_writer_and_verifier_reject_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot, packet = self._packet(root)
            result = replay_packet(packet, snapshot)
            output = root / "result.json"
            write_json_exclusive(output, result)
            validate_result(read_canonical_json(output))
            verify_result_against_packets(read_canonical_json(output), [packet], snapshot)
            original = output.read_bytes()
            with self.assertRaises(FileExistsError):
                write_json_exclusive(output, {"replacement": True})
            self.assertEqual(output.read_bytes(), original)

            tampered = copy.deepcopy(result)
            tampered["arms"]["used_q3_s1"]["packets"][0]["rank_delta"] += 1
            with self.assertRaisesRegex(ValueError, "rank delta mismatch"):
                validate_result(tampered)

            forged_receipt = copy.deepcopy(result)
            forged_receipt["arms"]["used_q3_s1"]["packets"][0]["source_use"]["events"][0]["changed"] = False
            validate_result(forged_receipt)
            with self.assertRaisesRegex(ValueError, "exact semantic replay"):
                verify_result_against_packets(forged_receipt, [packet], snapshot)

            registry = root / "registry"
            capture_packet(packet, registry)
            verify_result_against_registry(result, registry, snapshot)

    def test_result_free_artifacts_are_canonical_hashed_and_registered_outputs_absent(self) -> None:
        manifest = read_canonical_json(MANIFEST)
        self.assertEqual(manifest["protocol_id"], PROTOCOL_ID)
        self.assertTrue(manifest["result_free"])
        for relative, expected in manifest["artifact_sha256"].items():
            actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            self.assertEqual(actual, expected, relative)
        for relative in manifest["registered_outputs"].values():
            path = ROOT / relative
            self.assertTrue(not path.exists() or not any(path.iterdir()), relative)
        for name in (
            "real_task_shadow_v1.packet-schema.json",
            "real_task_shadow_v1.evidence-schema.json",
            "real_task_shadow_v1.result-schema.json",
            "real_task_shadow_v1.inclusion-rule.json",
            "real_task_shadow_v1.placeholder.json",
        ):
            read_canonical_json(FIXTURES / name)

    def test_core_defaults_and_mcp_registration_are_unchanged(self) -> None:
        config = EngineConfig()
        self.assertFalse(hasattr(config, "confirmed_outcome_reinforcement"))
        self.assertNotIn(
            "real_task_shadow",
            (ROOT / "src" / "neuron_graph_rag_mcp" / "server.py").read_text(encoding="utf-8"),
        )


if __name__ == "__main__":
    unittest.main()
