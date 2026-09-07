from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from neuron_graph_rag import github_retrieval_parity_v3 as parity

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "a" * 40


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8", errors="strict"))


def _copy(root: Path, relative: str) -> None:
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / relative, target)


class GitHubRetrievalParityV3Tests(unittest.TestCase):
    def _protocol_root(self) -> tempfile.TemporaryDirectory[str]:
        directory = tempfile.TemporaryDirectory()
        root = Path(directory.name)
        for relative in parity.protocol_file_inventory():
            _copy(root, relative)
        return directory

    def _capture(
        self, protocol: dict[str, object], stage: str, *, populated: bool
    ) -> dict[str, object]:
        corpus = protocol["corpus"]
        documents = {document.path: document for document in corpus.documents}
        queries = protocol["queries"]["stages"][stage]
        gold = {row["case_id"]: row for row in protocol["gold"]["stages"][stage]}
        defaults = protocol["queries"]["request_defaults"]
        rows = []
        for query in queries:
            gold_row = gold[query["case_id"]]
            result_rows = []
            fetched_rows = []
            graph_rows = []
            if populated:
                for index, source_id in enumerate(gold_row["expected_source_ids"]):
                    path = parity._path_from_source_id(corpus.repository, source_id)
                    document = documents[path]
                    vector_id = f"synthetic:{stage}:{query['case_id']}:{index}"
                    result_rows.append(
                        {
                            "vector_id": vector_id,
                            "repo": corpus.repository,
                            "type": "doc",
                            "doc_path": path,
                        }
                    )
                    content = parity._indexed_content(path, document.content)
                    fetched_rows.append(
                        {
                            "vector_id": vector_id,
                            "repo": corpus.repository,
                            "type": "doc",
                            "doc_path": path,
                            "content": content,
                            "content_chars": parity._js_length(content),
                            "content_truncated": parity._js_length(
                                path + "\n\n" + document.content
                            )
                            >= 8000,
                        }
                    )
                    if query["cohort"] == "relation_linked":
                        graph_rows.append(
                            {
                                "repo": corpus.repository,
                                "type": "doc",
                                "doc_path": path,
                                "graph_hop": 1,
                            }
                        )
            rows.append(
                {
                    "case_id": query["case_id"],
                    "request": {"query": query["query"], **defaults},
                    "raw_search": {
                        "mode": "search",
                        "filters_unmatched": [],
                        "results": result_rows,
                        "graph_results": graph_rows,
                    },
                    "raw_stored_content": {
                        "mode": "fetch",
                        "content_source": "index",
                        "content_max_chars": 8000,
                        "not_found": [],
                        "results": fetched_rows,
                    },
                }
            )
        return {
            "protocol_id": parity.PROTOCOL_ID,
            "protocol_commit": COMMIT,
            "stage": stage,
            "captured_at": "2030-01-01T00:00:00Z",
            "service": "github-rag-mcp",
            "tool": "search",
            "cases": rows,
        }

    def _write_capture_claim(
        self, root: Path, stage: str, *, populated: bool
    ) -> tuple[dict[str, object], dict[str, object]]:
        protocol = parity.load_protocol(root, require_result_free=False)
        capture = self._capture(protocol, stage, populated=populated)
        capture_path = parity._stage_path(protocol, stage, "capture")
        parity.write_json_exclusive(capture_path, capture)
        parity.write_json_exclusive(
            parity._stage_path(protocol, stage, "claim"),
            {
                "protocol_id": parity.PROTOCOL_ID,
                "protocol_commit": COMMIT,
                "stage": stage,
                "capture_sha256": hashlib.sha256(capture_path.read_bytes()).hexdigest(),
                "one_time_claim": True,
            },
        )
        return protocol, capture

    def _result(
        self,
        protocol: dict[str, object],
        capture: dict[str, object],
        stage: str,
        *,
        populated: bool,
    ) -> dict[str, object]:
        queries, gold = parity._stage_cases(protocol, stage)
        github = parity._validate_capture(capture, stage, COMMIT, protocol, queries)
        corpus = protocol["corpus"]
        documents = {document.path: document for document in corpus.documents}
        cases = []
        for query in queries:
            row = gold[query["case_id"]]
            hits = []
            if populated:
                for rank, source_id in enumerate(row["expected_source_ids"], start=1):
                    path = parity._path_from_source_id(corpus.repository, source_id)
                    document = documents[path]
                    relation = query["cohort"] == "relation_linked"
                    hits.append(
                        {
                            "rank": rank,
                            "source_id": source_id,
                            "repository": corpus.repository,
                            "path": path,
                            "source_url": document.source_url,
                            "commit": corpus.commit,
                            "blob_sha": document.blob_sha,
                            "content_sha256": document.content_sha256,
                            "explanation": {
                                "node_id": source_id,
                                "ranks": {"graph": rank if relation else None},
                                "paths": [{"steps": [{"edge_type": "links_to"}]}]
                                if relation
                                else [],
                            },
                        }
                    )
            cases.append(
                parity._case_metrics(
                    query,
                    row,
                    github[query["case_id"]],
                    hits,
                    parity._top_k(protocol),
                )
            )
        cohorts = parity._cohort_metrics(cases)
        gates = parity._evaluate_gates(cases, cohorts, True)
        all_pass = all(item["passed"] for item in gates)
        capture_path = parity._stage_path(protocol, stage, "capture")
        return {
            "protocol_id": parity.PROTOCOL_ID,
            "protocol_commit": COMMIT,
            "stage": stage,
            "status": "passed" if all_pass else "failed",
            "failure_code": None if all_pass else "hard-gate-failed",
            "capture_sha256": hashlib.sha256(capture_path.read_bytes()).hexdigest(),
            "protocol_hashes": parity._frozen_hashes(protocol["manifest"]),
            "source": {
                "repository": corpus.repository,
                "commit": corpus.commit,
                "fingerprint": corpus.fingerprint(),
                "document_count": len(corpus.documents),
            },
            "request_contract": dict(protocol["queries"]["request_defaults"]),
            "cases": cases,
            "cohorts": cohorts,
            "deterministic_replay": {
                "passed": True,
                "normalized_result_sha256": "b" * 64,
            },
            "resources": {"latency_hard_gate": False, "synthetic": True},
            "gates": gates,
            "all_hard_gates_pass": all_pass,
            "raw_github_rag_mcp_capture": capture,
            "interpretation_ja": "登録queryを実行しないsynthetic lifecycle state。",
        }

    def test_repository_root_is_result_free(self) -> None:
        result = parity.audit_result_free()
        self.assertEqual(result["status"], "result-free-protocol-valid")
        self.assertEqual(result["source_document_count"], 93)
        self.assertEqual(result["development_case_count"], 5)
        self.assertEqual(result["holdout_case_count"], 5)
        self.assertEqual(result["registered_observation_count"], 0)

    def test_source_snapshot_is_exact_pinned_full_markdown_surface(self) -> None:
        protocol = parity.load_protocol()
        corpus = protocol["corpus"]
        self.assertEqual(corpus.repository, "Liplus-Project/liplus-language")
        self.assertEqual(corpus.commit, "51623e200ab6128cf59cf5b5ac7d8115c31268d6")
        self.assertEqual(
            protocol["corpus_mapping"]["tree_sha"],
            "034101e7ff123f91d089640e0685514a3a890159",
        )
        self.assertEqual(len(corpus.documents), 93)
        for document in corpus.documents:
            self.assertTrue(document.path.endswith(".md"))
            self.assertFalse(document.path.startswith("/"))
            self.assertRegex(document.blob_sha, r"^[0-9a-f]{40}$")
            self.assertEqual(
                hashlib.sha256(document.content.encode("utf-8")).hexdigest(),
                document.content_sha256,
            )

    def test_queryless_surface_audit_matches_the_frozen_tree(self) -> None:
        protocol = parity.load_protocol()
        corpus_paths = sorted(document.path for document in protocol["corpus"].documents)
        surface = protocol["surface_audit"]
        self.assertTrue(surface["request_contract"]["query_omitted"])
        self.assertEqual(surface["indexed_document_count"], 93)
        self.assertEqual(surface["unique_path_count"], 93)
        self.assertEqual(surface["duplicate_path_count"], 0)
        self.assertEqual(surface["missing_path_count"], 0)
        self.assertEqual(surface["out_of_surface_path_count"], 0)
        self.assertEqual(surface["error_count"], 0)
        self.assertEqual(surface["indexed_paths"], corpus_paths)

    def test_split_and_predecessor_identities_are_disjoint(self) -> None:
        protocol = parity.load_protocol()
        stages = protocol["gold"]["stages"]

        def identities(stage: str) -> set[str]:
            result = set()
            for row in stages[stage]:
                for key in (
                    "expected_source_ids",
                    "forbidden_source_ids",
                    "protected_safe_source_ids",
                ):
                    result.update(row[key])
                if "relation_seed_source_id" in row:
                    result.add(row["relation_seed_source_id"])
            return result

        self.assertTrue(identities("development").isdisjoint(identities("holdout")))
        case_ids = {
            row["case_id"]
            for stage in protocol["queries"]["stages"].values()
            for row in stage
        }
        predecessor_ids = {
            case_id
            for predecessor in protocol["manifest"]["predecessors"]
            for case_id in predecessor["case_ids"]
        }
        self.assertTrue(case_ids.isdisjoint(predecessor_ids))
        query_hashes = {
            hashlib.sha256(row["query"].encode("utf-8")).hexdigest()
            for stage in protocol["queries"]["stages"].values()
            for row in stage
        }
        predecessor_query_hashes = {
            item
            for predecessor in protocol["manifest"]["predecessors"]
            for item in predecessor["query_text_sha256"]
        }
        self.assertTrue(query_hashes.isdisjoint(predecessor_query_hashes))
        predecessor_gold_sources = {
            item
            for predecessor in protocol["manifest"]["predecessors"]
            for item in predecessor["gold_source_ids"]
        }
        self.assertTrue(
            identities("development").isdisjoint(predecessor_gold_sources)
        )
        self.assertTrue(identities("holdout").isdisjoint(predecessor_gold_sources))

        for stage in parity.STAGES:
            queries = {
                row["case_id"]: row for row in protocol["queries"]["stages"][stage]
            }
            gold = {row["case_id"]: row for row in protocol["gold"]["stages"][stage]}
            case = next(
                row
                for row in queries.values()
                if row["cohort"] == "over_exclusion_control"
            )
            intent = parity.decompose_exclusion_intent(case["query"])
            self.assertTrue(intent.exclusion_clauses)
            for source_id in gold[case["case_id"]]["protected_safe_source_ids"]:
                decision = parity._candidate_exclusion_decision(
                    source_id,
                    intent,
                    protocol["corpus"],
                    {
                        document.path: document
                        for document in protocol["corpus"].documents
                    },
                )
                self.assertTrue(decision["accepted"])
                self.assertCountEqual(
                    decision["negated_mentions"], intent.exclusion_clauses
                )
            for source_id in gold[case["case_id"]]["forbidden_source_ids"]:
                decision = parity._candidate_exclusion_decision(
                    source_id,
                    intent,
                    protocol["corpus"],
                    {
                        document.path: document
                        for document in protocol["corpus"].documents
                    },
                )
                self.assertFalse(decision["accepted"])
                self.assertTrue(decision["matched_exclusions"])

    def test_frozen_predecessor_and_v3_artifact_hashes_match(self) -> None:
        protocol = parity.load_protocol()
        self.assertEqual(
            protocol["manifest"]["lifecycle_contract"]["freeze_identity_scope"],
            "protocol-artifacts-and-production-runtime",
        )
        self.assertEqual(
            tuple(protocol["manifest"]["runtime_sha256"]), parity.RUNTIME_PATHS
        )
        for predecessor in protocol["manifest"]["predecessors"]:
            for registry in ("identity_sha256", "observation_sha256"):
                for relative, expected in predecessor[registry].items():
                    self.assertEqual(
                        hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(),
                        expected,
                        relative,
                    )
        parity.verify_frozen_artifacts(protocol)

        with self._protocol_root() as directory:
            root = Path(directory)
            runtime = root / parity.RUNTIME_PATHS[0]
            runtime.write_text(
                runtime.read_text(encoding="utf-8") + "\n# drift\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "frozen artifact hash mismatch"):
                parity.load_protocol(root)

    def test_whole_module_lifecycle_accepts_closed_and_eligible_states(self) -> None:
        with self._protocol_root() as directory:
            root = Path(directory)
            self.assertEqual(
                parity.audit_repository_lifecycle(root)["phase"], "result-free"
            )
            protocol = parity.load_protocol(root)
            capture = self._capture(protocol, "development", populated=False)
            parity.write_json_exclusive(
                parity._stage_path(protocol, "development", "capture"), capture
            )
            self.assertEqual(
                parity.audit_repository_lifecycle(root)["phase"],
                "development-captured",
            )
            capture_path = parity._stage_path(protocol, "development", "capture")
            parity.write_json_exclusive(
                parity._stage_path(protocol, "development", "claim"),
                {
                    "protocol_id": parity.PROTOCOL_ID,
                    "protocol_commit": COMMIT,
                    "stage": "development",
                    "capture_sha256": hashlib.sha256(
                        capture_path.read_bytes()
                    ).hexdigest(),
                    "one_time_claim": True,
                },
            )
            self.assertEqual(
                parity.audit_repository_lifecycle(root)["phase"],
                "development-claimed",
            )
            result = self._result(protocol, capture, "development", populated=False)
            parity.write_json_exclusive(
                parity._stage_path(protocol, "development", "result"), result
            )
            self.assertEqual(
                parity.audit_repository_lifecycle(root)["phase"],
                "development-closed",
            )

        with self._protocol_root() as directory:
            root = Path(directory)
            protocol, capture = self._write_capture_claim(
                root, "development", populated=True
            )
            result = self._result(protocol, capture, "development", populated=True)
            self.assertTrue(result["all_hard_gates_pass"])
            parity.write_json_exclusive(
                parity._stage_path(protocol, "development", "result"), result
            )
            self.assertEqual(
                parity.audit_repository_lifecycle(root)["phase"],
                "holdout-eligible",
            )

    def test_exclusive_writer_and_holdout_gate_fail_closed(self) -> None:
        with self._protocol_root() as directory:
            root = Path(directory)
            protocol, capture = self._write_capture_claim(
                root, "development", populated=False
            )
            capture_path = parity._stage_path(protocol, "development", "capture")
            with self.assertRaises(FileExistsError):
                parity.write_json_exclusive(capture_path, {"replacement": True})
            result = self._result(protocol, capture, "development", populated=False)
            parity.write_json_exclusive(
                parity._stage_path(protocol, "development", "result"), result
            )
            with self.assertRaisesRegex(RuntimeError, "hard gates did not pass"):
                parity._assert_stage_can_start("holdout", protocol)

    def test_tampered_result_and_split_overlap_are_rejected(self) -> None:
        with self._protocol_root() as directory:
            root = Path(directory)
            protocol, capture = self._write_capture_claim(
                root, "development", populated=True
            )
            result = self._result(protocol, capture, "development", populated=True)
            result["cases"][0]["ngr"]["mrr"] = 0.0
            parity.write_json_exclusive(
                parity._stage_path(protocol, "development", "result"), result
            )
            with self.assertRaisesRegex(ValueError, "do not recompute"):
                parity.audit_repository_lifecycle(root)

        with self._protocol_root() as directory:
            root = Path(directory)
            protocol, capture = self._write_capture_claim(
                root, "development", populated=True
            )
            result = self._result(protocol, capture, "development", populated=True)
            result["cases"][0]["ngr"]["hits"][0]["source_url"] = "tampered"
            parity.write_json_exclusive(
                parity._stage_path(protocol, "development", "result"), result
            )
            with self.assertRaisesRegex(ValueError, "source provenance mismatch"):
                parity.audit_repository_lifecycle(root)

        protocol = parity.load_protocol()
        tampered = copy.deepcopy(protocol)
        tampered["gold"]["stages"]["holdout"][0]["expected_source_ids"] = list(
            protocol["gold"]["stages"]["development"][0]["expected_source_ids"]
        )
        with self.assertRaisesRegex(ValueError, "identities must be disjoint"):
            parity.validate_protocol(tampered)

        tampered = copy.deepcopy(protocol)
        over_query = next(
            row
            for row in tampered["queries"]["stages"]["development"]
            if row["cohort"] == "over_exclusion_control"
        )
        over_query["query"] = "operations specification for PR and release procedures"
        with self.assertRaisesRegex(ValueError, "explicit exclusion clause"):
            parity.validate_protocol(tampered)

        tampered = copy.deepcopy(protocol)
        over_gold = next(
            row
            for row in tampered["gold"]["stages"]["development"]
            if row["cohort"] == "over_exclusion_control"
        )
        over_gold["expected_source_ids"], over_gold["forbidden_source_ids"] = (
            over_gold["forbidden_source_ids"],
            over_gold["expected_source_ids"],
        )
        over_gold["protected_safe_source_ids"] = list(over_gold["expected_source_ids"])
        with self.assertRaisesRegex(ValueError, "candidate-side-negation premise"):
            parity.validate_protocol(tampered)

    def test_runner_audit_executes_no_registered_query(self) -> None:
        result = subprocess.run(
            [sys.executable, "tools/run_github_retrieval_parity_v3.py", "--audit"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        payload = json.loads(result.stdout)
        self.assertEqual(payload["status"], "result-free-protocol-valid")
        self.assertEqual(payload["performance"], "not assessed")


if __name__ == "__main__":
    unittest.main()
