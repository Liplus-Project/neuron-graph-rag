from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import re
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import full_corpus_rerank_oracle as v1
from . import structural_representation_length_bias_diagnostic as design

PROTOCOL_ID = design.PROTOCOL_ID
ROOT = Path(__file__).resolve().parents[2]
MANIFEST = design.MANIFEST
SCHEMA = Path("tests/fixtures/structural_representation_length_bias_ablation_v1.schema.json")
QUERY = Path("tests/fixtures/full_corpus_rerank_oracle_v2.query.json")
GOLD = Path("tests/fixtures/full_corpus_rerank_oracle_v2.gold.json")
CORPUS = design.CORPUS
MODEL_REGISTRY = Path("tests/fixtures/github_cross_encoder_precision_v8.models.json")
CLAIM = Path("tests/evidence/structural_representation_length_bias_ablation_v1/development.claim.json")
RESULT = Path("tests/evidence/structural_representation_length_bias_ablation_v1/development.observed.json")
ERROR = Path("tests/evidence/structural_representation_length_bias_ablation_v1/development.error.json")
ATTESTATION = Path("preflight.attestation.json")
MODEL_KINDS = ("base", "v2-m3")
REPRESENTATIONS = ("body", "structural")
ARMS = ("body_max", "structural_max", "body_nlme", "structural_nlme")
MAX_LENGTH = 512

canonical_json_bytes = v1.canonical_json_bytes
sha256_bytes = v1.sha256_bytes
sha256_file = v1.sha256_file
read_json = v1.read_json
write_json_exclusive = v1.write_json_exclusive


def _manifest(root: Path = ROOT) -> dict[str, Any]:
    value = read_json(root / MANIFEST)
    if value.get("protocol_id") != PROTOCOL_ID or value.get("status") != "frozen_pre_registered_execution":
        raise ValueError("frozen manifest identity mismatch")
    if [row.get("arm_id") for row in value.get("arms", [])] != list(ARMS):
        raise ValueError("frozen arm order mismatch")
    if value["aggregations"]["normalized_log_mean_exp"]["temperature"] != 1.0:
        raise ValueError("NLME temperature mismatch")
    if value["ranking"]["cutoff"] != 20:
        raise ValueError("cutoff mismatch")
    artifacts = value.get("artifact_sha256")
    if not isinstance(artifacts, dict):
        raise ValueError("artifact hash registry missing")
    for relative, digest in artifacts.items():
        if not re.fullmatch(r"[0-9a-f]{64}", str(digest)) or sha256_file(root / relative) != digest:
            raise ValueError(f"frozen artifact changed: {relative}")
    return value


def _query(root: Path) -> str:
    value = read_json(root / QUERY)
    if set(value) != {"schema_version", "protocol_id", "case_id", "query"}:
        raise ValueError("query bundle fields mismatch")
    return str(value["query"])


def _gold(root: Path) -> str:
    value = read_json(root / GOLD)
    if set(value) != {"schema_version", "protocol_id", "case_id", "expected_source_id"}:
        raise ValueError("gold bundle fields mismatch")
    return str(value["expected_source_id"])


def _documents(root: Path) -> list[dict[str, Any]]:
    return v1._documents(root)


def _model_specs(root: Path) -> list[dict[str, Any]]:
    manifest = _manifest(root)
    rows = read_json(root / MODEL_REGISTRY)["models"]
    result = []
    for expected in manifest["models"]:
        found = [row for row in rows if row["model_id"] == expected["model_id"] and row["revision"] == expected["revision"]]
        if len(found) != 1:
            raise ValueError("pinned model missing")
        result.append({**found[0], "kind": expected["kind"]})
    return result


def _registered_files(root: Path) -> set[str]:
    return {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file() and "__pycache__" not in path.parts}


def _validate_tree(root: Path, stage: str) -> dict[str, str]:
    manifest = _manifest(root)
    expected = set(manifest["worker_registered_files"])
    if stage in {"worker", "finalizer"}:
        expected.add(CLAIM.as_posix())
    if stage == "finalizer":
        expected.add(GOLD.as_posix())
    actual = _registered_files(root)
    if actual != expected:
        raise ValueError(f"registered allowlist mismatch: missing={sorted(expected-actual)!r} extra={sorted(actual-expected)!r}")
    for forbidden in manifest["forbidden_registered_paths"]:
        if (root / forbidden).exists():
            raise ValueError(f"forbidden registered path exists: {forbidden}")
    return {relative: sha256_file(root / relative) for relative in sorted(expected)}


def _rankdata(values: Sequence[float | int]) -> list[float]:
    return design._rankdata(values)


def _spearman(rows: Sequence[Mapping[str, Any]]) -> float:
    return design._pearson(_rankdata([int(row["chunk_count"]) for row in rows]), _rankdata([float(row["document_score"]) for row in rows]))


def _nlme(scores: Sequence[float]) -> float:
    maximum = max(scores)
    return maximum + math.log(sum(math.exp(value - maximum) for value in scores) / len(scores))


def _score_representation(query: str, documents: Sequence[Mapping[str, Any]], runtime: tuple[Any, Any, Any], representation: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tokenizer, model, torch = runtime
    output = []
    prefix_codepoints: list[int] = []
    prefix_tokens: list[int] = []
    exceeding = 0
    for document in documents:
        body = str(document["content"])
        chunks = v1.project_passages(body, window=480, overlap=80)
        passages = []
        chunk_rows = []
        for index, chunk in enumerate(chunks):
            prefix = "" if representation == "body" else design.structural_prefix(str(document["path"]), body, int(chunk["start_codepoint"]))
            passage = prefix + str(chunk["text"])
            prefix_token_length = len(tokenizer.encode(prefix, add_special_tokens=False)) if prefix else 0
            pair_length = len(tokenizer.encode(query, passage, add_special_tokens=True, truncation=False))
            prefix_codepoints.append(len(prefix)); prefix_tokens.append(prefix_token_length)
            exceeding += int(pair_length > MAX_LENGTH)
            passages.append(passage)
            chunk_rows.append({
                "chunk_index": index,
                "start_codepoint": chunk["start_codepoint"], "end_codepoint": chunk["end_codepoint"],
                "body_chunk_sha256": sha256_bytes(str(chunk["text"]).encode("utf-8")),
                "body_chunk_codepoint_length": len(str(chunk["text"])),
                "prefix_codepoint_length": len(prefix), "prefix_token_length_without_special_tokens": prefix_token_length,
                "pair_token_length_before_truncation": pair_length, "pair_exceeds_512_before_truncation": pair_length > MAX_LENGTH,
            })
        scores = []
        for offset in range(0, len(passages), 8):
            batch = passages[offset:offset+8]
            encoded = tokenizer([query]*len(batch), batch, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
            with torch.inference_mode():
                logits = model(**encoded).logits.reshape(-1).to(dtype=torch.float32).tolist()
            scores.extend(float(value) for value in logits)
        for row, score in zip(chunk_rows, scores):
            row["raw_logit"] = score
        output.append({"source_id": document["source_id"], "path": document["path"], "character_count": len(body), "chunk_count": len(chunks), "chunks": chunk_rows})
    def stats(values: Sequence[int]) -> dict[str, float | int]:
        return {"min": min(values), "max": max(values), "mean": sum(values)/len(values)}
    return output, {"pair_count": len(prefix_codepoints), "pairs_exceeding_512_before_truncation": exceeding, "prefix_codepoint_length": stats(prefix_codepoints), "prefix_token_length": stats(prefix_tokens)}


def _arms(representations: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    arms = []
    by_id = {row["representation"]: row for row in representations}
    for arm_id in ARMS:
        representation, aggregation = arm_id.rsplit("_", 1)
        rows = []
        for document in by_id[representation]["documents"]:
            scores = [float(chunk["raw_logit"]) for chunk in document["chunks"]]
            score = max(scores) if aggregation == "max" else _nlme(scores)
            rows.append({"source_id": document["source_id"], "path": document["path"], "chunk_count": document["chunk_count"], "document_score": score})
        rows.sort(key=lambda row: (-row["document_score"], row["source_id"]))
        for rank, row in enumerate(rows, 1): row["rank"] = rank
        arms.append({"arm_id": arm_id, "representation": representation, "aggregation": aggregation, "documents": rows, "spearman_chunk_count_vs_document_score": _spearman(rows), "ranking_sha256": sha256_bytes(canonical_json_bytes(rows))})
    return arms


def preflight(root: Path, cache: Path, runtime_root: Path, source_commit: str) -> dict[str, Any]:
    if platform.system() != "Linux" or platform.machine() != "x86_64" or not v1._network_is_disabled():
        raise RuntimeError("preflight requires offline Linux x86_64")
    hashes = _validate_tree(root, "preflight")
    if (root / GOLD).exists(): raise ValueError("gold must be absent from preflight")
    documents = _documents(root)
    for document in documents:
        chunks = v1.project_passages(document["content"], window=480, overlap=80)
        for chunk in chunks: design.structural_prefix(document["path"], document["content"], chunk["start_codepoint"])
    models = []
    for spec in _model_specs(root):
        model_hashes = v1._verify_model_files(spec, cache)
        runtime = v1._load_model(spec, cache); tokenizer, model, torch = runtime
        encoded = tokenizer(["synthetic query"], ["synthetic passage"], return_tensors="pt")
        with torch.inference_mode(): value = float(model(**encoded).logits.reshape(-1)[0])
        if not math.isfinite(value): raise ValueError("non-finite synthetic logit")
        models.append({"kind": spec["kind"], "model_files_sha256": model_hashes}); del runtime; gc.collect()
    if runtime_root.exists(): raise FileExistsError("runtime root must be fresh")
    runtime_root.mkdir(parents=True)
    payload = {"protocol_id": PROTOCOL_ID, "status": "preflight_valid", "source_commit": source_commit, "registered_query_execution_count": 0, "gold_present": False, "holdout_bearing_input_file_count": 0, "corpus_document_count": len(documents), "registered_source_sha256": hashes, "models": models}
    write_json_exclusive(runtime_root / ATTESTATION, payload); return payload


def claim(root: Path, runtime_root: Path, source_commit: str) -> dict[str, Any]:
    hashes = _validate_tree(root, "claim"); pre = read_json(runtime_root / ATTESTATION)
    if pre.get("source_commit") != source_commit or pre.get("registered_source_sha256") != hashes: raise ValueError("preflight binding mismatch")
    if any((root/path).exists() for path in (CLAIM, RESULT, ERROR)): raise FileExistsError("append-only evidence exists")
    payload = {"protocol_id": PROTOCOL_ID, "source_commit": source_commit, "manifest_sha256": sha256_file(root/MANIFEST), "preflight_attestation_sha256": sha256_file(runtime_root/ATTESTATION), "registered_query_count": 1, "retry_count": 0, "model_worker_count": 2, "gold_present_in_workers": False}
    write_json_exclusive(root/CLAIM, payload); return payload


def worker(root: Path, cache: Path, runtime_root: Path, kind: str) -> dict[str, Any]:
    _validate_tree(root, "worker")
    if (root/GOLD).exists(): raise ValueError("worker gold must be absent")
    specs = {row["kind"]: row for row in _model_specs(root)}; spec = specs[kind]
    query = _query(root); documents = _documents(root); started = time.perf_counter(); runtime = v1._load_model(spec, cache)
    reps=[]
    try:
        for representation in REPRESENTATIONS:
            rep_started=time.perf_counter(); docs, truncation = _score_representation(query, documents, runtime, representation)
            reps.append({"representation": representation, "runtime_seconds": time.perf_counter()-rep_started, "truncation": truncation, "documents": docs})
    finally: del runtime; gc.collect()
    body = {row["source_id"]: row for row in reps[0]["documents"]}; structural = {row["source_id"]: row for row in reps[1]["documents"]}
    increased=0
    for source_id in body:
        for left,right in zip(body[source_id]["chunks"], structural[source_id]["chunks"]):
            if (left["start_codepoint"],left["end_codepoint"],left["body_chunk_sha256"]) != (right["start_codepoint"],right["end_codepoint"],right["body_chunk_sha256"]): raise ValueError("body chunk identity drift")
            increased += int(not left["pair_exceeds_512_before_truncation"] and right["pair_exceeds_512_before_truncation"])
    import resource
    payload={"kind":kind,"model_id":spec["model_id"],"revision":spec["revision"],"model_files_sha256":v1._verify_model_files(spec,cache),"dependencies":v1._dependency_versions(),"runtime_seconds":time.perf_counter()-started,"peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,"representations":reps,"arms":_arms(reps),"body_at_or_below_512_but_structural_above_512_count":increased,"structural_increases_truncation":increased>0}
    write_json_exclusive(runtime_root/f"{kind}.json",payload); return {"kind":kind,"status":"worker_complete"}


def finalize(root: Path, runtime_root: Path, source_commit: str) -> dict[str, Any]:
    _validate_tree(root,"finalizer"); expected=_gold(root); workers=[read_json(runtime_root/f"{kind}.json") for kind in MODEL_KINDS]
    for worker_payload in workers:
        for arm in worker_payload["arms"]:
            match=next(row for row in arm["documents"] if row["source_id"]==expected); arm["expected_source_rank"]=match["rank"]; arm["expected_source_within_cutoff"]=match["rank"]<=20; arm["top_distractor"]=next(row for row in arm["documents"] if row["source_id"]!=expected)
    by_model={row["kind"]:{arm["arm_id"]:arm for arm in row["arms"]} for row in workers}
    primary=[arm for arm in ARMS[1:] if all(by_model[k][arm]["expected_source_rank"]<=20 for k in MODEL_KINDS)]
    directional=[arm for arm in ARMS[1:] if all(by_model[k][arm]["expected_source_rank"]<by_model[k]["body_max"]["expected_source_rank"] for k in MODEL_KINDS)]
    attenuation={rep: all(abs(by_model[k][f"{rep}_nlme"]["spearman_chunk_count_vs_document_score"])<abs(by_model[k][f"{rep}_max"]["spearman_chunk_count_vs_document_score"]) for k in MODEL_KINDS) for rep in REPRESENTATIONS}
    payload={"schema_version":1,"protocol_id":PROTOCOL_ID,"source_commit":source_commit,"expected_source_id":expected,"corpus_document_count":93,"cutoff":20,"primary_success":bool(primary),"primary_success_arms":primary,"directional_evidence_arms":directional,"length_bias_attenuation":attenuation,"environment":{"os":"linux","architecture":platform.machine(),"network":"disabled","fresh_runtime":True,"holdout_bearing_input_file_count":0,"shared_database_open_count":0},"inputs_sha256":{path.as_posix():sha256_file(root/path) for path in (MANIFEST,SCHEMA,QUERY,GOLD,CORPUS,MODEL_REGISTRY)},"models":workers}
    payload["payload_sha256"]=sha256_bytes(canonical_json_bytes(payload)); write_json_exclusive(root/RESULT,payload)
    return {"status":"observed_valid","primary_success":payload["primary_success"],"primary_success_arms":primary,"directional_evidence_arms":directional,"length_bias_attenuation":attenuation,"ranks":{k:{arm:by_model[k][arm]["expected_source_rank"] for arm in ARMS} for k in MODEL_KINDS},"payload_sha256":payload["payload_sha256"]}


def record_error(root: Path, source_commit: str, message: str) -> dict[str, Any]:
    if not (root/CLAIM).exists() or (root/RESULT).exists(): raise FileExistsError("error evidence requires claim and no result")
    payload={"protocol_id":PROTOCOL_ID,"source_commit":source_commit,"retry_count":0,"error":message}
    write_json_exclusive(root/ERROR,payload); return payload


def audit(root: Path=ROOT) -> dict[str, Any]:
    design.audit(root); _manifest(root); evidence={"claim":(root/CLAIM).exists(),"result":(root/RESULT).exists(),"error":(root/ERROR).exists()}
    if evidence["result"] and evidence["error"]: raise ValueError("result and error coexist")
    result=read_json(root/RESULT) if evidence["result"] else None
    if result and result.get("payload_sha256") != sha256_bytes(canonical_json_bytes({k:v for k,v in result.items() if k!="payload_sha256"})): raise ValueError("result payload hash mismatch")
    return {"protocol_id":PROTOCOL_ID,"status":"observed_valid" if result else "result_free_frozen","registered_query_execution_count":1 if result else 0,"holdout_bearing_input_file_count":0,"evidence":evidence,"primary_success":result.get("primary_success") if result else None}


def main(argv: Sequence[str]|None=None)->int:
    parser=argparse.ArgumentParser(); sub=parser.add_subparsers(dest="command",required=True)
    for name in ("audit","probe"): sub.add_parser(name).add_argument("--root",type=Path,default=ROOT)
    p=sub.add_parser("preflight"); p.add_argument("--root",type=Path,required=True); p.add_argument("--cache",type=Path,required=True); p.add_argument("--runtime-root",type=Path,required=True); p.add_argument("--source-commit",required=True)
    p=sub.add_parser("claim"); p.add_argument("--root",type=Path,required=True); p.add_argument("--runtime-root",type=Path,required=True); p.add_argument("--source-commit",required=True)
    p=sub.add_parser("worker"); p.add_argument("--root",type=Path,required=True); p.add_argument("--cache",type=Path,required=True); p.add_argument("--runtime-root",type=Path,required=True); p.add_argument("--kind",choices=MODEL_KINDS,required=True)
    p=sub.add_parser("finalize"); p.add_argument("--root",type=Path,required=True); p.add_argument("--runtime-root",type=Path,required=True); p.add_argument("--source-commit",required=True)
    p=sub.add_parser("record-error"); p.add_argument("--root",type=Path,required=True); p.add_argument("--source-commit",required=True); p.add_argument("--message",required=True)
    args=parser.parse_args(argv)
    if args.command in {"audit","probe"}: result=audit(args.root)
    elif args.command=="preflight": result=preflight(args.root,args.cache,args.runtime_root,args.source_commit)
    elif args.command=="claim": result=claim(args.root,args.runtime_root,args.source_commit)
    elif args.command=="worker": result=worker(args.root,args.cache,args.runtime_root,args.kind)
    elif args.command=="finalize": result=finalize(args.root,args.runtime_root,args.source_commit)
    else: result=record_error(args.root,args.source_commit,args.message)
    sys.stdout.buffer.write(canonical_json_bytes(result)+b"\n"); return 0


if __name__=="__main__": raise SystemExit(main())
