from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from neuron_graph_rag.github_source import GitHubSnapshot
from neuron_graph_rag.source_grounded_relation_observation_v3 import (
    MANIFEST_PATH,
    ROOT,
    extract_source_grounded_relations,
)


def _git(root: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, stdout=subprocess.PIPE
    ).stdout


def build(root: Path) -> dict[str, object]:
    manifest = json.loads(
        (root / MANIFEST_PATH.relative_to(ROOT)).read_text(encoding="utf-8")
    )
    inherited = manifest["inherited_protocol_artifacts"]["corpus"]
    predecessor = json.loads(
        (root / manifest["inherited_protocol_manifest"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    source = predecessor["source"]
    repository = source["repository"]
    commit = source["commit"]
    documents = []
    for path in source["paths"]:
        raw = _git(root, "show", f"{commit}:{path}")
        content = raw.decode("utf-8", errors="strict")
        blob_sha = _git(root, "rev-parse", f"{commit}:{path}").decode().strip()
        documents.append(
            {
                "path": path,
                "blob_sha": blob_sha,
                "content": content,
                "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
                "source_url": f"https://github.com/{repository}/blob/{commit}/{path}",
            }
        )
    payload = {
        "schema_version": 1,
        "repository": repository,
        "commit": commit,
        "documents": documents,
    }
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode(
        "utf-8"
    )
    if hashlib.sha256(encoded).hexdigest() != inherited["sha256"]:
        raise ValueError("rebuilt corpus does not match the inherited v2 identity")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify the exact v2 corpus inherited by relation v3."
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if not args.verify:
        raise SystemExit("v3 inherits v2 corpus bytes; acquisition is verify-only")
    manifest = json.loads(
        (args.root / MANIFEST_PATH.relative_to(ROOT)).read_text(encoding="utf-8")
    )
    inherited = manifest["inherited_protocol_artifacts"]["corpus"]
    expected_output = args.root / inherited["path"]
    if args.output.resolve() != expected_output.resolve():
        raise SystemExit("output must be the registered inherited v2 corpus path")
    payload = build(args.root)
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode(
        "utf-8"
    )
    if args.output.read_bytes() != encoded:
        raise SystemExit("inherited corpus bytes do not match git objects")
    snapshot = GitHubSnapshot.from_mapping(payload)
    if not extract_source_grounded_relations(snapshot):
        raise SystemExit("inherited corpus exposes no in-corpus relations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
