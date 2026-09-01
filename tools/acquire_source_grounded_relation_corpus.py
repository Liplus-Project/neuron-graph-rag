from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from neuron_graph_rag.github_source import GitHubSnapshot
from neuron_graph_rag.source_grounded_relation_observation import (
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
    source = manifest["source"]
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
    return {
        "schema_version": 1,
        "repository": repository,
        "commit": commit,
        "documents": documents,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Acquire or verify the fixed v1 relation corpus."
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    payload = build(args.root)
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    if args.verify:
        if args.output.read_bytes() != encoded:
            raise SystemExit("fixed corpus bytes do not match git objects")
        snapshot = GitHubSnapshot.from_mapping(payload)
        if not extract_source_grounded_relations(snapshot):
            raise SystemExit("fixed corpus exposes no in-corpus relations")
        return 0
    if args.output.exists():
        raise SystemExit("refusing to overwrite corpus")
    args.output.write_bytes(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
