from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

PROTOCOL_ID = "github-ngr-cross-encoder-precision-v23-real-tasks"
REPOSITORY = "Liplus-Project/neuron-graph-rag"
REPOSITORY_URL = "https://github.com/Liplus-Project/neuron-graph-rag"
SOURCE_COMMIT = "79b456d620f1b37746669ea1fe1e57c385f5e4ed"
SOURCE_PATHS = (
    "docs/Home.md",
    "docs/feedback-adaptation-experiment.md",
    "docs/historical-source-verification.md",
    "docs/sibling-relation-feedback-normalization.md",
    "docs/mcp-feedback-stabilization-settings.md",
    "docs/decision-wiki-pilot-migration.md",
    "docs/feedback-adaptation-reproduction-experiment.md",
    "docs/cross-encoder-precision-observation-v20.md",
    "docs/intent-aware-observation-engine.md",
    "src/neuron_graph_rag/d1_fixture.py",
    "src/neuron_graph_rag/evaluation.py",
    "src/neuron_graph_rag/config_provenance.py",
)
RELATIONSHIPS = (
    {
        "source_path": "docs/Home.md",
        "target_path": "docs/decision-wiki-pilot-migration.md",
        "edge_type": "informs",
    },
    {
        "source_path": "docs/feedback-adaptation-experiment.md",
        "target_path": "docs/feedback-adaptation-reproduction-experiment.md",
        "edge_type": "informs",
    },
    {
        "source_path": "docs/sibling-relation-feedback-normalization.md",
        "target_path": "docs/mcp-feedback-stabilization-settings.md",
        "edge_type": "informs",
    },
    {
        "source_path": "docs/Home.md",
        "target_path": "src/neuron_graph_rag/evaluation.py",
        "edge_type": "informs",
    },
)


def _git_blob(root: Path, relative: str) -> bytes:
    completed = subprocess.run(
        ["git", "show", f"{SOURCE_COMMIT}:{relative}"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return completed.stdout


def acquire(root: Path) -> dict[str, object]:
    documents = []
    hashes: dict[str, str] = {}
    for relative in SOURCE_PATHS:
        blob = _git_blob(root, relative)
        hashes[relative] = hashlib.sha256(blob).hexdigest()
        documents.append(
            {
                "path": relative,
                "text": blob.decode("utf-8", errors="strict"),
            }
        )
    return {
        "protocol_id": PROTOCOL_ID,
        "repository": REPOSITORY,
        "acquisition_provenance": {
            "repository_url": REPOSITORY_URL,
            "commit": SOURCE_COMMIT,
            "method": "git show <commit>:<path>",
            "read_only": True,
            "generated_by": "tools/acquire_cross_encoder_precision_v23_real_corpus.py",
            "source_paths": list(SOURCE_PATHS),
            "content_sha256": hashes,
        },
        "documents": documents,
        "relationships": list(RELATIONSHIPS),
    }


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    expected = _canonical_bytes(acquire(args.root.resolve()))
    output = args.output if args.output.is_absolute() else args.root / args.output
    if args.verify:
        if output.read_bytes() != expected:
            raise ValueError(
                "v23 real corpus differs from fixed Git object acquisition"
            )
        print(
            json.dumps(
                {"status": "verified", "sha256": hashlib.sha256(expected).hexdigest()}
            )
        )
        return 0
    with output.open("xb") as stream:
        stream.write(expected)
    print(
        json.dumps(
            {"status": "written", "sha256": hashlib.sha256(expected).hexdigest()}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
