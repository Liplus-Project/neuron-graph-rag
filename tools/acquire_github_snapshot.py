"""Acquire selected public GitHub files into a read-only, pinned snapshot."""

from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import quote
from urllib.request import Request, urlopen


API_ROOT = "https://api.github.com"


def github_get(path: str) -> dict[str, Any]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "neuron-graph-rag-read-only-snapshot",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(f"{API_ROOT}{path}", headers=headers, method="GET")
    with urlopen(request, timeout=30) as response:  # noqa: S310 - fixed GitHub API root
        return json.loads(response.read().decode("utf-8"))


def acquire_snapshot(repository: str, ref: str, paths: Sequence[str]) -> dict[str, Any]:
    if repository.count("/") != 1:
        raise ValueError("repository must be an owner/repository slug")
    if not paths:
        raise ValueError("at least one --path is required")
    commit = str(github_get(f"/repos/{repository}/commits/{quote(ref, safe='')}")["sha"])
    documents: list[dict[str, str]] = []
    for path in sorted(set(paths)):
        if not path or path.startswith("/") or ".." in Path(path).parts:
            raise ValueError(f"invalid GitHub path: {path}")
        payload = github_get(
            f"/repos/{repository}/contents/{quote(path, safe='/')}?ref={commit}"
        )
        if payload.get("type") != "file" or payload.get("encoding") != "base64":
            raise ValueError(f"GitHub path is not a base64 file: {path}")
        content = base64.b64decode(str(payload["content"])).decode("utf-8")
        documents.append(
            {
                "path": path,
                "blob_sha": str(payload["sha"]),
                "content": content,
                "source_url": f"https://github.com/{repository}/blob/{commit}/{path}",
            }
        )
    return {
        "schema_version": 1,
        "repository": repository,
        "commit": commit,
        "documents": documents,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--ref", default="main")
    parser.add_argument("--path", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    snapshot = acquire_snapshot(args.repo, args.ref, args.path)
    args.output.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
