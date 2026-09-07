"""Acquire and verify every Markdown blob in a pinned public GitHub commit."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tarfile
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen

API_ROOT = "https://api.github.com"


def _request(path: str) -> Request:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "neuron-graph-rag-read-only-markdown-snapshot-v3",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return Request(f"{API_ROOT}{path}", headers=headers, method="GET")


def _github_json(path: str) -> dict[str, Any]:
    with urlopen(_request(path), timeout=60) as response:
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, dict):
        raise TypeError("GitHub JSON object required")
    return value


def _github_bytes(path: str) -> bytes:
    with urlopen(_request(path), timeout=60) as response:
        return response.read()


def _blob_sha(content: bytes) -> str:
    header = f"blob {len(content)}\0".encode()
    return hashlib.sha1(header + content).hexdigest()


def acquire_snapshot(repository: str, ref: str) -> dict[str, Any]:
    if repository.count("/") != 1:
        raise ValueError("repository must be an owner/repository slug")
    commit_payload = _github_json(
        f"/repos/{repository}/git/commits/{quote(ref, safe='')}"
    )
    commit = str(commit_payload["sha"])
    tree_sha = str(commit_payload["tree"]["sha"])
    archive = _github_bytes(
        f"/repos/{repository}/tarball/{quote(commit, safe='')}"
    )
    documents: list[dict[str, str]] = []
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as bundle:
        for member in bundle.getmembers():
            if not member.isfile():
                continue
            parts = PurePosixPath(member.name).parts
            if len(parts) < 2:
                continue
            relative = PurePosixPath(*parts[1:]).as_posix()
            if not relative.endswith(".md"):
                continue
            handle = bundle.extractfile(member)
            if handle is None:
                raise ValueError(f"archive member is unreadable: {relative}")
            raw = handle.read()
            content = raw.decode("utf-8", errors="strict")
            documents.append(
                {
                    "path": relative,
                    "blob_sha": _blob_sha(raw),
                    "content": content,
                    "content_sha256": hashlib.sha256(raw).hexdigest(),
                    "source_url": (
                        f"https://github.com/{repository}/blob/{commit}/{relative}"
                    ),
                }
            )
    documents.sort(key=lambda row: row["path"])
    if not documents:
        raise ValueError("pinned commit contains no Markdown files")
    return {
        "schema_version": 1,
        "repository": repository,
        "commit": commit,
        "tree_sha": tree_sha,
        "documents": documents,
    }


def _encoded(snapshot: dict[str, Any]) -> bytes:
    return (
        json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--ref", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    encoded = _encoded(acquire_snapshot(args.repo, args.ref))
    if args.verify:
        if args.output.read_bytes() != encoded:
            raise ValueError("frozen Markdown snapshot differs from GitHub")
        print(f"verified {args.output}")
        return 0
    args.output.write_bytes(encoded)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
