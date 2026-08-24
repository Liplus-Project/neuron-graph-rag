"""Read-only GitHub source adapter for local NGR compatibility experiments.

This module deliberately has no HTTP client.  Acquisition is owned by the
``tools/acquire_github_snapshot.py`` boundary; the core engine receives only a
fixed, reviewable snapshot.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .engine import NeuronGraphRAG

SNAPSHOT_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class GitHubDocument:
    path: str
    blob_sha: str
    content: str
    source_url: str

    @property
    def content_sha256(self) -> str:
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class GitHubSnapshot:
    repository: str
    commit: str
    documents: tuple[GitHubDocument, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "GitHubSnapshot":
        if value.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
            raise ValueError("Unsupported GitHub snapshot schema version")
        repository = _required_string(value, "repository")
        commit = _required_string(value, "commit")
        documents_value = value.get("documents")
        if not isinstance(documents_value, list) or not documents_value:
            raise ValueError("GitHub snapshot must contain at least one document")
        documents: list[GitHubDocument] = []
        seen_paths: set[str] = set()
        for item in documents_value:
            if not isinstance(item, Mapping):
                raise ValueError("GitHub snapshot documents must be objects")
            path = _required_string(item, "path")
            if path in seen_paths:
                raise ValueError(f"Duplicate GitHub snapshot path: {path}")
            seen_paths.add(path)
            content = _required_string(item, "content")
            blob_sha = _required_string(item, "blob_sha")
            source_url = _required_string(item, "source_url")
            content_sha256 = item.get("content_sha256")
            actual_content_sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()
            if content_sha256 is not None and content_sha256 != actual_content_sha256:
                raise ValueError(f"GitHub snapshot content hash mismatch: {path}")
            expected_url = f"https://github.com/{repository}/blob/{commit}/{path}"
            if source_url != expected_url:
                raise ValueError(
                    "GitHub snapshot source_url must pin repository, commit, and path"
                )
            documents.append(GitHubDocument(path, blob_sha, content, source_url))
        return cls(
            repository, commit, tuple(sorted(documents, key=lambda item: item.path))
        )

    @classmethod
    def read(cls, path: str | Path) -> "GitHubSnapshot":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise ValueError("GitHub snapshot must be a JSON object")
        return cls.from_mapping(value)

    def document_id(self, document: GitHubDocument) -> str:
        return f"github:{self.repository}:{document.path}"

    def fingerprint(self) -> str:
        payload = {
            "repository": self.repository,
            "commit": self.commit,
            "documents": [
                {"path": document.path, "blob_sha": document.blob_sha}
                for document in self.documents
            ],
        }
        serialized = json.dumps(payload, separators=(",", ":"), sort_keys=True)
        return "sha256:" + hashlib.sha256(serialized.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class GitHubIndexReceipt:
    repository: str
    commit: str
    fingerprint: str
    node_ids: tuple[str, ...]
    source_urls: tuple[str, ...]


def index_github_snapshot(
    engine: NeuronGraphRAG, snapshot: GitHubSnapshot
) -> GitHubIndexReceipt:
    """Upsert a fixed GitHub snapshot into an existing local NGR index."""
    node_ids: list[str] = []
    source_urls: list[str] = []
    for document in snapshot.documents:
        node_id = snapshot.document_id(document)
        engine.add_document(
            node_id,
            f"{document.path}\n\n{document.content}",
            metadata={
                "source_adapter": "github_read_only_snapshot",
                "repository": snapshot.repository,
                "commit": snapshot.commit,
                "path": document.path,
                "blob_sha": document.blob_sha,
                "content_sha256": document.content_sha256,
                "source_url": document.source_url,
            },
        )
        node_ids.append(node_id)
        source_urls.append(document.source_url)
    return GitHubIndexReceipt(
        snapshot.repository,
        snapshot.commit,
        snapshot.fingerprint(),
        tuple(node_ids),
        tuple(source_urls),
    )


def changed_paths(before: GitHubSnapshot, after: GitHubSnapshot) -> tuple[str, ...]:
    if before.repository != after.repository:
        raise ValueError("Snapshots must refer to the same GitHub repository")
    previous = {document.path: document.blob_sha for document in before.documents}
    current = {document.path: document.blob_sha for document in after.documents}
    return tuple(
        path
        for path in sorted(set(previous) | set(current))
        if previous.get(path) != current.get(path)
    )


def _required_string(value: Mapping[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.strip():
        raise ValueError(f"GitHub snapshot {key} must be a non-empty string")
    return item
