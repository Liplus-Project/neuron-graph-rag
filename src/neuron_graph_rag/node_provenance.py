from __future__ import annotations

import re
from typing import Any


_SOURCE_REF = re.compile(
    r"https://github[.]com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/"
    r"(?:issues|pull)/[1-9][0-9]*/?"
)


def validate_node_provenance(metadata: dict[str, Any]) -> None:
    """Validate the optional reserved key without interpreting judgment provenance."""
    if "node_provenance" not in metadata:
        return
    value = metadata["node_provenance"]
    if not isinstance(value, dict) or set(value) != {"writer", "source_refs"}:
        raise ValueError("node_provenance must contain exactly writer and source_refs")
    writer = value["writer"]
    refs = value["source_refs"]
    if not isinstance(writer, str) or not writer.strip():
        raise ValueError("node_provenance.writer must be a non-empty string")
    if not isinstance(refs, list) or not refs or any(
        not isinstance(ref, str) or _SOURCE_REF.fullmatch(ref) is None for ref in refs
    ):
        raise ValueError("node_provenance.source_refs must be a non-empty list of HTTPS GitHub issue/PR URLs")
