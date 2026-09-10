"""Audit or execute the frozen GitHub retrieval parity v5 protocol."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
PACKAGE_ROOT = SOURCE_ROOT / "neuron_graph_rag"


def _bootstrap_source_tree() -> None:
    """Resolve the repository source tree before importing the package."""
    if not PACKAGE_ROOT.is_dir():
        raise RuntimeError(f"repository package directory is missing: {PACKAGE_ROOT}")
    source = str(SOURCE_ROOT)
    if source not in sys.path:
        sys.path.insert(0, source)


_bootstrap_source_tree()

_protocol = importlib.import_module(
    "neuron_graph_rag.github_retrieval_parity_v5"
)


def _import_smoke_payload() -> dict[str, object]:
    return {
        "status": "import-safe",
        "protocol_id": "github-rag-vs-ngr-retrieval-parity-v5",
        "repository_root_resolved": REPOSITORY_ROOT.is_dir(),
        "source_root_resolved": SOURCE_ROOT.is_dir(),
        "github_rag_request_count": 0,
        "ngr_search_count": 0,
        "sqlite_open_count": 0,
        "artifact_create_count": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--import-smoke", action="store_true")
    actions.add_argument("--audit", action="store_true")
    actions.add_argument("--lifecycle", action="store_true")
    actions.add_argument("--register-preflight", choices=("development", "holdout"))
    actions.add_argument("--register-capture", choices=("development", "holdout"))
    actions.add_argument("--stage", choices=("development", "holdout"))
    actions.add_argument("--verify", choices=("development", "holdout"))
    actions.add_argument("--verify-preflight", choices=("development", "holdout"))
    parser.add_argument("--input", type=Path)
    parser.add_argument("--protocol-commit")
    args = parser.parse_args()

    if args.import_smoke:
        print(json.dumps(_import_smoke_payload(), ensure_ascii=False, indent=2))
        return
    if args.audit:
        print(json.dumps(_protocol.audit_result_free(), ensure_ascii=False, indent=2))
        return
    if args.lifecycle:
        print(
            json.dumps(
                _protocol.audit_repository_lifecycle(), ensure_ascii=False, indent=2
            )
        )
        return
    if args.register_preflight:
        if args.input is None or args.protocol_commit is None:
            parser.error("--register-preflight requires --input and --protocol-commit")
        print(
            _protocol.register_preflight(
                args.register_preflight, args.input, args.protocol_commit
            )
        )
        return
    if args.register_capture:
        if args.input is None or args.protocol_commit is None:
            parser.error("--register-capture requires --input and --protocol-commit")
        print(
            _protocol.register_capture(
                args.register_capture, args.input, args.protocol_commit
            )
        )
        return
    if args.stage:
        if args.protocol_commit is None:
            parser.error("--stage requires --protocol-commit")
        print(_protocol.run_registered_stage(args.stage, args.protocol_commit))
        return
    if args.verify_preflight:
        _protocol.verify_registered_preflight(args.verify_preflight)
        print(f"{args.verify_preflight} preflight verification passed")
        return
    _protocol.verify_registered_result(str(args.verify))
    print(f"{args.verify} verification passed")


if __name__ == "__main__":
    main()
