from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


def coverage_index(report: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (str(row["repo"]), str(row["type"])): row
        for row in report.get("coverage", [])
    }


def compare_reports(
    previous: dict[str, Any], current: dict[str, Any]
) -> dict[str, Any]:
    before = coverage_index(previous)
    after = coverage_index(current)
    rows: list[dict[str, Any]] = []
    for repo, doc_type in sorted(set(before) | set(after)):
        old = before.get((repo, doc_type), {})
        new = after.get((repo, doc_type), {})
        old_count = int(old.get("source_count", 0))
        new_count = int(new.get("source_count", 0))
        old_commits = int(old.get("distinct_commit_count", 0))
        new_commits = int(new.get("distinct_commit_count", 0))
        rows.append(
            {
                "repo": repo,
                "type": doc_type,
                "source_count_before": old_count,
                "source_count_after": new_count,
                "source_count_delta": new_count - old_count,
                "distinct_commit_count_before": old_commits,
                "distinct_commit_count_after": new_commits,
                "distinct_commit_count_delta": new_commits - old_commits,
                "newest_updated_at_before": old.get("newest_updated_at"),
                "newest_updated_at_after": new.get("newest_updated_at"),
                "newest_extended": bool(
                    old.get("newest_updated_at")
                    and new.get("newest_updated_at")
                    and new["newest_updated_at"] > old["newest_updated_at"]
                ),
            }
        )
    return {"schema_version": 1, "coverage_changes": rows}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare coverage in two D1 fixture provenance reports."
    )
    parser.add_argument("previous", type=Path)
    parser.add_argument("current", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    previous = json.loads(args.previous.read_text(encoding="utf-8"))
    current = json.loads(args.current.read_text(encoding="utf-8"))
    print(
        json.dumps(
            compare_reports(previous, current),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
