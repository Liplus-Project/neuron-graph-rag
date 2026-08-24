from __future__ import annotations

import argparse
import json
from pathlib import Path

from neuron_graph_rag.database_home import migrate_database


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy one NGR SQLite database to a new explicit location"
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--backup", type=Path, required=True)
    arguments = parser.parse_args()
    result = migrate_database(
        arguments.source,
        arguments.destination,
        arguments.backup,
    )
    print(
        json.dumps(
            {
                "backup": str(result.backup),
                "destination": str(result.destination),
                "source": str(result.source),
                "sqlite_integrity": "ok",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
