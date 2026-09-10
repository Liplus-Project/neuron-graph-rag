"""Verify the recorded v5 development observation against its freeze commit."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from neuron_graph_rag.github_retrieval_parity_v5_observation import (
    verify_observation,
)


def main() -> int:
    print(json.dumps(verify_observation(_ROOT), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
