"""Opt-in unittest selection. Does not alter unittest discovery or frozen CI."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.test_suites import GROUPS, SUITES


def validate_inventory(root: Path = ROOT, groups: dict = GROUPS) -> None:
    classified = [name for group in groups.values() for name in group["modules"]]
    discovered = {
        path.relative_to(root / "tests").with_suffix("").as_posix().replace("/", ".")
        for path in (root / "tests").rglob("test*.py")
    }
    duplicates = sorted(name for name, count in Counter(classified).items() if count != 1)
    unknown = sorted(discovered - set(classified))
    stale = sorted(set(classified) - discovered)
    if duplicates or unknown or stale:
        raise ValueError(f"Test inventory mismatch: unknown={unknown}, stale={stale}, duplicate={duplicates}")
    if any(not group.get("reason") for group in groups.values()):
        raise ValueError("Every group needs a classification reason")


def selected_modules(suite: str) -> list[str]:
    return sorted(name for role in SUITES[suite] for name in GROUPS[role]["modules"])


def build_suite(suite: str) -> unittest.TestSuite:
    validate_inventory()
    loader = unittest.TestLoader()
    if suite == "all":
        # Preserve the established full-discovery path, including load_tests hooks.
        result = loader.discover(str(ROOT / "tests"))
    else:
        result = loader.loadTestsFromNames(["tests." + name for name in selected_modules(suite)])
    if loader.errors:
        raise ValueError("\n".join(loader.errors))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=SUITES)
    parser.add_argument("--list", action="store_true", help="Validate and list modules without importing tests")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    started = time.perf_counter()
    try:
        validate_inventory()
        modules = selected_modules(args.suite)
        if args.list:
            print(json.dumps({"suite": args.suite, "modules": modules, "module_count": len(modules)}, indent=2))
            return 0
        suite = build_suite(args.suite)
    except ValueError as exc:
        parser.exit(2, str(exc) + "\n")
    selected_count = suite.countTestCases()
    result = unittest.TextTestRunner(verbosity=2 if args.verbose else 1).run(suite)
    print(json.dumps({
        "suite": args.suite, "module_count": len(modules), "selected_tests": selected_count,
        "tests_run": result.testsRun, "skipped": len(result.skipped),
        "failures": len(result.failures), "errors": len(result.errors),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
