"""Keep frozen test bytes while routing their exact modules through the bridge."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import historical_readme_bridge as bridge

ORIGINAL = "_historical_readme_original_loader"
INSTALLED = "_historical_readme_router_installed"


def routed_load_tests(self, module, *args, **kwargs):
    original = getattr(unittest.TestLoader, ORIGINAL)
    name = getattr(module, "__name__", "")
    bare = name.removeprefix("tests.")
    if name not in (bare, "tests." + bare) or bare not in bridge.ROUTED_MODULE_COUNTS:
        return original(self, module, *args, **kwargs)
    expected_path = bridge.ROOT / "tests" / (bare + ".py")
    if Path(getattr(module, "__file__", "")).resolve() != expected_path:
        raise ValueError(f"historical routing module root mismatch: {name}")
    count = original(self, module, *args, **kwargs).countTestCases()
    if count != bridge.ROUTED_MODULE_COUNTS[bare]:
        raise ValueError(f"historical {bare} expected {bridge.ROUTED_MODULE_COUNTS[bare]} original tests, found {count}")
    def run_original_module():
        print(bridge.run_module(bare), end="")
    return self.suiteClass([unittest.FunctionTestCase(run_original_module,
                                                   description=f"historical {bridge.group_of(bare)}: {bare} ({count} original tests)")])


if not getattr(unittest.TestLoader, INSTALLED, False):
    setattr(unittest.TestLoader, ORIGINAL, unittest.TestLoader.loadTestsFromModule)
    unittest.TestLoader.loadTestsFromModule = routed_load_tests
    setattr(unittest.TestLoader, INSTALLED, True)
