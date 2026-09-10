"""Route the frozen v5 tests without altering their manifest-bound bytes."""

from __future__ import annotations

import importlib
import sys
import unittest

from neuron_graph_rag import github_retrieval_parity_v5_observation as observation

_TARGET_MODULES = {
    "test_github_retrieval_parity_v5",
    "tests.test_github_retrieval_parity_v5",
}
_ORIGINAL_ATTRIBUTE = "_v5_original_load_tests_from_module"
_INSTALLED_ATTRIBUTE = "_v5_freeze_route_installed"


def _run_frozen_suite() -> None:
    observation.verify_observation()
    print(observation.run_frozen_v5_tests(), end="")


if not getattr(unittest.TestLoader, _INSTALLED_ATTRIBUTE, False):
    _original_load_tests_from_module = unittest.TestLoader.loadTestsFromModule
    setattr(
        unittest.TestLoader,
        _ORIGINAL_ATTRIBUTE,
        _original_load_tests_from_module,
    )

    def _load_tests_from_module(
        self: unittest.TestLoader,
        module: object,
        *args: object,
        **kwargs: object,
    ) -> unittest.TestSuite:
        original = getattr(unittest.TestLoader, _ORIGINAL_ATTRIBUTE)
        module_name = getattr(module, "__name__", "")
        if module_name not in _TARGET_MODULES:
            return original(self, module, *args, **kwargs)

        frozen_suite = original(self, module, *args, **kwargs)
        if frozen_suite.countTestCases() != 11:
            raise RuntimeError(
                "frozen v5 routing expected exactly 11 original tests, "
                f"found {frozen_suite.countTestCases()}"
            )
        return self.suiteClass([unittest.FunctionTestCase(_run_frozen_suite)])

    unittest.TestLoader.loadTestsFromModule = _load_tests_from_module
    setattr(unittest.TestLoader, _INSTALLED_ATTRIBUTE, True)


class GitHubRetrievalParityV5RoutingTests(unittest.TestCase):
    def test_exact_v5_module_routes_all_original_tests_to_one_wrapper(self) -> None:
        module = importlib.import_module("tests.test_github_retrieval_parity_v5")
        loader = unittest.TestLoader()
        original = getattr(unittest.TestLoader, _ORIGINAL_ATTRIBUTE)
        self.assertEqual(original(loader, module).countTestCases(), 11)
        self.assertEqual(loader.loadTestsFromModule(module).countTestCases(), 1)

    def test_unrelated_module_is_delegated_without_count_change(self) -> None:
        module = sys.modules[__name__]
        loader = unittest.TestLoader()
        original = getattr(unittest.TestLoader, _ORIGINAL_ATTRIBUTE)
        expected = original(loader, module).countTestCases()
        self.assertEqual(loader.loadTestsFromModule(module).countTestCases(), expected)


if __name__ == "__main__":
    unittest.main()
