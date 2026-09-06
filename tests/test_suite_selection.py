from pathlib import Path
import tempfile
import unittest

from tools.run_tests import selected_modules, validate_inventory
from tools.test_suites import GROUPS


class SuiteSelectionTests(unittest.TestCase):
    def test_real_inventory_is_exhaustive(self):
        validate_inventory()

    def test_shared_contracts_in_both_paths_and_union_is_all(self):
        normal = set(selected_modules("normal"))
        experiments = set(selected_modules("experiments"))
        self.assertEqual(normal & experiments, set(GROUPS["shared"]["modules"]))
        self.assertEqual(normal | experiments, set(selected_modules("all")))

    def test_unknown_nested_stale_and_duplicate_modules_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "tests" / "nested").mkdir(parents=True)
            (root / "tests" / "test_known.py").touch()
            groups = {"product": {"reason": "contract", "modules": ["test_known"]}}
            validate_inventory(root, groups)
            extra = root / "tests" / "nested" / "test_new.py"
            extra.touch()
            with self.assertRaisesRegex(ValueError, "unknown=.*nested.test_new"):
                validate_inventory(root, groups)
            extra.unlink()
            groups["product"]["modules"].append("test_known")
            with self.assertRaisesRegex(ValueError, "duplicate=.*test_known"):
                validate_inventory(root, groups)
            groups["product"]["modules"] = ["test_missing"]
            with self.assertRaisesRegex(ValueError, "stale=.*test_missing"):
                validate_inventory(root, groups)
