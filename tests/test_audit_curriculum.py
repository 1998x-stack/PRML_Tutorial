"""Tests for the read-only curriculum inventory.

Run: python -m unittest discover -s tests -p 'test_*.py'
"""
import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_curriculum.py"
SPEC = importlib.util.spec_from_file_location("audit_curriculum", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CurriculumAuditTest(unittest.TestCase):
    def test_counts_matched_sections_and_small_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lesson = root / "2_Distributions" / "2.1_Bernoulli"
            lesson.mkdir(parents=True)
            (lesson / "00_Beta.md").write_text("explanation " * 50, encoding="utf-8")
            (lesson / "00_Beta.py").write_text("print('demo')\n", encoding="utf-8")
            (root / "1_Intro").mkdir()
            result = MODULE.inspect_curriculum(root)
            self.assertEqual(result["chapter_count"], 2)
            self.assertEqual(result["totals"]["pairs"], 1)
            self.assertEqual(result["totals"]["small_markdown"], 0)
            self.assertEqual(result["totals"]["small_python"], 1)
            self.assertEqual(result["unpaired"], [])

    def test_reports_missing_partner_by_relative_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lesson = root / "1_Intro" / "1.1_Basics"
            lesson.mkdir(parents=True)
            (lesson / "1.1_Basics.md").write_text("content", encoding="utf-8")
            result = MODULE.inspect_curriculum(root)
            self.assertEqual(result["totals"]["pairs"], 0)
            self.assertEqual(result["unpaired"], ["1_Intro/1.1_Basics/1.1_Basics (missing .py)"])

    def test_rejects_invalid_threshold_and_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(ValueError):
                MODULE.inspect_curriculum(root, -1)
            with self.assertRaises(ValueError):
                MODULE.inspect_curriculum(root / "unknown")


if __name__ == "__main__":
    unittest.main()
