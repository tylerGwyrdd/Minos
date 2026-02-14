"""Smoke tests for guidance examples."""

from __future__ import annotations

import os
import subprocess
import sys
import unittest


def _with_src_pythonpath() -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = "src" if not existing else f"src{os.pathsep}{existing}"
    return env


class TestGuidanceExamples(unittest.TestCase):
    def test_heading_scenarios_run(self) -> None:
        result = subprocess.run(
            [sys.executable, "examples/guidance_heading_scenarios.py", "--no-plot"],
            capture_output=True,
            text=True,
            check=False,
            env=_with_src_pythonpath(),
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("straight_line:", result.stdout)
        self.assertIn("step_turn_90deg_at_20s:", result.stdout)

    def test_t_approach_branch_examples_run(self) -> None:
        result = subprocess.run(
            [sys.executable, "examples/t_approach_branch_examples.py"],
            capture_output=True,
            text=True,
            check=False,
            env=_with_src_pythonpath(),
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("initialising:", result.stdout)
        self.assertIn("final_approach_with_flare:", result.stdout)


if __name__ == "__main__":
    unittest.main()
