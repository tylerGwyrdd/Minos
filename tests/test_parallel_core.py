"""Tests for generic parallel execution module."""

from __future__ import annotations

import unittest

from minos.parallel import ParallelRunner, TaskSpec, execute_task, resolve_callable


class TestParallelCore(unittest.TestCase):
    def test_resolve_callable(self) -> None:
        fn = resolve_callable("math:sqrt")
        self.assertEqual(fn(9.0), 3.0)

    def test_execute_task_success(self) -> None:
        out = execute_task(TaskSpec(task_id="a", callable_path="operator:add", args=(2, 3)))
        self.assertEqual(out.status, "ok")
        self.assertEqual(out.result, 5)

    def test_execute_task_error(self) -> None:
        out = execute_task(TaskSpec(task_id="b", callable_path="math:does_not_exist"))
        self.assertEqual(out.status, "error")
        self.assertIsNotNone(out.error)

    def test_parallel_runner_process(self) -> None:
        tasks = [
            TaskSpec(task_id="t1", callable_path="operator:add", args=(1, 2)),
            TaskSpec(task_id="t2", callable_path="operator:add", args=(5, 7)),
        ]
        runner = ParallelRunner(backend="process")
        try:
            out = runner.run(tasks, max_workers=2)
        except PermissionError as exc:
            self.skipTest(f"ProcessPool unavailable in this environment: {exc}")
        self.assertEqual(len(out), 2)
        self.assertEqual([o.result for o in out], [3, 12])

    def test_parallel_runner_thread(self) -> None:
        tasks = [
            TaskSpec(task_id="t1", callable_path="operator:add", args=(3, 4)),
            TaskSpec(task_id="t2", callable_path="operator:add", args=(6, 8)),
        ]
        runner = ParallelRunner(backend="thread")
        out = runner.run(tasks, max_workers=2)
        self.assertEqual([o.result for o in out], [7, 14])


if __name__ == "__main__":
    unittest.main()
