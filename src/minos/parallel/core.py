"""Generic parallel execution primitives."""

from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from importlib import import_module
from time import perf_counter
import traceback
from typing import Any, Callable, Iterable


TaskStatus = str


@dataclass(frozen=True)
class TaskSpec:
    """A serializable task definition for backend execution."""

    task_id: str
    callable_path: str
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskOutcome:
    """Outcome payload for one task."""

    task_id: str
    status: TaskStatus
    result: Any = None
    error: str | None = None
    runtime_s: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


ProgressCallback = Callable[[int, int, TaskOutcome], None]


def resolve_callable(path: str) -> Callable[..., Any]:
    """Resolve callable from ``module.submodule:function_name`` string."""
    # Callables are passed by import path so `TaskSpec` stays pickle-safe across
    # process boundaries. This avoids capturing closures/lambdas, which often
    # fail under Windows spawn semantics.
    if ":" not in path:
        raise ValueError(f"callable_path must be 'module:function', got '{path}'.")
    module_name, fn_name = path.split(":", 1)
    module = import_module(module_name)
    fn = getattr(module, fn_name, None)
    if fn is None or not callable(fn):
        raise ValueError(f"Callable '{fn_name}' not found in module '{module_name}'.")
    return fn


def execute_task(spec: TaskSpec) -> TaskOutcome:
    """Execute one task and capture result/error as a stable outcome."""
    # Runtime is measured in wall-clock seconds for batch throughput diagnostics.
    # No monotonic simulation-time assumptions are made here.
    t0 = perf_counter()
    try:
        fn = resolve_callable(spec.callable_path)
        result = fn(*spec.args, **spec.kwargs)
        return TaskOutcome(
            task_id=spec.task_id,
            status="ok",
            result=result,
            runtime_s=float(perf_counter() - t0),
            metadata=dict(spec.metadata),
        )
    except Exception:
        return TaskOutcome(
            task_id=spec.task_id,
            status="error",
            error=traceback.format_exc(),
            runtime_s=float(perf_counter() - t0),
            metadata=dict(spec.metadata),
        )


class ExecutorBackend(ABC):
    """Backend interface for running task batches."""

    @abstractmethod
    def run(
        self,
        tasks: Iterable[TaskSpec],
        *,
        max_workers: int | None = None,
        fail_fast: bool = False,
        progress_cb: ProgressCallback | None = None,
    ) -> list[TaskOutcome]:
        """Run tasks and return outcomes in input order."""


class SequentialBackend(ExecutorBackend):
    """Single-process deterministic backend."""

    def run(
        self,
        tasks: Iterable[TaskSpec],
        *,
        max_workers: int | None = None,
        fail_fast: bool = False,
        progress_cb: ProgressCallback | None = None,
    ) -> list[TaskOutcome]:
        del max_workers
        task_list = list(tasks)
        outcomes: list[TaskOutcome] = []
        total = len(task_list)
        for i, task in enumerate(task_list, start=1):
            out = execute_task(task)
            outcomes.append(out)
            if progress_cb is not None:
                progress_cb(i, total, out)
            if fail_fast and out.status != "ok":
                break
        return outcomes


class _FuturePoolBackend(ExecutorBackend):
    """Common logic for thread/process backends."""

    executor_cls: type[ThreadPoolExecutor] | type[ProcessPoolExecutor]

    def run(
        self,
        tasks: Iterable[TaskSpec],
        *,
        max_workers: int | None = None,
        fail_fast: bool = False,
        progress_cb: ProgressCallback | None = None,
    ) -> list[TaskOutcome]:
        task_list = list(tasks)
        n = len(task_list)
        if n == 0:
            return []
        workers = max_workers if max_workers is not None else None
        # Outcomes are written back by original task index. This preserves input
        # ordering even though futures complete out-of-order.
        outcomes: list[TaskOutcome | None] = [None] * n
        completed = 0
        with self.executor_cls(max_workers=workers) as pool:
            index_by_future = {
                pool.submit(execute_task, task): i
                for i, task in enumerate(task_list)
            }
            for fut in as_completed(index_by_future):
                idx = index_by_future[fut]
                out = fut.result()
                outcomes[idx] = out
                completed += 1
                if progress_cb is not None:
                    progress_cb(completed, n, out)
                if fail_fast and out.status != "ok":
                    # Safety logic: cancel only not-yet-started futures. Running
                    # tasks cannot be force-killed by Future.cancel().
                    # TODO: Intent unclear — appears to assume pending-cancel is sufficient for fail-fast semantics. Confirm.
                    for pending in index_by_future:
                        if not pending.done():
                            pending.cancel()
                    break
        return [o for o in outcomes if o is not None]


class ThreadPoolBackend(_FuturePoolBackend):
    """Thread-based execution backend."""

    # Assumption: workload is either I/O-bound or spends meaningful time in C
    # extensions that release the GIL. For pure Python CPU-bound tasks, process
    # backend usually scales better.
    executor_cls = ThreadPoolExecutor


class ProcessPoolBackend(_FuturePoolBackend):
    """Process-based execution backend."""

    # Each worker has its own interpreter state/GIL, so this backend is the
    # default for CPU-heavy simulation sweeps.
    executor_cls = ProcessPoolExecutor


class ParallelRunner:
    """Facade for selecting backend and running tasks."""

    def __init__(self, backend: str = "process") -> None:
        key = backend.strip().lower()
        if key == "sequential":
            self.backend: ExecutorBackend = SequentialBackend()
        elif key == "thread":
            self.backend = ThreadPoolBackend()
        elif key == "process":
            self.backend = ProcessPoolBackend()
        else:
            raise ValueError("backend must be one of: 'sequential', 'thread', 'process'.")

    def run(
        self,
        tasks: Iterable[TaskSpec],
        *,
        max_workers: int | None = None,
        fail_fast: bool = False,
        progress_cb: ProgressCallback | None = None,
    ) -> list[TaskOutcome]:
        """Execute tasks through selected backend."""
        return self.backend.run(
            tasks,
            max_workers=max_workers,
            fail_fast=fail_fast,
            progress_cb=progress_cb,
        )
