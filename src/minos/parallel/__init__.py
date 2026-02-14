"""Generic parallel execution module."""

from .core import (
    ExecutorBackend,
    ParallelRunner,
    ProcessPoolBackend,
    SequentialBackend,
    TaskOutcome,
    TaskSpec,
    ThreadPoolBackend,
    execute_task,
    resolve_callable,
)

__all__ = [
    "ExecutorBackend",
    "ParallelRunner",
    "ProcessPoolBackend",
    "SequentialBackend",
    "TaskOutcome",
    "TaskSpec",
    "ThreadPoolBackend",
    "execute_task",
    "resolve_callable",
]
