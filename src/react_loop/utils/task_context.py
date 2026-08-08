import contextvars
from typing import Any, Dict, Optional

_task_context: contextvars.ContextVar[Optional[Dict[str, Any]]] = contextvars.ContextVar(
    'task_context', default=None
)


def init_task_context() -> dict:
    ctx = {}
    _task_context.set(ctx)
    return ctx


def get_task_context() -> Dict[str, Any]:
    ctx = _task_context.get()
    if ctx is None:
        ctx = {}
        _task_context.set(ctx)
    return ctx


def clear_task_context() -> None:
    _task_context.set(None)
