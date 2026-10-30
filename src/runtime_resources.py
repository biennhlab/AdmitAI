"""Read-only memory diagnostics and allocation error classification."""
from __future__ import annotations

import ctypes
import sys


class StartupCheckError(RuntimeError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


def memory_status() -> dict[str, float] | None:
    if sys.platform != "win32":
        return None

    class MemoryStatus(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
            (key, ctypes.c_ulonglong) for key in (
                "total_physical", "available_physical", "total_commit",
                "available_commit", "total_virtual", "available_virtual", "extended",
            )
        ]

    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    gib = 1024**3
    return {
        "total_physical_gib": status.total_physical / gib,
        "available_physical_gib": status.available_physical / gib,
        "commit_limit_gib": status.total_commit / gib,
        "commit_charge_gib": (status.total_commit - status.available_commit) / gib,
        "commit_headroom_gib": status.available_commit / gib,
    }


def failure_code(exc: BaseException, stage: str) -> str:
    current: BaseException | None = exc
    while current is not None:
        if isinstance(current, StartupCheckError):
            return current.code
        message = str(current).casefold()
        if isinstance(current, MemoryError) or getattr(current, "winerror", None) in {8, 14, 1455} or any(
            term in message for term in ("out of memory", "paging file", "pagefile", "cannot allocate memory", "not enough memory", "defaultcpuallocator")
        ):
            return "INSUFFICIENT_MEMORY"
        current = current.__cause__ or current.__context__
    return {"local_index": "INDEX_INVALID", "qdrant": "QDRANT_UNAVAILABLE", "rag": "RAG_INITIALIZATION_FAILED"}.get(stage, "STARTUP_FAILED")
