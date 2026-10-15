"""Filesystem helpers."""

from __future__ import annotations

import os
import time
from pathlib import Path


def read_text_retry(path: Path, attempts: int = 12, delay: float = 0.02) -> str:
    """Path.read_text with the same tolerance for the transient sharing violation a concurrent os.replace can cause.
    A persistent PermissionError, or any other error, still raises."""
    for attempt in range(attempts):
        try:
            return Path(path).read_text(encoding="utf-8")
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(delay * (attempt + 1))
    raise AssertionError("unreachable")


def replace_atomic(source: Path, target: Path, attempts: int = 12, delay: float = 0.02) -> None:
    """os.replace with retries for transient Windows sharing violations.

    Antivirus scanners and indexers can briefly hold a freshly written file, making os.replace raise PermissionError
    (WinError 5/32). Retrying with backoff keeps state commits atomic without turning a hiccup into a failed step.
    Persistent failures still raise."""
    for attempt in range(attempts):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(delay * (attempt + 1))
