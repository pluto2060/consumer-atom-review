"""Append-only, hash-chained causal trace (JSONL) per run: the default AuditSink.

Every entry carries `seq`, `prev_hash`, and `entry_hash = sha256(canonical(entry without entry_hash))`, so editing,
deleting, reordering, or inserting a line breaks the chain. A separate `head.json` records the latest (seq, hash) so that
truncating the tail is detected too. This is tamper-EVIDENT, not tamper-PROOF: someone who can rewrite both files can
forge a consistent history. Anchor `head()` somewhere they cannot write (another host, a ledger) and pass it to
`verify(root, expected_head=...)` for a stronger guarantee.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any

from .fsutil import replace_atomic
from .state import now

GENESIS = "0" * 64
_LOCK = threading.Lock()


def _canonical(entry: dict[str, Any]) -> bytes:
    return json.dumps(entry, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def entry_hash(entry: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical({k: v for k, v in entry.items() if k != "entry_hash"})).hexdigest()


class MirroredAudit:
    """AuditSink decorator: every chained entry is also handed to `mirror(root, entry)` (e.g. an external ledger).

    Fail-closed by default: if the mirror cannot record an entry, the append raises so the step does not advance
    without its external audit record."""

    def __init__(self, primary: Any, mirror: Any, required: bool = True):
        self.primary, self.mirror, self.required = primary, mirror, required

    def append(self, root: Path, event: dict[str, Any]) -> dict[str, Any]:
        entry = self.primary.append(root, event)
        try:
            self.mirror(root, entry)
        except Exception:
            if self.required:
                raise
        return entry

    def read(self, root: Path) -> list[dict[str, Any]]:
        return self.primary.read(root)

    def verify(self, root: Path, expected_head: str | None = None) -> list[dict[str, Any]]:
        return self.primary.verify(root, expected_head)

    def head(self, root: Path) -> dict[str, Any] | None:
        return self.primary.head(root)


class Trace:
    """Default AuditSink: hash-chained JSONL files under `<run>/traces/`."""

    @staticmethod
    def _paths(root: Path) -> tuple[Path, Path]:
        return root / "traces" / "events.jsonl", root / "traces" / "head.json"

    @staticmethod
    def _last(path: Path) -> dict[str, Any] | None:
        """Last entry of the chain (an unchained/legacy last line starts a fresh chain at seq 1)."""
        if not path.exists():
            return None
        lines = path.read_text(encoding="utf-8").splitlines()
        if not lines:
            return None
        entry = json.loads(lines[-1])
        return entry if "entry_hash" in entry and "seq" in entry else None

    @classmethod
    def append(cls, root: Path, event: dict[str, Any]) -> dict[str, Any]:
        path, head_path = cls._paths(root)
        with _LOCK:
            last = cls._last(path)
            entry = {**event}
            entry.setdefault("timestamp", now())
            entry["seq"] = (last["seq"] + 1) if last else 1
            entry["prev_hash"] = last["entry_hash"] if last else GENESIS
            entry["entry_hash"] = entry_hash(entry)
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
            temp = head_path.with_suffix(".tmp")
            temp.write_text(json.dumps({"seq": entry["seq"], "entry_hash": entry["entry_hash"]}), encoding="utf-8")
            replace_atomic(temp, head_path)
        return entry

    @classmethod
    def read(cls, root: Path) -> list[dict[str, Any]]:
        path, _ = cls._paths(root)
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]

    @classmethod
    def head(cls, root: Path) -> dict[str, Any] | None:
        _, head_path = cls._paths(root)
        return json.loads(head_path.read_text(encoding="utf-8")) if head_path.exists() else None

    @classmethod
    def verify(cls, root: Path, expected_head: str | None = None) -> list[dict[str, Any]]:
        """Return a list of problems (empty = chain intact). `expected_head` is an externally anchored entry_hash."""
        path, head_path = cls._paths(root)
        problems: list[dict[str, Any]] = []
        if not path.exists():
            return [{"line": 0, "reason": "trace file is missing"}]
        previous, seq, last_hash = GENESIS, 0, None
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                problems.append({"line": number, "reason": "line is not valid JSON"})
                return problems
            if "entry_hash" not in entry:
                problems.append({"line": number, "reason": "entry is not chained (legacy or forged line)"})
                return problems
            seq += 1
            if entry.get("seq") != seq:
                problems.append({"line": number, "reason": f"sequence gap: expected {seq}, found {entry.get('seq')}"})
                return problems
            if entry.get("prev_hash") != previous:
                problems.append({"line": number, "reason": "prev_hash does not match the previous entry"})
                return problems
            if entry_hash(entry) != entry["entry_hash"]:
                problems.append({"line": number, "reason": "entry content does not match its hash"})
                return problems
            previous = last_hash = entry["entry_hash"]
        head = json.loads(head_path.read_text(encoding="utf-8")) if head_path.exists() else None
        if head is None:
            problems.append({"line": seq, "reason": "head.json is missing (cannot rule out truncation)"})
        elif head.get("entry_hash") != last_hash or head.get("seq") != seq:
            problems.append({"line": seq, "reason": f"trace was truncated or rewritten: head expects seq {head.get('seq')}"})
        if expected_head is not None and expected_head != last_hash:
            problems.append({"line": seq, "reason": "trace head does not match the externally anchored head"})
        return problems

