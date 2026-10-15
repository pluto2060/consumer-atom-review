"""Evaluator integrity (M2622 T9, T7): the code that decides pass or fail is hashed, pinned by a named human, and recorded in every run and selection entry.

Why: an evolution loop that can change the evaluator together with the policy can raise its own score without getting better. The evaluator is therefore a
separate approval CLASS: a change to its files is visible as a hash mismatch, auto-approval of policy changes pauses while it mismatches, and only a named
human (or an owner scope `evaluator`, never part of `all`) pins the new hash. A hidden validation set (run ids kept outside every path the proposers read)
is the second guard: proposals are made without seeing it and promotion is judged on it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .gate import SELF_APPROVERS
from .state import now

ROOT = Path(__file__).resolve().parent
EVALUATOR_GLOBS = ("*_validators.py", "validator_common.py", "judge*.py", "confidence.py", "evaluation.py", "trends.py", "principle_checks.py", "verifier.py", "gate.py", "promotion.py")
PIN_FILE = "evaluator_pin.json"
HIDDEN_FILE = "hidden_validation.json"


def manifest(root: Path | None = None) -> dict[str, str]:
    base = Path(root) if root else ROOT
    files = sorted({p for g in EVALUATOR_GLOBS for p in base.glob(g)})
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def evaluator_hash(files: dict[str, str] | None = None) -> str:
    files = files if files is not None else manifest()
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode("utf-8")).hexdigest()


class PinStore:
    """The pinned evaluator hash of one compiler runtime directory."""

    def __init__(self, directory: Path, root: Path | None = None):
        self.path = Path(directory) / PIN_FILE
        self.root = root

    def _load(self) -> dict[str, Any] | None:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def status(self) -> dict[str, Any]:
        current = manifest(self.root)
        pin = self._load()
        if pin is None:
            return {"pinned": False, "match": True, "current": evaluator_hash(current), "note": "no evaluator has been pinned yet, so nothing is enforced"}
        changed = sorted({n for n in set(current) | set(pin["manifest"]) if current.get(n) != pin["manifest"].get(n)})
        return {"pinned": True, "match": not changed, "pinned_hash": pin["hash"], "current": evaluator_hash(current), "changed_files": changed,
                "approved_by": pin["approved_by"], "approved_at": pin["approved_at"], "class": "evaluator", "history": len(pin.get("history", []))}

    def pin(self, approved_by: str, auto: dict[str, Any] | None = None) -> dict[str, Any]:
        """Pin the evaluator as it is now. A named human is required; an owner scope may stand in only when the server says so (`auto`)."""
        who = (approved_by or "").strip()
        if not auto and (len(who) < 2 or who.lower() in SELF_APPROVERS):
            raise PermissionError("pinning an evaluator needs a named human: a model cannot approve changes to its own evaluator")
        current = manifest(self.root)
        old = self._load() or {}
        record = {"hash": evaluator_hash(current), "manifest": current, "approved_by": who or (auto or {}).get("label", ""), "approved_at": now(), "class": "evaluator",
                  "auto": bool(auto), "history": [*old.get("history", []), {"hash": old.get("hash"), "approved_by": old.get("approved_by"), "approved_at": old.get("approved_at")}][-20:]}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return {k: v for k, v in record.items() if k != "manifest"}


class HiddenSet:
    """Run ids reserved for validation. Stored in its own file that no proposal path reads; `exclude` removes them from anything a proposer may see."""

    def __init__(self, directory: Path):
        self.path = Path(directory) / HIDDEN_FILE

    def ids(self) -> set[str]:
        try:
            return set(json.loads(self.path.read_text(encoding="utf-8")).get("run_ids", []))
        except (OSError, ValueError):
            return set()

    def reserve(self, run_ids: list[str], by: str) -> dict[str, Any]:
        who = (by or "").strip()
        if len(who) < 2 or who.lower() in SELF_APPROVERS:
            raise PermissionError("reserving the hidden validation set needs a named human")
        current = self.ids()
        merged = sorted(current | {r for r in run_ids if isinstance(r, str) and r})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"run_ids": merged, "reserved_by": who, "reserved_at": now()}, indent=2) + "\n", encoding="utf-8")
        return {"reserved": len(merged), "added": len(set(merged) - current)}

    def exclude(self, rows: list[dict[str, Any]], key: str = "id") -> list[dict[str, Any]]:
        hidden = self.ids()
        return [r for r in rows if r.get(key) not in hidden]

    def only(self, rows: list[dict[str, Any]], key: str = "id") -> list[dict[str, Any]]:
        hidden = self.ids()
        return [r for r in rows if r.get(key) in hidden]
