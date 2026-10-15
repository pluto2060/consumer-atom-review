"""Conditional pre-approval of run gates (M2602): a named human approves, BEFORE a run starts, the approval steps it will stop at, under conditions.

A policy run stops at an approval step until a named human approves it, bound to the artifacts they reviewed. That is the right default, but it makes
every run wait for a person even when the person already knows what they would accept. A pre-approval moves that decision to the start of the run
without making it blind: the named human states, per approval step, the CONDITIONS under which the step may start (a field of an upstream artifact must
equal or be one of given values, a list must stay within a size, a field must exist), the compiler records who said so in the run's hash-chained trace,
and when the run reaches the gate it evaluates the conditions on the artifacts that really exist. Met: the step is approved under the human's name
("pre-approved at run start") bound to the same artifact hashes as any approval. Not met: nothing is approved and the run waits for a human, exactly as
without a pre-approval. A pre-approval with no condition is refused (that would be approving whatever arrives), and the model never chooses who approved."""
from __future__ import annotations

from typing import Any, Callable

from .gate import SELF_APPROVERS

OPS = ("==", "!=", "in", "<=", ">=", "count<=", "count>=", "exists")
MAX_CONDITIONS = 12


class PreApprovalInvalid(ValueError):
    pass


def _get(data: Any, path: str) -> tuple[bool, Any]:
    cur = data
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return False, None
    return True, cur


def validate(items: Any, approval_required: list[str], steps: Any) -> list[dict[str, Any]]:
    """Normalised pre-approvals; raises PreApprovalInvalid. `steps` is any container of the policy's step ids."""
    if items is None:
        return []
    if not isinstance(items, list):
        raise PreApprovalInvalid("pre_approvals must be a list")
    out: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict) or set(item) - {"step", "by", "conditions", "note"}:
            raise PreApprovalInvalid("a pre-approval is {step, by, conditions, note?}")
        step, by = item.get("step"), item.get("by")
        if step not in approval_required:
            raise PreApprovalInvalid(f"{step!r} is not an approval step of this run {sorted(approval_required)}")
        if not isinstance(by, str) or by.strip().lower() in SELF_APPROVERS or len(by.strip()) < 2:
            raise PreApprovalInvalid("a named human is required to pre-approve; agents cannot")
        conditions = item.get("conditions")
        if not isinstance(conditions, list) or not 1 <= len(conditions) <= MAX_CONDITIONS:
            raise PreApprovalInvalid(f"conditions must be a list of 1-{MAX_CONDITIONS} (a pre-approval without a condition would approve anything)")
        clean = []
        for c in conditions:
            if not isinstance(c, dict) or set(c) - {"step", "path", "op", "value"} or c.get("step") not in steps \
                    or not isinstance(c.get("path"), str) or not c["path"] or c.get("op") not in OPS:
                raise PreApprovalInvalid(f"a condition is {{step (a step of the policy), path, op in {list(OPS)}, value}}, got {str(c)[:100]}")
            if c["op"] == "in" and not isinstance(c.get("value"), list):
                raise PreApprovalInvalid("op in needs a list value")
            if c["op"] in ("<=", ">=", "count<=", "count>=") and (isinstance(c.get("value"), bool) or not isinstance(c.get("value"), (int, float))):
                raise PreApprovalInvalid(f"op {c['op']} needs a number")
            clean.append({"step": c["step"], "path": c["path"], "op": c["op"], **({"value": c["value"]} if "value" in c else {})})
        out.append({"step": step, "by": by.strip(), "conditions": clean, "note": str(item.get("note", ""))[:300]})
    return out


def unmet(conditions: list[dict[str, Any]], artifact_of: Callable[[str], Any]) -> list[str]:
    """Why the conditions are not all met on the artifacts that exist (empty list = all met). Fail closed: a missing artifact or path is a failure."""
    problems = []
    for c in conditions:
        data = artifact_of(c["step"])
        found, value = _get(data, c["path"]) if isinstance(data, dict) else (False, None)
        label = f"{c['step']}.{c['path']} {c['op']} {c.get('value')!r}"
        if c["op"] == "exists":
            if not found:
                problems.append(f"{label}: not present")
            continue
        if not found:
            problems.append(f"{label}: not present in the committed artifact")
            continue
        op, want = c["op"], c.get("value")
        if op.startswith("count"):
            if not isinstance(value, (list, dict)):
                problems.append(f"{label}: is not a list")
                continue
            value, op = len(value), op[len("count"):]
        ok = {"==": lambda: value == want, "!=": lambda: value != want, "in": lambda: value in want,
              "<=": lambda: isinstance(value, (int, float)) and value <= want, ">=": lambda: isinstance(value, (int, float)) and value >= want}[op]()
        if not ok:
            problems.append(f"{label}: actual {value!r}")
    return problems
