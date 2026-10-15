"""Policy Gate: fail-closed authorization. Only PASS with every check satisfied is allowed."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .graph import Graph, Node


SELF_APPROVERS = {"", "claude", "agent", "compiler", "auto", "runtime", "system"}


class GateBlocked(Exception):
    def __init__(self, requested: str, required: str, reason: str):
        self.requested, self.required, self.reason = requested, required, reason
        super().__init__(f"BLOCK {requested}: {reason}; required next step is {required}")


@dataclass
class Decision:
    allowed: bool
    requested: str
    required: str = ""
    reason: str = ""
    checks: dict[str, str] = field(default_factory=dict)

    def raise_if_blocked(self) -> None:
        if not self.allowed:
            raise GateBlocked(self.requested, self.required, self.reason)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Gate:
    def __init__(self, graph: Graph):
        self.graph = graph

    def authorize(self, root: Path, state: dict[str, Any], node: Node) -> Decision:
        """May this step start? Unknown, missing, or tampered state is a BLOCK."""
        checks: dict[str, str] = {}
        expected = state.get("current_step")
        if expected != node.id:
            checks["current_step"] = "BLOCK"
            reason = (f"{node.id} requires {', '.join(node.requires) or 'the initial state'}; "
                      f"current step is {expected or 'none'}")
            return Decision(False, node.id, expected or "none", reason, checks)
        checks["current_step"] = "PASS"
        for required in node.requires:
            record = state["steps"].get(required)
            if not isinstance(record, dict) or record.get("status") != "pass":
                checks[f"prerequisite:{required}"] = "BLOCK"
                return Decision(False, node.id, required, f"prerequisite {required} has not passed", checks)
            artifact = root / "artifacts" / str(record.get("artifact"))
            if not artifact.is_file():
                checks[f"artifact:{required}"] = "MISSING"
                return Decision(False, node.id, required,
                                f"required artifact {record.get('artifact')} is missing", checks)
            if digest(artifact.read_bytes()) != record.get("artifact_hash"):
                checks[f"artifact:{required}"] = "HASH_MISMATCH"
                return Decision(False, node.id, required,
                                f"required artifact {record.get('artifact')} hash does not match state", checks)
            checks[f"prerequisite:{required}"] = "PASS"
        if node.id in state.get("approval_required", []):
            approval = state.get("approvals", {}).get(node.id)
            hashes = {name: state["steps"][name]["artifact_hash"] for name in node.requires}
            if not approval:
                checks["approval"] = "PENDING"
                return Decision(False, node.id, "approval", f"awaiting human approval before {node.id}", checks)
            if approval.get("input_hashes") != hashes:
                checks["approval"] = "STALE"
                return Decision(False, node.id, "approval",
                                f"approval for {node.id} is stale: upstream artifacts changed after it was granted", checks)
            checks["approval"] = "PASS"
        return Decision(True, node.id, checks=checks)

    def check_proposal(self, node: Node, proposal: Any) -> Decision:
        """Is the agent's proposal itself legal for this node? (step identity, tools, forbidden actions)"""
        checks: dict[str, str] = {}
        if not isinstance(proposal, dict):
            return Decision(False, node.id, node.id, "agent proposal must be a JSON object", {"shape": "BLOCK"})
        proposed = proposal.get("step")
        if proposed is not None and proposed != node.id:
            checks["step_identity"] = "BLOCK"
            return Decision(False, node.id, node.id, f"agent proposed {proposed}; runtime requested {node.id}", checks)
        checks["step_identity"] = "PASS"
        log = proposal.get("tool_log") or {}
        used = {call.get("name") for call in log.get("tool_calls", []) if isinstance(call, dict)}
        illegal = sorted(str(name) for name in used - set(node.tools))
        if illegal:
            checks["tools"] = "BLOCK"
            return Decision(False, node.id, node.id,
                            f"policy violation: {node.id} may not use tools {illegal}", checks)
        calls = [c for c in log.get("tool_calls", []) if isinstance(c, dict)]
        if self.graph.policy.get("require_guard") and calls:
            decisions = log.get("guard")
            if not isinstance(decisions, list) or len(decisions) < len(calls):
                checks["guard"] = "MISSING"       # the independent hook did not observe every call: fail closed
                return Decision(False, node.id, node.id,
                                f"policy violation: guard log covers {len(decisions or [])} of {len(calls)} tool calls", checks)
            checks["guard"] = "PASS"
        wanted = node.raw.get("model")
        recorded = log.get("models")
        if wanted and isinstance(recorded, list) and recorded:
            matches = [wanted.lower() in str(m).lower() for m in recorded]
            # a tool-using step may also record a helper model (e.g. WebFetch's summarizer); a tool-less step may not
            if not (any(matches) if node.tools else all(matches)):
                checks["model"] = "BLOCK"
                return Decision(False, node.id, node.id,
                                f"policy violation: {node.id} must run on model '{wanted}' but the CLI recorded {recorded}", checks)
            checks["model"] = "PASS"
        elif wanted:
            checks["model"] = "UNVERIFIED"            # the agent reported no model record, so the policy's model cannot be confirmed
        forbidden = set(self.graph.policy.get("forbidden_actions") or [])
        requested = proposal.get("requested_actions") or []
        if not isinstance(requested, list):
            return Decision(False, node.id, node.id, "requested_actions must be an array", {"actions": "BLOCK"})
        blocked = sorted(str(a) for a in requested if a in forbidden)
        if blocked:
            checks["actions"] = "BLOCK"
            return Decision(False, node.id, node.id, f"policy violation: forbidden actions {blocked}", checks)
        checks["tools"] = checks["actions"] = "PASS"
        return Decision(True, node.id, checks=checks)
