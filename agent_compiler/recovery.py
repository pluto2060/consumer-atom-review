"""Recovery: diagnose a failure, then retry or roll back to a verified state. The compiler decides legality."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .graph import Graph, Node


class RetryExhausted(RuntimeError):
    """A step failed on every allowed attempt (M2709.T3). `attempts` is how many times it ran, `errors` the message of each failure."""

    def __init__(self, attempts: int, errors: list[str]):
        super().__init__(f"step failed {attempts} time(s); last error: {errors[-1] if errors else 'none'}")
        self.attempts, self.errors = attempts, errors


def run_with_retry(step: Callable[[], Any], max_attempts: int, on_halt: Callable[[dict[str, Any]], None] | None = None) -> Any:
    """Call `step()` until it returns, at most `max_attempts` times, then stop with RetryExhausted (never an endless loop).
    Only Exception is retried; KeyboardInterrupt and SystemExit pass through. `on_halt(event)` runs once when the attempts are used up (e.g. `alerts.send`);
    a failing `on_halt` never hides the RetryExhausted."""
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 1:
        raise ValueError("max_attempts must be an integer >= 1")
    errors: list[str] = []
    for _ in range(max_attempts):
        try:
            return step()
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}"[:300])
    if on_halt is not None:
        try:
            on_halt({"kind": "halt", "message": f"step stopped after {max_attempts} failed attempt(s): {errors[-1]}", "attempts": max_attempts})
        except Exception:
            pass
    raise RetryExhausted(max_attempts, errors)


class Recovery:
    def __init__(self, graph: Graph):
        self.graph = graph

    def decide(self, state: dict[str, Any], node: Node, error: str, agent: Any) -> dict[str, Any] | None:
        """Return a retry/repair plan, or None to halt (fail-closed)."""
        attempts = state["steps"][node.id]["attempts"]
        if attempts < node.max_attempts:
            return {"action": "retry", "rerun_from": node.id, "attempts": attempts}
        rule = node.repair
        if rule and state.get("repairs", {}).get(node.id, 0) < rule["max"]:
            target, proposed_by = rule["default"], "policy"
            diagnose = getattr(agent, "diagnose", None)
            if callable(diagnose):
                try:
                    suggestion = diagnose({"failed_step": node.id, "error": error, "allowed_from": rule["allowed_from"],
                                           "steps": self.graph.order})
                except Exception:      # a failing diagnosis never widens authority; the policy default applies
                    suggestion = None
                candidate = suggestion.get("rerun_from") if isinstance(suggestion, dict) else None
                if candidate in rule["allowed_from"]:
                    target, proposed_by = candidate, "agent"
            return {"action": "repair", "rerun_from": target, "proposed_by": proposed_by, "attempts": attempts}
        return None

    def rollback(self, root: Path, state: dict[str, Any], target: str, failed: str, error: str) -> list[str]:
        """Discard artifacts from `target` through `failed`, restoring the last verified state before `target`."""
        order = self.graph.order
        reset = order[order.index(target): order.index(failed) + 1]
        for step_id in reset:
            record = state["steps"][step_id]
            artifact = root / "artifacts" / record["artifact"]
            if artifact.exists():
                artifact.unlink()
            record.update(status="pending", artifact_hash=None, error=None, attempts=0)
        state["repairs"][failed] = state.get("repairs", {}).get(failed, 0) + 1
        state["feedback"][target] = {"failed_step": failed, "error": error[:1500], "rolled_back": reset}
        state["current_step"] = target
        state["status"] = "repairing"
        return reset

    def reset_from(self, root: Path, state: dict[str, Any], target: str) -> list[str]:
        """M2642: a chosen restart point. Discard the artifacts of `target` and every later step so they run again; earlier steps stay committed.
        Unlike `rollback` it is not a repair: no repair is counted and no failure feedback is handed to the agent. Spend already recorded stays recorded."""
        order = self.graph.order
        reset = order[order.index(target):]
        for step_id in reset:
            record = state["steps"][step_id]
            artifact = root / "artifacts" / record["artifact"]
            if artifact.exists():
                artifact.unlink()
            record.update(status="pending", artifact_hash=None, error=None, attempts=0)
            state.get("approvals", {}).pop(step_id, None)
            state.get("feedback", {}).pop(step_id, None)
        state["current_step"] = target
        state["status"] = "ready"
        return reset
