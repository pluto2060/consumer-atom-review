"""What an agent backend can do, and what a policy needs from it.

`claude -p` is the only backend today. Other headless agents (codex, hermes, pi, ...) are expected later; each adapter will
DECLARE its capabilities, and a policy whose needs exceed them is refused before any step runs, instead of failing halfway.

The compiler's rule does not change per backend: only harness-recorded evidence counts. A backend that cannot make the
harness record something a policy relies on (tool calls, fetched pages, the model used) must not run that policy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class AgentCapabilities:
    name: str
    tool_guard: bool = False              # independent, fail-closed enforcement of tool allowlists outside the model
    records_tool_calls: bool = False      # the harness sees every tool call the agent made
    records_pages: bool = False           # ... and the text each call returned, keyed by url (quote grounding)
    records_models: bool = False          # ... and which model actually produced the answer
    records_usage: bool = False           # ... and tokens / cost
    supports_mcp: bool = False            # can be given a declared set of MCP servers
    supports_model_select: bool = False   # honours a per-step model
    parallel_safe: bool = False           # propose() may be called concurrently from several threads
    cancellable: bool = False             # a running call can be stopped

    def describe(self) -> dict[str, Any]:
        return asdict(self)


def requirements(policy: dict[str, Any], steps: list[dict[str, Any]]) -> list[tuple[str, str, bool]]:
    """(capability, why, hard) for the policy as it will actually run (`steps` are the effective per-step dicts)."""
    needs: dict[str, tuple[str, bool]] = {}

    def need(capability: str, why: str, hard: bool) -> None:
        if capability not in needs or (hard and not needs[capability][1]):
            needs[capability] = (why, hard)

    quote_policy = bool((policy.get("collect") or {}).get("require_web_quote"))
    for step in steps:
        tools = step.get("tools") or []
        sid = step["id"]
        if tools:
            need("records_tool_calls", f"step {sid} uses tools, and only harness-recorded calls count", True)
            if policy.get("require_guard"):
                need("tool_guard", "the policy sets require_guard: a step with tools needs an independent guard", True)
            else:
                need("tool_guard", f"step {sid} uses tools without an independent guard", False)
            if quote_policy or step.get("validator") in ("intent.observe", "radar.collect") or step.get("mcp_servers"):
                need("records_pages", f"step {sid} quotes are checked against the text the harness recorded", True)
        if step.get("mcp_servers"):
            need("supports_mcp", f"step {sid} uses MCP servers {sorted(step['mcp_servers'])}", True)
        if step.get("model"):
            need("supports_model_select", f"step {sid} pins model '{step['model']}'", True)
            need("records_models", f"the gate can only confirm model '{step['model']}' if the backend reports the model used", False)
        if step.get("parallel") is True:
            need("parallel_safe", f"step {sid} is parallel: its proposal may be requested concurrently", False)
    if policy.get("budget") or any(s.get("max_budget_usd") for s in steps):
        need("records_usage", "the policy sets a spending budget, which needs recorded spend to enforce", True)
    termination = ((policy.get("termination") or {}).get("condition") or "")
    if termination.split(" ")[0] in ("cost_usd", "tokens_total"):
        need("records_usage", f"the termination condition '{termination}' needs recorded usage", True)
    elif (policy.get("observability") or {}).get("trace", {}).get("tokens") or (policy.get("observability") or {}).get("trace", {}).get("cost"):
        need("records_usage", "the policy asks for tokens/cost in the trace", False)
    return [(cap, why, hard) for cap, (why, hard) in needs.items()]


def check(policy: dict[str, Any], steps: list[dict[str, Any]], caps: AgentCapabilities) -> dict[str, list[str]]:
    """{'missing_hard': [...], 'missing_soft': [...]} — hard means the policy cannot run on this backend."""
    result: dict[str, list[str]] = {"missing_hard": [], "missing_soft": []}
    for capability, why, hard in requirements(policy, steps):
        if not getattr(caps, capability, False):
            result["missing_hard" if hard else "missing_soft"].append(f"{capability}: {why}")
    return result
