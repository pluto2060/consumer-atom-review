"""Typed execution graph produced by the policy compiler."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

KNOWN_TOOLS = ("WebSearch", "WebFetch")


@dataclass(frozen=True)
class Node:
    """One executable workflow step: contracts, gate inputs, validators, and transitions."""

    id: str
    requires: tuple[str, ...]
    output: str
    next: tuple[str, ...]
    description: str = ""
    instruction: str = ""
    contract: dict[str, Any] = field(default_factory=dict)      # agent/output contract shown to Claude
    schema: dict[str, str] = field(default_factory=dict)        # required top-level fields -> type
    validator: str | None = None
    tools: tuple[str, ...] = ()                                 # tools Claude may use for this step only
    max_turns: int = 1
    timeout: int = 180
    max_attempts: int = 1
    repair: dict[str, Any] | None = None                        # recovery rule: allowed_from/default/max
    raw: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    @property
    def parallel(self) -> bool:
        """True if this step may have its agent proposal obtained concurrently with other ready parallel steps."""
        return self.raw.get("parallel") is True

    @property
    def actor(self) -> str | None:
        return self.raw.get("actor")

    def to_step(self) -> dict[str, Any]:
        """Dict view handed to agents (kept close to the policy source)."""
        step = dict(self.raw)
        step.update(id=self.id, requires=list(self.requires), output=self.output,
                    next=list(self.next), description=self.description)
        return step


@dataclass
class Graph:
    name: str
    version: str
    objective: str
    nodes: dict[str, Node]
    policy: dict[str, Any]

    @property
    def order(self) -> list[str]:
        return list(self.nodes)

    @property
    def first(self) -> str:
        return next(iter(self.nodes))

    def node(self, step_id: str) -> Node:
        try:
            return self.nodes[step_id]
        except KeyError:
            raise KeyError(f"unknown step: {step_id}") from None

    def ancestors(self, step_id: str) -> list[str]:
        """Declared-order steps up to and excluding step_id."""
        order = self.order
        return order[:order.index(step_id)]

    def describe(self) -> list[dict[str, Any]]:
        return [{"id": n.id, "requires": list(n.requires), "output": n.output, "next": list(n.next),
                 "validator": n.validator or "legacy", "tools": list(n.tools),
                 "max_attempts": n.max_attempts, "repair": n.repair, "actor": n.actor,
                 "action": n.raw.get("action"), "parallel": n.parallel} for n in self.nodes.values()]
