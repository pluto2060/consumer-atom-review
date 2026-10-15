"""Swappable layers. The defaults are self-contained; adapters (AgentLedger, CPEX, ...) implement the same protocols.

Compiler -> ToolGuard (what may run) -> Agent (reason/propose) -> AuditSink (what happened) -> DurableStore (where state lives)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable


class RunCancelled(Exception):
    """A human asked to stop the run. Not a failure: the step is left pending so the run can be resumed."""


@runtime_checkable
class Agent(Protocol):
    """Reasoning engine. Proposes; never authorizes."""

    def propose(self, step: dict[str, Any], intent: str, inputs: dict[str, Any]) -> dict[str, Any]: ...


@runtime_checkable
class ToolGuard(Protocol):
    """Independent tool-use enforcement installed around an agent call (e.g. a Claude Code PreToolUse hook)."""

    def settings(self, directory: Path, tools: tuple[str, ...], mcp: dict[str, Any] | None = None) -> Path | None:
        """Write per-call configuration into `directory` and return the settings file (None = no guard).
        `mcp` maps each MCP server used by the step to its rules (allow_operations, operation_field, allow_hosts)."""

    def decisions(self, directory: Path) -> list[dict[str, Any]]:
        """Return the guard's own decision log for the call (must be independent of the agent's claims)."""


@runtime_checkable
class AuditSink(Protocol):
    """Append-only, verifiable record of gate decisions and transitions."""

    def append(self, root: Path, event: dict[str, Any]) -> dict[str, Any]: ...
    def read(self, root: Path) -> list[dict[str, Any]]: ...
    def verify(self, root: Path, expected_head: str | None = None) -> list[dict[str, Any]]: ...
    def head(self, root: Path) -> dict[str, Any] | None: ...


@runtime_checkable
class DurableStore(Protocol):
    """Where run state lives. The executor only uses this surface."""

    def root(self, run_id: str) -> Path: ...
    def create(self, graph: Any, intent: str) -> tuple[Path, dict[str, Any]]: ...
    def load(self, run_id: str) -> dict[str, Any]: ...
    def save(self, root: Path, state: dict[str, Any]) -> None: ...
    def record_transition(self, state: dict[str, Any], event: str, source: str | None, target: str | None) -> None: ...
