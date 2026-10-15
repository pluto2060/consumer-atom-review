"""Executor: orchestrates gate -> agent proposal -> verification -> commit -> trace, with recovery and planning."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Protocol

from .fsutil import replace_atomic
from .gate import SELF_APPROVERS, Decision, Gate, GateBlocked, digest
from . import cost as cost_mod, integrity, pre_approval
from .graph import Graph, Node
from .policy_compiler import builtin_governance, builtin_sources, load_graph, required_approvals
from . import capabilities as capabilities_mod
from . import principle_checks
from .principle_policy import advise_lines
from .interfaces import AuditSink, DurableStore, RunCancelled
from .recovery import Recovery
from .state import StateStore, now
from .trace import Trace
from .verifier import ArtifactInvalid, Verifier


HOLD_ONLY_MARK = "; without it the tree would be under its cap"

class Agent(Protocol):
    def propose(self, step: dict[str, Any], intent: str, inputs: dict[str, Any]) -> dict[str, Any]: ...


def _cost_of(usage: Any) -> float:
    return float(usage["cost_usd"]) if isinstance(usage, dict) and isinstance(usage.get("cost_usd"), (int, float)) and not isinstance(usage.get("cost_usd"), bool) else 0.0


def _add_usage(store: dict[str, Any], step_id: str, usage: Any) -> None:
    if not isinstance(usage, dict):
        return
    total = store.setdefault(step_id, {})
    for key, value in usage.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            total[key] = round(total.get(key, 0) + value, 6)
    total["attempts_counted"] = total.get("attempts_counted", 0) + 1


def _accepted_log(tool_log: dict[str, Any], pooled: dict[str, Any], data: Any) -> dict[str, Any]:
    """Tool log stored with an accepted artifact: this attempt's record plus pages pooled from earlier attempts,
    keeping page text only for urls the artifact cites (replay re-verifies against exactly this log)."""
    cited = json.dumps(data, ensure_ascii=False)
    pages = {u: t for u, t in pooled["pages"].items() if u == "_search" or u in cited}
    return {**tool_log, "urls_seen": pooled["urls_seen"], "pages": pages}


def _hash_json(value: Any) -> str:
    return digest(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8"))


class GatedWorkflowRuntime:
    """State lives outside the model context and gates every step invocation (fail-closed)."""

    def __init__(self, policy_path: Path, runtime_dir: Path, audit: AuditSink | None = None,
                 store: DurableStore | None = None, principles: list[dict[str, Any]] | None = None,
                 principle_loader: Callable[[str, str], dict[str, Any]] | None = None):
        self.principles = list(principles or [])          # ACTIVE principle policies for new runs (advise/evaluate)
        self.principle_loader = principle_loader          # (id, version) -> document with approved proxies applied
        self.policy_path = Path(policy_path)
        self.graph: Graph = load_graph(self.policy_path)
        self.policy = self.graph.policy
        self.runtime_dir = Path(runtime_dir)
        self.store: DurableStore = store or StateStore(self.runtime_dir)
        self.audit: AuditSink = audit or Trace()
        self.gate, self.verifier, self.recovery = Gate(self.graph), Verifier(self.graph), Recovery(self.graph)
        self._locks: dict[str, threading.RLock] = {}
        self._locks_guard = threading.Lock()
        self._live_lock = threading.Lock()
        self._cancels: dict[str, threading.Event] = {}
        self._prefetched: dict[tuple[str, str], tuple[str, Any, float]] = {}     # (run, step) -> (input hash, proposal, seconds)
        self._prefetch_lock = threading.Lock()
        self._caps_checked: set[str] = set()                              # runs whose agent capabilities were already checked
        self._waves: dict[str, dict[str, Any]] = {}                       # run -> {"steps": [...], "begin_seq": int}

    def principles_of(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        """The principle policies a run was created with (from its snapshot). Runs without a snapshot (created before the
        feature, or by an embedding that passes `principles=`) fall back to the runtime's active list."""
        refs = state.get("principles")
        if refs is None or self.principle_loader is None:
            return self.principles if refs is None else [d for d in self.principles if any(d["id"] == r["id"] and d["version"] == r["version"] for r in refs)]
        docs = []
        for ref in refs:
            try:
                docs.append(self.principle_loader(ref["id"], ref["version"]))
            except (KeyError, OSError, ValueError):
                continue                                    # a removed policy simply stops applying; never blocks a run
        return docs

    def _run_lock(self, run_id: str) -> threading.RLock:
        """One lock per run: two runs may execute concurrently, one run never runs two steps at once."""
        with self._locks_guard:
            return self._locks.setdefault(run_id, threading.RLock())

    # ---- live progress, cancellation, run listing -------------------------------------------------

    def register_cancel(self, run_id: str, event: threading.Event) -> None:
        self._cancels[run_id] = event

    def unregister_cancel(self, run_id: str) -> None:
        self._cancels.pop(run_id, None)

    def _cancelled(self, run_id: str) -> bool:
        event = self._cancels.get(run_id)
        return bool(event and event.is_set())

    def _live(self, root: Path, step: str | None, kind: str, **fields: Any) -> None:
        """Best-effort progress feed for humans (not the audit trail: not chained, never authoritative)."""
        try:
            line = json.dumps({"at": now(), "step": step, "kind": kind, **fields}, ensure_ascii=False, default=str)
            with self._live_lock, (root / "live.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
        except OSError:
            pass

    def live_events(self, run_id: str, after: int = 0) -> tuple[list[dict[str, Any]], int]:
        """Progress events with index > `after`, and the next cursor."""
        path = self.store.root(run_id) / "live.jsonl"
        if not path.exists():
            return [], 0
        lines = path.read_text(encoding="utf-8").splitlines()
        return [{"i": i + 1, **json.loads(line)} for i, line in enumerate(lines) if i >= max(0, after)], len(lines)

    def archive_unused(self) -> list[str]:
        """Move runs that were created but never started into `<runtime>/archive/` (reversible; nothing is deleted).

        A run is unused only if every step is still pending with zero attempts, the run is `ready`, nothing has run
        (no approvals, no measurements) and its trace holds nothing but the creation event."""
        runs_dir, target = self.runtime_dir / "runs", self.runtime_dir / "archive"
        if not runs_dir.exists():
            return []
        moved: list[str] = []
        for path in sorted(runs_dir.iterdir()):
            if not path.is_dir():
                continue
            with self._run_lock(path.name):
                try:
                    state = self.store.load(path.name)
                except (KeyError, ValueError, OSError):
                    continue
                untouched = (state["status"] == "ready"
                             and all(s["status"] == "pending" and not s.get("attempts") for s in state["steps"].values())
                             and not state.get("approvals") and not state.get("measurements"))
                events = [e for e in self.audit.read(path) if e.get("event") != "run_created"] if untouched else [None]
                if untouched and not events and path.name not in self._cancels:
                    target.mkdir(parents=True, exist_ok=True)
                    path.rename(target / path.name)
                    moved.append(path.name)
        return moved

    def list_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        runs_dir = self.runtime_dir / "runs"
        if not runs_dir.exists():
            return []
        found = []
        for path in sorted(runs_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]:
            try:
                state = self.store.load(path.name)
            except (KeyError, ValueError, OSError):
                continue
            found.append({"id": state["id"], "status": state["status"], "current_step": state.get("current_step"),
                          "policy": state["policy"], "created_at": state["created_at"], "updated_at": state["updated_at"],
                          "intent": state["intent"][:120],
                          "passed": sum(1 for s in state["steps"].values() if s["status"] == "pass"),
                          "total": len(state["steps"])})
        return found

    # ---- runs ---------------------------------------------------------------------------------

    def create_run(self, intent: str, approval_required: list[str] | None = None, lineage: dict[str, Any] | None = None,
                   pre_approvals: list[dict[str, Any]] | None = None, task_ref: str | None = None,
                   envelope_cap_usd: float | None = None, request_cap_usd: float | None = None) -> dict[str, Any]:
        """`request_cap_usd` (M2664.9): the caller's own spending cap for THIS run in every permission mode; it only tightens (the run's cap is the lower of it and the
        policy's own `run_usd`), so a request above the policy cap changes nothing and a request for a policy without a cap makes the request the cap.
        `approval_required`: steps that must be approved by a named human before they may start.
        `lineage` ({parent, root, depth}) is set only by the server after it verified the parent and the depth: a run started on behalf of
        another run records who started it, so a nested tree of runs can be audited and bounded."""
        intent = intent.strip() if isinstance(intent, str) else ""
        if not intent or len(intent) > 40000:
            raise ValueError("intent must be a non-empty string up to 40,000 characters")
        visible = [c for c in intent if not c.isspace()]
        if len(visible) >= 8 and visible.count("?") / len(visible) > 0.25:
            raise ValueError("intent looks corrupted by an encoding error (many '?' placeholders); send the request body as UTF-8")
        approval_required = list(approval_required or [])
        if any(step not in self.graph.nodes for step in approval_required):
            raise ValueError(f"approval_required names unknown steps: {sorted(set(approval_required) - set(self.graph.nodes))}")
        if len(set(approval_required)) != len(approval_required):
            raise ValueError("approval_required lists the same step more than once")
        missing = [step for step in required_approvals(self.policy) if step not in approval_required]
        if missing:
            raise ValueError(f"governance requires approval for steps {missing}: list them in approval_required when creating the run")
        try:
            pre = pre_approval.validate(pre_approvals, approval_required, self.graph.nodes)
        except pre_approval.PreApprovalInvalid as exc:
            raise ValueError(str(exc)) from None
        if request_cap_usd is not None and (isinstance(request_cap_usd, bool) or not isinstance(request_cap_usd, (int, float)) or not request_cap_usd > 0):
            raise ValueError("max_usd must be a positive number")
        root, state = self.store.create(self.graph, intent, approval_required)
        if request_cap_usd is not None:
            state["request_cap_usd"] = round(float(request_cap_usd), 6)
        if lineage:
            state["lineage"] = dict(lineage)
        if pre:
            state["pre_approvals"] = pre
        if envelope_cap_usd is not None and not isinstance((self.policy.get("budget") or {}).get("run_usd"), (int, float)):
            state["envelope_cap_usd"] = float(envelope_cap_usd)               # only for a policy with no cap of its own
        if isinstance(task_ref, str) and task_ref.strip():
            state["task_ref"] = task_ref.strip()[:120]                        # joins the run to selection_log and the work report (M2629)
        state["evaluator_hash"] = integrity.evaluator_hash()                   # which evaluator code judged this run (M2622 T9)
        state["principles"] = [{"id": d["id"], "version": d["version"], "source_sha256": d["source"]["sha256"]} for d in self.principles]
        state["checks"] = [c for c in principle_checks.frozen_checks(self.principles) if c["step"] in self.graph.nodes]
        (root / f"policy.snapshot{self.policy_path.suffix}").write_bytes(self.policy_path.read_bytes())   # exact text this run began under
        self.store.save(root, state)                       # the run is evaluated against the principles it was created with
        self.audit.append(root, {"event": "run_created", "run_id": state["id"], "policy": self.graph.name,
                            "policy_version": self.graph.version, "principles": state["principles"], "evaluator_hash": state["evaluator_hash"],
                            **({"task_ref": state["task_ref"]} if state.get("task_ref") else {}), "use": self.policy.get("use", "internal"),
                            "policy_hash": digest(self.policy_path.read_bytes()),
                            **({"lineage": state["lineage"]} if lineage else {})})
        for item in pre:
            self.audit.append(root, {"event": "pre_approval_recorded", **item})
        return state

    def create_branch_run(self, source: "GatedWorkflowRuntime", source_run_id: str, through_step: str, intent: str | None = None,
                          approval_required: list[str] | None = None, lineage: dict[str, Any] | None = None, task_ref: str | None = None,
                          envelope_cap_usd: float | None = None, by: str = "", request_cap_usd: float | None = None) -> dict[str, Any]:
        """M2642: a NEW run whose steps up to `through_step` are copied byte for byte from a finished run of `source` (the same or another policy
        version). Nothing is trusted: the source run is verified whole first, the copy is replayed under THIS policy's validators and THIS run's intent
        (a changed intent or a stricter validator blocks the reuse), approvals are not copied, spend of the copied steps is not counted again. A failed
        replay leaves the new run `terminated` with the reason, never half-built."""
        order, src_order = self.graph.order, source.graph.order
        if through_step not in order:
            raise ValueError(f"unknown step '{through_step}'; this policy has: {', '.join(order)}")
        cut = order.index(through_step) + 1
        if src_order[:cut] != order[:cut]:
            raise ValueError(f"the steps up to {through_step} differ between the source policy ({', '.join(src_order[:cut])}) and this one ({', '.join(order[:cut])})")
        src_state = source._require_run(source_run_id)
        kept = order[:cut]
        not_done = [n for n in kept if src_state["steps"][n]["status"] != "pass"]
        if not_done:
            raise GateBlocked("BRANCH", "committed steps", f"source steps not committed: {', '.join(not_done)}")
        if any(rec["status"] == "running" for rec in src_state["steps"].values()):
            raise GateBlocked("BRANCH", "state", "the source run has a step still running")
        gated = [n for n in kept if n in (src_state.get("approval_required") or []) or n in (src_state.get("approvals") or {})]
        if gated:
            raise GateBlocked("BRANCH", "approvals", f"steps with approvals cannot be reused (branch before them): {', '.join(gated)}")
        problems = source.verify_run(source_run_id)
        if problems:
            raise GateBlocked("BRANCH", "integrity", f"source run has {len(problems)} integrity problem(s): {problems[0]['reason']}")
        state = self.create_run(intent or src_state["intent"], approval_required, lineage=lineage, task_ref=task_ref, envelope_cap_usd=envelope_cap_usd, request_cap_usd=request_cap_usd)
        run_id, root, src_root = state["id"], self.store.root(state["id"]), source.store.root(source_run_id)
        (root / "artifacts").mkdir(exist_ok=True)
        hashes = {}
        for name in kept:
            rec = src_state["steps"][name]
            (root / "artifacts" / rec["artifact"]).write_bytes((src_root / "artifacts" / rec["artifact"]).read_bytes())
            state["steps"][name].update(status="pass", artifact_hash=rec["artifact_hash"], error=None, attempts=0)
            hashes[name] = rec["artifact_hash"]
        reused_cost = round(sum(float(((src_state.get("usage") or {}).get(n) or {}).get("cost_usd", 0)) for n in kept), 6)
        state["branch"] = {"from_run": source_run_id, "through_step": through_step, "source_policy": source.graph.name, "source_policy_version": source.graph.version,
                           "source_hashes": hashes, "reused_cost_usd": reused_cost, "by": (by or "api")[:80]}
        state["current_step"] = order[cut] if cut < len(order) else order[-1]
        state["status"] = "ready" if cut < len(order) else "completed"
        self.store.save(root, state)
        failures = self.check_integrity(run_id)
        if failures:                                               # the copy does not hold under this policy and intent: stop, keep the record
            state["status"] = "terminated"
            state["branch"]["refused"] = failures[0]["reason"][:400]
            self.store.save(root, state)
            self.audit.append(root, {"event": "branch_refused", "from_run": source_run_id, "through_step": through_step, "problems": failures[:5]})
            raise GateBlocked("BRANCH", "replay", f"the copied steps do not verify under this policy and intent: {failures[0]['step']}: {failures[0]['reason'][:200]}")
        self.audit.append(root, {"event": "run_branched", "from_run": source_run_id, "through_step": through_step, "source_policy": source.graph.name,
                                 "source_policy_version": source.graph.version, "reused_cost_usd": reused_cost, "by": (by or "api")[:80]})
        for name in kept:
            self.audit.append(root, {"event": "step_reused", "step": name, "artifact_hash": hashes[name], "from_run": source_run_id})
        self._live(root, through_step, "run_branched", from_run=source_run_id)
        return self.get_run(run_id) or state

    def add_pre_approval(self, run_id: str, item: dict[str, Any]) -> dict[str, Any]:
        """Record one more pre-approval on a run that has not passed (or been approved at) that step yet."""
        with self._run_lock(run_id):
            root = self.store.root(run_id)
            state = self._require_run(run_id)
            try:
                (clean,) = pre_approval.validate([item], state.get("approval_required", []), self.graph.nodes)
            except pre_approval.PreApprovalInvalid as exc:
                raise ValueError(str(exc)) from None
            if state["steps"][clean["step"]]["status"] == "pass" or clean["step"] in state.get("approvals", {}):
                raise ValueError(f"{clean['step']} is already approved or has already run; a pre-approval must come before it")
            state.setdefault("pre_approvals", []).append(clean)
            self.store.save(root, state)
            self.audit.append(root, {"event": "pre_approval_recorded", **clean})
            return self.get_run(run_id) or state

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        try:
            state = self.store.load(run_id)
        except KeyError:
            return None
        root = self.store.root(run_id)
        state["trace"] = self.audit.read(root)
        state["measurements"] = self.measurements(run_id)
        return state

    def _require_run(self, run_id: str) -> dict[str, Any]:
        return self.store.load(run_id)

    # ---- single step ----------------------------------------------------------------------------

    def run_next(self, run_id: str, agent: Agent) -> dict[str, Any]:
        with self._run_lock(run_id):
            state = self._require_run(run_id)
            if not state["current_step"]:
                raise GateBlocked("none", "none", "run has no remaining step")
            return self.run_step(run_id, state["current_step"], agent)

    def _enforce_termination(self, run_id: str, root: Path, state: dict[str, Any], step: str) -> None:
        """Opt-in `termination.enforce`: once a committed step makes the condition true, the remaining steps are not run.
        Pure function of the committed trace; a measure with no value counts as false (terminating always needs evidence)."""
        cfg = self.policy.get("termination") or {}
        if cfg.get("enforce") is not True or not cfg.get("condition"):
            return
        from .evaluation import run_measures
        from .policy_compiler import COMPARATORS, parse_condition
        measure, comparator, threshold = parse_condition(cfg["condition"])
        value = run_measures(self, run_id).get(measure)
        if value is None or not COMPARATORS[comparator](value, threshold):
            return
        fresh = self._require_run(run_id)
        trace_seq = len(self.audit.read(root))
        fresh["status"], fresh["current_step"] = "terminated", None
        fresh["termination"] = {"step": step, "condition": cfg["condition"], "measure": measure, "value": value,
                                "threshold": threshold, "at": now(), "trace_seq": trace_seq}
        self.store.record_transition(fresh, "terminated", step, None)
        self.store.save(root, fresh)
        self.audit.append(root, {"event": "termination", "step": step, **fresh["termination"]})
        self._live(root, step, "terminated", measure=measure, value=value)
        state.update(fresh)

    def run_step(self, run_id: str, step_id: str, agent: Agent) -> dict[str, Any]:
        with self._run_lock(run_id):
            root = self.store.root(run_id)
            state = self._require_run(run_id)
            node = self.graph.node(step_id)
            decision = self.gate.authorize(root, state, node)
            if not decision.allowed:
                self.audit.append(root, {"event": "gate_block", "step": node.id, "stage": "authorize",
                                    "reason": decision.reason, "required_step": decision.required,
                                    "policy_check": decision.checks, "agent": _agent_name(agent)})
                if decision.checks.get("approval") in ("PENDING", "STALE"):
                    self._await_approval(root, state, node, decision)
                decision.raise_if_blocked()
            self._require_capabilities(agent, run_id, root, state)
            inputs = self._read_inputs(root, state, node)
            record = state["steps"][node.id]
            record.update(status="running", error=None, attempts=record.get("attempts", 0) + 1)
            state["status"] = "running"
            self.store.save(root, state)
            started = time.monotonic()
            self.audit.append(root, {"event": "step_started", "step": node.id, "attempt": record["attempts"],
                                "input_artifacts": [k for k in inputs if not k.startswith("_")],
                                "input_hashes": {k: state["steps"][k]["artifact_hash"] for k in node.requires},
                                "has_feedback": "_feedback" in inputs})
            stage, proposal, proposal_hash, checks = "agent", None, None, dict(decision.checks)
            reservation: str | None = None
            self._live(root, node.id, "step_started", attempt=record["attempts"],
                       feedback=("_feedback" in inputs), tools=list(node.tools))
            try:
                self._bind(agent, root, node.id, run_id)
                try:
                    step = self._step_for(node, state)
                    self._enforce_budget(root, state, node, step)
                    reservation = self._reserve(state, node.id, step.get("max_budget_usd"))
                    proposal, waited = self._take_prefetched(run_id, node.id, inputs)
                    if proposal is None:
                        proposal = agent.propose(step, state["intent"], inputs)
                    else:
                        started -= waited        # the agent's time was spent concurrently; still count it as this step's latency
                finally:
                    self._bind(agent, None, node.id, run_id)
                proposal_hash = _hash_json(proposal)
                self._record_usage(root, state, node.id, (proposal.get("tool_log") or {}).get("usage")
                                   if isinstance(proposal, dict) else None, reservation)       # every attempt counts, passed or not
                reservation = None
                stage = "proposal"
                verdict = self.gate.check_proposal(node, proposal)
                checks.update(verdict.checks)
                verdict.raise_if_blocked()
                data = proposal.get("data", {k: v for k, v in proposal.items() if k not in ("step", "tool_log", "requested_actions")})
                tool_log = proposal.get("tool_log") or {}
                stage = "verify"
                pool = state.setdefault("pool", {}).setdefault(node.id, {"urls_seen": [], "pages": {}})
                pooled = {"urls_seen": sorted(set(pool["urls_seen"]) | set(tool_log.get("urls_seen", []))),
                          "pages": {**pool["pages"], **tool_log.get("pages", {})}}
                try:
                    checks.update(self.verifier.verify(node, data, inputs, {"intent": state["intent"],
                                                                            "tool_log": {**tool_log, **pooled}}))
                    results = principle_checks.evaluate(state.get("checks", []), node.id, data)
                    for result in results:            # every evaluation is traced, including clean passes
                        self.audit.append(root, {"event": "principle_check", "step": node.id, **result})
                    blocking = [f"[{r['policy']}:{r['element']}] {v}" for r in results if r["mode"] == "enforce" for v in r["violations"]]
                    if blocking:
                        raise ValueError("principle check failed (enforced): " + " | ".join(blocking[:8]))
                    checks["principle_checks"] = ({r["element"]: ("violated" if r["violations"] else "ok") + f"/{r['mode']}" for r in results}
                                                  or "none")
                except Exception:
                    state["pool"][node.id] = pooled       # pages fetched in a failed attempt stay usable for the retry
                    state.setdefault("feedback_data", {})[node.id] = data
                    raise
                envelope = {"run_id": run_id, "step": node.id,
                            "input_hashes": {n: state["steps"][n]["artifact_hash"] for n in node.requires},
                            "created_at": now(), "data": data,
                            "audit": {"agent": _agent_name(agent), "proposal_hash": proposal_hash,
                                      "attempt": record["attempts"],
                                      "tool_log": _accepted_log(tool_log, pooled, data)}}
                serialized = (json.dumps(envelope, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
                artifact = root / "artifacts" / node.output
                temp = artifact.with_suffix(artifact.suffix + ".tmp")
                temp.write_bytes(serialized)
                replace_atomic(temp, artifact)
                record.update(status="pass", artifact_hash=digest(serialized), error=None)
                state["feedback"].pop(node.id, None)
                state.get("pool", {}).pop(node.id, None)
                state.get("feedback_data", {}).pop(node.id, None)
                _notify(agent, node.id, True, "")
                following = node.next[0] if node.next else None
                state["current_step"] = following
                state["status"] = "ready" if following else "completed"
                self.store.record_transition(state, "pass", node.id, following)
                self.store.save(root, state)
                self.audit.append(root, {"event": "gate_pass", "step": node.id, "agent": _agent_name(agent),
                                    "proposal_hash": proposal_hash, "policy_check": checks,
                                    "artifact": node.output, "artifact_hash": record["artifact_hash"],
                                    "verification": "PASS", "transition": following or "COMPLETE",
                                    "latency_ms": round((time.monotonic() - started) * 1000),
                                    **self._usage_fields(state.get("usage", {}).get(node.id))})
                self._live(root, node.id, "step_passed", latency_ms=round((time.monotonic() - started) * 1000),
                           transition=following or "COMPLETE")
                self._close_wave(run_id, root, state)
                if following:
                    self._enforce_termination(run_id, root, state, node.id)
            except RunCancelled:
                if reservation is not None and self.ledger is not None:
                    self.ledger.settle(reservation, None)                 # a killed call's cost is unknown: keep the whole reservation
                record.update(status="pending", error=None, attempts=max(0, record["attempts"] - 1))
                state["status"], state["current_step"] = "ready", node.id
                self.store.record_transition(state, "cancel", node.id, node.id)
                self.store.save(root, state)
                self.audit.append(root, {"event": "run_cancelled", "step": node.id, "stage": stage})
                self._live(root, node.id, "cancelled")
                raise
            except Exception as exc:
                reason = str(exc)[:2000]
                if proposal is None:                  # the call itself failed (budget stop, error result): it still cost what the CLI reported
                    self._record_usage(root, state, node.id, getattr(exc, "usage", None), reservation)
                    reservation = None
                self._live(root, node.id, "step_failed", stage=stage, error=reason[:300])
                if stage == "verify":
                    checks["verification"] = "FAIL"
                elif stage == "proposal":
                    checks["proposal"] = "BLOCK"
                else:
                    checks["agent"] = "ERROR"          # timeouts and agent errors are never treated as success
                record.update(status="failed", error=reason)
                _notify(agent, node.id, False, reason)
                state["status"] = "failed"
                state["current_step"] = node.id
                state["feedback"][node.id] = {"failed_step": node.id, "error": reason, "attempt": record["attempts"]}
                self.store.record_transition(state, "fail", node.id, node.id)
                self.store.save(root, state)
                self.audit.append(root, {"event": "gate_block", "step": node.id, "stage": stage, "reason": reason,
                                    "agent": _agent_name(agent), "proposal_hash": proposal_hash,
                                    "policy_check": checks, "attempt": record["attempts"],
                                    "latency_ms": round((time.monotonic() - started) * 1000)})
                raise
            return self.get_run(run_id) or state

    def spent_usd(self, state: dict[str, Any]) -> float:
        """Agent spend so far in this run: every attempt of every step, as reported by the CLI (not the model)."""
        return round(sum(u.get("cost_usd", 0) for u in (state.get("usage") or {}).values()), 6)

    auto_gate: Any = None                        # set by the server (auto_approval.step_gate): (policy name, step) -> approver label or None
    ledger: Any = None                           # set by the server on every runtime: the shared spend ledger of run trees (None = off)

    def _spend_log(self) -> Path:
        """The shared spend log lives next to the shared tree ledger (the server's runtime dir), so a test server writes to its own temp dir; AC_SPEND_LOG overrides."""
        return Path(os.environ.get("AC_SPEND_LOG") or self.ledger.path.with_name("spend_log.jsonl"))

    def _record_usage(self, root: Path, state: dict[str, Any], step_id: str, usage: Any, reservation: str | None = None) -> None:
        _add_usage(state.setdefault("usage", {}), step_id, usage)
        if self.ledger is None:
            return
        if isinstance(usage, dict):                                  # M2709.T23: one timestamped entry per paid call in the shared spend log (server runtimes only)
            try:
                cost_mod.record_spend(state["id"], usage.get("cost_usd"), step_id, log=self._spend_log())
            except OSError:
                pass                                                 # a log that cannot be written must not fail the run; check_budget reads what exists
        tree = (state.get("lineage") or {}).get("root") or state["id"]
        known = isinstance(usage, dict) and isinstance(usage.get("cost_usd"), (int, float)) and not isinstance(usage.get("cost_usd"), bool)
        if reservation is not None:
            self.ledger.settle(reservation, float(usage["cost_usd"]) if known else None)      # unknown cost = the whole reservation
        else:
            self.ledger.append(tree, state["id"], step_id, _cost_of(usage))

    def _reserve(self, state: dict[str, Any], step_id: str, ceiling: Any) -> str | None:
        """Hold this call's ceiling in the tree ledger before the call starts (only for runs that belong to a capped tree)."""
        if self.ledger is None or not isinstance(ceiling, (int, float)) or self._tree_budget(state, self.run_cap_usd(state)) is None:
            return None
        return self.ledger.reserve((state.get("lineage") or {}).get("root") or state["id"], state["id"], step_id, ceiling)

    def run_cap_usd(self, state: dict[str, Any]) -> float | None:
        """The run's spending cap: the policy's `run_usd` plus every extension a named human approved for THIS run (extend_budget)."""
        cap = (self.policy.get("budget") or {}).get("run_usd")
        if isinstance(cap, (int, float)):
            base: float | None = round(cap + float(state.get("budget_extra_usd") or 0), 6)
        else:
            envelope = state.get("envelope_cap_usd")                          # permission mode autonomous (M2630): the owner's per-run envelope
            base = float(envelope) if isinstance(envelope, (int, float)) and not isinstance(envelope, bool) else None
        request = state.get("request_cap_usd")                                # M2664.9: the caller's own cap, tightening only
        if isinstance(request, (int, float)) and not isinstance(request, bool):
            return float(request) if base is None else min(base, float(request))
        return base

    def extend_budget(self, run_id: str, add_usd: float, approved_by: str, note: str = "") -> dict[str, Any]:
        """A named human raises THIS run's spending cap (the policy file, and so the run's snapshot, stay untouched). The extension is
        written to the hash-chained trace and to the state, and integrity checks they agree. A step that failed (it ran out of budget
        or anything else) gets a fresh attempt allowance, because a human has now decided the spend is worth it. Fail-closed: agents
        cannot extend, a completed run cannot be extended, and one extension cannot exceed the policy's own run cap."""
        with self._run_lock(run_id):
            root = self.store.root(run_id)
            state = self._require_run(run_id)
            base = (self.policy.get("budget") or {}).get("run_usd")
            if not isinstance(base, (int, float)):
                raise GateBlocked("BUDGET", "policy", "the policy sets no run_usd cap, so there is nothing to extend")
            if not isinstance(approved_by, str) or approved_by.strip().lower() in SELF_APPROVERS:
                raise GateBlocked("BUDGET", "human approval", "a named human approver is required; agents cannot extend a budget")
            if isinstance(add_usd, bool) or not isinstance(add_usd, (int, float)) or not 0 < add_usd <= base:
                raise GateBlocked("BUDGET", "amount", f"an extension must be a positive amount of at most the policy's run cap (${base:g})")
            if state["status"] == "completed":
                raise GateBlocked("BUDGET", "state", "the run already completed; there is nothing to extend")
            spent = self.spent_usd(state)
            before = self.run_cap_usd(state)
            state["budget_extra_usd"] = round(float(state.get("budget_extra_usd") or 0) + float(add_usd), 6)
            if isinstance(state.get("request_cap_usd"), (int, float)):
                state["request_cap_usd"] = round(float(state["request_cap_usd"]) + float(add_usd), 6)       # the human's raise must not be swallowed by the caller's lower cap
            retried = [name for name, rec in state["steps"].items() if rec["status"] == "failed"]
            for name in retried:
                state["steps"][name].update(status="pending", error=None, attempts=0)
                state.get("feedback", {}).pop(name, None)             # the stop was about money, not about the content of the last attempt
                state["current_step"] = name
            if retried:
                state["status"] = "ready"
            self.store.record_transition(state, "budget_extend", state.get("current_step"), state.get("current_step"))
            self.store.save(root, state)
            self.audit.append(root, {"event": "budget_extended", "add_usd": round(float(add_usd), 6), "approved_by": approved_by.strip(),
                                     "note": str(note)[:500], "spent_usd": spent, "cap_before_usd": before,
                                     "cap_after_usd": self.run_cap_usd(state), "retry_steps": retried})
            return self.get_run(run_id) or state

    def _enforce_budget(self, root: Path, state: dict[str, Any], node: Node, step: dict[str, Any]) -> None:
        """Policy `budget: {run_usd, step_usd}` (and a per-step `max_budget_usd`): stop BEFORE the call that would exceed the run cap,
        and hand each call a ceiling (`--max-budget-usd`) of min(step cap, what is left of the run). Fail-closed: a run over budget
        fails; nothing is silently spent."""
        if self.ledger is not None and any(os.environ.get(n, "").strip() for n in ("AC_BUDGET_DAILY_USD", "AC_BUDGET_WEEKLY_USD")):     # M2709.T23: the owner's global daily / weekly budget (server runtimes only)
            check = cost_mod.check_budget(log=self._spend_log())
            if not check["ok"]:
                self.audit.append(root, {"event": "budget_exceeded", "step": node.id, "global": True, "reason": check["reason"]})
                raise cost_mod.BudgetExceeded(f"{check['reason']}; {node.id} was not started")
        budget = self.policy.get("budget") or {}
        run_cap = self.run_cap_usd(state)
        caps = [c for c in (node.raw.get("max_budget_usd"), budget.get("step_usd")) if isinstance(c, (int, float))]
        spent = self.spent_usd(state)
        if isinstance(run_cap, (int, float)):
            remaining = run_cap - spent
            if remaining <= 0:
                self.audit.append(root, {"event": "budget_exceeded", "step": node.id, "spent_usd": spent, "run_cap_usd": run_cap})
                raise RuntimeError(f"budget exhausted: ${spent:.2f} spent of the ${run_cap:.2f} run cap; {node.id} was not started")
            caps.append(remaining)
        tree = self._tree_budget(state, run_cap)
        if tree is not None:                                          # the whole run tree shares ONE cap: the root's
            tree_cap, tree_spent = tree
            if tree_cap - tree_spent <= 0:
                self.audit.append(root, {"event": "budget_exceeded", "step": node.id, "spent_usd": tree_spent, "tree_cap_usd": tree_cap})
                held = self.ledger.unsettled(state.get("lineage", {}).get("root") or state["id"])
                note = ""
                if held > 0:
                    only_hold = tree_spent - held < tree_cap                       # without the hold the tree would still be under its cap: the stop is caused by killed calls alone
                    note = f" (${held:.2f} of it is held for calls that never reported a cost, for example after a service restart; the real cost is unknown and at most that{HOLD_ONLY_MARK if only_hold else ''})"
                raise RuntimeError(f"tree budget exhausted: ${tree_spent:.2f} spent by this run and the runs it started, of the ${tree_cap:.2f} cap{note}; {node.id} was not started")
            caps.append(tree_cap - tree_spent)
        if caps:
            step["max_budget_usd"] = round(max(min(caps), 0.01), 4)

    def _tree_budget(self, state: dict[str, Any], run_cap: float | None) -> tuple[float, float] | None:
        """(cap, spent) of the run tree this run belongs to, or None when there is no ledger or no cap to share."""
        if self.ledger is None:
            return None
        lineage = state.get("lineage") or {}
        cap = lineage.get("tree_cap_usd") if lineage else run_cap          # a root's own cap; a child carries the root's cap it was started under
        if not isinstance(cap, (int, float)):
            return None
        return float(cap), self.ledger.total(lineage.get("root") or state["id"])

    def _usage_fields(self, usage: dict[str, Any] | None) -> dict[str, Any]:
        """Token / cost fields for the trace, honoring the policy's observability.trace flags (default: on)."""
        if not usage:
            return {}
        flags = ((self.policy.get("observability") or {}).get("trace")) or {}
        out: dict[str, Any] = {}
        if flags.get("tokens", True):
            out["tokens"] = {k: usage[k] for k in ("input_tokens", "output_tokens", "cache_creation_input_tokens",
                                                   "cache_read_input_tokens") if k in usage}
        if flags.get("cost", True):
            out["cost_usd"] = round(usage.get("cost_usd", 0), 6)
        return out

    def agent_capability_report(self, agent: Any) -> dict[str, Any]:
        """What this policy needs from `agent` versus what the agent declares (no side effects)."""
        caps = getattr(agent, "capabilities", None)
        if not isinstance(caps, capabilities_mod.AgentCapabilities):
            return {"declared": False, "agent": type(agent).__name__, "missing_hard": [], "missing_soft": []}
        steps = [self._step_for(n, {}) for n in self.graph.nodes.values()]
        return {"declared": True, "agent": caps.name, "capabilities": caps.describe(), **capabilities_mod.check(self.policy, steps, caps)}

    def _require_capabilities(self, agent: Any, run_id: str, root: Path, state: dict[str, Any]) -> None:
        """Once per run: record the agent's declared capabilities and refuse to start if the policy needs more (fail-closed).
        An agent that declares nothing is allowed but recorded as undeclared."""
        if run_id in self._caps_checked:
            return
        report = self.agent_capability_report(agent)
        self.audit.append(root, {"event": "agent_capabilities", **report})
        if report["missing_hard"]:
            raise GateBlocked("AGENT", "capabilities", f"agent '{report['agent']}' cannot run this policy: "
                              + "; ".join(report["missing_hard"]))
        self._caps_checked.add(run_id)

    def _step_for(self, node: Node, state: dict[str, Any]) -> dict[str, Any]:
        """The step dict handed to an agent: policy source + only the sources/principles this run may use."""
        step = node.to_step()
        declared = self.policy.get("mcp_servers") or {}
        used = {t.split("__")[1] for t in node.tools if t.startswith("mcp__")}
        if self.policy.get("use") == "commercial":     # sources whose terms are not cleared are switched off
            blocked = {n for n in used if (declared.get(n) or {}).get("commercial_ok") is not True}
            if blocked:
                step["tools"] = [t for t in node.tools if not (t.startswith("mcp__") and t.split("__")[1] in blocked)]
                used -= blocked
        if used:                       # only the servers this step names are handed to the agent
            step["mcp_servers"] = {n: declared[n] for n in sorted(used) if n in declared}
            for name, server in list(step["mcp_servers"].items()):
                if server.get("builtin") == "sources":            # the raw declaration carries no allow_tools; the guard needs the validated form (M2635: the first v5 run was denied its own tools)
                    step["mcp_servers"][name] = builtin_sources(name, server)
                if server.get("builtin") == "governance":          # a builtin governance server is told which run it serves; runs it creates are that run's children
                    step["mcp_servers"][name] = {**builtin_governance(name, server),
                                                 "run_context": {"run_id": state.get("id", ""), "children_depth": (state.get("lineage") or {}).get("depth", 0) + 1}}
        run_principles = self.principles_of(state)
        if run_principles:
            step["principles"] = advise_lines(run_principles)
        return step

    # ---- parallel steps: concurrent proposals, sequential gate ------------------------------------
    # The gate still authorizes, verifies and commits ONE step at a time in declared order. Only the slow part, the agent's
    # proposal, is obtained concurrently for ready `parallel` steps; a proposal is used only if the step's inputs are
    # byte-identical when its turn comes (retries carry feedback, so they never reuse a prefetched proposal).

    MAX_PARALLEL = 4

    def _prefetch(self, run_id: str, agent: Any) -> None:
        state = self._require_run(run_id)
        current = state.get("current_step")
        if not current or not self.graph.node(current).parallel or state["status"] in ("completed", "awaiting_approval"):
            return
        caps = getattr(agent, "capabilities", None)
        if isinstance(caps, capabilities_mod.AgentCapabilities) and not caps.parallel_safe:
            return                                   # a backend that cannot be called concurrently simply runs steps in order
        passed = {k for k, v in state["steps"].items() if v["status"] == "pass"}
        group = [current]
        for sid in self.graph.order:
            node = self.graph.node(sid)
            if (sid != current and node.parallel and state["steps"][sid]["status"] == "pending" and set(node.requires) <= passed
                    and sid not in state.get("approval_required", []) and sid not in state.get("feedback", {})):
                group.append(sid)
        if len(group) < 2 or current in state.get("feedback", {}):
            return
        group = group[: self.MAX_PARALLEL]
        root = self.store.root(run_id)
        wave_inputs = {sid: _hash_json(self._read_inputs(root, state, self.graph.node(sid))) for sid in group}
        begin = self.audit.append(root, {"event": "wave_begin", "steps": group, "commit_order": group,
                                         "input_hashes": wave_inputs, "snapshot_state": _hash_json(
                                             {k: v["artifact_hash"] for k, v in state["steps"].items()})})
        with self._prefetch_lock:
            self._waves[run_id] = {"steps": group, "begin_seq": (begin or {}).get("seq", 0) if isinstance(begin, dict) else 0}
        self._live(root, current, "prefetch", steps=group)
        self._bind(agent, root, "+".join(group), run_id)
        results: dict[str, Any] = {}

        def work(sid: str) -> None:
            node = self.graph.node(sid)
            inputs = self._read_inputs(root, state, node)
            began = time.monotonic()
            try:
                proposal = agent.propose(self._step_for(node, state), state["intent"], inputs)
                results[sid] = (_hash_json(inputs), proposal, time.monotonic() - began)
            except BaseException:                      # a failed prefetch is simply not used; the step will call the agent itself
                results[sid] = None

        threads = [threading.Thread(target=work, args=(sid,), daemon=True) for sid in group]
        try:
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        finally:
            self._bind(agent, None, "+".join(group), run_id)
        with self._prefetch_lock:
            for sid, value in results.items():
                if value is not None:
                    self._prefetched[(run_id, sid)] = value

    def _take_prefetched(self, run_id: str, step_id: str, inputs: dict[str, Any]) -> tuple[Any, float]:
        """(proposal, seconds the agent took) or (None, 0) when there is nothing usable for these exact inputs."""
        with self._prefetch_lock:
            entry = self._prefetched.pop((run_id, step_id), None)
        if entry is None or entry[0] != _hash_json(inputs):
            return None, 0.0
        return entry[1], entry[2]

    def _close_wave(self, run_id: str, root: Path, state: dict[str, Any]) -> None:
        """After a commit: if every member of the open wave has now passed, record wave_end with the actual commit order."""
        with self._prefetch_lock:
            wave = self._waves.get(run_id)
        if not wave or any(state["steps"][s]["status"] != "pass" for s in wave["steps"]):
            return
        with self._prefetch_lock:
            self._waves.pop(run_id, None)
        committed = [e["step"] for e in self.audit.read(root) if e.get("event") == "gate_pass"
                     and e.get("seq", 0) > wave["begin_seq"] and e.get("step") in wave["steps"]]
        self.audit.append(root, {"event": "wave_end", "steps": wave["steps"], "commit_order": committed,
                                 "in_declared_order": committed == wave["steps"]})

    def _drop_prefetched(self, run_id: str) -> None:
        with self._prefetch_lock:
            self._waves.pop(run_id, None)
            for key in [k for k in self._prefetched if k[0] == run_id]:
                del self._prefetched[key]

    def _bind(self, agent: Any, root: Path | None, step_id: str, run_id: str) -> None:
        """Give streaming-capable agents a live-activity callback and this run's cancel event (None unbinds)."""
        activity, cancel = getattr(agent, "bind_activity", None), getattr(agent, "bind_cancel", None)
        if callable(activity):
            activity((lambda event: self._live(root, step_id, "agent", activity=event.get("kind"),
                                               **{k: v for k, v in event.items() if k != "kind"})) if root else None)
        if callable(cancel):
            cancel(self._cancels.get(run_id) if root else None)

    def _read_inputs(self, root: Path, state: dict[str, Any], node: Node) -> dict[str, Any]:
        inputs: dict[str, Any] = {}
        for required in node.requires:
            filename = state["steps"][required]["artifact"]
            inputs[required] = json.loads((root / "artifacts" / filename).read_text(encoding="utf-8"))
        if node.id in state.get("feedback", {}):
            inputs["_feedback"] = dict(state["feedback"][node.id])
            previous = state.get("feedback_data", {}).get(node.id)
            if previous is not None and len(json.dumps(previous)) <= 16000:
                inputs["_feedback"]["previous_output"] = previous     # fix only the failing items; keep the rest as is
        return inputs

    # ---- whole workflow: retry, repair, optional planner ------------------------------------------

    def run_all(self, run_id: str, agent: Agent, planner: bool = False) -> dict[str, Any]:
        try:
            return self._run_all(run_id, agent, planner)
        finally:
            self._drop_prefetched(run_id)                     # never carry a proposal across runs of the loop

    def _run_all(self, run_id: str, agent: Agent, planner: bool = False) -> dict[str, Any]:
        while True:
            state = self._require_run(run_id)
            if state["status"] in ("completed", "terminated"):
                return self.get_run(run_id) or state
            if self._cancelled(run_id):                       # stop between steps; nothing is half-done
                self.audit.append(self.store.root(run_id), {"event": "run_cancelled", "step": state.get("current_step"),
                                                            "stage": "between_steps"})
                self._live(self.store.root(run_id), state.get("current_step"), "cancelled")
                return self.get_run(run_id) or state
            try:
                if planner and callable(getattr(agent, "plan", None)):
                    self.run_planned_step(run_id, agent)
                else:
                    self._prefetch(run_id, agent)
                    self.run_next(run_id, agent)
            except RunCancelled:
                return self.get_run(run_id) or state
            except GateBlocked as exc:
                if self._auto_approve(run_id):                 # the owner handed this approval gate to the runtime: grant it and go on
                    continue
                if not self._recover(run_id, agent, exc):
                    return self.get_run(run_id) or state
            except Exception as exc:
                if not self._recover(run_id, agent, exc):
                    return self.get_run(run_id) or state

    def _auto_approve(self, run_id: str) -> bool:
        """Auto-approval mode (auto_approval.py): when the run waits at an approval gate the owner listed, record the approval under the owner's label.
        Same record, same artifact-hash binding and same trace event as a human approval; the label says it was automatic."""
        state = self.get_run(run_id)
        if not state or state.get("status") != "awaiting_approval":
            return False
        step = state.get("current_step")
        pre = [p for p in state.get("pre_approvals") or [] if p["step"] == step]
        if pre:                                           # a named human stated conditions at run start: they decide, the server-wide list is not consulted
            root = self.store.root(run_id)
            artifact = lambda name: ((self.artifact(run_id, name) or {}).get("data"))
            report = [(p, pre_approval.unmet(p["conditions"], artifact)) for p in pre]
            met = next((p for p, problems in report if not problems), None)
            if met is None:
                self.audit.append(root, {"event": "pre_approval_not_met", "step": step, "problems": [pr for _, problems in report for pr in problems][:12]})
                return False
            self.audit.append(root, {"event": "pre_approval_applied", "step": step, "by": met["by"], "conditions": met["conditions"]})
            self.approve_step(run_id, step, f"{met['by']} (pre-approved at run start)", "pre-approval conditions met on the committed artifacts")
            return True
        if not self.auto_gate:
            return False
        label = self.auto_gate(self.policy.get("name"), step)
        if not label:
            return False
        self.approve_step(run_id, step, label, "auto-approval mode: granted without a human at this step")
        return True

    def _recover(self, run_id: str, agent: Agent, exc: Exception) -> bool:
        with self._run_lock(run_id):
            root = self.store.root(run_id)
            state = self._require_run(run_id)
            step_id = state.get("current_step")
            if not step_id or state["steps"][step_id]["status"] != "failed":
                return False                      # blocked before execution (authorization) -> stop, fail-closed
            node = self.graph.node(step_id)
            plan = self.recovery.decide(state, node, str(exc), agent)
            if plan is None:
                self.audit.append(root, {"event": "recovery_exhausted", "step": step_id, "reason": str(exc)[:500]})
                return False
            if plan["action"] == "retry":
                self.audit.append(root, {"event": "retry_scheduled", "step": step_id, "attempt": plan["attempts"] + 1})
                self._live(root, step_id, "retry", attempt=plan["attempts"] + 1, reason=str(exc)[:200])
                return True
            reset = self.recovery.rollback(root, state, plan["rerun_from"], step_id, str(exc))
            self.store.record_transition(state, "rollback", step_id, plan["rerun_from"])
            self.store.save(root, state)
            self.audit.append(root, {"event": "repair_scheduled", "step": step_id, "rerun_from": plan["rerun_from"],
                                "rolled_back": reset, "proposed_by": plan["proposed_by"], "reason": str(exc)[:500]})
            self._live(root, step_id, "repair", rerun_from=plan["rerun_from"], rolled_back=reset)
            return True

    def run_planned_step(self, run_id: str, agent: Any) -> dict[str, Any]:
        """Claude proposes the next action; the compiler authorizes or corrects it."""
        with self._run_lock(run_id):
            root = self.store.root(run_id)
            state = self._require_run(run_id)
            expected = state["current_step"]
            summary = {"intent": state["intent"], "current_step": expected,
                       "steps": {k: v["status"] for k, v in state["steps"].items()}, "order": self.graph.order}
            try:
                plan = agent.plan(summary)
            except Exception as exc:               # planner failure never blocks progress; the compiler chooses
                plan = {"next_action": expected, "reason": f"planner error: {exc}"}
            proposed = plan.get("next_action") if isinstance(plan, dict) else None
            self.audit.append(root, {"event": "plan_proposed", "proposed": proposed, "agent": _agent_name(agent),
                                "reason": str(plan.get("reason", ""))[:300] if isinstance(plan, dict) else ""})
            if proposed != expected:
                required = expected or "none"
                if proposed in self.graph.nodes:
                    decision = self.gate.authorize(root, state, self.graph.node(proposed))
                    reason, checks = decision.reason, decision.checks
                else:
                    reason, checks = f"unknown step: {proposed}", {"current_step": "BLOCK"}
                self.audit.append(root, {"event": "gate_block", "step": proposed, "stage": "plan", "reason": reason,
                                    "required_step": required, "policy_check": checks,
                                    "agent": _agent_name(agent)})
                self.audit.append(root, {"event": "plan_corrected", "from": proposed, "to": required})
            return self.run_step(run_id, expected, agent)

    # ---- human approval gates ---------------------------------------------------------------------

    def _await_approval(self, root: Path, state: dict[str, Any], node: Node, decision: Decision) -> None:
        if state["status"] != "awaiting_approval" or state.get("current_step") != node.id:
            state["status"] = "awaiting_approval"
            self._live(root, node.id, "awaiting_approval", reason=decision.reason)
            self.store.record_transition(state, "await_approval", node.id, node.id)
            self.store.save(root, state)
            self.audit.append(root, {"event": "approval_requested", "step": node.id, "reason": decision.reason,
                                     "input_hashes": {n: state["steps"][n]["artifact_hash"] for n in node.requires}})

    def approve_step(self, run_id: str, step_id: str, approved_by: str, note: str = "") -> dict[str, Any]:
        """A named human approves starting `step_id`, bound to the exact upstream artifacts they reviewed."""
        with self._run_lock(run_id):
            root = self.store.root(run_id)
            state = self._require_run(run_id)
            node = self.graph.node(step_id)
            if step_id not in state.get("approval_required", []):
                raise ValueError(f"{step_id} does not require approval in this run")
            if not isinstance(approved_by, str) or approved_by.strip().lower() in SELF_APPROVERS:
                raise GateBlocked("APPROVE", "human approval", "a named human approver is required; agents cannot approve")
            if any(state["steps"][name]["status"] != "pass" for name in node.requires):
                raise GateBlocked("APPROVE", "prerequisites", f"prerequisites of {step_id} have not passed; nothing to approve yet")
            hashes = {name: state["steps"][name]["artifact_hash"] for name in node.requires}
            state["approvals"][step_id] = {"by": approved_by.strip(), "note": str(note)[:500], "at": now(),
                                           "input_hashes": hashes}
            if state["status"] == "awaiting_approval":
                state["status"] = "ready"
            self.store.record_transition(state, "approve", step_id, step_id)
            self.store.save(root, state)
            self.audit.append(root, {"event": "approval_granted", "step": step_id, "approved_by": approved_by.strip(),
                                     "note": str(note)[:500], "input_hashes": hashes})
            return self.get_run(run_id) or state

    def deny_step(self, run_id: str, step_id: str, denied_by: str, reason: str = "") -> dict[str, Any]:
        """A named human refuses: the run fails at that step and nothing after it can run."""
        with self._run_lock(run_id):
            root = self.store.root(run_id)
            state = self._require_run(run_id)
            if step_id not in state.get("approval_required", []):
                raise ValueError(f"{step_id} does not require approval in this run")
            if not isinstance(denied_by, str) or denied_by.strip().lower() in SELF_APPROVERS:
                raise GateBlocked("DENY", "human approval", "a named human is required")
            state["steps"][step_id].update(status="failed", error=f"denied by {denied_by.strip()}: {str(reason)[:300]}")
            state["status"] = "failed"
            state["current_step"] = step_id
            state.get("approvals", {}).pop(step_id, None)
            self.store.record_transition(state, "deny", step_id, step_id)
            self.store.save(root, state)
            self.audit.append(root, {"event": "approval_denied", "step": step_id, "denied_by": denied_by.strip(),
                                     "reason": str(reason)[:300]})
            return self.get_run(run_id) or state

    # ---- resume after a crash / restart ---------------------------------------------------------------

    def resume(self, run_id: str, agent: Agent, planner: bool = False) -> dict[str, Any]:
        """Continue a persisted run in a new process: verify everything already committed, then run what is left.

        Fail-closed: any tampered artifact or broken trace chain stops the resume before the agent is called."""
        with self._run_lock(run_id):
            root = self.store.root(run_id)
            state = self._require_run(run_id)
            if state["status"] == "completed":
                return self.get_run(run_id) or state
            problems = self.verify_run(run_id)
            if problems:
                self.audit.append(root, {"event": "resume_blocked", "problems": problems})
                raise GateBlocked("RESUME", "integrity", f"{len(problems)} integrity problem(s): {problems[0]['reason']}")
            interrupted = [name for name, rec in state["steps"].items() if rec["status"] == "running"]
            for name in interrupted:               # the previous process died mid-step: the step is not committed
                state["steps"][name].update(status="pending", error="interrupted before commit (process restart)")
                state["current_step"] = name
            if interrupted or state["status"] in ("running", "repairing"):
                state["status"] = "ready"
            self.store.record_transition(state, "resume", state.get("current_step"), state.get("current_step"))
            self.store.save(root, state)
            self.audit.append(root, {"event": "run_resumed", "current_step": state.get("current_step"),
                                     "interrupted_steps": interrupted,
                                     "verified_steps": [n for n, r in state["steps"].items() if r["status"] == "pass"]})
        return self.run_all(run_id, agent, planner=planner)

    def rerun_from(self, run_id: str, from_step: str, reason: str = "", by: str = "") -> dict[str, Any]:
        """M2642: re-run a persisted run from a chosen step. Every earlier step must be committed and is verified (hash, validators, trace chain) before
        anything is discarded; `from_step` and every later step are reset and then run again by `resume`/`run_all`. Fail-closed like `resume`; the run's
        own spending cap still applies because the spend already recorded is kept."""
        with self._run_lock(run_id):
            root = self.store.root(run_id)
            state = self._require_run(run_id)
            order = self.graph.order
            if from_step not in order:
                raise ValueError(f"unknown step '{from_step}'; this policy has: {', '.join(order)}")
            if state["status"] == "terminated":
                raise GateBlocked("RERUN", "state", "a terminated run cannot be restarted")
            running = [n for n, rec in state["steps"].items() if rec["status"] == "running"]
            if running:
                raise GateBlocked("RERUN", "state", f"step(s) still marked running: {', '.join(running)}; resume the run first")
            earlier = order[:order.index(from_step)]
            missing = [n for n in earlier if state["steps"][n]["status"] != "pass"]
            if missing:
                raise GateBlocked("RERUN", "committed steps", f"steps before {from_step} are not committed: {', '.join(missing)}")
            problems = self.verify_run(run_id)
            if problems:
                self.audit.append(root, {"event": "rerun_blocked", "from_step": from_step, "problems": problems})
                raise GateBlocked("RERUN", "integrity", f"{len(problems)} integrity problem(s): {problems[0]['reason']}")
            reset = self.recovery.reset_from(root, state, from_step)
            self.store.record_transition(state, "rerun_from", from_step, from_step)
            self.store.save(root, state)
            self.audit.append(root, {"event": "rerun_from", "from_step": from_step, "kept_steps": earlier, "reset_steps": reset,
                                     "by": (by or "api")[:80], "reason": str(reason)[:300]})
            self._live(root, from_step, "rerun_from", reset=reset)
        return self.get_run(run_id) or state

    # ---- integrity ----------------------------------------------------------------------------

    def verify_trace(self, run_id: str, expected_head: str | None = None) -> list[dict[str, Any]]:
        return self.audit.verify(self.store.root(run_id), expected_head)

    def check_integrity(self, run_id: str, expected_head: str | None = None) -> list[dict[str, Any]]:
        """Read-only integrity check (writes nothing, not even to the trace). Empty list = intact."""
        root, state = self.store.root(run_id), self._require_run(run_id)
        failures = self.verifier.replay(root, state)
        failures += [{"step": "TRACE", "reason": p["reason"]} for p in self.verify_trace(run_id, expected_head)]
        granted = round(sum(float(e.get("add_usd") or 0) for e in self.audit.read(root) if e.get("event") == "budget_extended"), 6)
        if granted != round(float(state.get("budget_extra_usd") or 0), 6):
            failures.append({"step": "BUDGET", "reason": f"state carries ${state.get('budget_extra_usd') or 0} of approved budget extensions but the trace records ${granted}"})
        return failures

    def verify_run(self, run_id: str, expected_head: str | None = None) -> list[dict[str, Any]]:
        """Artifacts (hash + validators replay) and the hash-chained trace; records the check in the trace."""
        failures = self.check_integrity(run_id, expected_head)
        self.audit.append(self.store.root(run_id), {"event": "integrity_check", "failures": failures})
        return failures

    def artifact(self, run_id: str, step_id: str) -> dict[str, Any] | None:
        """The committed envelope of a declared step (data + audit), or None if not committed."""
        node = self.graph.node(step_id)
        path = self.store.root(run_id) / "artifacts" / node.output
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None

    # ---- measurements (OBSERVE -> reality) ---------------------------------------------------------

    def record_measurement(self, run_id: str, action_id: str, value: float, note: str = "") -> dict[str, Any]:
        with self._run_lock(run_id):
            root = self.store.root(run_id)
            state = self._require_run(run_id)
            if state["steps"].get("OBSERVE", {}).get("status") != "pass":
                raise GateBlocked("MEASURE", "OBSERVE", "measurements require a passed OBSERVE step")
            path = root / "artifacts" / state["steps"]["OBSERVE"]["artifact"]
            if not path.is_file() or digest(path.read_bytes()) != state["steps"]["OBSERVE"]["artifact_hash"]:
                raise GateBlocked("MEASURE", "OBSERVE", "OBSERVE artifact is missing or does not match state")
            observe = json.loads(path.read_text(encoding="utf-8"))
            entry = next((o for o in observe["data"]["observations"] if o["action_id"] == action_id), None)
            if entry is None:
                raise KeyError(f"unknown action: {action_id}")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("measurement value must be a number")
            comparator = entry.get("comparator", ">=")
            hit = value >= entry["predicted_value"] if comparator == ">=" else value <= entry["predicted_value"]
            record = {"action_id": action_id, "value": value, "predicted_value": entry["predicted_value"],
                      "comparator": comparator, "hit": bool(hit), "note": str(note)[:500], "recorded_at": now()}
            with (root / "measurements.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.audit.append(root, {"event": "measurement_recorded", **record})
            return record

    def measurements(self, run_id: str) -> list[dict[str, Any]]:
        path = self.store.root(run_id) / "measurements.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _notify(agent: Any, step_id: str, accepted: bool, reason: str) -> None:
    """Optional agent hook: tell wrappers (e.g. replay caches) whether the compiler accepted the proposal."""
    observe = getattr(agent, "observe", None)
    if callable(observe):
        try:
            observe(step_id, accepted, reason)
        except Exception:                        # observers never affect authority
            pass


def _agent_name(agent: Any) -> str:
    return getattr(agent, "name", type(agent).__name__)


