"""Compiler-owned run state. Conversation context is never authoritative."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from typing import Any, Callable
from uuid import uuid4

from .fsutil import read_text_retry, replace_atomic
from .graph import Graph

STEP_STATUSES = {"pending", "running", "pass", "failed"}
METRIC_SOURCES = ("real", "mock")          # M2709.T2: a metric says whether it was measured on a real channel or comes from a mock, so a mock can never count as a real integration
VERIFICATION_STATUSES = ("unverified", "verified", "rejected")   # M2709.T6: what is known about a collected source record; "verified" is never the default
HYPOTHESIS_STATUSES = ("needs_more_evidence", "hypothesis", "supported", "contradicted", "on_hold")   # M2709.T14: no "fact", "confirmed" or "verified": a hypothesis never becomes a fact here
EXPERIMENT_STATUSES = ("defined", "completed")
RUN_STATUSES = {"ready", "running", "completed", "failed", "repairing", "awaiting_approval", "terminated"}


class StateCorrupt(ValueError):
    """Persisted state is malformed. The runtime is fail-closed and refuses to continue."""


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class StateStore:
    def __init__(self, runtime_dir: Path):
        self.runtime_dir = Path(runtime_dir)
        self.runtime_dir.mkdir(parents=True, exist_ok=True)

    def root(self, run_id: str) -> Path:
        if not isinstance(run_id, str) or not re.fullmatch(r"[a-f0-9]{32}", run_id):
            raise KeyError("invalid run id")
        return self.runtime_dir / "runs" / run_id

    def create(self, graph: Graph, intent: str,
               approval_required: list[str] | None = None) -> tuple[Path, dict[str, Any]]:
        run_id = uuid4().hex
        root = self.root(run_id)
        (root / "artifacts").mkdir(parents=True)
        (root / "traces").mkdir()
        state = {
            "id": run_id, "policy": graph.name, "policy_version": graph.version, "intent": intent,
            "status": "ready", "current_step": graph.first, "created_at": now(), "updated_at": now(),
            "steps": {n.id: {"status": "pending", "artifact": n.output, "artifact_hash": None,
                             "error": None, "attempts": 0} for n in graph.nodes.values()},
            "repairs": {}, "feedback": {}, "transitions": [],
            "approval_required": list(approval_required or []), "approvals": {},
        }
        self.save(root, state)
        return root, state

    def load(self, run_id: str) -> dict[str, Any]:
        path = self.root(run_id) / "state.json"
        if not path.is_file():
            raise KeyError(f"run not found: {run_id}")
        try:
            state = json.loads(read_text_retry(path))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise StateCorrupt(f"state.json for {run_id} is not valid JSON") from exc
        self._check(state)
        return state

    @staticmethod
    def _check(state: Any) -> None:
        if not isinstance(state, dict) or state.get("status") not in RUN_STATUSES:
            raise StateCorrupt(f"unknown run status: {state.get('status') if isinstance(state, dict) else state!r}")
        steps = state.get("steps")
        if not isinstance(steps, dict) or not steps:
            raise StateCorrupt("state has no steps")
        for name, record in steps.items():
            if not isinstance(record, dict) or record.get("status") not in STEP_STATUSES:
                raise StateCorrupt(f"unknown status for step {name}")
        if state.get("current_step") is not None and state["current_step"] not in steps:
            raise StateCorrupt("current_step is not a declared step")

    @staticmethod
    def save(root: Path, state: dict[str, Any]) -> None:
        state["updated_at"] = now()
        temp = root / "state.json.tmp"
        temp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        replace_atomic(temp, root / "state.json")

    @staticmethod
    def record_transition(state: dict[str, Any], event: str, source: str | None, target: str | None) -> None:
        state.setdefault("transitions", []).append({"event": event, "from": source, "to": target, "at": now()})

    # ---- metrics (M2709.T2): one append-only JSONL file next to the runs; no database ----------------------------------------
    def _metrics_path(self) -> Path:
        return self.runtime_dir / "metrics.jsonl"

    def write_metric(self, record: dict[str, Any]) -> None:
        """Append one metric. Required: `name` (non-empty text), `value`, and `source` in METRIC_SOURCES; anything else is refused and nothing is written.
        The record is stored exactly as given (add `observed_at` yourself when the time matters)."""
        if not isinstance(record, dict):
            raise ValueError("metric must be an object")
        if record.get("source") not in METRIC_SOURCES:
            raise ValueError(f"metric needs source in {METRIC_SOURCES}, got {record.get('source')!r}")
        if not isinstance(record.get("name"), str) or not record["name"].strip() or "value" not in record:
            raise ValueError("metric needs a non-empty name and a value")
        self._append_jsonl(self._metrics_path(), record)

    @staticmethod
    def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
        line = (json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, line)                      # one write call: concurrent appenders never interleave inside a record
        finally:
            os.close(fd)

    @staticmethod
    def _read_jsonl(path: Path, valid: Callable[[Any], bool], what: str) -> list[dict[str, Any]]:
        """Records in write order. A malformed line or one that fails `valid` raises StateCorrupt (fail closed): it is never skipped or counted."""
        if not path.is_file():
            return []
        out = []
        for n, line in enumerate(read_text_retry(path).splitlines(), 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise StateCorrupt(f"{path.name} line {n} is not valid JSON") from exc
            if not isinstance(rec, dict) or not valid(rec):
                raise StateCorrupt(f"{path.name} line {n} is not a valid {what}")
            out.append(rec)
        return out

    def read_metrics(self, source: str | None = None) -> list[dict[str, Any]]:
        """All metrics in write order, or only those with `source`."""
        if source is not None and source not in METRIC_SOURCES:
            raise ValueError(f"unknown source {source!r}")
        rows = self._read_jsonl(self._metrics_path(), lambda r: r.get("source") in METRIC_SOURCES, "metric with a source tag")
        return [r for r in rows if source is None or r["source"] == source]

    # ---- source records (M2709.T6): what was collected, from where, when, and how far it is verified -----------------------------
    def _sources_path(self) -> Path:
        return self.runtime_dir / "source_records.jsonl"

    @staticmethod
    def _source_problem(rec: Any) -> str | None:
        if not isinstance(rec, dict):
            return "source record must be an object"
        for k in ("source", "url", "collected_at", "verification_status"):
            if not isinstance(rec.get(k), str) or not rec[k].strip():
                return f"source record needs a non-empty text field {k!r}"
        if not re.match(r"https?://[^\s/]+", rec["url"]):
            return "url must be an http(s) address"
        try:
            when = datetime.fromisoformat(rec["collected_at"])
        except ValueError:
            return "collected_at must be an ISO 8601 time"
        if when.tzinfo is None:
            return "collected_at must carry a UTC offset"
        if rec["verification_status"] not in VERIFICATION_STATUSES:
            return f"verification_status must be one of {VERIFICATION_STATUSES}"
        return None

    def write_source_record(self, record: dict[str, Any]) -> bool:
        """Append one source record; True when written, False when the same `source` + `url` is already stored (the stored record is kept, so a re-run never duplicates).
        Required text fields: source (the collector's name), url, collected_at (ISO time with offset), verification_status in VERIFICATION_STATUSES.
        Other fields (external id, published_at, content_hash, a reference to the raw text) are stored as given. Anything invalid raises ValueError and writes nothing."""
        problem = self._source_problem(record)
        if problem:
            raise ValueError(problem)
        if any((r["source"], r["url"]) == (record["source"], record["url"]) for r in self.read_source_records()):
            return False
        self._append_jsonl(self._sources_path(), record)
        return True

    def read_source_records(self, source: str | None = None) -> list[dict[str, Any]]:
        """All source records in write order, or only those collected by `source`. A line that fails the schema raises StateCorrupt."""
        rows = self._read_jsonl(self._sources_path(), lambda r: self._source_problem(r) is None, "source record")
        return [r for r in rows if source is None or r["source"] == source]

    # ---- post bodies (M2715): the text of a collected post, kept apart from the source record so retention can differ ----------------
    def _bodies_path(self) -> Path:
        return self.runtime_dir / "post_bodies.jsonl"

    @staticmethod
    def _body_problem(rec: Any) -> str | None:
        if not isinstance(rec, dict):
            return "post body must be an object"
        for k in ("source", "url", "text", "text_hash", "stored_at"):
            if not isinstance(rec.get(k), str) or not rec[k].strip():
                return f"post body needs a non-empty text field {k!r}"
        days = rec.get("retention_days")
        if days is not None and (isinstance(days, bool) or not isinstance(days, int) or days < 1):
            return "retention_days is null (kept permanently, an explicit owner choice) or an integer >= 1"
        return None

    def write_post_body(self, record: dict[str, Any]) -> bool:
        """Append the text of one stored source record; True when written, False when that `source` + `url` already has a body (the first one is kept).
        Required text fields: source, url (must match a stored source record), text, text_hash, stored_at; `retention_days` is null for permanent keeping or an integer.
        Invalid input raises ValueError and writes nothing."""
        problem = self._body_problem(record)
        if problem:
            raise ValueError(problem)
        key = (record["source"], record["url"])
        if key not in {(r["source"], r["url"]) for r in self.read_source_records()}:
            raise ValueError(f"post body links to a source record that is not stored: {record['source']} {record['url']}")
        if any((r["source"], r["url"]) == key for r in self.read_post_bodies()):
            return False
        self._append_jsonl(self._bodies_path(), record)
        return True

    def read_post_bodies(self, source: str | None = None) -> list[dict[str, Any]]:
        """All post bodies in write order, or only those of `source`. A line that fails the schema raises StateCorrupt."""
        rows = self._read_jsonl(self._bodies_path(), lambda r: self._body_problem(r) is None, "post body")
        return [r for r in rows if source is None or r["source"] == source]

    # ---- hypotheses (M2709.T14): append-only, the latest line per pain_id is the current record -----------------------------------
    def _hypotheses_path(self) -> Path:
        return self.runtime_dir / "hypotheses.jsonl"

    @staticmethod
    def _link_problem(ref: Any) -> str | None:
        if not isinstance(ref, dict) or not isinstance(ref.get("source"), str) or not isinstance(ref.get("url"), str) or not ref["source"] or not ref["url"]:
            return "a link is an object with the `source` and `url` of a stored source record"
        return None

    def _hypothesis_problem(self, rec: Any) -> str | None:
        """Everything that can be checked without the source records (shape, status, link forms, counter-evidence form, independence for `supported`)."""
        if not isinstance(rec, dict):
            return "hypothesis must be an object"
        for k in ("pain_id", "title", "claim"):
            if not isinstance(rec.get(k), str) or not rec[k].strip():
                return f"hypothesis needs a non-empty text field {k!r}"
        if rec.get("status") not in HYPOTHESIS_STATUSES:
            return f"status must be one of {HYPOTHESIS_STATUSES} (a hypothesis is never stored as a fact)"
        ev = rec.get("evidence")
        if not isinstance(ev, list) or not ev:
            return "hypothesis needs at least one evidence link"
        for ref in ev:
            if (bad := self._link_problem(ref)):
                return "evidence: " + bad
        counter = rec.get("counter_evidence")
        if not isinstance(counter, dict) or set(counter) not in ({"links"}, {"none_found"}):
            return "counter_evidence is {\"links\": [...]} or {\"none_found\": {\"searched_at\", \"method\"}}; an unsearched hypothesis is not valid"
        if "links" in counter:
            if not isinstance(counter["links"], list) or not counter["links"]:
                return "counter_evidence.links must list at least one link"
            for ref in counter["links"]:
                if (bad := self._link_problem(ref)):
                    return "counter_evidence: " + bad
        else:
            nf = counter["none_found"]
            if not isinstance(nf, dict) or not isinstance(nf.get("method"), str) or not nf["method"].strip():
                return "counter_evidence.none_found needs the `method` of the search"
            try:
                when = datetime.fromisoformat(nf.get("searched_at"))
            except (TypeError, ValueError):
                return "counter_evidence.none_found.searched_at must be an ISO 8601 time"
            if when.tzinfo is None:
                return "counter_evidence.none_found.searched_at must carry a UTC offset"
        if rec["status"] == "contradicted" and "links" not in counter:
            return "status contradicted needs counter-evidence links"
        if rec["status"] == "supported" and len({urlparse(r["url"]).hostname for r in ev}) < 2:
            return "status supported needs evidence from at least 2 different sites (independent sources)"
        return None

    def write_hypothesis(self, record: dict[str, Any]) -> None:
        """Append one hypothesis. Required: pain_id, title, claim, status in HYPOTHESIS_STATUSES, evidence (>= 1 link to a stored source record), counter_evidence (links, or a none-found
        search record with method and time). Every evidence and counter-evidence link must resolve to a stored source record (`source` + `url`). A later write with the same pain_id
        replaces the current record and keeps the older line as history. Invalid input raises ValueError and writes nothing."""
        problem = self._hypothesis_problem(record)
        if problem:
            raise ValueError(problem)
        stored = {(r["source"], r["url"]) for r in self.read_source_records()}
        links = list(record["evidence"]) + list(record["counter_evidence"].get("links", []))
        missing = [f"{r['source']} {r['url']}" for r in links if (r["source"], r["url"]) not in stored]
        if missing:
            raise ValueError(f"link target is not a stored source record: {missing[0]}")
        self._append_jsonl(self._hypotheses_path(), record)

    def _all_hypotheses(self) -> list[dict[str, Any]]:
        return self._read_jsonl(self._hypotheses_path(), lambda r: self._hypothesis_problem(r) is None, "hypothesis")

    def read_hypotheses(self, status: str | None = None) -> list[dict[str, Any]]:
        """Current record of each pain_id (the latest write), in order of first appearance; optional filter by status. An invalid line on disk raises StateCorrupt."""
        if status is not None and status not in HYPOTHESIS_STATUSES:
            raise ValueError(f"unknown status {status!r}")
        latest: dict[str, dict[str, Any]] = {}
        for rec in self._all_hypotheses():
            latest[rec["pain_id"]] = rec
        return [r for r in latest.values() if status is None or r["status"] == status]

    def hypothesis_history(self, pain_id: str) -> list[dict[str, Any]]:
        """Every written version of one hypothesis, oldest first."""
        return [r for r in self._all_hypotheses() if r["pain_id"] == pain_id]

    # ---- experiments (M2709.T17): append-only, latest line per experiment_id is current ------------------------------------------
    def _experiments_path(self) -> Path:
        return self.runtime_dir / "experiments.jsonl"

    @staticmethod
    def _experiment_problem(rec: Any) -> str | None:
        if not isinstance(rec, dict):
            return "experiment must be an object"
        for k in ("experiment_id", "pain_id", "target_customer", "method", "observe", "policy_impact"):
            if not isinstance(rec.get(k), str) or not rec[k].strip():
                return f"experiment needs a non-empty text field {k!r}"
        crit = rec.get("success_criteria")
        if not isinstance(crit, dict) or not isinstance(crit.get("kind"), str) or isinstance(crit.get("min_count"), bool) or not isinstance(crit.get("min_count"), int) or crit["min_count"] < 1:
            return "success_criteria needs a `kind` and an integer `min_count` >= 1"
        for k in ("expected_cost_usd", "budget_limit_usd"):
            if isinstance(rec.get(k), bool) or not isinstance(rec.get(k), (int, float)) or rec[k] < 0:
                return f"{k} must be a number >= 0"
        if rec["expected_cost_usd"] > rec["budget_limit_usd"]:
            return "expected_cost_usd is above budget_limit_usd"
        if rec.get("status") not in EXPERIMENT_STATUSES:
            return f"status must be one of {EXPERIMENT_STATUSES}"
        if (rec["status"] == "completed") != isinstance(rec.get("result"), dict):
            return "a completed experiment carries a result object and a defined one has none"
        return None

    def write_experiment(self, record: dict[str, Any]) -> None:
        """Append one experiment. `pain_id` must be a stored hypothesis (the link); structural problems raise ValueError and nothing is written. The latest line per experiment_id is current."""
        problem = self._experiment_problem(record)
        if problem:
            raise ValueError(problem)
        if record["pain_id"] not in {h["pain_id"] for h in self.read_hypotheses()}:
            raise ValueError(f"experiment links to hypothesis {record['pain_id']!r}, which is not stored")
        self._append_jsonl(self._experiments_path(), record)

    def _all_experiments(self) -> list[dict[str, Any]]:
        return self._read_jsonl(self._experiments_path(), lambda r: self._experiment_problem(r) is None, "experiment")

    def read_experiments(self, pain_id: str | None = None) -> list[dict[str, Any]]:
        """Current record of each experiment_id, in order of first appearance; optional filter by hypothesis."""
        latest: dict[str, dict[str, Any]] = {}
        for rec in self._all_experiments():
            latest[rec["experiment_id"]] = rec
        return [r for r in latest.values() if pain_id is None or r["pain_id"] == pain_id]

    def experiment_history(self, experiment_id: str) -> list[dict[str, Any]]:
        return [r for r in self._all_experiments() if r["experiment_id"] == experiment_id]
