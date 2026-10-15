"""Verifier: schema + step validators, independent of the agent's claims."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .gate import digest
from .graph import Graph, Node
from .radar_validators import VALIDATORS

_TYPES = {"string": str, "number": (int, float), "integer": int, "boolean": bool, "array": list, "object": dict}


class ArtifactInvalid(ValueError):
    pass


def _legacy(step_id: str, data: dict[str, Any], inputs: dict[str, Any]) -> None:
    """Validators for the original five-step market_radar policy (kept for compatibility)."""
    if step_id in ("COLLECT", "NORMALIZE"):
        items = data.get("items")
        if not isinstance(items, list):
            raise ArtifactInvalid(f"{step_id} requires an items array")
        ids = []
        for item in items:
            if not isinstance(item, dict) or not all(isinstance(item.get(f), str) and item[f].strip() for f in ("id", "text", "source")):
                raise ArtifactInvalid(f"{step_id} items require non-empty id, text, and source strings")
            ids.append(item["id"])
        if len(ids) != len(set(ids)):
            raise ArtifactInvalid(f"{step_id} item ids must be unique")
        if step_id == "NORMALIZE":
            original = {item["id"] for item in inputs["COLLECT"]["data"]["items"]}
            if set(ids) != original:
                raise ArtifactInvalid("NORMALIZE must preserve every COLLECT item id")
    elif step_id == "CLUSTER":
        clusters = data.get("clusters")
        if not isinstance(clusters, list) or not clusters:
            raise ArtifactInvalid("CLUSTER requires a non-empty clusters array")
        clustered: list[str] = []
        for cluster in clusters:
            if not isinstance(cluster, dict) or not isinstance(cluster.get("label"), str) or not isinstance(cluster.get("item_ids"), list):
                raise ArtifactInvalid("each cluster requires a label and item_ids array")
            clustered.extend(item for item in cluster["item_ids"] if isinstance(item, str))
        normalized = {item["id"] for item in inputs["NORMALIZE"]["data"]["items"]}
        if len(clustered) != len(set(clustered)) or set(clustered) != normalized:
            raise ArtifactInvalid("CLUSTER must include every normalized item exactly once")
    elif step_id == "VALIDATE":
        if not isinstance(data.get("valid"), bool) or not isinstance(data.get("issues"), list):
            raise ArtifactInvalid("VALIDATE requires boolean valid and an issues array")
        if not data["valid"] or data["issues"]:
            raise ArtifactInvalid("VALIDATE rejected the artifacts: " + "; ".join(map(str, data["issues"][:8])))
    elif step_id == "ACTION":
        actions = data.get("actions")
        if not isinstance(actions, list) or any(not isinstance(a, dict) or not isinstance(a.get("action"), str) or not a["action"].strip() for a in actions):
            raise ArtifactInvalid("ACTION requires an actions array with non-empty action strings")


def _generic_object(data: dict[str, Any], inputs: dict[str, Any], policy: dict[str, Any], ctx: dict[str, Any]) -> None:
    if not data:
        raise ValueError("artifact data must be a non-empty object")


def _squash_text(text: Any) -> str:
    return " ".join(str(text).split()).lower()


def _required_fields(node: Any, data: dict[str, Any], ctx: dict[str, Any] | None = None, inputs: dict[str, Any] | None = None) -> None:
    spec = node.raw.get("required_fields") or {}
    items = data.get(spec["list"])
    if not isinstance(items, list):
        raise ValueError(f"'{spec['list']}' must be an array")
    low, high = spec.get("min", 1), spec.get("max", 100)
    if not low <= len(items) <= high:
        raise ValueError(f"'{spec['list']}' has {len(items)} items, must be {low} to {high}")
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"{spec['list']}[{index}] must be an object")
        bad = [f for f in spec["fields"] if not isinstance(item.get(f), str) or not item[f].strip()]
        if bad:
            raise ValueError(f"{spec['list']}[{index}] needs non-empty text in: {', '.join(bad)}")
    if spec.get("unique"):
        values = [item[spec["unique"]].strip() for item in items]
        dupes = sorted({v for v in values if values.count(v) > 1})
        if dupes:
            raise ValueError(f"{spec['list']}.{spec['unique']} must be unique, repeated: {', '.join(dupes[:5])}")
    refs = spec.get("refs")
    if refs:
        known = {item[refs["key"]].strip() for item in items}
        for index, item in enumerate(items):
            value = item.get(refs["field"])
            wanted = [] if value in (None, "") else [value] if isinstance(value, str) else value
            if not isinstance(wanted, list) or any(not isinstance(v, str) for v in wanted):
                raise ValueError(f"{spec['list']}[{index}].{refs['field']} must be text or a list of text")
            missing = [v for v in wanted if v.strip() not in known]
            if missing:
                raise ValueError(f"{spec['list']}[{index}].{refs['field']} points to unknown {refs['key']}: {', '.join(missing[:5])}")
    ext = spec.get("ext_refs")
    if ext:
        step, _, listname = ext["from"].partition(".")
        upstream = ((inputs or {}).get(step) or {}).get("data", {}).get(listname)
        if not isinstance(upstream, list) or any(not isinstance(u, dict) for u in upstream):
            raise ValueError(f"upstream {ext['from']} must be an array of objects")
        known = {str(u[ext["key"]]).strip() for u in upstream if ext["key"] in u}
        for index, item in enumerate(items):
            value = item.get(ext["field"])
            wanted = [] if value in (None, "") else [value] if isinstance(value, str) else value
            if not isinstance(wanted, list) or any(not isinstance(v, str) for v in wanted):
                raise ValueError(f"{spec['list']}[{index}].{ext['field']} must be text or a list of text")
            missing = [v for v in wanted if v.strip() not in known]
            if missing:
                raise ValueError(f"{spec['list']}[{index}].{ext['field']} points to unknown {ext['key']} in {ext['from']}: {', '.join(missing[:5])}")
    within = spec.get("within")
    if within:
        source = (ctx or {}).get("intent", "") if within["in"] == "intent" else data.get(within["in"])
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"'{within['in']}' must hold the text that {within['field']} is checked against")
        haystack = _squash_text(source)
        for index, item in enumerate(items):
            if _squash_text(item[within["field"]]) not in haystack:
                raise ValueError(f"{spec['list']}[{index}].{within['field']} was not found in {within['in']}")


class Verifier:
    def __init__(self, graph: Graph):
        self.graph = graph

    def verify(self, node: Node, data: Any, inputs: dict[str, Any], ctx: dict[str, Any] | None = None) -> dict[str, str]:
        """Raise ArtifactInvalid unless the data satisfies schema and validator. Returns the check summary."""
        ctx = ctx or {}
        if not isinstance(data, dict):
            raise ArtifactInvalid(f"{node.id} artifact data must be an object")
        for name, kind in node.schema.items():
            value = data.get(name)
            ok = isinstance(value, _TYPES[kind]) and not (kind in ("number", "integer") and isinstance(value, bool))
            if not ok or (kind == "string" and not value.strip()):
                raise ArtifactInvalid(f"{node.id} schema: '{name}' must be a non-empty {kind}")
        if node.validator:
            try:
                if node.validator == "generic.required_fields":
                    _required_fields(node, data, ctx, inputs)
                else:
                    check = _generic_object if node.validator == "generic.object" else VALIDATORS[node.validator]
                    check(data, inputs, self.graph.policy, ctx)
            except ArtifactInvalid:
                raise
            except (ValueError, KeyError, TypeError, AttributeError) as exc:
                raise ArtifactInvalid(f"{node.id}: {exc}") from exc
        elif not node.schema:
            _legacy(node.id, data, inputs)
        return {"schema": "PASS", "validator": node.validator or "legacy"}

    def replay(self, root: Path, state: dict[str, Any]) -> list[dict[str, str]]:
        """Re-verify every committed artifact from disk. Detects tampering and rule drift."""
        failures: list[dict[str, str]] = []
        envelopes: dict[str, dict[str, Any]] = {}
        for node in self.graph.nodes.values():
            record = state["steps"].get(node.id, {})
            if record.get("status") != "pass":
                continue
            path = root / "artifacts" / node.output
            if not path.is_file():
                failures.append({"step": node.id, "reason": "artifact missing"})
                continue
            raw = path.read_bytes()
            if digest(raw) != record.get("artifact_hash"):
                failures.append({"step": node.id, "reason": "artifact hash mismatch"})
                continue
            envelope = json.loads(raw)
            envelopes[node.id] = envelope
            missing = [name for name in node.requires if name not in envelopes]
            if missing:
                failures.append({"step": node.id, "reason": f"not re-verified: upstream {', '.join(missing)} failed integrity"})
                continue
            try:
                inputs = {name: envelopes[name] for name in node.requires}
                ctx = {"intent": state["intent"], "tool_log": (envelope.get("audit") or {}).get("tool_log") or {}}
                self.verify(node, envelope["data"], inputs, ctx)
            except (ArtifactInvalid, KeyError) as exc:
                failures.append({"step": node.id, "reason": str(exc)[:500]})
        return failures
