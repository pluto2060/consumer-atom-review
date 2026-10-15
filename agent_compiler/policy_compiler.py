"""Policy Compiler: policy source (JSON or YAML subset) -> typed, validated execution Graph."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .graph import KNOWN_TOOLS, Graph, Node
from .radar_validators import VALIDATORS

SCHEMA_TYPES = {"string", "number", "integer", "boolean", "array", "object"}
KNOWN_VALIDATORS = set(VALIDATORS) | {"generic.object", "generic.required_fields"}
STEP_ID = re.compile(r"[A-Z][A-Z0-9_]*")


# ---------------------------------------------------------------------------
# Minimal YAML subset (block mappings/sequences, scalars, flow JSON). No dependencies.
# ---------------------------------------------------------------------------

def _strip_comment(line: str) -> str:
    quote = None
    for index, char in enumerate(line):
        if quote:
            if char == quote:
                quote = None
        elif char in "\"'":
            quote = char
        elif char == "#" and (index == 0 or line[index - 1] in " \t"):
            return line[:index]
    return line


def _scalar(text: str) -> Any:
    text = text.strip()
    if text == "":
        return None
    if text[0] in "[{":
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            if text[0] == "[" and text[-1] == "]":
                inner = text[1:-1].strip()
                return [_scalar(part) for part in inner.split(",")] if inner else []
            if text == "{}":
                return {}
            raise ValueError(f"unsupported flow value: {text}")
    if text[0] in "\"'":
        if len(text) < 2 or text[-1] != text[0]:
            raise ValueError(f"unterminated string: {text}")
        return json.loads(text) if text[0] == '"' else text[1:-1]
    lowered = text.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("null", "~"):
        return None
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    if re.fullmatch(r"-?\d+\.\d+", text):
        return float(text)
    return text


_KEY_LINE = re.compile(r"""^(?P<key>"[^"]*"|'[^']*'|[^\s"'\[\]{}#,:][^:]*?)\s*:(?:\s+(?P<value>.*))?$""")


def parse_yaml(text: str) -> Any:
    rows: list[tuple[int, str, int]] = []
    for number, raw in enumerate(text.splitlines(), 1):
        if "\t" in raw[: len(raw) - len(raw.lstrip())]:
            raise ValueError(f"yaml line {number}: tabs are not allowed for indentation")
        stripped = _strip_comment(raw).rstrip()
        if stripped.strip() and stripped.strip() not in ("---", "..."):
            rows.append((len(stripped) - len(stripped.lstrip(" ")), stripped.strip(), number))
    if not rows:
        raise ValueError("empty policy")
    value, index = _block(rows, 0, rows[0][0])
    if index != len(rows):
        raise ValueError(f"yaml line {rows[index][2]}: unexpected indentation")
    return value


def _block(rows: list[tuple[int, str, int]], index: int, indent: int) -> tuple[Any, int]:
    if rows[index][1] == "-" or rows[index][1].startswith("- "):
        return _sequence(rows, index, indent)
    return _mapping(rows, index, indent)


def _sequence(rows, index, indent):
    items: list[Any] = []
    while index < len(rows) and rows[index][0] == indent and (rows[index][1] == "-" or rows[index][1].startswith("- ")):
        rest = rows[index][1][1:].strip()
        number = rows[index][2]
        if not rest:
            if index + 1 < len(rows) and rows[index + 1][0] > indent:
                value, index = _block(rows, index + 1, rows[index + 1][0])
            else:
                value, index = None, index + 1
            items.append(value)
        elif _KEY_LINE.match(rest) and not rest.startswith(("[", "{")):
            rows[index] = (indent + 2, rest, number)
            value, index = _mapping(rows, index, indent + 2)
            items.append(value)
        else:
            items.append(_scalar(rest))
            index += 1
    return items, index


def _mapping(rows, index, indent):
    result: dict[str, Any] = {}
    while index < len(rows) and rows[index][0] == indent and not (rows[index][1] == "-" or rows[index][1].startswith("- ")):
        match = _KEY_LINE.match(rows[index][1])
        if not match:
            raise ValueError(f"yaml line {rows[index][2]}: expected 'key: value'")
        key = _scalar(match.group("key")) if match.group("key")[0] in "\"'" else match.group("key")
        key = str(key)
        if key in result:
            raise ValueError(f"yaml line {rows[index][2]}: duplicate key {key}")
        value_text = match.group("value")
        number = rows[index][2]
        if value_text is None or value_text.strip() == "":
            nxt = index + 1
            if nxt < len(rows) and (rows[nxt][0] > indent or (rows[nxt][0] == indent and (rows[nxt][1] == "-" or rows[nxt][1].startswith("- ")))):
                value, index = _block(rows, nxt, rows[nxt][0])
            else:
                value, index = None, nxt
        else:
            try:
                value = _scalar(value_text)
            except ValueError as exc:
                raise ValueError(f"yaml line {number}: {exc}") from exc
            index += 1
        result[key] = value
    return result, index


def parse_policy_text(text: str) -> dict[str, Any]:
    stripped = text.lstrip()
    if stripped.startswith("{"):
        value = json.loads(text)          # JSON is a YAML subset
    else:
        value = parse_yaml(text)
    if not isinstance(value, dict):
        raise ValueError("policy must be a mapping")
    return value


# ---------------------------------------------------------------------------
# Compiler
# ---------------------------------------------------------------------------

def _normalize_steps(raw: dict[str, Any]) -> list[dict[str, Any]]:
    steps = raw.get("steps")
    if isinstance(steps, dict):          # roadmap style: steps: {NAME: {...}}
        steps = [{"id": key, **(value or {})} for key, value in steps.items()]
    if not isinstance(steps, list) or not steps or not all(isinstance(s, dict) for s in steps):
        raise ValueError("policy must define a non-empty steps list or mapping")
    return steps


def _mcp_servers(raw: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Validated `mcp_servers`: {name: {url (https), allow_operations [..], operation_field, allow_hosts [..]}}."""
    servers = raw.get("mcp_servers") or {}
    if not isinstance(servers, dict):
        raise ValueError("mcp_servers must be a mapping of server name to its settings")
    out: dict[str, dict[str, Any]] = {}
    for name, cfg in servers.items():
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]*", name):
            raise ValueError(f"mcp server name {name!r} must be letters, digits and hyphens (no underscores: they delimit tool names)")
        if isinstance(cfg, dict) and cfg.get("builtin") == "sources":
            out[name] = builtin_sources(name, cfg)
            continue
        if isinstance(cfg, dict) and "builtin" in cfg:
            out[name] = builtin_governance(name, cfg)
            continue
        if not isinstance(cfg, dict) or not str(cfg.get("url", "")).startswith("https://"):
            raise ValueError(f"mcp server {name} needs an https `url`")
        ops = cfg.get("allow_operations") or []
        hosts = cfg.get("allow_hosts") or []
        if not isinstance(ops, list) or not all(isinstance(o, str) and o for o in ops) \
                or not isinstance(hosts, list) or not all(isinstance(h, str) and h for h in hosts):
            raise ValueError(f"mcp server {name}: allow_operations and allow_hosts must be lists of strings")
        out[name] = {"url": cfg["url"], "allow_operations": ops, "allow_hosts": hosts,
                     "operation_field": str(cfg.get("operation_field") or "operation_id"),
                     "commercial_ok": cfg.get("commercial_ok") is True}      # a named human sets this after reading the terms
    return out


def builtin_governance(name: str, cfg: dict[str, Any]) -> dict[str, Any]:
    """`{builtin: governance, tiers: [read, run]}`: the compiler's OWN MCP server, never an arbitrary command or url. A step that lists its tools
    (`mcp__<name>__<tool>`) may call only those; `run` tools start child runs (lineage, depth limit and a shared tree budget apply)."""
    from .mcp_server import TOOLS
    if cfg.get("builtin") != "governance" or set(cfg) - {"builtin", "tiers", "max_depth"}:
        raise ValueError(f"mcp server {name}: a builtin server must be exactly `builtin: governance` with optional `tiers` and `max_depth` (no url, no command)")
    depth = cfg.get("max_depth")
    if depth is not None and (isinstance(depth, bool) or not isinstance(depth, int) or not 1 <= depth <= 3):
        raise ValueError(f"mcp server {name}: max_depth must be an integer from 1 to 3 (how many levels of runs this policy's run tree may reach)")
    tiers = cfg.get("tiers") or ["read"]
    if not isinstance(tiers, list) or not tiers or any(t not in ("read", "run") for t in tiers):
        raise ValueError(f"mcp server {name}: tiers must be a list drawn from read, run")
    wanted = sorted(set(tiers) | {"read"})
    return {"builtin": "governance", "tiers": wanted, "allow_tools": sorted(n for n, spec in TOOLS.items() if spec[0] in wanted), "commercial_ok": True,
            **({"max_depth": depth} if depth is not None else {})}


def builtin_sources(name: str, cfg: dict[str, Any]) -> dict[str, Any]:
    """`{builtin: sources}`: the compiler's own read-only community-voice server (YouTube comments, Stack Exchange questions); no url, no command, no tiers."""
    from .sources_server import TOOLS as SOURCE_TOOLS
    if set(cfg) != {"builtin"}:
        raise ValueError(f"mcp server {name}: a `sources` builtin server is exactly `builtin: sources` (no url, no command, no other settings)")
    return {"builtin": "sources", "allow_tools": sorted(SOURCE_TOOLS), "commercial_ok": False}


def _mcp_tool_ok(tool: str, servers: dict[str, dict[str, Any]]) -> bool:
    parts = tool.split("__")
    if not (len(parts) == 3 and parts[0] == "mcp" and parts[1] in servers and bool(parts[2])):
        return False
    server = servers[parts[1]]
    return parts[2] in server["allow_tools"] if server.get("builtin") else True


def _check_required_fields_spec(sid: str, spec: Any, requires: list[str] | None = None) -> None:
    """M2631/M2642: `required_fields` = {list: top-level array field, fields: names every item must fill, min/max: item count} plus optional declarative rules:
    `unique` (an item field whose values must differ), `refs` ({field, key}: every value of `field` (text or list of text) must be a `key` value of an item of the same list)
    and `within` ({field, in}: the item's `field` text must appear in the run intent (`in: "intent"`) or in another top-level text field)."""
    allowed = {"list", "fields", "min", "max", "unique", "refs", "within", "ext_refs"}
    if not isinstance(spec, dict) or set(spec) - allowed:
        raise ValueError(f"step {sid}: validator generic.required_fields needs required_fields = {{list, fields, min?, max?, unique?, refs?, within?, ext_refs?}}")
    fields = spec.get("fields")
    if not isinstance(spec.get("list"), str) or not spec["list"].strip():
        raise ValueError(f"step {sid}: required_fields.list must be a field name")
    if not isinstance(fields, list) or not fields or any(not isinstance(f, str) or not f.strip() for f in fields):
        raise ValueError(f"step {sid}: required_fields.fields must be a non-empty list of field names")
    low, high = spec.get("min", 1), spec.get("max", 100)
    if any(isinstance(v, bool) or not isinstance(v, int) for v in (low, high)) or not 1 <= low <= high <= 1000:
        raise ValueError(f"step {sid}: required_fields.min/max must be integers with 1 <= min <= max <= 1000")
    if "unique" in spec and (not isinstance(spec["unique"], str) or spec["unique"] not in fields):
        raise ValueError(f"step {sid}: required_fields.unique must be one of the fields")
    refs = spec.get("refs")
    if refs is not None and (not isinstance(refs, dict) or set(refs) != {"field", "key"} or refs["key"] not in fields or not isinstance(refs["field"], str) or not refs["field"].strip()):
        raise ValueError(f"step {sid}: required_fields.refs must be {{field, key}} with key one of the fields")
    ext = spec.get("ext_refs")
    if ext is not None:
        source = ext.get("from") if isinstance(ext, dict) else None
        step, _, listname = source.partition(".") if isinstance(source, str) else ("", "", "")
        if (not isinstance(ext, dict) or set(ext) != {"field", "from", "key"} or not all(isinstance(ext[k], str) and ext[k].strip() for k in ext)
                or not step or not listname or step not in (requires or [])):
            raise ValueError(f"step {sid}: required_fields.ext_refs must be {{field, from: 'STEP.list', key}} where STEP is one of this step's requires")
    within = spec.get("within")
    if within is not None and (not isinstance(within, dict) or set(within) != {"field", "in"} or within["field"] not in fields or not isinstance(within["in"], str) or not within["in"].strip()):
        raise ValueError(f"step {sid}: required_fields.within must be {{field, in}} with field one of the fields and in 'intent' or a top-level field name")


IR_STEP_KEYS = ("validator", "schema", "tools", "max_turns", "timeout", "max_attempts", "contract", "repair", "output",
                "description", "instruction")
COMPARATORS = {">=": lambda a, b: a >= b, "<=": lambda a, b: a <= b, ">": lambda a, b: a > b, "<": lambda a, b: a < b,
               "==": lambda a, b: a == b}
CONDITION = re.compile(r"^\s*([a-z_]+)\s*(>=|<=|==|>|<)\s*(-?\d+(?:\.\d+)?)\s*$")
ROUTING = ("dependency_aware", "sequential")
# Fixed action vocabulary. `tools`: may the step use tools at all. `needs`: it must directly depend on a step of this action,
# so the structure itself guarantees the order evidence -> checking -> answer (compile-time, not a prompt).
ACTIONS: dict[str, dict[str, Any]] = {
    "retrieve": {"tools": True, "needs": None},
    "critique": {"tools": False, "needs": None},
    "verify": {"tools": False, "needs": "retrieve"},
    "synthesize": {"tools": False, "needs": "verify"},
}
MODEL_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._\-\[\]]{1,59}")
TRACE_FLAGS = ("inputs", "outputs", "tool_calls", "latency", "tokens", "cost")


def parse_condition(text: Any) -> tuple[str, str, float]:
    """`measure >= 0.85` -> (measure, comparator, threshold). The measure must be one the runtime can compute."""
    from .principle_policy import MEASURES
    found = CONDITION.match(str(text))
    if not found:
        raise ValueError(f"termination condition {text!r} must look like '<measure> >= <number>' (comparators: {sorted(COMPARATORS)})")
    measure, comparator, value = found.group(1), found.group(2), float(found.group(3))
    if measure not in MEASURES:
        raise ValueError(f"termination measure {measure!r} is not computable by the runtime; use one of {list(MEASURES)}")
    return measure, comparator, value


def _from_ir(raw: dict[str, Any]) -> dict[str, Any]:
    """Convert a `policy_ir` document (objective / actors / operations / control / observability) into the step format.

    Operation ids become upper-case step ids, `depends_on` becomes `requires`, and operations are put in a stable
    topological order (a dependency cycle or an unknown dependency is a compile error)."""
    ir = raw["policy_ir"]
    if not isinstance(ir, dict):
        raise ValueError("policy_ir must be a mapping")
    objective = ir.get("objective") or {}
    if not isinstance(objective, dict) or not objective.get("id"):
        raise ValueError("policy_ir.objective needs an `id` (and a `goal`)")
    actors_raw = ir.get("actors") or []
    specs: dict[str, dict[str, Any] | None] = {}
    if isinstance(actors_raw, dict):                       # mapping form: name -> {tools, model}; no `tools` means NO tools
        for name, spec in actors_raw.items():
            spec = spec or {}
            if not isinstance(name, str) or not name or not isinstance(spec, dict) or set(spec) - {"tools", "model"}:
                raise ValueError(f"actor {name!r}: settings may only contain `tools` and `model`")
            tools = spec.get("tools", [])
            if not isinstance(tools, list) or not all(isinstance(t, str) for t in tools):
                raise ValueError(f"actor {name}: tools must be a list of tool names")
            if "model" in spec and not (isinstance(spec["model"], str) and MODEL_NAME.fullmatch(spec["model"])):
                raise ValueError(f"actor {name}: model must be a model id or alias such as `sonnet`")
            specs[name] = {"tools": tools, **({"model": spec["model"]} if "model" in spec else {})}
        actors = list(specs)
    else:
        actors = actors_raw
        if not isinstance(actors, list) or not all(isinstance(a, str) and a for a in actors) or len(set(actors)) != len(actors):
            raise ValueError("policy_ir.actors must be a list of unique names or a mapping of name to {tools, model}")
        specs = {name: None for name in actors}
    operations = ir.get("operations")
    if not isinstance(operations, list) or not operations or not all(isinstance(o, dict) for o in operations):
        raise ValueError("policy_ir.operations must be a non-empty list of operations")
    by_id: dict[str, dict[str, Any]] = {}
    for op in operations:
        oid = op.get("id")
        if not isinstance(oid, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", oid):
            raise ValueError(f"operation id {oid!r} must be letters, digits and underscores")
        if oid.upper() in by_id:
            raise ValueError(f"duplicate operation id: {oid}")
        if op.get("actor") not in actors:
            raise ValueError(f"operation {oid}: actor {op.get('actor')!r} is not declared in actors {actors}")
        spec = specs[op["actor"]]
        if spec is not None:                                # least privilege: a step may use only what its actor is granted
            over = sorted(set(op.get("tools") or ()) - set(spec["tools"]))
            if over:
                raise ValueError(f"operation {oid}: tools {over} are not granted to actor {op['actor']} (granted: {spec['tools']})")
        if not isinstance(op.get("action"), str) or not op["action"].strip():
            raise ValueError(f"operation {oid} needs an `action`")
        deps = op.get("depends_on") or []
        if not isinstance(deps, list) or not all(isinstance(d, str) for d in deps):
            raise ValueError(f"operation {oid}: depends_on must be a list of operation ids")
        by_id[oid.upper()] = {**op, "_deps": [d.upper() for d in deps]}
    for sid, op in by_id.items():
        unknown = [d for d in op["_deps"] if d not in by_id]
        if unknown or sid in op["_deps"]:
            raise ValueError(f"operation {sid.lower()}: depends_on names unknown or itself: {[d.lower() for d in unknown] or [sid.lower()]}")
    ordered: list[str] = []
    pending = list(by_id)
    while pending:                                                    # stable Kahn: keep declared order among ready steps
        ready = [s for s in pending if set(by_id[s]["_deps"]) <= set(ordered)]
        if not ready:
            raise ValueError(f"dependency cycle among operations: {[s.lower() for s in pending]}")
        ordered.append(ready[0])
        pending.remove(ready[0])
    steps = []
    for position, sid in enumerate(ordered):
        op = by_id[sid]
        step: dict[str, Any] = {"id": sid, "requires": op["_deps"], "actor": op["actor"], "action": op["action"],
                                "parallel": op.get("parallel") is True,
                                "next": [ordered[position + 1]] if position + 1 < len(ordered) else []}
        step.setdefault("description", f"{op['actor']}: {op['action']}")
        for key in IR_STEP_KEYS:
            if key in op:
                step[key] = op[key]
        spec = specs[op["actor"]]
        if spec is not None:
            step.setdefault("tools", list(spec["tools"]))
            if "model" in spec:
                step["model"] = spec["model"]
        step.setdefault("description", f"{op['actor']}: {op['action']}")
        step.setdefault("instruction", f"Act as the {op['actor']} and perform '{op['action']}' for the intent, using only the "
                                       "upstream artifacts. Return a JSON object with your findings.")
        steps.append(step)
    control = ir.get("control") or {}
    routing = (control.get("routing") or {}).get("strategy", "dependency_aware")
    if routing not in ROUTING:
        raise ValueError(f"control.routing.strategy must be one of {list(ROUTING)}")
    termination = control.get("termination")
    if termination is not None:
        if not isinstance(termination, dict) or "condition" not in termination:
            raise ValueError("control.termination needs a `condition`")
        parse_condition(termination["condition"])
        _check_termination(termination)
    observability = (ir.get("observability") or {}).get("trace") or {}
    if not isinstance(observability, dict) or set(observability) - set(TRACE_FLAGS) \
            or not all(isinstance(v, bool) for v in observability.values()):
        raise ValueError(f"observability.trace keys must be booleans among {list(TRACE_FLAGS)}")
    out = {k: v for k, v in raw.items() if k != "policy_ir"}
    out.update(name=str(objective["id"]), version=str(ir.get("version", raw.get("version", "1.0"))),
               objective=str(objective.get("goal", "")), actors=actors, routing=routing, steps=steps,
               actor_specs={n: s for n, s in specs.items() if s is not None},
               max_attempts=ir.get("max_attempts", raw.get("max_attempts", 2)),
               observability={"trace": {flag: observability.get(flag, True) for flag in TRACE_FLAGS}})
    if termination is not None:
        out["termination"] = {"condition": str(termination["condition"]),
                              **({"enforce": True} if termination.get("enforce") is True else {})}
    return out


def _enforce_governance(gov: Any, ids: set[str]) -> None:
    """Hard rules for a policy that declares `governance` (policies without it are legacy and unchanged). A malformed block must not
    silently skip enforcement, so wrong shapes and unknown risk classes are rejected here too; the finer checks stay lint warnings."""
    if gov is None:
        return
    if not isinstance(gov, dict):
        raise ValueError("governance must be a mapping")
    risk, approval = gov.get("risk"), gov.get("approval")
    if risk is not None and not (isinstance(risk, dict) and risk.get("class") in ("low", "medium", "high")):
        raise ValueError("governance.risk must be {class: low|medium|high} (lower case)")
    if approval is not None and not (isinstance(approval, dict) and isinstance(approval.get("steps", []), list)
                                     and all(isinstance(s, str) for s in approval.get("steps", []))):
        raise ValueError("governance.approval must be {steps: [step ids]}")
    steps = (approval or {}).get("steps", [])
    unknown = sorted(set(steps) - ids)
    if unknown:
        raise ValueError(f"governance.approval.steps names unknown steps {unknown}")
    if (risk or {}).get("class") == "high" and not steps:
        raise ValueError("a high-risk policy (governance.risk.class: high) must list at least one step in governance.approval.steps")


def required_approvals(policy: dict[str, Any]) -> list[str]:
    """Steps that `governance.approval.steps` makes mandatory for every run (empty for a legacy policy)."""
    approval = (policy.get("governance") or {}).get("approval") if isinstance(policy.get("governance"), dict) else None
    steps = approval.get("steps") if isinstance(approval, dict) else None
    return [s for s in steps if isinstance(s, str)] if isinstance(steps, list) else []


ENFORCEABLE_MEASURES = ("tool_calls", "latency_ms", "gate_blocks", "retries", "repairs", "human_interventions",
                        "steps_passed", "steps_total", "first_pass_rate", "tokens_total", "cost_usd")


def _check_termination(termination: Any) -> None:
    """`termination.enforce: true` stops the run once the condition holds after a committed step. Only measures that are defined
    on a partial run may be enforced; without `enforce` the condition stays advisory (shown after the run)."""
    if termination is None:
        return
    if not isinstance(termination, dict):
        raise ValueError("termination must be a mapping")
    enforce = termination.get("enforce", False)
    if not isinstance(enforce, bool):
        raise ValueError("termination.enforce must be true or false")
    if not enforce:
        return
    measure = parse_condition(termination.get("condition"))[0]
    if measure not in ENFORCEABLE_MEASURES:
        raise ValueError(f"termination measure {measure!r} is only defined after the run completes; with enforce use one of {list(ENFORCEABLE_MEASURES)}")


def compile_policy(raw: dict[str, Any]) -> Graph:
    if "policy_ir" in raw:
        raw = _from_ir(raw)
    if "termination" in raw:
        parse_condition((raw["termination"] or {}).get("condition"))
        _check_termination(raw["termination"])
    if raw.get("use", "internal") not in ("internal", "commercial"):
        raise ValueError("`use` must be \"internal\" or \"commercial\" (commercial disables every source whose terms are not cleared)")
    budget = raw.get("budget")
    positive = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0
    if budget is not None and (not isinstance(budget, dict) or not budget or set(budget) - {"run_usd", "step_usd"}
                               or not all(positive(v) for v in budget.values())):
        raise ValueError("budget must be a mapping with `run_usd` and/or `step_usd`, each a positive number of US dollars")
    mcp_servers = _mcp_servers(raw)
    if any("run" in (m.get("tiers") or ()) for m in mcp_servers.values()) and not isinstance(((budget or {}).get("run_usd")), (int, float)):
        raise ValueError("a policy whose steps can start runs (governance tier `run`) must set budget.run_usd: the run tree shares that cap")
    if any("run" in (m.get("tiers") or ()) for m in mcp_servers.values()) and not isinstance(((budget or {}).get("step_usd")), (int, float)):
        raise ValueError("a policy whose steps can start runs must also set budget.step_usd: every call reserves its ceiling in the shared tree cap, "
                         "so without a per-call ceiling the first call would reserve the whole cap and starve the runs it starts")
    steps = _normalize_steps(raw)
    ids = [step.get("id") for step in steps]
    if any(not isinstance(item, str) or not STEP_ID.fullmatch(item) for item in ids):
        raise ValueError("step ids must be uppercase identifiers")
    if len(ids) != len(set(ids)):
        raise ValueError("step ids must be unique")
    for key in ("name", "version"):
        if not isinstance(raw.get(key), (str, int, float)) or not str(raw[key]).strip():
            raise ValueError(f"policy requires {key}")
    known = set(ids)
    outputs: set[str] = set()
    nodes: dict[str, Node] = {}
    for position, step in enumerate(steps):
        sid = step["id"]
        requires = step.get("requires") or []
        following = step.get("next")
        if following is None:
            following = [ids[position + 1]] if position + 1 < len(ids) else []
        if not isinstance(requires, list) or set(requires) - known:
            raise ValueError(f"invalid prerequisites for {sid}")
        if not isinstance(following, list) or set(following) - known:
            raise ValueError(f"invalid transitions for {sid}")
        output = Path(str(step.get("output") or f"{sid.lower()}.json"))
        if output.is_absolute() or ".." in output.parts or not output.name.endswith(".json"):
            raise ValueError(f"unsafe artifact path for {sid}")
        if output.as_posix() in outputs:
            raise ValueError(f"artifact path is reused: {output}")
        outputs.add(output.as_posix())
        validator = step.get("validator")
        if validator is not None and validator not in KNOWN_VALIDATORS:
            raise ValueError(f"unknown validator for {sid}: {validator}")
        if validator == "generic.required_fields":
            _check_required_fields_spec(sid, step.get("required_fields"), requires)
        schema = step.get("schema") or {}
        if not isinstance(schema, dict) or any(t not in SCHEMA_TYPES for t in schema.values()):
            raise ValueError(f"invalid schema for {sid}: types must be one of {sorted(SCHEMA_TYPES)}")
        tools = tuple(step.get("tools") or ())
        unknown = sorted(t for t in tools if t not in KNOWN_TOOLS and not _mcp_tool_ok(t, mcp_servers))
        if unknown:
            raise ValueError(f"unknown tools for {sid}: {unknown} (MCP tools must be `mcp__<server>__<tool>` for a server "
                             f"declared in mcp_servers: {sorted(mcp_servers)})")
        if "max_budget_usd" in step and not positive(step["max_budget_usd"]):
            raise ValueError(f"max_budget_usd for {sid} must be a positive number of US dollars")
        max_attempts = step.get("max_attempts", raw.get("max_attempts", 1))
        max_turns = step.get("max_turns", 1)
        timeout = step.get("timeout", 180)
        for label, value, low, high in (("max_attempts", max_attempts, 1, 5), ("max_turns", max_turns, 1, 60),
                                        ("timeout", timeout, 5, 1800)):
            if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
                raise ValueError(f"{label} for {sid} must be an integer between {low} and {high}")
        repair = step.get("repair")
        if repair is not None:
            if not isinstance(repair, dict):
                raise ValueError(f"repair for {sid} must be an object")
            earlier = ids[: position + 1]                     # a step may only roll back itself or predecessors
            allowed = repair.get("allowed_from") or [repair.get("rerun_from", sid)]
            default = repair.get("rerun_from", allowed[0])
            if not isinstance(allowed, list) or set(allowed) - set(earlier) or default not in allowed:
                raise ValueError(f"repair for {sid} must roll back to the step itself or an earlier step")
            repair = {"allowed_from": allowed, "default": default, "max": int(repair.get("max", 1))}
        nodes[sid] = Node(
            id=sid, requires=tuple(requires), output=output.as_posix(), next=tuple(following),
            description=str(step.get("description", "")), instruction=str(step.get("instruction", "")),
            contract=step.get("contract") or {}, schema=schema, validator=validator, tools=tools,
            max_turns=max_turns, timeout=timeout, max_attempts=max_attempts, repair=repair, raw=dict(step))
    for sid, node in nodes.items():                           # fixed action vocabulary and its structural rules
        action = node.raw.get("action")
        if action is None:
            continue
        rule = ACTIONS.get(action)
        if rule is None:
            raise ValueError(f"step {sid}: action {action!r} is not in the vocabulary {sorted(ACTIONS)}")
        if node.tools and not rule["tools"]:
            raise ValueError(f"step {sid}: a `{action}` step must not use tools (tools: {list(node.tools)}); only `retrieve` may")
        if rule["needs"] and not any(nodes[r].raw.get("action") == rule["needs"] for r in node.requires):
            raise ValueError(f"step {sid}: a `{action}` step must directly depend on a `{rule['needs']}` step "
                             f"(it depends on {list(node.requires) or 'nothing'})")
    seen: set[str] = set()
    for sid in ids:                                           # declared order must be a valid topological order
        if set(nodes[sid].requires) - seen:
            raise ValueError(f"policy order violates prerequisites for {sid}")
        seen.add(sid)
    for current, nxt in zip(ids, ids[1:]):
        if nxt not in nodes[current].next:
            raise ValueError(f"missing transition {current} -> {nxt}")
    if nodes[ids[-1]].next:
        raise ValueError("terminal step must not declare a next step")
    _enforce_governance(raw.get("governance"), set(ids))
    policy = dict(raw)
    policy["steps"] = [dict(step, requires=list(nodes[step["id"]].requires), next=list(nodes[step["id"]].next),
                            output=nodes[step["id"]].output, description=nodes[step["id"]].description)
                       for step in steps]
    return Graph(name=str(raw["name"]), version=str(raw["version"]), objective=str(raw.get("objective", "")),
                 nodes=nodes, policy=policy)


def load_graph(path: Path) -> Graph:
    return compile_policy(parse_policy_text(Path(path).read_text(encoding="utf-8")))
