"""Deterministic checks that turn a principle/rule into something the runtime can really enforce.

A check is attached to a rule or principle by a NAMED HUMAN (append-only overlay, never by the model) and names a step,
a predicate from the fixed library below, and a mode:

  shadow   evaluate and record only (default; safe rollout)
  warn     evaluate, record, and show in the review
  enforce  a violation fails verification of that step (the agent gets the reason and retries)

Checks can only TIGHTEN: they run after the workflow's own validators pass and can reject an artifact, never accept one
the validators rejected. Predicates read only the step's artifact data; they never call a model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from .state import now

MODES = ("shadow", "warn", "enforce")


class CheckInvalid(ValueError):
    pass


def resolve(data: Any, path: str) -> list[Any]:
    """Values at a dotted path; `name[*]` flattens a list (e.g. `actions[*].type`)."""
    current = [data]
    for part in path.split("."):
        wildcard = part.endswith("[*]")
        key = part[:-3] if wildcard else part
        nxt = []
        for value in current:
            if isinstance(value, dict) and key in value:
                found = value[key]
                nxt.extend(found if wildcard and isinstance(found, list) else [found])
        current = nxt
    return current


def _max_count(data, p):
    values = resolve(data, p["path"])
    n = len(values[0]) if len(values) == 1 and isinstance(values[0], list) else len(values)     # a list field counts its entries
    return [f"{p['path']} has {n} value(s); at most {p['max']} allowed"] if n > p["max"] else []


def _max_chars(data, p):
    return [f"{p['path']} value of {len(v)} chars exceeds {p['max']}: {v[:40]!r}…"
            for v in resolve(data, p["path"]) if isinstance(v, str) and len(v) > p["max"]]


def _nonempty(data, p):
    values = resolve(data, p["path"])
    return [] if values and all(v not in (None, "", [], {}) for v in values) else [f"{p['path']} must be present and non-empty"]


def _forbid_values(data, p):
    bad = sorted({str(v) for v in resolve(data, p["path"]) if v in p["values"]})
    return [f"{p['path']} contains forbidden value(s): {bad}"] if bad else []


def _min_share(data, p):
    values = resolve(data, p["path"])
    if not values:
        return []
    share = sum(1 for v in values if v in p["values"]) / len(values)
    return [f"only {share:.0%} of {p['path']} are in {p['values']}; at least {p['min']:.0%} required"] if share < p["min"] else []


def _number(v): return isinstance(v, (int, float)) and not isinstance(v, bool)


# name -> (required params with validators, function(data, params) -> violation messages)
PREDICATES: dict[str, tuple[dict[str, Callable[[Any], bool]], Callable[[Any, dict[str, Any]], list[str]]]] = {
    "max_count": ({"path": lambda v: isinstance(v, str) and bool(v), "max": lambda v: _number(v) and v >= 0}, _max_count),
    "max_chars": ({"path": lambda v: isinstance(v, str) and bool(v), "max": lambda v: _number(v) and v > 0}, _max_chars),
    "nonempty": ({"path": lambda v: isinstance(v, str) and bool(v)}, _nonempty),
    "forbid_values": ({"path": lambda v: isinstance(v, str) and bool(v),
                       "values": lambda v: isinstance(v, list) and bool(v)}, _forbid_values),
    "min_share": ({"path": lambda v: isinstance(v, str) and bool(v), "values": lambda v: isinstance(v, list) and bool(v),
                   "min": lambda v: _number(v) and 0 <= v <= 1}, _min_share),
}


def validate_check(check: Any, steps: set[str] | None = None) -> dict[str, Any]:
    if not isinstance(check, dict):
        raise CheckInvalid("check must be an object")
    name, params, step = check.get("predicate"), check.get("params"), check.get("step")
    if name not in PREDICATES:
        raise CheckInvalid(f"predicate must be one of {sorted(PREDICATES)}")
    spec = PREDICATES[name][0]
    if not isinstance(params, dict) or set(params) != set(spec) or not all(spec[k](params[k]) for k in spec):
        raise CheckInvalid(f"{name} needs exactly the params {sorted(spec)} with valid values")
    if not isinstance(step, str) or not step or (steps is not None and step not in steps):
        raise CheckInvalid(f"step must be one of the workflow's steps {sorted(steps or [])}")
    if check.get("mode", "shadow") not in MODES:
        raise CheckInvalid(f"mode must be one of {list(MODES)}")
    return {"step": step, "predicate": name, "params": params, "mode": check.get("mode", "shadow")}


# ---- overlay (append-only, named human) ---------------------------------------------------------------

def _path(directory: Path, doc_id: str) -> Path:
    return Path(directory) / f"{doc_id}.checks.json"


def load_checks(directory: Path, doc_id: str) -> dict[str, dict[str, Any]]:
    """Latest check per element id. An entry with check=null removes it."""
    path = _path(directory, doc_id)
    latest: dict[str, dict[str, Any]] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            entry = json.loads(line)
            if entry.get("check") is None:
                latest.pop(entry["element"], None)
            else:
                latest[entry["element"]] = entry
    return latest


def set_check(directory: Path, doc: dict[str, Any], element_id: str, check: dict[str, Any] | None, by: str,
              steps: set[str] | None = None) -> dict[str, Any]:
    from .gate import SELF_APPROVERS
    if not isinstance(by, str) or by.strip().lower() in SELF_APPROVERS:
        raise CheckInvalid("a named human is required; agents cannot attach or change checks")
    element = next((e for e in doc["elements"] if e["id"] == element_id and e["type"] in ("principle", "rule")), None)
    if element is None:
        raise CheckInvalid(f"unknown principle/rule element: {element_id}")
    entry = {"element": element_id, "check": validate_check(check, steps) if check is not None else None,
             "by": by.strip(), "at": now()}
    previous = load_checks(directory, doc["id"]).get(element_id)
    same = bool(previous) and all(previous["check"][k] == (entry["check"] or {}).get(k) for k in ("step", "predicate", "params"))
    if entry["check"] and entry["check"]["mode"] == "enforce" and not same:
        raise CheckInvalid("a new or changed check must run as shadow or warn first; promote to enforce afterwards")
    with _path(directory, doc["id"]).open("a", encoding="utf-8") as log:
        log.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


AUTO_PROMOTE_RUNS = 5


def auto_promote(directory: Path, runtimes: list[Any], min_runs: int = AUTO_PROMOTE_RUNS) -> list[dict[str, Any]]:
    """shadow -> warn, only, once a check has been evaluated in `min_runs` distinct finished runs.

    Warn changes nothing about what runs (it only surfaces violations), so this step needs no human. Promotion to
    `enforce` never happens here: that is the change that can block work and stays a named human's decision."""
    directory = Path(directory)
    promoted: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.checks.json")):
        doc_id = path.name[: -len(".checks.json")]
        for element, entry in load_checks(directory, doc_id).items():
            check = entry["check"]
            if check["mode"] != "shadow":
                continue
            runs = set()
            for runtime in runtimes:
                for run in runtime.list_runs(200):
                    if run["status"] not in ("completed", "failed"):
                        continue
                    trace = runtime.audit.read(runtime.store.root(run["id"]))
                    if any(e.get("event") == "principle_check" and e.get("policy") == doc_id and e.get("element") == element
                           and e.get("mode") == "shadow" for e in trace):
                        runs.add(run["id"])
            if len(runs) < min_runs:
                continue
            record = {"element": element, "check": {**check, "mode": "warn"}, "by": f"auto-promote (shadow in {len(runs)} runs)",
                      "at": now(), "auto": True}
            with _path(directory, doc_id).open("a", encoding="utf-8") as log:
                log.write(json.dumps(record, ensure_ascii=False) + "\n")
            promoted.append({"policy": doc_id, "element": element, "runs": len(runs)})
    return promoted


def apply_checks(document: dict[str, Any], checks: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out = json.loads(json.dumps(document))
    for element in out["elements"]:
        element.pop("check", None)                 # only a human-attached overlay may carry a check, never the document itself
        entry = checks.get(element["id"])
        if entry:
            element["check"] = {**entry["check"], "by": entry["by"], "at": entry["at"]}
    return out


# ---- runtime ------------------------------------------------------------------------------------------

def frozen_checks(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Checks of the active principle documents, copied into the run state so a run is judged by the rules it began with."""
    return [{"policy": d["id"], "version": d["version"], "element": e["id"], **{k: e["check"][k] for k in ("step", "predicate", "params", "mode")}}
            for d in documents for e in d["elements"] if e.get("check") and e["check"].get("by")]


def evaluate(checks: list[dict[str, Any]], step: str, data: Any) -> list[dict[str, Any]]:
    results = []
    for check in checks:
        if check["step"] != step:
            continue
        violations = PREDICATES[check["predicate"]][1](data, check["params"])
        results.append({"policy": check["policy"], "element": check["element"], "predicate": check["predicate"],
                        "mode": check["mode"], "violations": violations})
    return results
