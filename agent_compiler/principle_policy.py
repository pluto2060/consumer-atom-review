"""Principle policies: turn an unstructured policy document into a structured, source-grounded, registered policy.

A principle policy is NOT a workflow (step graph). It captures principles, conditional rules, metrics, failure
conditions, disclosure levels and evolution rules, each grounded in verbatim quotes of the source. Claude proposes the
structure; the compiler verifies grounding, coverage and honesty of the enforcement claims, then registers it.

Enforcement classes (what the runtime can honestly do with an element):
  advise    injected into agent prompts (never enforced)
  evaluate  measured or reported by evaluation (never blocks)
  verify    checked by an existing runtime mechanism (see MECHANISMS)
  gate      blocks by an existing runtime mechanism (see MECHANISMS)
`verify`/`gate` are accepted only when they name an existing mechanism; everything else must be advise/evaluate.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Callable

from .state import now

SCHEMA_VERSION = 1
ELEMENT_TYPES = ("principle", "rule", "metric", "failure_condition", "disclosure_level", "evolution_rule", "definition")
MODALITIES = ("obligation", "prohibition", "permission", "preference")
ENFORCEMENTS = ("advise", "evaluate", "verify", "gate")
MECHANISMS = {                      # runtime mechanisms that really exist, and the params each needs
    "tool_allowlist": ("tools",),
    "approval_before_step": ("step",),
    "artifact_verification": (),
    "fail_closed": (),
}
MEASURES = ("tool_calls", "latency_ms", "gate_blocks", "retries", "repairs", "human_interventions",
            "completion", "steps_passed", "steps_total",
            "first_pass_rate", "recovery_rate", "verified_evidence_ratio", "web_evidence_ratio",
            "confirmed_bottleneck_ratio",
            "claim_support_ratio", "judge_agreement",              # from a run's trace/artifacts; these two need a judged run
            "principle_violations", "principle_shadow_violations", # enforced-check violations / shadow+warn violations
            "intent_relevance_ratio", "bottleneck_relevance_ratio",
            "tokens_total", "cost_usd", "confidence")      # need a relevance-judged run
LEVELS = ("default", "useful", "debug")
ID_PATTERN = re.compile(r"[a-z][a-z0-9_]{2,60}")
ELEMENT_ID = re.compile(r"[a-z][a-z0-9_\-]{0,60}")


class PolicyDocInvalid(ValueError):
    pass


def squash(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---- source structure -----------------------------------------------------------------------------

def sections(source: str) -> dict[str, str]:
    """Split by numbered headings (`N. Title` on its own line). Text before the first heading is `preamble`;
    text after the last heading (e.g. closing remarks) belongs to that last section."""
    found = [(m.start(), m.group(1)) for m in re.finditer(r"(?m)^[ \t]*(\d{1,2})\.[ \t]+\S[^\n]*$", source)]
    result: dict[str, str] = {"preamble": source[: found[0][0]] if found else source}
    for index, (start, number) in enumerate(found):
        end = found[index + 1][0] if index + 1 < len(found) else len(source)
        result[number] = source[start:end]
    return result


# ---- validators (the compiler's checks; never the model's claims) ------------------------------------

def validate_segments(data: Any, source: str) -> None:
    if not isinstance(data, dict):
        raise PolicyDocInvalid("SEGMENT output must be an object")
    title, doc_id = data.get("title"), data.get("id")
    if not isinstance(title, str) or not title.strip() or len(title) > 120:
        raise PolicyDocInvalid("title must be a short non-empty string")
    if not isinstance(doc_id, str) or not ID_PATTERN.fullmatch(doc_id):
        raise PolicyDocInvalid("id must be lowercase snake_case, 3-61 chars, starting with a letter")
    statements = data.get("statements")
    if not isinstance(statements, list) or not 3 <= len(statements) <= 400:
        raise PolicyDocInvalid("statements must be an array of 3-400 items")
    spans = {key: squash(text) for key, text in sections(source).items()}
    seen, covered = set(), set()
    for item in statements:
        if not isinstance(item, dict) or not all(isinstance(item.get(k), str) and item[k].strip() for k in ("id", "section", "quote")):
            raise PolicyDocInvalid("each statement needs non-empty id, section, quote")
        if not ELEMENT_ID.fullmatch(item["id"]) or item["id"] in seen:
            raise PolicyDocInvalid(f"statement id invalid or duplicated: {item['id']}")
        seen.add(item["id"])
        quote = squash(item["quote"])
        if len(quote) < 8 or len(quote) > 700:
            raise PolicyDocInvalid(f"{item['id']}: quote must be 8-700 characters")
        section = item["section"].strip()
        if section not in spans:
            raise PolicyDocInvalid(f"{item['id']}: unknown section '{section}'; use one of {sorted(spans, key=lambda k: (k != 'preamble', int(k) if k.isdigit() else 0))}")
        if quote not in spans[section]:
            where = [k for k, text in spans.items() if quote in text]
            hint = f"; it appears in section {where[0]}" if where else "; it does not appear anywhere in the source (quote must be verbatim)"
            raise PolicyDocInvalid(f"{item['id']}: quote is not verbatim text of section '{section}'{hint}")
        covered.add(section)
    missing = [k for k, text in spans.items() if k != "preamble" and len(text.strip()) >= 30 and k not in covered]
    if missing:
        raise PolicyDocInvalid(f"sections without any statement: {sorted(missing, key=int)}; every numbered section must be covered")


def validate_structure(data: Any, statements: list[dict[str, Any]]) -> None:
    if not isinstance(data, dict):
        raise PolicyDocInvalid("STRUCTURE output must be an object")
    known = {s["id"] for s in statements}
    elements, ignored = data.get("elements"), data.get("ignored", [])
    if not isinstance(elements, list) or not elements or len(elements) > 300:
        raise PolicyDocInvalid("elements must be a non-empty array (max 300)")
    if not isinstance(ignored, list):
        raise PolicyDocInvalid("ignored must be an array")
    ids, cited, metrics = set(), set(), {}
    for element in elements:
        if not isinstance(element, dict):
            raise PolicyDocInvalid("each element must be an object")
        eid, etype = element.get("id"), element.get("type")
        if not isinstance(eid, str) or not ELEMENT_ID.fullmatch(eid) or eid in ids or eid in known:
            raise PolicyDocInvalid(f"element id invalid, duplicated, or clashes with a statement id: {eid!r}")
        ids.add(eid)
        if "check" in element:
            raise PolicyDocInvalid(f"{eid}: runtime checks are attached by a named human after registration, never by the document")
        if etype not in ELEMENT_TYPES:
            raise PolicyDocInvalid(f"{eid}: type must be one of {list(ELEMENT_TYPES)}")
        cites = element.get("cites")
        if not isinstance(cites, list) or not cites or not all(isinstance(c, str) and c in known for c in cites):
            raise PolicyDocInvalid(f"{eid}: cites must be a non-empty array of existing statement ids")
        cited.update(cites)
        if etype in ("principle", "rule"):
            _text(element, "statement", eid, 400)
            if element.get("modality") not in MODALITIES:
                raise PolicyDocInvalid(f"{eid}: modality must be one of {list(MODALITIES)}")
            _enforcement(element, eid)
            if etype == "rule":
                _text(element, "when", eid, 300)
                _text(element, "then", eid, 300)
        elif etype == "metric":
            _text(element, "name", eid, 120)
            if element.get("direction") not in ("up", "down"):
                raise PolicyDocInvalid(f"{eid}: direction must be 'up' or 'down'")
            if element.get("measure") is not None and element["measure"] not in MEASURES:
                raise PolicyDocInvalid(f"{eid}: measure must be null or one of {list(MEASURES)} (null = not automatically measurable)")
            metrics[eid] = element
        elif etype == "failure_condition":
            _text(element, "diagnosis", eid, 300)
            when = element.get("when")
            if not isinstance(when, list) or not when or not all(isinstance(w, dict) and w.get("trend") in ("up", "down") and isinstance(w.get("metric"), str) for w in when):
                raise PolicyDocInvalid(f"{eid}: when must be a non-empty array of {{metric, trend: up|down}}")
            remedies = element.get("remedies", [])
            if not isinstance(remedies, list) or not all(isinstance(r, str) and r.strip() for r in remedies):
                raise PolicyDocInvalid(f"{eid}: remedies must be an array of strings")
        elif etype == "disclosure_level":
            if element.get("level") not in LEVELS:
                raise PolicyDocInvalid(f"{eid}: level must be one of {list(LEVELS)}")
            _text(element, "description", eid, 300)
        elif etype == "evolution_rule":
            _text(element, "trigger", eid, 300)
            _text(element, "action", eid, 300)
        elif etype == "definition":
            _text(element, "term", eid, 120)
            _text(element, "meaning", eid, 400)
    for element in elements:                                    # cross references (after all metrics are known)
        if element["type"] == "failure_condition":
            for w in element["when"]:
                if w["metric"] not in metrics:
                    raise PolicyDocInvalid(f"{element['id']}: unknown metric '{w['metric']}' (metrics: {sorted(metrics)})")
    ignored_ids = set()
    for item in ignored:
        if not isinstance(item, dict) or item.get("statement") not in known or not isinstance(item.get("reason"), str) or not item["reason"].strip():
            raise PolicyDocInvalid("each ignored entry needs an existing statement id and a reason")
        ignored_ids.add(item["statement"])
    if ignored_ids & cited:
        raise PolicyDocInvalid(f"statements both cited and ignored: {sorted(ignored_ids & cited)}")
    unaccounted = sorted(known - cited - ignored_ids)
    if unaccounted:
        raise PolicyDocInvalid(f"statements neither cited by an element nor ignored with a reason: {unaccounted}")
    if len(ignored_ids) > max(2, len(known) // 4):
        raise PolicyDocInvalid(f"too many ignored statements ({len(ignored_ids)} of {len(known)}); structure them instead")


def _text(element: dict[str, Any], key: str, eid: str, limit: int) -> None:
    value = element.get(key)
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise PolicyDocInvalid(f"{eid}: {key} must be a non-empty string up to {limit} characters")


def _enforcement(element: dict[str, Any], eid: str) -> None:
    mode = element.get("enforcement")
    if mode not in ENFORCEMENTS:
        raise PolicyDocInvalid(f"{eid}: enforcement must be one of {list(ENFORCEMENTS)}")
    if mode in ("verify", "gate"):
        mechanism = element.get("mechanism")
        if mechanism not in MECHANISMS:
            raise PolicyDocInvalid(f"{eid}: '{mode}' is only allowed with an existing mechanism {sorted(MECHANISMS)}; "
                                   "use 'evaluate' or 'advise' for anything the runtime cannot really enforce")
        params = element.get("params") or {}
        if not isinstance(params, dict) or any(k not in params for k in MECHANISMS[mechanism]):
            raise PolicyDocInvalid(f"{eid}: mechanism {mechanism} needs params {list(MECHANISMS[mechanism])}")
    elif element.get("mechanism") is not None:
        raise PolicyDocInvalid(f"{eid}: mechanism is only for verify/gate elements")


# ---- ingestion -------------------------------------------------------------------------------------

SEGMENT = {
    "id": "SEGMENT",
    "description": "Split the source policy document into atomic, verbatim-quoted statements.",
    "instruction": ("Split the policy document (given as `intent`) into atomic statements that each carry one meaning: a "
                    "principle, rule, definition, metric, condition, level, or example. `quote` must be copied VERBATIM from the "
                    "source (exact characters; you may drop line breaks) and lie inside the section named by `section`. "
                    "`section` is the heading number as a string (e.g. \"3\"), or \"preamble\" for text before the first "
                    "numbered heading; unnumbered closing text belongs to the last numbered section. Cover every numbered "
                    "section. Also give a short `title` and a snake_case `id` for the whole document. Do not invent text."),
    "contract": {"title": "short document title", "id": "snake_case_id",
                 "statements": [{"id": "s1", "section": "\"0\"..\"16\" or \"preamble\"", "quote": "verbatim excerpt"}]},
}

STRUCTURE = {
    "id": "STRUCTURE",
    "description": "Classify the statements into structured policy elements.",
    "instruction": ("Turn the statements (input SEGMENT) into structured elements. Element types: principle/rule "
                    "(modality obligation|prohibition|permission|preference; `statement` = one-sentence normalization; rule also "
                    "has `when` and `then`), metric (name, direction up|down, `measure` null or one of "
                    "tool_calls|latency_ms|gate_blocks|retries|repairs|human_interventions|completion|steps_passed|steps_total|"
                    "first_pass_rate|recovery_rate|verified_evidence_ratio|web_evidence_ratio|confirmed_bottleneck_ratio|"
                    "claim_support_ratio|judge_agreement|principle_violations|principle_shadow_violations|intent_relevance_ratio|bottleneck_relevance_ratio|tokens_total|cost_usd|confidence), "
                    "failure_condition (when=[{metric,trend}] over metric ids you defined, diagnosis, remedies[]), "
                    "disclosure_level (level default|useful|debug, description), evolution_rule (trigger, action), definition "
                    "(term, meaning). Every element needs a unique `id` and `cites` (existing statement ids). Every statement must be "
                    "cited by some element or listed in `ignored` with a reason (at most 25%). ENFORCEMENT HONESTY: for principle/rule "
                    "choose enforcement advise (guidance in prompts) or evaluate (measured/reported). Use verify/gate ONLY if the "
                    "rule exactly matches a runtime mechanism (tool_allowlist{tools}, approval_before_step{step}, "
                    "artifact_verification, fail_closed) and name it in `mechanism` with `params`; abstract principles such as "
                    "'preserve intent' can only be advise or evaluate. Set metric.measure null when it cannot be computed "
                    "automatically. Do not add content that is not in the statements."),
    "contract": {"elements": [
        {"id": "p1", "type": "principle", "modality": "obligation", "statement": "...", "enforcement": "advise", "cites": ["s1"]},
        {"id": "r1", "type": "rule", "modality": "prohibition", "statement": "...", "when": "...", "then": "...", "enforcement": "evaluate", "cites": ["s2"]},
        {"id": "m1", "type": "metric", "name": "Human intervention rate", "direction": "down", "measure": "human_interventions", "cites": ["s3"]},
        {"id": "f1", "type": "failure_condition", "when": [{"metric": "m1", "trend": "up"}], "diagnosis": "...", "remedies": ["..."], "cites": ["s4"]},
        {"id": "d1", "type": "disclosure_level", "level": "default", "description": "...", "cites": ["s5"]},
        {"id": "e1", "type": "evolution_rule", "trigger": "...", "action": "...", "cites": ["s6"]},
        {"id": "t1", "type": "definition", "term": "...", "meaning": "...", "cites": ["s7"]}],
        "ignored": [{"statement": "s8", "reason": "purely decorative diagram"}]},
}


def _ask(agent: Any, spec: dict[str, Any], source: str, inputs: dict[str, Any], check: Callable[[Any], None],
         attempts: int, progress: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    feedback = None
    for attempt in range(1, attempts + 1):
        progress({"phase": spec["id"], "attempt": attempt, "status": "asking Claude" if attempt == 1 else "retrying with error feedback"})
        payload = dict(inputs)
        if feedback:
            payload["_feedback"] = {"error": feedback, "attempt": attempt}
        proposal = agent.propose({**spec, "requires": list(inputs)}, source, payload)
        data = proposal.get("data", {k: v for k, v in proposal.items() if k not in ("step", "tool_log")}) if isinstance(proposal, dict) else proposal
        try:
            check(data)
            progress({"phase": spec["id"], "attempt": attempt, "status": "verified"})
            return data
        except PolicyDocInvalid as exc:
            feedback = str(exc)
            progress({"phase": spec["id"], "attempt": attempt, "status": "rejected", "error": feedback[:300]})
    raise PolicyDocInvalid(f"{spec['id']} failed verification after {attempts} attempts: {feedback}")


def ingest(source: str, agent: Any, title: str | None = None, attempts: int = 3,
           progress: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    """Structure `source` with the agent; return an (unregistered) verified document. Raises PolicyDocInvalid."""
    progress = progress or (lambda _event: None)
    source = source.replace("\r\n", "\n").strip()
    if not 200 <= len(source) <= 60000:
        raise PolicyDocInvalid("source must be between 200 and 60,000 characters")
    segmented = _ask(agent, SEGMENT, source, {}, lambda d: validate_segments(d, source), attempts, progress)
    statements = segmented["statements"]
    structured = _ask(agent, STRUCTURE, source, {"SEGMENT": {"statements": statements}},
                      lambda d: validate_structure(d, statements), attempts, progress)
    document = {
        "kind": "principle_policy", "schema_version": SCHEMA_VERSION,
        "id": segmented["id"], "title": (title or segmented["title"]).strip(), "version": "1.0",
        "created_at": now(), "source": {"sha256": sha256(source), "chars": len(source)},
        "statements": statements, "elements": structured["elements"], "ignored": structured.get("ignored", []),
    }
    document["report"] = coverage_report(document, source)
    return document


def coverage_report(document: dict[str, Any], source: str) -> dict[str, Any]:
    numbered = [k for k in sections(source) if k != "preamble"]
    used = {s["section"] for s in document["statements"]}
    by_type: dict[str, int] = {}
    by_enforcement: dict[str, int] = {}
    for element in document["elements"]:
        by_type[element["type"]] = by_type.get(element["type"], 0) + 1
        if "enforcement" in element:
            by_enforcement[element["enforcement"]] = by_enforcement.get(element["enforcement"], 0) + 1
    return {"sections": len(numbered), "sections_covered": len([k for k in numbered if k in used]),
            "statements": len(document["statements"]), "elements": len(document["elements"]),
            "ignored": len(document["ignored"]), "by_type": by_type, "by_enforcement": by_enforcement,
            "metrics_measurable": sum(1 for e in document["elements"] if e["type"] == "metric" and e.get("measure")),
            "metrics_unmeasurable": [e["id"] for e in document["elements"] if e["type"] == "metric" and not e.get("measure")]}


# ---- registry --------------------------------------------------------------------------------------

def _bump(version: str) -> str:
    head, _, tail = version.rpartition(".")
    return f"{head}.{int(tail) + 1}" if head and tail.isdigit() else f"{version}.1"


def _versions(directory: Path, doc_id: str) -> list[str]:
    found = [re.fullmatch(re.escape(doc_id) + r"\.v(.+)\.json", p.name) for p in Path(directory).glob(f"{doc_id}.v*.json")]
    return sorted((m.group(1) for m in found if m), key=lambda v: [int(x) if x.isdigit() else 0 for x in v.split(".")])


def register(document: dict[str, Any], source: str, directory: Path) -> dict[str, Any]:
    """Write a NEW versioned file (never overwrites) plus the source text; returns the registered document."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    existing = _versions(directory, document["id"])
    document = {**document, "version": _bump(existing[-1]) if existing else document["version"]}
    if document["source"]["sha256"] != sha256(source.replace("\r\n", "\n").strip()):
        raise PolicyDocInvalid("document does not match the given source text")
    target = directory / f"{document['id']}.v{document['version']}.json"
    target.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (directory / f"{document['id']}.v{document['version']}.source.txt").write_text(source, encoding="utf-8")
    with (directory / "_ingest_log.jsonl").open("a", encoding="utf-8") as log:
        log.write(json.dumps({"at": now(), "id": document["id"], "version": document["version"],
                              "source_sha256": document["source"]["sha256"], "report": document["report"]}, ensure_ascii=False) + "\n")
    return document


def list_documents(directory: Path) -> list[dict[str, Any]]:
    out = []
    for path in sorted(Path(directory).glob("*.v*.json")) if Path(directory).exists() else []:
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if doc.get("kind") == "principle_policy":
            out.append({"id": doc["id"], "version": doc["version"], "title": doc["title"], "created_at": doc.get("created_at"),
                        "report": doc.get("report", {})})
    return out


def load_document(directory: Path, doc_id: str, version: str | None = None) -> dict[str, Any]:
    if not ID_PATTERN.fullmatch(doc_id or ""):
        raise KeyError("invalid principle policy id")
    versions = _versions(directory, doc_id)
    if not versions:
        raise KeyError(f"principle policy not found: {doc_id}")
    chosen = version or versions[-1]
    if chosen not in versions:
        raise KeyError(f"version not found: {doc_id} {chosen}")
    return json.loads((Path(directory) / f"{doc_id}.v{chosen}.json").read_text(encoding="utf-8"))


def verify_document(document: dict[str, Any], source: str) -> None:
    """Re-run every check on a stored document (detects edits to the JSON or the source after registration)."""
    if document.get("source", {}).get("sha256") != sha256(source.replace("\r\n", "\n").strip()):
        raise PolicyDocInvalid("source text changed since registration")
    validate_segments({"title": document["title"], "id": document["id"], "statements": document["statements"]}, source.replace("\r\n", "\n").strip())
    validate_structure({"elements": document["elements"], "ignored": document["ignored"]}, document["statements"])


# ---- human-approved proxy measures -----------------------------------------------------------------------

def _proxy_path(directory: Path, doc_id: str) -> Path:
    return Path(directory) / f"{doc_id}.proxies.json"


def load_proxies(directory: Path, doc_id: str) -> dict[str, dict[str, Any]]:
    """Latest approved proxy per metric id: {metric: {measure, approved_by, at, note}}."""
    path = _proxy_path(directory, doc_id)
    if not path.exists():
        return {}
    latest: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        if entry.get("measure") is None:
            latest.pop(entry["metric"], None)               # a revocation
        else:
            latest[entry["metric"]] = entry
    return latest


def set_proxy(directory: Path, doc_id: str, metric: str, measure: str | None, approved_by: str, note: str = "") -> dict[str, Any]:
    """A named human declares that an unmeasurable metric is to be evaluated through a computable proxy measure.

    Append-only log (revocation = measure null). The original document is never edited: proxies are overlaid at
    evaluation time and always labelled as proxies with their approver."""
    from .gate import SELF_APPROVERS
    if not isinstance(approved_by, str) or approved_by.strip().lower() in SELF_APPROVERS:
        raise PolicyDocInvalid("a named human approver is required; agents cannot approve proxy measures")
    document = load_document(directory, doc_id)
    element = next((e for e in document["elements"] if e["type"] == "metric" and e["id"] == metric), None)
    if element is None:
        raise PolicyDocInvalid(f"unknown metric: {metric}")
    if element.get("measure"):
        raise PolicyDocInvalid(f"metric {metric} already has its own measure '{element['measure']}'; proxies are only for unmeasurable metrics")
    if measure is not None and measure not in MEASURES:
        raise PolicyDocInvalid(f"measure must be one of {list(MEASURES)} or null to revoke")
    entry = {"metric": metric, "measure": measure, "approved_by": approved_by.strip(), "note": str(note)[:300], "at": now()}
    with _proxy_path(directory, doc_id).open("a", encoding="utf-8") as log:
        log.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def apply_proxies(document: dict[str, Any], proxies: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Copy of `document` whose unmeasurable metrics carry the approved proxy (marked `proxy`)."""
    out = json.loads(json.dumps(document))
    for element in out["elements"]:
        if element["type"] == "metric" and not element.get("measure") and element["id"] in proxies:
            p = proxies[element["id"]]
            element["measure"] = p["measure"]
            element["proxy"] = {"by": p["approved_by"], "at": p["at"], "note": p.get("note", "")}
    return out


def load_effective(directory: Path, doc_id: str, version: str | None = None) -> dict[str, Any]:
    """Registered document with the currently approved proxies and human-attached checks applied."""
    from . import principle_checks
    document = apply_proxies(load_document(directory, doc_id, version), load_proxies(directory, doc_id))
    return principle_checks.apply_checks(document, principle_checks.load_checks(directory, doc_id))


# ---- using a registered policy ------------------------------------------------------------------------

def advise_lines(documents: list[dict[str, Any]], limit: int = 12, chars: int = 1500) -> list[str]:
    """Advisory principles for agent prompts (never enforced; the compiler enforces separately)."""
    lines: list[str] = []
    for doc in documents:
        for element in doc["elements"]:
            if element["type"] in ("principle", "rule") and element.get("enforcement") == "advise":
                lines.append(f"[{element['modality']}] {element['statement']}"
                             + (f" (when {element['when']}: {element['then']})" if element["type"] == "rule" else ""))
    out, total = [], 0
    for line in lines[:limit]:
        if total + len(line) > chars:
            break
        out.append(line)
        total += len(line)
    return out

