"""Validators for the product_idea_value_review policy (M2778): the owner's "Value Layer 2.0" turned into compiler-checked steps.

The model writes claims and cites evidence; this module recomputes what can be derived (evidence grade per question, grade ceiling on confidence,
net customer value, opportunity potential, the verdict) and refuses what cannot be grounded. Nothing here proves a product will sell: a verdict of
"keep" only means the recorded evidence clears the thresholds below. The thresholds (confidence ceiling per grade, impact 4+, grade E2 floor,
two high-impact drops) are policy parameters chosen for this review, not research findings; the grade ladder E0-E4 is the owner's own internal
scheme, not a published standard."""
from __future__ import annotations

import re
from typing import Any

from .marketing_validators import _rows
from .validator_common import squash as _squash

AREAS = "ABCDEFGH"
QUESTION_IDS = [f"{a}{n}" for a in AREAS for n in (1, 2, 3, 4)]       # 8 areas x 4 = 32: the source text says 24 but its tables list 32
KIND_GRADE = {"assumption": 0, "statement": 1, "behavior": 2, "payment": 3, "repeat_outcome": 4}
GRADE_CAP = (2, 3, 4, 5, 5)             # highest confidence (1-5) a question may claim at grade E0..E4
DECISIONS = ("keep", "investigate", "drop")
SOURCES = ("cost_cut", "time_saved")
REALIZATIONS = ("payroll_reduced", "redeployed_with_revenue")
HIGH_IMPACT = 4
EVIDENCE_FLOOR = 2                      # E2 (observed behavior) is the lowest grade a keep verdict may rest on
DROP_LIMIT = 2


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_num(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value == value


def _text(value: Any, minimum: int) -> bool:
    return isinstance(value, str) and len(value.strip()) >= minimum


def _evidence_rows(ctx: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    """The run's EVIDENCE facts block as {id: row}; kinds come from the closed list, so a grade is never the model's word."""
    rows = _rows((ctx or {}).get("intent", ""), "EVIDENCE", "id", ("kind", "text"))
    for rid, row in rows.items():
        if row["kind"] not in KIND_GRADE:
            raise ValueError(f"evidence {rid}: kind must be one of {list(KIND_GRADE)}, got {row['kind']!r}")
        if not _text(row["text"], 12):
            raise ValueError(f"evidence {rid}: text must be at least 12 characters")
    return rows


def _grade_of(cited: Any, rows: dict[str, dict[str, Any]], where: str, problems: list[str]) -> int:
    """Highest grade among the cited evidence rows (0 when nothing is cited); every cited quote must sit word for word in that row's text."""
    if not isinstance(cited, list):
        problems.append(f"{where}: evidence must be a list (empty means an assumption, grade E0)")
        return 0
    best = 0
    for item in cited:
        if not isinstance(item, dict) or set(item) != {"id", "quote"}:
            problems.append(f"{where}: each evidence entry is exactly {{id, quote}}, got {str(item)[:60]!r}")
            continue
        row = rows.get(item["id"])
        if row is None:
            problems.append(f"{where}: evidence id {item['id']!r} is not in the EVIDENCE block {sorted(rows)}")
            continue
        if not _text(item["quote"], 12) or _squash(item["quote"]) not in _squash(row["text"]):
            problems.append(f"{where}: the quote for {item['id']} is not word for word in that evidence text")
            continue
        best = max(best, KIND_GRADE[row["kind"]])
    return best


def _fail(problems: list[str]) -> None:
    if problems:
        raise ValueError(f"{len(problems)} problem(s): " + "; ".join(problems[:10]))


def intake(data, inputs, policy, ctx=None):
    """The idea is restated only with words that sit in the idea statement, and the EVIDENCE block must exist and be well formed."""
    idea = _rows((ctx or {}).get("intent", ""), "IDEA", "id", ("statement",))
    if len(idea) != 1:
        raise ValueError("the IDEA facts block must hold exactly one idea row")
    statement = next(iter(idea.values()))["statement"]
    _evidence_rows(ctx)
    problems = []
    for key in ("product", "customer", "situation"):
        part = data.get(key)
        if not isinstance(part, dict) or set(part) != {"text", "quote"}:
            problems.append(f"{key} must be exactly {{text, quote}}")
            continue
        if not _text(part["text"], 8):
            problems.append(f"{key}.text must be a phrase of at least 8 characters")
        if not _text(part["quote"], 8) or _squash(part["quote"]) not in _squash(statement):
            problems.append(f"{key}.quote must be copied word for word from the idea statement")
    _fail(problems)


def questions(data, inputs, policy, ctx=None):
    """every listed question (A1..H4, 32 in the source tables although its text says 24) answered once. Per question the compiler derives the evidence grade (E0 when nothing is cited), caps confidence by that grade,
    refuses `keep` on a high-impact claim that rests on E0 and writes the summary lists the verdict step reads. Idempotent: derived fields are rewritten."""
    rows = _evidence_rows(ctx)
    items = data.get("questions")
    if not isinstance(items, list):
        raise ValueError("questions must be an array of the 32 question answers A1..H4")
    problems, seen = [], []
    for q in items:
        if not isinstance(q, dict) or q.get("id") not in QUESTION_IDS:
            problems.append(f"each answer needs an id from A1..H4, got {str(q)[:60]!r}")
            continue
        qid = q["id"]
        seen.append(qid)
        grade = _grade_of(q.get("evidence"), rows, qid, problems)
        q["grade"], q["grade_n"] = f"E{grade}", grade
        if not _text(q.get("claim"), 12):
            problems.append(f"{qid}: claim must be a sentence of at least 12 characters")
        counter = q.get("counterevidence")
        if counter != "none_found" and not _text(counter, 12):
            problems.append(f"{qid}: counterevidence must be a sentence of 12+ characters or exactly none_found")
        for key in ("confidence", "impact"):
            if not _is_int(q.get(key)) or not 1 <= q[key] <= 5:
                problems.append(f"{qid}: {key} must be an integer 1-5")
        if _is_int(q.get("confidence")) and q["confidence"] > GRADE_CAP[grade]:
            problems.append(f"{qid}: confidence {q['confidence']} is above the ceiling {GRADE_CAP[grade]} for evidence grade E{grade}")
        if q.get("decision") not in DECISIONS:
            problems.append(f"{qid}: decision must be one of {list(DECISIONS)}")
        elif q["decision"] == "keep" and _is_int(q.get("impact")) and (grade == 0 and q["impact"] >= HIGH_IMPACT):
            problems.append(f"{qid}: a claim with impact {q['impact']} resting on E0 (an assumption) cannot be marked keep")
        if grade < EVIDENCE_FLOOR and _is_int(q.get("impact")) and q["impact"] >= HIGH_IMPACT and not _text(q.get("experiment"), 20):
            problems.append(f"{qid}: a high-impact claim below E{EVIDENCE_FLOOR} needs an experiment sentence of 20+ characters that would settle it")
    if len(seen) != len(set(seen)) or set(seen) != set(QUESTION_IDS):
        problems.append(f"every question A1..H4 exactly once; missing {sorted(set(QUESTION_IDS) - set(seen))}, repeated {sorted({s for s in seen if seen.count(s) > 1})}")
    advanced = data.get("advanced")
    if not isinstance(advanced, dict) or set(advanced) != set(AREAS):
        problems.append("advanced must hold exactly one answer per area A..H")
    else:
        for area, text in advanced.items():
            if not _text(text, 20):
                problems.append(f"advanced {area}: an answer of at least 20 characters is required")
    _fail(problems)
    by_id = {q["id"]: q for q in items}
    data["by_grade"] = {f"E{g}": sum(1 for q in items if q["grade_n"] == g) for g in range(5)}
    data["weak_high_impact"] = sorted(q["id"] for q in items if q["grade_n"] < EVIDENCE_FLOOR and q["impact"] >= HIGH_IMPACT)
    data["high_impact_drops"] = sorted(q["id"] for q in items if q["decision"] == "drop" and q["impact"] >= HIGH_IMPACT)
    data["answered"] = len(by_id)


def economics(data, inputs, policy, ctx=None):
    """S, E, P, C, R, N and F as numbers with their evidence; the compiler computes V_net = S + E + P - C - R and opportunity = N x F x V_net.
    P is never given: the model gives the expected loss without and with the product and the compiler subtracts. Time saved counts only as S when it
    was realized as lower payroll or as redeployed hours that earned revenue, and only with evidence. Amounts are per value event in one currency;
    F is the number of value events per year (a monthly value that recurs twelve times a year has F = 12)."""
    rows = _evidence_rows(ctx)
    econ = data.get("economics")
    if not isinstance(econ, dict):
        raise ValueError("economics must be an object with S, E, P, C, R, N and F")
    need = {"S", "E", "P", "C", "R", "N", "F"}
    problems = []
    if set(econ) != need:
        problems.append(f"economics must hold exactly {sorted(need)}, got {sorted(econ)}")
    unit = data.get("unit")
    if not _text(unit, 2):
        problems.append("unit must name the currency and event, for example KRW per month")
    grades: list[int] = []
    for key in ("S", "E", "C", "R"):
        item = econ.get(key)
        if not isinstance(item, dict) or not _is_num(item.get("value")) or item["value"] < 0:
            problems.append(f"{key}: value must be a number >= 0")
            continue
        g = _grade_of(item.get("evidence"), rows, key, problems)
        item["grade"] = f"E{g}"
        if key in ("S", "E") and item["value"] > 0:
            grades.append(g)
    s = econ.get("S")
    if isinstance(s, dict) and _is_num(s.get("value")) and s["value"] > 0:
        if s.get("source") not in SOURCES:
            problems.append(f"S.source must be one of {list(SOURCES)}")
        elif s["source"] == "time_saved" and (s.get("realization") not in REALIZATIONS or s.get("grade") == "E0"):
            problems.append(f"S: saved time counts only with realization in {list(REALIZATIONS)} and evidence above an assumption")
    p = econ.get("P")
    if not isinstance(p, dict) or not _is_num(p.get("loss_without")) or not _is_num(p.get("loss_with")) or p["loss_without"] < 0 or p["loss_with"] < 0:
        problems.append("P must give loss_without and loss_with (expected loss per event without and with the product, numbers >= 0), not a total")
    else:
        if p["loss_with"] > p["loss_without"]:
            problems.append("P: loss_with is above loss_without, so nothing is avoided; use R for added risk")
        p.pop("value", None)
        p["value"] = p["loss_without"] - p["loss_with"]                          # derived, never taken from the model
        g = _grade_of(p.get("evidence"), rows, "P", problems)
        p["grade"] = f"E{g}"
        if p["value"] > 0:
            grades.append(g)
    n, f = econ.get("N"), econ.get("F")
    if not isinstance(n, dict) or not _is_int(n.get("value")) or n["value"] < 1:
        problems.append("N.value must be an integer >= 1 (eligible customers you can reach)")
    else:
        n["grade"] = f"E{_grade_of(n.get('evidence'), rows, 'N', problems)}"
    if not isinstance(f, dict) or not _is_num(f.get("value")) or f["value"] <= 0:
        problems.append("F.value must be a number > 0 (value events per year)")
    else:
        f["grade"] = f"E{_grade_of(f.get('evidence'), rows, 'F', problems)}"
    _fail(problems)
    v = econ["S"]["value"] + econ["E"]["value"] + econ["P"]["value"] - econ["C"]["value"] - econ["R"]["value"]
    data["V_net"] = round(v, 6)
    data["opportunity_per_year"] = round(econ["N"]["value"] * econ["F"]["value"] * v, 6)
    data["econ_grade_n"] = min(grades) if grades else 0
    data["econ_grade"] = f"E{data['econ_grade_n']}"
    data["note"] = "opportunity is a rough yearly total of net customer value, not a revenue forecast or a success probability"


_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _numbers(text: str) -> set[float]:
    out = set()
    for m in _NUM.findall(str(text)):
        try:
            out.add(float(m.replace(",", "")))
        except ValueError:
            pass
    return out


def _supported_grade(item: dict[str, Any], amount: float, rows: dict[str, dict[str, Any]], where: str, problems: list[str]) -> tuple[int, bool]:
    """v1.1 (M2778): grade of an amount. A positive amount counts above E0 only when at least one cited quote holds that very number; otherwise it is an
    assumption (E0, supported False). A zero with no evidence is unknown, not zero: E0 too. Returns (grade, supported)."""
    grade = _grade_of(item.get("evidence"), rows, where, problems)
    if not isinstance(item.get("evidence"), list):
        return 0, False
    if amount > 0:
        quoted = any(amount in _numbers(e.get("quote", "")) for e in item["evidence"] if isinstance(e, dict))
        if not quoted:
            return 0, False
    return grade, grade > 0


def economics_v2(data, inputs, policy, ctx=None):
    """economics (v1.0 stays unchanged: stored runs are re-verified with it) plus two rules found in the first real run (M2778): (1) a positive amount is
    graded above E0 only if a cited quote contains that number, else it is an assumption; (2) S, E and P all enter the weakest-evidence grade, so a zero
    without evidence counts as E0 (unknown) instead of dropping out. Everything else, including V_net and the opportunity, is computed as before."""
    rows = _evidence_rows(ctx)
    econ = data.get("economics")
    if not isinstance(econ, dict):
        raise ValueError("economics must be an object with S, E, P, C, R, N and F")
    problems = []
    if set(econ) != {"S", "E", "P", "C", "R", "N", "F"}:
        problems.append(f"economics must hold exactly ['C', 'E', 'F', 'N', 'P', 'R', 'S'], got {sorted(econ)}")
    if not _text(data.get("unit"), 2):
        problems.append("unit must name the currency and event, for example KRW per month")
    grades: dict[str, int] = {}
    for key in ("S", "E", "C", "R"):
        item = econ.get(key)
        if not isinstance(item, dict) or not _is_num(item.get("value")) or item["value"] < 0:
            problems.append(f"{key}: value must be a number >= 0")
            continue
        g, ok = _supported_grade(item, item["value"], rows, key, problems)
        item["grade"], item["supported"] = f"E{g}", ok
        if key in ("S", "E"):
            grades[key] = g
    s = econ.get("S")
    if isinstance(s, dict) and _is_num(s.get("value")) and s["value"] > 0:
        if s.get("source") not in SOURCES:
            problems.append(f"S.source must be one of {list(SOURCES)}")
        elif s["source"] == "time_saved" and (s.get("realization") not in REALIZATIONS or s.get("grade") == "E0"):
            problems.append(f"S: saved time counts only with realization in {list(REALIZATIONS)} and evidence above an assumption")
    p = econ.get("P")
    if not isinstance(p, dict) or not _is_num(p.get("loss_without")) or not _is_num(p.get("loss_with")) or p["loss_without"] < 0 or p["loss_with"] < 0:
        problems.append("P must give loss_without and loss_with (expected loss per event without and with the product, numbers >= 0), not a total")
    else:
        if p["loss_with"] > p["loss_without"]:
            problems.append("P: loss_with is above loss_without, so nothing is avoided; use R for added risk")
        p.pop("value", None)
        p["value"] = p["loss_without"] - p["loss_with"]
        quoted_ok = p["value"] == 0 or all(any(a in _numbers(e.get("quote", "")) for e in (p.get("evidence") or []) if isinstance(e, dict)) for a in (p["loss_without"], p["loss_with"]) if a > 0)
        g = _grade_of(p.get("evidence"), rows, "P", problems)
        g = g if (isinstance(p.get("evidence"), list) and quoted_ok) else 0
        p["grade"], p["supported"] = f"E{g}", g > 0
        grades["P"] = g
    n, f = econ.get("N"), econ.get("F")
    if not isinstance(n, dict) or not _is_int(n.get("value")) or n["value"] < 1:
        problems.append("N.value must be an integer >= 1 (eligible customers you can reach)")
    else:
        g, ok = _supported_grade(n, n["value"], rows, "N", problems)
        n["grade"], n["supported"] = f"E{g}", ok
    if not isinstance(f, dict) or not _is_num(f.get("value")) or f["value"] <= 0:
        problems.append("F.value must be a number > 0 (value events per year)")
    else:
        g, ok = _supported_grade(f, f["value"], rows, "F", problems)
        f["grade"], f["supported"] = f"E{g}", ok
    _fail(problems)
    v = econ["S"]["value"] + econ["E"]["value"] + econ["P"]["value"] - econ["C"]["value"] - econ["R"]["value"]
    data["V_net"] = round(v, 6)
    data["opportunity_per_year"] = round(econ["N"]["value"] * econ["F"]["value"] * v, 6)
    data["econ_grade_n"] = min(grades.values())
    data["econ_grade"] = f"E{data['econ_grade_n']}"
    data["unsupported_amounts"] = sorted(k for k in ("S", "E", "P", "C", "R", "N", "F") if (econ[k].get("value", 0) > 0 or k == "P" and econ[k].get("loss_without", 0) > 0) and not econ[k].get("supported"))
    data["note"] = "opportunity is a rough yearly total of net customer value, not a revenue forecast or a success probability; an amount without its number in a cited quote is an assumption (E0)"


def verdict(data, inputs, policy, ctx=None):
    """The verdict is computed from the recorded questions and economics: drop when V_net <= 0 or two high-impact claims are marked drop; keep when V_net > 0,
    the weakest of S/E/P evidence is at least E2 and no high-impact claim rests below E2; investigate otherwise. The model must state the same verdict and
    name an experiment for every high-impact claim below E2."""
    qd = inputs["QUESTIONS"]["data"]
    ed = inputs["ECONOMICS"]["data"]
    weak = list(qd["weak_high_impact"])
    if ed["V_net"] <= 0 or len(qd["high_impact_drops"]) >= DROP_LIMIT:
        computed = "drop"
    elif ed["econ_grade_n"] >= EVIDENCE_FLOOR and not weak:
        computed = "keep"
    else:
        computed = "investigate"
    if data.get("verdict") != computed:
        raise ValueError(f"verdict must be {computed!r} (computed from V_net {ed['V_net']}, evidence grade {ed['econ_grade']}, weak high-impact claims {weak}); got {data.get('verdict')!r}")
    if not _text(data.get("summary"), 40):
        raise ValueError("summary must be at least 40 characters")
    exps = data.get("experiments")
    if not isinstance(exps, list):
        raise ValueError("experiments must be an array")
    named = [e.get("question_id") for e in exps if isinstance(e, dict)]
    if computed != "drop" and (set(named) != set(weak) or len(named) != len(set(named))):
        raise ValueError(f"experiments must name each weak high-impact question exactly once {weak}, got {named}")
    for e in exps:
        if not isinstance(e, dict) or set(e) != {"question_id", "experiment"} or not _text(e.get("experiment"), 20):
            raise ValueError("each experiment is exactly {question_id, experiment} with a sentence of 20+ characters")
    data["verdict_basis"] = {"V_net": ed["V_net"], "econ_grade": ed["econ_grade"], "weak_high_impact": weak, "high_impact_drops": list(qd["high_impact_drops"])}


def verdict_v2(data, inputs, policy, ctx=None):
    """verdict (v1.0 stays unchanged: stored runs are re-verified with it) plus the rule the first two real runs exposed (M2778): no evidence is not no value.
    drop needs a reason that rests on stated facts: two claims with impact 4-5 marked drop, or V_net <= 0 with EVERY amount S, E, P, C and R supported by a cited
    quote that holds its number. With any unsupported amount the verdict is at best investigate, and the model must name each gap and how to close it.
    keep additionally needs every amount supported. The thresholds are policy parameters, not research findings."""
    qd = inputs["QUESTIONS"]["data"]
    ed = inputs["ECONOMICS"]["data"]
    weak = list(qd["weak_high_impact"])
    econ = ed.get("economics", {})
    gaps = [k for k in ("S", "E", "P", "C", "R") if not econ.get(k, {}).get("supported")]    # a zero without evidence is unknown, so it is a gap too
    if len(qd["high_impact_drops"]) >= DROP_LIMIT or (ed["V_net"] <= 0 and not gaps):
        computed = "drop"
    elif ed["V_net"] > 0 and ed["econ_grade_n"] >= EVIDENCE_FLOOR and not weak and not gaps:
        computed = "keep"
    else:
        computed = "investigate"
    if data.get("verdict") != computed:
        raise ValueError(f"verdict must be {computed!r} (computed from V_net {ed['V_net']}, evidence grade {ed['econ_grade']}, unsupported amounts {gaps}, weak high-impact claims {weak}); got {data.get('verdict')!r}")
    if not _text(data.get("summary"), 40):
        raise ValueError("summary must be at least 40 characters")
    problems = []
    exps = data.get("experiments")
    if not isinstance(exps, list):
        problems.append("experiments must be an array")
        exps = []
    named = [e.get("question_id") for e in exps if isinstance(e, dict)]
    if computed != "drop" and (set(named) != set(weak) or len(named) != len(set(named))):
        problems.append(f"experiments must name each weak high-impact question exactly once {weak}, got {named}")
    for e in exps:
        if not isinstance(e, dict) or set(e) != {"question_id", "experiment"} or not _text(e.get("experiment"), 20):
            problems.append("each experiment is exactly {question_id, experiment} with a sentence of 20+ characters")
            break
    found = data.get("evidence_gaps")
    if not isinstance(found, list):
        problems.append("evidence_gaps must be an array (empty when no amount is unsupported)")
        found = []
    amounts = [g.get("amount") for g in found if isinstance(g, dict)]
    if computed == "investigate" and (set(amounts) != set(gaps) or len(amounts) != len(set(amounts))):
        problems.append(f"evidence_gaps must name each unsupported amount exactly once {gaps}, got {amounts}")
    if computed != "investigate" and found:
        problems.append("evidence_gaps must be empty unless the verdict is investigate")
    for g in found:
        if not isinstance(g, dict) or set(g) != {"amount", "how"} or not _text(g.get("how"), 20):
            problems.append("each evidence gap is exactly {amount, how}: the number to obtain and how, in 20+ characters")
            break
    _fail(problems)
    data["verdict_basis"] = {"V_net": ed["V_net"], "econ_grade": ed["econ_grade"], "unsupported_amounts": gaps, "weak_high_impact": weak, "high_impact_drops": list(qd["high_impact_drops"])}


# Layers of the pasted text (section 6: Value Taxonomy, Discovery, Quantification, Capture, Evidence). The question-to-layer assignment is a design
# decision of this policy, not a research finding; layers marked pre_customer can be assessed from desk evidence (E1) before any customer contact.
LAYERS = (
    ("taxonomy", "Value Taxonomy", True, ("C1", "C2", "C3", "C4")),
    ("discovery", "Value Discovery", True, ("A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4", "E1", "E2", "E3", "E4", "H1", "H2", "H3", "H4")),
    ("quantification", "Value Quantification", False, ("D1", "D2", "D3", "D4")),
    ("capture", "Value Capture", False, ("F1", "F2", "F3", "F4", "G1", "G2", "G3", "G4")),
)
LEVERS = (("C1", "save"), ("C3", "protect"), ("C2", "earn"))     # tie order when two levers have the same grade


def layer_report(question_data: dict[str, Any], economics_data: dict[str, Any]) -> dict[str, Any]:
    """Compiler-derived per-layer view of the answers; nothing here comes from the model."""
    by_id = {q["id"]: q for q in question_data["questions"]}
    layers = []
    for key, name, pre_customer, ids in LAYERS:
        grades = [by_id[i]["grade_n"] for i in ids]
        floor = 1 if pre_customer else EVIDENCE_FLOOR       # desk statements can carry pre-customer layers; amounts, payment and repeat use need observed behavior
        status = "supported" if min(grades) >= floor else "unevidenced" if max(grades) == 0 else "partial"
        row = {"layer": key, "name": name, "pre_customer": pre_customer, "questions": len(ids), "status": status,
               "grade_counts": {f"E{g}": grades.count(g) for g in range(5)}, "at_e0": [i for i in ids if by_id[i]["grade_n"] == 0]}
        if key == "quantification":
            amounts = economics_data["economics"]
            row["amounts_supported"] = [k for k in ("S", "E", "P", "C", "R", "N", "F") if amounts[k].get("supported")]
            row["amounts_unsupported"] = [k for k in ("S", "E", "P", "C", "R", "N", "F") if not amounts[k].get("supported")]
        layers.append(row)
    counts = question_data["by_grade"]
    layers.append({"layer": "evidence", "name": "Value Evidence", "pre_customer": False, "questions": len(by_id), "status": "supported" if counts.get("E2", 0) + counts.get("E3", 0) + counts.get("E4", 0) >= len(by_id) else "partial" if counts.get("E0", 0) < len(by_id) else "unevidenced",
                   "grade_counts": dict(counts), "at_e0": [i for i, q in by_id.items() if q["grade_n"] == 0]})
    ranked = sorted(((by_id[q]["grade_n"], -n, name) for n, (q, name) in enumerate(LEVERS)), reverse=True)
    lever = {"primary": ranked[0][2] if ranked[0][0] > 0 else "undetermined", "secondary": [r[2] for r in ranked[1:] if r[0] > 0],
             "grades": {name: f"E{by_id[q]['grade_n']}" for q, name in LEVERS}}
    pre = [l for l in layers if l["pre_customer"]]
    gaps = sorted(i for l in pre for i in l["at_e0"])
    return {"layers": layers, "lever": lever, "pre_customer_ready": all(l["status"] == "supported" for l in pre), "pre_customer_gaps": gaps,
            "needs_customer_evidence": [l["layer"] for l in layers if not l["pre_customer"] and l["status"] != "supported"]}


def verdict_v3(data, inputs, policy, ctx=None):
    """verdict_v2 (unchanged, still used by stored runs) plus the layer view of section 6 of the pasted text, written by the compiler: which layers can be
    checked before customer contact, their gaps, and the value lever (save / protect / earn) read from C1-C3. A model-written layer_report is refused."""
    if "layer_report" in data:
        raise ValueError("layer_report is written by the compiler; remove it from the output")
    verdict_v2(data, inputs, policy, ctx)
    data["layer_report"] = layer_report(inputs["QUESTIONS"]["data"], inputs["ECONOMICS"]["data"])


VALIDATORS = {"value.intake": intake, "value.questions": questions, "value.economics": economics, "value.economics_v2": economics_v2, "value.verdict": verdict, "value.verdict_v2": verdict_v2, "value.verdict_v3": verdict_v3}
