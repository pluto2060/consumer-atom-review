"""Validators for the consumer_atom_layer_review policy (M2778): an input idea reviewed against the owner's consumer-psychology atom dictionary
(`consumer_atom_dictionary`: 8 layers L0-L7, one record per atomic factor) and critiqued per layer.

The model decides which atoms the idea touches, writes the mechanism and the critique, and cites evidence rows; the compiler derives everything it can:
layer, name and meaning of each atom, the evidence grade (from the kind of the cited row, never from the model), the confidence ceiling, the per-layer
table and the overall evidence status. Nothing here shows that an atom really drives purchase: the source text itself calls the model an integrating
framework that has to be validated atom by atom. The grade ladder and the confidence ceiling are the same policy parameters as in the value review."""
from __future__ import annotations

from typing import Any

from .consumer_atom_dictionary import (ATOMS, BEHAVIOR_EFFECTS, COMPLETE_GRADE, EVIDENCE_LABEL, JUDGMENTS, LAYERS, MODERATORS, PROCEED_SHARE, RATINGS, SCORE_CAP, STOP_MIN_AGAINST,
                                       THEORY_STATUS, VALIDATION_TASKS, layer_atoms)
from .marketing_validators import _rows
from .value_review_validators import GRADE_CAP, _evidence_rows, _fail, _grade_of, _is_int, _text, intake

LANGUAGES = ("ko", "en")
LANGUAGE_MIN_LETTERS = 8                 # shorter texts (a duration like "2주", a role) are only checked for the forbidden script
HANGUL_SHARE = 0.3                       # a Korean text may carry Latin terms (AI, token), but at least this share of its letters must be Hangul


def output_language(ctx) -> str:
    """The language the owner asked for: an optional LANGUAGE facts block {id: output, code: ko|en} in the intent; Korean when absent (owner, 2026-10-15: Korean first)."""
    intent = (ctx or {}).get("intent", "")
    if "LANGUAGE_JSON_BEGIN" not in intent:
        return "ko"
    code = _rows(intent, "LANGUAGE", "id", ("code",)).get("output", {}).get("code", "ko")
    if code not in LANGUAGES:
        raise ValueError(f"the LANGUAGE block must name ko or en, got {code!r}")
    return code


def _hangul(text: str) -> tuple[int, int]:
    hangul = sum(1 for ch in text if "\uac00" <= ch <= "\ud7a3" or "\u3131" <= ch <= "\u318e")
    latin = sum(1 for ch in text if ("a" <= ch.lower() <= "z"))
    return hangul, latin


def _check_language(texts: dict[str, Any], lang: str, problems: list[str]) -> None:
    """Free text must be in the requested language (the compiler checks the script, not the meaning); evidence quotes are never passed in."""
    for where, text in texts.items():
        if not isinstance(text, str):
            continue
        hangul, latin = _hangul(text)
        if lang == "en" and hangul:
            problems.append(f"{where}: the output language is en but the text contains Hangul")
        elif lang == "ko" and hangul + latin >= LANGUAGE_MIN_LETTERS and hangul / (hangul + latin) < HANGUL_SHARE:
            problems.append(f"{where}: the output language is ko but only {hangul}/{hangul + latin} letters are Hangul")


RELEVANCE = ("activated", "inactive", "unknown")
DIRECTIONS = ("supports", "hinders", "mixed")
BASE_KEYS = {"id", "relevance", "claim", "evidence", "critique", "confidence"}
ACTIVE_KEYS = BASE_KEYS | {"direction", "precondition", "mechanism", "indicator", "behavior_effect", "moderators"}
DERIVED_KEYS = {"layer", "layer_name", "name", "meaning", "example", "grade", "grade_n", "theory_status", "evidence_label", "review"}
BASE_KEYS_2 = {"id", "relevance", "claim", "evidence", "critique"}
ACTIVE_KEYS_2 = BASE_KEYS_2 | {"score", "direction", "precondition", "mechanism", "indicator", "behavior_effect", "moderators", "hypothesis", "recommendation"}
MAX_RISKS = 8


def _atom_step(layers: tuple[str, ...], v2: bool = False):
    expected = [i for layer in layers for i in layer_atoms(layer)]

    def validator(data, inputs, policy, ctx=None):
        rows = _evidence_rows(ctx)
        lang = output_language(ctx)
        items = data.get("atoms")
        if not isinstance(items, list):
            raise ValueError(f"atoms must be an array with one record for each of the {len(expected)} atoms of {list(layers)}")
        problems, seen = [], []
        for a in items:
            if not isinstance(a, dict) or a.get("id") not in expected:
                problems.append(f"each record needs an atom id from {expected[0]}..{expected[-1]} of layers {list(layers)}, got {str(a)[:60]!r}")
                continue
            aid = a["id"]
            seen.append(aid)
            for key in DERIVED_KEYS:
                a.pop(key, None)                                     # derived fields are rewritten, so the validator stays idempotent
            active = a.get("relevance") == "activated"
            if a.get("relevance") not in RELEVANCE:
                problems.append(f"{aid}: relevance must be one of {list(RELEVANCE)}")
            allowed = (ACTIVE_KEYS_2 if active else BASE_KEYS_2) if v2 else (ACTIVE_KEYS if active else BASE_KEYS)
            if set(a) != allowed:
                problems.append(f"{aid}: a {a.get('relevance')} record has exactly the keys {sorted(allowed)}, got {sorted(a)}")
                continue
            grade = _grade_of(a.get("evidence"), rows, aid, problems)
            if not _text(a.get("claim"), 12):
                problems.append(f"{aid}: claim must be a sentence of at least 12 characters")
            if not _text(a.get("critique"), 20 if active else 12):
                problems.append(f"{aid}: critique must be a sentence of at least {20 if active else 12} characters")
            if v2:
                if active:
                    if not _is_int(a.get("score")) or not 1 <= a["score"] <= 5:
                        problems.append(f"{aid}: score must be an integer 1-5")
                    elif a["score"] > SCORE_CAP[grade]:
                        problems.append(f"{aid}: score {a['score']} is above the ceiling {SCORE_CAP[grade]} for evidence grade E{grade}")
                    for key in ("hypothesis", "recommendation"):
                        if not _text(a.get(key), 12):
                            problems.append(f"{aid}: {key} must be a sentence of at least 12 characters")
            elif not _is_int(a.get("confidence")) or not 1 <= a["confidence"] <= 5:
                problems.append(f"{aid}: confidence must be an integer 1-5")
            elif a["confidence"] > GRADE_CAP[grade]:
                problems.append(f"{aid}: confidence {a['confidence']} is above the ceiling {GRADE_CAP[grade]} for evidence grade E{grade}")
            if active:
                if a.get("direction") not in DIRECTIONS:
                    problems.append(f"{aid}: direction must be one of {list(DIRECTIONS)}")
                for key in ("precondition", "mechanism", "indicator"):
                    if not _text(a.get(key), 12):
                        problems.append(f"{aid}: {key} must be a sentence of at least 12 characters")
                effects = a.get("behavior_effect")
                if not isinstance(effects, list) or not effects or any(e not in BEHAVIOR_EFFECTS for e in effects) or len(set(effects)) != len(effects):
                    problems.append(f"{aid}: behavior_effect must be a non-empty list without repeats from {list(BEHAVIOR_EFFECTS)}")
                mods = a.get("moderators")
                if not isinstance(mods, list) or any(m not in MODERATORS for m in mods) or len(set(mods)) != len(mods):
                    problems.append(f"{aid}: moderators must be a list without repeats from {list(MODERATORS)} (empty when none applies)")
            if v2:
                _check_language({f"{aid}.{k}": a.get(k) for k in ("claim", "critique", "precondition", "mechanism", "indicator", "hypothesis", "recommendation")}, lang, problems)
            meta = ATOMS[aid]
            a.update(layer=meta["layer"], layer_name=LAYERS[meta["layer"]], name=meta["name"], meaning=meta["meaning"], example=meta["example"],
                     grade=f"E{grade}", grade_n=grade, theory_status=THEORY_STATUS)
            if v2:
                a["evidence_label"] = EVIDENCE_LABEL[grade]
                if active:
                    a["review"] = "검증 완료" if grade >= COMPLETE_GRADE else "추가 검증 필요"
        if len(seen) != len(set(seen)) or set(seen) != set(expected):
            problems.append(f"every atom exactly once; missing {sorted(set(expected) - set(seen))}, repeated {sorted({s for s in seen if seen.count(s) > 1})}")
        _fail(problems)
        data["activated"] = sorted(a["id"] for a in items if a["relevance"] == "activated")

    return validator


atoms_l0_l3 = _atom_step(("L0", "L1", "L2", "L3"))
atoms_l4_l5 = _atom_step(("L4", "L5"))
atoms_l6_l7 = _atom_step(("L6", "L7"))
atoms2_l0_l3 = _atom_step(("L0", "L1", "L2", "L3"), v2=True)
atoms2_l4_l5 = _atom_step(("L4", "L5"), v2=True)
atoms2_l6_l7 = _atom_step(("L6", "L7"), v2=True)
ATOM_STEPS = ("ATOMS_L0_L3", "ATOMS_L4_L5", "ATOMS_L6_L7")


def all_atoms(inputs: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for step in ATOM_STEPS:
        for a in inputs[step]["data"]["atoms"]:
            out[a["id"]] = a
    return out


def layers(data, inputs, policy, ctx=None):
    """A critique for each of the 8 layers, the weakest activated atom per layer and the causal chain between activated atoms. The compiler derives which
    activated layers the chain reaches; a chain edge may point backwards (feedback from behavior to memory or self) because the source text says the order is not fixed."""
    atoms = all_atoms(inputs)
    activated = {i for i, a in atoms.items() if a["relevance"] == "activated"}
    rows = data.get("layers")
    problems = []
    if not isinstance(rows, list):
        raise ValueError("layers must be an array with one record for each layer L0..L7")
    seen = []
    for r in rows:
        if not isinstance(r, dict) or set(r) != {"layer", "critique", "weakest_atom"} or r.get("layer") not in LAYERS:
            problems.append(f"each layer record is exactly {{layer, critique, weakest_atom}} with a layer from {list(LAYERS)}, got {str(r)[:60]!r}")
            continue
        seen.append(r["layer"])
        if not _text(r["critique"], 40):
            problems.append(f"{r['layer']}: critique must be a sentence of at least 40 characters")
        own = [i for i in layer_atoms(r["layer"]) if i in activated]
        if own and r["weakest_atom"] not in own:
            problems.append(f"{r['layer']}: weakest_atom must be one of the activated atoms {own}, got {r['weakest_atom']!r}")
        if not own and r["weakest_atom"] != "none":
            problems.append(f"{r['layer']}: no atom of this layer is activated, so weakest_atom must be exactly none")
    if len(seen) != len(set(seen)) or set(seen) != set(LAYERS):
        problems.append(f"every layer L0..L7 exactly once; got {seen}")
    chain = data.get("chain")
    if not isinstance(chain, list) or len(chain) < 3:
        problems.append("chain must list at least 3 causal edges between activated atoms")
        chain = []
    pairs = []
    for e in chain:
        if not isinstance(e, dict) or set(e) != {"from", "to", "why"}:
            problems.append(f"each chain edge is exactly {{from, to, why}}, got {str(e)[:60]!r}")
            continue
        if e["from"] not in activated or e["to"] not in activated:
            problems.append(f"chain edge {e['from']!r} -> {e['to']!r}: both ends must be atoms activated in the atom steps")
        elif e["from"] == e["to"]:
            problems.append(f"chain edge {e['from']}: from and to are the same atom")
        if not _text(e.get("why"), 20):
            problems.append(f"chain edge {e.get('from')} -> {e.get('to')}: why must be a sentence of at least 20 characters")
        pairs.append((e.get("from"), e.get("to")))
    if len(pairs) != len(set(pairs)):
        problems.append("a chain edge appears twice")
    _fail(problems)
    touched = {atoms[i]["layer"] for e in chain for i in (e["from"], e["to"])}
    active_layers = sorted({atoms[i]["layer"] for i in activated})
    data["chain_coverage"] = {"layers_on_chain": sorted(touched), "activated_layers_off_chain": [l for l in active_layers if l not in touched],
                              "reaches_behavior": any(atoms[e["to"]]["layer"] == "L7" for e in chain)}


def intake2(data, inputs, policy, ctx=None):
    """intake plus the language check on the two text fields; the quotes stay in the idea statement's own language."""
    intake(data, inputs, policy, ctx)
    problems: list[str] = []
    _check_language({f"{k}.text": data[k]["text"] for k in ("product", "customer", "situation")}, output_language(ctx), problems)
    _fail(problems)


def layers2(data, inputs, policy, ctx=None):
    """layers (unchanged) plus the language check on every critique and chain reason."""
    layers(data, inputs, policy, ctx)
    problems: list[str] = []
    texts = {f"{r['layer']}.critique": r["critique"] for r in data["layers"]} | {f"chain {n}.why": e["why"] for n, e in enumerate(data["chain"], 1)}
    _check_language(texts, output_language(ctx), problems)
    _fail(problems)


def verdict(data, inputs, policy, ctx=None):
    """The closing critique. The compiler writes the per-layer table and the evidence status from the recorded atoms; the model names strengths, risks and the
    three validation tasks of section 5 of the source text, each tied to activated atoms."""
    atoms = all_atoms(inputs)
    activated = {i: a for i, a in atoms.items() if a["relevance"] == "activated"}
    problems = []
    if not activated:
        raise ValueError("no atom is activated, so there is nothing to critique; the atom steps must activate the atoms the idea touches")
    if not _text(data.get("overall"), 60):
        problems.append("overall must be a critique of at least 60 characters")
    strongest = data.get("strongest")
    if not isinstance(strongest, list) or len(strongest) > 5 or len(set(strongest)) != len(strongest) or any(s not in activated or activated[s]["direction"] != "supports" for s in strongest):
        problems.append("strongest must be a list of at most 5 distinct activated atoms whose direction is supports")
    risks = data.get("top_risks")
    hinders = {i for i, a in activated.items() if a["direction"] == "hinders"}
    ids = []
    if not isinstance(risks, list) or not 1 <= len(risks) <= max(MAX_RISKS, len(hinders)):
        problems.append(f"top_risks must hold 1-{max(MAX_RISKS, len(hinders))} records")
    else:
        for r in risks:
            if not isinstance(r, dict) or set(r) != {"atom_id", "why"} or r.get("atom_id") not in activated:
                problems.append(f"each risk is exactly {{atom_id, why}} with an activated atom, got {str(r)[:60]!r}")
                continue
            ids.append(r["atom_id"])
            if not _text(r["why"], 20):
                problems.append(f"risk {r['atom_id']}: why must be a sentence of at least 20 characters")
        if len(ids) != len(set(ids)):
            problems.append("an atom appears twice in top_risks")
        if hinders - set(ids):
            problems.append(f"every activated atom whose direction is hinders must be a listed risk; missing {sorted(hinders - set(ids))}")
    plan = data.get("validation_plan")
    tasks = []
    if not isinstance(plan, list):
        problems.append("validation_plan must be an array with one record for each of the three validation tasks")
    else:
        for p in plan:
            if not isinstance(p, dict) or set(p) != {"task", "atom_ids", "design"} or p.get("task") not in VALIDATION_TASKS:
                problems.append(f"each plan item is exactly {{task, atom_ids, design}} with a task from {list(VALIDATION_TASKS)}, got {str(p)[:60]!r}")
                continue
            tasks.append(p["task"])
            if not isinstance(p["atom_ids"], list) or not p["atom_ids"] or any(i not in activated for i in p["atom_ids"]):
                problems.append(f"{p['task']}: atom_ids must be a non-empty list of activated atoms")
            if not _text(p["design"], 30):
                problems.append(f"{p['task']}: design must be a sentence of at least 30 characters")
        if sorted(tasks) != sorted(VALIDATION_TASKS):
            problems.append(f"validation_plan must hold each of {list(VALIDATION_TASKS)} exactly once, got {tasks}")
    _fail(problems)
    table = []
    for layer, name in LAYERS.items():
        mine = [a for a in atoms.values() if a["layer"] == layer]
        act = [a for a in mine if a["relevance"] == "activated"]
        table.append({"layer": layer, "name": name, "atoms": len(mine), "activated": len(act), "inactive": sum(a["relevance"] == "inactive" for a in mine),
                      "unknown": sum(a["relevance"] == "unknown" for a in mine), "supports": sum(a["direction"] == "supports" for a in act),
                      "hinders": sum(a["direction"] == "hinders" for a in act), "mixed": sum(a["direction"] == "mixed" for a in act),
                      "grades": {f"E{g}": sum(a["grade_n"] == g for a in act) for g in range(5)}})
    top = max(a["grade_n"] for a in activated.values())
    data["layer_table"] = table
    data["evidence_status"] = "hypothesis_only" if top == 0 else "statements_only" if top == 1 else "has_behavior_evidence"
    data["activated_total"] = len(activated)
    data["note"] = "evidence_status is derived from the highest evidence grade among activated atoms (E0 only = hypothesis_only, E1 = statements_only, E2 or more = has_behavior_evidence); it says nothing about whether the idea will sell"


def verdict2(data, inputs, policy, ctx=None):
    """verdict (unchanged, still used by stored v1.0 runs) plus the sections of the owner's sample sheet (2026-10-15): idea summary, value boxes, judgment, ratings, action
    plan and additional notes written by the model, and per-layer mean scores, the completed share and the allowed judgments derived by the compiler. The judgment is
    limited by the recorded evidence: 진행 needs at least PROCEED_SHARE of the activated atoms at 검증 완료 (grade E3 or above), 중단 needs at least STOP_MIN_AGAINST
    hindering atoms backed by grade E2 or above. Ratings (높음/보통/낮음) stay model judgements."""
    verdict(data, inputs, policy, ctx)
    atoms = all_atoms(inputs)
    activated = {i: a for i, a in atoms.items() if a["relevance"] == "activated"}
    problems = []
    for key, minimum in (("idea_name", 2), ("problem", 20), ("core_hypothesis", 20), ("functional_value", 12), ("emotional_value", 12), ("self_value", 12),
                         ("biggest_uncertainty", 12), ("priority_experiment", 20), ("judgment_reason", 30)):
        if not _text(data.get(key), minimum):
            problems.append(f"{key} must be a sentence of at least {minimum} characters")
    ratings = data.get("ratings")
    if not isinstance(ratings, dict) or set(ratings) != {"potential", "feasibility", "appeal"} or any(v not in RATINGS for v in ratings.values()):
        problems.append(f"ratings must be exactly {{potential, feasibility, appeal}} each one of {list(RATINGS)}")
    plan = data.get("action_plan")
    if not isinstance(plan, list) or not 3 <= len(plan) <= 6:
        problems.append("action_plan must hold 3-6 items")
        plan = []
    for n, item in enumerate(plan, 1):
        if not isinstance(item, dict) or set(item) != {"priority", "action", "detail", "owner", "duration", "atom_ids"}:
            problems.append(f"action_plan item {n} is exactly {{priority, action, detail, owner, duration, atom_ids}}")
            continue
        if item["priority"] != n:
            problems.append(f"action_plan priorities must run 1..{len(plan)} in order, item {n} has {item['priority']!r}")
        for key, minimum in (("action", 6), ("detail", 20), ("owner", 2), ("duration", 2)):
            if not _text(item[key], minimum):
                problems.append(f"action_plan item {n}: {key} must be at least {minimum} characters")
        if not isinstance(item["atom_ids"], list) or not item["atom_ids"] or any(i not in activated for i in item["atom_ids"]):
            problems.append(f"action_plan item {n}: atom_ids must be a non-empty list of activated atoms")
    notes = data.get("additional_notes")
    if not isinstance(notes, list) or not 2 <= len(notes) <= 6 or any(not _text(x, 12) for x in notes):
        problems.append("additional_notes must be 2-6 sentences of at least 12 characters")
    done = [a for a in activated.values() if a["grade_n"] >= COMPLETE_GRADE]
    share = len(done) / len(activated)
    against = [a for a in activated.values() if a["direction"] == "hinders" and a["grade_n"] >= 2]
    allowed = ["조건부 진행", "보류"] + (["진행"] if share >= PROCEED_SHARE else []) + (["중단"] if len(against) >= STOP_MIN_AGAINST else [])
    if data.get("final_judgment") not in allowed:
        problems.append(f"final_judgment must be one of {allowed} (completed share {share:.2f}, hindering atoms at E2+ {len(against)}); got {data.get('final_judgment')!r}")
    texts = {k: data.get(k) for k in ("overall", "idea_name", "problem", "core_hypothesis", "functional_value", "emotional_value", "self_value", "biggest_uncertainty", "priority_experiment", "judgment_reason")}
    texts |= {f"top_risks {n}": r["why"] for n, r in enumerate(data["top_risks"], 1)} | {f"validation_plan {n}": p["design"] for n, p in enumerate(data["validation_plan"], 1)}
    texts |= {f"action_plan {n}.{k}": item[k] for n, item in enumerate(plan, 1) for k in ("action", "detail", "owner", "duration")} | {f"additional_notes {n}": x for n, x in enumerate(notes, 1)}
    lang = output_language(ctx)
    _check_language(texts, lang, problems)
    _fail(problems)
    data["language"] = lang
    for t in data["layer_table"]:
        scores = [a["score"] for a in activated.values() if a["layer"] == t["layer"]]
        t["mean_score"] = round(sum(scores) / len(scores), 2) if scores else None
        t["completed"] = sum(1 for a in activated.values() if a["layer"] == t["layer"] and a["grade_n"] >= COMPLETE_GRADE)
    all_scores = [a["score"] for a in activated.values()]
    data["overall_mean_score"] = round(sum(all_scores) / len(all_scores), 2)
    data["completed_share"] = round(share, 3)
    data["allowed_judgments"] = allowed


VALIDATORS = {"consumer.intake": intake, "consumer.atoms_l0_l3": atoms_l0_l3, "consumer.atoms_l4_l5": atoms_l4_l5, "consumer.atoms_l6_l7": atoms_l6_l7,
              "consumer.layers": layers, "consumer.verdict": verdict,
              "consumer.intake2": intake2, "consumer.layers2": layers2, "consumer.atoms2_l0_l3": atoms2_l0_l3, "consumer.atoms2_l4_l5": atoms2_l4_l5, "consumer.atoms2_l6_l7": atoms2_l6_l7, "consumer.verdict2": verdict2}
