"""Validators for the marketing_content policy: content may only assert what upstream verified intents and their recorded evidence support.

The upstream facts arrive inside the node's intent as `<KEY>_JSON_BEGIN ... <KEY>_JSON_END` blocks written by the group runtime from
committed artifacts (INTENTS, VALIDATIONS, BEHAVIORS). Nothing here trusts the model's account of what upstream said.
"""

from __future__ import annotations

import json
import re
from typing import Any
from .validator_common import squash as _squash


def _block(intent: str, key: str) -> list[dict[str, Any]]:
    found = re.search(rf"{key}_JSON_BEGIN\n(.*?)\n{key}_JSON_END", intent or "", re.S)
    if not found:
        raise ValueError(f"the intent carries no {key} facts block; content cannot be grounded")
    try:
        rows = json.loads(found.group(1))
    except ValueError as exc:
        raise ValueError(f"the {key} facts block is not valid JSON: {exc}") from exc
    if not isinstance(rows, list):
        raise ValueError(f"the {key} facts block must be a list")
    return rows


def _rows(intent: str, key: str, id_field: str, required: tuple[str, ...] = ()) -> dict[str, dict[str, Any]]:
    """Facts block as {id: row}; a row that is not an object, lacks its id or a required field is a validation error, not a crash."""
    out: dict[str, dict[str, Any]] = {}
    for row in _block(intent, key):
        if not isinstance(row, dict) or not isinstance(row.get(id_field), str) or any(f not in row for f in required):
            raise ValueError(f"the {key} facts block holds a malformed row {str(row)[:80]!r}; each row needs a string {id_field}"
                             + (f" and {list(required)}" if required else ""))
        out[row[id_field]] = row
    return out


def _marketing_facts(ctx: dict[str, Any]) -> dict[str, Any]:
    intent = ctx.get("intent", "")
    return {"intents": _rows(intent, "INTENTS", "id"), "status": {k: v.get("status") for k, v in _rows(intent, "VALIDATIONS", "intent_id").items()},
            "behaviors": _rows(intent, "BEHAVIORS", "id", ("evidence_quote",))}


def _posts(inputs: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {p["id"]: p for p in inputs["DRAFT"]["data"]["posts"]}


def draft(data, inputs, policy, ctx=None):
    facts, limits = _marketing_facts(ctx or {}), policy.get("channel_limits") or {}
    posts = data.get("posts")
    if not isinstance(posts, list) or not posts:
        raise ValueError("posts must be a non-empty list")
    seen: set[str] = set()
    problems: list[str] = []
    banned = [_squash(p) for p in policy.get("forbidden_phrases", [])]
    for post in posts:
        pid = post.get("id") if isinstance(post, dict) else None
        if not isinstance(pid, str) or not pid or pid in seen:
            problems.append(f"post id {pid!r} must be a unique non-empty string")
            continue
        seen.add(pid)
        text = post.get("text")
        if post.get("channel") not in limits:
            problems.append(f"{pid}: channel must be one of {sorted(limits)}")
        elif not isinstance(text, str) or not text.strip() or len(text) > limits[post["channel"]]:
            problems.append(f"{pid}: text must be 1..{limits[post['channel']]} characters for {post['channel']}")
        for key in ("claim", "cta"):
            if not isinstance(post.get(key), str) or not post[key].strip():
                problems.append(f"{pid}: {key} must be a non-empty string")
        iid = post.get("intent_id")
        if iid not in facts["intents"]:
            problems.append(f"{pid}: intent_id {iid!r} is not one of the upstream intents {sorted(facts['intents'])}")
        elif facts["status"].get(iid) != "confirmed":
            problems.append(f"{pid}: intent {iid} is '{facts['status'].get(iid)}', only confirmed intents may be marketed")
        if isinstance(text, str) and any(b in _squash(text) for b in banned):
            problems.append(f"{pid}: text uses a forbidden absolute claim ({policy.get('forbidden_phrases')})")
    if problems:
        raise ValueError(f"{len(problems)} problem(s): " + "; ".join(problems))


def ground(data, inputs, policy, ctx=None):
    facts, posts = _marketing_facts(ctx or {}), _posts(inputs)
    rows = data.get("groundings")
    if not isinstance(rows, list):
        raise ValueError("groundings must be a list")
    ids = [r.get("post_id") for r in rows if isinstance(r, dict)]
    if sorted(ids) != sorted(posts) or len(set(ids)) != len(ids):
        raise ValueError(f"groundings must cover every drafted post exactly once: expected {sorted(posts)}, got {sorted(map(str, ids))}")
    for row in rows:
        intent = facts["intents"][posts[row["post_id"]]["intent_id"]]
        evidence = [_squash(facts["behaviors"][b]["evidence_quote"]) for b in intent.get("behavior_ids", []) if b in facts["behaviors"]]
        quote = _squash(row.get("evidence_quote", ""))
        if not quote or not any(quote in e for e in evidence):
            raise ValueError(f"{row['post_id']}: evidence_quote must be copied from the recorded evidence of intent "
                             f"{intent['id']} (behaviors {intent.get('behavior_ids')}), not found")


def final(data, inputs, policy, ctx=None):
    posts = _posts(inputs)
    publish, report = data.get("publish"), data.get("report")
    limit = int((policy.get("thresholds") or {}).get("max_publish", 3))
    if not isinstance(publish, list) or not publish or len(set(publish)) != len(publish):
        raise ValueError("publish must be a non-empty list of distinct post ids")
    if len(publish) > limit:
        raise ValueError(f"at most {limit} posts may be published in one pass, got {len(publish)}")
    unknown = [p for p in publish if p not in posts]
    if unknown:
        raise ValueError(f"publish names posts that were never drafted and grounded: {unknown}")
    if not isinstance(report, list) or sorted(r.get("post_id") for r in report if isinstance(r, dict)) != sorted(publish):
        raise ValueError("report must hold exactly one row per published post")
    for row in report:
        if row.get("text") != posts[row["post_id"]]["text"] or row.get("channel") != posts[row["post_id"]]["channel"]:
            raise ValueError(f"{row['post_id']}: published text/channel must equal the drafted post exactly")


VALIDATORS = {"marketing.draft": draft, "marketing.ground": ground, "marketing.final": final}
