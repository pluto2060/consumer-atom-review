"""What a run cost: agent spend (every attempt, from the CLI's own usage report) plus the advisory judge calls."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


def _judge_spend(runtime: Any, run_id: str) -> tuple[float, int]:
    path = runtime.store.root(run_id) / "judgments.jsonl"
    if not path.is_file():
        return 0.0, 0
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return round(sum(r.get("cost_usd") or 0 for r in records), 4), len(records)


def run_cost(runtime: Any, run_id: str, state: dict[str, Any] | None = None) -> dict[str, Any]:
    """USD and tokens for one run. `known` is False for runs made before spend was recorded (their cost is unknown, not zero)."""
    state = state or runtime.get_run(run_id) or {}
    usage = state.get("usage") or {}
    by_step = {step: round(u.get("cost_usd", 0), 4) for step, u in usage.items()}
    agent_usd = round(sum(by_step.values()), 4)
    judge_usd, judgments = _judge_spend(runtime, run_id)
    tokens = {key: int(sum(u.get(key, 0) for u in usage.values()))
              for key in ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")}
    budget = (getattr(runtime, "policy", {}) or {}).get("budget") or {}
    cap = budget.get("run_usd")
    cap = round(cap + float(state.get("budget_extra_usd") or 0), 4) if cap else cap      # human-approved extensions raise the cap
    req = state.get("request_cap_usd")                                                    # the caller's own tighter cap (M2664.9)
    if isinstance(req, (int, float)) and not isinstance(req, bool):
        cap = round(min(cap, req), 4) if cap else round(float(req), 4)
    return {"known": bool(usage), "agent_usd": agent_usd, "judge_usd": judge_usd, "total_usd": round(agent_usd + judge_usd, 4),
            "by_step": by_step, "tokens": tokens, "judgments": judgments,
            "budget": ({"run_usd": cap, "remaining_usd": round(cap - agent_usd, 4)} if cap else None)}


# ---- global spend log and daily / weekly budget (M2709.T23) -----------------------------------------------------------------
# One append-only JSONL file shared by every runtime of one server (a per-runtime file would split the totals; it sits next to the server's tree ledger): {at, run_id, step, usd, source}. The windows are ROLLING (last 24 hours and last
# 7 days), so there is no midnight reset to game. Limits come from the owner through AC_BUDGET_DAILY_USD / AC_BUDGET_WEEKLY_USD; with neither set nothing is enforced by the
# executor (current behavior), and `check_budget(require_limits=True)` - what the autonomous loop calls - refuses to run without limits instead of treating "unset" as unlimited.
class BudgetExceeded(RuntimeError):
    """A global daily or weekly budget is used up (or unreadable): the paid call was not started."""


def spend_log_path() -> Path:
    """AC_SPEND_LOG, else `runtime/spend_log.jsonl` of this checkout: the default server runtime dir, where the server also keeps its shared tree ledger."""
    return Path(os.environ.get("AC_SPEND_LOG") or Path(__file__).resolve().parents[1] / "runtime" / "spend_log.jsonl")


def record_spend(run_id: str, usd: Any, step: str | None = None, log: Path | None = None) -> dict[str, Any] | None:
    """Append one cost entry; returns it, or None when `usd` is not a positive number (an unknown cost is not written as 0)."""
    if isinstance(usd, bool) or not isinstance(usd, (int, float)) or usd <= 0:
        return None
    entry = {"at": datetime.now(timezone.utc).isoformat(), "run_id": str(run_id), "step": step, "usd": round(float(usd), 6), "source": "claude -p"}
    path = Path(log) if log else spend_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def window_totals(log: Path | None = None, now: datetime | None = None) -> dict[str, float]:
    """USD spent in the last 24 hours and the last 7 days. An unreadable line raises ValueError (fail closed: a damaged ledger must not read as zero spend)."""
    path = Path(log) if log else spend_log_path()
    now = now or datetime.now(timezone.utc)
    day, week = 0.0, 0.0
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines() if path.is_file() else [], 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
            when, usd = datetime.fromisoformat(rec["at"]), float(rec["usd"])
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"spend log line {n} is not a valid entry") from exc
        age = now - when
        if age <= timedelta(days=7):
            week += usd
        if age <= timedelta(hours=24):
            day += usd
    return {"spent_24h": round(day, 6), "spent_7d": round(week, 6)}


def budget_limits() -> tuple[float | None, float | None]:
    """(daily, weekly) from the environment; a value that is not a positive number raises ValueError (a typo must not switch the budget off)."""
    out = []
    for name in ("AC_BUDGET_DAILY_USD", "AC_BUDGET_WEEKLY_USD"):
        raw = os.environ.get(name, "").strip()
        if not raw:
            out.append(None)
            continue
        try:
            value = float(raw)
        except ValueError:
            raise ValueError(f"{name} must be a number, got {raw!r}") from None
        if not value > 0:
            raise ValueError(f"{name} must be above 0, got {raw!r}")
        out.append(value)
    return out[0], out[1]


def check_budget(log: Path | None = None, now: datetime | None = None, daily: float | None = None, weekly: float | None = None, require_limits: bool = False) -> dict[str, Any]:
    """{ok, reason, spent_24h, spent_7d, daily_limit, weekly_limit}. ok is False when a limit is reached or exceeded, the ledger or a limit is unreadable,
    or (with require_limits) no limit is set. Limits default to the environment."""
    res: dict[str, Any] = {"ok": False, "reason": None, "spent_24h": None, "spent_7d": None, "daily_limit": None, "weekly_limit": None}
    try:
        if daily is None and weekly is None:
            daily, weekly = budget_limits()
        totals = window_totals(log, now)
    except ValueError as exc:
        return {**res, "reason": f"unreadable: {exc}"}
    res.update(totals, daily_limit=daily, weekly_limit=weekly)
    if daily is None and weekly is None:
        return {**res, "ok": not require_limits, "reason": "budget_unset" if require_limits else None}
    if daily is not None and totals["spent_24h"] >= daily:
        return {**res, "reason": f"daily budget reached: {totals['spent_24h']:.4f} of {daily:.4f} USD in 24 hours"}
    if weekly is not None and totals["spent_7d"] >= weekly:
        return {**res, "reason": f"weekly budget reached: {totals['spent_7d']:.4f} of {weekly:.4f} USD in 7 days"}
    return {**res, "ok": True}


def enforce(action: str, require_limits: bool = False, log: Path | None = None) -> dict[str, Any]:
    """Raise BudgetExceeded unless `check_budget` is ok; returns the check result otherwise."""
    res = check_budget(log=log, require_limits=require_limits)
    if not res["ok"]:
        raise BudgetExceeded(f"{res['reason']}; {action} was not started")
    return res
