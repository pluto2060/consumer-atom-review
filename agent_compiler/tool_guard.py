"""Claude Code PreToolUse hook: an independent, fail-closed second line of defense for tool use.

Installed per `claude -p` call through `--settings`. It does not trust the compiler process or the model:
any internal error, unknown tool, unsafe URL, or logging failure results in exit code 2 (block).
Every decision is appended to a JSONL guard log that the compiler cross-checks against the tool log.

Usage (set by claude_agent.py):  python tool_guard.py --allow WebSearch,WebFetch --log path/to/guard.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import re
import socket
import sys
import unicodedata
from datetime import datetime, timezone
from urllib.parse import urlparse

BLOCKED_SUFFIXES = (".local", ".internal", ".localhost", ".lan", ".home", ".corp")


def _private(address: str) -> bool:
    ip = ipaddress.ip_address(address.split("%")[0])
    if isinstance(ip, ipaddress.IPv6Address) and (ip.ipv4_mapped or ip.sixtofour or ip.teredo):
        embedded = ip.ipv4_mapped or ip.sixtofour or ip.teredo[1]      # judge the IPv4 address hidden inside the IPv6 one
        return _private(str(embedded))
    return (not ip.is_global) or ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved or ip.is_unspecified


def check_url(url: str) -> str | None:
    """Return a denial reason, or None if the URL is a public http(s) address."""
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in ("http", "https") or not host:
        return "only public http(s) URLs are allowed"
    if parsed.username or parsed.password:
        return "URLs with embedded credentials are not allowed"
    if host == "localhost" or host.endswith(BLOCKED_SUFFIXES):
        return f"host {host} is not public"
    try:
        if _private(host):
            return f"address {host} is not public"
        return None
    except ValueError:
        pass                                   # not a literal IP: resolve and check every address
    try:
        addresses = {info[4][0] for info in socket.getaddrinfo(host, None)}
    except OSError:
        return f"host {host} could not be resolved"
    if not addresses or any(_private(a) for a in addresses):
        return f"host {host} resolves to a non-public address"
    return None


def _walk_urls(value, found: list[str], depth: int = 0) -> None:
    """Every string that looks like a URL anywhere in a tool input (MCP inputs nest them under `parameters`)."""
    if depth > 6:
        return
    if isinstance(value, str):
        if value.strip().lower().startswith(("http://", "https://", "//", "ftp:", "file:")):
            found.append(value.strip())
    elif isinstance(value, dict):
        for item in value.values():
            _walk_urls(item, found, depth + 1)
    elif isinstance(value, list):
        for item in value:
            _walk_urls(item, found, depth + 1)


def input_url(tool_input: dict) -> str | None:
    """The URL a call targets, for the guard log (top-level `url` or the first nested one)."""
    if tool_input.get("url"):
        return str(tool_input["url"])
    found: list[str] = []
    _walk_urls(tool_input, found)
    return found[0] if found else None


def _host_allowed(url: str, hosts: list[str]) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return any(host == h or host.endswith("." + h) for h in hosts)


def decide_mcp(tool: str, tool_input: dict, rules: dict) -> tuple[str, str]:
    """Rules for an MCP tool `mcp__<server>__<name>`: read-only operations only, URLs public and on allowed hosts."""
    parts = tool.split("__")
    server = rules.get(parts[1]) if len(parts) == 3 else None
    if server is None:
        return "deny", f"MCP tool {tool!r} belongs to a server this step did not declare"
    if server.get("builtin"):                         # the compiler's own server: only the tools the policy listed for this tier, nothing else
        return ("allow", "allowed by step policy (builtin governance tool)") if parts[2] in set(server.get("allow_tools") or ()) \
            else ("deny", f"tool {parts[2]!r} is not a governance tool this step may call")
    field = server.get("operation_field") or "operation_id"
    allowed_ops = set(server.get("allow_operations") or [])
    if field in tool_input and tool_input[field] not in allowed_ops:
        return "deny", f"operation {tool_input[field]!r} is not in this step's read-only allowlist {sorted(allowed_ops)}"
    parameters = tool_input.get("parameters")
    if isinstance(parameters, dict) and field in parameters:
        return "deny", f"the operation field {field!r} must not be nested inside parameters"
    if tool.endswith("__execute_operation") or "operation_id" in tool_input:
        if field not in tool_input:
            return "deny", f"execute_operation without {field!r} is not allowed"
    urls: list[str] = []
    _walk_urls(tool_input, urls)
    for url in urls:
        reason = check_url(url)
        if reason:
            return "deny", reason
        if server.get("allow_hosts") and not _host_allowed(url, server["allow_hosts"]):
            return "deny", f"host of {url} is not in this server's allowed hosts {server['allow_hosts']}"
    return "allow", "allowed by step policy (MCP read-only operation)"


def decide(tool: str, tool_input: dict, allowed: set[str], mcp_rules: dict | None = None) -> tuple[str, str]:
    if tool not in allowed:
        return "deny", f"tool {tool!r} is not allowed for this step"
    if tool.startswith("mcp__"):
        return decide_mcp(tool, tool_input, mcp_rules or {})
    if tool == "WebFetch":
        reason = check_url(str(tool_input.get("url", "")))
        if reason:
            return "deny", reason
    if tool == "WebSearch" and not str(tool_input.get("query", "")).strip():
        return "deny", "empty search query"
    return "allow", "allowed by step policy"


# ---- Threads actions (M2709.T8) --------------------------------------------------------------------------------------------
THREADS_HOST = "graph.threads.net"


def content_hash(text: str) -> str:
    """Hash of a post text for the duplicate check: Unicode-normalised, case-folded, whitespace collapsed, so a re-posted text with other spacing or case still matches."""
    norm = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().casefold()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def check_threads_action(action: dict, state: dict, log) -> dict:
    """Decide whether one Threads action may run and append the decision to `log` (a JSONL path), allowed or not.

    action: {"endpoint": https url of the official API (host graph.threads.net), "text": post text, "approved_by": name of the human who approved this post}.
    state:  {"posted_hashes": hashes of earlier posts (see content_hash), "posts_today": int, "daily_limit": int >= 0}.
    Rules, first match wins: non_api_path, approval_missing, duplicate, daily_limit. Anything missing or malformed in `state` blocks (fail closed): a limit that is
    not an integer is `daily_limit_unset`, never unlimited. If the decision cannot be logged the action is blocked (`log_failed`), like the tool guard hook.
    Returns {"allow": bool, "rule": str | None, "reason": str}."""
    decision = _threads_decision(action, state)
    entry = {"at": datetime.now(timezone.utc).isoformat(), "allow": decision["allow"], "rule": decision["rule"], "reason": decision["reason"],
             "endpoint": str(action.get("endpoint"))[:200] if isinstance(action, dict) else None,
             "text_hash": content_hash(action["text"]) if isinstance(action, dict) and isinstance(action.get("text"), str) else None,
             "approved_by": action.get("approved_by") if isinstance(action, dict) else None}
    try:
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError as exc:
        return {"allow": False, "rule": "log_failed", "reason": f"decision could not be logged: {type(exc).__name__}"}
    return decision


def _threads_decision(action, state) -> dict:
    def block(rule: str, reason: str) -> dict:
        return {"allow": False, "rule": rule, "reason": reason}
    if not isinstance(action, dict) or not isinstance(state, dict):
        return block("malformed", "action and state must be objects")
    try:
        url = urlparse(str(action.get("endpoint") or ""))
        official = url.scheme == "https" and url.hostname == THREADS_HOST and not url.username and url.port in (None, 443)
    except ValueError:
        official = False
    if not official:
        return block("non_api_path", f"only https://{THREADS_HOST} (the official API) is allowed")
    text, approver = action.get("text"), action.get("approved_by")
    if not isinstance(approver, str) or not approver.strip():
        return block("approval_missing", "no named approver for this post")
    if not isinstance(text, str) or not text.strip():
        return block("malformed", "post text is empty")
    hashes = state.get("posted_hashes")
    if not isinstance(hashes, (list, set, tuple, frozenset)):
        return block("malformed", "posted_hashes is missing")
    if content_hash(text) in hashes:
        return block("duplicate", "an identical post was already published")
    limit, today = state.get("daily_limit"), state.get("posts_today")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0 or isinstance(today, bool) or not isinstance(today, int) or today < 0:
        return block("daily_limit_unset", "daily_limit and posts_today must be non-negative integers")
    if today >= limit:
        return block("daily_limit", f"{today} posts today, limit {limit}")
    return {"allow": True, "rule": None, "reason": "official API, approved, new content, within the daily limit"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow", default="")
    parser.add_argument("--log", required=True)
    parser.add_argument("--mcp-rules", default="")
    args = parser.parse_args(argv)
    allowed = {t for t in args.allow.split(",") if t}
    record = {"at": datetime.now(timezone.utc).isoformat(), "tool": None, "decision": "deny", "reason": "guard error"}
    try:
        event = json.load(sys.stdin)
        tool = event.get("tool_name")
        tool_input = event.get("tool_input") or {}
        tool_input = tool_input if isinstance(tool_input, dict) else {}
        record.update(tool=tool, url=input_url(tool_input))
        rules = json.load(open(args.mcp_rules, encoding="utf-8")) if args.mcp_rules else {}    # unreadable rules -> exception -> deny
        record["decision"], record["reason"] = decide(str(tool), tool_input, allowed, rules)
    except Exception as exc:                   # fail closed on anything unexpected
        record.update(decision="deny", reason=f"guard error: {type(exc).__name__}: {exc}")
    try:
        with open(args.log, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        record.update(decision="deny", reason="guard log is not writable")   # no audit trail -> no tool call
    if record["decision"] != "allow":
        print(f"Blocked by policy guard: {record['reason']}", file=sys.stderr)
        return 2
    return 0


class HookToolGuard:
    """ToolGuard implementation: registers this module as a fail-closed Claude Code PreToolUse hook."""

    def settings(self, directory, tools, mcp=None):
        from pathlib import Path
        # Exec form (`args`): shell-form commands silently failed on Windows (PowerShell), and a failing hook
        # that does not exit 2 is fail-open. The gate additionally requires one guard decision per tool call.
        args = [str(Path(__file__).resolve()), "--allow", ",".join(tools), "--log", str(Path(directory) / "guard.jsonl")]
        if mcp:
            rules = Path(directory) / "mcp_rules.json"
            rules.write_text(json.dumps(mcp), encoding="utf-8")
            args += ["--mcp-rules", str(rules)]
        hook = {"type": "command", "command": sys.executable, "args": args}
        target = Path(directory) / "settings.json"
        target.write_text(json.dumps({"hooks": {"PreToolUse": [{"matcher": "*", "hooks": [hook]}]}}), encoding="utf-8")
        return target

    def decisions(self, directory):
        from pathlib import Path
        log = Path(directory) / "guard.jsonl"
        return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line] if log.exists() else []


class NoToolGuard:
    """Explicitly no guard. Steps under `require_guard` policies will be blocked by the gate."""

    def settings(self, directory, tools, mcp=None):
        return None

    def decisions(self, directory):
        return []


if __name__ == "__main__":
    sys.exit(main())
