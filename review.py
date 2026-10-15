"""One-command runner: an idea (text file) plus evidence rows (CSV: id,kind,text) in, a review workbook (xlsx) out.

    python3 review.py --idea examples/idea.txt --evidence examples/evidence.csv --lang en --cap 1.5 --out review.xlsx

kind is one of: assumption, statement, behavior, payment, repeat_outcome (it sets the evidence grade E0-E4; the compiler caps each score by it).
It calls your own logged-in `claude` CLI, so the cost is billed to your account; --cap is a hard spending limit in USD for this run."""
import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from agent_compiler.claude_agent import ClaudeCliAgent  # noqa: E402
from agent_compiler.executor import GatedWorkflowRuntime  # noqa: E402

KINDS = ("assumption", "statement", "behavior", "payment", "repeat_outcome")


def block(key, rows):
    return f"{key}_JSON_BEGIN\n{json.dumps(rows, ensure_ascii=False)}\n{key}_JSON_END"


def build_intent(idea: str, evidence: list[dict], lang: str) -> str:
    return "\n".join(["Review this idea.", block("IDEA", [{"id": "idea", "statement": idea.strip()}]), block("EVIDENCE", evidence),
                      block("LANGUAGE", [{"id": "output", "code": lang}])])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--idea", required=True, type=Path)
    ap.add_argument("--evidence", required=True, type=Path)
    ap.add_argument("--lang", choices=("en", "ko"), default="en")
    ap.add_argument("--cap", type=float, required=True, help="spending cap in USD for this run (a run of the sample costs about 1 USD)")
    ap.add_argument("--out", type=Path, default=Path("review.xlsx"))
    ap.add_argument("--runs", type=Path, default=HERE / "runs")
    a = ap.parse_args()
    evidence = list(csv.DictReader(open(a.evidence, encoding="utf-8", newline="")))
    for r in evidence:
        if r.get("kind") not in KINDS:
            raise SystemExit(f"evidence {r.get('id')}: kind must be one of {KINDS}, got {r.get('kind')!r}")
    rt = GatedWorkflowRuntime(HERE / "policy.json", a.runs)
    state = rt.create_run(build_intent(a.idea.read_text(encoding="utf-8"), evidence, a.lang), request_cap_usd=a.cap)
    print("run", state["id"], "cap", state.get("request_cap_usd"))
    final = rt.run_all(state["id"], ClaudeCliAgent(), planner=False)
    cost = sum((v or {}).get("cost_usd", 0) for v in (final.get("usage") or {}).values())
    print("status", final["status"], "cost_usd", round(cost, 4))
    if final["status"] != "completed":
        raise SystemExit("the run did not complete; see the run directory under " + str(a.runs))
    import consumer_atom_xlsx as x
    run_dir = next(p for p in a.runs.rglob(state["id"]) if p.is_dir())
    x.build_workbook(x.read_run(run_dir)).save(a.out)
    print("workbook", a.out)


if __name__ == "__main__":
    main()
