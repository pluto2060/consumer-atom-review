"""Builds the final deliverable of the consumer_atom_layer_review policy: an Excel workbook with the per-layer atomic-factor review and critique of one idea.

Main sheet "소비심리 기반 아이디어 검수" follows the owner's sample (image_205, 2026-10-15): summary block with the final judgment, the layer table (one row per activated
atom with a numeric 평가 점수 and a 검수 결과), the priority action plan, additional notes and the overall rating box. Appendix sheets hold all 75 atoms and the other views.
Every cell is copied from the committed artifacts of one run (values the compiler derived or the model wrote and the compiler accepted); the only formula is the 검수 결과 cell.
Runs of policy v1.0 lack some fields (score, hypothesis, recommendation, summary boxes): the cells say so instead of inventing values.

Run: python3 scripts/consumer_atom_review_xlsx.py --run-dir <runtime dir of the run> --out <file.xlsx>"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from agent_compiler.consumer_atom_dictionary import (  # noqa: E402
    ATOMS, BEHAVIOR_EFFECTS, COMPLETE_GRADE, EVIDENCE_LABEL, LAYERS, MANAGEMENT_FIELDS, MODERATORS, THEORY_STATUS, VALIDATION_TASKS)

STATUS_KO = {"hypothesis_only": "가설 수준(E0만 있음)", "statements_only": "진술 수준(E1까지)", "has_behavior_evidence": "행동 근거 있음(E2 이상)"}
REL_KO = {"activated": "작동", "inactive": "해당 없음", "unknown": "판단 불가"}
DIR_KO = {"supports": "구매에 유리", "hinders": "구매에 불리", "mixed": "혼합"}
STEP_FILES = ("atoms_l0_l3", "atoms_l4_l5", "atoms_l6_l7")
MISSING = "(v1.0 산출물에 없음)"
LAYER_FILL = {"L0": "DDEBF7", "L1": "BDD7EE", "L2": "D9E1F2", "L3": "E2EFDA", "L4": "C6E0B4", "L5": "E4DFEC", "L6": "FCE4D6", "L7": "FFF2CC"}
LABEL_FILL = {"확립됨": "C6EFCE", "간접적 지지": "FFEB9C", "제한적 근거": "FCE4D6", "근거 없음(가정)": "F8CBAD"}
EN = {"소비심리 기반 아이디어 검수": "Consumer-psychology idea review", "1. 아이디어 개요 및 핵심 요약": "1. Idea overview and key summary", "아이디어명": "Idea name", "핵심 고객": "Core customer",
      "해결하려는 문제": "Problem to solve", "핵심 소비 심리 가설": "Core consumer-psychology hypothesis", "근거 상태": "Evidence status", "기능적 가치": "Functional value", "정서적 가치": "Emotional value",
      "자아 형성 가치": "Self-formation value", "가장 큰 불확실성": "Biggest uncertainty", "최우선 검증 실험": "Top-priority experiment", "최종 판정": "Final judgment",
      "코드": "Code", "평가 영역": "Area", "원자 요인": "Atomic factor", "아이디어 적용 메커니즘": "How the idea applies", "검증 가설": "Hypothesis to verify", "검증 지표": "Indicator",
      "근거 수준": "Evidence level", "위험 및 반론": "Risks and counter-arguments", "개선 권고": "Improvement advice", "평가 점수": "Score", "검수 결과": "Review result", "방향": "Direction",
      "검증 완료": "Verified", "추가 검증 필요": "Needs more verification", "확립됨": "Established", "간접적 지지": "Indirect support", "제한적 근거": "Limited evidence", "근거 없음(가정)": "No evidence (assumption)",
      "(작동한 원자 없음)": "(no activated atom)", "2. 우선순위 액션 플랜": "2. Priority action plan", "추가 검토 사항": "Additional notes", "우선순위": "Priority", "실행 항목": "Action",
      "주요 내용": "Details", "담당 부서": "Owner", "예상 기간": "Duration", "종합 평가": "Overall rating", "아이디어 잠재력": "Idea potential", "실행 가능성": "Feasibility", "시장 매력도": "Market appeal",
      "평균 평가 점수": "Mean score", "검증 완료 비율": "Share verified", "추천 방향": "Recommended direction", "구매에 유리": "Favors purchase", "구매에 불리": "Hinders purchase", "혼합": "Mixed",
      "작동": "Activated", "해당 없음": "Not applicable", "판단 불가": "Unknown", "높음": "High", "보통": "Medium", "낮음": "Low", "진행": "Go", "조건부 진행": "Conditional go", "보류": "Hold", "중단": "Stop",
      "가설 수준(E0만 있음)": "Hypothesis only (E0)", "진술 수준(E1까지)": "Statements only (up to E1)", "행동 근거 있음(E2 이상)": "Behavior evidence (E2 or above)",
      "(v1.0 산출물에 없음)": "(not in the v1.0 output)", "원자 전체(75)": "All atoms (75)", "레이어별 평론": "Layer critique", "조절 변수": "Moderators", "인과 경로": "Causal path", "검증 과제": "Validation tasks",
      "증거 목록": "Evidence list", "레이어": "Layer", "이름": "Name", "원자 수": "Atoms", "해당 없음 수": "N/A", "유리": "Supports", "불리": "Hinders", "근거 등급(작동 원자)": "Grades (activated)",
      "평균 평가 점수 ": "Mean score ", "평론": "Critique", "가장 약한 원자": "Weakest atom", "없음": "none", "원인 원자": "Cause atom", "결과 원자": "Effect atom", "이유": "Reason", "과제": "Task",
      "대상 원자": "Atoms", "설계": "Design", "종류": "Kind", "내용": "Text", "인용한 원자": "Citing atoms", "(인용 없음)": "(not cited)", "모델 성격": "Model nature", "출처": "Source"}

HEADERS = ["코드", "평가 영역", "원자 요인", "아이디어 적용 메커니즘", "검증 가설", "검증 지표", "근거 수준", "위험 및 반론", "개선 권고", "평가 점수", "검수 결과", "방향"]
WIDTHS = [7, 14, 16, 40, 40, 32, 14, 40, 40, 9, 14, 12]


def _load(run_dir: Path, name: str) -> dict:
    a = json.load(open(run_dir / "artifacts" / f"{name}.json", encoding="utf-8"))
    return a.get("data", a)


def read_run(run_dir: Path) -> dict:
    state = json.load(open(run_dir / "state.json", encoding="utf-8"))
    m = re.search(r"EVIDENCE_JSON_BEGIN\n(.*)\nEVIDENCE_JSON_END", state["intent"], re.S)
    idea = re.search(r"IDEA_JSON_BEGIN\n(.*)\nIDEA_JSON_END", state["intent"], re.S)
    return {"run_id": state.get("id", run_dir.name), "status": state["status"], "intake": _load(run_dir, "intake"), "atoms": [a for n in STEP_FILES for a in _load(run_dir, n)["atoms"]],
            "layers": _load(run_dir, "layers"), "verdict": _load(run_dir, "verdict"), "evidence": json.loads(m.group(1)) if m else [],
            "idea": json.loads(idea.group(1))[0]["statement"] if idea else ""}


def build_workbook(run: dict):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

    thin = Side(style="thin", color="BFBFBF")
    box = Border(left=thin, right=thin, top=thin, bottom=thin)
    wb = Workbook()
    atoms = {a["id"]: a for a in run["atoms"]}
    verdict, layers, intake = run["verdict"], run["layers"], run["intake"]
    v2 = "final_judgment" in verdict
    lang = verdict.get("language", "ko")

    def tr(text):
        return EN.get(text, text) if lang == "en" else text

    def fill(color):
        return PatternFill("solid", fgColor=color)

    def put(ws, ref, value, bold=False, color=None, font_color=None, wrap=True, center=False, size=None):
        c = ws[ref]
        c.value = value
        c.font = Font(bold=bold, color=font_color or "000000", size=size or 10)
        c.alignment = Alignment(wrap_text=wrap, vertical="center", horizontal="center" if center else "left")
        if color:
            c.fill = fill(color)
        c.border = box
        return c

    def merged(ws, rng, value, **kw):
        ws.merge_cells(rng)
        first = rng.split(":")[0]
        put(ws, first, value, **kw)
        for row in ws[rng]:
            for c in row:
                c.border = box
                if kw.get("color"):
                    c.fill = fill(kw["color"])

    def band(ws, row, text):
        merged(ws, f"A{row}:L{row}", text, bold=True, color="1F3864", font_color="FFFFFF", size=11)

    ws = wb.active
    ws.title = tr("소비심리 기반 아이디어 검수")
    for i, w in enumerate(WIDTHS):
        ws.column_dimensions[chr(65 + i)].width = w

    band(ws, 1, tr("1. 아이디어 개요 및 핵심 요약"))
    name = verdict.get("idea_name") or intake["product"]["text"]
    left = [("아이디어명", name), ("핵심 고객", intake["customer"]["text"]), ("해결하려는 문제", verdict.get("problem") or intake["situation"]["text"]),
            ("핵심 소비 심리 가설", verdict.get("core_hypothesis") or verdict["overall"]), ("근거 상태", f"{tr(STATUS_KO[verdict['evidence_status']])} · {tr('작동')} {verdict['activated_total']}/{len(atoms)}")]
    mid = [("기능적 가치", verdict.get("functional_value", tr(MISSING))), ("정서적 가치", verdict.get("emotional_value", tr(MISSING))), ("자아 형성 가치", verdict.get("self_value", tr(MISSING))),
           ("가장 큰 불확실성", verdict.get("biggest_uncertainty", tr(MISSING))), ("최우선 검증 실험", verdict.get("priority_experiment", tr(MISSING)))]
    for n, ((lk, lv), (mk, mv)) in enumerate(zip(left, mid)):
        r = 2 + n
        merged(ws, f"A{r}:B{r}", tr(lk), bold=True, color="DDEBF7")
        merged(ws, f"C{r}:D{r}", lv)
        put(ws, f"E{r}", tr(mk), bold=True, color="DDEBF7")
        merged(ws, f"F{r}:H{r}", mv)
        ws.row_dimensions[r].height = 42
    merged(ws, "I2:J6", tr("최종 판정"), bold=True, color="DDEBF7", center=True)
    judgment = verdict.get("final_judgment")
    merged(ws, "K2:L6", (f"{tr(judgment)}\n\n{verdict['judgment_reason']}" if judgment else f"{tr(MISSING)}\n\n{verdict['overall']}"), bold=bool(judgment), color="FFF2CC")

    header_row = 8
    for i, h in enumerate(HEADERS):
        put(ws, f"{chr(65 + i)}{header_row}", tr(h), bold=True, color="1F3864", font_color="FFFFFF", center=True)
    ws.row_dimensions[header_row].height = 24
    ws.freeze_panes = f"A{header_row + 1}"
    r = header_row + 1
    for layer, lname in LAYERS.items():
        rows = [a for a in run["atoms"] if a["layer"] == layer and a["relevance"] == "activated"]
        start = r
        if not rows:
            put(ws, f"C{r}", tr("(작동한 원자 없음)"))
            r += 1
        for a in rows:
            label = a["evidence_label"] if "evidence_label" in a else EVIDENCE_LABEL[a["grade_n"]]
            score = a["score"] if "score" in a else a["confidence"]
            cells = [a["name"], a["mechanism"], a.get("hypothesis", a["claim"]), a["indicator"], label, a["critique"], a.get("recommendation", tr(MISSING)), score]
            cells[4] = tr(label)
            for col, value in zip("CDEFGHIJ", cells):
                put(ws, f"{col}{r}", value, center=col in "GJ")
            ws[f"G{r}"].fill = fill(LABEL_FILL[label])
            ws[f"J{r}"].fill = fill("C6EFCE" if score >= 4 else "FFEB9C" if score == 3 else "F8CBAD")
            put(ws, f"K{r}", f'=IF(G{r}="{tr("확립됨")}","{tr("검증 완료")}","{tr("추가 검증 필요")}")', bold=True, center=True,
                color="C6EFCE" if a["grade_n"] >= COMPLETE_GRADE else "FFEB9C")
            put(ws, f"L{r}", tr(DIR_KO[a["direction"]]), center=True)
            ws.row_dimensions[r].height = 62
            r += 1
        merged(ws, f"A{start}:A{r - 1}", layer, bold=True, color=LAYER_FILL[layer], center=True)
        merged(ws, f"B{start}:B{r - 1}", lname, bold=True, color=LAYER_FILL[layer], center=True)

    r += 1
    section = r
    merged(ws, f"A{r}:H{r}", tr("2. 우선순위 액션 플랜"), bold=True, color="1F3864", font_color="FFFFFF", size=11)
    merged(ws, f"I{r}:L{r}", tr("추가 검토 사항"), bold=True, color="1F3864", font_color="FFFFFF", size=11)
    r += 1
    for rng, text in (("A", "우선순위"), ("B:C", "실행 항목"), ("D:F", "주요 내용"), ("G", "담당 부서"), ("H", "예상 기간")):
        if ":" in rng:
            a, b = rng.split(":")
            merged(ws, f"{a}{r}:{b}{r}", tr(text), bold=True, color="DDEBF7", center=True)
        else:
            put(ws, f"{rng}{r}", tr(text), bold=True, color="DDEBF7", center=True)
    plan = verdict.get("action_plan")
    if plan:
        items = [(p["priority"], p["action"], p["detail"], p["owner"], p["duration"]) for p in plan]
    else:
        items = [(n, VALIDATION_TASKS[p["task"]], p["design"], MISSING, MISSING) for n, p in enumerate(verdict["validation_plan"], 1)]
    notes = verdict.get("additional_notes") or [f"{x['atom_id']} {atoms[x['atom_id']]['name']}: {x['why']}" for x in verdict["top_risks"]]
    top = r + 1
    for n, (pri, action, detail, owner, dur) in enumerate(items):
        rr = top + n
        put(ws, f"A{rr}", pri, center=True)
        merged(ws, f"B{rr}:C{rr}", action)
        merged(ws, f"D{rr}:F{rr}", detail)
        put(ws, f"G{rr}", owner, center=True)
        put(ws, f"H{rr}", dur, center=True)
        ws.row_dimensions[rr].height = 44
    end = max(top + len(items) - 1, top + 5)
    merged(ws, f"I{section + 1}:L{end}", "\n".join(f"• {x}" for x in notes))
    box_row = end + 2
    merged(ws, f"I{box_row}:L{box_row}", tr("종합 평가"), bold=True, color="1F3864", font_color="FFFFFF", size=11)
    rat = verdict.get("ratings") or {}
    mean = verdict.get("overall_mean_score")
    rows_box = [("아이디어 잠재력", rat.get("potential", MISSING)), ("실행 가능성", rat.get("feasibility", MISSING)), ("시장 매력도", rat.get("appeal", MISSING)),
                ("평균 평가 점수", f"{mean} / 5 (작동 원자 {verdict['activated_total']}개)" if mean is not None else MISSING),
                ("검증 완료 비율", f"{round(verdict['completed_share'] * 100)}%" if "completed_share" in verdict else MISSING), ("추천 방향", judgment or MISSING)]
    for n, (k, v) in enumerate(rows_box, 1):
        merged(ws, f"I{box_row + n}:J{box_row + n}", tr(k), bold=True, color="DDEBF7")
        merged(ws, f"K{box_row + n}:L{box_row + n}", tr(v), bold=True, color="FFF2CC" if v != "높음" else "C6EFCE", center=True)
    ws.sheet_view.zoomScale = 85

    def sheet(title, header, widths):
        s = wb.create_sheet(title)
        s.append(header)
        for c in s[1]:
            c.font = Font(bold=True, color="FFFFFF")
            c.fill = fill("305496")
            c.alignment = Alignment(wrap_text=True, vertical="top")
        for i, w in enumerate(widths):
            s.column_dimensions[chr(65 + i)].width = w
        s.freeze_panes = "A2"
        return s

    def wrap(s):
        for row in s.iter_rows(min_row=2):
            for c in row:
                c.alignment = Alignment(wrap_text=True, vertical="top")

    risk_ids = {x["atom_id"] for x in verdict["top_risks"]}
    s = sheet(tr("원자 전체(75)"), list(MANAGEMENT_FIELDS) + ["원자 이름", "관측 예시(사전)", "관련성", "방향", "이 아이디어에 대한 주장", "인용 증거", "평론", "평가 점수", "검증 가설", "개선 권고", "주요 위험"],
              [9, 16, 30, 30, 30, 30, 14, 24, 34, 16, 26, 10, 12, 50, 50, 60, 9, 40, 40, 9])
    for aid in ATOMS:
        a = atoms[aid]
        act = a["relevance"] == "activated"
        s.append([aid, f"{a['layer']} {a['layer_name']}", a["meaning"], a.get("precondition", "-"), a.get("mechanism", "-"), a.get("indicator", "-"),
                  ", ".join(BEHAVIOR_EFFECTS[e] for e in a["behavior_effect"]) if act else "-", (", ".join(MODERATORS[m][0] for m in a["moderators"]) or "(없음)") if act else "-",
                  f"{a['grade']}(이 아이디어) / 이론적 제안", a["name"], a["example"], REL_KO[a["relevance"]], DIR_KO[a["direction"]] if act else "-", a["claim"],
                  "\n".join(f"{e['id']}: {e['quote']}" for e in a["evidence"]) or "(없음: 가정)", a["critique"], (a.get("score", a.get("confidence")) if act else ""),
                  a.get("hypothesis", "-") if act else "-", a.get("recommendation", "-") if act else "-", "예" if aid in risk_ids else ""])
    wrap(s)

    s = sheet(tr("레이어별 평론"), ["레이어", "이름", "원자 수", "작동", "해당 없음", "판단 불가", "유리", "불리", "혼합", "근거 등급(작동 원자)", "평균 평가 점수", "평론", "가장 약한 원자"], [8, 14, 8, 8, 10, 10, 8, 8, 8, 28, 12, 80, 26])
    crit = {x["layer"]: x for x in layers["layers"]}
    for t in verdict["layer_table"]:
        weak = crit[t["layer"]]["weakest_atom"]
        s.append([t["layer"], t["name"], t["atoms"], t["activated"], t["inactive"], t["unknown"], t["supports"], t["hinders"], t["mixed"], " ".join(f"{k} {v}" for k, v in t["grades"].items()),
                  t.get("mean_score", "-"), crit[t["layer"]]["critique"], "없음" if weak == "none" else f"{weak} {atoms[weak]['name']}"])
    wrap(s)

    s = sheet(tr("조절 변수"), ["ID", "조절 변수", "작동 방식(사전)", "연결된 원자 수", "연결된 원자"], [6, 14, 50, 12, 90])
    for mid, (mname, how) in MODERATORS.items():
        linked = [i for i, a in atoms.items() if a["relevance"] == "activated" and mid in a["moderators"]]
        s.append([mid, mname, how, len(linked), ", ".join(f"{i} {atoms[i]['name']}" for i in linked) or "(없음)"])
    wrap(s)

    s = sheet(tr("인과 경로"), ["#", "원인 원자", "결과 원자", "이유", "원인 레이어→결과 레이어"], [5, 26, 26, 90, 20])
    for n, e in enumerate(layers["chain"], 1):
        s.append([n, f"{e['from']} {atoms[e['from']]['name']}", f"{e['to']} {atoms[e['to']]['name']}", e["why"], f"{atoms[e['from']]['layer']}→{atoms[e['to']]['layer']}"])
    cov = layers["chain_coverage"]
    s.append([])
    s.append(["", "경로가 지나는 레이어", ", ".join(cov["layers_on_chain"]), "구매 후 행동(L7)까지 닿음" if cov["reaches_behavior"] else "L7에 닿지 않음", ""])
    s.append(["", "경로에서 빠진 작동 레이어", ", ".join(cov["activated_layers_off_chain"]) or "없음", "", ""])
    wrap(s)

    s = sheet(tr("검증 과제"), ["과제", "대상 원자", "설계"], [26, 50, 100])
    for p in verdict["validation_plan"]:
        s.append([VALIDATION_TASKS[p["task"]], ", ".join(f"{i} {atoms[i]['name']}" for i in p["atom_ids"]), p["design"]])
    wrap(s)

    s = sheet(tr("증거 목록"), ["ID", "종류", "내용", "인용한 원자"], [6, 14, 110, 50])
    for e in run["evidence"]:
        used = [i for i, a in atoms.items() if any(x["id"] == e["id"] for x in a["evidence"])]
        s.append([e["id"], e["kind"], e["text"], ", ".join(used) or "(인용 없음)"])
    s.append([])
    s.append(["", "모델 성격", THEORY_STATUS, ""])
    s.append(["", "출처", f"run {run['run_id']} 산출물에서 그대로 옮김(재계산 없음)", ""])
    wrap(s)
    return wb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    run = read_run(args.run_dir)
    if run["status"] != "completed":
        raise SystemExit(f"run status is {run['status']}, the workbook is built only from a completed run")
    build_workbook(run).save(args.out)
    print(args.out)


if __name__ == "__main__":
    main()
