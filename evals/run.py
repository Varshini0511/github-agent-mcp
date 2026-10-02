"""Run the golden evaluation set and report pass/fail per case.

    python evals/run.py                 # run every case
    python evals/run.py rag-code-locator  # run one case by id
    python evals/run.py --list          # list case ids
    python evals/run.py --json report.json

Exit code is 0 only if every case passes, so this doubles as a CI gate.

Each case is scored two ways -- exact deterministic checks (tools called,
strings present/absent) AND an LLM-as-judge (evals/judge.py). A case
passes only if both agree.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
import uuid
from pathlib import Path

# Make the project root importable whether this is launched as
# `python evals/run.py` or `python -m evals.run`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
from langchain.chat_models import init_chat_model

from agent_core import build_agent
from config import configure_logging, get_settings
from evals.judge import judge_answer
from multi_agent import build_supervisor

CASES_FILE = Path(__file__).with_name("cases.yaml")


def load_cases() -> list[dict]:
    return yaml.safe_load(CASES_FILE.read_text(encoding="utf-8"))


def tool_names(messages) -> set[str]:
    """Every tool name invoked anywhere in a run's message trace."""
    names: set[str] = set()
    for m in messages:
        for tc in getattr(m, "tool_calls", None) or []:
            names.add(tc["name"] if isinstance(tc, dict) else getattr(tc, "name", ""))
    return {n for n in names if n}


def deterministic_checks(case: dict, answer: str, tools: set[str]) -> list[dict]:
    """Cheap exact checks. Each returns {name, ok, detail}."""
    out: list[dict] = []
    low = answer.lower()

    for t in case.get("expect_tools", []):
        out.append({"name": f"used tool `{t}`", "ok": t in tools,
                    "detail": f"tools called: {sorted(tools) or 'none'}"})

    for s in case.get("expect_contains", []):
        out.append({"name": f"answer contains '{s}'", "ok": s.lower() in low, "detail": ""})

    any_list = case.get("expect_any", [])
    if any_list:
        ok = any(s.lower() in low for s in any_list)
        out.append({"name": f"answer contains any of {any_list}", "ok": ok, "detail": ""})

    for s in case.get("forbid_contains", []):
        out.append({"name": f"answer avoids '{s}'", "ok": s.lower() not in low, "detail": ""})

    return out


async def run_case(agent, judge_model, case: dict) -> dict:
    """Execute one case and score it. A fresh thread_id keeps cases from
    sharing memory."""
    config = {"configurable": {"thread_id": f"eval-{case['id']}-{uuid.uuid4().hex[:6]}"}}
    result = await agent.ainvoke({"messages": [("user", case["question"])]}, config)
    answer = result["messages"][-1].content or ""
    tools = tool_names(result["messages"])

    checks = deterministic_checks(case, answer, tools)
    det_ok = all(c["ok"] for c in checks)

    verdict = await judge_answer(judge_model, case["question"], answer, case["rubric"])

    passed = det_ok and verdict["verdict"] == "pass"
    return {
        "id": case["id"],
        "passed": passed,
        "deterministic_ok": det_ok,
        "checks": checks,
        "judge": verdict,
        "answer": answer,
        "tools": sorted(tools),
    }


async def main(argv: list[str]) -> int:
    if "--list" in argv:
        for c in load_cases():
            print(f"{c['id']:24} [{c.get('mode', 'single')}] {c['question'][:60]}")
        return 0

    json_out = None
    if "--json" in argv:
        i = argv.index("--json")
        json_out = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]

    wanted = [a for a in argv if not a.startswith("--")]
    cases = [c for c in load_cases() if not wanted or c["id"] in wanted]
    if not cases:
        print(f"No cases matched {wanted}. Try --list.")
        return 2

    configure_logging()
    settings = get_settings()
    judge_model = init_chat_model(settings.chat_model)

    # Group by mode so each agent (and its MCP subprocess) is built once.
    by_mode: dict[str, list[dict]] = {}
    for c in cases:
        by_mode.setdefault(c.get("mode", "single"), []).append(c)

    results: list[dict] = []
    for mode, mode_cases in by_mode.items():
        builder = build_supervisor if mode == "multi" else build_agent
        async with builder() as agent:
            for n, case in enumerate(mode_cases):
                print(f"  running {case['id']} ...", flush=True)
                try:
                    results.append(await run_case(agent, judge_model, case))
                except Exception as exc:  # a crash is itself a failed case
                    results.append({
                        "id": case["id"], "passed": False, "deterministic_ok": False,
                        "checks": [], "judge": {"score": 0, "verdict": "fail",
                        "reasoning": f"run crashed: {type(exc).__name__}: {exc}"},
                        "answer": "", "tools": [],
                    })
                if n + 1 < len(mode_cases):
                    time.sleep(3)  # gentle on the Groq free-tier rate limit

    _print_report(results)
    if json_out:
        Path(json_out).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nWrote {json_out}")

    return 0 if all(r["passed"] for r in results) else 1


def _print_report(results: list[dict]) -> None:
    print("\n" + "=" * 72)
    print(f"{'CASE':24} {'RESULT':8} {'JUDGE':6} DETAIL")
    print("-" * 72)
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        score = r["judge"].get("score", 0)
        failed_checks = [c["name"] for c in r["checks"] if not c["ok"]]
        detail = ""
        if failed_checks:
            detail = "failed: " + "; ".join(failed_checks)
        elif r["judge"]["verdict"] == "fail":
            detail = "judge: " + r["judge"].get("reasoning", "")
        print(f"{r['id']:24} {status:8} {score}/5    {detail[:44]}")
    passed = sum(r["passed"] for r in results)
    print("-" * 72)
    print(f"{passed}/{len(results)} passed")
    print("=" * 72)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(asyncio.run(main(sys.argv[1:])))
