"""
Runs the eval query set against the live agent (real Groq calls -- needs
GROQ_API_KEY) and prints a pass/fail report. This is a lightweight,
assertion-based eval, not a full RAGAS-style scored eval -- see DESIGN.md
for why, and for what a next iteration would add.

    python -m eval.run_eval
"""
import json
import sys
from pathlib import Path

from app.config import settings
from app.graph import run_query
from app.vectorstore import build_vectorstore

EVAL_FILE = Path(__file__).parent / "eval_queries.json"


def check_case(case: dict, result: dict) -> tuple[bool, str]:
    answer = result["messages"][-1].content
    sources = {d.metadata.get("source", "") for d in result["retrieved_docs"]}
    tool_called = any(
        getattr(m, "name", None) == "check_rebate_eligibility" for m in result["messages"]
    )

    if "expect_sources_any_of" in case:
        if not sources.intersection(case["expect_sources_any_of"]):
            return False, f"expected one of {case['expect_sources_any_of']}, got {sources}"

    if "expect_tool_call" in case and not tool_called:
        return False, "expected the eligibility tool to be called, but it wasn't"

    if case.get("expect_eligible") is not None:
        wants = "True" if case["expect_eligible"] else "False"
        tool_msgs = [m.content for m in result["messages"] if getattr(m, "name", None) == "check_rebate_eligibility"]
        if not any(f"'eligible': {wants}" in t for t in tool_msgs):
            return False, f"expected eligible={case['expect_eligible']} in tool result, got {tool_msgs}"

    if case.get("expect_clarifying_question"):
        if tool_called:
            return False, "expected a clarifying question, but the tool was called with (likely guessed) values"
        if "?" not in answer:
            return False, "expected a clarifying question but answer contains no '?'"

    if "expect_answer_contains_any_of" in case:
        low = answer.lower()
        if not any(phrase.lower() in low for phrase in case["expect_answer_contains_any_of"]):
            return False, f"expected answer to mention one of {case['expect_answer_contains_any_of']}"

    if case.get("expect_out_of_scope"):
        # heuristic: should not confidently assert unrelated real-world facts
        if any(w in answer.lower() for w in ["degrees", "sunny", "rain", "forecast"]):
            return False, "agent appears to have answered an out-of-scope question instead of declining"

    return True, "ok"


def main() -> int:
    if not settings.groq_api_key:
        print("GROQ_API_KEY is not set -- eval needs live LLM calls.", file=sys.stderr)
        return 1

    cases = json.loads(EVAL_FILE.read_text())
    vectorstore = build_vectorstore()

    passed = 0
    for case in cases:
        result = run_query(vectorstore, case["question"])
        ok, detail = check_case(case, result)
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {case['id']}: {detail}")
        if ok:
            passed += 1

    print(f"\n{passed}/{len(cases)} passed")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
