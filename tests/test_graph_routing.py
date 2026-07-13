from langchain_core.messages import AIMessage
from langgraph.graph import END

from app.config import settings
from app.graph import route_after_decide, route_after_grade, run_tool


def _base_state(**overrides):
    state = {
        "messages": [],
        "query": "test",
        "retrieved_docs": [],
        "retry_count": 0,
        "tool_already_run": False,
        "last_grade": None,
    }
    state.update(overrides)
    return state


def test_route_after_grade_relevant_goes_to_respond():
    state = _base_state(last_grade="yes")
    assert route_after_grade(state) == "decide_and_respond"


def test_route_after_grade_irrelevant_with_retries_left_rewrites():
    state = _base_state(last_grade="no", retry_count=0)
    assert route_after_grade(state) == "rewrite_query"


def test_route_after_grade_irrelevant_but_retries_exhausted_gives_up_and_responds():
    state = _base_state(last_grade="no", retry_count=settings.max_retrieval_retries)
    assert route_after_grade(state) == "decide_and_respond"


def test_route_after_decide_with_tool_call_goes_to_run_tool():
    ai_msg = AIMessage(
        content="",
        tool_calls=[{"name": "check_rebate_eligibility", "args": {}, "id": "call_1"}],
    )
    state = _base_state(messages=[ai_msg])
    assert route_after_decide(state) == "run_tool"


def test_route_after_decide_without_tool_call_ends():
    ai_msg = AIMessage(content="Here is your answer.")
    state = _base_state(messages=[ai_msg])
    assert route_after_decide(state) == END


def test_run_tool_executes_eligibility_check_and_marks_tool_already_run():
    ai_msg = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "check_rebate_eligibility",
                "args": {
                    "household_zip": "94101",
                    "annual_income_usd": 80000,
                    "system_size_kw": 5.0,
                    "installer_approved": True,
                },
                "id": "call_1",
            }
        ],
    )
    state = _base_state(messages=[ai_msg])
    new_state = run_tool(state)

    assert new_state["tool_already_run"] is True
    tool_msg = new_state["messages"][0]
    assert tool_msg.name == "check_rebate_eligibility"
    assert "'eligible': True" in tool_msg.content


def test_run_tool_handles_unknown_tool_gracefully():
    ai_msg = AIMessage(
        content="",
        tool_calls=[{"name": "not_a_real_tool", "args": {}, "id": "call_2"}],
    )
    state = _base_state(messages=[ai_msg])
    new_state = run_tool(state)
    assert "Unknown tool" in new_state["messages"][0].content
