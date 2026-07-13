"""
The agentic RAG graph.

Flow
----
    retrieve --> grade_documents --+-- (relevant / retries exhausted) --> decide_and_respond
                       ^            |
                       |            +-- (not relevant, retries left) --> rewrite_query --+
                       +---------------------------------------------------------------+

    decide_and_respond --+-- (LLM calls check_rebate_eligibility) --> run_tool --> finalize --> END
                          +-- (LLM answers directly, or asks a clarifying question) --> END

Why a graph instead of a single retrieve-then-generate chain
--------------------------------------------------------------
Three failure modes a fixed pipeline can't recover from on its own:
  1. The first retrieval is off-topic (bad phrasing, acronym, typo) -> the
     `grade_documents` node catches this and `rewrite_query` gives the
     retriever another shot, bounded by `max_retrieval_retries`.
  2. The question requires a live determination (e.g. "am I eligible?")
     that the docs alone can't answer -> the model has the eligibility
     tool bound and is instructed to call it once it has all four
     required inputs, rather than inventing an answer from prose.
  3. The eligibility question is missing required inputs -> the model is
     instructed to ask the user for them instead of assuming values.

Retrieval itself is deliberately NOT filtered by category (see
DESIGN.md). It's full-corpus semantic search every time; category tags
are attached only as citation metadata. This is what lets the
deliberately cross-cutting doc (06) surface for both rebate-reversal and
billing-adjustment style queries without us having to force it into one
bucket.
"""
import logging
from typing import Annotated, List, Literal, Optional, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from app.config import settings
from app.tools import ALL_TOOLS, check_eligibility_tool, format_docs_for_context

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    query: str
    retrieved_docs: List[Document]
    retry_count: int
    tool_already_run: bool
    last_grade: Optional[str]


class GradeDocuments(BaseModel):
    """Structured grade for whether retrieved chunks can answer the query."""

    binary_score: Literal["yes", "no"] = Field(
        description="'yes' if the retrieved chunks contain enough information to answer the query, else 'no'."
    )
    reasoning: str = Field(description="One sentence explaining the grade.")


class RewrittenQuery(BaseModel):
    rewritten_query: str = Field(description="A clearer, more retrievable rephrasing of the user's question.")


SYSTEM_PROMPT = """You are the SunGrid Cooperative Copilot, an internal assistant for SunGrid \
Cooperative staff and members. Answer using ONLY the retrieved document context you are given \
-- do not use outside knowledge about solar policy or make up SunGrid-specific numbers, dates, \
or rules that aren't in the context.

Rules:
- Always cite sources using the bracketed numbers from the context, e.g. "[1]".
- If the retrieved context touches more than one policy area (e.g. a rebate reversal that is \
also a billing matter), say so explicitly rather than picking one and ignoring the other.
- If the question asks for a concrete eligibility determination or rebate estimate for a \
specific household, use the check_rebate_eligibility tool. Only call it once you have all four \
inputs: household_zip, annual_income_usd, system_size_kw, and installer_approved. If any are \
missing, ask the user for them plainly -- never guess or assume a default.
- If the context doesn't contain the answer, say so plainly instead of guessing.
- Be concise. This is a staff/member support tool, not a marketing document.
"""


def _get_llm(temperature: Optional[float] = None) -> ChatGroq:
    return ChatGroq(
        model=settings.groq_model,
        api_key=settings.groq_api_key,
        temperature=settings.llm_temperature if temperature is None else temperature,
    )


# --------------------------------------------------------------------------
# Nodes
# --------------------------------------------------------------------------
def make_retrieve_node(vectorstore):
    def retrieve(state: AgentState) -> AgentState:
        query = state["query"]
        docs = vectorstore.similarity_search(query, k=settings.top_k)
        logger.debug("Retrieved %d docs for query=%r", len(docs), query)
        return {**state, "retrieved_docs": docs}

    return retrieve


def grade_documents(state: AgentState) -> AgentState:
    llm = _get_llm(temperature=0.0)
    grader = llm.with_structured_output(GradeDocuments)
    context = format_docs_for_context(state["retrieved_docs"])
    result: GradeDocuments = grader.invoke(
        [
            SystemMessage(
                content="Grade whether the CONTEXT below contains enough information to answer the QUESTION. "
                "Respond with a binary score only."
            ),
            HumanMessage(content=f"QUESTION: {state['query']}\n\nCONTEXT:\n{context}"),
        ]
    )
    logger.debug("Grade=%s reasoning=%s", result.binary_score, result.reasoning)
    return {**state, "last_grade": result.binary_score}


def rewrite_query(state: AgentState) -> AgentState:
    llm = _get_llm(temperature=0.2)
    rewriter = llm.with_structured_output(RewrittenQuery)
    result: RewrittenQuery = rewriter.invoke(
        [
            SystemMessage(
                content="The user's question didn't retrieve good results from a document search. "
                "Rewrite it to be clearer and more likely to match relevant internal policy documents. "
                "Expand acronyms, fix ambiguity, keep the original intent."
            ),
            HumanMessage(content=state["query"]),
        ]
    )
    logger.debug("Rewrote query: %r -> %r", state["query"], result.rewritten_query)
    return {
        **state,
        "query": result.rewritten_query,
        "retry_count": state["retry_count"] + 1,
    }


def decide_and_respond(state: AgentState) -> AgentState:
    llm = _get_llm()
    llm_with_tools = llm.bind_tools(ALL_TOOLS) if not state["tool_already_run"] else llm

    context = format_docs_for_context(state["retrieved_docs"])
    history = state["messages"]

    if state["tool_already_run"]:
        # We already have a tool result appended to `messages` as a ToolMessage;
        # just ask the model to produce the final answer, no further tool calls.
        prompt_messages = [SystemMessage(content=SYSTEM_PROMPT)] + history
    else:
        prompt_messages = (
            [SystemMessage(content=SYSTEM_PROMPT)]
            + history
            + [HumanMessage(content=f"CONTEXT:\n{context}\n\nQUESTION: {state['query']}")]
        )

    response: AIMessage = llm_with_tools.invoke(prompt_messages)
    return {**state, "messages": [response]}


def run_tool(state: AgentState) -> AgentState:
    last_message = state["messages"][-1]
    tool_messages = []
    for call in last_message.tool_calls:
        if call["name"] == check_eligibility_tool.name:
            result = check_eligibility_tool.invoke(call["args"])
            tool_messages.append(
                ToolMessage(content=str(result), tool_call_id=call["id"], name=call["name"])
            )
        else:
            tool_messages.append(
                ToolMessage(
                    content=f"Unknown tool: {call['name']}", tool_call_id=call["id"], name=call["name"]
                )
            )
    return {**state, "messages": tool_messages, "tool_already_run": True}


# --------------------------------------------------------------------------
# Conditional edges
# --------------------------------------------------------------------------
def route_after_grade(state: AgentState) -> Literal["rewrite_query", "decide_and_respond"]:
    grade = state.get("last_grade", "yes")
    if grade == "no" and state["retry_count"] < settings.max_retrieval_retries:
        return "rewrite_query"
    return "decide_and_respond"


def route_after_decide(state: AgentState) -> Literal["run_tool", "__end__"]:
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "run_tool"
    return END


# --------------------------------------------------------------------------
# Graph assembly
# --------------------------------------------------------------------------
def build_graph(vectorstore):
    graph = StateGraph(AgentState)

    graph.add_node("retrieve", make_retrieve_node(vectorstore))
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("decide_and_respond", decide_and_respond)
    graph.add_node("run_tool", run_tool)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents",
        route_after_grade,
        {"rewrite_query": "rewrite_query", "decide_and_respond": "decide_and_respond"},
    )
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_conditional_edges(
        "decide_and_respond",
        route_after_decide,
        {"run_tool": "run_tool", END: END},
    )
    graph.add_edge("run_tool", "decide_and_respond")

    return graph.compile()


def run_query(vectorstore, question: str, history: Optional[List[BaseMessage]] = None) -> AgentState:
    app = build_graph(vectorstore)
    initial_state: AgentState = {
        "messages": (history or []) + [HumanMessage(content=question)],
        "query": question,
        "retrieved_docs": [],
        "retry_count": 0,
        "tool_already_run": False,
        "last_grade": None,
    }
    return app.invoke(initial_state, config={"recursion_limit": settings.max_agent_steps * 4})
