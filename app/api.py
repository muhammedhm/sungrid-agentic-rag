"""
FastAPI service around the agent, with chat-thread support so the
frontend (or any client) gets history-aware conversation across turns.

    uvicorn app.api:app --reload

Then open http://localhost:8000/ for the bundled frontend, or call the
API directly:

    curl -X POST localhost:8000/api/chat \
      -H 'Content-Type: application/json' \
      -d '{"question": "What happens if my autopay fails?"}'
"""
import logging
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import BaseModel

from app.config import settings
from app.graph import run_query
from app.threads import store
from app.vectorstore import build_vectorstore

logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = REPO_ROOT / "frontend"

app = FastAPI(title="SunGrid Cooperative Copilot")

# Dev-friendly default. If you serve the frontend from a different origin
# during development (e.g. a live-reload static server on another port),
# CORS needs to allow it. Tighten this to specific origins in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_vectorstore = None


def _get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = build_vectorstore()
    return _vectorstore


class ChatRequest(BaseModel):
    question: str
    thread_id: Optional[str] = None


class ChatResponse(BaseModel):
    thread_id: str
    answer: str
    sources: List[str]


class MessageOut(BaseModel):
    role: str  # "human" | "ai" | "tool"
    content: str


class ThreadSummary(BaseModel):
    thread_id: str
    title: str
    created_at: str


class ThreadDetail(ThreadSummary):
    messages: List[MessageOut]


def _role_for(message) -> str:
    if isinstance(message, HumanMessage):
        return "human"
    if isinstance(message, ToolMessage):
        return "tool"
    return "ai"


def _visible_messages(messages) -> List[MessageOut]:
    """Filter out tool-call scaffolding the UI doesn't need to render."""
    out = []
    for m in messages:
        if isinstance(m, AIMessage) and not m.content:
            continue  # an AIMessage that's just a tool_call, no text yet
        if isinstance(m, ToolMessage):
            continue  # raw tool result, not conversational content
        out.append(MessageOut(role=_role_for(m), content=m.content))
    return out


@app.on_event("startup")
def _startup() -> None:
    if not settings.groq_api_key:
        logger.warning("GROQ_API_KEY is not set -- requests will fail until it is.")
    _get_vectorstore()  # warm the index on startup rather than on first request


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    if not settings.groq_api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is not configured on the server.")
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty.")

    thread = store.get_or_create(req.thread_id, first_message=req.question)

    result = run_query(_get_vectorstore(), req.question, history=list(thread.messages))

    # `run_query` returns the full state; `messages` includes the whole
    # running history plus this turn's new messages. Persist all of it so
    # the next call in this thread has full context.
    store.append(thread.thread_id, result["messages"][len(thread.messages) :])

    answer = result["messages"][-1].content
    sources = sorted({d.metadata.get("source", "unknown") for d in result["retrieved_docs"]})
    return ChatResponse(thread_id=thread.thread_id, answer=answer, sources=sources)


@app.post("/api/threads", response_model=ThreadSummary)
def create_thread() -> ThreadSummary:
    thread = store.create()
    return ThreadSummary(
        thread_id=thread.thread_id, title=thread.title, created_at=thread.created_at.isoformat()
    )


@app.get("/api/threads", response_model=List[ThreadSummary])
def list_threads() -> List[ThreadSummary]:
    return [
        ThreadSummary(thread_id=t.thread_id, title=t.title, created_at=t.created_at.isoformat())
        for t in store.list_threads()
    ]


@app.get("/api/threads/{thread_id}", response_model=ThreadDetail)
def get_thread(thread_id: str) -> ThreadDetail:
    thread = store.get(thread_id)
    if thread is None:
        raise HTTPException(status_code=404, detail="thread not found")
    return ThreadDetail(
        thread_id=thread.thread_id,
        title=thread.title,
        created_at=thread.created_at.isoformat(),
        messages=_visible_messages(thread.messages),
    )


@app.delete("/api/threads/{thread_id}")
def delete_thread(thread_id: str) -> dict:
    deleted = store.delete(thread_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="thread not found")
    return {"deleted": True}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


# Mounted last and at "/" so it only catches requests that don't match an
# API route above -- it serves the bundled frontend (index.html, JS, CSS).
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
