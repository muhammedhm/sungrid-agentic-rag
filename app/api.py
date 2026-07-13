"""
Minimal FastAPI service around the agent, for anyone who wants an HTTP
endpoint instead of the CLI.

    uvicorn app.api:app --reload

    curl -X POST localhost:8000/chat \\
      -H 'Content-Type: application/json' \\
      -d '{"question": "What happens if my autopay fails?"}'
"""
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.config import settings
from app.graph import run_query
from app.vectorstore import build_vectorstore

logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
logger = logging.getLogger(__name__)

app = FastAPI(title="SunGrid Cooperative Copilot")

_vectorstore = None


def _get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = build_vectorstore()
    return _vectorstore


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]


@app.on_event("startup")
def _startup() -> None:
    if not settings.groq_api_key:
        logger.warning("GROQ_API_KEY is not set -- requests will fail until it is.")
    _get_vectorstore()  # warm the index on startup rather than on first request


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    if not settings.groq_api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is not configured on the server.")
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty.")

    result = run_query(_get_vectorstore(), req.question)
    answer = result["messages"][-1].content
    sources = sorted({d.metadata.get("source", "unknown") for d in result["retrieved_docs"]})
    return ChatResponse(answer=answer, sources=sources)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
