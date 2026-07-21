"""
In-memory chat-thread store.

Keeps a LangChain message history per thread_id so `/api/chat` can pass
prior turns into `run_query` for history-aware conversation (e.g.
answering a follow-up "ZIP 94101, $80k, approved installer" after the
agent asked a clarifying question about a missing input).

This is intentionally simple for a take-home deliverable: a single
process's in-memory dict, guarded by a lock since FastAPI can run
handlers concurrently. It does NOT survive a restart.

For a real deployment, swap `ThreadStore` for a persisted backend
(Postgres, Redis, or a `langgraph.checkpoint` implementation) behind the
same interface -- nothing else in the app would need to change.
"""
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

from langchain_core.messages import BaseMessage


@dataclass
class Thread:
    thread_id: str
    title: str
    created_at: datetime
    messages: List[BaseMessage] = field(default_factory=list)


class ThreadStore:
    def __init__(self) -> None:
        self._threads: dict[str, Thread] = {}
        self._lock = threading.Lock()

    def create(self, first_message: Optional[str] = None) -> Thread:
        thread_id = uuid.uuid4().hex[:12]
        title = (first_message or "New chat").strip()[:60]
        thread = Thread(thread_id=thread_id, title=title, created_at=datetime.now(timezone.utc))
        with self._lock:
            self._threads[thread_id] = thread
        return thread

    def get(self, thread_id: str) -> Optional[Thread]:
        with self._lock:
            return self._threads.get(thread_id)

    def get_or_create(self, thread_id: Optional[str], first_message: Optional[str] = None) -> Thread:
        if thread_id:
            existing = self.get(thread_id)
            if existing is not None:
                return existing
        return self.create(first_message=first_message)

    def append(self, thread_id: str, messages: List[BaseMessage]) -> None:
        with self._lock:
            thread = self._threads.get(thread_id)
            if thread is None:
                raise KeyError(thread_id)
            thread.messages.extend(messages)

    def list_threads(self) -> List[Thread]:
        with self._lock:
            return sorted(self._threads.values(), key=lambda t: t.created_at, reverse=True)

    def delete(self, thread_id: str) -> bool:
        with self._lock:
            return self._threads.pop(thread_id, None) is not None


# Process-wide singleton -- see module docstring re: persistence limitations.
store = ThreadStore()
