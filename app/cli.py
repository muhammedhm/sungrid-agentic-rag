"""
Interactive CLI for the SunGrid Cooperative Copilot.

    python -m app.cli

Type a question, get an answer with citations. Conversation history is
kept for the session so follow-up answers to a clarifying question (e.g.
supplying a ZIP code after being asked for one) work naturally.
"""
import logging
import sys

from app.config import settings
from app.graph import run_query
from app.vectorstore import build_vectorstore


def main() -> None:
    logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))

    if not settings.groq_api_key:
        print(
            "ERROR: GROQ_API_KEY is not set. Copy .env.example to .env and fill it in, "
            "or export GROQ_API_KEY in your shell.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Building / loading index...")
    vectorstore = build_vectorstore()

    print("SunGrid Cooperative Copilot. Ctrl-C or 'exit' to quit.\n")
    history = []
    while True:
        try:
            question = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break

        result = run_query(vectorstore, question, history=history)
        answer = result["messages"][-1].content
        print(f"\ncopilot> {answer}\n")

        history = result["messages"]


if __name__ == "__main__":
    main()
