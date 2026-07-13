"""Run once (or after editing docs/) to (re)build the persisted Chroma index.

    python -m scripts.build_index
"""
import logging

from app.vectorstore import build_vectorstore

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_vectorstore(force_rebuild=True)
    print("Index built.")
