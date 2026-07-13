"""
Vector store construction and loading.

Embeddings are local (sentence-transformers, called directly -- see
`LocalSentenceTransformerEmbeddings` below), so building and querying the
index never requires the Groq API key or any network call beyond the
one-time model download from Hugging Face on first run.

Note: we implement a tiny `Embeddings` class here instead of depending on
`langchain-huggingface`, because that package currently caps its
`langchain-core` requirement below 1.0, which conflicts with
`langchain-groq`/`langgraph`'s `langchain-core>=1.4` requirement. This
avoids an unresolvable dependency conflict while adding no real extra code.
"""
import logging
from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

from app.config import settings
from app.ingest import load_and_chunk

logger = logging.getLogger(__name__)

_embeddings = None


class LocalSentenceTransformerEmbeddings(Embeddings):
    """Minimal LangChain-compatible wrapper around a local sentence-transformers model."""

    def __init__(self, model_name: str):
        self._model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._model.encode(list(texts), convert_to_numpy=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self._model.encode([text], convert_to_numpy=True)[0].tolist()


def get_embeddings() -> LocalSentenceTransformerEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = LocalSentenceTransformerEmbeddings(settings.embedding_model)
    return _embeddings


def build_vectorstore(force_rebuild: bool = False) -> Chroma:
    """
    Load an existing persisted Chroma collection, or build one from
    `docs_dir` if it doesn't exist yet (or `force_rebuild=True`).
    """
    persist_dir = Path(settings.chroma_persist_dir)
    embeddings = get_embeddings()

    already_built = persist_dir.exists() and any(persist_dir.iterdir())

    if already_built and not force_rebuild:
        logger.info("Loading existing Chroma index from %s", persist_dir)
        return Chroma(
            persist_directory=str(persist_dir),
            embedding_function=embeddings,
            collection_name=settings.collection_name,
        )

    logger.info("Building Chroma index from %s", settings.docs_dir)
    docs = load_and_chunk(
        settings.docs_dir,
        max_characters=settings.max_chunk_characters,
        new_after_n_chars=settings.new_chunk_after_characters,
        combine_text_under_n_chars=settings.combine_chunk_under_characters,
    )
    if not docs:
        raise RuntimeError(f"No documents found in {settings.docs_dir}")

    vs = Chroma.from_documents(
        docs,
        embedding=embeddings,
        persist_directory=str(persist_dir),
        collection_name=settings.collection_name,
    )
    logger.info("Indexed %d chunks", len(docs))
    return vs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_vectorstore(force_rebuild=True)
