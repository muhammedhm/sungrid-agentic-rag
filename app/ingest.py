"""
Ingestion pipeline.

Uses `unstructured` to partition each markdown doc into semantically
meaningful elements (Title, NarrativeText, Table, ListItem, ...), then
`chunk_by_title` to group elements into chunks that respect the document's
own `##` section boundaries rather than an arbitrary character window.

Category tagging
-----------------
Every doc is tagged with one or more `categories`, derived from its
filename. This is metadata for citation/traceability and optional
filtering -- it is deliberately NOT used to gate retrieval (see
DESIGN.md for why). One doc, `06_ambiguous_rebate_billing_adjustments.md`,
is explicitly cross-cutting: it says so in its own text ("does not sit
cleanly under either... document alone"), so it is tagged with BOTH of
the categories it bridges instead of being forced into one.
"""
import re
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from unstructured.chunking.title import chunk_by_title
from unstructured.partition.md import partition_md

# Explicit overrides for docs whose category can't be inferred cleanly
# from the filename alone. Add to this dict rather than fighting the
# filename-slug heuristic below.
CATEGORY_OVERRIDES: dict[str, list[str]] = {
    "06_ambiguous_rebate_billing_adjustments.md": [
        "incentive_rebate_programs",
        "billing_faqs",
    ],
}

AMBIGUOUS_DOCS = set(CATEGORY_OVERRIDES.keys())


def _slug_from_filename(filename: str) -> str:
    """`02_incentive_rebate_programs.md` -> `incentive_rebate_programs`"""
    stem = Path(filename).stem
    return re.sub(r"^\d+_", "", stem)


def _categories_for(filename: str) -> List[str]:
    return CATEGORY_OVERRIDES.get(filename, [_slug_from_filename(filename)])


def load_and_chunk(
    docs_dir: str,
    max_characters: int = 1200,
    new_after_n_chars: int = 1000,
    combine_text_under_n_chars: int = 200,
) -> List[Document]:
    """Partition + chunk every .md file in `docs_dir` into LangChain Documents."""
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        raise FileNotFoundError(f"docs_dir does not exist: {docs_dir}")

    all_chunks: List[Document] = []

    for path in sorted(docs_path.glob("*.md")):
        elements = partition_md(filename=str(path))
        chunks = chunk_by_title(
            elements,
            max_characters=max_characters,
            new_after_n_chars=new_after_n_chars,
            combine_text_under_n_chars=combine_text_under_n_chars,
        )

        categories = _categories_for(path.name)
        ambiguous = path.name in AMBIGUOUS_DOCS

        for i, chunk in enumerate(chunks):
            text = str(chunk).strip()
            if not text:
                continue
            all_chunks.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": path.name,
                        # Chroma metadata values must be scalars, not lists ->
                        # store as a comma-joined string, parse back with
                        # `categories_list()` below when needed.
                        "categories": ",".join(categories),
                        "ambiguous": ambiguous,
                        "chunk_index": i,
                    },
                )
            )

    return all_chunks


def categories_list(doc: Document) -> List[str]:
    return doc.metadata.get("categories", "").split(",")


if __name__ == "__main__":
    from app.config import settings

    chunks = load_and_chunk(settings.docs_dir)
    print(f"Loaded {len(chunks)} chunks from {settings.docs_dir}")
    for c in chunks[:3]:
        print("---")
        print(c.metadata)
        print(c.page_content[:150])
