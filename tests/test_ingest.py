from app.config import settings
from app.ingest import categories_list, load_and_chunk


def test_loads_all_docs():
    chunks = load_and_chunk(settings.docs_dir)
    sources = {c.metadata["source"] for c in chunks}
    assert len(sources) == 14, f"expected 14 source docs, found {len(sources)}: {sources}"


def test_no_empty_chunks():
    chunks = load_and_chunk(settings.docs_dir)
    for c in chunks:
        assert c.page_content.strip(), f"empty chunk from {c.metadata['source']}"


def test_ambiguous_doc_gets_two_categories():
    chunks = load_and_chunk(settings.docs_dir)
    doc_06_chunks = [c for c in chunks if c.metadata["source"] == "06_ambiguous_rebate_billing_adjustments.md"]
    assert doc_06_chunks, "doc 06 produced no chunks"
    for c in doc_06_chunks:
        cats = categories_list(c)
        assert set(cats) == {"incentive_rebate_programs", "billing_faqs"}
        assert c.metadata["ambiguous"] is True


def test_non_ambiguous_doc_gets_single_category():
    chunks = load_and_chunk(settings.docs_dir)
    doc_01_chunks = [c for c in chunks if c.metadata["source"] == "01_program_policies.md"]
    assert doc_01_chunks
    for c in doc_01_chunks:
        assert categories_list(c) == ["program_policies"]
        assert c.metadata["ambiguous"] is False


def test_chunks_respect_reasonable_size_bounds():
    chunks = load_and_chunk(settings.docs_dir, max_characters=1200)
    # generous upper bound check; unstructured's chunker can slightly exceed
    # max_characters when a single element is longer than the limit
    for c in chunks:
        assert len(c.page_content) < 2500, f"suspiciously large chunk from {c.metadata['source']}"
