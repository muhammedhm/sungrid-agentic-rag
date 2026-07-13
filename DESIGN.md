# Design Write-Up

## 1. What "agentic" means here

I avoided building a single fixed retrieve → generate chain. The graph
(`app/graph.py`, built with LangGraph) has three points where control flow
actually branches based on a model decision, not just a fixed pipeline:

1. **Retrieval quality gate.** After retrieving, a small structured-output
   call grades whether the retrieved chunks can actually answer the
   question. If not, the query gets rewritten (expand acronyms, resolve
   ambiguity) and retried, up to `max_retrieval_retries`. This catches the
   common RAG failure where the first phrasing just doesn't embed near the
   right chunk.
2. **Tool-use decision.** The main response node has `check_rebate_eligibility`
   bound as a tool. The model decides whether a question needs a live
   determination (an eligibility/rebate-amount question) versus something
   answerable from policy text alone, and only calls the tool when it has
   all four required inputs.
3. **Clarify vs. guess.** If an eligibility question is missing inputs
   (e.g. no income given), the system prompt explicitly instructs the model
   to ask for them rather than assume a default. This is enforced through
   instruction + the tool's docstring rather than a separate code path,
   which is a deliberate simplicity/robustness tradeoff (see §5).

Retrieval → grade → (retry loop) → decide/answer/tool-call → (tool) → finalize.

## 2. Ingestion: why `unstructured` + title-based chunking

Each doc is short (13–29 lines) but organized into clean `##` sections that
already represent the right retrieval unit — e.g. "What happens if I miss a
payment?" is one coherent idea, and splitting it mid-sentence with a
fixed-size window would be strictly worse. `unstructured.partition.md` parses
each file into typed elements (Title, NarrativeText, Table, ListItem), and
`chunk_by_title` groups elements under their nearest heading, combining
short adjacent sections and splitting long ones. This gave one-chunk-per-section
chunks for nearly every doc with no manual tuning, and it correctly extracts
the billing-tier table in `03_billing_faqs.md` as a distinct `Table` element
rather than mangling it into prose.

A markdown-header-aware splitter is not novel, but starting from
`unstructured`'s element classification rather than a naive text
splitter avoids two classes of bugs on documents like these: page notes,
metadata, or unusual short docs getting fixed-window boundaries that cut a
sentence in half, and the table getting split from its own header.

## 3. Handling the ambiguous document (`06_...`)

Doc 06 is explicit about its own ambiguity in its text: *"it does not sit
cleanly under either the Incentive & Rebate Programs document or the
Billing & Account FAQs document alone."* Two ways I considered handling this:

- **Force a single category** (e.g. classify it as "billing" because it's
  about billing adjustments). Rejected — this is exactly the kind of
  categorization the doc itself is warning against, and it would make the
  system wrong in the same way a human filing this into one drawer would
  be wrong.
- **Full-corpus semantic retrieval, multi-label metadata for citation.**
  What I built: retrieval is never gated by category. Every query does a
  plain similarity search across the whole indexed corpus. Category tags
  (`app/ingest.py::CATEGORY_OVERRIDES`) exist only as metadata attached to
  each chunk for citation/traceability and possible future filtering — they
  never narrow *what gets searched*. Doc 06 is tagged with **both**
  `incentive_rebate_programs` and `billing_faqs`, and its `ambiguous` flag is
  set to `True` in metadata for downstream tooling (e.g. an eval check or a
  UI badge) without affecting retrieval at all.

This means a query about "my rebate got reversed and now I have a weird
charge" surfaces doc 06 on its semantic merits regardless of which
"team" a human indexer might have filed it under, and the system prompt
explicitly instructs the model to say when a retrieved answer spans more
than one policy area rather than silently picking one (see
`eval/eval_queries.json::ambiguous_doc_1/2`).

The same principle — semantic search first, category metadata second —
also naturally handles the *other* cross-references baked into this
corpus (installer certification lapses affecting rebate eligibility,
autopay failure affecting billing-adjustment collection, grievance vs.
appeals process), none of which are called out as "the ambiguous one"
but are just as real. I didn't want a design that only worked for the one
document the assessment flagged.

## 4. The eligibility tool

`stub_tools.check_rebate_eligibility` is used exactly as provided — no
signature change, no reimplemented logic. `app/tools.py` adds a LangChain
`@tool`-decorated wrapper with a Pydantic arg schema so the LLM can call it
directly via Groq's tool-calling support. The tool's docstring is where the
"ask, don't guess" instruction is anchored, reinforced in the system prompt.

I chose direct LLM tool-calling over a rule-based intent classifier ("is
this an eligibility question? extract slots") because the four required
inputs are heterogeneous (a ZIP, a dollar figure, a kW figure, a boolean)
and a general-purpose LLM is already good at extracting them from natural
phrasing ("I make about 80k a year" → `annual_income_usd: 80000`) without
brittle regex.

## 5. Production considerations included

- **Config** via `pydantic-settings` (`app/config.py`), `.env`-driven, no
  secrets in the repo (`.env.example` only).
- **Local embeddings** (`sentence-transformers/all-MiniLM-L6-v2`) so the
  only paid/rate-limited external call is the Groq LLM itself — index
  build and retrieval work offline once the embedding model is cached.
- **Tests that don't require an API key.** `tests/test_ingest.py` and
  `tests/test_tools.py` exercise deterministic code paths directly.
  `tests/test_graph_routing.py` tests the conditional-edge functions and
  the tool-execution node with hand-built `AIMessage`/state objects,
  so CI can run the whole suite with a dummy `GROQ_API_KEY` and zero live
  calls (see `.github`-free but CI-friendly command in the README).
- **A scenario eval set** (`eval/eval_queries.json` +
  `eval/run_eval.py`) that specifically targets the ambiguous doc, the
  tool-call path (with and without complete inputs), a doc that explicitly
  disclaims itself (`05_company_updates.md`'s unreleased tier), and an
  out-of-scope question — because "does it retrieve reasonably" isn't the
  same as "does it handle the traps this corpus was built with."
- **Bounded loops.** `max_retrieval_retries` and a `recursion_limit` on the
  compiled graph prevent a pathological rewrite loop from running forever.
- **Docker** image with the index built at build time for fast cold starts.
- **Both a CLI and a minimal FastAPI endpoint**, since a take-home reviewer
  might want to poke at either.

## 6. What I'd do next with more time

- **Re-ranking.** Right now retrieval is single-stage dense similarity
  search (`k=4`). A cross-encoder re-rank step over a larger initial `k`
  would likely help on borderline cases like doc 06 competing against its
  two "sibling" docs.
- **A real eval harness (RAGAS or similar)** with faithfulness/answer-relevance
  scoring rather than the current assertion-based checks, plus a larger,
  versioned query set with human-labeled expected answers.
- **Persisted conversation memory** beyond a single process's in-memory
  history — the CLI/session history is passed through `run_query`'s
  `history` param but isn't persisted across restarts or across concurrent
  API users; a real deployment would need per-session state (Redis, a
  DB-backed checkpointer via `langgraph.checkpoint`) instead.
- **Table-aware chunk formatting.** `unstructured` extracts `03_billing_faqs.md`'s
  pricing table with a clean `text_as_html` representation
  (`element.metadata.text_as_html`), which I currently flatten to plain text
  for embedding/generation. Re-rendering it as markdown in the chunk shown
  to the LLM would likely improve answer quality on tier-pricing questions.
- **Observability.** Structured logging exists but there's no tracing
  (e.g. LangSmith) wired in to inspect retrieval quality and tool-call
  decisions in production — useful given how much of "agentic" behavior
  here is judgment calls a grader (or human op) would want to audit.
- **Auth/rate limiting on the FastAPI endpoint** — currently unauthenticated,
  which is fine for a local take-home demo but not for anything real.
