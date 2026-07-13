# SunGrid Cooperative Copilot

An agentic RAG assistant over SunGrid Cooperative's internal knowledge base
(program policies, rebates, billing, technical/installation guidance,
governance, and company updates), with a real tool call into a downstream
eligibility-check service.

See [`DESIGN.md`](DESIGN.md) for the design write-up (architecture, the
ambiguous-document handling, tradeoffs, and what I'd do next with more time).

## What's here

```
app/
  config.py       # settings (pydantic-settings, .env-driven)
  ingest.py       # unstructured-based markdown partitioning + title-aware chunking
  vectorstore.py  # Chroma index build/load (local HF embeddings)
  stub_tools.py   # provided eligibility stub, unmodified
  tools.py        # LangChain tool wrapper around the stub
  graph.py        # the LangGraph agentic RAG graph
  cli.py          # interactive terminal chat
  api.py          # optional FastAPI HTTP endpoint
docs/             # the 14 provided knowledge-base documents (unmodified)
scripts/
  build_index.py  # one-off script to (re)build the vector index
tests/            # pytest unit tests (no API key required)
eval/
  eval_queries.json  # scenario-based eval set (ambiguous doc, tool use, clarification, etc.)
  run_eval.py         # runs the eval set against the live agent
Dockerfile
requirements.txt
.env.example
```

## Install

Requires Python 3.11+.

```bash
git clone <this-repo-url>
cd sungrid-copilot
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# Recommended on CPU-only machines: install the CPU-only torch build first,
# otherwise `sentence-transformers` will pull several GB of CUDA packages
# you don't need just to embed short text chunks.
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
```

## Configure

1. Get a free Groq API key: https://console.groq.com/keys
2. Copy the env template and fill it in:

```bash
cp .env.example .env
# edit .env, set GROQ_API_KEY=your_key_here
```

Everything else in `.env.example` is optional — sensible defaults are set in
`app/config.py`. Notably, **embeddings run locally** via
`sentence-transformers/all-MiniLM-L6-v2` (downloaded once from Hugging Face on
first run), so only the LLM calls hit an external API.

## Run

### Build the index (first time, or after editing `docs/`)

```bash
python -m scripts.build_index
```

This creates a persisted Chroma collection under `.chroma/`.

### Chat via the CLI

```bash
python -m app.cli
```

```
you> What happens if my autopay fails?
copilot> ...answer with citations like [1], [2]...

you> Is a household at ZIP 94101, income $80,000, 5kW system, approved
     installer, eligible for the rooftop rebate and how much?
copilot> ...calls the eligibility tool and reports the result...
```

### Or run the HTTP API

```bash
uvicorn app.api:app --reload
```

```bash
curl -X POST localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"question": "What happens if my autopay fails?"}'
```

### Docker

```bash
docker build -t sungrid-copilot .
docker run -p 8000:8000 --env-file .env sungrid-copilot
```

## Tests

Unit tests cover ingestion/chunking, the eligibility tool wrapper, and the
graph's routing logic. **None of them require a Groq API key** — they test
deterministic code paths and use fake messages/state for the routing tests.

```bash
pytest
```

## Eval

A small scenario-based eval set lives in `eval/eval_queries.json`, covering:
the ambiguous document, cross-referenced policies, eligibility-tool
invocation (with and without complete inputs), a document that explicitly
warns not to be relied on, and an out-of-scope question. This **does**
require a live `GROQ_API_KEY` since it runs the real agent end to end:

```bash
python -m eval.run_eval
```

## Notes on `stub_tools.py`

Left byte-for-byte as provided. `app/tools.py` wraps
`check_rebate_eligibility` with a LangChain tool schema so the LLM can call
it directly; no eligibility logic is reimplemented anywhere else.
