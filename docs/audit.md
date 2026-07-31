# Pipeline Audit (P0-2)

State of the pipeline after Phase 0 restructure. A reader should be able to trace
scrape → chunk → embed → store → retrieve → answer end-to-end from this doc alone.

## Layout

| Path | Role |
|---|---|
| `ingest/` | scrape → chunk → embed → persist (FastAPI/Pydantic docs) |
| `app/` | search, RAG flow, LLM wrapper, Streamlit UI |
| `eval/evaluate.py` | end-to-end evaluation script (legacy course harness; Phase 3 replaces it) |
| `notebooks/` | legacy course notebooks + `notebooks/ground_truth.py` (superseded by Phase 3) |
| `corpus/sources.yaml` | source manifest (P0-3 schema), `corpus/raw|parsed|chunks/` generated (gitignored) |
| `retrieval/`, `codegen/` | empty placeholders (Phases 4, 5) |
| `database/`, `grafana/`, `demo/`, `screenshots/` | course artifacts, unchanged |

## Current pipeline (one pass)

1. **Scrape** — `ingest/scrape.py`: fetch sitemap, parse `<loc>` URLs, fetch each page
   with `httpx` (User-Agent set, follows redirects, 30s timeout), return HTML + metadata.
2. **Chunk** — `ingest/chunk.py::chunk_document`: BeautifulSoup HTML; splits at
   `h1-h4` heading boundaries; heading text excluded from content; optional overlap
   (last element carries to next chunk; CLI default on); chunk fields:
   `id, title, section, content, url, doc_library`; IDs = `{library}-{slug}-{index:03d}`.
   **Type: heading-boundary splitter (not fixed-size).** Params: `overlap`, `start_index`.
3. **Embed** — `ingest/index.py::build_vector_index`: `sentence-transformers`
   **`all-MiniLM-L6-v2`** (384-dim), `model.encode(...)` in one batch.
4. **Store** — file-based, **no vector DB** (no Qdrant/PGVector):
   - `data/processed/{library}/chunks.json` (chunk metadata)
   - `data/processed/{library}/embeddings.npy` (embedding matrix, row = chunk)
   - `data/processed/{library}/minsearch.pkl` (pickled minsearch index)
   `data/` is gitignored. Orchestrated by `ingest/run.py` (`uv run python -m ingest.run --library fastapi`).
5. **Retrieve** — `app/search.py`:
   - **keyword**: `minsearch` (TF-IDF/BM25-like) over fields `title, section, content`
     with keyword fields `id, doc_library`; optional `boost_dict`.
   - **vector**: cosine similarity on query embedding vs. matrix (numpy, no ANN).
   - **hybrid**: RRF fusion of keyword + vector, `RRF_K = 60`, returns top-k.
   - Extras: `rerank` (CrossEncoder `cross-encoder/ms-marco-MiniLM-L-6-v2`),
     `rewrite_query` (abbreviation dict or LLM).
   **Currently dense + keyword hybrid; no BM25/Elasticsearch.**
6. **LLM** — `app/llm.py` (P0-4): `litellm.completion`, model from `LLM_MODEL` env
   (default `gpt-4o-mini`); returns `(text, tokens)`; `llm_stream` for streaming;
   `calculate_cost` prices `gpt-4o-mini`/`gpt-4o` only, 0.0 for other providers.
7. **RAG** — `app/rag.py`: hybrid search top-5 → prompt with citations → LLM →
   relevance judge (`app/evaluation.py::evaluate_relevance`) → result dict with
   tokens, cost, latency; LRU cache (max 128).
8. **UI** — Streamlit `app/main.py`; PostgreSQL `database/init.sql` + Grafana
   `grafana/` for monitoring; `init.py` provisions both.

## Hardcoded paths & configs (P2/P3 cleanup candidates)

| Where | Hardcode |
|---|---|
| `ingest/index.py` | output base `data/processed/{library}/` |
| `app/search.py` | `data/processed/{library}/` load path; default `library="fastapi"` |
| `eval/evaluate.py` | `data/processed/fastapi/chunks.json`, `embeddings.npy`, `data/ground_truth.jsonl` |
| `ingest/run.py` | `SITEMAP_MAP` (fastapi, pydantic) |
| `ingest/index.py`, `app/search.py` | `EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"` |
| `app/llm.py` | `MODEL_PRICING` (gpt-4o-mini, gpt-4o) |
| `app/search.py` | `RRF_K = 60` |

## Known gaps

- No structure-aware chunking for PDFs / command tables (Phase 1-2 adds this).
- No eval harness over a question set (Phase 3; legacy `eval/evaluate.py` uses
  LLM-generated ground truth on FastAPI chunks only).
- No sparse-only retrieval or BM25 (Phase 4).
- No code generation layer (Phase 5).
- Corpus is empty (`corpus/sources.yaml` has 1 entry; no raw docs downloaded yet).
