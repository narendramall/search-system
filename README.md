# 🚗 Car Brochure Search System

An intelligent chatbot that ingests car brochure PDFs and provides expert car recommendations, detailed specifications, and side-by-side comparisons — powered by **Elasticsearch**, **Pinecone**, and **Google Gemini**.

---

## Architecture

```
User Query → Chat Engine → Search Router
                              ├── Phase 1: Elasticsearch (BM25 keyword search)
                              └── Phase 2: Pinecone (semantic vector search)
                                    ↓
                              Reranker (Reciprocal Rank Fusion)
                                    ↓
                              Gemini LLM → Natural Language Response
```

> Full architecture details: [docs/architecture.md](docs/architecture.md)

---

## Project Structure

```
search-system/
├── app.py                          # FastAPI entry point
├── pyproject.toml                  # Poetry project & dependencies
├── .env                            # API keys & configuration
├── config/
│   └── __init__.py                 # Centralized settings (Pydantic)
├── data/                           # Drop your car brochure PDFs here
├── src/
│   ├── ingestion/
│   │   ├── pdf_parser.py           # PDF text extraction (LlamaParse + PyMuPDF fallback)
│   │   ├── chunker.py              # Intelligent text chunking with overlap
│   │   ├── es_indexer.py           # Elasticsearch bulk indexing
│   │   └── pinecone_indexer.py     # Pinecone vector indexing (Phase 2)
│   ├── search/
│   │   ├── es_search.py            # Elasticsearch BM25 search with boosting
│   │   ├── vector_search.py        # Pinecone semantic search (Phase 2)
│   │   └── reranker.py             # Reciprocal Rank Fusion (Phase 2)
│   ├── llm/
│   │   └── gemini_client.py        # Gemini LLM for answers & embeddings
│   └── chat/
│       └── engine.py               # Chat orchestration engine
├── scripts/
│   ├── ingest.py                   # CLI: Ingest PDFs into search backends
│   └── chat_cli.py                 # CLI: Interactive terminal chatbot
└── docs/
    └── architecture.md             # Detailed architecture documentation
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- Elasticsearch 8.x running locally (or a cloud instance)
- Gemini API key (already in `.env`)

### 1. Install Dependencies

```bash
# Install Poetry (if not already installed)
curl -sSL https://install.python-poetry.org | python3 -

# Install all dependencies
poetry install

# Activate the virtual environment
poetry shell
```

### 2. Start Elasticsearch

```bash
# Using Docker (quickest way)
docker run -d --name elasticsearch \
  -p 9200:9200 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  docker.elastic.co/elasticsearch/elasticsearch:8.15.0
```

### 3. Add Car Brochures

Drop your PDF brochure files into the `data/` folder:

```
data/
├── Hyundai_Creta_2024.pdf
├── Tata_Nexon_2024.pdf
├── Maruti_Brezza_2024.pdf
└── ...
```

**Naming convention** (recommended): `Brand_Model_Year.pdf` — the system auto-extracts car brand and model from filenames.

### 4. Ingest PDFs

```bash
# Phase 1: Index into Elasticsearch
python scripts/ingest.py

# With fresh re-index
python scripts/ingest.py --fresh
```

### 5. Start Chatting

**Option A: Terminal CLI** (recommended for testing)
```bash
python scripts/chat_cli.py
```

**Option B: REST API**
```bash
python app.py
# or
uvicorn app:app --reload --port 8000
```

Then send queries:
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the top speed of Hyundai Creta?"}'
```

---

## Phase 1: Elasticsearch Only (Current)

**What works:**
- PDF ingestion with table extraction
- Intelligent text chunking (512 tokens, 50-token overlap)
- BM25 keyword search with field boosting (model^3, brand^2, section^2)
- Specialized spec queries (boosts specification/engine/performance sections)
- Fuzzy matching for typo tolerance
- Gemini LLM generates natural language answers
- Conversation history for multi-turn chats
- Query intent classification (specification / comparison / recommendation / general)

**Example queries:**
- "What is the ground clearance of Tata Nexon?"
- "Compare Creta and Brezza fuel efficiency"
- "Which car has the best boot space?"
- "Recommend a car under 15 lakhs with sunroof"

---

## Phase 2: Hybrid Search (Elasticsearch + Pinecone)

### Setup

1. Get a [Pinecone API key](https://www.pinecone.io/) and add to `.env`:
   ```
   PINECONE_API_KEY=your_pinecone_key
   SEARCH_MODE=hybrid
   ```

2. Re-ingest with hybrid mode:
   ```bash
   python scripts/ingest.py --mode hybrid --fresh
   ```

**What Phase 2 adds:**
- Semantic vector search via Pinecone (understands meaning, not just keywords)
- Gemini gemini-embedding-2 for 768-dim embeddings
- Reciprocal Rank Fusion (RRF) merges keyword + semantic results
- Better handling of natural language queries ("which car feels premium?")
- Near-duplicate deduplication

---

## Configuration

All settings in `.env`:

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | — | Google Gemini API key |
| `ELASTICSEARCH_URL` | `http://localhost:9200` | ES cluster URL |
| `ELASTICSEARCH_API_KEY` | — | ES auth key (optional for local) |
| `ES_INDEX_NAME` | `car_brochures` | ES index name |
| `PINECONE_API_KEY` | — | Pinecone API key (Phase 2) |
| `PINECONE_INDEX_NAME` | `car-brochures` | Pinecone index name |
| `SEARCH_MODE` | `elasticsearch` | `elasticsearch` / `pinecone` / `hybrid` |
| `CHUNK_SIZE` | `512` | Tokens per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `TOP_K` | `10` | Results per search backend |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/chat` | Send a question, get an answer |
| `POST` | `/chat/reset` | Clear conversation history |
| `GET` | `/health` | System health check |
| `GET` | `/stats` | Index statistics |

---

## How It Works

1. **Ingestion**: PDFs are parsed → chunked (512 tokens, overlapping) → indexed into ES (and Pinecone in Phase 2)
2. **Query**: User asks a question → intent is classified → routed to search backend(s)
3. **Search**: ES does BM25 keyword matching; Pinecone does cosine similarity on embeddings
4. **Rerank**: In hybrid mode, Reciprocal Rank Fusion merges both result lists
5. **Generate**: Top chunks are sent to Gemini as context → generates a conversational answer
6. **Respond**: Answer + source references returned to user
