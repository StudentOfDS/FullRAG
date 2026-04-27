# FullRAG

Production-ready Retrieval-Augmented Generation platform with modular architecture:

- ingestion (text, markdown, PDF, codebase)
- chunking (adaptive overlapping windows)
- indexing (BM25 + embedding-powered dense retrieval)
- retrieval (3-stage hybrid + optional cross-encoder reranker)
- generation (multi-provider routed LLM completion)
- semantic caching (persistent SQLite-backed cache)
- guardrails (prompt-injection and low-signal query blocking)
- orchestration and tombstoning
- monitoring (JSON snapshot + Prometheus text exposition)
- API + tests

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
uvicorn fullrag.main:app --reload
```

## API

- `POST /ingest` with `{ "paths": ["/path/to/file.txt"] }`
- `POST /query` with `{ "query": "your question" }`
- `GET /metrics`
- `GET /metrics/prometheus`

## Deployment

- Docker image: `Dockerfile`
- Kubernetes manifest: `deploy/k8s/deployment.yaml`
- CI workflow: `.github/workflows/ci.yml`

## Configuration and secrets

Runtime config is centralized in `config.json` and secret values are loaded from environment variables:

- `OPENAI_API_KEY`
- `PINECONE_API_KEY`
- `ANTHROPIC_API_KEY`
- `GEMINI_API_KEY`
- `DEEPSEEK_API_KEY`
- `HUGGINGFACE_API_KEY`
- `OLLAMA_BASE_URL`

## Architecture notes

- Deterministic IDs are generated using content hashing.
- Tombstoning is supported before async cleanup jobs.
- Hybrid retrieval combines BM25 and dense similarity via min-max normalization.
- Cross-encoder rerank execution path is non-blocking through `asyncio.to_thread()`.
- Dense retrieval is embedding-client based with normalized vectors and retries.
- PDF ingestion uses binary parsing when `pypdf` is available and automatically falls back safely.
