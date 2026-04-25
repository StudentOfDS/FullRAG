# FullRAG

Production-minded Retrieval-Augmented Generation platform with clear modular layers:

- ingestion
- chunking
- indexing (sparse + dense)
- retrieval (3-stage hybrid)
- generation (multi-provider router)
- semantic caching
- guardrails
- orchestration
- monitoring
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

## Configuration and secrets

- Runtime config is centralized in `config.json`.
- Secrets are injected through environment variables (`.env.example`).

## Architecture notes

- Deterministic IDs are generated using content hashing.
- Tombstoning is supported before async cleanup jobs.
- Hybrid retrieval combines BM25 and dense similarity via min-max normalization.
- Cross-encoder rerank execution path is non-blocking through `asyncio.to_thread()`.
