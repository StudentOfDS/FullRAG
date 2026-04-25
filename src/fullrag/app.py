from __future__ import annotations

from fastapi import FastAPI

from fullrag.api.schemas import IngestRequest, QueryRequest
from fullrag.core.config import ConfigLoader
from fullrag.orchestration.service import FullRAGOrchestrator
from fullrag.utils.logging import setup_logging


def create_app() -> FastAPI:
    config = ConfigLoader().load()
    setup_logging(config["app"]["log_level"])

    orchestrator = FullRAGOrchestrator(config)

    app = FastAPI(title="FullRAG")

    @app.post("/ingest")
    def ingest(payload: IngestRequest):
        return orchestrator.ingest_paths(payload.paths)

    @app.post("/query")
    async def query(payload: QueryRequest):
        return await orchestrator.query(payload.query)

    @app.get("/metrics")
    def metrics():
        return orchestrator.metrics.snapshot()

    return app
