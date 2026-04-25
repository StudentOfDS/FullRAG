from __future__ import annotations

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    paths: list[str] = Field(default_factory=list)


class QueryRequest(BaseModel):
    query: str
