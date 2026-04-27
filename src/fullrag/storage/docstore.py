from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from fullrag.models.entities import Chunk


class DocumentStore:
    def __init__(self, db_path: str = "./data/docstore.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._init_db()

    def _init_db(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                text TEXT NOT NULL,
                abstract TEXT NOT NULL,
                section_path TEXT NOT NULL,
                source_file TEXT NOT NULL,
                page INTEGER,
                tombstoned INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(document_id)")
        self._conn.commit()

    def upsert_chunks(self, chunks: list[Chunk]) -> None:
        rows = [
            (
                c.chunk_id,
                c.document_id,
                c.text,
                c.abstract,
                json.dumps(c.section_path),
                c.source_file,
                c.page,
                int(c.tombstoned),
            )
            for c in chunks
        ]
        self._conn.executemany(
            """
            INSERT INTO chunks(chunk_id, document_id, text, abstract, section_path, source_file, page, tombstoned)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(chunk_id) DO UPDATE SET
              document_id=excluded.document_id,
              text=excluded.text,
              abstract=excluded.abstract,
              section_path=excluded.section_path,
              source_file=excluded.source_file,
              page=excluded.page,
              tombstoned=excluded.tombstoned
            """,
            rows,
        )
        self._conn.commit()

    def get(self, chunk_id: str) -> Chunk | None:
        row = self._conn.execute("SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)).fetchone()
        return self._row_to_chunk(row) if row else None

    def get_many(self, chunk_ids: list[str]) -> list[Chunk]:
        return [chunk for cid in chunk_ids if (chunk := self.get(cid))]

    def mark_tombstoned(self, document_id: str) -> int:
        cursor = self._conn.execute("UPDATE chunks SET tombstoned = 1 WHERE document_id = ? AND tombstoned = 0", (document_id,))
        self._conn.commit()
        return cursor.rowcount

    def active_chunks(self) -> list[Chunk]:
        rows = self._conn.execute("SELECT * FROM chunks WHERE tombstoned = 0").fetchall()
        return [self._row_to_chunk(row) for row in rows]

    @staticmethod
    def _row_to_chunk(row) -> Chunk:
        return Chunk(
            chunk_id=row[0],
            document_id=row[1],
            text=row[2],
            abstract=row[3],
            section_path=json.loads(row[4]),
            source_file=row[5],
            page=row[6],
            tombstoned=bool(row[7]),
        )
