"""LanceDB storage for vector embeddings and full-text search."""

from pathlib import Path
from typing import Optional

import lancedb
import pyarrow as pa

from ..models import Chunk
from ..config import Config, EmbeddingConfig
from ..logging_config import logger
from ..exceptions import LoadError


class LanceDBStore:
    """LanceDB storage for vector embeddings with hybrid search support."""

    TABLE_NAME = "chunks"

    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.db_path = str(self.config.paths.lancedb_dir)
        self._db = None
        self._table = None

    def _get_db(self) -> lancedb.DBConnection:
        """Get or create database connection."""
        if self._db is None:
            self._db = lancedb.connect(self.db_path)
            logger.info(f"Connected to LanceDB: {self.db_path}")
        return self._db

    def _get_schema(self) -> pa.Schema:
        """Get PyArrow schema for chunks table."""
        embedding_dim = self.config.embedding.dimension

        return pa.schema([
            pa.field("id", pa.string()),
            pa.field("document_id", pa.string()),
            pa.field("document_title", pa.string()),
            pa.field("section_h1", pa.string()),
            pa.field("section_h2", pa.string()),
            pa.field("text", pa.string()),
            pa.field("page_numbers", pa.list_(pa.int32())),
            pa.field("chunk_index", pa.int32()),
            pa.field("vector", pa.list_(pa.float32(), embedding_dim)),
        ])

    def _ensure_table(self):
        """Ensure the chunks table exists."""
        db = self._get_db()

        if self.TABLE_NAME not in db.table_names():
            # Create empty table with schema
            schema = self._get_schema()
            self._table = db.create_table(self.TABLE_NAME, schema=schema)
            logger.info(f"Created LanceDB table: {self.TABLE_NAME}")
        else:
            self._table = db.open_table(self.TABLE_NAME)

        return self._table

    def insert_chunks(self, chunks: list[Chunk]):
        """
        Insert chunks with embeddings into LanceDB.

        Args:
            chunks: List of Chunk objects with embeddings
        """
        if not chunks:
            return

        table = self._ensure_table()

        # Convert chunks to records
        records = []
        for chunk in chunks:
            if chunk.embedding is None:
                logger.warning(
                    f"Chunk {chunk.id} has no embedding, skipping",
                    extra={"document_id": chunk.document_id},
                )
                continue

            records.append({
                "id": chunk.id,
                "document_id": chunk.document_id,
                "document_title": chunk.document_title,
                "section_h1": chunk.section_h1 or "",
                "section_h2": chunk.section_h2 or "",
                "text": chunk.text,
                "page_numbers": chunk.page_numbers,
                "chunk_index": chunk.chunk_index,
                "vector": chunk.embedding,
            })

        if records:
            table.add(records)
            logger.info(
                f"Inserted {len(records)} chunks into LanceDB",
                extra={"chunk_count": len(records)},
            )

    def create_fts_index(self):
        """Create full-text search index for hybrid search."""
        table = self._ensure_table()

        try:
            table.create_fts_index(["document_title", "text"], replace=True)
            logger.info("Created FTS index on document_title and text")
        except Exception as e:
            logger.warning(f"FTS index creation failed (may already exist): {e}")

    def delete_by_document(self, document_id: str):
        """Delete all chunks for a document."""
        table = self._ensure_table()

        try:
            table.delete(f"document_id = '{document_id}'")
            logger.info(
                f"Deleted chunks for document from LanceDB",
                extra={"document_id": document_id},
            )
        except Exception as e:
            logger.warning(f"Delete failed (table may be empty): {e}")

    def vector_search(
        self,
        query_vector: list[float],
        limit: int = 10,
        filter_expr: Optional[str] = None,
    ) -> list[dict]:
        """
        Perform vector similarity search.

        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results
            filter_expr: Optional SQL filter expression

        Returns:
            List of matching records with scores
        """
        table = self._ensure_table()

        query = table.search(query_vector).limit(limit)

        if filter_expr:
            query = query.where(filter_expr)

        results = query.to_list()
        return results

    def hybrid_search(
        self,
        query_text: str,
        query_vector: list[float],
        limit: int = 10,
        filter_expr: Optional[str] = None,
    ) -> list[dict]:
        """
        Perform hybrid search (vector + full-text).

        Args:
            query_text: Text query for BM25
            query_vector: Query embedding vector
            limit: Maximum number of results
            filter_expr: Optional SQL filter expression

        Returns:
            List of matching records with combined scores
        """
        table = self._ensure_table()

        try:
            query = (
                table.search(query_vector, query_type="hybrid")
                .limit(limit)
            )

            if filter_expr:
                query = query.where(filter_expr)

            results = query.to_list()
            return results

        except Exception as e:
            # Fallback to vector search if hybrid fails
            logger.warning(f"Hybrid search failed, falling back to vector: {e}")
            return self.vector_search(query_vector, limit, filter_expr)

    def fts_search(
        self,
        query_text: str,
        limit: int = 10,
        filter_expr: Optional[str] = None,
    ) -> list[dict]:
        """
        Perform full-text search (BM25).

        Args:
            query_text: Text query
            limit: Maximum number of results
            filter_expr: Optional SQL filter expression

        Returns:
            List of matching records
        """
        table = self._ensure_table()

        try:
            query = table.search(query_text, query_type="fts").limit(limit)

            if filter_expr:
                query = query.where(filter_expr)

            results = query.to_list()
            return results

        except Exception as e:
            logger.warning(f"FTS search failed: {e}")
            return []

    def count(self) -> int:
        """Get total number of chunks."""
        table = self._ensure_table()
        return table.count_rows()

    def get_chunks_by_document(self, document_id: str, limit: int = 100) -> list[dict]:
        """
        Get all chunks for a document.

        Args:
            document_id: Document ID to filter by
            limit: Maximum number of chunks to return

        Returns:
            List of chunk records
        """
        table = self._ensure_table()

        try:
            # Use SQL query to filter by document_id
            results = (
                table.search()
                .where(f"document_id = '{document_id}'")
                .limit(limit)
                .to_list()
            )
            # Sort by chunk_index
            results.sort(key=lambda x: x.get("chunk_index", 0))
            return results
        except Exception as e:
            logger.warning(f"Get chunks by document failed: {e}")
            return []

    def get_chunk(self, chunk_id: str) -> Optional[dict]:
        """
        Get a single chunk by ID.

        Args:
            chunk_id: Chunk ID

        Returns:
            Chunk record or None if not found
        """
        table = self._ensure_table()

        try:
            results = (
                table.search()
                .where(f"id = '{chunk_id}'")
                .limit(1)
                .to_list()
            )
            return results[0] if results else None
        except Exception as e:
            logger.warning(f"Get chunk failed: {e}")
            return None

    def close(self):
        """Close the database connection."""
        self._db = None
        self._table = None
